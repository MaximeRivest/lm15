"""
lm15.router — Minimalist model-string router.

The router is a lookup table you can read, not a framework.  Four
resolution rungs, in fixed order, with no configuration of the order
itself:

  0. object attribute  a ``provider`` attribute carried by the model
                       value itself                   -> source="object"
                       (catalog packages ship model ids as str
                       subclasses that know their provider; duck-typed —
                       lm15 names no package.  An attribute that names
                       nothing routable falls through.)
  1. explicit prefix   ``"openai:gpt-4.1-mini"``       -> source="prefix"
  2. catalog           match against ``registry.list()`` -> source="catalog"
                       entries by id or alias; exact-id matches beat
                       alias matches; multiple providers (or multiple
                       same-provider entries) raise AmbiguousModelError
                       (only if you passed a registry; catalogs are
                       opt-in via ``ModelRegistry.discover()``)
  3. built-in rules    ``DEFAULT_RULES`` prefix match  -> source="rule"

Nothing else.  No plugins, no callbacks, no fallback chains.

A provider is routable when it has a first-class adapter (a key of
:data:`ADAPTERS`) or an OpenAI Chat Completions compat preset (a key of
:data:`CHAT_PRESET_ROUTES`: groq, openrouter, ollama, vllm, sglang).
Preset providers route to ``OpenAIChatLM(compat=<preset>)``, which also
supplies that server's default base_url.

Model-string grammar
--------------------
A model string is split on the FIRST ``:``.  If the head is a routable
provider string, the remainder is the model id sent on the wire.
Otherwise the whole string (colons and all) is treated as a bare model
id and falls through to the catalog and rule rungs.  Consequence: a
fine-tune id like ``ft:gpt-4.1:org`` needs the explicit form
``openai:ft:gpt-4.1:org``.

Credentials
-----------
``resolve()`` is pure: it touches no network and reads no secret values
(it records WHICH env var would be used, never the value).  ``lm()``
reads the key — first from ``RouterConfig.api_keys`` (explicit,
repr-suppressed; values may be static strings or zero-argument
credential-provider callables, resolved per request by the adapter),
then from the env mapping via the provider's ``ProviderManifest.env_keys``
or the preset's own convention (first hit wins), then a keyless local
server's placeholder key.  OAuth providers (``claude-code``,
``openai-codex``) declare no env keys and pass through to their
self-resolving constructors.

The direct LM classes remain first-class; the router is just the
recommended front door.  Needing a custom ``base_url``/transport/compat
(azure, a self-hosted gateway) is the documented escape hatch: ``lm()``
returns the ordinary provider LM — keep it and configure it yourself
next time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import AsyncIterator, Iterator, Mapping

from .errors import LM15Error, NotConfiguredError
from .models import ModelInfo, ModelRegistry
from .providers import (
    AnthropicLM,
    AsyncAnthropicLM,
    AsyncClaudeCodeLM,
    AsyncGeminiLM,
    AsyncOpenAIChatLM,
    AsyncOpenAICodexLM,
    AsyncOpenAILM,
    ClaudeCodeLM,
    Credential,
    GeminiLM,
    AsyncXaiLM,
    OpenAIChatLM,
    OpenAICodexLM,
    OpenAILM,
    XaiLM,
)
from .types import Request, Response, StreamEvent

__all__ = [
    "RouteRule",
    "DEFAULT_RULES",
    "ADAPTERS",
    "ASYNC_ADAPTERS",
    "PresetRoute",
    "CHAT_PRESET_ROUTES",
    "Resolution",
    "RouterConfig",
    "RouterError",
    "UnknownModelError",
    "AmbiguousModelError",
    "MissingCredentialError",
    "LMRouter",
    "AsyncLMRouter",
]


# ---------------------------------------------------------------- rules ----


@dataclass(frozen=True, slots=True)
class RouteRule:
    """Maps a model-id prefix to a provider.  That's all a rule is.

    ``note`` is a short human rationale surfaced in docs and
    :meth:`Resolution.describe` output.
    """

    prefix: str
    provider: str
    note: str = ""


# The complete built-in knowledge of the router.  Inspectable, printable,
# overridable by passing rules=... in RouterConfig.  First match wins.
# This table is a convenience, not a registry of truth: new model
# families need a release (or a catalog, or the provider: prefix).
DEFAULT_RULES: tuple[RouteRule, ...] = (
    RouteRule("claude-", "anthropic", note="Anthropic Claude family"),
    RouteRule("gpt-", "openai", note="OpenAI GPT family (Responses API; use openai-chat: for Chat Completions)"),
    RouteRule("o1", "openai", note="OpenAI o1 reasoning family"),
    RouteRule("o3", "openai", note="OpenAI o3 reasoning family"),
    RouteRule("o4", "openai", note="OpenAI o4 reasoning family"),
    RouteRule("gemini-", "gemini", note="Google Gemini family"),
    RouteRule("grok-", "xai", note="xAI Grok family (XAI_API_KEY or subscription OAuth)"),
    RouteRule("sora-", "openai", note="OpenAI Sora video generation"),
    RouteRule("veo-", "gemini", note="Google Veo video generation"),
)


# provider string -> LM class.  Hardcoded, exported, inspectable.
# Values are the *sync* classes; AsyncLMRouter uses ASYNC_ADAPTERS.
ADAPTERS: Mapping[str, type] = {
    "openai": OpenAILM,
    "openai-chat": OpenAIChatLM,
    "anthropic": AnthropicLM,
    "gemini": GeminiLM,
    "claude-code": ClaudeCodeLM,
    "openai-codex": OpenAICodexLM,
    "xai": XaiLM,
}

ASYNC_ADAPTERS: Mapping[str, type] = {
    "openai": AsyncOpenAILM,
    "openai-chat": AsyncOpenAIChatLM,
    "anthropic": AsyncAnthropicLM,
    "gemini": AsyncGeminiLM,
    "claude-code": AsyncClaudeCodeLM,
    "openai-codex": AsyncOpenAICodexLM,
    "xai": AsyncXaiLM,
}

def _credential_policy(provider: str, adapters: Mapping[str, type] | None = None) -> str:
    """The provider's declared ``ProviderManifest.credential_policy``.

    Chat-preset routes (groq, openrouter, local servers) have no manifest
    of their own and are always ordinary ``"key"`` providers.  This is the
    single source the router and doctor consult — there is deliberately no
    hardcoded provider-name list that could drift from the manifests.
    """
    lookup = ADAPTERS if adapters is None else adapters
    cls = lookup.get(provider)
    if cls is None:
        return "key"
    return cls.manifest.credential_policy


@dataclass(frozen=True, slots=True)
class PresetRoute:
    """A provider routable through ``OpenAIChatLM`` with a named compat
    preset.  Pure data: the provider string doubles as the preset name
    (which also supplies the server's default base_url); ``env_keys`` is
    that server's own key convention; ``default_key`` is the placeholder
    local servers accept when no key is configured."""

    provider: str
    env_keys: tuple[str, ...] = ()
    default_key: str | None = None
    note: str = ""


# Chat Completions preset providers, keyed by provider string.  Same
# spirit as DEFAULT_RULES: the complete built-in knowledge, inspectable
# and printable.  Only presets whose base_url is pinned in
# OPENAI_CHAT_PRESET_BASE_URLS belong here; new entries land with live
# receipts first, like every provider behavior.
CHAT_PRESET_ROUTES: Mapping[str, PresetRoute] = {
    "groq": PresetRoute("groq", env_keys=("GROQ_API_KEY",), note="Groq Cloud (Chat Completions dialect)"),
    "openrouter": PresetRoute("openrouter", env_keys=("OPENROUTER_API_KEY",), note="OpenRouter (Chat Completions dialect)"),
    "ollama": PresetRoute("ollama", default_key="ollama", note="local ollama server (keyless)"),
    "vllm": PresetRoute("vllm", default_key="EMPTY", note="local vLLM server (keyless)"),
    "sglang": PresetRoute("sglang", default_key="EMPTY", note="local SGLang server (keyless)"),
}


def _canonical_provider(name: str) -> str:
    """Provider strings are hyphenated (``openai-chat``); the underscore
    spelling is accepted everywhere as a permanent alias."""
    return name.replace("_", "-")


def _routable(provider: str, adapters: Mapping[str, type]) -> bool:
    return provider in adapters or provider in CHAT_PRESET_ROUTES


def _adapter_for(provider: str, adapters: Mapping[str, type]) -> type:
    if provider in adapters:
        return adapters[provider]
    return adapters["openai-chat"]  # preset route


# --------------------------------------------------------------- errors ----


class RouterError(LM15Error):
    """Base for all routing failures."""

    default_code = "router"


class UnknownModelError(RouterError):
    """No resolution rung matched the model string."""

    default_code = "unknown_model"

    def __init__(
        self,
        message: str = "",
        *,
        model: str = "",
        rules_tried: tuple[RouteRule, ...] = (),
        catalog_searched: bool = False,
        **kwargs,
    ) -> None:
        self.model = model
        self.rules_tried = tuple(rules_tried)
        self.catalog_searched = catalog_searched
        super().__init__(message, **kwargs)


class AmbiguousModelError(RouterError):
    """Catalog matched the model id under more than one provider."""

    default_code = "ambiguous_model"

    def __init__(
        self,
        message: str = "",
        *,
        model: str = "",
        providers: tuple[str, ...] = (),
        **kwargs,
    ) -> None:
        self.model = model
        self.providers = tuple(providers)
        self.candidates = self.providers  # alias: full candidate list
        super().__init__(message, **kwargs)


class MissingCredentialError(RouterError, NotConfiguredError):
    """Provider resolved but no API key was found.

    Subclasses the existing :class:`lm15.errors.NotConfiguredError` —
    semantically the credential case IS not-configured — so existing
    ``except NotConfiguredError`` handlers keep working.  Carries
    ``provider`` and ``env_keys`` straight from the ProviderManifest.
    """

    default_code = "not_configured"


# ----------------------------------------------------------- resolution ----


@dataclass(frozen=True, slots=True)
class Resolution:
    """The complete answer to "how did you route this string".

    ``resolve()`` returning this IS the explain() method — there is no
    separate one.
    """

    requested: str                  # verbatim input string
    model: str                      # id sent on the wire (prefix stripped)
    provider: str                   # canonical provider string
    adapter: str                    # LM class name, e.g. "AnthropicLM"
    source: str                     # "object" | "prefix" | "catalog" | "rule"
    rule: RouteRule | None = None   # the matching rule when source == "rule"
    env_key: str | None = None      # env var the key would be read from;
                                    # None for OAuth providers or when an
                                    # explicit api_keys entry overrides env
    model_info: ModelInfo | None = None  # catalog metadata when source == "catalog"
    compat: str | None = None       # OpenAIChatLM preset name when routed
                                    # through CHAT_PRESET_ROUTES

    def describe(self) -> str:
        """One-paragraph human-readable explanation of this resolution."""
        parts = [f"{self.requested!r} -> provider {self.provider!r} ({self.adapter})"]
        if self.source == "object":
            parts.append("via provider attribute on the model value")
        elif self.source == "prefix":
            parts.append("via explicit provider prefix")
        elif self.source == "catalog":
            parts.append("via catalog match")
        elif self.source == "rule" and self.rule is not None:
            note = f" — {self.rule.note}" if self.rule.note else ""
            parts.append(f"via built-in rule prefix={self.rule.prefix!r}{note}")
        if self.compat is not None:
            parts.append(f"Chat Completions compat preset {self.compat!r}")
        parts.append(f"wire model {self.model!r}")
        route = CHAT_PRESET_ROUTES.get(self.provider)
        policy = _credential_policy(self.provider)
        if self.env_key is not None:
            parts.append(f"key from ${self.env_key}")
        elif policy == "oauth":
            parts.append("local OAuth credential (no env key)")
        elif route is not None and route.default_key is not None:
            parts.append("key from explicit api_keys or the preset's local-server default")
        else:
            parts.append("key from explicit api_keys")
        if policy == "key-then-oauth":
            parts.append("else the stored local OAuth credential")
        return "; ".join(parts) + "."

    def __str__(self) -> str:
        return self.describe()


# ---------------------------------------------------------------- config ----


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """Everything the router consults.  All explicit, nothing discovered
    behind your back.  Catalog use is opt-in: pass
    ``registry=ModelRegistry.discover()``.

    ``api_keys`` maps provider string -> credential and beats env (repr-
    suppressed; lets hermetic tests pass ``env={}``).  A credential is a
    static key string or a zero-argument provider callable, resolved per
    request by the adapter.  ``env`` defaults to ``os.environ`` at
    lookup time.
    """

    registry: ModelRegistry | None = None
    rules: tuple[RouteRule, ...] = DEFAULT_RULES
    env: Mapping[str, str] | None = None
    api_keys: Mapping[str, Credential] | None = field(default=None, repr=False)
    transport: object | None = None  # SyncTransport for LMRouter, AsyncTransport
                                     # for AsyncLMRouter; passed to every LM the
                                     # router constructs (tests, custom pooling)


# ------------------------------------------------------------- internals ----


def _resolution(
    *,
    requested: str,
    wire_model: str,
    provider: str,
    source: str,
    config: RouterConfig,
    adapters: Mapping[str, type],
    rule: RouteRule | None = None,
    model_info: ModelInfo | None = None,
) -> Resolution:
    route = CHAT_PRESET_ROUTES.get(provider) if provider not in adapters else None
    return Resolution(
        requested=requested,
        model=wire_model,
        provider=provider,
        adapter=_adapter_for(provider, adapters).__name__,
        source=source,
        rule=rule,
        env_key=_env_key_for(provider, config, adapters),
        model_info=model_info,
        compat=route.provider if route is not None else None,
    )


def _resolve(model: str, config: RouterConfig, adapters: Mapping[str, type]) -> Resolution:
    if not isinstance(model, str) or not model:
        raise UnknownModelError(
            "model must be a non-empty string", model=str(model),
            rules_tried=config.rules, catalog_searched=config.registry is not None,
        )

    requested = model

    # Rung 0: a provider attribute carried by the model value itself.
    # Catalog packages ship model ids as str subclasses that know their
    # provider; duck-typed, so lm15 names no package.  An attribute that
    # names nothing routable falls through (and is mentioned if nothing
    # else matches either).
    object_provider = getattr(model, "provider", None)
    if isinstance(object_provider, str) and object_provider:
        object_provider = _canonical_provider(object_provider)
    else:
        object_provider = None
    if object_provider is not None and _routable(object_provider, adapters):
        return _resolution(
            requested=requested,
            wire_model=str(model),
            provider=object_provider,
            source="object",
            config=config,
            adapters=adapters,
        )

    # Rung 1: explicit provider prefix (split on FIRST colon).
    if ":" in model:
        raw_head, rest = model.split(":", 1)
        head = _canonical_provider(raw_head)
        if _routable(head, adapters) and rest:
            return _resolution(
                requested=requested,
                wire_model=rest,
                provider=head,
                source="prefix",
                config=config,
                adapters=adapters,
            )

    # Rung 2: catalog (only if a registry was explicitly supplied).
    if config.registry is not None:
        matches = tuple(
            info
            for info in config.registry.list()
            if info.id == model or model in info.aliases
        )
        providers = tuple(dict.fromkeys(info.provider for info in matches))
        if len(providers) > 1:
            options = " or ".join(f'"{p}:{model}"' for p in providers)
            raise AmbiguousModelError(
                f"model {model!r} is offered by multiple providers: "
                f"{', '.join(providers)}. Fix: use the explicit form, e.g. "
                f"Request(model={providers[0] + ':' + model!r}) — options: {options}.",
                model=model,
                providers=providers,
            )
        if matches:
            # Exact-id matches beat alias matches; an alias must never
            # shadow an entry whose canonical id IS the requested string.
            exact = tuple(info for info in matches if info.id == model)
            narrowed = exact if exact else matches
            if len(narrowed) > 1:
                # Same provider (multi-provider was caught above), multiple
                # entries: never pick one by insertion order.
                ids = ", ".join(info.id for info in narrowed)
                raise AmbiguousModelError(
                    f"model {model!r} matches multiple catalog entries "
                    f"({ids}) under provider {narrowed[0].provider!r}. "
                    "Fix: request a canonical id directly.",
                    model=model,
                    providers=providers,
                )
            info = narrowed[0]
            catalog_provider = _canonical_provider(info.provider)
            if not _routable(catalog_provider, adapters):
                raise UnknownModelError(
                    f"model {model!r} resolved in the catalog to provider "
                    f"{info.provider!r}, but lm15 has no adapter or compat "
                    f"preset for it. Known providers: {_known_providers(adapters)}. "
                    "Construct a provider LM directly (e.g. OpenAIChatLM with "
                    "a custom base_url) for OpenAI-compatible servers.",
                    model=model,
                    rules_tried=config.rules,
                    catalog_searched=True,
                )
            return _resolution(
                requested=requested,
                wire_model=info.id if model in info.aliases else str(model),
                provider=catalog_provider,
                source="catalog",
                config=config,
                adapters=adapters,
                model_info=info,
            )

    # Rung 3: built-in prefix rules, first match wins.
    for rule in config.rules:
        if model.startswith(rule.prefix):
            rule_provider = _canonical_provider(rule.provider)
            if not _routable(rule_provider, adapters):
                raise UnknownModelError(
                    f"rule {rule!r} names provider {rule.provider!r}, which has "
                    f"no adapter. Known providers: {_known_providers(adapters)}.",
                    model=model,
                    rules_tried=config.rules,
                    catalog_searched=config.registry is not None,
                )
            return _resolution(
                requested=requested,
                wire_model=str(model),
                provider=rule_provider,
                source="rule",
                config=config,
                adapters=adapters,
                rule=rule,
            )

    hints = []
    if ":" in model:
        import difflib

        head = _canonical_provider(model.split(":", 1)[0])
        close = difflib.get_close_matches(head, sorted({*adapters, *CHAT_PRESET_ROUTES}), n=1, cutoff=0.75)
        if close:
            hints.append(f'Did you mean "{close[0]}:{model.split(":", 1)[1]}"?')
    if object_provider is not None:
        hints.append(
            f"The model value carries provider {object_provider!r}, which has "
            f"no adapter or compat preset (known providers: {_known_providers(adapters)})."
        )
    hints.append(
        f'Use an explicit provider prefix — "provider:{model}" with provider '
        f"one of: {_known_providers(adapters)}."
    )
    if config.registry is None:
        hints.append(
            "Or pass a model catalog: "
            "LMRouter(config=RouterConfig(registry=ModelRegistry.discover())) "
            "— install a catalog package such as 'aimo' first."
        )
    raise UnknownModelError(
        f"could not route model {model!r}: no provider prefix, "
        f"{'no catalog match' if config.registry is not None else 'no catalog supplied'}, "
        f"and none of the {len(config.rules)} built-in rules matched. "
        + " ".join(hints),
        model=model,
        rules_tried=config.rules,
        catalog_searched=config.registry is not None,
    )


def _known_providers(adapters: Mapping[str, type]) -> str:
    return ", ".join(sorted({*adapters, *CHAT_PRESET_ROUTES}))


def _declared_env_keys(provider: str, adapters: Mapping[str, type]) -> tuple[str, ...]:
    route = CHAT_PRESET_ROUTES.get(provider) if provider not in adapters else None
    if route is not None:
        return route.env_keys
    return adapters[provider].manifest.env_keys


def _api_keys_entry(config: RouterConfig, provider: str) -> tuple[Credential | None, bool]:
    """Explicit api_keys entry for a provider, matching either spelling."""
    if config.api_keys is None:
        return None, False
    for key, value in config.api_keys.items():
        if _canonical_provider(key) == provider:
            return value, True
    return None, False


def _env_key_for(provider: str, config: RouterConfig, adapters: Mapping[str, type]) -> str | None:
    """WHICH env var lm() would read for this provider (never the value).

    None when the provider is OAuth-based or a keyless local preset
    (declares no env keys) or when an explicit ``api_keys`` entry
    overrides env lookup entirely.
    """
    if _api_keys_entry(config, provider)[1]:
        return None
    env_keys = _declared_env_keys(provider, adapters)
    if not env_keys:
        return None
    env = config.env if config.env is not None else os.environ
    for key in env_keys:
        if env.get(key):
            return key
    return env_keys[0]


def _build_lm(resolution: Resolution, config: RouterConfig, adapters: Mapping[str, type]):
    cls = _adapter_for(resolution.provider, adapters)
    route = CHAT_PRESET_ROUTES.get(resolution.provider) if resolution.provider not in adapters else None
    extra: dict = {}
    if config.transport is not None:
        extra["transport"] = config.transport
    policy = _credential_policy(resolution.provider, adapters)
    if policy == "oauth":
        return cls(**extra)  # self-resolving local OAuth constructor
    api_key, _ = _api_keys_entry(config, resolution.provider)
    if api_key is None:
        env = config.env if config.env is not None else os.environ
        for key in _declared_env_keys(resolution.provider, adapters):
            value = env.get(key)
            if value:
                api_key = value
                break
    if api_key is None and route is not None and route.default_key is not None:
        api_key = route.default_key
    if not api_key and policy == "key-then-oauth":
        return cls(**extra)  # self-resolving local OAuth constructor
    if not api_key:
        env_keys = _declared_env_keys(resolution.provider, adapters)
        raise MissingCredentialError(
            f"no API key found for provider {resolution.provider!r}. "
            f"Set {' or '.join(env_keys)} in the environment, or pass "
            f"RouterConfig(api_keys={{{resolution.provider!r}: \"...\"}}).",
            provider=resolution.provider,
            env_keys=env_keys,
        )
    if route is not None:
        return cls(api_key=api_key, compat=route.provider, **extra)
    return cls(api_key=api_key, **extra)


def _routed_request(request: Request, resolution: Resolution) -> Request:
    if request.model == resolution.model:
        return request
    return replace(request, model=resolution.model)


# ---------------------------------------------------------------- router ----


class LMRouter:
    """Routes model strings to provider LMs.

    Four methods, no state you can't see: config is frozen; the only
    mutation is an LM cache keyed by provider (one LM per provider,
    built lazily, reused).
    """

    def __init__(self, config: RouterConfig = RouterConfig()) -> None:
        self.config = config
        self._lms: dict[str, object] = {}

    _adapters: Mapping[str, type] = ADAPTERS

    def resolve(self, model: str) -> Resolution:
        """Pure lookup; touches no network and reads no secret values.

        Raises UnknownModelError / AmbiguousModelError.  This IS the
        explain() method — there is no separate one.
        """
        return _resolve(model, self.config, self._adapters)

    def lm(self, model: str):
        """resolve(), then construct-or-reuse the provider LM.

        Raises MissingCredentialError when no key is found.  The
        returned LM is an ordinary OpenAILM/AnthropicLM/... — the escape
        hatch is built in: keep it, configure transports yourself.
        """
        resolution = self.resolve(model)
        lm = self._lms.get(resolution.provider)
        if lm is None:
            lm = _build_lm(resolution, self.config, self._adapters)
            self._lms[resolution.provider] = lm
        return lm

    def complete(self, request: Request) -> Response:
        resolution = self.resolve(request.model)
        return self.lm(request.model).complete(_routed_request(request, resolution))

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        resolution = self.resolve(request.model)
        return self.lm(request.model).stream(_routed_request(request, resolution))


class AsyncLMRouter:
    """Async mirror of :class:`LMRouter`.

    Same four methods; ``lm()`` returns Async* mirrors; ``complete`` is
    a coroutine and ``stream`` returns an async iterator.  ``resolve()``
    stays sync (it is pure).
    """

    def __init__(self, config: RouterConfig = RouterConfig()) -> None:
        self.config = config
        self._lms: dict[str, object] = {}

    _adapters: Mapping[str, type] = ASYNC_ADAPTERS

    def resolve(self, model: str) -> Resolution:
        return _resolve(model, self.config, self._adapters)

    def lm(self, model: str):
        resolution = self.resolve(model)
        lm = self._lms.get(resolution.provider)
        if lm is None:
            lm = _build_lm(resolution, self.config, self._adapters)
            self._lms[resolution.provider] = lm
        return lm

    async def complete(self, request: Request) -> Response:
        resolution = self.resolve(request.model)
        return await self.lm(request.model).complete(_routed_request(request, resolution))

    def stream(self, request: Request) -> AsyncIterator[StreamEvent]:
        resolution = self.resolve(request.model)
        return self.lm(request.model).stream(_routed_request(request, resolution))
