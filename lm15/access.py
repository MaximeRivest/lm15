"""
lm15.access — how a dialect adapter reaches a backend, as a value.

An lm15 adapter is three composed things:

- the **dialect**: the class that speaks a wire format (Anthropic Messages,
  OpenAI Responses, OpenAI Chat Completions, Gemini);
- for the chat dialect, a **compat** value describing a server's quirks
  (``OpenAIChatCompat``);
- an **access policy**: this module. Which credential, which headers,
  which endpoint surfaces, which backend variant, which login hint.

Before 2026-09-02 the third thing was a subclass: ``ClaudeCodeLM`` extended
``AnthropicLM``, ``OpenAICodexLM`` extended ``OpenAILM``, ``XaiLM`` extended
``OpenAIChatLM``, each overriding headers, payload defaults, error hints,
and endpoint blocks. Go and Rust have no inheritance, so every port would
have invented its own shape for the same facts. An ``AccessPolicy`` is a
frozen value; a port copies the table below as data and consults it at the
same named points the reference does (spec/auth.md AUTH-9).

What stays behaviour, deliberately:

- ``backend`` names a variant the dialect adapter switches on at a small,
  documented set of points (the ChatGPT Codex backend answers a different
  error envelope, a different models endpoint, and is streaming-first).
  That is a ``match`` in every language, keyed on data.
- Loading a stored subscription credential is per-language code, keyed by
  ``provider`` (``_CREDENTIAL_LOADERS``); the policy says *that* a stored
  credential is used, the loader says *how*.

``ProviderManifest`` is the same class under its earlier name: an adapter's
manifest is its access policy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .auth import (
    CLAUDE_CODE_LOGIN_HINT,
    OPENAI_CODEX_LOGIN_HINT,
    XAI_LOGIN_HINT,
    extract_chatgpt_account_id,
    get_claude_code_access_token,
    get_codex_cli_access_token,
    get_xai_access_token,
    usable_xai_credential,
)
from .compat import OPENAI_CHAT_PRESET_BASE_URLS
from .errors import NotConfiguredError
from .features import AccessPolicy, AuthHeader, CredentialPolicy, EndpointSupport  # noqa: F401 — re-exported

Credential = str | Callable[[], str]

# ─── The table ───────────────────────────────────────────────────────

ANTHROPIC_API = AccessPolicy(
    provider="anthropic",
    supports=EndpointSupport(complete=True, stream=True, files=True, batches=True, models=True),
    auth_modes=("x-api-key",),
    env_keys=("ANTHROPIC_API_KEY",),
    auth_header="x-api-key",
)

DEFAULT_CLAUDE_CODE_VERSION = "2.1.170"
DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT = "You are Claude Code, Anthropic's official CLI for Claude."

# models=True: the Anthropic /v1/models endpoint answers to the OAuth
# headers (validated live 2026-08-31, HTTP 200). Files and batch are
# API-key surfaces the subscription token does not carry.
CLAUDE_CODE = AccessPolicy(
    provider="claude-code",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    credential_policy="oauth",
    auth_modes=("claude-code-oauth", "bearer-oauth"),
    auth_header="bearer",
    headers=(
        ("anthropic-dangerous-direct-browser-access", "true"),
        ("anthropic-beta", "claude-code-20250219,oauth-2025-04-20"),
        ("x-app", "cli"),
        ("user-agent", f"claude-cli/{DEFAULT_CLAUDE_CODE_VERSION}"),
    ),
    login_hint=CLAUDE_CODE_LOGIN_HINT,
    backend="claude-code",
    system_prefix=DEFAULT_CLAUDE_CODE_SYSTEM_PROMPT,
)

OPENAI_API = AccessPolicy(
    provider="openai",
    supports=EndpointSupport(
        complete=True, stream=True, live=True, files=True, batches=True,
        images=True, speech=True, video=True, responses_api=True, models=True,
    ),
    auth_modes=("bearer",),
    env_keys=("OPENAI_API_KEY",),
    enterprise_variants=("azure-openai",),
)

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_ORIGINATOR = "lm15"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."
# The backend's /models endpoint requires a client_version query parameter
# (a Codex CLI release); any recent release is accepted.
DEFAULT_CODEX_CLIENT_VERSION = "0.147.0"

OPENAI_CODEX = AccessPolicy(
    provider="openai-codex",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    credential_policy="oauth",
    auth_modes=("chatgpt-oauth", "bearer-oauth"),
    headers=(
        ("OpenAI-Beta", "responses=experimental"),
        ("originator", DEFAULT_CODEX_ORIGINATOR),
    ),
    login_hint=OPENAI_CODEX_LOGIN_HINT,
    backend="chatgpt-codex",
    backend_options={"client_version": DEFAULT_CODEX_CLIENT_VERSION},
    system_prefix=DEFAULT_CODEX_INSTRUCTIONS,
    base_url=DEFAULT_CODEX_BASE_URL,
)

OPENAI_CHAT_API = AccessPolicy(
    provider="openai_chat",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("OPENAI_API_KEY",),
)

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"

XAI = AccessPolicy(
    provider="xai",
    supports=EndpointSupport(complete=True, stream=True, models=True, images=True, video=True),
    credential_policy="oauth-unless-explicit",
    auth_modes=("bearer", "xai-oauth"),
    env_keys=("XAI_API_KEY",),
    login_hint=XAI_LOGIN_HINT,
    base_url=DEFAULT_XAI_BASE_URL,
)

GEMINI_API = AccessPolicy(
    provider="gemini",
    supports=EndpointSupport(
        complete=True, stream=True, live=True, files=True, batches=True,
        images=True, speech=True, video=True, models=True, caches=True,
    ),
    auth_modes=("query-api-key", "x-goog-api-key"),
    env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    auth_header="x-api-key",  # the dialect renders it as x-goog-api-key
)

# ─── Chat Completions servers reached through OpenAIChatLM ───────────
#
# These policies are bound onto the chat dialect by the provider registry
# (lm15.registry): the class is OpenAIChatLM, the compat preset of the
# same name names the server's quirks, and this value names the credential
# and the surfaces.  base_url points at the compat table so each URL has
# one copy.

GROQ = AccessPolicy(
    provider="groq",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("GROQ_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["groq"],
)

OPENROUTER = AccessPolicy(
    provider="openrouter",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("OPENROUTER_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["openrouter"],
)

# DeepSeek (api-docs.deepseek.com, scraped 2026-09-03): bearer key from
# platform.deepseek.com, prepaid balance (a drained balance is HTTP 402,
# never a surprise bill).  The same key also opens an Anthropic-format
# endpoint (/anthropic) and a Responses-format endpoint; `deepseek` names
# the Chat Completions wire only — a provider string never guesses a
# protocol.
DEEPSEEK = AccessPolicy(
    provider="deepseek",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("DEEPSEEK_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["deepseek"],
)

# Z.AI (docs.z.ai, scraped 2026-09-03): bearer key from z.ai/manage-apikey,
# general endpoint only.  The GLM Coding Plan is a subscription with its own
# endpoint and its own terms (docs.z.ai/devpack); lm15 does not name it.
ZAI = AccessPolicy(
    provider="zai",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    env_keys=("ZAI_API_KEY",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["zai"],
)

# Keyless local servers: no env key; the registry supplies the placeholder
# the server accepts when nothing is configured.
OLLAMA = AccessPolicy(
    provider="ollama",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["ollama"],
)

VLLM = AccessPolicy(
    provider="vllm",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["vllm"],
)

SGLANG = AccessPolicy(
    provider="sglang",
    supports=EndpointSupport(complete=True, stream=True, models=True),
    auth_modes=("bearer",),
    base_url=OPENAI_CHAT_PRESET_BASE_URLS["sglang"],
)


# ─── Credential loading, keyed by provider ───────────────────────────


@dataclass(frozen=True, slots=True)
class LoadedCredential:
    credential: Credential
    account_id: str | None = None
    source: str = "explicit"  # "explicit" (caller-supplied) or "stored" (local login)


def _load_claude_code(path: str | os.PathLike[str] | None) -> LoadedCredential:
    # Validate now (typed, re-login-guided errors), then re-resolve per
    # request so a long-lived client always sends a fresh token.
    get_claude_code_access_token(path)
    return LoadedCredential(lambda: get_claude_code_access_token(path), source="stored")


def _load_openai_codex(path: str | os.PathLike[str] | None) -> LoadedCredential:
    initial = get_codex_cli_access_token(path)
    account_id = initial.account_id or extract_chatgpt_account_id(initial.access_token)
    return LoadedCredential(lambda: get_codex_cli_access_token(path).access_token, account_id, source="stored")


def _load_xai(path: str | os.PathLike[str] | None) -> LoadedCredential:
    get_xai_access_token(path)
    return LoadedCredential(lambda: get_xai_access_token(path), source="stored")


_CREDENTIAL_LOADERS: dict[str, Callable[[str | os.PathLike[str] | None], LoadedCredential]] = {
    "claude-code": _load_claude_code,
    "openai-codex": _load_openai_codex,
    "xai": _load_xai,
}

_STORED_PROBES: dict[str, Callable[[], bool]] = {
    "xai": usable_xai_credential,
}


def load_credential(
    policy: AccessPolicy,
    api_key: Credential | None,
    *,
    credentials_path: str | os.PathLike[str] | None = None,
) -> LoadedCredential:
    """Resolve the credential an adapter will send under ``policy``.

    An explicit ``api_key`` always wins (AUTH-1). Otherwise a stored-login
    policy loads through its provider's loader; a ``key`` policy with no
    key is a typed not-configured error naming the env keys.
    """
    if api_key is not None and api_key != "":
        return LoadedCredential(api_key)
    loader = _CREDENTIAL_LOADERS.get(policy.provider) if policy.credential_policy != "key" else None
    if loader is not None:
        return loader(credentials_path)
    raise NotConfiguredError(
        f"{policy.provider}: no credential given"
        + (f"; set {' or '.join(policy.env_keys)} or pass api_key=" if policy.env_keys else "; pass api_key="),
        provider=policy.provider,
        env_keys=policy.env_keys,
        credential_hint=policy.login_hint,
    )


def has_stored_credential(policy: AccessPolicy) -> bool:
    """Offline probe (reads files, never the network) for the router's
    ``oauth-unless-explicit`` chain: is a usable login stored locally?"""
    probe = _STORED_PROBES.get(policy.provider)
    return bool(probe()) if probe is not None else False


def auth_header(policy: AccessPolicy, token: str, *, api_key_header: str = "x-api-key") -> tuple[str, str]:
    """The (name, value) pair that carries ``token`` under this policy."""
    if policy.auth_header == "bearer":
        return ("Authorization", f"Bearer {token}")
    return (api_key_header, token)
