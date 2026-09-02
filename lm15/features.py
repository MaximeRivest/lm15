"""
lm15.features — Provider capability and endpoint declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping

# How a provider's credential is obtained (spec/auth.md AUTH-1):
#
# - "key"   — the ordinary chain: explicit api_keys entry, then the declared
#             env vars in order, then (local-server presets only) the
#             preset's placeholder key.
# - "oauth" — self-resolving local OAuth only.  The adapter reads a local
#             credential file; the chain above never runs and a failed OAuth
#             load never falls back to an env var.
# - "oauth-unless-explicit" — an explicit api_keys entry wins; otherwise a
#             usable stored OAuth login wins; env vars are consulted only
#             when no usable login is stored.  Rationale: deliberate
#             in-process configuration always wins, but between two kinds of
#             stored state — a subscription login and an ambient env var —
#             the subscription wins because it spends no money per token
#             (lm15's durable constraint: normal inference must not
#             unexpectedly spend money).  Stated trade-off: with a login
#             stored, a set env key is silently ignored; pass the key
#             explicitly to force it.  Adapters declaring this policy MUST
#             expose ``has_stored_credential()`` — an offline, classmethod
#             probe the router calls to decide the chain.
#
# The router and doctor derive their behavior from this declaration; there
# is no parallel provider-name list to keep in sync.
CredentialPolicy = Literal["key", "oauth", "oauth-unless-explicit"]


@dataclass(frozen=True, slots=True)
class EndpointSupport:
    complete: bool = True
    stream: bool = True
    live: bool = False
    files: bool = False
    batches: bool = False
    images: bool = False
    speech: bool = False
    video: bool = False
    responses_api: bool = False
    models: bool = False
    caches: bool = False

    # Escape hatch for endpoint names not yet promoted to typed booleans.
    extra: frozenset[str] = field(default_factory=frozenset)

    def supports_endpoint(self, name: str) -> bool:
        if name in self.extra:
            return True
        return bool(getattr(self, name, False))


AuthHeader = Literal["bearer", "x-api-key"]


@dataclass(frozen=True, slots=True)
class AccessPolicy:
    """How an adapter reaches a backend. Pure data; ports copy it as a table.

    See ``lm15.access`` for the table of policies and the rationale
    (spec/auth.md AUTH-9).

    ``provider``        canonical provider string (errors, routing, doctor).
    ``supports``        endpoint surfaces this access path carries; a
                        dialect that implements a surface still RAISES when
                        the policy does not carry it (a subscription token
                        has no files or batch).
    ``credential_policy`` spec/auth.md AUTH-1: key | oauth |
                        oauth-unless-explicit.
    ``auth_modes``, ``env_keys``, ``enterprise_variants``: as before
                        (support-matrix pinned).
    ``auth_header``     how the credential travels: ``Authorization: Bearer``
                        or an API-key header (``x-api-key`` on Anthropic,
                        ``x-goog-api-key`` on Gemini — the dialect names it).
    ``headers``         static headers on every request, in order. A header
                        the dialect also sets is merged by the dialect's
                        stated rule (Anthropic joins ``anthropic-beta``).
    ``login_hint``      appended to auth errors when the credential is a
                        local login and there is no env var to set.
    ``backend``         dialect-consulted variant name; ``"api"`` is the
                        provider's public API.
    ``backend_options`` string knobs the backend variant needs (the Codex
                        models endpoint wants a ``client_version``).
    ``system_prefix``   text the backend requires first in the system
                        prompt / instructions (Claude Code, Codex).
    ``base_url``        this access path's default base URL, when it is
                        not the dialect's.
    """

    provider: str
    supports: EndpointSupport = field(default_factory=EndpointSupport)
    auth_modes: tuple[str, ...] = field(default_factory=tuple)
    enterprise_variants: tuple[str, ...] = field(default_factory=tuple)
    env_keys: tuple[str, ...] = field(default_factory=tuple)
    credential_policy: CredentialPolicy = "key"
    auth_header: AuthHeader = "bearer"
    headers: tuple[tuple[str, str], ...] = ()
    login_hint: str | None = None
    backend: str = "api"
    backend_options: Mapping[str, str] = field(default_factory=dict)
    system_prefix: str | None = None
    base_url: str | None = None

    def __post_init__(self) -> None:
        if not self.provider:
            raise ValueError("AccessPolicy.provider must be non-empty")
        if self.credential_policy == "oauth" and self.env_keys:
            # AUTH-1: an oauth manifest declares no environment keys.
            raise ValueError(f"{self.provider}: an 'oauth' access policy declares no env_keys")
        object.__setattr__(self, "auth_modes", tuple(self.auth_modes))
        object.__setattr__(self, "env_keys", tuple(self.env_keys))
        object.__setattr__(self, "enterprise_variants", tuple(self.enterprise_variants))
        object.__setattr__(self, "headers", tuple((str(k), str(v)) for k, v in self.headers))
        object.__setattr__(self, "backend_options", dict(self.backend_options))

    def with_headers(self, headers: Mapping[str, str]) -> "AccessPolicy":
        """A copy with these static headers replaced or appended (names compared case-insensitively)."""
        lowered = {k.lower() for k in headers}
        kept = tuple((k, v) for k, v in self.headers if k.lower() not in lowered)
        return replace(self, headers=kept + tuple(headers.items()))


# The earlier name: an adapter's manifest is its access policy.
ProviderManifest = AccessPolicy
