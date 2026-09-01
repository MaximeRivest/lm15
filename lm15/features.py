"""
lm15.features — Provider capability and endpoint declarations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# How a provider's credential is obtained (spec/auth.md AUTH-1):
#
# - "key"            — the ordinary chain: explicit api_keys entry, then the
#                      declared env vars in order, then (local-server presets
#                      only) the preset's placeholder key.
# - "oauth"          — self-resolving local OAuth only.  The adapter reads a
#                      local credential file; the chain above never runs and a
#                      failed OAuth load never falls back to an env var.
# - "key-then-oauth" — the ordinary chain first; when it yields nothing, the
#                      adapter self-resolves local OAuth as the final rung.
#                      Stated trade-off: a configured key silently wins over a
#                      stored subscription login, which can move billing from
#                      subscription to per-token.  Explicit-beats-ambient is
#                      the rule everywhere else; doctor.explain_auth shows
#                      which rung won.
#
# The router and doctor derive their behavior from this declaration; there
# is no parallel provider-name list to keep in sync.
CredentialPolicy = Literal["key", "oauth", "key-then-oauth"]


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

    # Escape hatch for endpoint names not yet promoted to typed booleans.
    extra: frozenset[str] = field(default_factory=frozenset)

    def supports_endpoint(self, name: str) -> bool:
        if name in self.extra:
            return True
        return bool(getattr(self, name, False))


@dataclass(frozen=True, slots=True)
class ProviderManifest:
    provider: str
    supports: EndpointSupport
    auth_modes: tuple[str, ...] = field(default_factory=tuple)
    enterprise_variants: tuple[str, ...] = field(default_factory=tuple)
    env_keys: tuple[str, ...] = field(default_factory=tuple)
    credential_policy: CredentialPolicy = "key"
