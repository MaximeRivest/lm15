"""
lm15.providers.xai — xAI Grok adapter (Chat Completions dialect).

xAI's API speaks the Chat Completions dialect at ``https://api.x.ai/v1``
(compat preset ``"xai"``, pinned live 2026-09-01).  What makes it a
first-class adapter instead of a preset route is authentication: xAI sells
subscription access (SuperGrok / X Premium) through a device-code OAuth
flow, and the resulting access token is sent as an ordinary bearer key.

Credential resolution order:

1. an explicit ``api_key`` argument (static string or callable);
2. the router's usual key resolution (``RouterConfig.api_keys`` /
   ``XAI_API_KEY``) — the router passes the result as ``api_key``;
3. self-resolved subscription OAuth: lm15's own credential store, then the
   Pi agent store (``~/.pi/agent/auth.json``); refreshed tokens are written
   back to their source file (xAI rotates refresh tokens).
"""

from __future__ import annotations

import os
from typing import ClassVar

from ..auth import XAI_LOGIN_HINT, get_xai_access_token
from ..errors import ProviderError, with_credential_hint
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities
from .base import Credential, SyncTransport, default_transport
from .openai_chat import OpenAIChatLM

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"

XAI_CAPABILITIES = Capabilities(
    input_modalities=frozenset({"text", "image"}),
    output_modalities=frozenset({"text"}),
    features=frozenset({"streaming", "tools", "json_output", "reasoning"}),
)


class XaiLM(OpenAIChatLM):
    """Chat Completions adapter for xAI, with subscription OAuth fallback."""

    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True, models=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="xai",
        supports=supports,
        auth_modes=("bearer", "xai-oauth"),
        env_keys=("XAI_API_KEY",),
    )

    def __init__(
        self,
        api_key: Credential | None = None,
        *,
        credentials_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_XAI_BASE_URL,
    ) -> None:
        self._oauth = api_key is None
        if api_key is not None:
            credential: Credential = api_key
        else:
            # Validate now — get_xai_access_token raises typed, re-login-guided
            # errors (NotConfiguredError / AuthError) — then re-resolve per
            # request so a long-lived client always sends a fresh token
            # (rotations on disk are picked up, expiry refreshes and persists).
            get_xai_access_token(credentials_path)

            def credential() -> str:
                return get_xai_access_token(credentials_path)

        super().__init__(
            api_key=credential,
            transport=transport or default_transport(),
            base_url=base_url,
            compat="xai",
            provider="xai",
            capabilities=XAI_CAPABILITIES,
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        error = super().normalize_error(status, body)
        # Auth failures on the subscription path guide the user back to
        # login; on the API-key path the generic message already fits.
        return with_credential_hint(error, XAI_LOGIN_HINT) if self._oauth else error
