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

import base64
import os
from typing import Any, ClassVar

from ..auth import XAI_LOGIN_HINT, get_xai_access_token
from ..errors import ProviderError, UnsupportedFeatureError, with_credential_hint
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities
from ..transports import TransportRequest
from ..types import ImageGenerationRequest, ImageGenerationResponse, ImagePart, Usage
from .base import Credential, HttpResponse, SyncTransport, default_transport
from .common import make_json_request
from .openai_chat import OpenAIChatLM

DEFAULT_XAI_BASE_URL = "https://api.x.ai/v1"

XAI_CAPABILITIES = Capabilities(
    input_modalities=frozenset({"text", "image"}),
    output_modalities=frozenset({"text"}),
    features=frozenset({"streaming", "tools", "json_output", "reasoning"}),
)


class XaiLM(OpenAIChatLM):
    """Chat Completions adapter for xAI, with subscription OAuth fallback."""

    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True, models=True, images=True)
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

    def _image_generate_request(self, request: ImageGenerationRequest) -> TransportRequest:
        base = self.base_url.rstrip("/")
        payload: dict[str, Any] = {"model": request.model, "prompt": request.prompt, **(request.extensions or {})}
        if request.size is not None:
            # No wire slot: xAI sizes through quality/resolution knobs with
            # their own names (extensions).  Raising beats guessing a mapping.
            raise UnsupportedFeatureError(
                "xai: size has no wire slot; use extensions for xAI's quality/resolution fields",
                provider=self.provider,
            )
        if not request.images:
            return make_json_request(method="POST", url=f"{base}/images/generations", headers=self._headers(), payload=payload, read_timeout=300.0)
        if len(request.images) > 1:
            raise UnsupportedFeatureError(
                "xai: image edits take exactly one input image; the wire has no slot for more",
                provider=self.provider,
            )
        payload["image"] = _xai_image_input(request.images[0], self.provider)
        return make_json_request(method="POST", url=f"{base}/images/edits", headers=self._headers(), payload=payload, read_timeout=300.0)

    def _image_generation_from_response(self, request: ImageGenerationRequest, resp: HttpResponse) -> ImageGenerationResponse:
        data = resp.json()
        images: list[ImagePart] = []
        for item in data.get("data", []) or []:
            if not isinstance(item, dict):
                continue
            mime = item.get("mime_type")
            media_type = mime if isinstance(mime, str) and mime else "application/octet-stream"
            if item.get("b64_json"):
                images.append(ImagePart(media_type=media_type, data=str(item["b64_json"])))
            elif item.get("url"):
                images.append(ImagePart(media_type=media_type, url=str(item["url"])))
        if not images:
            raise ProviderError("xai: image response carries no images", provider=self.provider)
        # Captured: usage reports cost_in_usd_ticks only — no token counts
        # exist, so Usage stays empty and the figure lives in provider_data.
        return ImageGenerationResponse(images=tuple(images), usage=Usage(), provider_data=data)

    def normalize_error(self, status: int, body: str) -> ProviderError:
        error = super().normalize_error(status, body)
        # Auth failures on the subscription path guide the user back to
        # login; on the API-key path the generic message already fits.
        return with_credential_hint(error, XAI_LOGIN_HINT) if self._oauth else error


# ─── Media generation (captured live 2026-09-01) ─────────────────────
#
# Images: /images/generations for text-to-image; /images/edits for
# image-to-image.  The split matters: `generations` silently IGNORES
# input images (verified by pixel check), so edits must never route
# there.  The edit input is `image:{url|file_id}` — an https URL, a
# data URI (verified honored), or an xAI file id.  Exactly one input
# image; the wire has no slot for more.  Responses state `mime_type`
# per image (JPEG, captured); usage reports only a dollar-tick cost,
# which stays verbatim in provider_data (no token counts exist).
# Speech: no endpoint (voice is app-only) — the base hooks raise.


def _xai_image_input(part: "ImagePart", provider: str) -> dict[str, str]:
    if part.url is not None:
        return {"url": part.url}
    if part.file_id is not None:
        return {"file_id": part.file_id}
    if part.data is not None:
        return {"url": f"data:{part.media_type};base64,{part.data}"}
    if part.path is not None:
        encoded = base64.b64encode(part.path.read_bytes()).decode("ascii")
        return {"url": f"data:{part.media_type};base64,{encoded}"}
    raise UnsupportedFeatureError(f"{provider}: input image carries no content", provider=provider)
