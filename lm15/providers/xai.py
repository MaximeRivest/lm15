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

Billing trade-off, stated: a configured key wins over a stored subscription
login.  A stray ``XAI_API_KEY`` in the environment therefore silently moves
you from subscription (prepaid) to per-token billing.  Explicit-beats-ambient
is the resolution rule everywhere in lm15; run
``lm15.doctor.explain_auth("xai")`` to see which rung won.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, ClassVar

from ..auth import XAI_LOGIN_HINT, get_xai_access_token
from ..errors import ProviderError, UnsupportedFeatureError, with_credential_hint
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities
from ..transports import TransportRequest
from ..types import ImageGenerationRequest, ImageGenerationResponse, ImagePart, Usage, VideoGenerationRequest, VideoJobInfo, VideoPart
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

    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True, models=True, images=True, video=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="xai",
        supports=supports,
        auth_modes=("bearer", "xai-oauth"),
        env_keys=("XAI_API_KEY",),
        credential_policy="key-then-oauth",
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

    # ─── Video generation (grok-imagine; captured live 2026-09-01) ──────
    #
    # POST /videos/generations -> {"request_id"}; GET /videos/{id} ->
    # pending + progress %, then done + a PUBLIC MP4 URL (downloads with
    # no auth, verified) — so the result is URL-addressed, no fetch step.
    # There is NO list endpoint (probed: 404): the ticket you store is
    # the only copy.

    _VIDEO_STATUS_MAP: ClassVar[dict[str, str]] = {
        "pending": "running",
        "done": "completed",
        "failed": "failed",
    }

    def _video_submit_request(self, request: VideoGenerationRequest) -> TransportRequest:
        if request.seconds is not None:
            raise UnsupportedFeatureError(
                "xai: video duration has no wire slot", provider=self.provider,
            )
        if request.images:
            # The image-generation wire silently IGNORES unknown fields
            # (pixel-verified 2026-09-01); an unverified image-input mapping
            # here could silently produce prompt-only videos.  Raise until
            # the field is live-receipted.
            raise UnsupportedFeatureError(
                "xai: video input images are not mapped yet; "
                "use extensions until the mapping is live-receipted",
                provider=self.provider,
            )
        payload: dict[str, Any] = {"model": request.model, "prompt": request.prompt, **(request.extensions or {})}
        return make_json_request(
            method="POST", url=f"{self.base_url.rstrip('/')}/videos/generations",
            headers=self._headers(), payload=payload, read_timeout=120.0,
        )

    def _video_job_from_body(self, body: str, video_id: "str | None" = None) -> VideoJobInfo:
        data = json.loads(body)
        request_id = data.get("request_id")
        if isinstance(request_id, str) and request_id:
            # The submit acknowledgement: a bare ticket, not yet started.
            return VideoJobInfo(id=request_id, status="queued", provider_data=data)
        if video_id is None:
            raise ProviderError("xai: video body carries no request_id", provider=self.provider)
        return self._video_status_info(video_id, data)

    def _video_status_request(self, video_id: str) -> TransportRequest:
        return make_json_request(
            method="GET", url=f"{self.base_url.rstrip('/')}/videos/{video_id}",
            headers=self._headers(), read_timeout=60.0,
        )

    def _video_status_info(self, video_id: str, data: "dict[str, Any]") -> VideoJobInfo:
        wire_status = str(data.get("status") or "")
        status = self._VIDEO_STATUS_MAP.get(wire_status)
        if status is None:
            raise ProviderError(f"xai: unknown video status {wire_status!r}", provider=self.provider)
        progress = data.get("progress")
        return VideoJobInfo(
            id=video_id,
            status=status,
            progress=int(progress) if isinstance(progress, (int, float)) and not isinstance(progress, bool) else None,
            model=data.get("model"),
            provider_data=data,
        )

    def _video_list_request(self, limit: int, model: "str | None") -> TransportRequest:
        raise UnsupportedFeatureError(
            "xai: the wire has no video list endpoint (probed 2026-09-01: 404) — "
            "the ticket you stored is the only copy",
            provider=self.provider,
        )

    def _video_result_fetch(self, status_body: "dict[str, Any]") -> None:
        return None  # the terminal body carries a public URL

    def _video_part(self, status_body: "dict[str, Any]", fetched: object) -> VideoPart:
        video = status_body.get("video") if isinstance(status_body.get("video"), dict) else {}
        url = video.get("url")
        if not isinstance(url, str) or not url:
            raise ProviderError("xai: terminal video carries no url", provider=self.provider)
        return VideoPart(media_type="video/mp4", url=url)

    def normalize_error(self, status: int, body: str) -> ProviderError:
        # xAI's own envelope is {"code": str, "error": str} (captured
        # 2026-09-01: model-not-found 400, unauthenticated 401) — refold it
        # into the OpenAI shape so the shared mapping preserves the wire
        # code as provider_code instead of dropping it.
        try:
            data = json.loads(body)
            if isinstance(data, dict) and isinstance(data.get("error"), str):
                body = json.dumps({"error": {"message": data["error"], "code": data.get("code")}})
        except ValueError:
            pass
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
