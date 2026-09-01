from __future__ import annotations

import json
import os
from typing import ClassVar, Iterator

from ..auth import (
    OPENAI_CODEX_LOGIN_HINT,
    extract_chatgpt_account_id,
    get_codex_cli_access_token,
)
from ..errors import (
    NotConfiguredError,
    ProviderError,
    UnsupportedFeatureError,
    UnsupportedModelError,
    map_http_error,
    with_credential_hint,
)
from ..features import EndpointSupport, ProviderManifest
from ..protocols import Capabilities, LiveSession
from ..result import materialize_response
from ..types import (
    SpeechGenerationRequest,
    SpeechGenerationResponse,
    BatchEntry,
    BatchJobInfo,
    BatchRequest,
    FileInfo,
    FilePage,
    FileUploadRequest,
    ImageGenerationRequest,
    ImageGenerationResponse,
    LiveConfig,
    Request,
    Response,
    StreamEvent,
)
from .base import BaseProviderLM, Credential, SyncTransport, default_transport, resolve_credential
from .common import make_json_request, model_infos_from_entries
from .openai import OpenAILM

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"
DEFAULT_CODEX_ORIGINATOR = "lm15"
DEFAULT_CODEX_INSTRUCTIONS = "You are a helpful assistant."
# The backend's /models endpoint requires a client_version query parameter
# (a Codex CLI release); any recent release is accepted.
DEFAULT_CODEX_CLIENT_VERSION = "0.147.0"
MODEL_LIST_HINT = "List the models your subscription accepts: call .list_models() on this client."


class OpenAICodexLM(OpenAILM):
    """OpenAI Responses adapter authenticated with local Codex CLI OAuth."""

    supports: ClassVar[EndpointSupport] = EndpointSupport(complete=True, stream=True, models=True)
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="openai-codex",
        supports=supports,
        auth_modes=("chatgpt-oauth", "bearer-oauth"),
        env_keys=(),
    )
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "document", "binary"}),
        output_modalities=frozenset({"text"}),
        features=frozenset({"streaming", "tools", "json_output", "reasoning"}),
    )

    def __init__(
        self,
        api_key: Credential | None = None,
        *,
        account_id: str | None = None,
        auth_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        originator: str = DEFAULT_CODEX_ORIGINATOR,
        client_version: str = DEFAULT_CODEX_CLIENT_VERSION,
    ) -> None:
        self.client_version = client_version
        if api_key:
            credential: Credential = api_key
            resolved_account_id = account_id or extract_chatgpt_account_id(resolve_credential(api_key))
        else:
            # Validate now — get_codex_cli_access_token raises typed,
            # re-login-guided errors (NotConfiguredError / AuthError) — then
            # re-resolve per request so a long-lived client always sends a
            # fresh token (rotations on disk are picked up, expiry refreshes).
            # The account id is stable across refreshes; resolve it once.
            initial = get_codex_cli_access_token(auth_path)
            resolved_account_id = account_id or initial.account_id or extract_chatgpt_account_id(initial.access_token)

            def credential() -> str:
                return get_codex_cli_access_token(auth_path).access_token

        if not resolved_account_id:
            raise NotConfiguredError(
                "No ChatGPT account id found in the Codex OAuth token.",
                provider="openai-codex",
                credential_hint=OPENAI_CODEX_LOGIN_HINT,
            )
        self.account_id = resolved_account_id
        self.originator = originator
        super().__init__(
            api_key=credential,
            transport=transport or default_transport(),
            base_url=base_url,
            profile=None,
            provider="openai-codex",
            capabilities=self.capabilities,
        )

    @classmethod
    def from_codex_cli(
        cls,
        *,
        auth_path: str | os.PathLike[str] | None = None,
        transport: SyncTransport | None = None,
        base_url: str = DEFAULT_CODEX_BASE_URL,
        originator: str = DEFAULT_CODEX_ORIGINATOR,
    ) -> "OpenAICodexLM":
        return cls(auth_path=auth_path, transport=transport, base_url=base_url, originator=originator)

    def __repr__(self) -> str:  # never leak the OAuth token (dataclass repr would)
        return (
            f"{type(self).__name__}(provider={self.provider!r}, "
            f"base_url={self.base_url!r}, account_id={self.account_id!r}, api_key=<redacted>)"
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        # The ChatGPT Codex backend does not always use the OpenAI error
        # envelope ({"error": {...}}); rejections arrive as {"detail": "..."}
        # (e.g. an unknown model slug -> HTTP 400 with a plain-text reason).
        # Recover that message and classify it first; otherwise fall through
        # to the canonical OpenAI mapping.  Either way, auth failures guide
        # the user to re-login (there is no env var for subscription auth).
        error = self._normalize_detail_error(status, body) or super().normalize_error(status, body)
        return with_credential_hint(error, OPENAI_CODEX_LOGIN_HINT)

    def _normalize_detail_error(self, status: int, body: str) -> ProviderError | None:
        try:
            data = json.loads(body)
        except ValueError:
            return None
        if not isinstance(data, dict):
            return None
        detail = data.get("detail")
        if not isinstance(detail, str) or not detail.strip():
            return None
        detail = detail.strip()
        if self._is_model_error(detail):
            return self._provider_error(
                UnsupportedModelError, f"{detail}\n{MODEL_LIST_HINT}", status=status
            )
        return map_http_error(status, detail, provider=self.provider)

    # ─── Model catalog ────────────────────────────────────────────────

    def _models_request(self):
        return make_json_request(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/models",
            params={"client_version": self.client_version},
            headers=self._headers(),
            read_timeout=30.0,
        )

    def _models_from_body(self, body: str):
        # The Codex backend's usable model names are the `slug` values, and
        # the list lives under "models" (not the OpenAI "data" envelope).
        data = json.loads(body)
        entries = data.get("models") if isinstance(data, dict) else None
        return model_infos_from_entries(
            entries,
            provider=self.provider,
            api_family="openai_responses",
            id_of=lambda entry: entry.get("slug"),
        )

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {resolve_credential(self.api_key)}",
            "Content-Type": content_type,
            "chatgpt-account-id": self.account_id,
            "OpenAI-Beta": "responses=experimental",
            "originator": self.originator,
        }

    def _payload(self, request: Request, stream: bool) -> dict[str, object]:
        payload = super()._payload(request, stream=True)
        payload.setdefault("instructions", DEFAULT_CODEX_INSTRUCTIONS)
        payload["store"] = False
        payload["stream"] = True
        payload.pop("max_output_tokens", None)
        payload.pop("max_completion_tokens", None)
        payload.pop("max_tokens", None)
        return payload

    def complete(self, request: Request) -> Response:
        # The Codex subscription backend is streaming-first.  Materialize the
        # stream so callers get the same synchronous complete() surface.
        return materialize_response(self.stream(request), request)

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        # Bypass OpenAILM.stream's realtime/websocket dispatch and go straight
        # through BaseProviderLM.stream, which applies the MAP-3 coalescer:
        # the Codex backend sends `response.completed` (usage) and then
        # `[DONE]` — two adapter-level end frames that must merge into exactly
        # one final StreamEndEvent with usage intact.
        yield from BaseProviderLM.stream(self, request)

    def live(self, config: LiveConfig) -> LiveSession:
        raise UnsupportedFeatureError("openai-codex: live is not supported", provider=self.provider)

    # Files are an API-key surface; the subscription credential does not
    # carry them. Block every inherited driver, not just upload.
    def file_upload(self, request: FileUploadRequest) -> FileInfo:
        raise UnsupportedFeatureError("openai-codex: files are not supported", provider=self.provider)

    def file_get(self, file_id: str) -> FileInfo:
        raise UnsupportedFeatureError("openai-codex: files are not supported", provider=self.provider)

    def file_list(self, limit: int = 20, cursor: str | None = None) -> FilePage:
        raise UnsupportedFeatureError("openai-codex: files are not supported", provider=self.provider)

    def file_delete(self, file_id: str) -> None:
        raise UnsupportedFeatureError("openai-codex: files are not supported", provider=self.provider)

    def file_download(self, file_id: str) -> bytes:
        raise UnsupportedFeatureError("openai-codex: files are not supported", provider=self.provider)

    # Batch is an API-key surface; the subscription credential does not
    # carry it. Block every inherited driver, not just submit.
    def batch_submit(self, request: BatchRequest) -> BatchJobInfo:
        raise UnsupportedFeatureError("openai-codex: batch is not supported", provider=self.provider)

    def batch_status(self, batch_id: str) -> BatchJobInfo:
        raise UnsupportedFeatureError("openai-codex: batch is not supported", provider=self.provider)

    def batch_results(self, batch_id: str) -> tuple[BatchEntry, ...]:
        raise UnsupportedFeatureError("openai-codex: batch is not supported", provider=self.provider)

    def batch_cancel(self, batch_id: str) -> BatchJobInfo:
        raise UnsupportedFeatureError("openai-codex: batch is not supported", provider=self.provider)

    def batch_list(self, limit: int = 20) -> tuple[BatchJobInfo, ...]:
        raise UnsupportedFeatureError("openai-codex: batch is not supported", provider=self.provider)

    def image_generate(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        raise UnsupportedFeatureError("openai-codex: image generation is not supported", provider=self.provider)

    def speech_generate(self, request: SpeechGenerationRequest) -> SpeechGenerationResponse:
        raise UnsupportedFeatureError("openai-codex: speech generation is not supported", provider=self.provider)

    # Video is an API-key surface; the subscription credential does not
    # carry it.  Block the pure hooks so every driver and mirror raises.
    def _video_submit_request(self, request):
        raise UnsupportedFeatureError("openai-codex: video generation is not supported", provider=self.provider)

    def _video_status_request(self, video_id: str):
        raise UnsupportedFeatureError("openai-codex: video generation is not supported", provider=self.provider)

    def _video_list_request(self, limit: int, model: str | None):
        raise UnsupportedFeatureError("openai-codex: video generation is not supported", provider=self.provider)
