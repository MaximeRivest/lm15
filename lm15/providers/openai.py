from __future__ import annotations

import base64
import json
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

from ..errors import (
    AuthError,
    BillingError,
    ContextLengthError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
    ServerError,
    TimeoutError,
    UnsupportedFeatureError,
    UnsupportedModelError,
    canonical_error_code,
    map_http_error,
)
from ..features import EndpointSupport, ProviderManifest
from ..live import WebSocketLiveSession, require_websocket_sync_connect
from ..profiles import ProviderProfile, ResolvedOpenAIResponsesCompat, resolve_openai_responses_compat
from ..protocols import Capabilities
from ..sse import SSEEvent
from ..transports import TransportRequest
from ..types import (
    continuation_data,
    VideoGenerationRequest,
    VideoJobInfo,
    VideoPart,
    AudioDelta,
    AudioFormat,
    SpeechGenerationRequest,
    SpeechGenerationResponse,
    AudioPart,
    BatchEntry,
    BatchJobInfo,
    BatchRequest,
    BuiltinTool,
    CitationDelta,
    ContinuationState,
    CitationPart,
    ErrorDetail,
    FileInfo,
    FilePage,
    FileUploadRequest,
    FunctionTool,
    ImageDelta,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    LiveClientAudioEvent,
    LiveClientEndAudioEvent,
    LiveClientEvent,
    LiveClientImageEvent,
    LiveClientInterruptEvent,
    LiveClientTextEvent,
    LiveClientToolResultEvent,
    LiveClientTurnEvent,
    LiveConfig,
    LiveServerAudioEvent,
    LiveServerErrorEvent,
    LiveServerInterruptedEvent,
    LiveServerTextEvent,
    LiveServerToolCallDeltaEvent,
    LiveServerToolCallEvent,
    LiveServerTurnEndEvent,
    Message,
    RefusalPart,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from .base import (
    BaseProviderLM,
    Credential,
    HttpResponse,
    SyncTransport,
    batch_entry_http,
    batch_entry_request,
    default_transport,
    resolve_credential,
)
from .common import (
    iso_utc,
    make_json_request,
    model_infos_from_entries,
    multipart_form_body,
    openai_token_logprobs,
    parse_json_object,
    part_to_openai_input,
    parts_to_text,
)

# Canonical builtin tool name → OpenAI Responses API tool type
_OPENAI_BUILTIN_MAP: dict[str, str] = {
    "web_search": "web_search_preview",
    "code_execution": "code_interpreter",
    "file_search": "file_search",
    "computer_use": "computer_use_preview",
}

OPENAI_PROVIDER_EXECUTED_ITEMS = {
    "web_search_call",
    "file_search_call",
    "code_interpreter_call",
    "computer_call",
    "computer_use_call",
}


def _attach_unmapped(provider_data: dict[str, Any], unmapped: list[dict[str, str]]) -> dict[str, Any]:
    if not unmapped:
        return provider_data
    out = dict(provider_data)
    out["_lm15_unmapped"] = unmapped
    return out


_GPT_VERSION_RE = re.compile(r"^gpt-(\d+)\.(\d+)")


def openai_model_has_cache_options(model: str) -> bool:
    """True for the GPT-5.6-and-later model class (MAP-6, mode="off").

    The off switch for cache WRITES is ``prompt_cache_options`` and only
    that class accepts it (older models reject it with HTTP 400, and their
    writes are free anyway).  Deciding client-side needs a model check —
    a table that rots.  Stated trade-off: a future family whose name does
    not match ``gpt-<major>.<minor>`` keeps the provider's implicit writes
    on ``mode="off"``; ``extensions={"prompt_cache_options": ...}`` is the
    override.
    """
    m = _GPT_VERSION_RE.match(model.lower())
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (5, 6)


def _cache_breakpoint_index(request: Request, cache_control: str) -> int | None:
    """Message index that carries the explicit prompt-cache breakpoint.

    ``CacheConfig.prefix_until_index`` marks the end of the reusable prefix.
    Both OpenAI dialects place ``prompt_cache_breakpoint`` on a text
    content block (gpt-5.6+); the mapping is active only when the compat
    policy names OpenAI's cache control (compat servers that declare
    ``cache_control="none"`` get nothing, as for prompt_cache_key).  The
    index is clamped to the last message, the Anthropic adapter's
    precedent.  ``prefix="history"`` sends nothing: OpenAI's implicit
    trailing breakpoint already marks the latest message (MAP-6 A2).
    ``prefix="stable"`` is handled by the system-message rendering, not
    here.
    """
    cache_cfg = request.config.cache
    if cache_cfg is None or cache_cfg.mode == "off" or cache_cfg.prefix_until_index is None:
        return None
    if cache_control != "openai":
        return None
    return min(cache_cfg.prefix_until_index, len(request.messages) - 1)


def _cache_stable_prefix(request: Request, cache_control: str) -> bool:
    """``prefix="stable"``: mark the end of system + tools (MAP-6 A2)."""
    cache_cfg = request.config.cache
    return (
        cache_cfg is not None and cache_cfg.mode != "off" and cache_cfg.prefix == "stable"
        and cache_control == "openai"
    )


def _cache_common_payload(request: Request, payload: dict, cache_control: str, provider: str) -> None:
    """Shared MAP-6 fields for both OpenAI dialects: off switch, key, retention."""
    cache_cfg = request.config.cache
    if cache_cfg is None or cache_control != "openai":
        return
    if cache_cfg.mode == "off":
        # Option 2 (ratified 2026-09-01): the real off switch where the
        # model class has one; nothing where writes are free anyway.
        if openai_model_has_cache_options(request.model):
            payload["prompt_cache_options"] = {"mode": "explicit"}
        return
    if cache_cfg.key:
        payload["prompt_cache_key"] = cache_cfg.key
    if cache_cfg.retention == "long":
        if openai_model_has_cache_options(request.model):
            raise UnsupportedFeatureError(
                f"{provider}: cache.retention='long' is not available on the gpt-5.6+ "
                "model class — prompt_cache_options.ttl accepts only '30m' (the default). "
                "Older OpenAI models take prompt_cache_retention='24h'; Anthropic takes a 1h TTL",
                provider=provider,
            )
        payload["prompt_cache_retention"] = "24h"
    if cache_cfg.resource is not None:
        raise UnsupportedFeatureError(
            f"{provider}: cache.resource is not supported — this provider has no stored-cache "
            "tier; it caches by marks on blocks (prefix / prefix_until_index) and automatically",
            provider=provider,
        )


def _breakpoint_unsupported(provider: str, index: int, role: str) -> UnsupportedFeatureError:
    return UnsupportedFeatureError(
        f"{provider}: cache.prefix_until_index={index} points at a {role} message "
        "whose last block is not text — the wire carries prompt_cache_breakpoint "
        "on text input blocks only. Point the prefix at a user/developer message "
        "that ends with text, or omit prefix_until_index (implicit caching still "
        "applies).",
        provider=provider,
    )


def _record_unmapped(unmapped: list[dict[str, str]], path: str, typ: Any) -> None:
    unmapped.append({"path": path, "type": str(typ or "<missing>")})


def _builtin_to_openai(tool: BuiltinTool) -> dict[str, Any]:
    out: dict[str, Any] = {"type": _OPENAI_BUILTIN_MAP.get(tool.name, tool.name)}
    if tool.config:
        out.update(tool.config)
    return out


def _response_format_to_openai_text(format_config: dict[str, Any]) -> dict[str, Any]:
    """Canonical response_format (INV-050) -> OpenAI Responses `text` config."""
    if format_config["type"] == "json_object":
        return {"format": {"type": "json_object"}}
    fmt: dict[str, Any] = {"type": "json_schema", "name": format_config.get("name") or "response", "schema": format_config["schema"]}
    if "strict" in format_config:
        fmt["strict"] = format_config["strict"]
    return {"format": fmt}


def _finish_from_status(data: dict[str, Any], *, has_tool_call: bool = False) -> str:
    if has_tool_call:
        return "tool_call"
    status = str(data.get("status") or "").lower()
    incomplete = data.get("incomplete_details") or {}
    reason = str(incomplete.get("reason") or "").lower() if isinstance(incomplete, dict) else ""
    if status == "incomplete" and "token" in reason:
        return "length"
    if "content_filter" in reason or "safety" in reason:
        return "content_filter"
    return "stop"


def _openai_batch_status(status: str) -> str:
    """Map an OpenAI Batch status to the canonical BatchStatus."""
    status = status.lower()
    if status in {"completed", "failed", "cancelled", "expired"}:
        return status
    if status in {"cancelling", "canceling"}:
        return "cancelling"
    if status in {"in_progress", "finalizing"}:
        return "running"
    return "queued"  # validating / queued / anything pre-run


def _str_or_none(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _annotation_text(annotation: dict[str, Any], source_text: str | None) -> str | None:
    for key in ("text", "snippet", "cited_text", "quote"):
        text = _str_or_none(annotation.get(key))
        if text is not None:
            return text

    start = _int_or_none(annotation.get("start_index"))
    end = _int_or_none(annotation.get("end_index"))
    if source_text is not None and start is not None and end is not None:
        if 0 <= start < end <= len(source_text):
            return source_text[start:end]
    return None


def _citation_from_openai_annotation(annotation: dict[str, Any], source_text: str | None) -> CitationPart | None:
    url = _str_or_none(annotation.get("url") or annotation.get("uri"))
    title = _str_or_none(
        annotation.get("title")
        or annotation.get("filename")
        or annotation.get("file_id")
    )
    text = _annotation_text(annotation, source_text)
    if url is None and title is None and text is None:
        return None
    return CitationPart(url=url, title=title, text=text)


def _citation_delta_from_openai_annotation(
    annotation: dict[str, Any],
    *,
    part_index: int,
    source_text: str | None = None,
) -> CitationDelta | None:
    citation = _citation_from_openai_annotation(annotation, source_text)
    if citation is None:
        return None
    return CitationDelta(
        text=citation.text,
        url=citation.url,
        title=citation.title,
        part_index=part_index,
    )


@dataclass(slots=True)
class OpenAILM(BaseProviderLM):
    api_key: Credential = field(repr=False)
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = "https://api.openai.com/v1"
    profile: ProviderProfile | None = None

    provider: str = "openai"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text", "image", "audio", "video", "document", "binary"}),
        output_modalities=frozenset({"text", "audio", "image"}),
        features=frozenset({"streaming", "tools", "json_output", "reasoning", "live", "files", "batch", "images", "audio"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(
        complete=True,
        stream=True,
        live=True,
        files=True,
        batches=True,
        images=True,
        speech=True,
        video=True,
        responses_api=True,
        models=True,
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="openai",
        supports=supports,
        auth_modes=("bearer",),
        enterprise_variants=("azure-openai",),
        env_keys=("OPENAI_API_KEY",),
    )

    _response_error_code_map: ClassVar[dict[str, type[ProviderError]]] = {
        "server_error": ServerError,
        "rate_limit_exceeded": RateLimitError,
        "invalid_prompt": InvalidRequestError,
        "vector_store_timeout": TimeoutError,
        "invalid_image": InvalidRequestError,
        "invalid_image_format": InvalidRequestError,
        "invalid_base64_image": InvalidRequestError,
        "invalid_image_url": InvalidRequestError,
        "image_too_large": InvalidRequestError,
        "image_too_small": InvalidRequestError,
        "image_parse_error": InvalidRequestError,
        "image_content_policy_violation": InvalidRequestError,
        "invalid_image_mode": InvalidRequestError,
        "image_file_too_large": InvalidRequestError,
        "unsupported_image_media_type": InvalidRequestError,
        "empty_image_file": InvalidRequestError,
        "failed_to_download_image": InvalidRequestError,
        "image_file_not_found": InvalidRequestError,
        "model_not_found": UnsupportedModelError,
        "model_not_available": UnsupportedModelError,
        "unsupported_model": UnsupportedModelError,
    }

    _model_error_codes: ClassVar[frozenset[str]] = frozenset(
        {"model_not_found", "model_not_available", "unsupported_model"}
    )

    _stream_error_code_map: ClassVar[dict[str, type[ProviderError]]] = {
        **_response_error_code_map,
        "context_length_exceeded": ContextLengthError,
        "invalid_api_key": AuthError,
        "insufficient_quota": BillingError,
        "authentication_error": AuthError,
        "rate_limit_error": RateLimitError,
    }

    @classmethod
    def from_profile(
        cls,
        *,
        api_key: Credential,
        profile: ProviderProfile,
        transport: SyncTransport | None = None,
    ) -> "OpenAILM":
        endpoint = profile.endpoint("inference")
        base_url = endpoint.base_url if endpoint and endpoint.base_url else "https://api.openai.com/v1"
        return cls(
            api_key=api_key,
            transport=transport or default_transport(),
            base_url=base_url,
            profile=profile,
        )

    def _headers(self, content_type: str = "application/json") -> dict[str, str]:
        return {
            "Authorization": f"Bearer {resolve_credential(self.api_key)}",
            "Content-Type": content_type,
        }

    @staticmethod
    def _is_model_error(message: str, *codes: str) -> bool:
        lowered = " ".join(value for value in (message, *codes) if value).lower()
        return "model" in lowered and any(
            marker in lowered
            for marker in (
                "not found",
                "does not exist",
                "not exist",
                "not supported",
                "unsupported",
                "not available",
                "unknown",
            )
        )

    def _response_error(self, code: str, message: str) -> ProviderError:
        cls = self._response_error_code_map.get(code, ServerError)
        msg = message or code or "provider error"
        return self._provider_error(cls, msg, provider_code=code or None)

    def _error_detail(self, provider_code: str, message: str) -> ErrorDetail:
        cls = self._stream_error_code_map.get(provider_code, ProviderError)
        return ErrorDetail(
            code=canonical_error_code(cls),
            message=message or provider_code or "provider error",
            provider_code=provider_code or "provider",
        )

    def normalize_error(self, status: int, body: str) -> ProviderError:
        """Extract message from OpenAI error shape."""
        try:
            data = json.loads(body)
            err = data.get("error", {}) if isinstance(data, dict) else {}
            msg = err.get("message", "") if isinstance(err, dict) else str(err)
            code = str(err.get("code") or "") if isinstance(err, dict) else ""
            err_type = str(err.get("type") or "") if isinstance(err, dict) else ""

            provider_code = code or err_type or None

            if code == "context_length_exceeded":
                return self._provider_error(
                    ContextLengthError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code in self._model_error_codes or (
                status == 404 and self._is_model_error(msg, code, err_type)
            ):
                return self._provider_error(
                    UnsupportedModelError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code == "insufficient_quota" or err_type == "insufficient_quota":
                return self._provider_error(
                    BillingError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code == "invalid_api_key" or err_type == "authentication_error":
                return self._provider_error(
                    AuthError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code == "rate_limit_exceeded" or err_type == "rate_limit_error":
                return self._provider_error(
                    RateLimitError,
                    msg,
                    status=status,
                    provider_code=provider_code,
                )
            if code and code not in msg:
                msg = f"{msg} ({code})"
        except Exception:
            msg = body.strip()[:500] or f"HTTP {status}"
            provider_code = None
        return map_http_error(
            status,
            msg,
            provider=self.provider,
            env_keys=self.manifest.env_keys,
            provider_code=provider_code,
        )

    # ─── Request serialization ──────────────────────────────────────

    def _compat(self, request: Request) -> ResolvedOpenAIResponsesCompat:
        return resolve_openai_responses_compat(
            base_url=self.base_url,
            model=request.model,
            profile=self.profile,
            request_extensions=request.config.extensions,
        )

    def _build_input(
        self,
        messages: tuple[Message, ...],
        compat: ResolvedOpenAIResponsesCompat,
        *,
        breakpoint_index: int | None = None,
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for msg_index, msg in enumerate(messages):
            if msg_index == breakpoint_index and msg.role in ("assistant", "tool"):
                raise _breakpoint_unsupported(self.provider, msg_index, msg.role)
            if msg.role == "tool":
                for part in msg.parts:
                    if isinstance(part, ToolResultPart):
                        output = parts_to_text(part.content)
                        if not output:
                            output = json.dumps([{"type": p.type} for p in part.content])
                        item = {
                            "type": "function_call_output",
                            "call_id": part.id,
                            "output": output,
                        }
                        if compat.tool_result_name == "include" and part.name:
                            item["name"] = part.name
                        items.append(item)
                continue

            if msg.role == "assistant":
                content_parts = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        content_parts.append({"type": "output_text", "text": part.text})
                    elif isinstance(part, RefusalPart):
                        content_parts.append({"type": "refusal", "refusal": part.text})
                    elif isinstance(part, ThinkingPart):
                        state = continuation_data(part, "openai", "reasoning_item")
                        if state:
                            # Native replay (MAP-7 rule 8): the reasoning item
                            # goes back as its own input item, before the
                            # message it preceded.
                            item = {"type": "reasoning", **{k: v for k, v in state.items() if k in ("id", "encrypted_content")}}
                            # `summary` is required on a replayed item, even
                            # empty (HTTP 400 "Missing required parameter:
                            # input[1].summary", live 2026-09-02).
                            item["summary"] = [{"type": "summary_text", "text": part.text}] if part.text else []
                            items.append(item)
                        elif part.text:
                            # No native state: replay as assistant text
                            # (decision G, 2026-09-01) rather than drop it.
                            content_parts.append({"type": "output_text", "text": part.text})
            else:
                content_parts = [
                    part_to_openai_input(part)
                    for part in msg.parts
                    if not isinstance(part, (ToolCallPart, ToolResultPart))
                ]
            if msg_index == breakpoint_index:
                # CacheConfig.prefix_until_index -> an explicit prompt-cache
                # breakpoint on the last text block of that message (the
                # Anthropic cache_control precedent).  The Responses wire
                # carries prompt_cache_breakpoint on input_text blocks only
                # (gpt-5.6+, verified live 2026-09-01: cache_write_tokens on
                # the first call, cached_tokens on the next; pre-5.6 models
                # reject with HTTP 400 — that loud failure is the contract).
                if not content_parts or content_parts[-1].get("type") != "input_text":
                    raise _breakpoint_unsupported(self.provider, msg_index, msg.role)
                content_parts[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
            if content_parts:
                role = msg.role
                if role == "developer":
                    role = compat.developer_role
                items.append({"role": role, "content": content_parts})

            for part in msg.parts:
                if isinstance(part, ToolCallPart):
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": part.id,
                            "name": part.name,
                            "arguments": json.dumps(part.input, separators=(",", ":")),
                        }
                    )
        return items

    def _tool_choice_payload(self, request: Request) -> Any:
        tc = request.config.tool_choice
        if tc is None:
            return None
        if tc.mode == "none":
            return "none"
        if tc.allowed:
            # Resolve names against Request.tools (INV-031 guarantees
            # presence) and emit the kind-correct wire form.  Verified live
            # 2026-09-01: {"type": "web_search_preview"} forcing and mixed
            # allowed_tools both accepted.
            by_name = {t.name: t for t in request.tools}
            entries = [by_name[name] for name in tc.allowed]
            if len(entries) == 1 and tc.mode == "required":
                tool = entries[0]
                if isinstance(tool, BuiltinTool):
                    return {"type": _OPENAI_BUILTIN_MAP.get(tool.name, tool.name)}
                return {"type": "function", "name": tool.name}
            # mode="auto" restriction or multi-tool subset: allowed_tools
            # keeps auto semantics honest — the old single-name mapping
            # forced a call even when the caller said mode="auto".
            wire_tools: list[dict[str, Any]] = []
            for tool in entries:
                if isinstance(tool, BuiltinTool):
                    wire_tools.append({"type": _OPENAI_BUILTIN_MAP.get(tool.name, tool.name)})
                else:
                    wire_tools.append({"type": "function", "name": tool.name})
            return {"type": "allowed_tools", "mode": tc.mode, "tools": wire_tools}
        if tc.mode == "required":
            return "required"
        return "auto"

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        compat = self._compat(request)
        payload: dict[str, Any] = {
            "model": request.model,
            "input": self._build_input(
                request.messages,
                compat,
                breakpoint_index=_cache_breakpoint_index(request, compat.cache_control),
            ),
            "stream": stream,
        }
        if request.system:
            system_text = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            if _cache_stable_prefix(request, compat.cache_control):
                # Top-level `instructions` cannot carry a breakpoint
                # (prompt-caching guide: "place them in an input_text block
                # inside a developer message").  The stable-prefix intent
                # therefore renders the system prompt as the first input
                # item with the mark on it.
                payload["input"].insert(0, {
                    "role": compat.developer_role,
                    "content": [{"type": "input_text", "text": system_text, "prompt_cache_breakpoint": {"mode": "explicit"}}],
                })
            else:
                payload["instructions"] = system_text
        if request.config.max_tokens is not None:
            payload[compat.max_output_tokens_field] = request.config.max_tokens
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.config.top_p is not None:
            payload["top_p"] = request.config.top_p
        if request.config.logprobs is not None:
            # Verified live 2026-09-01: include triggers per-token logprobs;
            # top_logprobs (0–20) controls the alternatives count.
            payload["top_logprobs"] = request.config.logprobs
            payload["include"] = ["message.output_text.logprobs"]
        if request.tools:
            tools_wire: list[dict[str, Any]] = []
            for tool in request.tools:
                if isinstance(tool, FunctionTool):
                    tool_payload = {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    if compat.strict_tools == "include":
                        tool_payload["strict"] = False
                    tools_wire.append(tool_payload)
                elif isinstance(tool, BuiltinTool):
                    tools_wire.append(_builtin_to_openai(tool))
            payload["tools"] = tools_wire
        tool_choice = self._tool_choice_payload(request)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if request.config.tool_choice and request.config.tool_choice.parallel is not None:
            payload["parallel_tool_calls"] = request.config.tool_choice.parallel
        if request.config.response_format:
            payload["text"] = _response_format_to_openai_text(request.config.response_format)
        if request.config.reasoning:
            reasoning = request.config.reasoning
            if not reasoning.is_off:
                # MAP-7: the word goes verbatim; models reject unsupported
                # levels with a 400 that lists the supported set (live
                # 2026-09-02: gpt-5.6-sol rejects minimal, gpt-5.4-mini
                # rejects max).  No budget exists on this wire.
                if reasoning.thinking_budget is not None:
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.thinking_budget is not supported — this wire "
                        "has no thinking token budget; use effort (Anthropic's manual class and "
                        "Gemini take a budget)",
                        provider=self.provider,
                    )
                effort = reasoning.effort
                if reasoning.summary in ("concise", "detailed") and compat.reasoning_format != "responses_reasoning":
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.summary={reasoning.summary!r} is an OpenAI Responses "
                        "detail level; this wire has no summary levels (use 'auto')",
                        provider=self.provider,
                    )
                if compat.reasoning_format == "responses_reasoning":
                    reasoning_payload: dict[str, Any] = {"effort": effort}
                    if reasoning.summary is not None:
                        reasoning_payload["summary"] = reasoning.summary
                    payload["reasoning"] = reasoning_payload
                elif compat.reasoning_format == "reasoning_effort":
                    payload["reasoning_effort"] = effort
                elif compat.reasoning_format == "openrouter":
                    payload["reasoning"] = {"effort": effort}
                elif compat.reasoning_format == "deepseek":
                    payload["thinking"] = {"type": "enabled"}
                    payload["reasoning_effort"] = effort
                elif compat.reasoning_format in {"qwen", "zai"}:
                    payload["enable_thinking"] = True
                elif compat.reasoning_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {
                        "enable_thinking": True,
                        "preserve_thinking": True,
                    }
            else:
                # Explicit off must reach the wire.  Sending nothing lets
                # reasoning-by-default models (gpt-5 family, o-series) burn
                # hidden reasoning tokens the caller asked to disable —
                # verified live 2026-09-01: gpt-5-mini spent 64 reasoning
                # tokens on a request with effort="off" when the field was
                # omitted.  Models whose floor is "minimal" reject
                # "none" with a clear 400; that loud failure is deliberate
                # (a silent default-effort run spends money on hidden
                # reasoning tokens the caller asked to disable).
                if compat.reasoning_format == "responses_reasoning":
                    payload["reasoning"] = {"effort": "none"}
                elif compat.reasoning_format == "reasoning_effort":
                    payload["reasoning_effort"] = "none"
                elif compat.reasoning_format == "openrouter":
                    payload["reasoning"] = {"enabled": False}
                elif compat.reasoning_format == "deepseek":
                    payload["thinking"] = {"type": "disabled"}
                elif compat.reasoning_format in {"qwen", "zai"}:
                    payload["enable_thinking"] = False
                elif compat.reasoning_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Prompt caching (MAP-6): off switch, key, retention, resource.
        _cache_common_payload(request, payload, compat.cache_control, self.provider)

        if compat.routing is not None:
            payload["provider"] = compat.routing

        # Promoted cross-provider knobs (changes/2026-09-01-extensions-burn-down):
        # canonical spelling in, provider spelling out. user_id maps to
        # safety_identifier — OpenAI's current end-user attribution field
        # (`user` is the deprecated legacy spelling; still available verbatim
        # through extensions).
        if request.config.service_tier is not None:
            payload["service_tier"] = request.config.service_tier
        if request.config.user_id is not None:
            payload["safety_identifier"] = request.config.user_id
        if request.config.store is not None:
            payload["store"] = request.config.store

        if request.config.extensions:
            reserved = {
                "prompt_caching",
                "cache",
                "compat",
                "openai_compat",
                "openai_responses_compat",
            }
            passthrough = {k: v for k, v in request.config.extensions.items() if k not in reserved}
            payload.update(passthrough)
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        return make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/responses",
            headers=self._headers(),
            payload=self._payload(request, stream=stream),
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    def parse_response(self, request: Request, response: HttpResponse) -> Response:
        data = response.json()

        resp_error = data.get("error") if isinstance(data, dict) else None
        if isinstance(resp_error, dict):
            raise self._response_error(
                str(resp_error.get("code") or ""),
                str(resp_error.get("message") or resp_error),
            )

        parts: list[Any] = []
        unmapped: list[dict[str, str]] = []
        logprob_seq: list[Any] = []
        for item_index, item in enumerate(data.get("output", []) or []):
            if not isinstance(item, dict):
                _record_unmapped(unmapped, f"output[{item_index}]", type(item).__name__)
                continue
            item_type = item.get("type")
            if item_type == "message":
                for content_index, content in enumerate(item.get("content", []) or []):
                    if not isinstance(content, dict):
                        _record_unmapped(unmapped, f"output[{item_index}].content[{content_index}]", type(content).__name__)
                        continue
                    ctype = content.get("type")
                    if ctype in ("output_text", "text"):
                        text = str(content.get("text") or "")
                        parts.append(TextPart(text=text))
                        # Per-block wire lists concatenate in document order
                        # into the message-level canonical sequence.
                        logprob_seq.extend(openai_token_logprobs(content.get("logprobs")))
                        for annotation in content.get("annotations", []) or []:
                            if not isinstance(annotation, dict):
                                continue
                            citation = _citation_from_openai_annotation(annotation, text)
                            if citation is not None:
                                parts.append(citation)
                    elif ctype == "refusal":
                        text = str(content.get("refusal") or content.get("text") or "")
                        parts.append(RefusalPart(text=text) if text else TextPart(text=""))
                    elif ctype == "output_image":
                        b64 = content.get("b64_json") or content.get("image_base64") or ""
                        if b64:
                            parts.append(ImagePart(media_type="image/png", data=str(b64)))
                    elif ctype == "output_audio":
                        audio_payload = content.get("audio") if isinstance(content.get("audio"), dict) else {}
                        b64 = audio_payload.get("data") or content.get("b64_json") or ""
                        if b64:
                            parts.append(AudioPart(media_type="audio/wav", data=str(b64)))
                    else:
                        _record_unmapped(unmapped, f"output[{item_index}].content[{content_index}]", ctype)
            elif item_type == "function_call":
                parts.append(
                    ToolCallPart(
                        id=str(item.get("call_id") or item.get("id") or f"call_{len(parts)}"),
                        name=str(item.get("name") or "tool"),
                        input=parse_json_object(item.get("arguments")),
                    )
                )
            elif item_type == "reasoning":
                # MAP-7 rule 8: the reasoning item is replay state.  Its
                # summary (when requested) is the visible text; its id and
                # encrypted_content ride as continuation so the next turn
                # can send the item back verbatim (live 2026-09-02:
                # encrypted_content is present on every reasoning item).
                summary = item.get("summary")
                if isinstance(summary, list):
                    text = "\n".join(str(x.get("text") if isinstance(x, dict) else x) for x in summary)
                else:
                    text = str(summary or item.get("text") or "")
                state: dict[str, Any] = {}
                if item.get("id"):
                    state["id"] = str(item["id"])
                if item.get("encrypted_content"):
                    state["encrypted_content"] = str(item["encrypted_content"])
                continuation = (ContinuationState(provider="openai", kind="reasoning_item", data=state),) if state else ()
                if text or continuation:
                    parts.append(ThinkingPart(text=text, continuation=continuation))
            elif item_type in OPENAI_PROVIDER_EXECUTED_ITEMS:
                continue
            else:
                _record_unmapped(unmapped, f"output[{item_index}]", item_type)

        if not parts:
            parts = [TextPart(text=str(data.get("output_text") or ""))]

        usage_data = data.get("usage", {}) or {}
        input_details = usage_data.get("input_tokens_details") or {}
        output_details = usage_data.get("output_tokens_details") or {}
        usage = Usage(
            input_tokens=int(usage_data.get("input_tokens", 0) or 0),
            output_tokens=int(usage_data.get("output_tokens", 0) or 0),
            total_tokens=usage_data.get("total_tokens"),
            reasoning_tokens=output_details.get("reasoning_tokens"),
            cache_read_tokens=input_details.get("cached_tokens"),
            cache_write_tokens=input_details.get("cache_write_tokens"),
            input_audio_tokens=input_details.get("audio_tokens"),
            output_audio_tokens=output_details.get("audio_tokens"),
        )

        has_tool = any(isinstance(part, ToolCallPart) for part in parts)
        message_continuation: tuple[ContinuationState, ...] = ()
        if data.get("id"):
            message_continuation = (
                ContinuationState(
                    provider="openai",
                    kind="response_id",
                    data={"id": str(data.get("id"))},
                ),
            )
        return Response(
            id=str(data.get("id")) if data.get("id") else None,
            model=str(data.get("model") or request.model),
            message=Message(role="assistant", parts=tuple(parts), continuation=message_continuation),
            finish_reason=_finish_from_status(data, has_tool_call=has_tool),
            usage=usage,
            logprobs=tuple(logprob_seq) if logprob_seq else None,
            provider_data=_attach_unmapped(data, unmapped),
        )

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        event = self._parse_single_stream_event(request, raw_event)
        if event is not None:
            yield event

    def _parse_single_stream_event(self, request: Request, raw_event: SSEEvent) -> StreamEvent | None:
        if not raw_event.data:
            return None
        if raw_event.data == "[DONE]":
            return StreamEndEvent(finish_reason="stop")
        payload = json.loads(raw_event.data)
        et = str(payload.get("type") or "")

        if et == "response.created":
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            return StreamStartEvent(
                id=str(response.get("id")) if response.get("id") else None,
                model=str(response.get("model") or request.model),
            )

        if et in {"response.output_text.delta", "response.refusal.delta"}:
            return StreamDeltaEvent(
                delta=TextDelta(
                    text=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                    # Verified live 2026-09-01: each output_text.delta event
                    # carries the logprobs for exactly its own tokens.
                    logprobs=openai_token_logprobs(payload.get("logprobs")),
                )
            )

        if et in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            return StreamDeltaEvent(
                delta=ThinkingDelta(
                    text=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                )
            )

        if et == "response.output_text.annotation.added":
            annotation = payload.get("annotation")
            if isinstance(annotation, dict):
                delta = _citation_delta_from_openai_annotation(
                    annotation,
                    part_index=int(payload.get("output_index", 0) or 0),
                )
                if delta is not None:
                    return StreamDeltaEvent(delta=delta)
            return None

        if et == "response.output_audio.delta":
            return StreamDeltaEvent(
                delta=AudioDelta(
                    data=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                    media_type="audio/wav",
                )
            )

        if et in {"response.output_image.delta", "response.image.delta"}:
            return StreamDeltaEvent(
                delta=ImageDelta(
                    data=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                    media_type="image/png",
                )
            )

        if et == "response.output_item.added":
            item = payload.get("item", {}) if isinstance(payload.get("item"), dict) else {}
            if item.get("type") == "function_call":
                return StreamDeltaEvent(
                    delta=ToolCallDelta(
                        input=str(item.get("arguments") or ""),
                        part_index=int(payload.get("output_index", 0) or 0),
                        id=str(item.get("call_id") or item.get("id") or "") or None,
                        name=str(item.get("name") or "") or None,
                    )
                )
            return None

        if et == "response.function_call_arguments.delta":
            return StreamDeltaEvent(
                delta=ToolCallDelta(
                    input=str(payload.get("delta") or ""),
                    part_index=int(payload.get("output_index", 0) or 0),
                    id=str(payload.get("call_id") or payload.get("id") or "") or None,
                    name=str(payload.get("name") or "") or None,
                )
            )

        if et == "response.completed":
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            usage_data = response.get("usage", {}) if isinstance(response, dict) else {}
            input_details = usage_data.get("input_tokens_details") or {}
            output_details = usage_data.get("output_tokens_details") or {}
            usage = Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=output_details.get("reasoning_tokens"),
                cache_read_tokens=input_details.get("cached_tokens"),
                cache_write_tokens=input_details.get("cache_write_tokens"),
                input_audio_tokens=input_details.get("audio_tokens"),
                output_audio_tokens=output_details.get("audio_tokens"),
            )
            output = response.get("output", []) if isinstance(response, dict) else []
            has_tool = any(isinstance(item, dict) and item.get("type") == "function_call" for item in output)
            return StreamEndEvent(
                finish_reason="tool_call" if has_tool else "stop",
                usage=usage,
                provider_data=response if isinstance(response, dict) else None,
            )

        if et in {"response.error", "error"}:
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("code") or err.get("type") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            return StreamErrorEvent(error=self._error_detail(provider_code, message))

        return None

    # ─── Streaming over OpenAI Realtime for live models ──────────────

    def stream(self, request: Request) -> Iterator[StreamEvent]:
        if self._should_use_live_completion(request):
            yield from self._stream_via_live_completion(request)
            return
        yield from BaseProviderLM.stream(self, request)

    def _should_use_live_completion(self, request: Request) -> bool:
        extensions = request.config.extensions or {}
        transport_mode = str(extensions.get("transport") or "").lower()
        if transport_mode in {"live", "websocket", "ws"}:
            return True
        model_name = request.model.lower()
        return "realtime" in model_name or "-live" in model_name

    def _stream_via_live_completion(self, request: Request) -> Iterator[StreamEvent]:
        ws = self._live_connect(self._live_url(request.model), self._live_headers())
        saw_tool_call = False
        usage = Usage()
        try:
            ws.send(json.dumps(self._live_session_update_from_request(request)))
            for frame in self._live_message_frames_for_request(request):
                ws.send(json.dumps(frame))

            yield StreamStartEvent(model=request.model)
            while True:
                raw = ws.recv()
                for event in self._decode_live_completion_stream_events(request, raw):
                    if event.type == "delta":
                        if isinstance(event.delta, ToolCallDelta):
                            saw_tool_call = True
                        yield event
                    elif event.type == "error":
                        yield event
                        return
                    elif event.type == "end":
                        if event.usage is not None:
                            usage = event.usage
                        yield StreamEndEvent(
                            finish_reason="tool_call" if saw_tool_call else (event.finish_reason or "stop"),
                            usage=usage,
                        )
                        return
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _live_session_update_from_request(self, request: Request) -> dict[str, Any]:
        extensions = dict(request.config.extensions or {})
        extensions.pop("transport", None)
        extensions.pop("prompt_caching", None)
        extensions.pop("output", None)
        config = LiveConfig(
            model=request.model,
            system=request.system,
            tools=request.tools,
            extensions=extensions or None,
        )
        return self._live_session_update_payload(config)

    def _live_message_frames_for_request(self, request: Request) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []
        for message in request.messages:
            if message.role == "tool":
                for part in message.parts:
                    if not isinstance(part, ToolResultPart):
                        continue
                    output = parts_to_text(part.content) or json.dumps([{"type": p.type} for p in part.content])
                    frames.append({"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": part.id, "output": output}})
                continue

            content = [part_to_openai_input(p) for p in message.parts if not isinstance(p, (ToolCallPart, ToolResultPart))]
            if content:
                frames.append({"type": "conversation.item.create", "item": {"type": "message", "role": message.role, "content": content}})
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    frames.append({"type": "conversation.item.create", "item": {"type": "function_call", "call_id": part.id, "name": part.name, "arguments": json.dumps(part.input)}})

        response_create: dict[str, Any] = {"type": "response.create"}
        if (request.config.extensions or {}).get("output") == "audio":
            response_create["response"] = {"output_modalities": ["audio"]}  # GA name
        frames.append(response_create)
        return frames

    def _decode_live_completion_stream_events(self, request: Request, raw: str | bytes) -> list[StreamEvent]:
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        et = str(payload.get("type") or "")

        if et in {"response.output_text.delta", "response.text.delta",
                  "response.output_audio_transcript.delta", "response.audio_transcript.delta"}:
            delta = str(payload.get("delta") or payload.get("text") or "")
            return [StreamDeltaEvent(delta=TextDelta(text=delta))] if delta else []
        if et == "response.output_audio.delta":
            delta = str(payload.get("delta") or "")
            return [StreamDeltaEvent(delta=AudioDelta(data=delta, media_type="audio/wav"))] if delta else []
        if et in {"response.output_item.added", "response.function_call_arguments.delta", "response.function_call_arguments.done", "response.output_item.done"}:
            if et in {"response.output_item.added", "response.output_item.done"}:
                item = payload.get("item", {}) if isinstance(payload.get("item"), dict) else {}
                if item.get("type") != "function_call":
                    return []
                call_id = str(item.get("call_id") or item.get("id") or "")
                name = str(item.get("name") or "tool")
                arguments = item.get("arguments") or ""
            else:
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                name = str(payload.get("name") or "tool")
                arguments = payload.get("delta") if et.endswith("delta") else payload.get("arguments")
            return [StreamDeltaEvent(delta=ToolCallDelta(input=arguments if isinstance(arguments, str) else json.dumps(arguments or {}), id=call_id or None, name=name))]
        if et in {"response.done", "response.completed"}:
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            usage_data = response.get("usage", {}) if isinstance(response, dict) else {}
            u_in = usage_data.get("input_token_details") or usage_data.get("input_tokens_details") or {}
            u_out = usage_data.get("output_token_details") or usage_data.get("output_tokens_details") or {}
            usage = Usage(
                input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                total_tokens=usage_data.get("total_tokens"),
                reasoning_tokens=u_out.get("reasoning_tokens"),
                cache_read_tokens=u_in.get("cached_tokens"),
                cache_write_tokens=u_in.get("cache_write_tokens"),
                input_audio_tokens=u_in.get("audio_tokens"),
                output_audio_tokens=u_out.get("audio_tokens"),
            )
            return [StreamEndEvent(finish_reason="stop", usage=usage, provider_data=response if isinstance(response, dict) else None)]
        if et in {"error", "response.error"}:
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("code") or err.get("type") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            return [StreamErrorEvent(error=self._error_detail(provider_code, message))]
        return []

    # ─── Live sessions ──────────────────────────────────────────────

    def live(self, config: LiveConfig):
        ws = self._live_connect(self._live_url(config.model), self._live_headers())
        for frame in self._live_setup_frames(config):
            ws.send(json.dumps(frame))

        return WebSocketLiveSession(
            ws=ws,
            encode_event=self._live_encoder(config),
            decode_event=self._decode_live_server_event,
        )

    # Pure live-codec hooks (uniform across providers; the vet shim's
    # replay_live op and the async twin drive these, never the socket).

    def _live_setup_frames(self, config: LiveConfig) -> list[dict[str, Any]]:
        return [self._live_session_update_payload(config)]

    def _live_encoder(self, config: LiveConfig):
        return self._encode_live_client_event

    def _live_connect(self, url: str, headers: dict[str, str]):
        connect = require_websocket_sync_connect()
        return connect(url, additional_headers=headers)

    def _live_url(self, model: str) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        base_path = parsed.path.rstrip("/")
        path = f"{base_path}/realtime" if base_path else "/realtime"
        query = urllib.parse.urlencode({"model": model})
        return urllib.parse.urlunparse((scheme, parsed.netloc, path, "", query, ""))

    def _live_headers(self) -> dict[str, str]:
        # GA Realtime: the beta header now HARD-CLOSES the socket (4000
        # `beta_api_shape_disabled`, observed live 2026-09-01).
        return {"Authorization": f"Bearer {resolve_credential(self.api_key)}"}

    @staticmethod
    def _live_audio_format(fmt: AudioFormat) -> dict[str, Any]:
        # GA wire format objects: {"type": "audio/pcm", "rate": N} (pcmu/
        # pcma carry no rate). Canonical pcm16 maps to audio/pcm.
        if fmt.encoding == "pcm16":
            return {"type": "audio/pcm", "rate": fmt.sample_rate}
        return {"type": f"audio/{fmt.encoding}"}

    def _live_session_update_payload(self, config: LiveConfig) -> dict[str, Any]:
        # GA Realtime session shape (verified live 2026-09-01,
        # curl-fixtures/live-2026-09-01/): session.type is required,
        # `output_modalities` replaces `modalities`, and audio config
        # nests under session.audio.{input,output}.
        session: dict[str, Any] = {"type": "realtime"}
        if config.system:
            session["instructions"] = config.system if isinstance(config.system, str) else parts_to_text(config.system)
        audio: dict[str, Any] = {}
        if config.output_format is not None or config.voice:
            session["output_modalities"] = ["audio"]
            output: dict[str, Any] = {}
            if config.output_format is not None:
                output["format"] = self._live_audio_format(config.output_format)
            if config.voice:
                output["voice"] = config.voice
            audio["output"] = output
        else:
            session["output_modalities"] = ["text"]
        if config.input_format is not None:
            # turn_detection null = server VAD OFF: a turn happens exactly
            # when the caller sends end_audio() (commit + response.create),
            # deterministic library behavior. Re-enable VAD through
            # extensions when you want the server to segment speech.
            audio["input"] = {"format": self._live_audio_format(config.input_format), "turn_detection": None}
        if audio:
            session["audio"] = audio
        if config.tools:
            session["tools"] = [
                {"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters}
                for t in config.tools
                if isinstance(t, FunctionTool)
            ]
        if config.extensions:
            session.update(config.extensions)
        return {"type": "session.update", "session": session}

    def _encode_live_client_event(self, event: LiveClientEvent) -> list[dict[str, Any]]:
        if isinstance(event, LiveClientAudioEvent):
            return [{"type": "input_audio_buffer.append", "audio": event.data}]
        if isinstance(event, LiveClientEndAudioEvent):
            return [{"type": "input_audio_buffer.commit"}, {"type": "response.create"}]
        if isinstance(event, LiveClientInterruptEvent):
            return [{"type": "response.cancel"}]
        if isinstance(event, LiveClientTextEvent):
            return [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": event.text}]}},
                {"type": "response.create"},
            ]
        if isinstance(event, LiveClientTurnEvent):
            return [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [part_to_openai_input(part) for part in event.parts]}},
                {"type": "response.create"},
            ] if event.turn_complete else [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [part_to_openai_input(part) for part in event.parts]}},
            ]
        if isinstance(event, LiveClientImageEvent):
            return [
                {"type": "conversation.item.create", "item": {"type": "message", "role": "user", "content": [{"type": "input_image", "image_url": f"data:{event.media_type};base64,{event.data}"}]}},
                {"type": "response.create"},
            ]
        if isinstance(event, LiveClientToolResultEvent):
            output = parts_to_text(event.content) or json.dumps([{"type": p.type} for p in event.content])
            return [
                {"type": "conversation.item.create", "item": {"type": "function_call_output", "call_id": event.id, "output": output}},
                {"type": "response.create"},
            ]
        return []

    def _decode_live_server_event(self, raw: str | bytes):
        try:
            payload = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            return []
        if not isinstance(payload, dict):
            return []
        et = str(payload.get("type") or "")
        events: list[Any] = []
        # GA server event names verified live 2026-09-01
        # (curl-fixtures/live-2026-09-01/); legacy beta names kept where
        # harmless. Audio-native turns speak through
        # `response.output_audio_transcript.delta` — mapped to text so
        # transcripts arrive uniformly across providers.
        if et in {"response.output_text.delta", "response.text.delta",
                  "response.output_audio_transcript.delta", "response.audio_transcript.delta"}:
            delta = str(payload.get("delta") or payload.get("text") or "")
            if delta:
                events.append(LiveServerTextEvent(text=delta))
        elif et == "response.output_audio.delta":
            delta = str(payload.get("delta") or "")
            if delta:
                events.append(LiveServerAudioEvent(data=delta))
        elif et == "response.function_call_arguments.delta":
            delta = str(payload.get("delta") or "")
            if delta:
                events.append(LiveServerToolCallDeltaEvent(input_delta=delta, id=str(payload.get("call_id") or payload.get("id") or "") or None, name=str(payload.get("name") or "") or None))
        elif et == "response.output_item.done":
            # The ONLY tool-call emission point. `function_call_arguments.done`
            # also carries the full call, but mapping both double-fires the
            # event and double-sends tool results (observed live 2026-09-01:
            # both frames arrive for one call).
            item = payload.get("item", {}) if isinstance(payload.get("item"), dict) else {}
            if item.get("type") == "function_call":
                call_id = str(item.get("call_id") or item.get("id") or "")
                if call_id:
                    events.append(LiveServerToolCallEvent(
                        id=call_id, name=str(item.get("name") or "tool"),
                        input=parse_json_object(item.get("arguments"))))
        elif et in {"response.done", "response.completed"}:
            response = payload.get("response", {}) if isinstance(payload.get("response"), dict) else {}
            output = response.get("output") if isinstance(response.get("output"), list) else []
            if str(response.get("status") or "") == "cancelled":
                # GA signals barge-in via response.done status=cancelled
                # (status_details.reason=client_cancelled); parallel to
                # Gemini's interrupted frame. The cancelled turn's usage is
                # dropped with it — stated, mirrors Gemini.
                events.append(LiveServerInterruptedEvent())
            elif any(isinstance(i, dict) and i.get("type") == "function_call" for i in output):
                # A response that requests tool calls does NOT end the turn:
                # the model is waiting for results, and the continuation
                # arrives as a further wire response (observed live
                # 2026-09-01). Gemini keeps the turn open here; emitting
                # turn_end would break the shared tool-dispatch loop.
                pass
            else:
                usage_data = response.get("usage", {}) if isinstance(response, dict) else {}
                u_in = usage_data.get("input_token_details") or usage_data.get("input_tokens_details") or {}
                u_out = usage_data.get("output_token_details") or usage_data.get("output_tokens_details") or {}
                events.append(LiveServerTurnEndEvent(usage=Usage(
                    input_tokens=int(usage_data.get("input_tokens", 0) or 0),
                    output_tokens=int(usage_data.get("output_tokens", 0) or 0),
                    total_tokens=usage_data.get("total_tokens"),
                    reasoning_tokens=u_out.get("reasoning_tokens"),
                    cache_read_tokens=u_in.get("cached_tokens"),
                    cache_write_tokens=u_in.get("cache_write_tokens"),
                    input_audio_tokens=u_in.get("audio_tokens"),
                    output_audio_tokens=u_out.get("audio_tokens"),
                )))
        elif et in {"response.cancelled", "response.canceled"}:
            events.append(LiveServerInterruptedEvent())
        elif et in {"error", "response.error"}:
            err = payload.get("error")
            if isinstance(err, dict):
                provider_code = str(err.get("code") or err.get("type") or payload.get("code") or "provider")
                message = str(err.get("message") or payload.get("message") or "")
            else:
                provider_code = str(payload.get("code") or payload.get("error_type") or "provider")
                message = str(payload.get("message") or "")
            if provider_code == "response_cancel_not_active":
                # Benign barge-in race (captured live 2026-09-01): the
                # response finished before the cancel arrived, or interrupt()
                # was pressed twice. Gemini tolerates repeated interrupts;
                # surfacing this as an error event breaks the parallel.
                return events
            events.append(LiveServerErrorEvent(error=self._error_detail(provider_code, message)))
        return events

    # ─── Other endpoints ────────────────────────────────────────────

    def _models_request(self):
        return make_json_request(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/models",
            headers=self._headers(),
            read_timeout=30.0,
        )

    def _models_from_body(self, body: str):
        data = json.loads(body)
        entries = data.get("data") if isinstance(data, dict) else None
        return model_infos_from_entries(
            entries,
            provider=self.provider,
            api_family="openai_responses",
            id_of=lambda entry: entry.get("id"),
        )

    # ─── File hooks (Files API) ─────────────────────────────────
    #
    # Wire shapes verified live 2026-08-31 (curl-fixtures/files-2026-08-31/):
    # multipart upload with a required `purpose` form field (an OpenAI
    # storage classification, NOT part of the portable surface — lm15
    # defaults to `user_data` per current OpenAI guidance and sets `batch`
    # itself for batch inputs; extensions["purpose"] overrides).  List
    # pages with `after=<last id>` + `has_more`.  Download is refused for
    # some purposes (observed: 400 for user_data) — forwarded typed.

    def _file_upload_request(self, request: FileUploadRequest) -> TransportRequest:
        extensions = dict(request.extensions or {})
        purpose = str(extensions.pop("purpose", "user_data"))
        fields = [("purpose", purpose)] + [(k, str(v)) for k, v in extensions.items()]
        content_type, body = multipart_form_body(
            fields=fields,
            files=[("file", request.filename, request.media_type, request.bytes)],
        )
        return TransportRequest(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/files",
            headers=list(self._headers(content_type=content_type).items()),
            body=body,
            read_timeout=300.0,
        )

    def _file_info_from_body(self, body: str) -> FileInfo:
        return self._file_info(json.loads(body))

    def _file_info(self, data: dict[str, Any]) -> FileInfo:
        file_id = data.get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ProviderError("openai: file object carries no id", provider=self.provider)
        status = str(data.get("status") or "")
        if status == "error":
            readiness = "failed"
        elif status == "uploaded":
            readiness = "pending"
        else:  # "processed", absent (the field is deprecated), or unknown
            readiness = "ready"
        filename = data.get("filename")
        return FileInfo(
            id=file_id,
            filename=filename if isinstance(filename, str) and filename else None,
            media_type=None,  # OpenAI file metadata reports no MIME type
            size_bytes=data.get("bytes") if isinstance(data.get("bytes"), int) else None,
            created_at=iso_utc(data.get("created_at")),
            expires_at=iso_utc(data.get("expires_at")),
            readiness=readiness,
            downloadable=None,  # purpose-dependent policy, not reported per file
            provider_data=data,
        )

    def _file_get_request(self, file_id: str) -> TransportRequest:
        return make_json_request(
            method="GET", url=f"{self.base_url.rstrip('/')}/files/{file_id}",
            headers=self._headers(), read_timeout=60.0,
        )

    def _file_list_request(self, limit: int, cursor: str | None) -> TransportRequest:
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["after"] = cursor
        return make_json_request(
            method="GET", url=f"{self.base_url.rstrip('/')}/files",
            params=params, headers=self._headers(), read_timeout=60.0,
        )

    def _file_page_from_list_body(self, body: str) -> FilePage:
        data = json.loads(body)
        entries = data.get("data") if isinstance(data.get("data"), list) else []
        items = tuple(self._file_info(entry) for entry in entries if isinstance(entry, dict))
        cursor = data.get("last_id") if data.get("has_more") and items else None
        return FilePage(items=items, next_cursor=cursor if isinstance(cursor, str) and cursor else None)

    def _file_delete_request(self, file_id: str) -> TransportRequest:
        return make_json_request(
            method="DELETE", url=f"{self.base_url.rstrip('/')}/files/{file_id}",
            headers=self._headers(), read_timeout=60.0,
        )

    def _file_download_request(self, file_id: str) -> TransportRequest:
        return make_json_request(
            method="GET", url=f"{self.base_url.rstrip('/')}/files/{file_id}/content",
            headers=self._headers(), read_timeout=300.0,
        )

    # ─── Batch hooks (Batch API over /v1/responses) ──────────────────
    #
    # OpenAI batches are two-step: the requests travel as an uploaded JSONL
    # file, then /batches references it. The file is wire syntax, not
    # semantics — batch() owns the upload; the file id stays visible in
    # provider_data. lm15 assigns positional custom_ids and re-sorts
    # results so entry order always equals submission order.

    def _batch_upload_request(self, request: BatchRequest) -> TransportRequest:
        lines = []
        for i, nested in enumerate(request.requests):
            lines.append(json.dumps({
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/responses",
                "body": self._payload(nested, stream=False),
            }, separators=(",", ":"), ensure_ascii=False))
        data = ("\n".join(lines) + "\n").encode("utf-8")
        content_type, body = multipart_form_body(
            fields=[("purpose", "batch")],
            files=[("file", "lm15-batch.jsonl", "application/jsonl", data)],
        )
        return TransportRequest(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/files",
            headers=list(self._headers(content_type=content_type).items()),
            body=body,
            read_timeout=300.0,
        )

    def _batch_submit_request(self, request: BatchRequest, upload_body: dict[str, Any] | None) -> TransportRequest:
        input_file_id = (upload_body or {}).get("id")
        if not isinstance(input_file_id, str) or not input_file_id:
            raise ProviderError("openai: batch input file upload returned no id", provider=self.provider)
        extensions = dict(request.extensions or {})
        payload: dict[str, Any] = {
            "input_file_id": input_file_id,
            "endpoint": extensions.pop("endpoint", "/v1/responses"),
            "completion_window": extensions.pop("completion_window", "24h"),
        }
        if request.label is not None:
            payload["metadata"] = {"label": request.label}
        payload.update(extensions)
        return make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/batches",
            headers=self._headers(),
            payload=payload,
            read_timeout=120.0,
        )

    def _batch_job_from_body(self, body: str) -> BatchJobInfo:
        return self._batch_job_info(json.loads(body))

    def _batch_job_info(self, data: dict[str, Any]) -> BatchJobInfo:
        batch_id = data.get("id")
        if not isinstance(batch_id, str) or not batch_id:
            raise ProviderError("openai: batch object carries no id", provider=self.provider)
        metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        label = metadata.get("label")
        return BatchJobInfo(
            id=batch_id,
            status=_openai_batch_status(str(data.get("status") or "")),
            label=label if isinstance(label, str) and label else None,
            created_at=iso_utc(data.get("created_at")),
            provider_data=data,
        )

    def _batch_status_request(self, batch_id: str) -> TransportRequest:
        return make_json_request(
            method="GET", url=f"{self.base_url.rstrip('/')}/batches/{batch_id}",
            headers=self._headers(), read_timeout=60.0,
        )

    def _batch_cancel_request(self, batch_id: str) -> TransportRequest:
        return make_json_request(
            method="POST", url=f"{self.base_url.rstrip('/')}/batches/{batch_id}/cancel",
            headers=self._headers(), read_timeout=60.0,
        )

    def _batch_result_fetches(self, status_body: dict[str, Any]) -> tuple[TransportRequest, ...]:
        fetches = []
        for key in ("output_file_id", "error_file_id"):
            file_id = status_body.get(key)
            if isinstance(file_id, str) and file_id:
                fetches.append(make_json_request(
                    method="GET", url=f"{self.base_url.rstrip('/')}/files/{file_id}/content",
                    headers=self._headers(), read_timeout=300.0,
                ))
        return tuple(fetches)

    def _batch_entries(self, status_body: dict[str, Any], fetched: tuple[str, ...]) -> tuple[BatchEntry, ...]:
        job_status = _openai_batch_status(str(status_body.get("status") or ""))
        found: dict[int, BatchEntry] = {}
        for text in fetched:
            for line in text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                index = int(str(item.get("custom_id")))
                response_obj = item.get("response") if isinstance(item.get("response"), dict) else {}
                status_code = int(response_obj.get("status_code") or 0)
                body_obj = response_obj.get("body") if isinstance(response_obj.get("body"), dict) else {}
                if status_code == 200 and body_obj:
                    response = self.parse_response(
                        batch_entry_request(body_obj.get("model")), batch_entry_http(body_obj)
                    )
                    found[index] = BatchEntry(index=index, outcome="succeeded", response=response)
                else:
                    err_source = body_obj or item.get("error") or {}
                    err = self.normalize_error(status_code or 400, json.dumps(err_source))
                    found[index] = BatchEntry(
                        index=index,
                        outcome="errored",
                        error=ErrorDetail(
                            code=canonical_error_code(err),
                            message=err.message or "batch entry errored",
                            provider_code=err.provider_code,
                        ),
                    )
        # Entries the output files never mention (an expired or cancelled
        # batch stops mid-flight): fill from the job's terminal status.
        # Live edge (observed 2026-09-01): a batch cancelled during
        # `validating` reports request_counts.total=0 — the provider never
        # registered the requests — so results() is honestly EMPTY; lm15
        # does not fabricate entries from the input file side-channel.
        counts = status_body.get("request_counts") if isinstance(status_body.get("request_counts"), dict) else {}
        total = int(counts.get("total") or 0) or (max(found) + 1 if found else 0)
        fill: str = "expired" if job_status == "expired" else "cancelled" if job_status == "cancelled" else "errored"
        entries = []
        for index in range(total):
            if index in found:
                entries.append(found[index])
            elif fill == "errored":
                entries.append(BatchEntry(
                    index=index, outcome="errored",
                    error=ErrorDetail(code="provider", message="entry missing from batch output files"),
                ))
            else:
                entries.append(BatchEntry(index=index, outcome=fill))
        return tuple(entries)

    def _batch_list_request(self, limit: int) -> TransportRequest:
        return make_json_request(
            method="GET", url=f"{self.base_url.rstrip('/')}/batches",
            params={"limit": int(limit)}, headers=self._headers(), read_timeout=60.0,
        )

    def _batch_jobs_from_list_body(self, body: str) -> tuple[BatchJobInfo, ...]:
        data = json.loads(body)
        items = data.get("data") if isinstance(data, dict) else None
        return tuple(self._batch_job_info(item) for item in (items or []) if isinstance(item, dict))

    # ─── Video generation (Sora; captured live 2026-09-01) ──────────────
    #
    # Jobs at /v1/videos: submit -> {status: queued, progress}, poll ->
    # in_progress -> completed, then /videos/{id}/content streams the MP4
    # bytes (media type from the content-type header).  The list endpoint
    # is account-wide.

    _VIDEO_STATUS_MAP: ClassVar[dict[str, str]] = {
        "queued": "queued",
        "in_progress": "running",
        "completed": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }

    def _video_submit_request(self, request: VideoGenerationRequest) -> TransportRequest:
        if request.images:
            # Sora's image input (input_reference) is a multipart upload; the
            # mapping is unverified against the live wire, and Sora also
            # constrains reference sizes.  Raising beats shipping a guess.
            raise UnsupportedFeatureError(
                "openai: video input images (input_reference) are not mapped yet; "
                "use the provider door until the mapping is live-receipted",
                provider=self.provider,
            )
        payload: dict[str, Any] = {"model": request.model, "prompt": request.prompt, **(request.extensions or {})}
        if request.seconds is not None:
            payload["seconds"] = str(request.seconds)  # the wire wants a string enum
        return make_json_request(method="POST", url=f"{self.base_url.rstrip('/')}/videos", headers=self._headers(), payload=payload, read_timeout=120.0)

    def _video_job_from_body(self, body: str, video_id: "str | None" = None) -> VideoJobInfo:
        return self._video_job_info(json.loads(body))

    def _video_job_info(self, data: dict[str, Any]) -> VideoJobInfo:
        video_id = data.get("id")
        if not isinstance(video_id, str) or not video_id:
            raise ProviderError("openai: video object carries no id", provider=self.provider)
        wire_status = str(data.get("status") or "")
        status = self._VIDEO_STATUS_MAP.get(wire_status)
        if status is None:
            raise ProviderError(f"openai: unknown video status {wire_status!r}", provider=self.provider)
        progress = data.get("progress")
        return VideoJobInfo(
            id=video_id,
            status=status,
            progress=int(progress) if isinstance(progress, (int, float)) and not isinstance(progress, bool) else None,
            created_at=iso_utc(data.get("created_at")),
            model=data.get("model"),
            provider_data=data,
        )

    def _video_status_request(self, video_id: str) -> TransportRequest:
        return make_json_request(method="GET", url=f"{self.base_url.rstrip('/')}/videos/{video_id}", headers=self._headers(), read_timeout=60.0)

    def _video_result_fetch(self, status_body: dict[str, Any]) -> TransportRequest:
        return make_json_request(
            method="GET",
            url=f"{self.base_url.rstrip('/')}/videos/{status_body.get('id')}/content",
            headers=self._headers(),
            read_timeout=600.0,
        )

    def _video_part(self, status_body: dict[str, Any], fetched: "HttpResponse | None") -> VideoPart:
        if fetched is None:
            raise ProviderError("openai: video content fetch is required", provider=self.provider)
        content_type = (fetched.header("content-type") or "").split(";", 1)[0].strip()
        if not content_type:
            raise ProviderError("openai: video content carries no content-type", provider=self.provider)
        return VideoPart(media_type=content_type, data=base64.b64encode(fetched.body).decode("ascii"))

    def _video_list_request(self, limit: int, model: str | None) -> TransportRequest:
        return make_json_request(
            method="GET", url=f"{self.base_url.rstrip('/')}/videos",
            params={"limit": int(limit)}, headers=self._headers(), read_timeout=60.0,
        )

    def _video_jobs_from_list_body(self, body: str) -> tuple[VideoJobInfo, ...]:
        data = json.loads(body)
        items = data.get("data") if isinstance(data, dict) else None
        return tuple(self._video_job_info(item) for item in (items or []) if isinstance(item, dict))

    # ─── Media generation (captured live 2026-09-01) ────────────────────
    #
    # Images: /images/generations (JSON) or, when input images are present,
    # /images/edits (multipart) — verified honored by pixel check.  The
    # response states the format in `output_format`; the media type is read
    # from the wire, never assumed.  Usage carries real token counts.
    # Speech: /audio/speech returns RAW BYTES; the media type lives in the
    # content-type header (server default is audio/mpeg — lm15 injects no
    # voice/format defaults of its own).

    def _image_generate_request(self, request: ImageGenerationRequest) -> TransportRequest:
        base = self.base_url.rstrip("/")
        if not request.images:
            payload = {"model": request.model, "prompt": request.prompt, "size": request.size, **(request.extensions or {})}
            payload = {k: v for k, v in payload.items() if v is not None}
            return make_json_request(method="POST", url=f"{base}/images/generations", headers=self._headers(), payload=payload, read_timeout=300.0)
        # Edits are multipart: the wire takes uploaded bytes only.
        for part in request.images:
            if part.data is None and part.path is None:
                raise UnsupportedFeatureError(
                    "openai: image edits take inline data or a local path; "
                    "url/file_id-addressed input images have no wire slot",
                    provider=self.provider,
                )
        fields = [("model", str(request.model)), ("prompt", request.prompt)]
        if request.size is not None:
            fields.append(("size", request.size))
        fields += [(k, str(v)) for k, v in (request.extensions or {}).items()]
        files = [
            ("image[]", f"image-{i}", part.media_type, part.bytes)
            for i, part in enumerate(request.images)
        ]
        content_type, body = multipart_form_body(fields=fields, files=files)
        return TransportRequest(
            method="POST",
            url=f"{base}/images/edits",
            headers=list(self._headers(content_type=content_type).items()),
            body=body,
            read_timeout=300.0,
        )

    def _image_generation_from_response(self, request: ImageGenerationRequest, resp: HttpResponse) -> ImageGenerationResponse:
        data = resp.json()
        output_format = data.get("output_format")
        media_type = f"image/{output_format}" if isinstance(output_format, str) and output_format else None
        images: list[ImagePart] = []
        for item in data.get("data", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("b64_json"):
                images.append(ImagePart(media_type=media_type or "application/octet-stream", data=str(item["b64_json"])))
            elif item.get("url"):
                images.append(ImagePart(media_type=media_type or "application/octet-stream", url=str(item["url"])))
        usage_obj = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = Usage(
            input_tokens=usage_obj.get("input_tokens"),
            output_tokens=usage_obj.get("output_tokens"),
            total_tokens=usage_obj.get("total_tokens"),
        )
        # Captured: the images response carries no id and no model echo.
        return ImageGenerationResponse(images=tuple(images), usage=usage, provider_data=data)

    def _speech_generate_request(self, request: SpeechGenerationRequest) -> TransportRequest:
        payload: dict[str, Any] = {"model": request.model, "input": request.prompt, **(request.extensions or {})}
        if request.voice is not None:
            payload["voice"] = request.voice
        if request.format is not None:
            payload["response_format"] = request.format
        return make_json_request(method="POST", url=f"{self.base_url.rstrip('/')}/audio/speech", headers=self._headers(), payload=payload, read_timeout=300.0)

    def _speech_generation_from_response(self, request: SpeechGenerationRequest, resp: HttpResponse) -> SpeechGenerationResponse:
        content_type = (resp.header("content-type") or "").split(";", 1)[0].strip()
        if not content_type:
            raise ProviderError("openai: speech response carries no content-type", provider=self.provider)
        audio = AudioPart(media_type=content_type, data=base64.b64encode(resp.body).decode("ascii"))
        # The body is raw media: no usage, no id, no model echo exist.
        return SpeechGenerationResponse(audio=audio, provider_data={"content_type": content_type})
