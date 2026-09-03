"""
lm15.providers.openai_chat — OpenAI Chat Completions adapter.

The Chat Completions dialect is the wire format spoken by OpenAI's legacy
endpoint and by most OpenAI-compatible servers: ollama, Groq, OpenRouter,
vLLM, SGLang, DeepSeek, and friends.  Provider quirks are described by
``OpenAIChatCompat`` policies (see lm15.compat); named presets bundle a
policy with that server's default base URL.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, ClassVar, Iterator

from ..compat import (
    OPENAI_CHAT_PRESET_BASE_URLS,
    OpenAIChatCompat,
    ResolvedOpenAIChatCompat,
    resolve_openai_chat_compat,
)
from ..errors import ProviderError, UnsupportedFeatureError
from ..access import OPENAI_CHAT_API, auth_header
from ..features import ProviderManifest
from ..sse import SSEEvent
from ..transports import TransportRequest
from ..types import (
    BuiltinTool,
    FunctionTool,
    Message,
    RefusalPart,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    StreamErrorEvent,
    StreamEvent,
    TextDelta,
    TextPart,
    ThinkingDelta,
    ThinkingPart,
    ToolCallDelta,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from .base import BaseProviderLM, Credential, HttpResponse, SyncTransport, default_transport, resolve_credential
from .common import (
    make_json_request,
    media_data_uri,
    model_infos_from_entries,
    openai_token_logprobs,
    parse_json_object,
    parts_to_text,
)
from .openai import (
    OpenAILM,
    _attach_unmapped,
    _breakpoint_unsupported,
    _cache_breakpoint_index,
    _cache_common_payload,
    _cache_stable_prefix,
    _record_unmapped,
)

_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Canonical builtin tool name → Groq server-executed tool type (compat
# builtin_tools="groq"; both verified live 2026-09-01: Groq runs them
# server-side and reports the trace in message.executed_tools, which
# stays in provider_data per MAP-1).
_GROQ_BUILTIN_MAP: dict[str, str] = {
    "web_search": "browser_search",
    "code_execution": "code_interpreter",
}

_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_call",
    "function_call": "tool_call",
    "content_filter": "content_filter",
}


def _chat_content_parts(msg: Message, *, force_array: bool = False) -> str | list[dict[str, Any]]:
    """Map non-assistant message parts to chat-completions content.

    Single text part → plain string; anything multimodal → content array.
    ``force_array`` keeps the array form for a lone text part (a cache
    breakpoint rides on a text content block, never on a bare string).
    """
    parts = [p for p in msg.parts if not isinstance(p, (ToolCallPart, ToolResultPart))]
    if len(parts) == 1 and isinstance(parts[0], TextPart) and not force_array:
        return parts[0].text
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, TextPart):
            out.append({"type": "text", "text": part.text})
        elif part.type == "image":
            url = part.url if part.url is not None else media_data_uri(part)
            payload: dict[str, Any] = {"url": url}
            if getattr(part, "detail", None):
                payload["detail"] = part.detail
            out.append({"type": "image_url", "image_url": payload})
        elif isinstance(part, ThinkingPart):
            continue  # thinking is never replayed as user content
        else:
            # Lossy text rendering for part kinds the dialect cannot carry.
            text = parts_to_text((part,))
            if text:
                out.append({"type": "text", "text": text})
    return out


def _response_format_to_chat(format_config: dict[str, Any]) -> dict[str, Any]:
    """Canonical response_format (INV-050) -> chat-completions response_format."""
    if format_config["type"] == "json_object":
        return {"type": "json_object"}
    inner: dict[str, Any] = {"name": format_config.get("name") or "response", "schema": format_config["schema"]}
    if "strict" in format_config:
        inner["strict"] = format_config["strict"]
    return {"type": "json_schema", "json_schema": inner}


def _usage_from_chat(usage_data: dict[str, Any]) -> Usage:
    prompt_details = usage_data.get("prompt_tokens_details") or {}
    completion_details = usage_data.get("completion_tokens_details") or {}
    return Usage(
        input_tokens=usage_data.get("prompt_tokens"),
        output_tokens=usage_data.get("completion_tokens"),
        total_tokens=usage_data.get("total_tokens"),
        reasoning_tokens=completion_details.get("reasoning_tokens"),
        cache_read_tokens=prompt_details.get("cached_tokens"),
        cache_write_tokens=prompt_details.get("cache_write_tokens"),
        input_audio_tokens=prompt_details.get("audio_tokens"),
        output_audio_tokens=completion_details.get("audio_tokens"),
    )


@dataclass(slots=True)
class OpenAIChatLM(BaseProviderLM):
    """Adapter for the OpenAI Chat Completions wire dialect.

    ``compat`` may be an :class:`OpenAIChatCompat`, a preset name
    (``"ollama"``, ``"groq"``, ``"openrouter"``, ``"vllm"``, ``"sglang"``,
    ``"openai"``, …), or None (plain OpenAI policy).  A preset name also
    supplies that server's default ``base_url``; an explicit non-default
    ``base_url`` argument always wins.
    """

    api_key: Credential | None = field(default=None, repr=False)
    transport: SyncTransport = field(default_factory=default_transport)
    base_url: str = _DEFAULT_BASE_URL
    compat: OpenAIChatCompat | str | None = None
    access: ProviderManifest | None = field(default=None, repr=False)
    credentials_path: "str | os.PathLike[str] | None" = field(default=None, repr=False)

    provider: str = field(default="openai_chat", init=False)
    account_id: str | None = field(default=None, init=False, repr=False)
    manifest: ClassVar[ProviderManifest] = OPENAI_CHAT_API

    # OpenAI-compatible servers reuse the same error envelope family;
    # share the Responses adapter's mapping verbatim.
    _response_error_code_map: ClassVar[dict[str, type[ProviderError]]] = OpenAILM._response_error_code_map
    _model_error_codes: ClassVar[frozenset[str]] = OpenAILM._model_error_codes
    _stream_error_code_map: ClassVar[dict[str, type[ProviderError]]] = OpenAILM._stream_error_code_map

    _is_model_error = staticmethod(OpenAILM._is_model_error)
    _response_error = OpenAILM._response_error
    _error_detail = OpenAILM._error_detail
    normalize_error = OpenAILM.normalize_error

    def __post_init__(self) -> None:
        self._bind_access(self.access, credentials_path=self.credentials_path, default_base_url=_DEFAULT_BASE_URL)
        compat = self.compat
        if isinstance(compat, str):
            preset_key = compat.lower().replace("-", "_").replace(" ", "_")
            resolved = resolve_openai_chat_compat(OpenAIChatCompat.preset(compat))
            if self.base_url == _DEFAULT_BASE_URL:
                self.base_url = OPENAI_CHAT_PRESET_BASE_URLS.get(preset_key, _DEFAULT_BASE_URL)
        elif isinstance(compat, OpenAIChatCompat):
            resolved = resolve_openai_chat_compat(compat)
        else:
            resolved = resolve_openai_chat_compat(OpenAIChatCompat())
        self._resolved_compat = resolved

    _resolved_compat: ResolvedOpenAIChatCompat = field(init=False, repr=False, default=ResolvedOpenAIChatCompat())

    def _headers(self) -> dict[str, str]:
        name, value = auth_header(self.access, resolve_credential(self.api_key))
        headers = {name: value, "Content-Type": "application/json"}
        for key, static in self.access.headers:
            headers[key] = static
        return headers

    # ─── Live model listing (provisional endpoint) ──────────────────────

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
            api_family="openai_chat",
            id_of=lambda entry: entry.get("id"),
        )

    # ─── Request serialization ──────────────────────────────────────

    def _build_messages(self, request: Request, compat: ResolvedOpenAIChatCompat) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        if request.system:
            system_text = request.system if isinstance(request.system, str) else parts_to_text(request.system)
            if _cache_stable_prefix(request, compat.cache_control):
                # prefix="stable": the mark rides on the system message's
                # text content part (array form; a bare string cannot carry it).
                messages.append({"role": compat.instruction_role, "content": [
                    {"type": "text", "text": system_text, "prompt_cache_breakpoint": {"mode": "explicit"}}
                ]})
            else:
                messages.append({"role": compat.instruction_role, "content": system_text})

        breakpoint_index = _cache_breakpoint_index(request, compat.cache_control)
        for msg_index, msg in enumerate(request.messages):
            if msg_index == breakpoint_index and msg.role in ("assistant", "tool"):
                raise _breakpoint_unsupported(self.provider, msg_index, msg.role)
            if msg.role == "tool":
                for part in msg.parts:
                    if isinstance(part, ToolResultPart):
                        output = parts_to_text(part.content)
                        if not output:
                            output = json.dumps([{"type": p.type} for p in part.content])
                        item: dict[str, Any] = {
                            "role": "tool",
                            "tool_call_id": part.id,
                            "content": output,
                        }
                        if compat.tool_result_name == "include" and part.name:
                            item["name"] = part.name
                        messages.append(item)
                continue

            if msg.role == "assistant":
                text_bits: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextPart):
                        text_bits.append(part.text)
                    elif isinstance(part, RefusalPart) and part.text:
                        text_bits.append(part.text)
                    elif isinstance(part, ThinkingPart) and compat.thinking_replay == "as_text" and part.text:
                        text_bits.append(part.text)
                tool_calls = [
                    {
                        "id": part.id,
                        "type": "function",
                        "function": {
                            "name": part.name,
                            "arguments": json.dumps(part.input, separators=(",", ":")),
                        },
                    }
                    for part in msg.parts
                    if isinstance(part, ToolCallPart)
                ]
                item = {"role": "assistant", "content": "\n".join(text_bits) if text_bits else None}
                if compat.thinking_replay == "native":
                    thinking = "\n".join(p.text for p in msg.parts if isinstance(p, ThinkingPart) and p.text)
                    if thinking or compat.assistant_reasoning_content == "include_empty":
                        item["reasoning_content"] = thinking
                if tool_calls:
                    item["tool_calls"] = tool_calls
                messages.append(item)
                continue

            role = compat.instruction_role if msg.role == "developer" else msg.role
            at_breakpoint = msg_index == breakpoint_index
            content = _chat_content_parts(msg, force_array=at_breakpoint)
            if at_breakpoint:
                # Same rule as the Responses dialect: the breakpoint rides on
                # the last text content block of the prefix message
                # (chat--create.md: ChatCompletionContentPartText carries
                # prompt_cache_breakpoint).
                if not isinstance(content, list) or not content or content[-1].get("type") != "text":
                    raise _breakpoint_unsupported(self.provider, msg_index, msg.role)
                content[-1]["prompt_cache_breakpoint"] = {"mode": "explicit"}
            if content or content == "":
                messages.append({"role": role, "content": content})
        return messages

    def _builtin_tool_payload(self, tool: BuiltinTool, compat: ResolvedOpenAIChatCompat) -> dict[str, Any]:
        """Map a BuiltinTool for the chat dialect, or raise.

        The base Chat Completions wire carries function/custom tools only,
        and some compat servers silently IGNORE unknown tool types
        (OpenRouter returned 200 with no search, verified live 2026-09-01)
        — passthrough would fabricate wire shapes and silent no-ops, so
        every unproven target raises.
        """
        if compat.builtin_tools == "groq":
            wire_type = _GROQ_BUILTIN_MAP.get(tool.name)
            if wire_type is None:
                raise UnsupportedFeatureError(
                    f"{self.provider}: builtin tool {tool.name!r} has no Groq wire "
                    f"mapping — supported: {sorted(_GROQ_BUILTIN_MAP)}",
                    provider=self.provider,
                )
            entry: dict[str, Any] = {"type": wire_type}
            if tool.config:
                entry.update(tool.config)
            return entry
        raise UnsupportedFeatureError(
            f"{self.provider}: builtin tool {tool.name!r} is not supported on this "
            "server — the Chat Completions wire carries function tools only, and "
            "unproven servers may silently ignore unknown tool types. Use "
            "compat='groq' for Groq's server-executed tools, or the OpenAI "
            "Responses / Anthropic / Gemini providers",
            provider=self.provider,
        )

    def _tool_choice_payload(self, request: Request) -> Any:
        tc = request.config.tool_choice
        if tc is None:
            return None
        if tc.mode == "none":
            return "none"
        if tc.allowed:
            by_name = {t.name: t for t in request.tools}
            entries = [by_name[name] for name in tc.allowed]
            builtins = [t.name for t in entries if isinstance(t, BuiltinTool)]
            if builtins:
                raise UnsupportedFeatureError(
                    f"{self.provider}: cannot force builtin tools {builtins} — the "
                    "Chat Completions wire has no hosted-tool tool_choice form "
                    "(OpenAI Responses and Anthropic carry it)",
                    provider=self.provider,
                )
            if len(entries) == 1 and tc.mode == "required":
                return {"type": "function", "function": {"name": entries[0].name}}
            # mode="auto" restriction or multi-tool subset: the dialect's
            # allowed_tools form (nested spelling, function tools only).
            return {
                "type": "allowed_tools",
                "allowed_tools": {
                    "mode": tc.mode,
                    "tools": [
                        {"type": "function", "function": {"name": t.name}} for t in entries
                    ],
                },
            }
        if tc.mode == "required":
            return "required"
        return "auto"

    def _payload(self, request: Request, stream: bool) -> dict[str, Any]:
        compat = self._resolved_compat
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": self._build_messages(request, compat),
        }
        if stream:
            payload["stream"] = True
            if compat.stream_usage == "include":
                payload["stream_options"] = {"include_usage": True}
        if request.config.max_tokens is not None:
            payload[compat.max_tokens_field] = request.config.max_tokens
        if request.config.temperature is not None:
            payload["temperature"] = request.config.temperature
        if request.config.top_p is not None:
            payload["top_p"] = request.config.top_p
        if request.config.stop:
            payload["stop"] = list(request.config.stop)
        if request.config.logprobs is not None:
            # Verified live 2026-09-01: logprobs=true alone returns chosen
            # tokens only; top_logprobs adds the alternatives.
            payload["logprobs"] = True
            if request.config.logprobs > 0:
                payload["top_logprobs"] = request.config.logprobs
        if request.tools:
            tools_wire: list[dict[str, Any]] = []
            for tool in request.tools:
                if isinstance(tool, FunctionTool):
                    function_payload: dict[str, Any] = {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    if compat.strict_tools == "include":
                        function_payload["strict"] = False
                    tools_wire.append({"type": "function", "function": function_payload})
                elif isinstance(tool, BuiltinTool):
                    tools_wire.append(self._builtin_tool_payload(tool, compat))
            if tools_wire:
                payload["tools"] = tools_wire
        tool_choice = self._tool_choice_payload(request)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if request.config.tool_choice and request.config.tool_choice.parallel is not None:
            payload["parallel_tool_calls"] = request.config.tool_choice.parallel
        if request.config.response_format:
            payload["response_format"] = _response_format_to_chat(request.config.response_format)
        if request.config.reasoning:
            reasoning = request.config.reasoning
            if not reasoning.is_off:
                # MAP-7: verbatim effort; no budget on this wire; summary
                # levels are Responses-only; "auto" maps to the dialect's
                # visibility knob where one exists (Groq include_reasoning).
                if reasoning.thinking_budget is not None:
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.thinking_budget is not supported — the Chat "
                        "Completions wire has no thinking token budget; use effort",
                        provider=self.provider,
                    )
                if reasoning.summary in ("concise", "detailed"):
                    raise UnsupportedFeatureError(
                        f"{self.provider}: reasoning.summary={reasoning.summary!r} is an OpenAI Responses "
                        "detail level; the Chat Completions wire has none (use 'auto')",
                        provider=self.provider,
                    )
                effort = reasoning.effort
                if compat.builtin_tools == "groq" and reasoning.summary == "auto":
                    # Groq's visibility knob (MAP-7 rule 7): "parsed" returns
                    # the trace as message.reasoning.  Live 2026-09-02: Qwen
                    # 3.6's default leaks a raw <think> block into
                    # message.content; this is the wire's fix.  Qwen's dial
                    # accepts only none|default, so an effort word still
                    # fails loudly there — extensions={"reasoning_format":
                    # "parsed"} with reasoning absent is the documented door.
                    payload["reasoning_format"] = "parsed"
                if compat.thinking_format == "reasoning_effort":
                    payload["reasoning_effort"] = effort
                elif compat.thinking_format == "openrouter":
                    payload["reasoning"] = {"effort": effort}
                elif compat.thinking_format == "deepseek":
                    payload["thinking"] = {"type": "enabled"}
                    payload["reasoning_effort"] = effort
                elif compat.thinking_format in {"qwen", "zai"}:
                    payload["enable_thinking"] = True
                elif compat.thinking_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {
                        "enable_thinking": True,
                        "preserve_thinking": True,
                    }
            else:
                # Explicit off must reach the wire; omission lets
                # reasoning-by-default models spend hidden reasoning tokens
                # (verified live 2026-09-01 on Groq: gpt-oss-20b spent 45
                # reasoning tokens when the field was omitted).  Servers
                # whose models cannot disable reasoning reject "none" with
                # a clear 400 — loud failure over a silent paid no-op.
                if compat.thinking_format == "reasoning_effort":
                    payload["reasoning_effort"] = "none"
                elif compat.thinking_format == "openrouter":
                    payload["reasoning"] = {"enabled": False}
                elif compat.thinking_format == "deepseek":
                    payload["thinking"] = {"type": "disabled"}
                elif compat.thinking_format in {"qwen", "zai"}:
                    payload["enable_thinking"] = False
                elif compat.thinking_format == "qwen_chat_template":
                    payload["chat_template_kwargs"] = {"enable_thinking": False}

        # Prompt caching (MAP-6): off switch, key, retention, resource.
        _cache_common_payload(request, payload, compat.cache_control, self.provider)

        if compat.routing is not None:
            payload["provider"] = compat.routing

        # Promoted cross-provider knobs (changes/2026-09-01-extensions-burn-down):
        # Chat Completions spellings — user_id rides the dialect's `user`
        # field (safety_identifier is Responses-only), or the server's own
        # name when the compat says so (DeepSeek: `user_id`). Servers that
        # do not implement a field reject it themselves; the dialect adapter
        # cannot know each compat server's support statically.
        if request.config.service_tier is not None:
            payload["service_tier"] = request.config.service_tier
        if request.config.user_id is not None:
            payload[compat.user_field] = request.config.user_id
        if request.config.store is not None:
            payload["store"] = request.config.store

        if request.config.extensions:
            reserved = {
                "prompt_caching",
                "cache",
                "compat",
                "openai_compat",
                "openai_chat_compat",
            }
            passthrough = {k: v for k, v in request.config.extensions.items() if k not in reserved}
            payload.update(passthrough)
        return payload

    def build_request(self, request: Request, stream: bool) -> TransportRequest:
        return make_json_request(
            method="POST",
            url=f"{self.base_url.rstrip('/')}/chat/completions",
            headers=self._headers(),
            payload=self._payload(request, stream=stream),
            read_timeout=120.0 if stream else 60.0,
        )

    # ─── Response parsing ───────────────────────────────────────────

    @staticmethod
    def _finish_reason(raw: Any, *, has_tool_call: bool, unmapped: list[dict[str, str]]) -> str:
        if has_tool_call:
            return "tool_call"
        if raw is None or raw == "":
            return "stop"
        mapped = _FINISH_REASON_MAP.get(str(raw))
        if mapped is None:
            _record_unmapped(unmapped, "choices[0].finish_reason", raw)
            return "stop"
        return mapped

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
        choices = data.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        if choices and not isinstance(choices[0], dict):
            _record_unmapped(unmapped, "choices[0]", type(choices[0]).__name__)
        message = choice.get("message") if isinstance(choice.get("message"), dict) else {}

        reasoning_text = message.get("reasoning_content") or message.get("reasoning")
        if reasoning_text:
            parts.append(ThinkingPart(text=str(reasoning_text)))

        content = message.get("content")
        if isinstance(content, str):
            if content:
                parts.append(TextPart(text=content))
        elif isinstance(content, list):
            for content_index, item in enumerate(content):
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(TextPart(text=str(item.get("text") or "")))
                else:
                    _record_unmapped(
                        unmapped,
                        f"choices[0].message.content[{content_index}]",
                        item.get("type") if isinstance(item, dict) else type(item).__name__,
                    )
        elif content is not None:
            _record_unmapped(unmapped, "choices[0].message.content", type(content).__name__)

        refusal = message.get("refusal")
        if refusal:
            parts.append(RefusalPart(text=str(refusal)))

        for call_index, call in enumerate(message.get("tool_calls") or []):
            if not isinstance(call, dict):
                _record_unmapped(unmapped, f"choices[0].message.tool_calls[{call_index}]", type(call).__name__)
                continue
            call_type = call.get("type") or "function"
            if call_type != "function":
                _record_unmapped(unmapped, f"choices[0].message.tool_calls[{call_index}]", call_type)
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            parts.append(
                ToolCallPart(
                    id=str(call.get("id") or f"call_{len(parts)}"),
                    name=str(function.get("name") or "tool"),
                    input=parse_json_object(function.get("arguments")),
                )
            )

        if not parts:
            # MAP-2: a response message is never empty.
            parts = [TextPart(text="")]

        has_tool = any(isinstance(part, ToolCallPart) for part in parts)
        usage = _usage_from_chat(data.get("usage") or {})
        # choices[0].logprobs.content is the message-level token sequence.
        # Refusal logprobs (choices[0].logprobs.refusal) stay in
        # provider_data — no canonical refusal-token concept.
        logprobs_payload = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else {}
        logprob_seq = openai_token_logprobs(logprobs_payload.get("content"))
        return Response(
            id=str(data.get("id")) if data.get("id") else None,
            model=str(data.get("model") or request.model),
            message=Message(role="assistant", parts=tuple(parts)),
            finish_reason=self._finish_reason(choice.get("finish_reason"), has_tool_call=has_tool, unmapped=unmapped),
            usage=usage,
            logprobs=logprob_seq or None,
            provider_data=_attach_unmapped(data, unmapped),
        )

    # ─── Stream parsing ──────────────────────────────────────────────

    def parse_stream_events(self, request: Request, raw_event: SSEEvent) -> Iterator[StreamEvent]:
        if not raw_event.data:
            return
        if raw_event.data == "[DONE]":
            yield StreamEndEvent()
            return
        payload = json.loads(raw_event.data)
        if not isinstance(payload, dict):
            return

        err = payload.get("error")
        if isinstance(err, dict):
            provider_code = str(err.get("code") or err.get("type") or "provider")
            yield StreamErrorEvent(error=self._error_detail(provider_code, str(err.get("message") or "")))
            return

        choices = payload.get("choices") or []
        choice = choices[0] if choices and isinstance(choices[0], dict) else {}
        delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}

        reasoning_text = delta.get("reasoning_content") or delta.get("reasoning")
        if reasoning_text:
            yield StreamDeltaEvent(delta=ThinkingDelta(text=str(reasoning_text)))

        content = delta.get("content")
        if isinstance(content, str) and content:
            logprobs_payload = choice.get("logprobs") if isinstance(choice.get("logprobs"), dict) else {}
            yield StreamDeltaEvent(
                delta=TextDelta(
                    text=content,
                    logprobs=openai_token_logprobs(logprobs_payload.get("content")),
                )
            )

        for call in delta.get("tool_calls") or []:
            if not isinstance(call, dict):
                continue
            function = call.get("function") if isinstance(call.get("function"), dict) else {}
            yield StreamDeltaEvent(
                delta=ToolCallDelta(
                    input=str(function.get("arguments") or ""),
                    part_index=int(call.get("index", 0) or 0),
                    id=str(call.get("id") or "") or None,
                    name=str(function.get("name") or "") or None,
                )
            )

        finish_raw = choice.get("finish_reason")
        usage_data = payload.get("usage")
        if finish_raw:
            yield StreamEndEvent(
                finish_reason=_FINISH_REASON_MAP.get(str(finish_raw), "stop"),
                usage=_usage_from_chat(usage_data) if isinstance(usage_data, dict) else None,
            )
        elif isinstance(usage_data, dict):
            # Final usage-only chunk (stream_options.include_usage).
            yield StreamEndEvent(usage=_usage_from_chat(usage_data))
