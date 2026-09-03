"""
lm15.compat — Typed provider/API compatibility policies.

Compatibility policies describe how a provider adapter should serialize a
canonical lm15 request for a specific API dialect. They are intentionally
separate from lm15.types: Request/Response describe *what* the caller wants;
compat profiles describe provider wire-format quirks.

Fields default to None, which means "inherit from the parent profile". The
string value "auto" is an explicit policy: ask the adapter to use its automatic
heuristic for that field. Keeping None distinct from "auto" matters because
profiles are layered.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, TypeAlias, get_args

from .types import JsonObject


# ─── Shared helpers ──────────────────────────────────────────────────


def _check_literal_or_none(value: object, literal_alias: object, field_name: str) -> None:
    if value is not None and value not in get_args(literal_alias):  # type: ignore[arg-type]
        raise ValueError(f"unsupported {field_name}: {value!r}")


def _check_json_object_or_none(value: object, field_name: str) -> None:
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a JSON object or None")


def _merge_json_object(a: JsonObject | None, b: JsonObject | None) -> JsonObject | None:
    if a is None:
        return b
    if b is None:
        return a
    return {**a, **b}


# ─── OpenAI Responses API compatibility ──────────────────────────────

OpenAIResponsesDeveloperRole = Literal["auto", "developer", "system"]
OpenAIResponsesMaxOutputTokensField = Literal[
    "auto",
    "max_output_tokens",
    "max_completion_tokens",
    "max_tokens",
]
OpenAIResponsesReasoningFormat = Literal[
    "auto",
    "none",
    "responses_reasoning",
    "reasoning_effort",
    "openrouter",
    "deepseek",
    "qwen",
    "qwen_chat_template",
    "zai",
]
OpenAIToolResultName = Literal["auto", "include", "omit"]
OpenAIStrictTools = Literal["auto", "include", "omit"]
OpenAICacheControl = Literal["auto", "none", "openai", "anthropic"]


@dataclass(frozen=True, slots=True)
class OpenAIResponsesCompat:
    """Partial compatibility policy for OpenAI Responses-family APIs.

    None means "inherit". Non-None values override parent profiles. The value
    "auto" means "explicitly use adapter auto-detection".
    """

    developer_role: OpenAIResponsesDeveloperRole | None = None
    max_output_tokens_field: OpenAIResponsesMaxOutputTokensField | None = None
    reasoning_format: OpenAIResponsesReasoningFormat | None = None
    tool_result_name: OpenAIToolResultName | None = None
    strict_tools: OpenAIStrictTools | None = None
    cache_control: OpenAICacheControl | None = None
    routing: JsonObject | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_literal_or_none(self.developer_role, OpenAIResponsesDeveloperRole, "developer_role")
        _check_literal_or_none(
            self.max_output_tokens_field,
            OpenAIResponsesMaxOutputTokensField,
            "max_output_tokens_field",
        )
        _check_literal_or_none(self.reasoning_format, OpenAIResponsesReasoningFormat, "reasoning_format")
        _check_literal_or_none(self.tool_result_name, OpenAIToolResultName, "tool_result_name")
        _check_literal_or_none(self.strict_tools, OpenAIStrictTools, "strict_tools")
        _check_literal_or_none(self.cache_control, OpenAICacheControl, "cache_control")
        _check_json_object_or_none(self.routing, "routing")
        _check_json_object_or_none(self.extensions, "extensions")

    @classmethod
    def preset(cls, name: str) -> "OpenAIResponsesCompat":
        key = name.lower().replace("-", "_").replace(" ", "_")

        if key in {"openai", "responses", "openai_responses"}:
            return cls(
                developer_role="developer",
                max_output_tokens_field="max_output_tokens",
                reasoning_format="responses_reasoning",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="openai",
            )

        if key == "openrouter":
            return cls(
                developer_role="developer",
                max_output_tokens_field="max_tokens",
                reasoning_format="openrouter",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="openai",
            )

        if key in {"ollama", "lmstudio", "lm_studio"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="none",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key in {"vllm", "sglang"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="reasoning_effort",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key in {"qwen", "dashscope_qwen"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="qwen",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key == "deepseek":
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="deepseek",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        if key in {"zai", "z_ai"}:
            return cls(
                developer_role="system",
                max_output_tokens_field="max_tokens",
                reasoning_format="zai",
                tool_result_name="omit",
                strict_tools="omit",
                cache_control="none",
            )

        raise ValueError(f"unknown OpenAIResponsesCompat preset: {name!r}")


@dataclass(frozen=True, slots=True)
class ResolvedOpenAIResponsesCompat:
    """Fully resolved OpenAI Responses compatibility policy."""

    developer_role: Literal["developer", "system"] = "developer"
    max_output_tokens_field: Literal["max_output_tokens", "max_completion_tokens", "max_tokens"] = "max_output_tokens"
    reasoning_format: Literal[
        "none",
        "responses_reasoning",
        "reasoning_effort",
        "openrouter",
        "deepseek",
        "qwen",
        "qwen_chat_template",
        "zai",
    ] = "responses_reasoning"
    tool_result_name: Literal["include", "omit"] = "omit"
    strict_tools: Literal["include", "omit"] = "omit"
    cache_control: Literal["none", "openai", "anthropic"] = "openai"
    routing: JsonObject | None = None
    extensions: JsonObject | None = None


# ─── OpenAI Chat Completions compatibility ───────────────────────────

OpenAIChatInstructionRole = Literal["auto", "developer", "system"]
OpenAIChatMaxTokensField = Literal["auto", "max_completion_tokens", "max_tokens"]
OpenAIChatStreamUsage = Literal["auto", "include", "omit"]
OpenAIChatAssistantAfterToolResult = Literal["auto", "insert", "omit"]
OpenAIChatThinkingReplay = Literal["auto", "native", "as_text", "omit"]
OpenAIChatAssistantReasoningContent = Literal["auto", "include_empty", "omit"]
# "deepseek" names a wire SHAPE — thinking={"type": enabled|disabled} plus
# reasoning_effort — not a company: DeepSeek, xAI and Z.AI all speak it
# (docs.z.ai chat--create.md ChatThinking; the earlier "zai" value sent
# Qwen's enable_thinking and was never live-validated; removed 2026-09-03).
OpenAIChatThinkingFormat = Literal[
    "auto",
    "none",
    "reasoning_effort",
    "openrouter",
    "deepseek",
    "qwen",
    "qwen_chat_template",
]
# BuiltinTool policy for the chat dialect. The base Chat Completions wire
# carries function/custom tools ONLY (doc: chat--create.md), and some
# compat servers silently IGNORE unknown tool types (OpenRouter, verified
# live 2026-09-01) — so "reject" (raise) is the only safe default.
# "groq" maps canonical builtin names onto Groq's server-executed tool
# types (browser_search / code_interpreter, both verified live
# 2026-09-01).
OpenAIChatBuiltinTools = Literal["auto", "reject", "groq"]
# Which request field carries Config.user_id.  OpenAI's dialect spells it
# `user`; DeepSeek documents `user_id` (content-safety, KV-cache and
# scheduling isolation, rate-limit.md) and accepts `user` silently (live
# 2026-09-03: 200 either way, no echo) — so the documented name is the
# only one that can be trusted to do anything.
OpenAIChatUserField = Literal["auto", "user", "user_id"]
# Whether the server honours a tool_choice other than "auto" (required,
# none, a named function, an allowlist).  "reject" makes the adapter raise
# UnsupportedFeatureError before the wire.  Z.AI documents auto only and
# ignores the rest silently (live 2026-09-03: mode=required answered text,
# mode=none called the tool) — a silent widen is worse than an error (MAP-8).
OpenAIChatForcedToolChoice = Literal["auto", "send", "reject"]
# Whether the server honours response_format.type=json_schema.  "reject"
# raises before the wire.  DeepSeek answers 400 (loud, nothing to do); Z.AI
# answers 200 with free-form, fenced JSON that ignores the schema (live
# 2026-09-03) — silent, so the adapter must refuse.
OpenAIChatJsonSchema = Literal["auto", "send", "reject"]


@dataclass(frozen=True, slots=True)
class OpenAIChatCompat:
    """Partial compatibility policy for OpenAI Chat Completions-family APIs.

    This class is consumed by OpenAIChatLM (lm15.providers.openai_chat); it is
    kept separate so profiles can describe chat-completions style endpoints
    without overloading OpenAIResponsesCompat.
    """

    instruction_role: OpenAIChatInstructionRole | None = None
    max_tokens_field: OpenAIChatMaxTokensField | None = None
    stream_usage: OpenAIChatStreamUsage | None = None
    tool_result_name: OpenAIToolResultName | None = None
    assistant_after_tool_result: OpenAIChatAssistantAfterToolResult | None = None
    thinking_format: OpenAIChatThinkingFormat | None = None
    thinking_replay: OpenAIChatThinkingReplay | None = None
    assistant_reasoning_content: OpenAIChatAssistantReasoningContent | None = None
    strict_tools: OpenAIStrictTools | None = None
    builtin_tools: OpenAIChatBuiltinTools | None = None
    cache_control: OpenAICacheControl | None = None
    user_field: OpenAIChatUserField | None = None
    forced_tool_choice: OpenAIChatForcedToolChoice | None = None
    json_schema: OpenAIChatJsonSchema | None = None
    routing: JsonObject | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_literal_or_none(self.instruction_role, OpenAIChatInstructionRole, "instruction_role")
        _check_literal_or_none(self.max_tokens_field, OpenAIChatMaxTokensField, "max_tokens_field")
        _check_literal_or_none(self.stream_usage, OpenAIChatStreamUsage, "stream_usage")
        _check_literal_or_none(self.tool_result_name, OpenAIToolResultName, "tool_result_name")
        _check_literal_or_none(
            self.assistant_after_tool_result,
            OpenAIChatAssistantAfterToolResult,
            "assistant_after_tool_result",
        )
        _check_literal_or_none(self.thinking_format, OpenAIChatThinkingFormat, "thinking_format")
        _check_literal_or_none(self.thinking_replay, OpenAIChatThinkingReplay, "thinking_replay")
        _check_literal_or_none(
            self.assistant_reasoning_content,
            OpenAIChatAssistantReasoningContent,
            "assistant_reasoning_content",
        )
        _check_literal_or_none(self.strict_tools, OpenAIStrictTools, "strict_tools")
        _check_literal_or_none(self.builtin_tools, OpenAIChatBuiltinTools, "builtin_tools")
        _check_literal_or_none(self.cache_control, OpenAICacheControl, "cache_control")
        _check_literal_or_none(self.user_field, OpenAIChatUserField, "user_field")
        _check_literal_or_none(self.forced_tool_choice, OpenAIChatForcedToolChoice, "forced_tool_choice")
        _check_literal_or_none(self.json_schema, OpenAIChatJsonSchema, "json_schema")
        _check_json_object_or_none(self.routing, "routing")
        _check_json_object_or_none(self.extensions, "extensions")


    @classmethod
    def preset(cls, name: str) -> "OpenAIChatCompat":
        """The named server dialect from :data:`OPENAI_CHAT_PRESETS`.

        Accepts the permanent spelling aliases (``lm-studio``, ``z.ai``,
        ``openai_chat``); raises ``ValueError`` for an unknown name.
        """
        key = _preset_key(name)
        try:
            return OPENAI_CHAT_PRESETS[key]
        except KeyError:
            raise ValueError(f"unknown OpenAIChatCompat preset: {name!r}") from None


def _preset_key(name: str) -> str:
    key = name.lower().replace("-", "_").replace(" ", "_").replace(".", "_")
    return _OPENAI_CHAT_PRESET_ALIASES.get(key, key)


# Spelling aliases → canonical preset key.  Every alias is permanent.
_OPENAI_CHAT_PRESET_ALIASES: dict[str, str] = {
    "openai_chat": "openai",
    "chat": "openai",
    "chat_completions": "openai",
    "lmstudio": "ollama",
    "lm_studio": "ollama",
    "dashscope_qwen": "qwen",
    "z_ai": "zai",
}

# The Chat Completions server dialects lm15 knows, as data.  A preset
# describes ONE server's quirks (instruction role, token-limit field,
# thinking wire shape, …); it says nothing about credentials or routing —
# that is the provider registry (lm15.registry), which names a preset by
# key.  Every field value here is pinned by a live receipt or the server's
# own documentation, cited inline; a preset changes with new evidence, not
# with a hunch.
OPENAI_CHAT_PRESETS: dict[str, OpenAIChatCompat] = {
    "openai": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_completion_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai",
    ),
    # ollama / LM Studio: max_tokens, no reasoning dial on the wire.
    "ollama": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="none",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
    ),
    # Groq: server-executed builtin tools (browser_search / code_interpreter,
    # live 2026-09-01); reasoning_effort dial; no cache_control field.
    "groq": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        builtin_tools="groq",
        cache_control="none",
    ),
    # OpenRouter: unified reasoning object; OpenAI-shaped cache_control.
    "openrouter": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="openrouter",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="openai",
    ),
    # xAI, pinned live 2026-09-01 against grok-4.6: max_tokens accepted,
    # reasoning arrives as message.reasoning_content (deepseek shape),
    # stream_options.include_usage honored, no cache_control field (prompt
    # caching is automatic; usage reports cached_tokens).
    "xai": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="deepseek",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
    ),
    "vllm": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
    ),
    "sglang": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="reasoning_effort",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
    ),
    # DeepSeek (api-docs.deepseek.com, scraped 2026-09-03 —
    # lm15-contract/scrapes/deepseek/pages): thinking={"type": enabled|
    # disabled} + reasoning_effort (guide--thinking-mode.md); max_tokens
    # (chat--create.md); usage in the final stream chunk via
    # stream_options.include_usage; context caching is automatic on disk,
    # no cache_control field (guide--kv-cache.md).  thinking_replay=native
    # + include_empty: with tools present, every assistant turn must carry
    # reasoning_content back — even turns that made no call — or the API
    # answers 400 (guide--thinking-mode.md § Tool Calls).
    "deepseek": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="deepseek",
        thinking_replay="native",
        assistant_reasoning_content="include_empty",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        user_field="user_id",
    ),
    "qwen": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="qwen",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
    ),
    # Z.AI (docs.z.ai, scraped 2026-09-03 — lm15-contract/scrapes/zai/pages):
    # thinking={"type": enabled|disabled} + reasoning_effort (chat--create.md
    # ChatThinking, the deepseek wire shape; GLM-5.3 cannot disable and the
    # request fails loudly — model--glm-5.3.md); max_tokens; interleaved
    # thinking asks for reasoning_content to be returned with tool results
    # (guide--thinking-mode.md); implicit context caching, usage in
    # prompt_tokens_details.cached_tokens (guide--cache.md); user_id 6–128
    # chars is the documented identity field (chat--create.md).
    "zai": OpenAIChatCompat(
        instruction_role="system",
        max_tokens_field="max_tokens",
        stream_usage="include",
        thinking_format="deepseek",
        thinking_replay="native",
        tool_result_name="omit",
        strict_tools="omit",
        cache_control="none",
        user_field="user_id",
        forced_tool_choice="reject",
        json_schema="reject",
    ),
}


# Default base URLs for the Chat Completions presets that name a server.
# Used by OpenAIChatLM when a compat preset is given by name and no
# explicit base_url overrides it; the provider registry's access policies
# point here so there is one copy of each URL.
OPENAI_CHAT_PRESET_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "xai": "https://api.x.ai/v1",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    # api-docs.deepseek.com/first-call.md: base_url https://api.deepseek.com
    # (no /v1; the site also answers /v1, but the documented form wins).
    "deepseek": "https://api.deepseek.com",
    # docs.z.ai introduction.md: the general endpoint.  The GLM Coding Plan
    # uses a different, subscription-bound endpoint that lm15 does not name.
    "zai": "https://api.z.ai/api/paas/v4",
}


@dataclass(frozen=True, slots=True)
class ResolvedOpenAIChatCompat:
    """Fully resolved OpenAI Chat Completions compatibility policy."""

    instruction_role: Literal["developer", "system"] = "system"
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens"
    stream_usage: Literal["include", "omit"] = "include"
    tool_result_name: Literal["include", "omit"] = "omit"
    assistant_after_tool_result: Literal["insert", "omit"] = "omit"
    thinking_format: Literal[
        "none",
        "reasoning_effort",
        "openrouter",
        "deepseek",
        "qwen",
        "qwen_chat_template",
    ] = "reasoning_effort"
    thinking_replay: Literal["native", "as_text", "omit"] = "as_text"
    assistant_reasoning_content: Literal["include_empty", "omit"] = "omit"
    strict_tools: Literal["include", "omit"] = "omit"
    builtin_tools: Literal["reject", "groq"] = "reject"
    cache_control: Literal["none", "openai", "anthropic"] = "openai"
    user_field: Literal["user", "user_id"] = "user"
    forced_tool_choice: Literal["send", "reject"] = "send"
    json_schema: Literal["send", "reject"] = "send"
    routing: JsonObject | None = None
    extensions: JsonObject | None = None


_CHAT_AUTO_DEFAULTS: dict[str, str] = {
    "instruction_role": "system",
    "max_tokens_field": "max_completion_tokens",
    "stream_usage": "include",
    "tool_result_name": "omit",
    "assistant_after_tool_result": "omit",
    "thinking_format": "reasoning_effort",
    "thinking_replay": "as_text",  # decision G (2026-09-01): unsigned thinking is replayed as text, never dropped
    "assistant_reasoning_content": "omit",
    "strict_tools": "omit",
    "builtin_tools": "reject",
    "cache_control": "openai",
    "user_field": "user",
    "forced_tool_choice": "send",
    "json_schema": "send",
}


def merge_openai_chat_compat(
    base: OpenAIChatCompat,
    override: OpenAIChatCompat | None,
) -> OpenAIChatCompat:
    """Merge partial OpenAI Chat compat objects.

    None fields inherit. Non-None fields, including "auto", override.
    """
    if override is None:
        return base

    kwargs = {}
    for f in fields(OpenAIChatCompat):
        value = getattr(override, f.name)
        if f.name == "extensions":
            kwargs[f.name] = _merge_json_object(base.extensions, override.extensions)
        else:
            kwargs[f.name] = getattr(base, f.name) if value is None else value
    return OpenAIChatCompat(**kwargs)


def resolve_openai_chat_compat(partial: OpenAIChatCompat) -> ResolvedOpenAIChatCompat:
    """Resolve a partial chat compat object into concrete serializer policy."""
    kwargs: dict[str, object] = {}
    for field_name, default in _CHAT_AUTO_DEFAULTS.items():
        value = getattr(partial, field_name)
        kwargs[field_name] = default if value in {None, "auto"} else value
    return ResolvedOpenAIChatCompat(
        routing=partial.routing,
        extensions=partial.extensions,
        **kwargs,  # type: ignore[arg-type]
    )


# ─── Anthropic Messages compatibility ───────────────────────────────
#
# The Messages wire has few dialects, so this policy is small.  It exists
# for servers that speak the format but not all of it: DeepSeek's Anthropic
# endpoint (api-docs.deepseek.com guide--anthropic-api.md, live 2026-09-03)
# is the first.  Fields default to None (inherit), "auto" is the dialect's
# own behavior, same convention as the OpenAI policies above.

# How reasoning is switched and steered.  "anthropic": the model-class rule
# (MAP-7 rule 10: manual class takes thinking.type=enabled + budget_tokens,
# adaptive class takes thinking.type=adaptive + output_config.effort; off
# is absence).  "deepseek": thinking.type=enabled|disabled +
# output_config.effort, no budget (the server ignores budget_tokens), and
# off MUST be sent — absence means on for DeepSeek.
AnthropicThinkingFormat = Literal["auto", "anthropic", "deepseek"]
# Whether cache_control marks are placed.  "none": the server ignores marks
# and caches implicitly; nothing is placed and, as on the chat dialect's
# "none", an explicit CacheConfig is not an error (implicit caching applies).
AnthropicCacheControl = Literal["auto", "anthropic", "none"]
# Whether output_config.format (a JSON schema) is honoured.  "reject" raises
# before the wire: DeepSeek answers 200 and ignores the schema (live
# 2026-09-03) — silent, so the adapter refuses.
AnthropicStructuredOutput = Literal["auto", "send", "reject"]
# Whether disable_parallel_tool_use is honoured.  DeepSeek documents it as
# ignored; a silent no-op on ToolChoice.parallel=False is refused (MAP-8 §2).
AnthropicParallelToolCalls = Literal["auto", "send", "reject"]


@dataclass(frozen=True, slots=True)
class AnthropicCompat:
    """Partial compatibility policy for Anthropic Messages-family APIs.

    ``model_prefixes`` refuses, before the wire, any model id that does not
    start with one of the prefixes: DeepSeek's endpoint silently serves
    ``claude-opus*`` as ``deepseek-v4-pro`` and ``claude-haiku*`` /
    ``claude-sonnet*`` as ``deepseek-v4-flash`` (docs and live 2026-09-03);
    lm15 does not take part in a substitution the caller cannot see.
    None means any model id goes out as typed.
    """

    thinking_format: AnthropicThinkingFormat | None = None
    cache_control: AnthropicCacheControl | None = None
    structured_output: AnthropicStructuredOutput | None = None
    parallel_tool_calls: AnthropicParallelToolCalls | None = None
    model_prefixes: tuple[str, ...] | None = None
    extensions: JsonObject | None = None

    def __post_init__(self) -> None:
        _check_literal_or_none(self.thinking_format, AnthropicThinkingFormat, "thinking_format")
        _check_literal_or_none(self.cache_control, AnthropicCacheControl, "cache_control")
        _check_literal_or_none(self.structured_output, AnthropicStructuredOutput, "structured_output")
        _check_literal_or_none(self.parallel_tool_calls, AnthropicParallelToolCalls, "parallel_tool_calls")
        if self.model_prefixes is not None:
            object.__setattr__(self, "model_prefixes", tuple(str(p) for p in self.model_prefixes))
            if not self.model_prefixes:
                raise ValueError("model_prefixes must be None or a non-empty tuple")
        _check_json_object_or_none(self.extensions, "extensions")

    @classmethod
    def preset(cls, name: str) -> "AnthropicCompat":
        key = _preset_key(name)
        try:
            return ANTHROPIC_PRESETS[key]
        except KeyError:
            raise ValueError(f"unknown AnthropicCompat preset: {name!r}") from None


ANTHROPIC_PRESETS: dict[str, AnthropicCompat] = {
    "anthropic": AnthropicCompat(),
    # DeepSeek over the Anthropic wire (api-docs.deepseek.com
    # guide--anthropic-api.md; live 2026-09-03, receipts/2026-09-03-deepseek-anthropic):
    # thinking={"type": enabled|disabled} + output_config.effort, budget_tokens
    # ignored; cache_control ignored (implicit caching reports
    # cache_read_input_tokens); output_config.format accepted and ignored;
    # disable_parallel_tool_use ignored; claude-* model names silently
    # remapped to DeepSeek models.
    "deepseek": AnthropicCompat(
        thinking_format="deepseek",
        cache_control="none",
        structured_output="reject",
        parallel_tool_calls="reject",
        model_prefixes=("deepseek-",),
    ),
}

# Default base URLs for the Anthropic presets that name a server.
ANTHROPIC_PRESET_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com/v1",
    # api-docs.deepseek.com first-call.md: base_url https://api.deepseek.com/anthropic,
    # which the Anthropic SDK completes to /anthropic/v1/messages (both the
    # /anthropic and /anthropic/v1 roots answer, live 2026-09-03).
    "deepseek": "https://api.deepseek.com/anthropic/v1",
}


@dataclass(frozen=True, slots=True)
class ResolvedAnthropicCompat:
    """Fully resolved Anthropic Messages compatibility policy."""

    thinking_format: Literal["anthropic", "deepseek"] = "anthropic"
    cache_control: Literal["anthropic", "none"] = "anthropic"
    structured_output: Literal["send", "reject"] = "send"
    parallel_tool_calls: Literal["send", "reject"] = "send"
    model_prefixes: tuple[str, ...] | None = None
    extensions: JsonObject | None = None


_ANTHROPIC_AUTO_DEFAULTS: dict[str, str] = {
    "thinking_format": "anthropic",
    "cache_control": "anthropic",
    "structured_output": "send",
    "parallel_tool_calls": "send",
}


def resolve_anthropic_compat(partial: AnthropicCompat) -> ResolvedAnthropicCompat:
    """Resolve a partial Anthropic compat object into concrete serializer policy."""
    kwargs: dict[str, object] = {}
    for field_name, default in _ANTHROPIC_AUTO_DEFAULTS.items():
        value = getattr(partial, field_name)
        kwargs[field_name] = default if value in {None, "auto"} else value
    return ResolvedAnthropicCompat(
        model_prefixes=partial.model_prefixes,
        extensions=partial.extensions,
        **kwargs,  # type: ignore[arg-type]
    )


CompatProfile: TypeAlias = OpenAIResponsesCompat | OpenAIChatCompat | AnthropicCompat


# ─── Merge helpers ──────────────────────────────────────────────────


def merge_openai_responses_compat(
    base: OpenAIResponsesCompat,
    override: OpenAIResponsesCompat | None,
) -> OpenAIResponsesCompat:
    """Merge partial OpenAI Responses compat objects.

    None fields inherit. Non-None fields, including "auto", override.
    """
    if override is None:
        return base

    kwargs = {}
    for f in fields(OpenAIResponsesCompat):
        value = getattr(override, f.name)
        if f.name == "extensions":
            kwargs[f.name] = _merge_json_object(base.extensions, override.extensions)
        else:
            kwargs[f.name] = getattr(base, f.name) if value is None else value
    return OpenAIResponsesCompat(**kwargs)


def resolve_openai_responses_compat(partial: OpenAIResponsesCompat) -> ResolvedOpenAIResponsesCompat:
    """Resolve a partial compat object into concrete serializer policy."""
    developer_role = partial.developer_role
    if developer_role in {None, "auto"}:
        developer_role = "developer"

    max_field = partial.max_output_tokens_field
    if max_field in {None, "auto"}:
        max_field = "max_output_tokens"

    reasoning_format = partial.reasoning_format
    if reasoning_format in {None, "auto"}:
        reasoning_format = "responses_reasoning"

    tool_result_name = partial.tool_result_name
    if tool_result_name in {None, "auto"}:
        tool_result_name = "omit"

    strict_tools = partial.strict_tools
    if strict_tools in {None, "auto"}:
        strict_tools = "omit"

    cache_control = partial.cache_control
    if cache_control in {None, "auto"}:
        cache_control = "openai"

    return ResolvedOpenAIResponsesCompat(
        developer_role=developer_role,  # type: ignore[arg-type]
        max_output_tokens_field=max_field,  # type: ignore[arg-type]
        reasoning_format=reasoning_format,  # type: ignore[arg-type]
        tool_result_name=tool_result_name,  # type: ignore[arg-type]
        strict_tools=strict_tools,  # type: ignore[arg-type]
        cache_control=cache_control,  # type: ignore[arg-type]
        routing=partial.routing,
        extensions=partial.extensions,
    )
