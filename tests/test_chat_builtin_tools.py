"""BuiltinTool policy on the Chat Completions dialect (2026-09-01).

The base chat wire carries function/custom tools only, and unproven
compat servers may silently IGNORE unknown tool types (OpenRouter,
verified live 2026-09-01: 200 OK, no search, no error). The dialect
therefore raises by default and maps only where a live capture proves
server-side execution — Groq's browser_search / code_interpreter
(bodies in curl-fixtures/chat-builtin-tools-2026-09-01/).
"""
from __future__ import annotations

import pytest

from lm15 import BuiltinTool, FunctionTool, Message, Request, UnsupportedFeatureError
from lm15.compat import OpenAIChatCompat, resolve_openai_chat_compat
from lm15.providers.openai_chat import OpenAIChatLM
from lm15.testing import FakeTransport


def req(tools) -> Request:
    return Request(model="m", messages=(Message.user("hi"),), tools=tuple(tools))


def lm(**kwargs) -> OpenAIChatLM:
    return OpenAIChatLM(api_key="k", transport=FakeTransport([]), **kwargs)


def test_default_rejects_builtin_tools() -> None:
    with pytest.raises(UnsupportedFeatureError, match="builtin tool 'web_search'"):
        lm()._payload(req([BuiltinTool(name="web_search")]), stream=False)


def test_openrouter_preset_rejects() -> None:
    # OpenRouter accepts the request and silently ignores unknown tool
    # types (live 2026-09-01) — the client-side raise is the protection.
    with pytest.raises(UnsupportedFeatureError, match="not supported"):
        lm(compat="openrouter")._payload(req([BuiltinTool(name="web_search")]), stream=False)


def test_groq_maps_builtin_tools() -> None:
    body = lm(compat="groq")._payload(
        req([BuiltinTool(name="web_search"), BuiltinTool(name="code_execution")]), stream=False
    )
    assert body["tools"] == [{"type": "browser_search"}, {"type": "code_interpreter"}]


def test_groq_builtin_config_merges() -> None:
    body = lm(compat="groq")._payload(
        req([BuiltinTool(name="web_search", config={"search_settings": {"include_domains": ["arxiv.org"]}})]),
        stream=False,
    )
    assert body["tools"] == [
        {"type": "browser_search", "search_settings": {"include_domains": ["arxiv.org"]}}
    ]


def test_groq_unmapped_builtin_name_raises() -> None:
    with pytest.raises(UnsupportedFeatureError, match="no Groq wire mapping"):
        lm(compat="groq")._payload(req([BuiltinTool(name="computer_use")]), stream=False)


def test_groq_mixed_function_and_builtin() -> None:
    body = lm(compat="groq")._payload(
        req([FunctionTool(name="add", parameters={"type": "object", "properties": {}}),
             BuiltinTool(name="web_search")]),
        stream=False,
    )
    assert [t["type"] for t in body["tools"]] == ["function", "browser_search"]


def test_builtin_tools_knob_resolution() -> None:
    assert resolve_openai_chat_compat(OpenAIChatCompat()).builtin_tools == "reject"
    assert resolve_openai_chat_compat(OpenAIChatCompat(builtin_tools="auto")).builtin_tools == "reject"
    assert resolve_openai_chat_compat(OpenAIChatCompat.preset("groq")).builtin_tools == "groq"
    with pytest.raises(ValueError):
        OpenAIChatCompat(builtin_tools="banana")  # type: ignore[arg-type]


def test_named_builtin_forcing_still_raises_on_groq() -> None:
    # Offering is proven; NAMED forcing has no documented Groq wire form.
    # Generic mode="required" (no allowlist) still flows through as the
    # plain "required" string — Groq accepts it at the wire.
    from lm15 import Config, ToolChoice

    r = Request(
        model="m", messages=(Message.user("hi"),),
        tools=(BuiltinTool(name="web_search"),),
        config=Config(tool_choice=ToolChoice(mode="required", allowed=("web_search",))),
    )
    with pytest.raises(UnsupportedFeatureError, match="cannot force builtin"):
        lm(compat="groq")._payload(r, stream=False)
    r2 = Request(
        model="m", messages=(Message.user("hi"),),
        tools=(BuiltinTool(name="web_search"),),
        config=Config(tool_choice=ToolChoice(mode="required")),
    )
    assert lm(compat="groq")._payload(r2, stream=False)["tool_choice"] == "required"
