"""MAP-7 reasoning: the mapping table, the raises, the replay (2026-09-02).

Wire bytes are pinned by the contract (anthropic.reasoning_adaptive,
anthropic.reasoning_budget, gemini.thinking_level, openai.reasoning_replay,
gemini.thinking); these pin the table and the no-silent-drop raises.
Receipts: lm15-contract/research/reasoning/.
"""
from __future__ import annotations

import json

import pytest

from lm15 import (AnthropicLM, Config, ContinuationState, GeminiLM, Message, OpenAIChatLM, OpenAILM, Reasoning, Request,
                  ThinkingPart, ToolCallPart, UnsupportedFeatureError, XaiLM)
from lm15.providers import HttpResponse
from lm15.providers.anthropic import anthropic_adaptive_class
from lm15.providers.common import EFFORT_THINKING_BUDGETS
from lm15.providers.gemini import gemini_level_class
from lm15.testing import FakeTransport


def _req(model: str, reasoning: Reasoning | None, **cfg) -> Request:
    return Request(model=model, messages=[Message.user("q")], config=Config(reasoning=reasoning, **cfg))


def _body(lm, request: Request) -> dict:
    return json.loads(lm.build_request(request, stream=False).body)


def test_model_class_tables() -> None:
    assert anthropic_adaptive_class("claude-sonnet-5") and anthropic_adaptive_class("claude-opus-4-8") and anthropic_adaptive_class("claude-fable-5-1")
    assert not anthropic_adaptive_class("claude-sonnet-4-5-20250929") and not anthropic_adaptive_class("claude-haiku-4-5-20251001")
    assert gemini_level_class("gemini-3.7-flash") and gemini_level_class("models/gemini-3.1-pro-preview")
    assert not gemini_level_class("gemini-2.5-flash")
    assert EFFORT_THINKING_BUDGETS == {"minimal": 1024, "low": 2048, "medium": 8192, "high": 16384, "xhigh": 24576, "max": 32768}


# ─── OpenAI ──────────────────────────────────────────────────────────

def test_openai_effort_verbatim_budget_raises() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    for eff in ("minimal", "low", "medium", "high", "xhigh", "max"):
        assert _body(lm, _req("gpt-5.6-sol", Reasoning(effort=eff)))["reasoning"] == {"effort": eff}
    assert _body(lm, _req("gpt-5.6-sol", Reasoning(effort="off")))["reasoning"] == {"effort": "none"}
    assert _body(lm, _req("gpt-5.6-sol", Reasoning(effort="low", summary="detailed")))["reasoning"] == {"effort": "low", "summary": "detailed"}
    with pytest.raises(UnsupportedFeatureError, match="thinking_budget"):
        _body(lm, _req("gpt-5.6-sol", Reasoning(effort="low", thinking_budget=1024)))


def test_openai_reasoning_item_round_trip() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = {"id": "resp_1", "model": "gpt-5.4-mini", "status": "completed",
            "output": [{"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA", "summary": []},
                       {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{\"n\": 7}"}],
            "usage": {"input_tokens": 10, "output_tokens": 20, "output_tokens_details": {"reasoning_tokens": 9}}}
    resp = lm.parse_response(_req("gpt-5.4-mini", Reasoning(effort="low")), HttpResponse(200, "OK", [], json.dumps(body).encode()))
    think = resp.message.parts[0]
    assert isinstance(think, ThinkingPart) and think.text == ""
    assert think.continuation[0] == ContinuationState(provider="openai", kind="reasoning_item", data={"id": "rs_1", "encrypted_content": "gAAA"})
    assert resp.usage.reasoning_tokens == 9
    # replay: the item goes back verbatim, summary required even when empty
    nxt = Request(model="gpt-5.4-mini", messages=[Message.user("q"), resp.message, Message.tool({"c1": "49"})], config=Config(reasoning=Reasoning(effort="low")))
    items = _body(lm, nxt)["input"]
    assert items[1] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAA", "summary": []}
    assert items[2]["type"] == "function_call" and items[3]["type"] == "function_call_output"


def test_openai_unsigned_thinking_replays_as_text() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    req = Request(model="gpt-5.6-sol", messages=[Message.user("q"), Message.assistant((ThinkingPart("I reasoned"), )), Message.user("more")])
    items = _body(lm, req)["input"]
    assert items[1] == {"role": "assistant", "content": [{"type": "output_text", "text": "I reasoned"}]}


# ─── Chat dialect ────────────────────────────────────────────────────

def test_chat_dialect_effort_summary_and_groq_visibility() -> None:
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    assert _body(lm, _req("gpt-5.6-sol", Reasoning(effort="max")))["reasoning_effort"] == "max"
    with pytest.raises(UnsupportedFeatureError, match="thinking_budget"):
        _body(lm, _req("gpt-5.6-sol", Reasoning(effort="low", thinking_budget=100)))
    with pytest.raises(UnsupportedFeatureError, match="detail level"):
        _body(lm, _req("gpt-5.6-sol", Reasoning(effort="low", summary="concise")))
    groq = OpenAIChatLM(api_key="k", transport=FakeTransport([]), compat="groq")
    assert _body(groq, _req("openai/gpt-oss-20b", Reasoning(effort="low", summary="auto")))["reasoning_format"] == "parsed"
    assert "reasoning_format" not in _body(groq, _req("openai/gpt-oss-20b", Reasoning(effort="low")))
    # decision G: unsigned thinking is replayed as text by default on the dialect
    req = Request(model="gpt-5.6-sol", messages=[Message.user("q"), Message.assistant((ThinkingPart("I reasoned"), )), Message.user("more")])
    assert _body(lm, req)["messages"][1]["content"] == "I reasoned"


# ─── Anthropic ───────────────────────────────────────────────────────

def test_anthropic_adaptive_class() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    b = _body(lm, _req("claude-sonnet-5", Reasoning(effort="xhigh"), max_tokens=500))
    assert b["thinking"] == {"type": "adaptive"} and b["output_config"] == {"effort": "xhigh"} and b["max_tokens"] == 500
    assert "thinking" not in _body(lm, _req("claude-sonnet-5", Reasoning(effort="off")))
    with pytest.raises(UnsupportedFeatureError, match="thinking_budget"):
        _body(lm, _req("claude-sonnet-5", Reasoning(effort="high", thinking_budget=2048)))
    with pytest.raises(UnsupportedFeatureError, match="minimal"):
        _body(lm, _req("claude-sonnet-5", Reasoning(effort="minimal")))
    with pytest.raises(UnsupportedFeatureError, match="detail level"):
        _body(lm, _req("claude-sonnet-5", Reasoning(effort="high", summary="detailed")))
    # summary=auto is satisfied: thinking blocks are always returned
    assert "summary" not in json.dumps(_body(lm, _req("claude-sonnet-5", Reasoning(effort="high", summary="auto"))))
    # response_format and effort share output_config
    b = _body(lm, _req("claude-sonnet-5", Reasoning(effort="low"), response_format={"type": "json_object"}))
    assert b["output_config"]["effort"] == "low" and "format" in b["output_config"]


def test_anthropic_manual_class() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    b = _body(lm, _req("claude-sonnet-4-5", Reasoning(effort="medium"), max_tokens=500))
    assert b["thinking"] == {"type": "enabled", "budget_tokens": 8192} and b["max_tokens"] == 8192 + 500
    b = _body(lm, _req("claude-haiku-4-5-20251001", Reasoning(effort="high", thinking_budget=1500), max_tokens=500))
    assert b["thinking"] == {"type": "enabled", "budget_tokens": 1500} and b["max_tokens"] == 2000
    assert "output_config" not in b


# ─── Gemini ──────────────────────────────────────────────────────────

def test_gemini_two_classes() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    tc = lambda m, r: _body(lm, _req(m, r))["generationConfig"]["thinkingConfig"]  # noqa: E731
    assert tc("gemini-2.5-flash", Reasoning(effort="low")) == {"thinkingBudget": 2048}
    assert tc("gemini-2.5-flash", Reasoning(effort="off")) == {"thinkingBudget": 0}
    assert tc("gemini-3.7-flash", Reasoning(effort="medium", summary="auto")) == {"includeThoughts": True, "thinkingLevel": "medium"}
    assert tc("gemini-3.7-flash", Reasoning(effort="medium", thinking_budget=512)) == {"thinkingBudget": 512}
    for eff in ("xhigh", "max"):
        with pytest.raises(UnsupportedFeatureError, match="no thinkingLevel"):
            tc("gemini-3.7-flash", Reasoning(effort=eff))
    with pytest.raises(UnsupportedFeatureError, match="cannot be disabled"):
        tc("gemini-3.7-flash", Reasoning(effort="off"))


# ─── xAI ─────────────────────────────────────────────────────────────

def test_xai_effort_verbatim_off_raises() -> None:
    lm = XaiLM(api_key="k", transport=FakeTransport([]))
    assert _body(lm, _req("grok-4.6", Reasoning(effort="xhigh")))["reasoning_effort"] == "xhigh"
    with pytest.raises(UnsupportedFeatureError, match="cannot be disabled"):
        _body(lm, _req("grok-4.6", Reasoning(effort="off")))
