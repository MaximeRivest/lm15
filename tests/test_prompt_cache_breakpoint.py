"""OpenAI explicit prompt-cache breakpoints (gpt-5.6+, 2026-09-01).

Both OpenAI dialects now carry ``prompt_cache_breakpoint: {"mode":
"explicit"}`` on text content blocks — Anthropic's ``cache_control``
model arriving on OpenAI.  ``CacheConfig.prefix_until_index`` maps to a
breakpoint on the last text block of that message.  Live receipt
2026-09-01 (gpt-5.6-sol, ~3k-token prefix): first call
``input_tokens_details.cache_write_tokens == 3066``, second call
``cached_tokens == 3066``; gpt-4.1-mini rejects the field with HTTP 400
(loud failure is the contract, no client-side model gating).

Wire equality for the pinned shapes is the contract's job
(cases openai.prompt_cache_breakpoint, openai_chat.prompt_cache_breakpoint);
these tests pin the mapping table, the raises, and the new usage field.
"""
from __future__ import annotations

import json

import pytest

from lm15 import CacheConfig, Config, Message, Request, UnsupportedFeatureError
from lm15.providers import HttpResponse, OpenAIChatLM, OpenAILM
from lm15.testing import FakeTransport
from lm15.types import ImagePart, TextPart, tool_call, tool_result


def _req(idx: int | None, *, mode: str = "auto", messages=None) -> Request:
    cache = CacheConfig(mode=mode, prefix_until_index=idx) if (idx is not None or mode != "auto") else None
    return Request(
        model="gpt-5.6-sol",
        messages=messages or (
            Message.user("A long reusable prefix."),
            Message.assistant("ok"),
            Message.user("The question."),
        ),
        config=Config(cache=cache) if cache is not None else Config(),
    )


def _body(lm, request: Request) -> dict:
    return json.loads(lm.build_request(request, stream=False).body)


# ─── Responses dialect ───────────────────────────────────────────────

def test_responses_breakpoint_on_last_text_block_of_prefix_message() -> None:
    body = _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req(0))
    first = body["input"][0]["content"][-1]
    assert first["type"] == "input_text"
    assert first["prompt_cache_breakpoint"] == {"mode": "explicit"}
    # Only the prefix message carries it; no request-level option is added
    # (OpenAI's implicit mode stays the server default).
    assert "prompt_cache_breakpoint" not in json.dumps(body["input"][1:])
    assert "prompt_cache_options" not in body


def test_responses_no_breakpoint_without_prefix_index() -> None:
    body = _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req(None))
    assert "prompt_cache_breakpoint" not in json.dumps(body)


def test_responses_breakpoint_index_clamps_to_last_message() -> None:
    # The Anthropic adapter's precedent: an index past the end means "the
    # whole transcript is the prefix".
    body = _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req(99))
    assert body["input"][-1]["content"][-1]["prompt_cache_breakpoint"] == {"mode": "explicit"}


def test_responses_cache_off_sends_nothing() -> None:
    request = Request(
        model="gpt-5.6-sol",
        messages=(Message.user("prefix"), Message.user("q")),
        config=Config(cache=CacheConfig(mode="off")),
    )
    body = _body(OpenAILM(api_key="k", transport=FakeTransport([])), request)
    assert "prompt_cache_breakpoint" not in json.dumps(body)


def test_responses_breakpoint_on_assistant_message_raises() -> None:
    with pytest.raises(UnsupportedFeatureError, match="prefix_until_index=1 points at a assistant"):
        _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req(1))


def test_responses_breakpoint_on_tool_message_raises() -> None:
    messages = (
        Message.user("prefix"),
        Message.assistant(tool_call("c1", "f", {})),
        Message.tool(tool_result("c1", "out")),
        Message.user("q"),
    )
    with pytest.raises(UnsupportedFeatureError, match="points at a tool"):
        _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req(2, messages=messages))


def test_responses_breakpoint_needs_trailing_text_block() -> None:
    messages = (
        Message.user((TextPart("look"), ImagePart(url="https://x/y.png"))),
        Message.user("q"),
    )
    with pytest.raises(UnsupportedFeatureError, match="last block is not text"):
        _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req(0, messages=messages))


def test_responses_parse_cache_write_tokens() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = {
        "id": "resp_1", "model": "gpt-5.6-sol", "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": "3861"}]}],
        "usage": {"input_tokens": 3083, "output_tokens": 6, "total_tokens": 3089,
                  "input_tokens_details": {"cache_write_tokens": 3066, "cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }
    resp = lm.parse_response(_req(0), HttpResponse(200, "OK", [], json.dumps(body).encode()))
    assert resp.usage.cache_write_tokens == 3066
    assert resp.usage.cache_read_tokens == 0


# ─── Chat Completions dialect ────────────────────────────────────────

def test_chat_breakpoint_forces_array_content() -> None:
    body = _body(OpenAIChatLM(api_key="k", transport=FakeTransport([])), _req(0))
    first = body["messages"][0]
    assert isinstance(first["content"], list)
    assert first["content"][-1] == {
        "type": "text", "text": "A long reusable prefix.",
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    # Non-prefix single-text messages keep the plain-string form.
    assert body["messages"][-1]["content"] == "The question."


def test_chat_breakpoint_skipped_when_compat_has_no_cache_control() -> None:
    # groq/vllm/ollama presets declare cache_control="none": the same
    # gating as prompt_cache_key — nothing is sent.
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]), compat="groq")
    body = _body(lm, _req(0))
    assert "prompt_cache_breakpoint" not in json.dumps(body)
    assert body["messages"][0]["content"] == "A long reusable prefix."


def test_chat_breakpoint_on_assistant_message_raises() -> None:
    with pytest.raises(UnsupportedFeatureError, match="points at a assistant"):
        _body(OpenAIChatLM(api_key="k", transport=FakeTransport([])), _req(1))


def test_chat_parse_cache_write_tokens() -> None:
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    body = {
        "id": "chatcmpl-1", "model": "gpt-5.6-sol",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "3861"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3083, "completion_tokens": 6, "total_tokens": 3089,
                  "prompt_tokens_details": {"cache_write_tokens": 3066, "cached_tokens": 0}},
    }
    resp = lm.parse_response(_req(0), HttpResponse(200, "OK", [], json.dumps(body).encode()))
    assert resp.usage.cache_write_tokens == 3066
