"""Logprobs output surface + kind-aware ToolChoice resolution (2026-09-01).

Wire equality for the promoted spellings is pinned by the contract's
request direction (cases openai.top_logprobs, openai.include,
openai.tool_choice_builtin, openai.tool_choice_allowed,
anthropic.tool_choice_builtin, openai_chat.logprobs). These tests pin
the mapping table, the no-silent-drop raises, and the stream
materialization semantics.
"""
from __future__ import annotations

import json

import pytest

from lm15 import (
    BuiltinTool,
    Config,
    FunctionTool,
    Message,
    Request,
    Response,
    TextDelta,
    TokenLogprob,
    ToolChoice,
    TopLogprob,
    UnsupportedFeatureError,
    Usage,
)
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM
from lm15.providers.base import HttpResponse
from lm15.providers.openai_chat import OpenAIChatLM
from lm15.result import StreamAccumulator, materialize_response, response_to_events
from lm15.serde import (
    config_from_dict,
    config_to_dict,
    delta_from_dict,
    delta_to_dict,
    response_from_dict,
    response_to_dict,
)
from lm15.testing import FakeTransport
from lm15.types import StreamDeltaEvent, StreamEndEvent, StreamStartEvent


def req(**config) -> Request:
    return Request(model="m", messages=(Message.user("hi"),), config=Config(**config))


def http(body: dict) -> HttpResponse:
    raw = json.dumps(body).encode()
    return HttpResponse(status=200, reason="OK", headers={}, body=raw)


TOKEN = TokenLogprob(
    token="hi",
    logprob=-0.25,
    bytes=(104, 105),
    top=(TopLogprob(token="hi", logprob=-0.25), TopLogprob(token="Hi", logprob=-2.5)),
)


# ─── Types ───────────────────────────────────────────────────────────

def test_config_logprobs_validation() -> None:
    assert Config(logprobs=0).logprobs == 0
    assert Config(logprobs=3.0).logprobs == 3  # float-coerced
    with pytest.raises(ValueError):
        Config(logprobs=-1)
    with pytest.raises(TypeError):
        Config(logprobs=True)  # bool is not the canonical int form


def test_token_logprob_validation() -> None:
    t = TokenLogprob(token="", logprob=0)  # "" and 0.0 are legal data
    assert t.logprob == 0.0 and isinstance(t.logprob, float)
    assert TOKEN.top[0].token == "hi"
    with pytest.raises(TypeError):
        TokenLogprob(token="x", logprob=-1.0, bytes=(-1,))
    with pytest.raises(TypeError):
        TokenLogprob(token="x", logprob=-1.0, top=("not a top",))  # type: ignore[arg-type]


def test_response_logprobs_field() -> None:
    r = Response(
        id=None, model="m", message=Message.assistant("hi"),
        finish_reason="stop", usage=Usage(), logprobs=(TOKEN,),
    )
    assert r.logprobs == (TOKEN,)
    assert "logprobs=<1 tokens>" in repr(r)


# ─── Serde ───────────────────────────────────────────────────────────

def test_config_serde_zero_is_data() -> None:
    d = config_to_dict(Config(logprobs=0))
    assert d == {"logprobs": 0}
    assert config_from_dict(d).logprobs == 0
    assert "logprobs" not in config_to_dict(Config())


def test_delta_and_response_serde_roundtrip() -> None:
    delta = TextDelta(text="hi", logprobs=(TOKEN,))
    assert delta_from_dict(delta_to_dict(delta)) == delta
    assert "logprobs" not in delta_to_dict(TextDelta(text="hi"))

    r = Response(
        id="r1", model="m", message=Message.assistant("hi"),
        finish_reason="stop", usage=Usage(input_tokens=1, output_tokens=1),
        logprobs=(TOKEN,),
    )
    d = response_to_dict(r)
    assert d["logprobs"][0]["token"] == "hi"
    assert d["logprobs"][0]["top"][1] == {"token": "Hi", "logprob": -2.5}
    assert response_from_dict(d).logprobs == (TOKEN,)
    assert "logprobs" not in response_to_dict(
        Response(id=None, model="m", message=Message.assistant("x"), finish_reason="stop", usage=Usage())
    )


# ─── Request mapping: logprobs ───────────────────────────────────────

def test_openai_logprobs_payload() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(req(logprobs=2), stream=False)
    assert body["top_logprobs"] == 2
    assert body["include"] == ["message.output_text.logprobs"]
    assert "top_logprobs" not in lm._payload(req(), stream=False)


def test_openai_chat_logprobs_payload() -> None:
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(req(logprobs=0), stream=False)
    assert body["logprobs"] is True
    assert "top_logprobs" not in body  # 0 = chosen only
    assert lm._payload(req(logprobs=5), stream=False)["top_logprobs"] == 5


def test_gemini_logprobs_payload() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    gc = lm._payload(req(logprobs=0))["generationConfig"]
    assert gc["responseLogprobs"] is True
    assert "logprobs" not in gc
    assert lm._payload(req(logprobs=4))["generationConfig"]["logprobs"] == 4


def test_anthropic_logprobs_raises() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="logprobs"):
        lm._payload(req(logprobs=0), stream=False)


# ─── Response mapping: logprobs ──────────────────────────────────────

def test_openai_parse_logprobs() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = {
        "id": "resp_1", "model": "m", "status": "completed",
        "output": [{"type": "message", "content": [{
            "type": "output_text", "text": "hi", "annotations": [],
            "logprobs": [{"token": "hi", "logprob": -0.25, "bytes": [104, 105],
                          "top_logprobs": [{"token": "hi", "logprob": -0.25},
                                           {"token": "Hi", "logprob": -2.5}]}],
        }]}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    resp = lm.parse_response(req(logprobs=2), http(body))
    assert resp.logprobs == (TOKEN,)
    # Empty wire list (logprobs not requested) stays None, not ().
    body["output"][0]["content"][0]["logprobs"] = []
    assert lm.parse_response(req(), http(body)).logprobs is None


def test_openai_chat_parse_logprobs() -> None:
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    body = {
        "id": "c1", "model": "m",
        "choices": [{
            "message": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "logprobs": {"content": [{"token": "hi", "logprob": -0.25, "bytes": [104, 105],
                                      "top_logprobs": [{"token": "hi", "logprob": -0.25},
                                                       {"token": "Hi", "logprob": -2.5}]}],
                         "refusal": None},
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }
    assert lm.parse_response(req(logprobs=2), http(body)).logprobs == (TOKEN,)


def test_gemini_parse_logprobs() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    body = {
        "candidates": [{
            "content": {"parts": [{"text": "hi"}], "role": "model"},
            "finishReason": "STOP",
            "logprobsResult": {
                "chosenCandidates": [{"token": "hi", "logProbability": -0.25, "tokenId": 544}],
                "topCandidates": [{"candidates": [
                    {"token": "hi", "logProbability": -0.25, "tokenId": 544},
                    {"token": "yo", "logProbability": -3.0, "tokenId": 921},
                ]}],
                "logProbabilitySum": -0.25,
            },
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
    }
    resp = lm.parse_response(req(logprobs=2), http(body))
    assert resp.logprobs == (TokenLogprob(
        token="hi", logprob=-0.25, token_id=544,
        top=(TopLogprob(token="hi", logprob=-0.25, token_id=544),
             TopLogprob(token="yo", logprob=-3.0, token_id=921)),
    ),)


# ─── Streaming: logprobs ─────────────────────────────────────────────

def test_stream_materialization_concatenates_logprobs() -> None:
    t2 = TokenLogprob(token=" world", logprob=-0.5)
    events = [
        StreamStartEvent(id="r1", model="m"),
        StreamDeltaEvent(delta=TextDelta(text="hi", logprobs=(TOKEN,))),
        StreamDeltaEvent(delta=TextDelta(text=" world", logprobs=(t2,))),
        StreamEndEvent(finish_reason="stop", usage=Usage(input_tokens=1, output_tokens=2)),
    ]
    resp = materialize_response(iter(events), req())
    assert resp.text == "hi world"
    assert resp.logprobs == (TOKEN, t2)


def test_response_to_events_roundtrip_with_logprobs() -> None:
    r = Response(
        id="r1", model="m", message=Message.assistant("hi"),
        finish_reason="stop", usage=Usage(input_tokens=1, output_tokens=1),
        logprobs=(TOKEN,),
    )
    assert materialize_response(response_to_events(r), req()).logprobs == (TOKEN,)


# ─── ToolChoice: kind-aware resolution ───────────────────────────────

WEATHER = FunctionTool(name="get_weather", parameters={"type": "object", "properties": {}})
SEARCH = BuiltinTool(name="web_search")


def tooled(tools, **tc) -> Request:
    return Request(
        model="m", messages=(Message.user("hi"),), tools=tuple(tools),
        config=Config(tool_choice=ToolChoice(**tc)),
    )


def test_openai_forces_builtin() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(tooled([SEARCH], mode="required", allowed=("web_search",)), stream=False)
    assert body["tool_choice"] == {"type": "web_search_preview"}


def test_openai_forces_function() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(tooled([WEATHER], mode="required", allowed=("get_weather",)), stream=False)
    assert body["tool_choice"] == {"type": "function", "name": "get_weather"}


def test_openai_auto_single_name_no_longer_forces() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(tooled([WEATHER, SEARCH], mode="auto", allowed=("get_weather",)), stream=False)
    assert body["tool_choice"] == {
        "type": "allowed_tools", "mode": "auto",
        "tools": [{"type": "function", "name": "get_weather"}],
    }


def test_openai_mixed_allowlist() -> None:
    lm = OpenAILM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(tooled([WEATHER, SEARCH], mode="required", allowed=("get_weather", "web_search")), stream=False)
    assert body["tool_choice"] == {
        "type": "allowed_tools", "mode": "required",
        "tools": [{"type": "function", "name": "get_weather"}, {"type": "web_search_preview"}],
    }


def test_anthropic_forces_builtin_by_name() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(tooled([SEARCH], mode="required", allowed=("web_search",)), stream=False)
    assert body["tool_choice"] == {"type": "tool", "name": "web_search"}


def test_anthropic_full_allowlist_is_no_restriction() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    body = lm._payload(
        tooled([WEATHER, SEARCH], mode="required", allowed=("get_weather", "web_search")), stream=False
    )
    assert body["tool_choice"] == {"type": "any"}
    body = lm._payload(
        tooled([WEATHER, SEARCH], mode="auto", allowed=("get_weather", "web_search")), stream=False
    )
    assert body["tool_choice"] == {"type": "auto"}


def test_anthropic_proper_subset_raises() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="subset"):
        lm._payload(tooled([WEATHER, SEARCH], mode="auto", allowed=("get_weather",)), stream=False)


def test_gemini_builtin_forcing_raises() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="builtin"):
        lm._payload(tooled([SEARCH], mode="required", allowed=("web_search",)))


def test_gemini_auto_subset_uses_validated_mode() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    cfg = lm._payload(tooled([WEATHER], mode="auto", allowed=("get_weather",)))["toolConfig"]
    assert cfg["functionCallingConfig"] == {"mode": "VALIDATED", "allowedFunctionNames": ["get_weather"]}
    cfg = lm._payload(tooled([WEATHER], mode="required", allowed=("get_weather",)))["toolConfig"]
    assert cfg["functionCallingConfig"] == {"mode": "ANY", "allowedFunctionNames": ["get_weather"]}


def test_openai_chat_builtin_forcing_raises_and_allowlist_nested() -> None:
    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="builtin"):
        lm._payload(tooled([SEARCH], mode="required", allowed=("web_search",)), stream=False)
    body = lm._payload(tooled([WEATHER], mode="auto", allowed=("get_weather",)), stream=False)
    assert body["tool_choice"] == {
        "type": "allowed_tools",
        "allowed_tools": {"mode": "auto", "tools": [{"type": "function", "function": {"name": "get_weather"}}]},
    }
    body = lm._payload(tooled([WEATHER], mode="required", allowed=("get_weather",)), stream=False)
    assert body["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
