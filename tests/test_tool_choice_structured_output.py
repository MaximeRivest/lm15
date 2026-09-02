"""MAP-8: tool-choice silent cells + the canonical response_format (INV-050).

Receipts: lm15-contract/research/tool-choice/ (2026-09-02).
"""
from __future__ import annotations

import json

import pytest

from lm15 import AnthropicLM, Config, FunctionTool, GeminiLM, Message, OpenAIChatLM, OpenAILM, Request, ToolChoice, UnsupportedFeatureError, XaiLM
from lm15.testing import FakeTransport

TOOLS = (FunctionTool(name="lookup"), FunctionTool(name="weather"))
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}


def _req(model: str, **cfg) -> Request:
    return Request(model=model, messages=[Message.user("q")], tools=TOOLS, config=Config(**cfg))


def _body(lm, request: Request) -> dict:
    return json.loads(lm.build_request(request, stream=False).body)


# ─── INV-050 ─────────────────────────────────────────────────────────

def test_response_format_has_exactly_two_shapes() -> None:
    Config(response_format={"type": "json_object"})
    Config(response_format={"type": "json_schema", "schema": SCHEMA, "name": "person", "strict": True})
    for bad in ({"format": {"type": "json_schema"}}, {"response_mime_type": "application/json"},
                {"type": "json_schema", "json_schema": {"name": "x"}}, {"type": "object", "properties": {}},
                {"type": "json_object", "schema": SCHEMA}, {"type": "json_schema"}):
        with pytest.raises((ValueError, TypeError), match="response_format"):
            Config(response_format=bad)


def test_schema_is_verbatim_on_every_wire() -> None:
    rf = {"type": "json_schema", "schema": {**SCHEMA, "properties": {"age": {"type": "integer", "minimum": 0}}}, "name": "p", "strict": True}
    assert _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req("gpt-5.6-sol", response_format=rf))["text"] == {
        "format": {"type": "json_schema", "name": "p", "schema": rf["schema"], "strict": True}}
    assert _body(OpenAIChatLM(api_key="k", transport=FakeTransport([])), _req("gpt-5.4-mini", response_format=rf))["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "p", "schema": rf["schema"], "strict": True}}
    # Anthropic rejects minimum server-side; lm15 does not strip it
    assert _body(AnthropicLM(api_key="k", transport=FakeTransport([])), _req("claude-sonnet-5", response_format=rf))["output_config"] == {
        "format": {"type": "json_schema", "schema": rf["schema"]}}
    g = _body(GeminiLM(api_key="k", transport=FakeTransport([])), _req("gemini-2.5-flash", response_format=rf))["generationConfig"]
    assert g["responseMimeType"] == "application/json" and g["responseJsonSchema"] == rf["schema"]


def test_json_object_per_provider() -> None:
    rf = {"type": "json_object"}
    assert _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req("gpt-5.6-sol", response_format=rf))["text"] == {"format": {"type": "json_object"}}
    assert _body(GeminiLM(api_key="k", transport=FakeTransport([])), _req("gemini-2.5-flash", response_format=rf))["generationConfig"] == {"responseMimeType": "application/json"}
    with pytest.raises(UnsupportedFeatureError, match="any-JSON"):
        _body(AnthropicLM(api_key="k", transport=FakeTransport([])), _req("claude-sonnet-5", response_format=rf))


def test_openai_name_defaults_to_response() -> None:
    b = _body(OpenAILM(api_key="k", transport=FakeTransport([])), _req("gpt-5.6-sol", response_format={"type": "json_schema", "schema": SCHEMA}))
    assert b["text"]["format"]["name"] == "response" and "strict" not in b["text"]["format"]


# ─── tool choice silent cells ────────────────────────────────────────

def test_gemini_parallel_false_raises() -> None:
    lm = GeminiLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="parallel"):
        _body(lm, _req("gemini-2.5-flash", tool_choice=ToolChoice(parallel=False)))
    assert _body(lm, _req("gemini-2.5-flash", tool_choice=ToolChoice(parallel=True)))["toolConfig"] == {"functionCallingConfig": {"mode": "AUTO"}}


def test_xai_allowlist_and_forced_with_format_raise() -> None:
    lm = XaiLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="allowed subsets"):
        _body(lm, _req("grok-4.6", tool_choice=ToolChoice(allowed=("lookup",))))
    with pytest.raises(UnsupportedFeatureError, match="allowed subsets"):
        _body(lm, _req("grok-4.6", tool_choice=ToolChoice(mode="required", allowed=("lookup", "weather"))))
    assert _body(lm, _req("grok-4.6", tool_choice=ToolChoice(mode="required", allowed=("lookup",))))["tool_choice"] == {"type": "function", "function": {"name": "lookup"}}
    with pytest.raises(UnsupportedFeatureError, match="forced tool"):
        _body(lm, _req("grok-4.6", tool_choice=ToolChoice(mode="required"), response_format={"type": "json_object"}))
