"""The whole public path a tool loop takes, with media in the result:

    provider response (live body) → canonical Response → serde round trip
    → the application appends a ToolResultPart with an image → build_request

Bodies are the contract's captured ones (turn 1 of the 2026-09-07 tool-result
matrix and the pinned streaming_tool_call captures); the second wire is the
one the live matrix proved the servers accept. No network.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from lm15 import AnthropicLM, GeminiLM, ImagePart, Message, OpenAILM, Request, TextPart, ToolCallPart, XaiLM, serde, tool_result
from lm15.providers.base import HttpResponse
from lm15.result import materialize_response
from lm15.types import FunctionTool

CONTRACT = Path(os.environ.get("LM15_CONTRACT_DIR", Path(__file__).resolve().parents[2] / "lm15-contract"))
RECEIPTS = CONTRACT / "receipts" / "2026-09-07-tool-result-media"
PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfake").decode()
TOOL = FunctionTool(name="fetch_panel", description="Retrieve a visual panel by its label.",
                    parameters={"type": "object", "properties": {"label": {"type": "string", "enum": ["A", "B"]}}, "required": ["label"]})
pytestmark = pytest.mark.skipif(not RECEIPTS.exists(), reason="contract receipts not checked out")


def latest(provider: str, cell: str) -> Path:
    runs = sorted(RECEIPTS.glob(f"*/{provider}/{cell}/turn1-response.txt"))
    if not runs:
        pytest.skip(f"no {provider}/{cell} receipt")
    return runs[-1]


def turn2(lm, provider: str) -> dict:
    first = Request(model="m", messages=(Message.user("call the tool"),), tools=(TOOL,))
    raw = latest(provider, "pair").read_bytes()
    response = lm.parse_response(first, HttpResponse(200, "OK", [], raw))
    # serde round trip: what a stored transcript goes through
    response = serde.response_from_dict(json.loads(json.dumps(serde.response_to_dict(response))))
    calls = [p for p in response.message.parts if isinstance(p, ToolCallPart)]
    assert {c.input["label"] for c in calls} == {"A", "B"}
    results = tuple(tool_result(c.id, (TextPart(text="panel"), ImagePart(media_type="image/png", data=PNG)) if c.input["label"] == "B"
                                else ImagePart(media_type="image/png", data=PNG)) for c in calls)
    second = Request(model="m", messages=(*first.messages, response.message, Message.tool(results)), tools=first.tools)
    return json.loads(lm.build_request(second, stream=False).body), calls


def test_openai_responses_replay_carries_both_images_under_their_calls():
    body, calls = turn2(OpenAILM(api_key="k"), "openai")
    outputs = [i for i in body["input"] if i.get("type") == "function_call_output"]
    assert [o["call_id"] for o in outputs] == [c.id for c in calls]
    assert all(any(b["type"] == "input_image" for b in o["output"]) for o in outputs)
    # the model's own turn is replayed before the results, ids intact
    fc = [i for i in body["input"] if i.get("type") == "function_call"]
    assert [i["call_id"] for i in fc] == [c.id for c in calls]


def test_anthropic_replay_nests_images_in_tool_result_blocks():
    body, calls = turn2(AnthropicLM(api_key="k"), "anthropic")
    blocks = body["messages"][-1]["content"]
    assert [b["tool_use_id"] for b in blocks] == [c.id for c in calls]
    assert all(any(x["type"] == "image" for x in b["content"]) for b in blocks)
    assert body["messages"][-2]["role"] == "assistant"


def test_gemini_replay_resolves_names_and_keeps_thought_signatures():
    body, calls = turn2(GeminiLM(api_key="k"), "gemini")
    frs = [p["functionResponse"] for p in body["contents"][-1]["parts"]]
    assert [f["name"] for f in frs] == ["fetch_panel", "fetch_panel"]
    assert all(f["parts"][0]["inlineData"]["mimeType"] == "image/png" for f in frs)
    model_turn = body["contents"][-2]
    assert model_turn["role"] == "model" and any("functionCall" in p for p in model_turn["parts"])


def test_xai_chat_replay_uses_the_content_array():
    body, calls = turn2(XaiLM(api_key="k"), "xai")
    rows = [m for m in body["messages"] if m["role"] == "tool"]
    assert [r["tool_call_id"] for r in rows] == [c.id for c in calls]
    assert all(any(b["type"] == "image_url" for b in r["content"]) for r in rows)


def test_streamed_first_turn_then_image_result():
    """The first turn arrives as SSE (the pinned streaming_tool_call body);
    the accumulator's message is what gets replayed."""
    from lm15.vet import _parse_stream_body
    lm = OpenAILM(api_key="k")
    first = Request(model="m", messages=(Message.user("weather in Paris?"),),
                    tools=(FunctionTool(name="get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}}),))
    body = next((CONTRACT / "bodies" / "openai.streaming_tool_call").glob("*.txt")).read_bytes()
    events = _parse_stream_body(lm, first, body)
    response = materialize_response(iter(events), first)
    calls = [p for p in response.message.parts if isinstance(p, ToolCallPart)]
    assert calls
    second = Request(model="m", messages=(*first.messages, response.message,
                                           Message.tool((tool_result(calls[0].id, ImagePart(media_type="image/png", data=PNG)),))), tools=first.tools)
    wire = json.loads(lm.build_request(second, stream=False).body)
    assert wire["input"][-1] == {"type": "function_call_output", "call_id": calls[0].id,
                                 "output": [{"type": "input_image", "image_url": f"data:image/png;base64,{PNG}"}]}
