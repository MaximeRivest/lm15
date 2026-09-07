"""MAP-7.8–9: a Responses stream keeps thinking and final replay state."""

import json

import pytest

from lm15 import Message, OpenAILM, Request, ThinkingPart
from lm15.result import StreamAccumulator
from lm15.sse import SSEEvent
from lm15.testing import FakeTransport


def parse(lm, request, payload):
    return list(lm.parse_stream_events(request, SSEEvent(event=payload["type"], data=json.dumps(payload))))


@pytest.mark.parametrize("summary", ["", "Some summary."])
@pytest.mark.parametrize("encrypted", [None, "final-opaque-payload"])
def test_stream_reasoning_retains_final_state_without_repeating_summary(summary, encrypted):
    lm = OpenAILM(api_key="test", transport=FakeTransport([]))
    request = Request(model="gpt-5.4-mini", messages=[Message.user("hello")])
    acc = StreamAccumulator(request)
    item = {"type": "reasoning", "id": "rs_test", "summary": []}
    events = parse(lm, request, {"type": "response.output_item.added", "output_index": 2, "item": item})
    assert len(events) == 1
    assert events[0].delta.type == "thinking"
    assert events[0].delta.text == ""
    assert events[0].delta.part_index == 2
    if summary:
        events += parse(lm, request, {"type": "response.reasoning_summary_text.delta", "output_index": 2, "delta": summary})
    final_item = {**item, "summary": [{"type": "summary_text", "text": summary}] if summary else []}
    if encrypted:
        final_item["encrypted_content"] = encrypted
    final_events = parse(lm, request, {"type": "response.output_item.done", "output_index": 2, "item": final_item})
    assert len(final_events) == 1
    assert final_events[0].delta.type == "continuation"
    assert final_events[0].delta.part_index == 2
    for event in events + final_events:
        acc.push(event)
    response = acc.response()
    assert len(response.message.parts) == 1
    part = response.message.parts[0]
    assert isinstance(part, ThinkingPart)
    assert part.text == summary
    state = {"id": "rs_test", **({"encrypted_content": encrypted} if encrypted else {})}
    assert len(part.continuation) == 1
    assert part.continuation[0].provider == "openai"
    assert part.continuation[0].kind == "reasoning_item"
    assert part.continuation[0].data == state
    replay = Request(model=request.model, messages=[Message.user("hello"), response.message, Message.user("continue")])
    wire = json.loads(lm.build_request(replay, stream=False).body)
    assert wire["input"][1] == {"type": "reasoning", **state,
                                 "summary": [{"type": "summary_text", "text": summary}] if summary else []}


def test_unfinished_reasoning_item_still_has_an_empty_thinking_part():
    lm = OpenAILM(api_key="test", transport=FakeTransport([]))
    request = Request(model="test", messages=[Message.user("hello")])
    acc = StreamAccumulator(request)
    for event in parse(lm, request, {"type": "response.output_item.added", "output_index": 0,
                                    "item": {"type": "reasoning", "id": "rs_test", "summary": []}}):
        acc.push(event)
    assert acc.response().message.parts == (ThinkingPart(text=""),)


def test_missing_replay_fields_do_not_create_empty_continuation():
    lm = OpenAILM(api_key="test", transport=FakeTransport([]))
    request = Request(model="test", messages=[Message.user("hello")])
    assert parse(lm, request, {"type": "response.output_item.done", "output_index": 0,
                              "item": {"type": "reasoning", "summary": []}}) == []
