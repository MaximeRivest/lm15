from __future__ import annotations

import asyncio
import base64

import pytest

from lm15.result import (
    AsyncResponseStream,
    ResponseStream,
    StreamAccumulator,
    amaterialize_response,
    materialize_response,
    response_to_events,
)
from lm15.types import (
    AudioPart,
    ContinuationDelta,
    ContinuationState,
    DocumentPart,
    FunctionTool,
    ImageDelta,
    ImagePart,
    Message,
    RefusalPart,
    Request,
    Response,
    StreamDeltaEvent,
    StreamEndEvent,
    TextDelta,
    TextPart,
    ToolCallDelta,
    Usage,
    VideoPart,
)

_REQ = Request(model="m", messages=(Message.user("hi"),))


def _b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def _response_with(part) -> Response:
    return Response(
        id="r1",
        model="m",
        message=Message.assistant(part),
        finish_reason="stop",
        usage=Usage(),
    )


def test_response_to_events_preserves_image_file_ids() -> None:
    image = ImagePart(file_id="file_123")
    events = list(response_to_events(_response_with(image)))

    assert events[0].type == "start"
    assert isinstance(events[1].delta, ImageDelta)
    assert events[1].delta.file_id == "file_123"
    assert events[-1].type == "end"


def test_materialize_preserves_continuation_only_part_index() -> None:
    response = materialize_response(
        iter((
            StreamDeltaEvent(
                delta=ContinuationDelta(
                    provider="anthropic",
                    kind="redacted_thinking",
                    data={"data": "opaque"},
                    part_index=0,
                )
            ),
            StreamEndEvent(finish_reason="stop"),
        )),
        _REQ,
    )

    assert response.message.parts == (
        TextPart(
            "",
            continuation=(
                ContinuationState(provider="anthropic", kind="redacted_thinking", data={"data": "opaque"}),
            ),
        ),
    )


def test_response_to_events_and_materialize_preserve_continuation_state() -> None:
    response = Response(
        id="r1",
        model="m",
        message=Message(
            role="assistant",
            parts=(
                TextPart(
                    "hello",
                    continuation=(
                        ContinuationState(provider="openai", kind="response_item_id", data={"id": "item_1"}),
                    ),
                ),
            ),
            continuation=(
                ContinuationState(provider="openai", kind="response_id", data={"id": "resp_1"}),
            ),
        ),
        finish_reason="stop",
        usage=Usage(),
    )

    events = list(response_to_events(response))
    assert any(
        event.type == "delta" and isinstance(event.delta, ContinuationDelta) and event.delta.part_index is None
        for event in events
    )
    rebuilt = ResponseStream(iter(events), _REQ).response
    assert rebuilt == response


@pytest.mark.parametrize(
    "part",
    [
        # AudioPart by reference is non-streamable: AudioDelta requires inline data.
        AudioPart(url="https://example.com/audio.wav"),
        VideoPart(data=_b64(b"video")),
        DocumentPart(data=_b64(b"pdf")),
        RefusalPart("no"),
    ],
)
def test_response_to_events_raises_for_parts_without_delta_variants(part) -> None:
    with pytest.raises(TypeError, match="Cannot convert"):
        list(response_to_events(_response_with(part)))


def _text_stream_events() -> tuple:
    return (
        StreamDeltaEvent(delta=TextDelta(text="Hel", part_index=0)),
        StreamDeltaEvent(delta=TextDelta(text="lo", part_index=0)),
        StreamEndEvent(finish_reason="stop", usage=Usage(input_tokens=1, output_tokens=2)),
    )


def test_response_stream_yields_text_then_response() -> None:
    rs = ResponseStream(iter(_text_stream_events()), _REQ)
    assert list(rs) == ["Hel", "lo"]
    assert rs.response.text == "Hello"
    assert rs.usage.input_tokens == 1
    assert rs.finish_reason == "stop"


def test_response_stream_events_are_canonical() -> None:
    rs = ResponseStream(iter(_text_stream_events()), _REQ)
    events = list(rs.events())
    assert [e.type for e in events] == ["delta", "delta", "end"]
    assert events[0].delta.type == "text"
    assert rs.response.text == "Hello"


def test_response_stream_positional_construction() -> None:
    # The taught form: ResponseStream(lm.stream(req), req) — both positional.
    rs = ResponseStream(iter(_text_stream_events()), _REQ)
    assert rs.response.model == "m"


def test_response_stream_accessors_mirror_response_minimal_set() -> None:
    rs = ResponseStream(iter(_text_stream_events()), _REQ)
    rs.response  # consume
    assert rs.text == "Hello"
    assert rs.tool_calls == []
    assert rs.citations == []
    assert rs.model == "m"
    # Richer accessors deliberately do not exist; message.first(...) is THE
    # variant accessor on both the streaming and non-streaming paths.
    for gone in ("image", "images", "audio", "video", "videos",
                 "document", "documents", "image_bytes", "audio_bytes",
                 "video_bytes", "document_bytes", "thinking"):
        assert not hasattr(rs, gone)


def test_stream_accumulator_push_then_response() -> None:
    acc = StreamAccumulator(_REQ)
    for event in _text_stream_events():
        acc.push(event)
    response = acc.response()
    assert response.text == "Hello"
    assert response.usage.output_tokens == 2


def test_async_response_stream_mirrors_sync() -> None:
    async def source():
        for event in _text_stream_events():
            yield event

    async def main() -> None:
        rs = AsyncResponseStream(source(), _REQ)
        texts = [t async for t in rs]
        assert texts == ["Hel", "lo"]
        response = await rs.response()
        assert response.text == "Hello"

    asyncio.run(main())


def test_amaterialize_response() -> None:
    async def source():
        for event in _text_stream_events():
            yield event

    response = asyncio.run(amaterialize_response(source(), _REQ))
    assert response.text == "Hello"


def test_response_stream_rejects_tool_loop_parameters() -> None:
    """ResponseStream is a pure stream assembler: the automatic
    tool-execution loop was removed (positioning decision, 2026-06-11),
    and its constructor hooks went with it (API review, 2026-07-13)."""
    for kwarg in ("callable_registry", "on_tool_call", "max_tool_rounds",
                  "retries", "start_stream", "on_finished"):
        with pytest.raises(TypeError):
            ResponseStream(iter(()), _REQ, **{kwarg: None})


def test_result_module_has_no_tool_execution_helpers() -> None:
    import lm15.result as result_mod

    for name in ("_invoke_tool", "_normalize_tool_output", "_preview_parts",
                 "_ExecutedTool", "Result", "AsyncResult", "StreamChunk"):
        assert not hasattr(result_mod, name)


# ─── MAP-9: a missing tool-call name is filled from the request ──────


def _assemble(tools, events):
    request = Request(model="m", messages=(Message.user("hi"),), tools=tuple(tools))
    return materialize_response(iter(events), request)


def _call(idx: int, name: str | None = None) -> StreamDeltaEvent:
    return StreamDeltaEvent(delta=ToolCallDelta(input='{"q": 1}', part_index=idx, name=name))


def test_map9_single_declared_tool_names_the_call() -> None:
    resp = _assemble([FunctionTool(name="lookup")], [_call(0), StreamEndEvent(finish_reason="tool_call")])
    (call,) = resp.tool_calls
    assert call.name == "lookup"
    assert call.id == "tool_call_0"


def test_map9_position_among_all_parts_picks_the_tool() -> None:
    # The text part at index 0 occupies position 0, so the unnamed call at
    # index 1 takes the tool at position 1 — the stated part-position rule.
    events = [
        StreamDeltaEvent(delta=TextDelta(text="thinking", part_index=0)),
        _call(1),
        StreamEndEvent(finish_reason="tool_call"),
    ]
    resp = _assemble([FunctionTool(name="alpha"), FunctionTool(name="beta")], events)
    (call,) = resp.tool_calls
    assert call.name == "beta"


def test_map9_no_candidate_falls_back_to_literal_tool() -> None:
    events = [
        StreamDeltaEvent(delta=TextDelta(text="a", part_index=0)),
        StreamDeltaEvent(delta=TextDelta(text="b", part_index=1)),
        _call(2),
        StreamEndEvent(finish_reason="tool_call"),
    ]
    resp = _assemble([FunctionTool(name="alpha"), FunctionTool(name="beta")], events)
    (call,) = resp.tool_calls
    assert call.name == "tool"


def test_map9_delivered_name_wins_over_fallback() -> None:
    events = [_call(0, name="beta"), _call(0), StreamEndEvent(finish_reason="tool_call")]
    resp = _assemble([FunctionTool(name="alpha"), FunctionTool(name="beta")], events)
    (call,) = resp.tool_calls
    assert call.name == "beta"
