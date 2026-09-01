"""OpenAI GA Realtime live mapping.

Wire frames below are verbatim captures from 2026-09-01
(lm15-dev/curl-fixtures/live-2026-09-01/): the beta API shape is DEAD
(connecting with the `OpenAI-Beta: realtime=v1` header hard-closes the
socket, 4000 `beta_api_shape_disabled`), so every mapping here targets
the GA shape and was verified live end to end (text, audio out with
transcript, audio in via append/commit, tools, interrupt).
"""
from __future__ import annotations

import json

import pytest

from lm15 import LiveConfig
from lm15.providers import OpenAILM
from lm15.testing import FakeTransport
from lm15.types import AudioFormat, FunctionTool


def lm() -> OpenAILM:
    return OpenAILM(api_key="k", transport=FakeTransport([]))


def decode(payload: dict):
    return lm()._decode_live_server_event(json.dumps(payload))


# ─── Connection / session shape ──────────────────────────────────────

def test_no_beta_header() -> None:
    # Live 2026-09-01: the beta header closes the socket with 4000
    # beta_api_shape_disabled before any frame is exchanged.
    assert "OpenAI-Beta" not in lm()._live_headers()


def test_session_update_ga_shape() -> None:
    payload = lm()._live_session_update_payload(LiveConfig(
        model="gpt-realtime-mini", system="Be terse.", voice="alloy",
        input_format=AudioFormat(encoding="pcm16", sample_rate=24000),
        output_format=AudioFormat(encoding="pcm16", sample_rate=24000),
        tools=(FunctionTool(name="lookup", parameters={"type": "object", "properties": {}}),),
    ))
    session = payload["session"]
    assert session["type"] == "realtime"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "alloy"}
    assert session["audio"]["input"]["turn_detection"] is None  # deterministic end_audio()


def test_session_update_text_only() -> None:
    session = lm()._live_session_update_payload(LiveConfig(model="gpt-realtime-mini"))["session"]
    assert session["output_modalities"] == ["text"]
    assert "audio" not in session


# ─── Server event decoding (captured frames) ─────────────────────────

def test_text_and_transcript_deltas_map_to_text() -> None:
    assert decode({"type": "response.output_text.delta", "delta": "Live"})[0].text == "Live"
    # audio-native turns speak through the transcript event (GA name)
    assert decode({"type": "response.output_audio_transcript.delta", "delta": "hello"})[0].text == "hello"


def test_audio_delta() -> None:
    (event,) = decode({"type": "response.output_audio.delta", "delta": "aGk="})
    assert event.type == "audio" and event.data == "aGk="


def test_tool_call_fires_exactly_once() -> None:
    # Live 2026-09-01: BOTH function_call_arguments.done and
    # output_item.done arrive for one call; mapping both double-sends
    # tool results. Only output_item.done emits the tool_call event.
    args_done = {"type": "response.function_call_arguments.done", "call_id": "call_1",
                 "name": "get_weather", "arguments": "{\"city\": \"Montreal\"}"}
    item_done = {"type": "response.output_item.done", "item": {
        "type": "function_call", "call_id": "call_1", "id": "item_1",
        "name": "get_weather", "arguments": "{\"city\": \"Montreal\"}", "status": "completed"}}
    assert decode(args_done) == []
    (event,) = decode(item_done)
    assert event.type == "tool_call" and event.id == "call_1"
    assert event.input == {"city": "Montreal"}


def test_tool_call_delta_carries_call_id() -> None:
    (event,) = decode({"type": "response.function_call_arguments.delta",
                       "call_id": "call_1", "delta": "{\"ci"})
    assert event.type == "tool_call_delta" and event.id == "call_1" and event.input_delta == "{\"ci"


def test_function_call_response_done_does_not_end_turn() -> None:
    # The semantic turn stays open while the model waits for tool
    # results (the continuation is a further wire response). Gemini
    # keeps the turn open here; parity keeps the shared dispatch loop.
    done = {"type": "response.done", "response": {
        "status": "completed",
        "output": [{"type": "function_call", "call_id": "call_1", "name": "f", "arguments": "{}"}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}}}
    assert decode(done) == []


def test_turn_end_reads_ga_usage_keys() -> None:
    # GA renamed the detail keys: input_token_details (no "s" on token).
    done = {"type": "response.done", "response": {
        "status": "completed", "output": [{"type": "message"}],
        "usage": {"total_tokens": 18, "input_tokens": 13, "output_tokens": 5,
                  "input_token_details": {"text_tokens": 13, "audio_tokens": 2, "cached_tokens": 1},
                  "output_token_details": {"text_tokens": 5, "audio_tokens": 3}}}}
    (event,) = decode(done)
    assert event.type == "turn_end"
    assert event.usage.input_tokens == 13 and event.usage.cache_read_tokens == 1
    assert event.usage.input_audio_tokens == 2 and event.usage.output_audio_tokens == 3


def test_cancelled_response_maps_to_interrupted() -> None:
    done = {"type": "response.done", "response": {
        "status": "cancelled", "status_details": {"type": "cancelled", "reason": "client_cancelled"},
        "output": [], "usage": {"input_tokens": 40, "output_tokens": 2, "total_tokens": 42}}}
    (event,) = decode(done)
    assert event.type == "interrupted"


def test_cancel_race_error_is_benign() -> None:
    # Captured live 2026-09-01: pressing interrupt twice (or after the
    # response finished) yields this error; surfacing it breaks the
    # Gemini parallel where repeated interrupts are tolerated.
    error = {"type": "error", "error": {"type": "invalid_request_error",
             "code": "response_cancel_not_active",
             "message": "Cancellation failed: no active response found"}}
    assert decode(error) == []


def test_real_errors_still_surface() -> None:
    error = {"type": "error", "error": {"type": "invalid_request_error",
             "code": "unknown_parameter", "message": "bad session config"}}
    (event,) = decode(error)
    assert event.type == "error" and event.error.provider_code == "unknown_parameter"


# ─── Client event encoding ───────────────────────────────────────────

def test_end_audio_commits_and_creates_response() -> None:
    from lm15.types import LiveClientEndAudioEvent

    frames = lm()._encode_live_client_event(LiveClientEndAudioEvent())
    assert [f["type"] for f in frames] == ["input_audio_buffer.commit", "response.create"]


def test_interrupt_sends_cancel() -> None:
    from lm15.types import LiveClientInterruptEvent

    frames = lm()._encode_live_client_event(LiveClientInterruptEvent())
    assert frames == [{"type": "response.cancel"}]
