from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lm15.api as api
from lm15.client import UniversalLM
from lm15.model import Model
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.types import Config, LMRequest, Message, PartDelta, Usage


class _DummyTransport:
    def request(self, req):  # pragma: no cover
        raise AssertionError("not used")

    def stream(self, req):  # pragma: no cover
        yield from ()


class _FakeWS:
    def __init__(self, recv_payloads: list[dict]):
        self._recv = [json.dumps(x) for x in recv_payloads]
        self.sent: list[str] = []
        self.closed = False

    def send(self, payload: str) -> None:
        self.sent.append(payload)

    def recv(self) -> str:
        if not self._recv:
            raise RuntimeError("no more websocket frames")
        return self._recv.pop(0)

    def close(self) -> None:
        self.closed = True


class _OpenAICompletionAdapter(OpenAIAdapter):
    def __init__(self, sessions: list[_FakeWS]):
        super().__init__(api_key="k", transport=_DummyTransport())
        self._sessions = sessions

    def _live_connect(self, url: str, headers: dict[str, str]):
        return self._sessions.pop(0)


class _GeminiCompletionAdapter(GeminiAdapter):
    def __init__(self, sessions: list[_FakeWS]):
        super().__init__(api_key="k", transport=_DummyTransport())
        self._sessions = sessions

    def _live_connect(self, url: str):
        return self._sessions.pop(0)


class LiveCompletionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old_build_default = api.build_default
        api._client_cache.clear()

    def tearDown(self) -> None:
        api.build_default = self._old_build_default
        api._client_cache.clear()
        api.configure()

    def test_openai_live_model_streams_via_websocket_completion(self):
        ws = _FakeWS(
            [
                {"type": "response.output_text.delta", "delta": "ok"},
                {"type": "response.done", "response": {"usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}}},
            ]
        )
        adapter = _OpenAICompletionAdapter([ws])

        req = LMRequest(model="gpt-4o-realtime-preview", messages=(Message.user("hi"),), config=Config())
        events = list(adapter.stream(req))

        self.assertEqual(events[0].type, "start")
        self.assertEqual(events[1].type, "delta")
        self.assertEqual(events[1].delta.type, "text")
        self.assertEqual(events[1].delta.text, "ok")
        self.assertEqual(events[-1].type, "end")
        self.assertEqual(events[-1].finish_reason, "stop")

    def test_gemini_live_model_streams_via_websocket_completion(self):
        ws = _FakeWS(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "ok"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 1, "responseTokenCount": 1, "totalTokenCount": 2},
                }
            ]
        )
        adapter = _GeminiCompletionAdapter([ws])

        req = LMRequest(model="gemini-2.0-flash-live", messages=(Message.user("hi"),), config=Config())
        events = list(adapter.stream(req))

        self.assertEqual(events[0].type, "start")
        self.assertEqual(events[1].type, "delta")
        self.assertEqual(events[1].delta.type, "text")
        self.assertEqual(events[1].delta.text, "ok")
        self.assertEqual(events[-1].type, "end")
        self.assertEqual(events[-1].finish_reason, "stop")

    def test_openai_live_model_supports_tool_loop_in_model_call(self):
        ws_round_1 = _FakeWS(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "get_weather",
                        "arguments": '{"city":"Montreal"}',
                    },
                },
                {"type": "response.done", "response": {"usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}},
            ]
        )
        ws_round_2 = _FakeWS(
            [
                {"type": "response.output_text.delta", "delta": "Tool says: 22C in Montreal"},
                {"type": "response.done", "response": {"usage": {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4}}},
            ]
        )
        lm = UniversalLM()
        lm.register(_OpenAICompletionAdapter([ws_round_1, ws_round_2]))

        def get_weather(city: str) -> str:
            return f"22C in {city}"

        agent = Model(lm=lm, model="gpt-4o-realtime-preview", provider="openai")
        resp = agent.call("weather", tools=[get_weather])
        self.assertIn("22C", resp.text or "")

    def test_module_call_uses_live_completion_transport(self):
        ws = _FakeWS(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "ok"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 1, "responseTokenCount": 1, "totalTokenCount": 2},
                }
            ]
        )
        lm = UniversalLM()
        lm.register(_GeminiCompletionAdapter([ws]))
        api.build_default = lambda **_kw: lm

        resp = api.call("gemini-2.0-flash-live", "hi", provider="gemini")
        self.assertEqual(resp.text, "ok")

    def test_gemini_live_model_supports_tool_loop_in_model_call(self):
        ws_round_1 = _FakeWS(
            [
                {"setupComplete": {}},
                {
                    "toolCall": {
                        "functionCalls": [
                            {
                                "id": "call_1",
                                "name": "get_weather",
                                "args": {"city": "Montreal"},
                            }
                        ]
                    },
                    "serverContent": {
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 2, "responseTokenCount": 1, "totalTokenCount": 3},
                }
            ]
        )
        ws_round_2 = _FakeWS(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "Tool says: 22C in Montreal"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 2, "responseTokenCount": 2, "totalTokenCount": 4},
                }
            ]
        )
        lm = UniversalLM()
        lm.register(_GeminiCompletionAdapter([ws_round_1, ws_round_2]))

        def get_weather(city: str) -> str:
            return f"22C in {city}"

        agent = Model(lm=lm, model="gemini-2.0-flash-live", provider="gemini")
        resp = agent.call("weather", tools=[get_weather])
        self.assertIn("22C", resp.text or "")

    def test_gemini_live_completion_waits_for_setup_before_sending_prompt(self):
        ws = _FakeWS(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {"parts": [{"text": "ok"}]},
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 1, "responseTokenCount": 1, "totalTokenCount": 2},
                },
            ]
        )
        adapter = _GeminiCompletionAdapter([ws])

        req = LMRequest(model="gemini-2.0-flash-live", messages=(Message.user("hi"),), config=Config())
        events = list(adapter.stream(req))

        self.assertEqual(events[-1].type, "end")
        self.assertEqual(set(json.loads(ws.sent[0]).keys()), {"setup"})
        self.assertIn("realtimeInput", json.loads(ws.sent[1]))

    def test_gemini_audio_native_model_uses_audio_modality_and_realtime_input(self):
        """Audio-native live models (live-preview) must use AUDIO modality,
        outputAudioTranscription, and send all input via realtimeInput."""
        from lm15.types import Part as P
        import base64

        # Build a tiny valid WAV: 44-byte header + 4 bytes PCM
        import struct
        pcm = b"\x00\x01\x02\x03"
        wav = (
            b"RIFF"
            + struct.pack("<I", 36 + len(pcm))
            + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16)
            + b"data"
            + struct.pack("<I", len(pcm))
            + pcm
        )

        ws = _FakeWS(
            [
                {"setupComplete": {}},
                # transcription text from outputTranscription
                {
                    "serverContent": {
                        "modelTurn": {
                            "parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AAA="}}]
                        },
                        "outputTranscription": {"text": "Hello world"},
                    },
                },
                {
                    "serverContent": {"turnComplete": True},
                    "usageMetadata": {"promptTokenCount": 10, "responseTokenCount": 2, "totalTokenCount": 12},
                },
            ]
        )
        adapter = _GeminiCompletionAdapter([ws])

        audio_part = P.audio(data=wav, media_type="audio/wav")
        msg = Message(role="user", parts=(audio_part, P.text_part("Transcribe this.")))
        req = LMRequest(model="gemini-3.1-flash-live-preview", messages=(msg,), config=Config())
        events = list(adapter.stream(req))

        # Check setup uses AUDIO modality + outputAudioTranscription
        setup = json.loads(ws.sent[0])["setup"]
        self.assertEqual(setup["generationConfig"]["responseModalities"], ["AUDIO"])
        self.assertIn("outputAudioTranscription", setup)

        # Content sent as realtimeInput, not clientContent
        content_msgs = [json.loads(m) for m in ws.sent[1:]]
        for m in content_msgs:
            self.assertNotIn("clientContent", m)

        # Audio is sent as realtimeInput.audio with PCM mime
        audio_msgs = [m for m in content_msgs if "realtimeInput" in m and "audio" in m.get("realtimeInput", {})]
        self.assertTrue(len(audio_msgs) >= 1)
        self.assertIn("audio/pcm;rate=16000", audio_msgs[0]["realtimeInput"]["audio"]["mimeType"])

        # Text is sent as realtimeInput.text
        text_msgs = [m for m in content_msgs if "realtimeInput" in m and "text" in m.get("realtimeInput", {})]
        self.assertTrue(len(text_msgs) >= 1)

        # outputTranscription appears as a text delta
        text_deltas = [e for e in events if e.type == "delta" and isinstance(e.delta, PartDelta) and e.delta.type == "text"]
        self.assertTrue(len(text_deltas) >= 1)
        self.assertEqual(text_deltas[0].delta.text, "Hello world")

    def test_gemini_audio_native_output_audio_skips_transcription_config(self):
        """When output='audio', no outputAudioTranscription should be added."""
        ws = _FakeWS(
            [
                {"setupComplete": {}},
                {
                    "serverContent": {
                        "modelTurn": {
                            "parts": [{"inlineData": {"mimeType": "audio/pcm;rate=24000", "data": "AAA="}}]
                        },
                        "turnComplete": True,
                    },
                    "usageMetadata": {"promptTokenCount": 1, "responseTokenCount": 1, "totalTokenCount": 2},
                },
            ]
        )
        adapter = _GeminiCompletionAdapter([ws])

        req = LMRequest(
            model="gemini-3.1-flash-live-preview",
            messages=(Message.user("Say hello"),),
            config=Config(provider={"output": "audio"}),
        )
        events = list(adapter.stream(req))

        setup = json.loads(ws.sent[0])["setup"]
        self.assertEqual(setup["generationConfig"]["responseModalities"], ["AUDIO"])
        self.assertNotIn("outputAudioTranscription", setup)


if __name__ == "__main__":
    unittest.main()
