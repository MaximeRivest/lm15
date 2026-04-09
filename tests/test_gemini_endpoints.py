from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.errors import InvalidRequestError
from lm15.providers.gemini import GeminiAdapter
from lm15.sse import SSEEvent
from lm15.transports.base import HttpRequest, HttpResponse
from lm15.types import AudioGenerationRequest, BatchRequest, Config, EmbeddingRequest, FileUploadRequest, ImageGenerationRequest, LMRequest, Message


class RouterTransport:
    def request(self, req: HttpRequest) -> HttpResponse:
        url = req.url
        if url.endswith(":embedContent"):
            return HttpResponse(200, {}, json.dumps({"embedding": {"values": [0.1, 0.2, 0.3]}}).encode())
        if url.endswith(":batchEmbedContents"):
            return HttpResponse(200, {}, json.dumps({"embeddings": [{"values": [0.1, 0.2]}, {"values": [0.3, 0.4]}]}).encode())
        if "/upload/v1beta/files" in url:
            return HttpResponse(200, {}, json.dumps({"file": {"name": "files/abc"}}).encode())
        if url.endswith(":generateContent"):
            body = req.json_body or {}
            gen_cfg = body.get("generationConfig", {})
            modalities = gen_cfg.get("responseModalities", [])
            if "IMAGE" in modalities:
                payload = {
                    "responseId": "img1",
                    "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]}}],
                    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
                }
                return HttpResponse(200, {}, json.dumps(payload).encode())
            if "AUDIO" in modalities:
                payload = {
                    "responseId": "aud1",
                    "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}}]}}],
                    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 3, "totalTokenCount": 5},
                }
                return HttpResponse(200, {}, json.dumps(payload).encode())
            payload = {
                "responseId": "r1",
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
            }
            return HttpResponse(200, {}, json.dumps(payload).encode())
        return HttpResponse(404, {}, b"{}")

    def stream(self, req: HttpRequest):
        yield b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\n'
        yield b"\n"


class GeminiEndpointsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = GeminiAdapter(api_key="k", transport=RouterTransport())

    def test_embeddings_single(self):
        resp = self.adapter.embeddings(EmbeddingRequest(model="gemini-embedding-001", inputs=("hello",)))
        self.assertEqual(len(resp.vectors), 1)
        self.assertEqual(len(resp.vectors[0]), 3)

    def test_embeddings_batch(self):
        resp = self.adapter.embeddings(EmbeddingRequest(model="gemini-embedding-001", inputs=("a", "b")))
        self.assertEqual(len(resp.vectors), 2)

    def test_file_upload(self):
        resp = self.adapter.file_upload(FileUploadRequest(filename="x.txt", bytes_data=b"hello", media_type="text/plain"))
        self.assertEqual(resp.id, "files/abc")

    def test_batch_submit(self):
        req = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("hi"),), config=Config())
        batch = self.adapter.batch_submit(BatchRequest(model="gemini-2.0-flash-lite", requests=(req, req)))
        self.assertEqual(batch.status, "completed")
        self.assertEqual(len(batch.provider["results"]), 2)

    def test_image_generate(self):
        out = self.adapter.image_generate(ImageGenerationRequest(model="gemini-2.0-flash-lite", prompt="draw a cat"))
        self.assertEqual(len(out.images), 1)
        self.assertEqual(out.images[0].media_type, "image/png")

    def test_audio_generate(self):
        out = self.adapter.audio_generate(AudioGenerationRequest(model="gemini-2.0-flash-lite", prompt="say hi"))
        self.assertEqual(out.audio.media_type, "audio/wav")

    def test_parse_stream_event_error_maps_to_canonical(self):
        req = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("hi"),), config=Config())
        ev = self.adapter.parse_stream_event(
            req,
            SSEEvent(event=None, data='{"error":{"status":"RESOURCE_EXHAUSTED","message":"quota hit"}}'),
        )
        self.assertIsNotNone(ev)
        self.assertEqual(ev.type, "error")
        assert ev.error is not None
        self.assertEqual(ev.error["code"], "rate_limit")
        self.assertEqual(ev.error["provider_code"], "RESOURCE_EXHAUSTED")
        self.assertEqual(ev.error["message"], "quota hit")

    def test_parse_response_prompt_feedback_blocked_raises(self):
        req = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("hi"),), config=Config())
        resp = HttpResponse(200, {}, json.dumps({"promptFeedback": {"blockReason": "SAFETY"}}).encode())
        with self.assertRaises(InvalidRequestError):
            self.adapter.parse_response(req, resp)

    def test_parse_response_finish_reason_blocked_raises(self):
        req = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("hi"),), config=Config())
        resp = HttpResponse(
            200,
            {},
            json.dumps({"candidates": [{"finishReason": "RECITATION", "finishMessage": "recitation detected"}]}).encode(),
        )
        with self.assertRaises(InvalidRequestError):
            self.adapter.parse_response(req, resp)

    def test_parse_stream_event_prompt_feedback_blocked_maps_error(self):
        req = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("hi"),), config=Config())
        ev = self.adapter.parse_stream_event(req, SSEEvent(event=None, data='{"promptFeedback":{"blockReason":"BLOCKLIST"}}'))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.type, "error")
        assert ev.error is not None
        self.assertEqual(ev.error["code"], "invalid_request")
        self.assertEqual(ev.error["provider_code"], "inband_finish_reason")


if __name__ == "__main__":
    unittest.main()
