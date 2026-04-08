from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.providers.openai import OpenAIAdapter
from lm15.transports.base import HttpRequest, HttpResponse
from lm15.types import AudioGenerationRequest, BatchRequest, Config, EmbeddingRequest, FileUploadRequest, ImageGenerationRequest, LMRequest, Message


class RouterTransport:
    def request(self, req: HttpRequest) -> HttpResponse:
        if req.url.endswith("/embeddings"):
            return HttpResponse(200, {}, json.dumps({"model": "text-embedding-3-small", "data": [{"embedding": [0.1, 0.2]}], "usage": {"prompt_tokens": 2, "total_tokens": 2}}).encode())
        if req.url.endswith("/files"):
            return HttpResponse(200, {}, json.dumps({"id": "file_abc"}).encode())
        if req.url.endswith("/batches"):
            return HttpResponse(200, {}, json.dumps({"id": "batch_abc", "status": "in_progress"}).encode())
        if req.url.endswith("/images/generations"):
            return HttpResponse(200, {}, json.dumps({"data": [{"b64_json": "iVBORw0KGgo="}]}).encode())
        if req.url.endswith("/audio/speech"):
            return HttpResponse(200, {"content-type": "audio/wav"}, b"RIFF....")
        if req.url.endswith("/responses"):
            return HttpResponse(
                200,
                {},
                json.dumps({"id": "resp_1", "model": "gpt-4.1-mini", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}], "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}).encode(),
            )
        return HttpResponse(404, {}, b"{}")

    def stream(self, req: HttpRequest):
        yield b'data: {"type":"response.created","response":{"id":"resp_1"}}\n'
        yield b"\n"
        yield b'data: {"type":"response.output_text.delta","delta":"ok"}\n'
        yield b"\n"
        yield b"data: [DONE]\n"
        yield b"\n"


class OpenAIEndpointsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="k", transport=RouterTransport())

    def test_embeddings(self):
        out = self.adapter.embeddings(EmbeddingRequest(model="text-embedding-3-small", inputs=("hi",)))
        self.assertEqual(len(out.vectors), 1)
        self.assertEqual(len(out.vectors[0]), 2)

    def test_file_upload(self):
        out = self.adapter.file_upload(FileUploadRequest(filename="x.txt", bytes_data=b"hello", media_type="text/plain"))
        self.assertEqual(out.id, "file_abc")

    def test_batch_submit_native(self):
        req = LMRequest(model="gpt-4.1-mini", messages=(Message.user("hi"),), config=Config())
        out = self.adapter.batch_submit(BatchRequest(model="gpt-4.1-mini", requests=(req,), provider={"input_file_id": "file_abc"}))
        self.assertEqual(out.id, "batch_abc")
        self.assertEqual(out.status, "in_progress")

    def test_image_generate(self):
        out = self.adapter.image_generate(ImageGenerationRequest(model="gpt-image-1", prompt="draw a cat"))
        self.assertEqual(len(out.images), 1)
        self.assertEqual(out.images[0].media_type, "image/png")

    def test_audio_generate(self):
        out = self.adapter.audio_generate(AudioGenerationRequest(model="gpt-4o-mini-tts", prompt="say hi"))
        self.assertEqual(out.audio.media_type, "audio/wav")
        self.assertTrue(out.audio.data)


if __name__ == "__main__":
    unittest.main()
