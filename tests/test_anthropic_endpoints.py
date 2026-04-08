from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.providers.anthropic import AnthropicAdapter
from lm15.transports.base import HttpRequest, HttpResponse
from lm15.types import BatchRequest, Config, FileUploadRequest, LMRequest, Message


class RouterTransport:
    def request(self, req: HttpRequest) -> HttpResponse:
        if req.url.endswith("/files"):
            return HttpResponse(200, {}, json.dumps({"id": "file_abc"}).encode())
        if req.url.endswith("/messages/batches"):
            return HttpResponse(200, {}, json.dumps({"id": "batch_abc", "processing_status": "in_progress"}).encode())
        if req.url.endswith("/messages"):
            return HttpResponse(
                200,
                {},
                json.dumps(
                    {
                        "id": "msg_1",
                        "model": "claude-sonnet-4-5",
                        "content": [{"type": "text", "text": "ok"}],
                        "usage": {"input_tokens": 2, "output_tokens": 1},
                    }
                ).encode(),
            )
        return HttpResponse(404, {}, b"{}")

    def stream(self, req: HttpRequest):
        yield b'data: {"type":"message_start","message":{"id":"m1","model":"claude"}}\n'
        yield b"\n"
        yield b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n'
        yield b"\n"
        yield b'data: {"type":"message_stop"}\n'
        yield b"\n"


class AnthropicEndpointsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = AnthropicAdapter(api_key="k", transport=RouterTransport())

    def test_file_upload(self):
        out = self.adapter.file_upload(FileUploadRequest(filename="x.txt", bytes_data=b"hello", media_type="text/plain"))
        self.assertEqual(out.id, "file_abc")

    def test_batch_submit(self):
        req = LMRequest(model="claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config())
        out = self.adapter.batch_submit(BatchRequest(model="claude-sonnet-4-5", requests=(req, req)))
        self.assertEqual(out.id, "batch_abc")
        self.assertEqual(out.status, "in_progress")


if __name__ == "__main__":
    unittest.main()
