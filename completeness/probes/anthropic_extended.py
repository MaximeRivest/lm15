from __future__ import annotations

import json
from pathlib import Path

from lm15.providers.anthropic import AnthropicAdapter
from lm15.transports.base import HttpRequest, HttpResponse
from lm15.types import BatchRequest, Config, FileUploadRequest, LMRequest, Message

from ._helpers import ProbeResult


class _RouterTransport:
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


def run(test: dict, root: Path) -> ProbeResult:
    a = AnthropicAdapter(api_key="k", transport=_RouterTransport())

    f = a.file_upload(FileUploadRequest(filename="x.txt", bytes_data=b"x", media_type="text/plain"))

    req = LMRequest(model="claude-sonnet-4-5", messages=(Message.user("hi"),), config=Config())
    b = a.batch_submit(BatchRequest(model="claude-sonnet-4-5", requests=(req, req)))

    if f.id == "file_abc" and b.id == "batch_abc":
        return ProbeResult(status="pass", details="anthropic files+batches normalized")
    return ProbeResult(status="fail", details="anthropic files/batches mapping failed")
