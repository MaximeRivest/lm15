from __future__ import annotations

import json
from pathlib import Path

from lm15.providers.openai import OpenAIAdapter
from lm15.transports.base import HttpRequest, HttpResponse
from lm15.types import AudioGenerationRequest, BatchRequest, Config, EmbeddingRequest, FileUploadRequest, ImageGenerationRequest, LMRequest, Message

from ._helpers import ProbeResult


class _RouterTransport:
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
            return HttpResponse(200, {}, json.dumps({"id": "resp_1", "model": "gpt-4.1-mini", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}], "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}}).encode())
        return HttpResponse(404, {}, b"{}")

    def stream(self, req: HttpRequest):
        yield b'data: {"type":"response.created","response":{"id":"resp_1"}}\n'
        yield b"\n"
        yield b'data: {"type":"response.output_text.delta","delta":"ok"}\n'
        yield b"\n"
        yield b"data: [DONE]\n"
        yield b"\n"


def run(test: dict, root: Path) -> ProbeResult:
    a = OpenAIAdapter(api_key="k", transport=_RouterTransport())

    emb = a.embeddings(EmbeddingRequest(model="text-embedding-3-small", inputs=("x",)))
    f = a.file_upload(FileUploadRequest(filename="x.txt", bytes_data=b"x", media_type="text/plain"))

    req = LMRequest(model="gpt-4.1-mini", messages=(Message.user("hi"),), config=Config())
    b = a.batch_submit(BatchRequest(model="gpt-4.1-mini", requests=(req,), provider={"input_file_id": "file_abc"}))
    img = a.image_generate(ImageGenerationRequest(model="gpt-image-1", prompt="draw"))
    aud = a.audio_generate(AudioGenerationRequest(model="gpt-4o-mini-tts", prompt="say"))

    ok = (
        len(emb.vectors) == 1
        and f.id == "file_abc"
        and b.id == "batch_abc"
        and len(img.images) == 1
        and aud.audio.media_type.startswith("audio/")
    )

    if ok:
        return ProbeResult(status="pass", details="openai extended endpoints normalized")
    return ProbeResult(status="fail", details="openai extended endpoint mapping failed")
