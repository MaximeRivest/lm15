from __future__ import annotations

import json
from pathlib import Path

from lm15.providers.gemini import GeminiAdapter
from lm15.transports.base import HttpRequest, HttpResponse
from lm15.types import AudioGenerationRequest, BatchRequest, Config, EmbeddingRequest, FileUploadRequest, ImageGenerationRequest, LMRequest, Message

from ._helpers import ProbeResult


class _RouterTransport:
    def request(self, req: HttpRequest) -> HttpResponse:
        url = req.url
        if url.endswith(":embedContent"):
            return HttpResponse(200, {}, json.dumps({"embedding": {"values": [0.1, 0.2]}}).encode())
        if url.endswith(":batchEmbedContents"):
            return HttpResponse(200, {}, json.dumps({"embeddings": [{"values": [0.1]}, {"values": [0.2]}]}).encode())
        if "/upload/v1beta/files" in url:
            return HttpResponse(200, {}, json.dumps({"file": {"name": "files/abc"}}).encode())
        if url.endswith(":generateContent"):
            body = req.json_body or {}
            cfg = body.get("generationConfig", {})
            modalities = cfg.get("responseModalities", [])
            if "IMAGE" in modalities:
                payload = {
                    "responseId": "img1",
                    "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": "iVBORw0KGgo="}}]}}],
                }
                return HttpResponse(200, {}, json.dumps(payload).encode())
            if "AUDIO" in modalities:
                payload = {
                    "responseId": "aud1",
                    "candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "audio/wav", "data": "UklGRg=="}}]}}],
                }
                return HttpResponse(200, {}, json.dumps(payload).encode())
            payload = {"responseId": "r1", "candidates": [{"content": {"parts": [{"text": "ok"}]}}]}
            return HttpResponse(200, {}, json.dumps(payload).encode())
        return HttpResponse(404, {}, b"{}")

    def stream(self, req: HttpRequest):
        yield b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\n'
        yield b"\n"


def run(test: dict, root: Path) -> ProbeResult:
    a = GeminiAdapter(api_key="k", transport=_RouterTransport())

    emb_single = a.embeddings(EmbeddingRequest(model="gemini-embedding-001", inputs=("x",)))
    emb_batch = a.embeddings(EmbeddingRequest(model="gemini-embedding-001", inputs=("a", "b")))
    fu = a.file_upload(FileUploadRequest(filename="x.txt", bytes_data=b"x", media_type="text/plain"))

    req = LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("hi"),), config=Config())
    batch = a.batch_submit(BatchRequest(model="gemini-2.0-flash-lite", requests=(req, req)))
    img = a.image_generate(ImageGenerationRequest(model="gemini-2.0-flash-lite", prompt="draw"))
    aud = a.audio_generate(AudioGenerationRequest(model="gemini-2.0-flash-lite", prompt="say"))

    ok = (
        len(emb_single.vectors) == 1
        and len(emb_batch.vectors) == 2
        and fu.id == "files/abc"
        and batch.status == "completed"
        and len(img.images) >= 1
        and aud.audio.media_type.startswith("audio/")
    )
    if ok:
        return ProbeResult(status="pass", details="gemini extended endpoints normalized")
    return ProbeResult(status="fail", details="gemini extended endpoint mapping failed")
