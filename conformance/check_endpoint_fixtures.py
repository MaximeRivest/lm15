#!/usr/bin/env python3
"""Offline conformance checks for non-chat/generation endpoints."""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lm15.providers import AnthropicLM, GeminiLM, OpenAILM  # noqa: E402
from lm15.types import (  # noqa: E402
    AudioFormat,
    AudioGenerationRequest,
    AudioGenerationResponse,
    BatchRequest,
    BatchResponse,
    FileUploadRequest,
    FileUploadResponse,
    FunctionTool,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    LiveConfig,
    Message,
    Request,
    Response,
    TextPart,
)

REPORT_DIR = ROOT / "reports"
JsonObject = dict[str, Any]


@dataclass
class FakeResponse:
    status: int
    body: bytes
    headers: list[tuple[str, str]] | None = None
    reason: str = "OK"
    http_version: str = "HTTP/1.1"

    def __post_init__(self) -> None:
        if self.headers is None:
            self.headers = [("content-type", "application/json")]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return self.body

    def header(self, name: str) -> str | None:
        lname = name.lower()
        for key, value in self.headers or []:
            if key.lower() == lname:
                return value
        return None


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[Any] = []

    def stream(self, request: Any) -> FakeResponse:
        self.requests.append(request)
        return self.responses.pop(0)


def json_response(payload: JsonObject, *, status: int = 200) -> FakeResponse:
    return FakeResponse(status=status, body=json.dumps(payload).encode("utf-8"))


def request_body(req: Any) -> Any:
    if not req.body:
        return None
    try:
        return json.loads(req.body.decode("utf-8"))
    except Exception:
        return req.body


def assert_header(req: Any, name: str, expected: str | None = None) -> None:
    lname = name.lower()
    found = None
    for key, value in req.headers:
        if key.lower() == lname:
            found = value
            break
    assert found is not None, f"missing header {name}"
    if expected is not None:
        assert found == expected, f"{name}: expected {expected!r}, got {found!r}"


@dataclass(frozen=True)
class EndpointResult:
    case_id: str
    status: str
    reason: str | None = None


def run_case(case_id: str, fn: Callable[[], None]) -> EndpointResult:
    try:
        fn()
        return EndpointResult(case_id, "pass")
    except Exception as exc:
        return EndpointResult(case_id, "fail", str(exc))


def openai_file_upload() -> None:
    transport = FakeTransport([json_response({"id": "file_1"})])
    lm = OpenAILM(api_key="test", transport=transport)
    out = lm.file_upload(FileUploadRequest(filename="hello.txt", bytes_data=b"hello", media_type="text/plain"))
    req = transport.requests[0]
    assert req.method == "POST"
    assert req.url == "https://api.openai.com/v1/files"
    assert_header(req, "Content-Type")
    assert b"hello.txt" in req.body
    assert isinstance(out, FileUploadResponse)
    assert out.id == "file_1"


def openai_batch_submit() -> None:
    transport = FakeTransport([json_response({"id": "batch_1", "status": "validating"})])
    lm = OpenAILM(api_key="test", transport=transport)
    out = lm.batch_submit(BatchRequest(requests=(Request(model="gpt-test", messages=(Message.user("hi"),)),), extensions={"input_file_id": "file_1"}))
    req = transport.requests[0]
    assert req.url == "https://api.openai.com/v1/batches"
    assert request_body(req)["input_file_id"] == "file_1"
    assert isinstance(out, BatchResponse)
    assert out.status == "queued"


def openai_image_generate() -> None:
    transport = FakeTransport([json_response({"id": "img_1", "model": "gpt-image-1", "data": [{"b64_json": "aGk="}]})])
    lm = OpenAILM(api_key="test", transport=transport)
    out = lm.image_generate(ImageGenerationRequest(model="gpt-image-1", prompt="a cat", size="1024x1024"))
    req = transport.requests[0]
    assert req.url == "https://api.openai.com/v1/images/generations"
    assert request_body(req)["prompt"] == "a cat"
    assert isinstance(out, ImageGenerationResponse)
    assert out.images[0].data == "aGk="


def openai_audio_generate() -> None:
    transport = FakeTransport([FakeResponse(200, b"WAV", headers=[("content-type", "audio/wav")])])
    lm = OpenAILM(api_key="test", transport=transport)
    out = lm.audio_generate(AudioGenerationRequest(model="tts", prompt="hello", voice="alloy", format="wav"))
    req = transport.requests[0]
    assert req.url == "https://api.openai.com/v1/audio/speech"
    assert isinstance(out, AudioGenerationResponse)
    assert out.audio.media_type == "audio/wav"
    assert base64.b64decode(out.audio.data) == b"WAV"


def openai_live_url_and_headers() -> None:
    lm = OpenAILM(api_key="sk-test")
    try:
        url = lm._live_url("gpt-realtime")
        headers = lm._live_headers()
    finally:
        lm.close()
    assert url == "wss://api.openai.com/v1/realtime?model=gpt-realtime"
    assert headers["Authorization"] == "Bearer sk-test"
    assert headers["OpenAI-Beta"] == "realtime=v1"


def anthropic_file_upload() -> None:
    transport = FakeTransport([json_response({"id": "file_anth_1"})])
    lm = AnthropicLM(api_key="test", transport=transport)
    out = lm.file_upload(FileUploadRequest(filename="hello.txt", bytes_data=b"hello", media_type="text/plain"))
    req = transport.requests[0]
    assert req.method == "POST"
    assert "/v1/files" in req.url
    assert isinstance(out, FileUploadResponse)
    assert out.id == "file_anth_1"


def anthropic_batch_submit() -> None:
    transport = FakeTransport([json_response({"id": "batch_anth_1", "processing_status": "in_progress"})])
    lm = AnthropicLM(api_key="test", transport=transport)
    out = lm.batch_submit(BatchRequest(requests=(Request(model="claude-test", messages=(Message.user("hi"),)),)))
    req = transport.requests[0]
    assert "/v1/messages/batches" in req.url
    assert isinstance(out, BatchResponse)
    assert out.status == "running"


def gemini_file_upload() -> None:
    transport = FakeTransport([json_response({"file": {"name": "files/abc"}})])
    lm = GeminiLM(api_key="test", transport=transport)
    out = lm.file_upload(FileUploadRequest(filename="hello.txt", bytes_data=b"hello", media_type="text/plain"))
    req = transport.requests[0]
    assert "/upload/v1beta/files" in req.url
    assert isinstance(out, FileUploadResponse)
    assert out.id == "files/abc"


def gemini_image_generate() -> None:
    transport = FakeTransport([
        json_response({
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {"inlineData": {"mimeType": "image/png", "data": "aGk="}}
                        ]
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
        })
    ])
    lm = GeminiLM(api_key="test", transport=transport)
    out = lm.image_generate(ImageGenerationRequest(model="gemini-2.5-flash-image-preview", prompt="a cat"))
    req = transport.requests[0]
    assert ":generateContent" in req.url
    assert isinstance(out, ImageGenerationResponse)
    assert out.images[0].media_type == "image/png"


def gemini_live_url_shape() -> None:
    lm = GeminiLM(api_key="sk-gem")
    try:
        url = lm._live_url()
    finally:
        lm.close()
    assert url.startswith("wss://")
    assert "BidiGenerateContent" in url
    assert "key=sk-gem" in url


def gemini_live_setup_payload_shape() -> None:
    lm = GeminiLM(api_key="sk-gem")
    try:
        payload = lm._live_setup_payload(LiveConfig(
            model="gemini-2.5-flash-preview-native-audio-dialog",
            system="be helpful",
            tools=(FunctionTool(name="lookup", parameters={"type": "object", "properties": {}}),),
            voice="Puck",
            output_format=AudioFormat(encoding="pcm16", sample_rate=24000),
        ))
    finally:
        lm.close()
    setup = payload["setup"]
    assert setup["model"].startswith("models/")
    assert setup["systemInstruction"]["parts"][0]["text"] == "be helpful"
    assert setup["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    generation = setup["generationConfig"]
    assert generation["responseModalities"] == ["AUDIO"]
    assert generation["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"


def openai_live_session_payload_shape() -> None:
    lm = OpenAILM(api_key="sk-test")
    try:
        payload = lm._live_session_update_payload(LiveConfig(
            model="gpt-realtime",
            system="you are helpful",
            tools=(FunctionTool(name="lookup", parameters={"type": "object", "properties": {}}),),
            voice="alloy",
            input_format=AudioFormat(encoding="pcm16", sample_rate=24000),
            output_format=AudioFormat(encoding="pcm16", sample_rate=24000),
        ))
    finally:
        lm.close()
    assert payload["type"] == "session.update"
    session = payload["session"]
    assert session["instructions"] == "you are helpful"
    assert session["voice"] == "alloy"
    assert session["modalities"] == ["text", "audio"]
    assert session["output_audio_format"] == "pcm16"
    assert session["input_audio_format"] == "pcm16"
    assert session["tools"][0]["name"] == "lookup"


CASES: tuple[tuple[str, Callable[[], None]], ...] = (
    ("openai.file_upload", openai_file_upload),
    ("openai.batch_submit", openai_batch_submit),
    ("openai.image_generate", openai_image_generate),
    ("openai.audio_generate", openai_audio_generate),
    ("openai.live_url_and_headers", openai_live_url_and_headers),
    ("openai.live_session_payload_shape", openai_live_session_payload_shape),
    ("anthropic.file_upload", anthropic_file_upload),
    ("anthropic.batch_submit", anthropic_batch_submit),
    ("gemini.file_upload", gemini_file_upload),
    ("gemini.image_generate", gemini_image_generate),
    ("gemini.live_url_shape", gemini_live_url_shape),
    ("gemini.live_setup_payload_shape", gemini_live_setup_payload_shape),
)


def result_to_dict(result: EndpointResult) -> JsonObject:
    return {"id": result.case_id, "status": result.status, "reason": result.reason}


def write_markdown(results: list[EndpointResult], path: Path) -> None:
    counts: JsonObject = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    lines = [
        "# Endpoint conformance",
        "",
        "Generated by `conformance/check_endpoint_fixtures.py`.",
        "",
        "## Summary",
        "",
        "| Status | Count |",
        "|---|---:|",
    ]
    for status in ("pass", "fail"):
        lines.append(f"| {status} | {counts.get(status, 0)} |")
    lines.extend(["", "## Cases", "", "| Case | Status | Reason |", "|---|---|---|"])
    for result in results:
        reason = (result.reason or "").replace("|", "\\|")
        lines.append(f"| `{result.case_id}` | {result.status} | {reason} |")
    path.write_text("\n".join(lines) + "\n")


def iter_cases() -> Iterator[tuple[str, Callable[[], None]]]:
    return iter(CASES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", help="run only one case id")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)

    cases = list(iter_cases())
    if args.case:
        cases = [(name, fn) for name, fn in cases if name == args.case]
        if not cases:
            raise SystemExit(f"unknown case: {args.case}")

    results = [run_case(name, fn) for name, fn in cases]
    counts: JsonObject = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1

    REPORT_DIR.mkdir(exist_ok=True)
    json_path = args.json or REPORT_DIR / "endpoint-fixtures.json"
    md_path = args.markdown or REPORT_DIR / "endpoint-fixtures.md"
    json_path.write_text(json.dumps({"summary": counts, "total": len(results), "results": [result_to_dict(r) for r in results]}, indent=2, sort_keys=True) + "\n")
    write_markdown(results, md_path)

    print(f"endpoint conformance: {counts} / total={len(results)}")
    print(f"json: {json_path}")
    print(f"markdown: {md_path}")

    if args.strict and any(result.status != "pass" for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
