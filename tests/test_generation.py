"""Media generation (images / speech) tests.

Every wire shape here is a live capture from 2026-09-01
(curl-fixtures/genmedia-2026-09-01/ and xai-2026-09-01/), including the
facts that killed the prototype: OpenAI states output_format, xAI
returns JPEG with an explicit mime_type, Gemini narrates text next to
images, and OpenAI speech is raw bytes typed only by its header.
"""

from __future__ import annotations

import base64
import json

import pytest

from lm15 import (
    AudioPart,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImagePart,
    SpeechGenerationRequest,
    SpeechGenerationResponse,
    Usage,
)
from lm15.errors import UnsupportedFeatureError
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM, XaiLM
from lm15.providers.base import HttpResponse
from lm15.serde import (
    image_generation_request_from_dict,
    image_generation_request_to_dict,
    image_generation_response_from_dict,
    image_generation_response_to_dict,
    speech_generation_request_from_dict,
    speech_generation_request_to_dict,
    speech_generation_response_from_dict,
    speech_generation_response_to_dict,
)

PNG_B64 = base64.b64encode(b"fake-png-bytes").decode()


def _http(body: dict | bytes, headers: list[tuple[str, str]] | None = None) -> HttpResponse:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return HttpResponse(status=200, reason="OK", headers=headers or [], body=raw)


# ─── OpenAI ─────────────────────────────────────────────────────────

OPENAI_IMAGE_BODY = {  # captured shape (b64 truncated)
    "created": 1788221972,
    "background": "opaque",
    "data": [{"b64_json": PNG_B64}],
    "output_format": "png",
    "quality": "low",
    "size": "1024x1024",
    "usage": {"input_tokens": 16, "output_tokens": 272, "total_tokens": 288},
}


def test_openai_image_generate_routes_to_generations():
    lm = OpenAILM(api_key="k")
    tr = lm._image_generate_request(ImageGenerationRequest(model="gpt-image-1-mini", prompt="p", size="1024x1024"))
    assert tr.url.endswith("/images/generations")
    assert json.loads(tr.body) == {"model": "gpt-image-1-mini", "prompt": "p", "size": "1024x1024"}


def test_openai_image_media_type_comes_from_output_format():
    lm = OpenAILM(api_key="k")
    out = lm._image_generation_from_response(None, _http(OPENAI_IMAGE_BODY))
    assert out.images[0].media_type == "image/png"  # from the wire, not assumed
    assert out.usage == Usage(input_tokens=16, output_tokens=272, total_tokens=288)
    assert out.text is None and out.id is None and out.model is None  # captured: absent


def test_openai_edit_routes_to_edits_multipart():
    lm = OpenAILM(api_key="k")
    req = ImageGenerationRequest(
        model="gpt-image-1-mini", prompt="p",
        images=(ImagePart(media_type="image/png", data=PNG_B64),),
    )
    tr = lm._image_generate_request(req)
    assert tr.url.endswith("/images/edits")
    ctype = dict((k.lower(), v) for k, v in tr.headers)["content-type"]
    assert ctype.startswith("multipart/form-data")
    assert b"fake-png-bytes" in tr.body


def test_openai_edit_rejects_url_addressed_input():
    lm = OpenAILM(api_key="k")
    req = ImageGenerationRequest(
        model="m", prompt="p", images=(ImagePart(media_type="image/png", url="https://x/y.png"),),
    )
    with pytest.raises(UnsupportedFeatureError):
        lm._image_generate_request(req)


def test_openai_speech_injects_no_defaults():
    lm = OpenAILM(api_key="k")
    tr = lm._speech_generate_request(SpeechGenerationRequest(model="gpt-4o-mini-tts", prompt="hi"))
    body = json.loads(tr.body)
    assert "voice" not in body and "response_format" not in body  # server decides


def test_openai_speech_media_type_from_header():
    lm = OpenAILM(api_key="k")
    # Captured: the server default is MP3, reported in the header.
    out = lm._speech_generation_from_response(None, _http(b"ID3rawbytes", [("content-type", "audio/mpeg")]))
    assert isinstance(out, SpeechGenerationResponse)
    assert out.audio.media_type == "audio/mpeg"
    assert base64.b64decode(out.audio.data) == b"ID3rawbytes"
    assert out.usage == Usage()  # raw body: no usage exists


# ─── Gemini ─────────────────────────────────────────────────────────

GEMINI_IMAGE_BODY = {  # captured shape: narration text NEXT TO the image
    "candidates": [{
        "content": {"parts": [
            {"text": "Here's a red circle: "},
            {"inlineData": {"mimeType": "image/png", "data": PNG_B64}},
        ], "role": "model"},
        "finishReason": "STOP", "index": 0,
    }],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1305, "totalTokenCount": 1315},
    "modelVersion": "gemini-2.5-flash-image",
    "responseId": "LRqW",
}


def test_gemini_image_keeps_narration_text():
    lm = GeminiLM(api_key="k")
    req = ImageGenerationRequest(model="gemini-2.5-flash-image", prompt="p")
    out = lm._image_generation_from_response(req, _http(GEMINI_IMAGE_BODY))
    assert out.images[0].media_type == "image/png"
    assert out.text == "Here's a red circle: "  # dropped by the old prototype
    assert out.id == "LRqW" and out.model == "gemini-2.5-flash-image"
    assert out.usage.output_tokens == 1305


def test_gemini_image_edit_is_the_same_chat_call_with_parts():
    lm = GeminiLM(api_key="k")
    req = ImageGenerationRequest(
        model="gemini-2.5-flash-image", prompt="p",
        images=(ImagePart(media_type="image/png", data=PNG_B64),),
    )
    tr = lm._image_generate_request(req)
    body = json.loads(tr.body)
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "p"}
    assert parts[1]["inlineData"]["mimeType"] == "image/png"


def test_gemini_size_maps_to_aspect_ratio():
    lm = GeminiLM(api_key="k")
    tr = lm._image_generate_request(ImageGenerationRequest(model="m", prompt="p", size="16:9"))
    body = json.loads(tr.body)
    assert body["generationConfig"]["imageConfig"]["aspectRatio"] == "16:9"


GEMINI_SPEECH_BODY = {
    "candidates": [{
        "content": {"parts": [
            {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000", "data": PNG_B64}},
        ], "role": "model"},
        "finishReason": "STOP", "index": 0,
    }],
    "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 59, "totalTokenCount": 65},
    "modelVersion": "gemini-2.5-flash-preview-tts",
    "responseId": "RBqW",
}


def test_gemini_speech_parameterized_mime_verbatim():
    lm = GeminiLM(api_key="k")
    req = SpeechGenerationRequest(model="gemini-2.5-flash-preview-tts", prompt="p", voice="Kore")
    out = lm._speech_generation_from_response(req, _http(GEMINI_SPEECH_BODY))
    assert out.audio.media_type == "audio/L16;codec=pcm;rate=24000"  # captured, verbatim
    assert out.usage.output_tokens == 59


def test_gemini_speech_format_raises_no_wire_slot():
    lm = GeminiLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm._speech_generate_request(SpeechGenerationRequest(model="m", prompt="p", format="wav"))


def test_gemini_speech_voice_rides_speech_config():
    lm = GeminiLM(api_key="k")
    tr = lm._speech_generate_request(SpeechGenerationRequest(model="m", prompt="p", voice="Kore"))
    cfg = json.loads(tr.body)["generationConfig"]
    assert cfg["responseModalities"] == ["AUDIO"]
    assert cfg["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"


# ─── xAI ────────────────────────────────────────────────────────────

XAI_IMAGE_BODY = {  # captured: JPEG with explicit mime_type; ticks-only usage
    "data": [{"b64_json": PNG_B64, "mime_type": "image/jpeg"}],
    "usage": {"cost_in_usd_ticks": 200000000},
}


def test_xai_image_mime_type_from_wire_is_jpeg():
    lm = XaiLM(api_key="k")
    out = lm._image_generation_from_response(None, _http(XAI_IMAGE_BODY))
    assert out.images[0].media_type == "image/jpeg"  # png would be a lie here
    assert out.usage == Usage()  # ticks are not tokens
    assert out.provider_data["usage"]["cost_in_usd_ticks"] == 200000000


def test_xai_edit_routes_to_edits_never_generations():
    # Captured trap: /images/generations silently IGNORES input images.
    lm = XaiLM(api_key="k")
    req = ImageGenerationRequest(
        model="grok-imagine-image", prompt="p",
        images=(ImagePart(media_type="image/png", data=PNG_B64),),
    )
    tr = lm._image_generate_request(req)
    assert tr.url.endswith("/images/edits")
    assert json.loads(tr.body)["image"]["url"].startswith("data:image/png;base64,")


def test_xai_edit_url_and_file_id_addressing():
    lm = XaiLM(api_key="k")
    by_url = lm._image_generate_request(ImageGenerationRequest(
        model="m", prompt="p", images=(ImagePart(media_type="image/png", url="https://x/y.png"),)))
    assert json.loads(by_url.body)["image"] == {"url": "https://x/y.png"}
    by_id = lm._image_generate_request(ImageGenerationRequest(
        model="m", prompt="p", images=(ImagePart(media_type="image/png", file_id="file-1"),)))
    assert json.loads(by_id.body)["image"] == {"file_id": "file-1"}


def test_xai_rejects_multiple_input_images_and_size():
    lm = XaiLM(api_key="k")
    two = (ImagePart(media_type="image/png", data=PNG_B64),) * 2
    with pytest.raises(UnsupportedFeatureError):
        lm._image_generate_request(ImageGenerationRequest(model="m", prompt="p", images=two))
    with pytest.raises(UnsupportedFeatureError):
        lm._image_generate_request(ImageGenerationRequest(model="m", prompt="p", size="1k"))


def test_xai_speech_unsupported():
    lm = XaiLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm.speech_generate(SpeechGenerationRequest(model="m", prompt="p"))


# ─── Providers without the endpoint ─────────────────────────────────

def test_anthropic_generation_unsupported():
    lm = AnthropicLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm.image_generate(ImageGenerationRequest(model="m", prompt="p"))
    with pytest.raises(UnsupportedFeatureError):
        lm.speech_generate(SpeechGenerationRequest(model="m", prompt="p"))


# ─── Serde round-trips ──────────────────────────────────────────────

def test_serde_roundtrips():
    req = ImageGenerationRequest(
        model="m", prompt="p", size="16:9",
        images=(ImagePart(media_type="image/png", data=PNG_B64),),
        extensions={"quality": "low"},
    )
    assert image_generation_request_from_dict(image_generation_request_to_dict(req)) == req

    resp = ImageGenerationResponse(
        images=(ImagePart(media_type="image/jpeg", data=PNG_B64),),
        text="narration", id="i", model="m",
        usage=Usage(input_tokens=1, output_tokens=2, total_tokens=3),
        provider_data={"k": "v"},
    )
    assert image_generation_response_from_dict(image_generation_response_to_dict(resp)) == resp

    sreq = SpeechGenerationRequest(model="m", prompt="p", voice="Kore", format="wav")
    assert speech_generation_request_from_dict(speech_generation_request_to_dict(sreq)) == sreq

    sresp = SpeechGenerationResponse(
        audio=AudioPart(media_type="audio/L16;codec=pcm;rate=24000", data=PNG_B64),
        usage=Usage(output_tokens=59),
    )
    assert speech_generation_response_from_dict(speech_generation_response_to_dict(sresp)) == sresp
