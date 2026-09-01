"""Video generation tests.

Wire shapes captured live 2026-09-01 (curl-fixtures/video-2026-09-01/
and xai-2026-09-01/): Sora's /v1/videos job objects, Veo's operations
(key-bound download URI, 403 without the header), and grok-imagine's
request_id ticket with a public MP4 URL.
"""

from __future__ import annotations

import base64
import json

import pytest

from lm15 import VideoGenerationRequest, VideoJob, VideoJobInfo, serde
from lm15.errors import UnsupportedFeatureError
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM, XaiLM
from lm15.providers.base import HttpResponse

SORA_DONE = {
    "id": "video_1", "object": "video", "created_at": 1788266776, "status": "completed",
    "model": "sora-2", "progress": 100, "seconds": "4", "size": "720x1280",
}
VEO_DONE = {
    "name": "models/veo-3.1-lite-generate-preview/operations/op1", "done": True,
    "response": {"generateVideoResponse": {"generatedSamples": [
        {"video": {"uri": "https://generativelanguage.googleapis.com/v1beta/files/f1:download?alt=media"}}]}},
}
XAI_DONE = {"status": "done", "progress": 100, "model": "grok-imagine-video",
            "video": {"url": "https://vidgen.x.ai/bucket/v.mp4", "duration": 8}}


# ─── Status mapping (canonical vocabulary, wire words in provider_data) ──

def test_sora_status_mapping():
    lm = OpenAILM(api_key="k")
    assert lm._video_job_from_body(json.dumps({"id": "v", "status": "queued"})).status == "queued"
    assert lm._video_job_from_body(json.dumps({"id": "v", "status": "in_progress", "progress": 37})).progress == 37
    done = lm._video_job_from_body(json.dumps(SORA_DONE))
    assert done.status == "completed" and done.created_at == "2026-09-01T12:46:16Z" and done.model == "sora-2"


def test_veo_status_mapping():
    lm = GeminiLM(api_key="k")
    running = lm._video_job_from_body(json.dumps({"name": "models/m/operations/op1"}))
    assert running.status == "running"  # operations expose no queued/running split
    assert lm._video_job_from_body(json.dumps(VEO_DONE)).status == "completed"
    failed = lm._video_job_from_body(json.dumps({"name": "n", "done": True, "error": {"code": 3}}))
    assert failed.status == "failed"


def test_xai_status_needs_injected_id():
    lm = XaiLM(api_key="k")
    submit = lm._video_job_from_body(json.dumps({"request_id": "req-1"}))
    assert (submit.id, submit.status) == ("req-1", "queued")
    status = lm._video_job_from_body(json.dumps(XAI_DONE), "req-1")
    assert (status.id, status.status, status.progress) == ("req-1", "completed", 100)


# ─── Result delivery modes ──────────────────────────────────────────

def test_sora_result_is_fetched_bytes():
    lm = OpenAILM(api_key="k")
    fetch = lm._video_result_fetch(SORA_DONE)
    assert fetch.url.endswith("/videos/video_1/content")
    part = lm._video_part(SORA_DONE, HttpResponse(200, "OK", [("content-type", "video/mp4")], b"MP4!"))
    assert part.media_type == "video/mp4" and base64.b64decode(part.data) == b"MP4!"


def test_veo_result_is_fetched_bytes_with_auth():
    # The URI is key-bound (403 without the header, verified live) —
    # a URL the user cannot open is not an honest VideoPart.
    lm = GeminiLM(api_key="k")
    fetch = lm._video_result_fetch(VEO_DONE)
    assert fetch.url.startswith("https://generativelanguage.googleapis.com/v1beta/files/")
    assert any(k.lower() == "x-goog-api-key" for k, _ in fetch.headers)


def test_xai_result_is_the_public_url():
    lm = XaiLM(api_key="k")
    assert lm._video_result_fetch(XAI_DONE) is None
    part = lm._video_part(XAI_DONE, None)
    assert part.url == "https://vidgen.x.ai/bucket/v.mp4" and part.media_type == "video/mp4"


# ─── Submit builds and honest raises ────────────────────────────────

def test_sora_seconds_becomes_string_enum():
    lm = OpenAILM(api_key="k")
    tr = lm._video_submit_request(VideoGenerationRequest(model="sora-2", prompt="p", seconds=4))
    assert json.loads(tr.body)["seconds"] == "4"


def test_veo_seconds_maps_to_duration():
    lm = GeminiLM(api_key="k")
    tr = lm._video_submit_request(VideoGenerationRequest(model="veo-3.1-lite-generate-preview", prompt="p", seconds=4))
    body = json.loads(tr.body)
    assert body["instances"] == [{"prompt": "p"}]
    assert body["parameters"]["durationSeconds"] == 4
    assert tr.url.endswith(":predictLongRunning")


def test_xai_seconds_and_images_raise():
    lm = XaiLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm._video_submit_request(VideoGenerationRequest(model="grok-imagine-video", prompt="p", seconds=4))


def test_unreceipted_image_inputs_raise_everywhere():
    from lm15 import ImagePart

    img = (ImagePart(media_type="image/png", data="aGk="),)
    for lm in (OpenAILM(api_key="k"), GeminiLM(api_key="k"), XaiLM(api_key="k")):
        with pytest.raises(UnsupportedFeatureError):
            lm._video_submit_request(VideoGenerationRequest(model="m", prompt="p", images=img))


# ─── Enumerability honesty ──────────────────────────────────────────

def test_openai_lists_account_wide():
    lm = OpenAILM(api_key="k")
    tr = lm._video_list_request(5, None)
    assert "/videos" in tr.url


def test_gemini_lists_per_model_only():
    lm = GeminiLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm._video_list_request(5, None)
    tr = lm._video_list_request(5, "veo-3.1-lite-generate-preview")
    assert "/operations" in tr.url


def test_xai_has_no_list():
    lm = XaiLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm._video_list_request(5, None)


def test_anthropic_video_unsupported():
    lm = AnthropicLM(api_key="k")
    with pytest.raises(UnsupportedFeatureError):
        lm.video_submit(VideoGenerationRequest(model="m", prompt="p"))


# ─── Not-finished guard and serde ───────────────────────────────────

def test_result_raises_while_running(monkeypatch):
    lm = XaiLM(api_key="k")
    monkeypatch.setattr(lm, "_send", lambda req: HttpResponse(200, "OK", [], json.dumps(
        {"status": "pending", "progress": 3}).encode()))
    with pytest.raises(ValueError, match="not finished"):
        lm.video_result("req-1")


def test_serde_roundtrips():
    req = VideoGenerationRequest(model="sora-2", prompt="p", seconds=8, extensions={"size": "720x1280"})
    assert serde.video_generation_request_from_dict(serde.video_generation_request_to_dict(req)) == req
    job = VideoJobInfo(id="v", status="running", progress=42, created_at="2026-09-01T12:46:16Z",
                       model="sora-2", provider_data={"k": "v"})
    assert serde.video_job_from_dict(serde.video_job_to_dict(job)) == job
