"""Batch jobs: the third execution mode (complete / stream / batch).

Synthetic wire bodies below mirror live captures from 2026-08-31
(/v1/messages/batches, /v1/batches + files, :batchGenerateContent) —
including the observed fact that Anthropic returns results OUT of
submission order, which pins the re-sort rule.
"""
from __future__ import annotations

import json

import pytest

from lm15 import (
    BatchEntry,
    BatchJob,
    BatchJobInfo,
    BatchRequest,
    Message,
    Request,
    UnsupportedFeatureError,
)
from lm15.batch import AsyncBatchJob
from lm15.providers import AnthropicLM, GeminiLM, OpenAILM
from lm15.serde import (
    batch_entry_from_dict,
    batch_entry_to_dict,
    batch_job_from_dict,
    batch_job_to_dict,
    batch_request_from_dict,
    batch_request_to_dict,
)
from lm15.testing import FakeResponse, FakeTransport
from lm15.types import BATCH_STATUSES, BATCH_TERMINAL_STATUSES


def wire(payload: dict | str, status: int = 200) -> FakeResponse:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return FakeResponse(status=status, body=body.encode("utf-8"))


def req(text: str = "hi", model: str = "claude-haiku-4-5") -> Request:
    return Request(model=model, messages=(Message.user(text),))


def body_of(transport_request) -> dict:
    return json.loads(transport_request.body.decode("utf-8"))


# ─── Anthropic message body used inside batch results ────────────────

def anthropic_message(text: str) -> dict:
    return {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": "claude-haiku-4-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 8, "output_tokens": 3},
    }


# ─── Vocabulary ──────────────────────────────────────────────────────

def test_status_vocabulary_shape() -> None:
    assert BATCH_STATUSES == {"queued", "running", "cancelling", "completed", "failed", "cancelled", "expired"}
    assert BATCH_TERMINAL_STATUSES == {"completed", "failed", "cancelled", "expired"}
    assert BATCH_TERMINAL_STATUSES < BATCH_STATUSES


def test_entry_outcome_invariants() -> None:
    with pytest.raises(ValueError, match="carry a Response"):
        BatchEntry(index=0, outcome="succeeded")
    with pytest.raises(ValueError, match="carry an ErrorDetail"):
        BatchEntry(index=0, outcome="errored")
    with pytest.raises(ValueError, match="neither"):
        from lm15.types import ErrorDetail

        BatchEntry(index=0, outcome="expired", error=ErrorDetail(code="provider", message="x"))
    with pytest.raises(ValueError, match="non-negative"):
        BatchEntry(index=-1, outcome="cancelled")


# ─── Anthropic ───────────────────────────────────────────────────────

def test_anthropic_submit_positional_custom_ids() -> None:
    transport = FakeTransport([wire({"id": "msgbatch_1", "processing_status": "in_progress",
                                     "created_at": "2026-08-31T19:22:09.697866+00:00"})])
    lm = AnthropicLM(api_key="k", transport=transport)
    job = lm.batch_submit(BatchRequest(requests=(req("a"), req("b"))))
    sent = body_of(transport.requests[0])
    assert [r["custom_id"] for r in sent["requests"]] == ["0", "1"]
    assert job.id == "msgbatch_1" and job.status == "running"
    assert job.created_at == "2026-08-31T19:22:09Z"


def test_anthropic_label_raises_honestly() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="label"):
        lm.batch_submit(BatchRequest(requests=(req(),), label="nightly"))


def test_anthropic_ended_status_splits_on_counts() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    def status(counts):
        return lm._batch_job_info({"id": "b", "processing_status": "ended", "request_counts": counts}).status
    assert status({"succeeded": 2, "canceled": 0, "expired": 0, "errored": 0}) == "completed"
    assert status({"succeeded": 1, "canceled": 1, "expired": 0, "errored": 0}) == "completed"
    assert status({"succeeded": 0, "canceled": 2, "expired": 0, "errored": 0}) == "cancelled"
    assert status({"succeeded": 0, "canceled": 0, "expired": 2, "errored": 0}) == "expired"


def test_anthropic_results_resort_out_of_order_lines() -> None:
    # Live capture 2026-08-31: results arrived custom_id 1 before 0.
    status_body = {"id": "msgbatch_1", "processing_status": "ended",
                   "request_counts": {"succeeded": 2, "errored": 0, "canceled": 0, "expired": 0},
                   "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_1/results"}
    jsonl = "\n".join([
        json.dumps({"custom_id": "1", "result": {"type": "succeeded", "message": anthropic_message("Tokyo")}}),
        json.dumps({"custom_id": "0", "result": {"type": "succeeded", "message": anthropic_message("Paris")}}),
    ])
    transport = FakeTransport([wire(status_body), wire(jsonl)])
    lm = AnthropicLM(api_key="k", transport=transport)
    entries = lm.batch_results("msgbatch_1")
    assert [e.index for e in entries] == [0, 1]
    assert [e.response.text for e in entries] == ["Paris", "Tokyo"]
    assert all(e.ok for e in entries)
    assert transport.requests[1].url == status_body["results_url"]


def test_anthropic_errored_entry_maps_through_frozen_error_path() -> None:
    status_body = {"id": "b", "processing_status": "ended",
                   "request_counts": {"succeeded": 1, "errored": 1, "canceled": 0, "expired": 0},
                   "results_url": "https://api.anthropic.com/x"}
    jsonl = "\n".join([
        json.dumps({"custom_id": "0", "result": {"type": "succeeded", "message": anthropic_message("ok")}}),
        json.dumps({"custom_id": "1", "result": {"type": "errored", "error": {
            "type": "error", "error": {"type": "invalid_request_error", "message": "bad model"}}}}),
    ])
    lm = AnthropicLM(api_key="k", transport=FakeTransport([wire(status_body), wire(jsonl)]))
    entries = lm.batch_results("b")
    assert entries[1].outcome == "errored" and not entries[1].ok
    assert entries[1].error.code == "invalid_request"
    assert "bad model" in entries[1].error.message


def test_results_before_terminal_raises_valueerror() -> None:
    lm = AnthropicLM(api_key="k", transport=FakeTransport([
        wire({"id": "b", "processing_status": "in_progress"})]))
    with pytest.raises(ValueError, match="not finished.*running"):
        lm.batch_results("b")


def test_anthropic_list_and_cancel() -> None:
    transport = FakeTransport([
        wire({"data": [{"id": "b2", "processing_status": "in_progress"},
                        {"id": "b1", "processing_status": "ended",
                         "request_counts": {"succeeded": 1, "errored": 0, "canceled": 0, "expired": 0}}]}),
        wire({"id": "b2", "processing_status": "canceling"}),
    ])
    lm = AnthropicLM(api_key="k", transport=transport)
    jobs = lm.batch_list(limit=2)
    assert [j.id for j in jobs] == ["b2", "b1"]
    assert jobs[1].status == "completed"
    assert "limit=2" in transport.requests[0].url
    assert lm.batch_cancel("b2").status == "cancelling"


# ─── OpenAI ──────────────────────────────────────────────────────────

def test_openai_submit_uploads_jsonl_then_creates() -> None:
    transport = FakeTransport([
        wire({"id": "file_1"}),
        wire({"id": "batch_1", "status": "validating", "created_at": 1756666000,
              "metadata": {"label": "nightly"}}),
    ])
    lm = OpenAILM(api_key="k", transport=transport)
    job = lm.batch_submit(BatchRequest(requests=(req(model="gpt-5-nano"),), label="nightly"))
    upload, submit = transport.requests
    assert upload.url.endswith("/files")
    assert b"lm15-batch.jsonl" in upload.body and b"batch" in upload.body
    sent = body_of(submit)
    assert sent["input_file_id"] == "file_1"
    assert sent["endpoint"] == "/v1/responses"
    assert sent["completion_window"] == "24h"
    assert sent["metadata"] == {"label": "nightly"}
    assert job.status == "queued" and job.label == "nightly"
    assert job.created_at == "2025-08-31T18:46:40Z"


def test_openai_results_two_files_and_fill_in() -> None:
    responses_body = {
        "id": "resp_1", "object": "response", "status": "completed", "model": "gpt-5-nano",
        "output": [{"type": "message", "id": "m1", "role": "assistant",
                     "content": [{"type": "output_text", "text": "Paris"}]}],
        "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
    }
    status_body = {"id": "batch_1", "status": "expired", "output_file_id": "fo", "error_file_id": "fe",
                   "request_counts": {"total": 3, "completed": 1, "failed": 1}}
    out_jsonl = json.dumps({"custom_id": "0", "response": {"status_code": 200, "body": responses_body}})
    err_jsonl = json.dumps({"custom_id": "1", "response": {"status_code": 429, "body": {
        "error": {"type": "rate_limit_error", "message": "slow down"}}}})
    transport = FakeTransport([wire(status_body), wire(out_jsonl), wire(err_jsonl)])
    lm = OpenAILM(api_key="k", transport=transport)
    entries = lm.batch_results("batch_1")
    assert [e.outcome for e in entries] == ["succeeded", "errored", "expired"]
    assert entries[0].response.text == "Paris"
    assert entries[1].error is not None
    # Entry 2 never reached the output files; the expired job explains it.
    assert entries[2].response is None and entries[2].error is None


def test_openai_cancelled_during_validating_has_no_entries() -> None:
    # Live wire body captured 2026-09-01 (openai-cancel.json): a batch
    # cancelled while `validating` reports request_counts.total=0 — the
    # provider never registered the requests. results() faithfully
    # mirrors that accounting instead of fabricating entries from the
    # (expiring) input file side-channel.
    status_body = {"id": "batch_c", "status": "cancelled",
                   "output_file_id": None, "error_file_id": None,
                   "request_counts": {"total": 0, "completed": 0, "failed": 0},
                   "metadata": {"label": "lm15-cancel-capture-2026-09-01"}}
    transport = FakeTransport([wire(status_body)])
    lm = OpenAILM(api_key="k", transport=transport)
    assert lm.batch_results("batch_c") == ()


# ─── Gemini (wire shapes verified live 2026-08-31) ───────────────────

def gemini_status_body(state: str, done: bool = False, extra: dict | None = None) -> dict:
    body = {"name": "batches/abc", "metadata": {"state": state, "createTime": "2026-08-31T19:21:52.726672467Z",
                                                  "displayName": "nightly"}}
    if done:
        body["done"] = True
    if extra:
        body.update(extra)
    return body


def test_gemini_submit_inline_with_keys() -> None:
    transport = FakeTransport([wire(gemini_status_body("BATCH_STATE_PENDING"))])
    lm = GeminiLM(api_key="k", transport=transport)
    job = lm.batch_submit(BatchRequest(
        requests=(req(model="gemini-2.5-flash"), req(model="gemini-2.5-flash")), label="nightly"))
    sent = body_of(transport.requests[0])
    inline = sent["batch"]["inputConfig"]["requests"]["requests"]
    assert [item["metadata"]["key"] for item in inline] == ["0", "1"]
    assert sent["batch"]["displayName"] == "nightly"
    assert ":batchGenerateContent" in transport.requests[0].url
    assert job.id == "batches/abc" and job.status == "queued"
    assert job.label == "nightly" and job.created_at == "2026-08-31T19:21:52Z"


def test_gemini_submit_without_label_omits_display_name() -> None:
    transport = FakeTransport([wire(gemini_status_body("BATCH_STATE_PENDING"))])
    lm = GeminiLM(api_key="k", transport=transport)
    lm.batch_submit(BatchRequest(requests=(req(model="gemini-2.5-flash"),)))
    assert "displayName" not in body_of(transport.requests[0])["batch"]


def test_gemini_inline_results_parse_and_sort() -> None:
    gen = {"candidates": [{"content": {"parts": [{"text": "Tokyo"}], "role": "model"},
                             "finishReason": "STOP", "index": 0}],
           "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 1, "totalTokenCount": 9},
           "modelVersion": "gemini-2.5-flash", "responseId": "r1"}
    gen2 = dict(gen, candidates=[{"content": {"parts": [{"text": "Paris"}], "role": "model"},
                                   "finishReason": "STOP", "index": 0}])
    body = gemini_status_body("BATCH_STATE_SUCCEEDED", done=True, extra={
        "response": {"inlinedResponses": {"inlinedResponses": [
            {"metadata": {"key": "1"}, "response": gen},
            {"metadata": {"key": "0"}, "response": gen2},
        ]}}})
    lm = GeminiLM(api_key="k", transport=FakeTransport([wire(body)]))
    entries = lm.batch_results("batches/abc")
    assert [e.index for e in entries] == [0, 1]
    assert [e.response.text for e in entries] == ["Paris", "Tokyo"]


def test_gemini_list_maps_operations() -> None:
    body = {"operations": [gemini_status_body("BATCH_STATE_RUNNING")]}
    transport = FakeTransport([wire(body)])
    lm = GeminiLM(api_key="k", transport=transport)
    jobs = lm.batch_list(limit=3)
    assert jobs[0].status == "running"
    assert "pageSize=3" in transport.requests[0].url


# ─── The handle ──────────────────────────────────────────────────────

def test_batch_verb_returns_ticket_and_batches_enumerates() -> None:
    transport = FakeTransport([
        wire({"id": "msgbatch_1", "processing_status": "in_progress"}),
        wire({"data": [{"id": "msgbatch_1", "processing_status": "in_progress"}]}),
    ])
    lm = AnthropicLM(api_key="k", transport=transport)
    job = lm.batch([req("a")])
    assert isinstance(job, BatchJob)
    assert job.id == "msgbatch_1" and job.status == "running" and not job.done
    listed = lm.batches(limit=1)
    assert listed[0].id == job.id


def test_handle_wait_polls_until_terminal(monkeypatch) -> None:
    transport = FakeTransport([
        wire({"id": "b", "processing_status": "in_progress"}),
        wire({"id": "b", "processing_status": "in_progress"}),
        wire({"id": "b", "processing_status": "ended",
              "request_counts": {"succeeded": 1, "errored": 0, "canceled": 0, "expired": 0}}),
    ])
    lm = AnthropicLM(api_key="k", transport=transport)
    monkeypatch.setattr("lm15.batch.time.sleep", lambda s: None)
    job = lm.batch_job("b").wait(poll_every=0.0)
    assert job.status == "completed" and job.done


def test_handle_wait_timeout_raises_builtin_timeout(monkeypatch) -> None:
    transport = FakeTransport([wire({"id": "b", "processing_status": "in_progress"})])
    lm = AnthropicLM(api_key="k", transport=transport)
    monkeypatch.setattr("lm15.batch.time.sleep", lambda s: None)
    clock = iter([0.0, 100.0, 200.0])
    monkeypatch.setattr("lm15.batch.time.monotonic", lambda: next(clock))
    with pytest.raises(TimeoutError, match="still 'running'"):
        lm.batch_job("b").wait(poll_every=0.0, timeout=50.0)


def test_reattach_by_id_fails_fast_on_unknown_id() -> None:
    transport = FakeTransport([wire({"type": "error", "error": {
        "type": "not_found_error", "message": "no such batch"}}, status=404)])
    lm = AnthropicLM(api_key="k", transport=transport)
    with pytest.raises(Exception, match="no such batch"):
        lm.batch_job("msgbatch_nope")


# ─── Async twins ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_async_batch_flow() -> None:
    from lm15.providers.async_base import AsyncAnthropicLM
    from lm15.transports import AsyncTransportResponse

    class ScriptedAsyncTransport:
        """Replays scripted bodies over the real AsyncTransportResponse shape."""

        def __init__(self, responses):
            self.responses = list(responses)
            self.requests = []

        def stream(self, request):
            self.requests.append(request)
            fake = self.responses.pop(0)

            async def chunks():
                yield fake.body

            async def release(body_consumed: bool) -> None:
                return None

            return AsyncTransportResponse(
                status=fake.status, reason="OK",
                headers=[("content-type", "application/json")],
                http_version="HTTP/1.1", chunks=chunks(), release=release,
            )

    status_body = {"id": "b", "processing_status": "ended",
                   "request_counts": {"succeeded": 1, "errored": 0, "canceled": 0, "expired": 0},
                   "results_url": "https://api.anthropic.com/r"}
    jsonl = json.dumps({"custom_id": "0", "result": {"type": "succeeded", "message": anthropic_message("Oslo")}})
    transport = ScriptedAsyncTransport([
        wire({"id": "b", "processing_status": "in_progress"}),  # submit
        wire(status_body),                                        # results: status
        wire(jsonl),                                              # results: fetch
    ])
    lm = AsyncAnthropicLM(api_key="k", transport=transport)
    job = await lm.batch([req("a")])
    assert isinstance(job, AsyncBatchJob) and job.status == "running"
    entries = await job.results()
    assert entries[0].response.text == "Oslo"


# ─── Serde round trips ───────────────────────────────────────────────

def test_batch_serde_roundtrips() -> None:
    breq = BatchRequest(requests=(req("a"), req("b")), label="nightly", extensions={"completion_window": "24h"})
    assert batch_request_from_dict(batch_request_to_dict(breq)) == breq

    info = BatchJobInfo(id="b1", status="running", label="nightly",
                        created_at="2026-08-31T19:22:09Z", provider_data={"raw": True})
    assert batch_job_from_dict(batch_job_to_dict(info)) == info

    lm = AnthropicLM(api_key="k", transport=FakeTransport([]))
    from lm15.providers.base import batch_entry_http, batch_entry_request

    message = anthropic_message("Paris")
    response = lm.parse_response(batch_entry_request(message["model"]), batch_entry_http(message))
    entry = BatchEntry(index=0, outcome="succeeded", response=response)
    round_tripped = batch_entry_from_dict(batch_entry_to_dict(entry))
    assert round_tripped.index == 0 and round_tripped.ok
    assert round_tripped.response.text == "Paris"

    from lm15.types import ErrorDetail

    errored = BatchEntry(index=1, outcome="errored",
                         error=ErrorDetail(code="rate_limit", message="slow down"))
    assert batch_entry_from_dict(batch_entry_to_dict(errored)) == errored
    bare = BatchEntry(index=2, outcome="expired")
    assert batch_entry_from_dict(batch_entry_to_dict(bare)) == bare


# ─── No silent fallback, anywhere ────────────────────────────────────

def test_unsupported_providers_raise_never_fan_out() -> None:
    from lm15.providers import OpenAIChatLM

    lm = OpenAIChatLM(api_key="k", transport=FakeTransport([]))
    with pytest.raises(UnsupportedFeatureError, match="batch"):
        lm.batch([req(model="qwen")])
    with pytest.raises(UnsupportedFeatureError, match="batch"):
        lm.batches()
