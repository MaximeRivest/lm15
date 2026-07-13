"""lm15.testing — shipped test doubles (API review A4) and live
Retry-After parsing (A5)."""

from __future__ import annotations

import json

import pytest

from lm15 import (
    AnthropicLM,
    LMRouter,
    Message,
    RateLimitError,
    Request,
    Response,
    ResponseStream,
    RouterConfig,
    Usage,
)
from lm15.testing import FakeLM, FakeResponse, FakeTransport
from lm15.transports import TransportError as WireTransportError
from lm15.types import TextPart


REQ = Request(model="m", messages=(Message.user("hi"),))


def _response(text: str) -> Response:
    return Response(
        id="r1", model="m",
        message=Message.assistant(TextPart(text)),
        finish_reason="stop", usage=Usage(),
    )


class TestFakeLM:
    def test_scripted_responses_and_str_shorthand(self) -> None:
        lm = FakeLM(responses=[_response("one"), "two"])
        assert lm.complete(REQ).text == "one"
        assert lm.complete(REQ).text == "two"
        assert len(lm.requests) == 2

    def test_stream_replays_canonical_events(self) -> None:
        lm = FakeLM(responses=["hello"])
        rs = ResponseStream(lm.stream(REQ), REQ)
        assert "".join(rs) == "hello"
        assert rs.response.finish_reason == "stop"

    def test_scripted_exceptions_raise(self) -> None:
        lm = FakeLM(responses=[RateLimitError("slow down", retry_after=1.0), "ok"])
        with pytest.raises(RateLimitError):
            lm.complete(REQ)
        assert lm.complete(REQ).text == "ok"

    def test_exhausted_script_is_a_clear_error(self) -> None:
        lm = FakeLM(responses=[])
        with pytest.raises(AssertionError, match="scripted"):
            lm.complete(REQ)


class TestFakeTransport:
    def test_real_adapter_serde_offline(self) -> None:
        body = json.dumps({
            "id": "msg_1", "model": "m", "role": "assistant",
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }).encode()
        transport = FakeTransport([FakeResponse(status=200, body=body)])
        lm = AnthropicLM(api_key="k", transport=transport)
        response = lm.complete(REQ)
        assert response.text == "hi there"
        # the fake records the real wire request
        sent = transport.requests[0]
        assert sent.url.endswith("/messages")

    def test_scripted_transport_error(self) -> None:
        transport = FakeTransport([WireTransportError("connection dropped")])
        lm = AnthropicLM(api_key="k", transport=transport)
        from lm15 import TransportError
        with pytest.raises(TransportError):
            lm.complete(REQ)


class TestRouterTransportInjection:
    def test_router_config_transport_reaches_the_adapter(self) -> None:
        body = json.dumps({
            "id": "msg_1", "model": "m", "role": "assistant",
            "content": [{"type": "text", "text": "routed"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        transport = FakeTransport([FakeResponse(status=200, body=body)])
        router = LMRouter(RouterConfig(
            env={}, api_keys={"anthropic": "k"}, transport=transport,
        ))
        response = router.complete(Request(model="anthropic:m",
                                           messages=(Message.user("hi"),)))
        assert response.text == "routed"
        assert transport.requests  # the fake actually served it


class TestRetryAfterParsing:
    def _rate_limited(self, headers) -> FakeTransport:
        body = json.dumps({
            "error": {"type": "rate_limit_error", "message": "slow down"},
        }).encode()
        return FakeTransport([FakeResponse(status=429, body=body, headers=headers)])

    def test_seconds_form_is_parsed(self) -> None:
        lm = AnthropicLM(api_key="k", transport=self._rate_limited(
            [("content-type", "application/json"), ("Retry-After", "30")]
        ))
        with pytest.raises(RateLimitError) as exc_info:
            lm.complete(REQ)
        assert exc_info.value.retry_after == 30.0

    def test_http_date_form_is_parsed(self) -> None:
        from email.utils import format_datetime
        from datetime import datetime, timedelta, timezone
        when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60))
        lm = AnthropicLM(api_key="k", transport=self._rate_limited(
            [("content-type", "application/json"), ("Retry-After", when)]
        ))
        with pytest.raises(RateLimitError) as exc_info:
            lm.complete(REQ)
        assert exc_info.value.retry_after is not None
        assert 50.0 <= exc_info.value.retry_after <= 60.0

    def test_absent_header_leaves_none(self) -> None:
        lm = AnthropicLM(api_key="k", transport=self._rate_limited(
            [("content-type", "application/json")]
        ))
        with pytest.raises(RateLimitError) as exc_info:
            lm.complete(REQ)
        assert exc_info.value.retry_after is None

    def test_garbage_header_is_ignored(self) -> None:
        lm = AnthropicLM(api_key="k", transport=self._rate_limited(
            [("content-type", "application/json"), ("Retry-After", "soonish")]
        ))
        with pytest.raises(RateLimitError) as exc_info:
            lm.complete(REQ)
        assert exc_info.value.retry_after is None
