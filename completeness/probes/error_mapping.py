from __future__ import annotations

from pathlib import Path

from lm15.errors import AuthError, InvalidRequestError, RateLimitError, ServerError, TimeoutError, map_http_error

from ._helpers import ProbeResult


EXPECTED = {
    401: AuthError,
    403: AuthError,
    408: TimeoutError,
    429: RateLimitError,
    400: InvalidRequestError,
    422: InvalidRequestError,
    500: ServerError,
    503: ServerError,
}


def run(test: dict, root: Path) -> ProbeResult:
    for status, klass in EXPECTED.items():
        got = map_http_error(status, "err")
        if not isinstance(got, klass):
            return ProbeResult(status="fail", details=f"status {status} -> {type(got).__name__}, expected {klass.__name__}")
    return ProbeResult(status="pass", details="http error mapping matches taxonomy")
