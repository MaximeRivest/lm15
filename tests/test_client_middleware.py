from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.client import UniversalLM
from lm15.errors import InvalidRequestError, RateLimitError
from lm15.features import EndpointSupport, ProviderManifest
from lm15.middleware import with_cache, with_history, with_retries
from lm15.types import LMRequest, LMResponse, Message, Part, Usage


class EchoAdapter:
    provider = "echo"
    capabilities = None
    supports = EndpointSupport(complete=True, stream=True)
    manifest = ProviderManifest(provider="echo", supports=supports)

    def __init__(self):
        self.calls = 0

    def complete(self, request: LMRequest) -> LMResponse:
        self.calls += 1
        return LMResponse(
            id="e1",
            model=request.model,
            message=Message(role="assistant", parts=(Part.text_part("ok"),)),
            finish_reason="stop",
            usage=Usage(),
        )

    def stream(self, request: LMRequest):
        yield from ()


class MiddlewareTests(unittest.TestCase):
    def test_history_and_cache(self):
        lm = UniversalLM()
        adapter = EchoAdapter()
        lm.register(adapter)

        hist: list[dict] = []
        cache: dict = {}
        lm.middleware.add(with_cache(cache))
        lm.middleware.add(with_history(hist))

        req = LMRequest(model="echo-model", messages=(Message.user("hi"),))
        _ = lm.complete(req, provider="echo")
        _ = lm.complete(req, provider="echo")

        self.assertEqual(adapter.calls, 1)
        self.assertEqual(len(hist), 1)

    def test_retries_only_transient_errors(self):
        class ErrorAdapter(EchoAdapter):
            def __init__(self, error: Exception):
                super().__init__()
                self._error = error

            def complete(self, request: LMRequest) -> LMResponse:
                self.calls += 1
                raise self._error

        req = LMRequest(model="echo-model", messages=(Message.user("hi"),))

        lm_non_transient = UniversalLM()
        a1 = ErrorAdapter(InvalidRequestError("bad input"))
        lm_non_transient.register(a1)
        lm_non_transient.middleware.add(with_retries(max_retries=2, sleep_base=0))
        with self.assertRaises(InvalidRequestError):
            lm_non_transient.complete(req, provider="echo")
        self.assertEqual(a1.calls, 1)

        lm_transient = UniversalLM()
        a2 = ErrorAdapter(RateLimitError("slow down"))
        lm_transient.register(a2)
        lm_transient.middleware.add(with_retries(max_retries=2, sleep_base=0))
        with self.assertRaises(RateLimitError):
            lm_transient.complete(req, provider="echo")
        self.assertEqual(a2.calls, 3)


if __name__ == "__main__":
    unittest.main()
