from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator
from urllib.error import HTTPError

from ..errors import TransportError
from .base import HttpRequest, HttpResponse, Transport, TransportPolicy


@dataclass(slots=True)
class UrlLibTransport(Transport):
    policy: TransportPolicy = field(default_factory=TransportPolicy)

    def _build_url(self, req: HttpRequest) -> str:
        if not req.params:
            return req.url
        return f"{req.url}?{urllib.parse.urlencode(req.params)}"

    def _prepare(self, req: HttpRequest) -> tuple[urllib.request.Request, float]:
        url = self._build_url(req)
        headers = dict(req.headers)
        body = req.body
        if req.json_body is not None:
            body = json.dumps(req.json_body, separators=(",", ":")).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        timeout = req.timeout if req.timeout is not None else self.policy.timeout
        native_req = urllib.request.Request(url=url, data=body, method=req.method, headers=headers)
        return native_req, timeout

    def request(self, req: HttpRequest) -> HttpResponse:
        native_req, timeout = self._prepare(req)
        attempts = max(1, self.policy.max_retries + 1)
        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(native_req, timeout=timeout) as r:
                    headers = {k.lower(): v for k, v in r.headers.items()}
                    return HttpResponse(status=r.status, headers=headers, body=r.read())
            except HTTPError as e:
                if attempt + 1 == attempts:
                    # Return as HttpResponse so the adapter's normalize_error
                    # can produce a typed, user-friendly error (AuthError, etc.)
                    body = b""
                    try:
                        body = e.read()
                    except Exception:
                        pass
                    headers = {k.lower(): v for k, v in e.headers.items()} if e.headers else {}
                    return HttpResponse(status=e.code, headers=headers, body=body)
                time.sleep((self.policy.backoff_base_ms / 1000.0) * (2**attempt))
            except Exception as e:
                if attempt + 1 == attempts:
                    raise TransportError(str(e)) from e
                time.sleep((self.policy.backoff_base_ms / 1000.0) * (2**attempt))
        raise TransportError("request failed")

    def stream(self, req: HttpRequest) -> Iterator[bytes]:
        native_req, timeout = self._prepare(req)
        try:
            with urllib.request.urlopen(native_req, timeout=timeout) as r:
                while True:
                    line = r.readline()
                    if not line:
                        break
                    yield line
        except HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            msg = f"HTTP {e.code}: {body}" if body else str(e)
            raise TransportError(msg) from e
        except Exception as e:
            raise TransportError(str(e)) from e
