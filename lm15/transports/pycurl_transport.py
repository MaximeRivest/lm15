from __future__ import annotations

import io
import json
import queue
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Iterator

from ..errors import TransportError
from .base import HttpRequest, HttpResponse, Transport, TransportPolicy


@dataclass(slots=True)
class PyCurlTransport(Transport):
    policy: TransportPolicy = field(default_factory=TransportPolicy)

    def _build_url(self, req: HttpRequest) -> str:
        if not req.params:
            return req.url
        return f"{req.url}?{urllib.parse.urlencode(req.params)}"

    def _curl(self):
        try:
            import pycurl  # lazy import

            return pycurl
        except Exception as e:
            raise TransportError("pycurl is not installed") from e

    def _prepare(self, req: HttpRequest):
        pycurl = self._curl()
        c = pycurl.Curl()
        url = self._build_url(req)
        headers = dict(req.headers)
        body = req.body
        if req.json_body is not None:
            body = json.dumps(req.json_body, separators=(",", ":")).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        c.setopt(c.URL, url)
        c.setopt(c.CUSTOMREQUEST, req.method)
        timeout = req.timeout if req.timeout is not None else self.policy.timeout
        c.setopt(c.TIMEOUT_MS, int(timeout * 1000))
        c.setopt(c.CONNECTTIMEOUT_MS, int(self.policy.connect_timeout * 1000))
        if body is not None:
            c.setopt(c.POSTFIELDS, body)
        if headers:
            c.setopt(c.HTTPHEADER, [f"{k}: {v}" for k, v in headers.items()])
        if self.policy.proxy:
            c.setopt(c.PROXY, self.policy.proxy)
        if self.policy.http2:
            c.setopt(c.HTTP_VERSION, pycurl.CURL_HTTP_VERSION_2_0)
        return c

    def request(self, req: HttpRequest) -> HttpResponse:
        pycurl = self._curl()
        buf = io.BytesIO()
        hdr_buf = io.BytesIO()
        c = self._prepare(req)
        try:
            c.setopt(c.WRITEDATA, buf)
            c.setopt(c.HEADERFUNCTION, hdr_buf.write)
            c.perform()
            status = int(c.getinfo(c.RESPONSE_CODE))
            raw_headers = hdr_buf.getvalue().decode("utf-8", errors="replace").splitlines()
            parsed = {}
            for line in raw_headers:
                if ":" in line:
                    k, v = line.split(":", 1)
                    parsed[k.strip().lower()] = v.strip()
            return HttpResponse(status=status, headers=parsed, body=buf.getvalue())
        except Exception as e:
            raise TransportError(str(e)) from e
        finally:
            c.close()

    def stream(self, req: HttpRequest) -> Iterator[bytes]:
        c = self._prepare(req)
        q: queue.Queue[bytes | None | Exception] = queue.Queue()

        class LineBuffer:
            __slots__ = ("q", "buf")

            def __init__(self, q: queue.Queue[bytes | None | Exception]):
                self.q = q
                self.buf = bytearray()

            def feed(self, chunk: bytes) -> int:
                self.buf.extend(chunk)
                while True:
                    idx = self.buf.find(b"\n")
                    if idx < 0:
                        break
                    line = bytes(self.buf[: idx + 1])
                    del self.buf[: idx + 1]
                    self.q.put(line)
                return len(chunk)

            def flush(self) -> None:
                if self.buf:
                    self.q.put(bytes(self.buf))
                    self.buf.clear()

        linebuf = LineBuffer(q)

        def run() -> None:
            try:
                c.setopt(c.WRITEFUNCTION, linebuf.feed)
                c.perform()
                status = int(c.getinfo(c.RESPONSE_CODE))
                if status >= 400:
                    linebuf.flush()
                    payload = b""
                    while not q.empty():
                        item = q.get_nowait()
                        if isinstance(item, bytes):
                            payload += item
                    message = payload.decode("utf-8", errors="replace") if payload else ""
                    raise TransportError(f"HTTP {status}: {message}" if message else f"HTTP {status}")
                linebuf.flush()
                q.put(None)
            except Exception as e:  # pragma: no cover - transport thread exceptions are surfaced in iterator
                q.put(e)
            finally:
                c.close()

        t = threading.Thread(target=run, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is None:
                break
            if isinstance(item, Exception):
                raise TransportError(str(item)) from item
            yield item
