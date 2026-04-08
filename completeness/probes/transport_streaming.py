from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from lm15.transports.base import HttpRequest
from lm15.transports.pycurl_transport import PyCurlTransport

from ._helpers import ProbeResult


class _SSEHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        if self.path != "/sse":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in [b"data: a\n\n", b"data: b\n\n", b"data: c\n\n"]:
            self.wfile.write(chunk)
            self.wfile.flush()
            time.sleep(0.002)


def run(test: dict, root: Path) -> ProbeResult:
    try:
        import pycurl  # noqa: F401
    except Exception:
        return ProbeResult(status="skip", details="pycurl not installed")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        transport = PyCurlTransport()
        req = HttpRequest(method="GET", url=f"http://127.0.0.1:{srv.server_port}/sse")
        lines = list(transport.stream(req))
        joined = b"".join(lines)
        if b"data: a" in joined and b"data: b" in joined and b"data: c" in joined:
            return ProbeResult(status="pass", details=f"lines={len(lines)}")
        return ProbeResult(status="fail", details="missing expected stream chunks")
    finally:
        srv.shutdown()
        srv.server_close()
