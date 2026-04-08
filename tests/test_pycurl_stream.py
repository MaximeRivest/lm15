from __future__ import annotations

import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.transports.base import HttpRequest
from lm15.transports.pycurl_transport import PyCurlTransport


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
            time.sleep(0.005)


class PyCurlStreamTests(unittest.TestCase):
    def test_incremental_lines(self):
        try:
            import pycurl  # noqa: F401
        except Exception:
            self.skipTest("pycurl not installed")

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            t = PyCurlTransport()
            req = HttpRequest(method="GET", url=f"http://127.0.0.1:{srv.server_port}/sse")
            lines = list(t.stream(req))
            self.assertGreaterEqual(len(lines), 3)
            self.assertTrue(any(b"data: a" in x for x in lines))
            self.assertTrue(any(b"data: b" in x for x in lines))
            self.assertTrue(any(b"data: c" in x for x in lines))
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
