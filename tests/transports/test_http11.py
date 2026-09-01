"""Unit tests for the HTTP/1.1 codec (pure bytes in, pure bytes out).

These tests exercise the protocol logic without any I/O.  The same codec is
shared between the sync and async transports; correctness here is a
prerequisite for the integration tests.
"""
from __future__ import annotations

import pytest

from lm15.transports._http11 import (
    ChunkedDecoder,
    ContentLengthDecoder,
    EOFDecoder,
    ProtocolError,
    ResponseHeadParser,
    build_request_head,
)


class TestBuildRequestHead:
    def test_basic_get(self) -> None:
        head = build_request_head(
            method="GET", target="/v1/x", host="example.com",
            port=443, is_tls=True, headers=[], body_length=None,
        )
        lines = head.split(b"\r\n")
        assert lines[0] == b"GET /v1/x HTTP/1.1"
        # Host header should NOT include the default port
        assert b"Host: example.com" in lines
        # The blank line terminator
        assert lines[-2] == b""
        assert lines[-1] == b""

    def test_non_default_port_in_host_header(self) -> None:
        head = build_request_head(
            method="GET", target="/", host="example.com",
            port=8443, is_tls=True, headers=[], body_length=None,
        )
        assert b"Host: example.com:8443" in head.split(b"\r\n")

    def test_http_default_port_omitted(self) -> None:
        head = build_request_head(
            method="GET", target="/", host="example.com",
            port=80, is_tls=False, headers=[], body_length=None,
        )
        assert b"Host: example.com" in head.split(b"\r\n")

    def test_ipv6_host_header(self) -> None:
        head = build_request_head(
            method="GET", target="/", host="::1",
            port=8080, is_tls=False, headers=[], body_length=None,
        )
        assert b"Host: [::1]:8080" in head.split(b"\r\n")

    def test_post_with_body_adds_content_length(self) -> None:
        head = build_request_head(
            method="POST", target="/v1/x", host="example.com",
            port=443, is_tls=True,
            headers=[("Content-Type", "application/json")],
            body_length=42,
        )
        assert b"Content-Length: 42" in head.split(b"\r\n")

    def test_post_no_body_has_zero_length(self) -> None:
        head = build_request_head(
            method="POST", target="/", host="example.com",
            port=443, is_tls=True, headers=[], body_length=0,
        )
        assert b"Content-Length: 0" in head.split(b"\r\n")

    def test_user_headers_preserved(self) -> None:
        head = build_request_head(
            method="GET", target="/", host="example.com",
            port=443, is_tls=True,
            headers=[("Authorization", "Bearer sk-xxx"), ("X-Custom", "v")],
            body_length=None,
        )
        lines = head.split(b"\r\n")
        assert b"Authorization: Bearer sk-xxx" in lines
        assert b"X-Custom: v" in lines

    def test_user_host_override(self) -> None:
        """If the caller provides Host, don't duplicate it."""
        head = build_request_head(
            method="GET", target="/", host="example.com",
            port=443, is_tls=True,
            headers=[("Host", "other.com")],
            body_length=None,
        )
        lines = head.split(b"\r\n")
        hosts = [l for l in lines if l.lower().startswith(b"host:")]
        assert hosts == [b"Host: other.com"]

    def test_connection_keepalive_default(self) -> None:
        head = build_request_head(
            method="GET", target="/", host="example.com",
            port=443, is_tls=True, headers=[], body_length=None,
        )
        # We rely on HTTP/1.1 default (keep-alive), do NOT send Connection: close
        assert b"connection: close" not in head.lower()

    def test_forbids_bare_newline_in_header(self) -> None:
        with pytest.raises(ProtocolError):
            build_request_head(
                method="GET", target="/", host="example.com",
                port=443, is_tls=True,
                headers=[("X-Bad", "value\r\nInjected: yes")],
                body_length=None,
            )

    def test_forbids_bare_newline_in_target(self) -> None:
        with pytest.raises(ProtocolError):
            build_request_head(
                method="GET", target="/x\r\nInjected: yes", host="example.com",
                port=443, is_tls=True, headers=[], body_length=None,
            )

    def test_forbids_bare_newline_in_host(self) -> None:
        # The Host header is built from the host argument; it must get the
        # same CRLF-injection screening as caller-supplied headers.
        with pytest.raises(ProtocolError):
            build_request_head(
                method="GET", target="/", host="example.com\r\nInjected: yes",
                port=443, is_tls=True, headers=[], body_length=None,
            )


class TestResponseHeadParser:
    def _feed(self, parser: ResponseHeadParser, data: bytes) -> bool:
        """Feed bytes in chunks of 7 to exercise incremental parsing."""
        for i in range(0, len(data), 7):
            parser.feed(data[i:i + 7])
        return parser.complete

    def test_simple_response(self) -> None:
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 2\r\n"
            b"\r\n"
        )
        p = ResponseHeadParser()
        assert self._feed(p, raw)
        assert p.status == 200
        assert p.reason == "OK"
        assert p.http_version == "HTTP/1.1"
        assert p.header("content-type") == "text/plain"
        assert p.header("Content-Length") == "2"
        assert p.leftover == b""

    def test_incomplete_returns_false(self) -> None:
        p = ResponseHeadParser()
        p.feed(b"HTTP/1.1 200 OK\r\nContent-Type: te")
        assert not p.complete

    def test_leftover_after_headers(self) -> None:
        """Body bytes arriving with the head must be preserved."""
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 5\r\n"
            b"\r\n"
            b"hello"
        )
        p = ResponseHeadParser()
        p.feed(raw)
        assert p.complete
        assert p.leftover == b"hello"

    def test_multiple_same_header(self) -> None:
        raw = (
            b"HTTP/1.1 200 OK\r\n"
            b"Set-Cookie: a=1\r\n"
            b"Set-Cookie: b=2\r\n"
            b"Content-Length: 0\r\n"
            b"\r\n"
        )
        p = ResponseHeadParser()
        p.feed(raw)
        assert p.complete
        assert p.headers_all("set-cookie") == ["a=1", "b=2"]

    def test_case_insensitive_lookup(self) -> None:
        raw = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n"
        p = ResponseHeadParser()
        p.feed(raw)
        assert p.header("CONTENT-TYPE") == "text/plain"

    def test_status_line_malformed_raises(self) -> None:
        p = ResponseHeadParser()
        with pytest.raises(ProtocolError):
            p.feed(b"garbage\r\n\r\n")

    def test_header_without_colon_raises(self) -> None:
        p = ResponseHeadParser()
        with pytest.raises(ProtocolError):
            p.feed(b"HTTP/1.1 200 OK\r\nbadheader\r\n\r\n")

    def test_empty_reason_phrase_ok(self) -> None:
        raw = b"HTTP/1.1 204 \r\nContent-Length: 0\r\n\r\n"
        p = ResponseHeadParser()
        p.feed(raw)
        assert p.complete
        assert p.status == 204

    def test_whitespace_in_header_value(self) -> None:
        raw = b"HTTP/1.1 200 OK\r\nX-Foo:    padded   \r\nContent-Length: 0\r\n\r\n"
        p = ResponseHeadParser()
        p.feed(raw)
        assert p.header("x-foo") == "padded"

    def test_max_header_size_enforced(self) -> None:
        big = b"X-Big: " + (b"a" * 200_000) + b"\r\n"
        p = ResponseHeadParser(max_head_bytes=100_000)
        with pytest.raises(ProtocolError):
            p.feed(b"HTTP/1.1 200 OK\r\n" + big + b"\r\n")


class TestContentLengthDecoder:
    def test_reads_exactly_n_bytes(self) -> None:
        d = ContentLengthDecoder(length=5)
        out = list(d.feed(b"hel")) + list(d.feed(b"lo"))
        assert b"".join(out) == b"hello"
        assert d.complete

    def test_rejects_extra_bytes(self) -> None:
        d = ContentLengthDecoder(length=5)
        out = list(d.feed(b"hello"))
        assert d.complete
        # Additional bytes should not be consumed — they belong to next response
        leftover = d.leftover
        assert leftover == b""
        # Feeding more after complete is a bug: should raise
        with pytest.raises(ProtocolError):
            list(d.feed(b"extra"))

    def test_partial_and_complete(self) -> None:
        d = ContentLengthDecoder(length=10)
        list(d.feed(b"12345"))
        assert not d.complete
        list(d.feed(b"67890"))
        assert d.complete

    def test_leftover_when_chunk_overshoots(self) -> None:
        d = ContentLengthDecoder(length=5)
        out = list(d.feed(b"helloWORLD"))
        assert b"".join(out) == b"hello"
        assert d.complete
        assert d.leftover == b"WORLD"

    def test_zero_length(self) -> None:
        d = ContentLengthDecoder(length=0)
        assert d.complete
        assert list(d.feed(b"")) == []

    def test_eof_midstream_raises(self) -> None:
        d = ContentLengthDecoder(length=10)
        list(d.feed(b"abc"))
        with pytest.raises(ProtocolError):
            d.eof()


class TestChunkedDecoder:
    def test_simple(self) -> None:
        data = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
        d = ChunkedDecoder()
        out = b"".join(d.feed(data))
        assert out == b"hello world"
        assert d.complete
        assert d.leftover == b""

    def test_chunked_hex_size(self) -> None:
        # 0x10 = 16 bytes
        data = b"10\r\n" + (b"a" * 16) + b"\r\n0\r\n\r\n"
        d = ChunkedDecoder()
        assert b"".join(d.feed(data)) == b"a" * 16
        assert d.complete

    def test_incremental(self) -> None:
        d = ChunkedDecoder()
        data = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
        out = b""
        for b in data:
            out += b"".join(d.feed(bytes([b])))
        assert out == b"hello world"
        assert d.complete

    def test_chunk_extensions_discarded(self) -> None:
        # "5;name=value\r\n..." — RFC allows extensions after size
        data = b"5;ext=x\r\nhello\r\n0\r\n\r\n"
        d = ChunkedDecoder()
        assert b"".join(d.feed(data)) == b"hello"
        assert d.complete

    def test_trailing_headers_after_zero_chunk(self) -> None:
        # Zero-chunk may be followed by trailers then blank line
        data = b"5\r\nhello\r\n0\r\nX-Trailer: v\r\n\r\n"
        d = ChunkedDecoder()
        assert b"".join(d.feed(data)) == b"hello"
        assert d.complete

    def test_malformed_size_raises(self) -> None:
        d = ChunkedDecoder()
        with pytest.raises(ProtocolError):
            list(d.feed(b"ZZZ\r\n"))

    def test_leftover_after_end(self) -> None:
        data = b"5\r\nhello\r\n0\r\n\r\nNEXT_RESP"
        d = ChunkedDecoder()
        out = b""
        for chunks in [data]:
            out += b"".join(d.feed(chunks))
        assert out == b"hello"
        assert d.complete
        assert d.leftover == b"NEXT_RESP"

    def test_eof_before_zero_chunk_raises(self) -> None:
        d = ChunkedDecoder()
        list(d.feed(b"5\r\nhel"))
        with pytest.raises(ProtocolError):
            d.eof()


class TestEOFDecoder:
    """HTTP/1.0-style: body ends when the connection closes."""

    def test_reads_until_eof(self) -> None:
        d = EOFDecoder()
        out = b"".join(d.feed(b"hello "))
        out += b"".join(d.feed(b"world"))
        assert out == b"hello world"
        assert not d.complete
        d.eof()
        assert d.complete
