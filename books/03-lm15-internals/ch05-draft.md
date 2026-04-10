# Chapter 5: The Transport

At the bottom of the stack, something has to make an HTTP request. In most Python libraries, that something is `requests` or `httpx` — mature, battle-tested HTTP clients with connection pooling, HTTP/2, async support, and decades of collective development. lm15 uses `urllib.request.urlopen`. This is the most controversial decision in the library, and this chapter explains why, what it costs, and when to use the alternative.

## The Decision

The argument for `urllib` isn't that it's good at HTTP. It isn't. It has no connection pooling, no HTTP/2, no async, no built-in retry logic, no streaming progress callbacks. Every dedicated HTTP library is better at HTTP.

The argument is that `urllib` ships with Python. It's always there. It can't conflict with another package's version requirements. It can't break during a security patch. It can't fail to compile a C extension on an unusual platform. When you `pip install lm15`, the HTTP transport is already installed.

This matters more than it sounds. `requests` depends on `urllib3`, `certifi`, `charset-normalizer`, and `idna`. `httpx` depends on `httpcore`, `h11`, `anyio`, `sniffio`, and `certifi`. Each dependency is a surface for `pip install` failures, version conflicts with other packages in the user's environment, and security advisories that require emergency updates. lm15 chose to be worse at HTTP in exchange for being impossible to break from the outside.

The cost is real. No connection reuse means every API call opens a new TCP connection and TLS handshake. For a batch of 100 sequential calls, that's 100 handshakes — roughly 200ms of overhead per call at typical latencies. For interactive use (one call at a time), it's negligible. For high-throughput batch processing, it matters, and that's when you reach for pycurl.

## urllib Transport

The implementation is in `transports/urllib_transport.py` — 80 lines. The core is two methods: `request()` for blocking calls and `stream()` for SSE.

**`request()`** builds a `urllib.request.Request` with the URL, headers, and JSON body, then calls `urlopen()` with a timeout. If the call fails with an HTTP error, it reads the error body and returns it as an `HttpResponse` — the adapter will call `map_http_error()` to create a typed exception. If it fails with a non-HTTP error (network down, DNS failure, SSL error), it raises `TransportError`.

The transport has its own retry logic via `TransportPolicy.max_retries`, but it defaults to 0 — retries are handled by middleware or the Model's retry logic, not by the transport. Having retries at two levels would cause confusion.

**`stream()`** opens the connection and reads lines one at a time with `readline()`. It yields raw bytes — each line ending with `\n`. The SSE parser (in `sse.py`) handles the interpretation; the transport just delivers bytes. When the server closes the connection, `readline()` returns empty bytes and the generator ends. If an error occurs mid-stream, the transport raises `TransportError` immediately — there's no way to recover a half-consumed SSE stream.

## The SSE Parser

`sse.py` is 50 lines — a standalone SSE parser that takes an iterator of byte lines and yields `SSEEvent` objects:

```python
@dataclass(slots=True, frozen=True)
class SSEEvent:
    event: str | None
    data: str
```

The parser follows the SSE specification: lines starting with `data:` accumulate into the event's data. Lines starting with `event:` set the event name. An empty line flushes the accumulated data as a complete event. Lines starting with `:` are comments (keep-alive pings) and are ignored.

The parser has two safety limits: `max_line_bytes` (64KB default) prevents a single malformed line from consuming unlimited memory, and `max_event_bytes` (1MB default) prevents a malformed event from doing the same. These are defensive — normal LLM responses don't approach these limits, but a misbehaving proxy or a stuck connection might send unbounded data.

Provider differences in SSE format:
- **OpenAI** sends `data: [DONE]` as the final event
- **Anthropic** sends `event: message_stop` followed by data
- **Gemini** sends standard data events and closes the connection

The parser doesn't know about any of this — it yields raw events. Each adapter interprets the events in its own `stream()` method.

## HttpRequest and HttpResponse

The transport works with its own types, defined in `transports/base.py`:

```python
@dataclass(slots=True, frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)
    json_body: dict | list | None = None
    body: bytes | None = None
    timeout: float | None = None

@dataclass(slots=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes
```

These types are deliberately simple — no content negotiation, no encoding handling, no redirect following. They're thin wrappers over "send these bytes, get those bytes." The adapter builds the `HttpRequest` with the provider's URL, auth headers, and JSON body. The transport sends it and returns the raw response. The adapter parses the response body.

`HttpResponse` is the only mutable dataclass in lm15 (no `frozen=True`). This is pragmatic — the response body is read lazily in some error paths, and mutability avoids constructing a new object.

## TransportPolicy

```python
@dataclass(slots=True, frozen=True)
class TransportPolicy:
    timeout: float = 30.0
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    max_retries: int = 0
    backoff_base_ms: int = 100
    proxy: str | None = None
    http2: bool = False
```

Policy is set once at `build_default()` time and shared by all adapters through the transport. The `proxy` and `http2` fields are recognized by the pycurl transport but ignored by urllib (urllib doesn't support either natively).

## pycurl Transport

`transports/pycurl_transport.py` is the optional high-performance transport. It uses pycurl — a Python binding for libcurl — which provides connection reuse, HTTP/2, and better streaming performance. The interface is identical to `UrlLibTransport`: `request()` and `stream()` with the same `HttpRequest`/`HttpResponse` types.

pycurl is loaded lazily — `build_default()` tries to import it and falls back to urllib if it's not installed. No import error, no warning, just a silent fallback. You install it with `pip install pycurl` and lm15 uses it automatically.

When to use pycurl:
- **Batch processing**: connection reuse eliminates TLS handshake overhead
- **High-throughput**: HTTP/2 multiplexing reduces latency
- **Long-running agents**: connection keep-alive avoids re-establishing connections per turn
- **Latency-sensitive applications**: the handshake overhead of urllib (200-500ms per call on cold connections) disappears

When urllib is fine:
- **Interactive use**: one call at a time, handshake overhead is unnoticeable
- **Simple scripts**: `pip install pycurl` has build dependencies (libcurl, OpenSSL headers) that can fail on some systems
- **CI environments**: adding pycurl to CI means installing system libraries

## The Transport's Place

The transport is the most boring layer in lm15, and that's deliberate. It does one thing — move bytes over HTTP — and every interesting decision happens above it. The adapter decides what JSON to send. The client decides which adapter to use. The model decides when to retry. The transport just sends and receives.

This separation is what makes the transport swappable. You can replace urllib with pycurl without touching any adapter. You could build a mock transport for testing that returns canned responses without hitting any network. You could build a logging transport that records all HTTP traffic. The transport protocol is small enough to implement in 30 lines.

We've now traced our request through all five layers — from `lm15.complete()` through Model, Client, Adapter, and Transport to the wire and back. The next five chapters cover the cross-cutting concerns: streaming internals, middleware, discovery, auth, and errors. These systems don't sit in one layer — they span multiple layers, and understanding them requires seeing the whole stack.
