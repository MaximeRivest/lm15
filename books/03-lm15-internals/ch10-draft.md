# Chapter 10: Errors

A provider returns HTTP 429 with this body:

```json
{"error": {"message": "Rate limit exceeded. Please retry after 30 seconds.", "type": "rate_limit_error"}}
```

By the time this reaches user code, it's:

```python
except RateLimitError as e:
    print(e)  # "Rate limit exceeded. Please retry after 30 seconds."
```

Between the HTTP response and the Python exception, four things happen: the transport returns the status code, the adapter extracts the error message from provider-specific JSON, a mapper turns the status code into an error class, and the exception propagates up through middleware, client, model, and sugar. This chapter traces that path and explains the error hierarchy.

## The Hierarchy

```
ULMError
├── TransportError                  ← network failures, before HTTP
└── ProviderError                   ← API returned an error
    ├── AuthError                   ← 401, 403
    ├── BillingError                ← 402
    ├── RateLimitError              ← 429
    ├── InvalidRequestError         ← 400, 404, 422
    │   └── ContextLengthError      ← input too long
    ├── TimeoutError                ← 408, 504
    ├── ServerError                 ← 5xx
    ├── UnsupportedModelError       ← model not found
    ├── UnsupportedFeatureError     ← endpoint not supported
    └── NotConfiguredError          ← no API key
```

Every error is a subclass of `ULMError`. `except ULMError` catches everything lm15 can throw. Narrower catches let you handle each failure mode differently.

`TransportError` fires before HTTP — the connection failed, DNS didn't resolve, the SSL handshake was rejected, the socket timed out before any bytes arrived. These are network-level failures.

`ProviderError` and its subclasses fire after HTTP — the provider's server responded, but the response indicates an error. The status code determines which subclass.

## The Mapping Pipeline

The transport returns an `HttpResponse` with status and body. It doesn't interpret the status — a 429 is just `HttpResponse(status=429, body=b"...")`. The adapter checks the status and converts it:

1. **Extract the message.** Each adapter has a `normalize_error()` method (or equivalent) that parses the provider's JSON error body. OpenAI puts the message in `error.message`. Anthropic uses the same structure. Gemini uses `error.message` too, but the JSON shape differs.

2. **Map status to class.** `map_http_error(status, message)` in `errors.py`:

```python
def map_http_error(status: int, message: str) -> ProviderError:
    if status in (401, 403):
        return AuthError(message)
    if status == 402:
        return BillingError(message)
    if status == 429:
        return RateLimitError(message)
    if status in (400, 404, 409, 413, 422):
        return InvalidRequestError(message)
    if 500 <= status <= 599:
        return ServerError(message)
    return ProviderError(message)
```

Clean, predictable. Every HTTP error becomes a specific Python exception. The message is whatever the provider said — "Rate limit exceeded", "Invalid API key", "Model not found" — passed through verbatim.

3. **Raise.** The adapter raises the exception. It propagates up through the call stack.

`ContextLengthError` is a special case — it's a subclass of `InvalidRequestError` (both are status 400), but some adapters detect it by inspecting the error message (looking for "context length" or similar) and raise the more specific exception. This lets users catch context overflows separately from other invalid request errors.

## Canonical Error Codes

`canonical_error_code()` maps error classes to string codes:

```python
canonical_error_code(RateLimitError("..."))  # → "rate_limit"
canonical_error_code(AuthError("..."))        # → "auth"
canonical_error_code(TimeoutError("..."))     # → "timeout"
```

And `error_class_for_canonical_code()` does the reverse:

```python
error_class_for_canonical_code("rate_limit")  # → RateLimitError
```

These exist for middleware and plugins. A middleware that logs errors wants a stable string code, not an `isinstance()` check. A plugin that receives error information as a dict (from streaming error events) needs to reconstruct the right exception class.

## Errors in Streaming

Streaming errors arrive differently. Instead of an HTTP error status on the response, the error can occur mid-stream — after some data has already been received.

Adapters yield errors as `StreamEvent(type="error", error={"code": "rate_limit", "message": "..."})`. The `Stream` class converts these to exceptions:

```python
if event.type == "error":
    code = err.get("code", "provider")
    message = err.get("message", "stream error")
    exc_cls = error_class_for_canonical_code(code)
    raise exc_cls(message)
```

This means a streaming call can partially succeed — you might receive 90% of the text and then get a `ServerError`. The text you received is available through `stream._text_parts`, and `_materialize_response()` will build a partial `LMResponse` if accessed.

Connection drops are different. The transport raises `TransportError` directly, which propagates through the adapter's generator and the `Stream` to the user. There's no event — just an exception interrupting iteration.

## Retryability

`Model._is_retryable_error()` classifies which errors are worth retrying:

```python
@staticmethod
def _is_retryable_error(exc: Exception) -> bool:
    return isinstance(exc, (RateLimitError, TimeoutError, ServerError, TransportError))
```

The same classification is used by `with_retries()` middleware. The logic is the same in both places — transient failures (rate limits, timeouts, server errors, network failures) are retryable. Permanent failures (bad key, invalid request, billing issues) are not.

This classification is hardcoded, not configurable. The reasoning: if an `AuthError` were retryable, you'd retry the same bad key and get the same error. If an `InvalidRequestError` were retryable, you'd send the same bad request and get the same rejection. Retrying these wastes time and tokens.

## Propagation Through the Stack

An error at the adapter level propagates upward through every layer:

1. **Adapter** raises `RateLimitError`
2. **Middleware** catches it (if `with_retries` is active), waits, retries, possibly re-raises
3. **UniversalLM.complete()** doesn't catch — propagates to caller
4. **Model.__call__()** catches (if `retries=` > 0), waits, retries, possibly re-raises
5. **api.complete()** doesn't catch — propagates to user

Each layer has one chance to handle the error. Middleware and Model are the two retry points. Everything else passes through.

## What You Now Know

Ten chapters. One request, traced through every layer.

You've seen the sugar (`api.py` — 7 lines of delegation), the model (`model.py` — state, tools, history), the client (`client.py` — routing, middleware, dispatch), the adapters (three providers, three wire formats, one contract), the transport (`urllib` — bytes in, bytes out), the streaming pipeline (bytes → SSE → events → chunks → response), the middleware (retry, cache, observe), discovery (live queries, fallback catalogs, merge), auth (keys from files to headers), and errors (status codes to typed exceptions).

The entire library is 2,408 lines across 30 files. Every line is reachable from the public API. Every design choice — frozen dataclasses, zero dependencies, five layers, prefix-based routing, middleware-based retries — is visible in the source and explained in this book.

When something breaks, you now know where to look. When something needs to change, you know what it will affect. And when someone asks "how does lm15 work?" — you know. All of it.
