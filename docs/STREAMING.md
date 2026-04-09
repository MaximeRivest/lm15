# Streaming

LM15 normalizes SSE into `StreamEvent`.

## Event Types

- `start`
- `part_start`
- `delta`
- `part_end`
- `end`
- `error`

## Transport Behavior

- `UrlLibTransport`: line-based reads from response stream.
- `PyCurlTransport`: callback-driven chunk intake + line-buffer emission.

## Adapter Responsibilities

- Convert provider-specific stream payloads into normalized deltas.
- Emit `error` on provider stream error events.
- Emit `end` on completion markers.

## Stream Error Shape

`StreamEvent(type="error")` uses normalized error metadata:

- `error.code`: canonical lm15 code (`auth`, `billing`, `rate_limit`, `invalid_request`, `context_length`, `timeout`, `server`, `provider`)
- `error.provider_code`: provider-native code/status/type
- `error.message`: provider message
