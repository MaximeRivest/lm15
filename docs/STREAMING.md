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
