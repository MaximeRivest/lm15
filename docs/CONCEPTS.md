# Concepts

## Universal Contract

Core normalized types are in `types.py`.

- `LMRequest`
- `LMResponse`
- `Message`
- `Part`
- `StreamEvent`
- extended operation types (embeddings/files/batch/image/audio/live)

Provider adapters map wire payloads to these types.

## Adapter Plugin Contract

Defined in `protocols.py` and `providers/base.py`.

Adapters must expose:

- `provider`
- `capabilities`
- `supports`
- `manifest`

Operations:

- `complete`, `stream`, `live`, `embeddings`, `file_upload`, `batch_submit`, `image_generate`, `audio_generate`

Unsupported operations raise `UnsupportedFeatureError`.

## Transport Boundary

Transports live in `transports/*`.

- `UrlLibTransport`
- `PyCurlTransport`

Adapters must not contain socket/HTTP client policy logic beyond request assembly.

## Middleware

`middleware.py` wraps behavior without modifying adapters.

- `with_cache`
- `with_history`
- `with_retries`

## Capability Resolution

`capabilities.py` resolves provider and capabilities by model id.
Optional hydration from `models.dev` via `model_catalog.py`.

## Completeness Harness

`completeness/runner.py` executes fixture/live probes and emits score reports.

- `report.json`
- `report.md`

Required fixture probes are CI-quality gates.
