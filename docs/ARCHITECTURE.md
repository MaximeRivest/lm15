# LM15 Architecture

## Layering

1. **Contract layer** (`types.py`)
   - Universal request/response/stream/live datatypes.
   - Runtime validation in dataclass `__post_init__`.

2. **Provider plugin layer** (`providers/*`)
   - One adapter per provider.
   - Shared plugin contract includes complete/stream/live/embeddings/files/batch/images/audio.
   - `supports` + `manifest` declare endpoint availability and auth variants.

3. **Transport layer** (`transports/*`)
   - HTTP mechanics only.
   - `TransportPolicy` controls timeout/retry/proxy/http2.
   - `pycurl` transport does incremental callback streaming.

4. **Capability/model layer** (`capabilities.py`, `model_catalog.py`)
   - Static fallback resolver for provider/model patterns.
   - Optional hydration from models.dev API.

5. **Client layer** (`client.py`)
   - Adapter registration and provider resolution.
   - Runtime dispatch guards against unsupported operations.

6. **Cross-cutting layer** (`middleware.py`)
   - Retry/history/cache wrappers.

7. **Conformance layer** (`completeness/*`, `tests/*`)
   - Fixture conformance, stream conformance, error taxonomy checks, adapter contract checks.

## Design Rules

- Keep provider SDKs out of core runtime.
- Keep universal schema independent from provider wire schema.
- Keep stream normalization independent from transport implementation.
- Keep parity tracking measurable through completeness score output.
