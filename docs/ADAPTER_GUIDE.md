# Adapter Guide

## Implement a Provider Plugin Adapter

Subclass `BaseProviderAdapter` and implement request/response translation for supported operations.

### Required plugin surface

- `complete(request)`
- `stream(request)`
- `live(config)`
- `embeddings(request)`
- `file_upload(request)`
- `batch_submit(request)`
- `image_generate(request)`
- `audio_generate(request)`

Unsupported operations must raise `UnsupportedFeatureError`.

### HTTP translation hooks

Implement:

1. `build_request(request, stream)`
2. `parse_response(request, http_response)`
3. `parse_stream_event(request, sse_event)`
4. Optionally override `normalize_error(status, body)`

### Manifest and support declaration

Every adapter declares:

- `supports: EndpointSupport`
- `manifest: ProviderManifest`

These are consumed by completeness probes and runtime dispatch guards.

### Registration

```python
from lm15.client import UniversalLM

lm = UniversalLM()
lm.register(MyProviderAdapter(...))
```
