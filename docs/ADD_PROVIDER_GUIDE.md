# Add a New Provider to LM15 (Complete Guide)

This guide defines the **full mechanical path** for adding a provider plugin to ULM, from architecture fit to completeness score.

---

## 0) Preconditions

- Python environment available.
- LM15 tests passing before you start.

```bash
python -m unittest discover -s tests -v
python completeness/runner.py --mode fixture --fail-under 1.0
```

---

## 1) Understand the LM15 plugin contract

A provider adapter must satisfy the interface in `protocols.py` and `providers/base.py`:

- `complete(request)`
- `stream(request)`
- `live(config)`
- `embeddings(request)`
- `file_upload(request)`
- `batch_submit(request)`
- `image_generate(request)`
- `audio_generate(request)`

And expose:

- `provider: str`
- `capabilities: Capabilities`
- `supports: EndpointSupport`
- `manifest: ProviderManifest`

If the provider does not support a feature, raise `UnsupportedFeatureError` from base behavior.

---

## 2) Add adapter file

Create a new file:

- `providers/<provider>.py`

Use this skeleton:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from lm15.features import EndpointSupport, ProviderManifest
from lm15.protocols import Capabilities
from lm15.sse import SSEEvent
from lm15.transports.base import HttpRequest, HttpResponse, Transport
from lm15.types import LMRequest, LMResponse, Message, Part, StreamEvent, Usage
from lm15.providers.base import BaseProviderAdapter


@dataclass(slots=True)
class NewProviderAdapter(BaseProviderAdapter):
    api_key: str
    transport: Transport
    base_url: str = "https://api.example.com/v1"

    provider: str = "newprovider"
    capabilities: Capabilities = Capabilities(
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        features=frozenset({"streaming"}),
    )
    supports: ClassVar[EndpointSupport] = EndpointSupport(
        complete=True,
        stream=True,
        live=False,
        embeddings=False,
        files=False,
        batches=False,
        images=False,
        audio=False,
    )
    manifest: ClassVar[ProviderManifest] = ProviderManifest(
        provider="newprovider",
        supports=supports,
        auth_modes=("bearer",),
        enterprise_variants=(),
    )

    def build_request(self, request: LMRequest, stream: bool) -> HttpRequest:
        ...

    def parse_response(self, request: LMRequest, response: HttpResponse) -> LMResponse:
        ...

    def parse_stream_event(self, request: LMRequest, raw_event: SSEEvent) -> StreamEvent | None:
        ...
```

---

## 3) Map auth and headers

Choose one or more auth modes and reflect in `manifest.auth_modes`:

- `bearer`
- `x-api-key`
- `query-api-key`

Build headers/params in `build_request()`.

### Rules

- Keep auth logic local to adapter.
- Keep transport generic (do not hardcode provider rules in transport).

---

## 4) Implement `build_request()`

Map LM15 types to provider wire shape.

### Inputs to map

- `LMRequest.model`
- `LMRequest.messages[]`
- `LMRequest.system`
- `LMRequest.tools`
- `LMRequest.config` (`max_tokens`, `temperature`, `response_format`, `reasoning`, `provider` passthrough)

### Required behavior

- For stream mode, set provider stream flag and endpoint.
- For non-stream mode, use standard completion endpoint.
- Preserve unknown provider options via `config.provider` passthrough.

---

## 5) Implement `parse_response()`

Convert provider JSON to normalized `LMResponse`:

- `id`
- `model`
- `message=Message(role="assistant", parts=...)`
- `finish_reason`
- `usage`
- `provider` (raw payload)

### Parts normalization targets

- Text -> `TextPart(...)`
- Tool call -> `ToolCallPart(...)`
- Refusal -> `RefusalPart(...)`
- Thinking -> `ThinkingPart(...)`
- Citation -> `CitationPart(...)`

---

## 6) Implement `parse_stream_event()`

Map provider stream frames to LM15 `StreamEvent`:

- `start`
- `part_start`
- `delta`
- `part_end`
- `end`
- `error`

### Rules

- Return `None` for ignorable events.
- Emit `error` with normalized `{code, provider_code, message}`.
- Never leak provider event shapes directly.

---

## 7) Register adapter in package exports and factory

Update:

- `providers/__init__.py`
- `factory.py` (env key + registration)

Example factory registration:

```python
new_key = os.getenv("NEWPROVIDER_API_KEY")
if new_key:
    client.register(NewProviderAdapter(api_key=new_key, transport=transport))
```

---

## 8) Add fixture payloads (golden wire)

Add fixtures under:

- `tests/fixtures/<provider>_response.json`
- `tests/fixtures/<provider>_tool_response.json` (if tool support)

Add representative stream lines in conformance tests.

---

## 9) Add/extend tests

At minimum:

1. Roundtrip complete parse test.
2. Stream event normalization test.
3. Tool-call normalization test (if supported).
4. Plugin contract test (`supports` + `manifest` present).
5. Unsupported feature guard tests for methods not supported.

---

## 10) Add completeness probes

Update `completeness/spec_matrix.json` with required fixture probes for new provider.

If live API is available, add optional probe entry in `completeness/live_matrix.json`.

---

## 11) Validate locally

```bash
python -m unittest discover -s tests -v
python completeness/runner.py --mode fixture --fail-under 1.0
python completeness/runner.py --mode all --fail-under 1.0
```

If live keys are present:

```bash
python completeness/runner.py --mode live --fail-under 0.0
```

---

## 12) External provider package (no core PR)

You can ship a provider as a standalone package `lm15-x-<provider>`.

### pyproject entry-point

```toml
[project.entry-points."lm15.providers"]
newprovider = "lm15_x_newprovider:build_adapter"
```

Entry-point target can be:

- a factory function returning adapter instance
- an adapter class (instantiated by loader)
- an adapter instance

### Runtime loading

```python
from lm15 import build_default

lm = build_default(discover_plugins=True)
# any installed lm15.providers entry-points are auto-registered
```

### Optional explicit loading

```python
from lm15 import UniversalLM, load_plugins

lm = UniversalLM()
result = load_plugins(lm)
print(result.loaded, result.failed)
```

## 13) Community contribution checklist (PR-ready)

1. Add `providers/<provider>.py`.
2. Register in `providers/__init__.py` and `factory.py`.
3. Add fixtures.
4. Add tests.
5. Update completeness matrices.
6. Run test + completeness commands.
7. Include test/completeness outputs in PR description.

---

## 14) Quality bar for “complete provider support” in ULM

A provider is considered complete when all are true:

- Every `supports.* == True` operation is fully implemented and tested.
- No unsupported operation is accidentally reachable.
- Fixture completeness has required probes green.
- Live probe runs (if credentials available) produce stable pass or documented skip.
- Error mapping and stream normalization are verified.

---

## 15) Common implementation mistakes

- Mapping only text and ignoring tool/refusal/thinking blocks.
- Returning provider-native response shape instead of `LMResponse`.
- Mixing auth logic into transport layer.
- Leaving `supports` inconsistent with implemented methods.
- Adding provider-specific params without passthrough in `config.provider`.

---

## 16) Minimal provider bootstrap template (copy/paste order)

1. Create adapter file.
2. Implement `build_request` + `parse_response` + `parse_stream_event`.
3. Set `supports` and `manifest`.
4. Register in factory and exports.
5. Add fixtures + tests.
6. Add completeness entries.
7. Run test/completeness commands.

This order minimizes rework and keeps architecture integrity intact.
