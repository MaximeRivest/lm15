# Chapter 8: Discovery

`lm15.models()` returns a list of every model available across all configured providers. In the demo output from Book 1, that was 82 models in 1.3 seconds. Where did they come from? Not from a hardcoded list — that would go stale. Not from a single API — there are three providers with three different list-models endpoints (and one that doesn't have a list endpoint at all). The answer involves live API queries, a fallback catalog, a merge strategy, and in-memory caching.

## The Data Flow

```
lm15.models()
    │
    ├── Live: query each provider's list-models API
    │     ├── OpenAI: GET /v1/models
    │     ├── Anthropic: GET /v1/models  
    │     └── Gemini: GET /v1beta/models?key=...
    │
    ├── Fallback: fetch_models_dev() → models.dev catalog
    │
    └── Merge: live ∪ fallback, live takes priority
         │
         └── Filter: supports=, input_modalities=, output_modalities=
              │
              └── Return: list[ModelSpec]
```

## Live Discovery

`discovery.py` has a fetcher function per provider. Each one calls the provider's list-models API and returns a list of `ModelSpec` objects.

OpenAI's fetcher hits `GET /v1/models` with a Bearer token. The response is a JSON array of model objects. Each becomes a `ModelSpec` with the model ID and provider name. OpenAI's API returns minimal metadata — no context window, no modality information — so most capability fields are empty.

Anthropic's fetcher hits `GET /v1/models` with an `x-api-key` header. Same pattern — extract model IDs, build `ModelSpec` objects.

Gemini's fetcher hits `GET /v1beta/models?key=...` (the key goes in the URL, not a header — Gemini's auth is different). Gemini's response is richer: it includes `inputTokenLimit` and `outputTokenLimit`, which map to `context_window` and `max_output` on `ModelSpec`.

Each fetcher runs with a timeout (default 5 seconds). If a provider doesn't respond, the fetcher returns an empty list — no error, no warning. Discovery is best-effort.

Results are cached in `_LIVE_CACHE` — a module-level dict keyed by provider name. Each entry stores the timestamp and the fetched specs. The cache TTL is 300 seconds. Subsequent calls to `models()` within 5 minutes get cached results. `refresh=True` bypasses the cache for that call.

## The Fallback Catalog

What about models that don't show up in live discovery? Anthropic might not expose all models through their list endpoint. A provider might be temporarily down. The user might be working offline.

`model_catalog.py` provides `fetch_models_dev()`, which fetches from `models.dev/api.json` — a community-maintained catalog of LLM models with capability metadata. This returns `ModelSpec` objects enriched with context windows, modalities, tool support, and reasoning flags.

The fallback is fetched lazily and cached in memory. If `models.dev` is unreachable, the function returns an empty list. Discovery degrades gracefully — you get fewer models, not an error.

## ModelSpec

The normalized model metadata type:

```python
@dataclass(slots=True, frozen=True)
class ModelSpec:
    id: str
    provider: str
    context_window: int | None = None
    max_output: int | None = None
    input_modalities: tuple[str, ...] = ()
    output_modalities: tuple[str, ...] = ()
    tool_call: bool = False
    structured_output: bool = False
    reasoning: bool = False
    raw: dict = field(default_factory=dict)
```

The `raw` field carries the provider's original model object — everything lm15 didn't normalize. For OpenAI, this includes `owned_by` and `created`. For Gemini, it includes `displayName` and supported generation methods.

## The Merge Strategy

`_merge_specs()` combines live and fallback specs:

1. Live specs are primary — they define the initial set
2. For each fallback spec, if it's not in the live set, add it
3. If a model appears in both, merge them — live data takes priority for each field, fallback fills in what live didn't provide

The merge is field-by-field: if live has `context_window=None` and fallback has `context_window=200000`, the merged spec gets `200000`. If live has `tool_call=False` (because the provider didn't report it) and fallback has `tool_call=True`, the merged spec gets `True`.

This merge strategy means live data is never overwritten by fallback, but fallback enriches what live discovery left empty. OpenAI's list endpoint returns almost no capability metadata, but models.dev knows the context windows and modalities, so the merged result is richer than either source alone.

## Filtering

`_filter_specs()` applies the user's filters:

```python
models = lm15.models(supports={"tools", "reasoning"}, input_modalities={"image"})
```

For each `ModelSpec`, the function builds a feature set from boolean flags (`tool_call=True` → `"tools"`, `reasoning=True` → `"reasoning"`), then checks subset containment. `supports={"tools", "reasoning"}` passes only if both flags are true. Modality filters check the `input_modalities` and `output_modalities` tuples.

Filters compose via intersection — all conditions must be met. There's no "or" filtering; if you need that, call `models()` multiple times and merge the results.

## providers_info()

`providers_info()` is built on top of `models()`:

```python
def providers_info(...):
    keys = _resolve_api_keys(...)
    specs = models(...)
    counts = count_by_provider(specs)
    return {
        provider: {"env_keys": ..., "configured": provider in keys, "model_count": ...}
        for provider in all_providers
    }
```

It tells you which providers have keys (`configured`), which environment variables they read (`env_keys`), and how many models are discoverable (`model_count`). Simple, but useful as a diagnostic — "why can't I use Claude?" → `providers_info()` → `anthropic: configured=False`.

## Hydration

The capability resolver in `capabilities.py` (which maps model names to providers and capabilities) can be enriched with model catalog data:

```python
hydrate_with_specs(fetch_models_dev())
```

This populates the resolver's `_model_index` dict, enabling exact-match provider resolution for models whose names don't follow the prefix convention (like `o1-preview`, which doesn't start with `gpt-`).

`build_default(hydrate_models_dev_catalog=True)` does this automatically. Without hydration, provider resolution relies purely on prefix matching. With hydration, it also checks the model catalog.

The next chapter covers auth — how keys move from `.env` files and environment variables into adapter constructors and HTTP headers.
