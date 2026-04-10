# Chapter 3: The Client

Our request — `lm15.complete("claude-sonnet-4-5", "Hello.")` — has passed through the sugar and the model. It's now an `LMRequest` arriving at `UniversalLM.complete()`. Something has to decide: this request goes to Anthropic, not OpenAI, not Gemini. Who decides, and how?

## The Router

Open `client.py`. The entire file is 91 lines, and `UniversalLM` is a dataclass with two fields:

```python
@dataclass(slots=True)
class UniversalLM:
    adapters: dict[str, LMAdapter] = field(default_factory=dict)
    middleware: MiddlewarePipeline = field(default_factory=MiddlewarePipeline)
```

A dict of adapters keyed by provider name. A middleware pipeline. That's the entire state of the client. It doesn't know how to talk to any provider — it just knows which adapter to ask.

`complete()` is seven lines:

```python
def complete(self, request: LMRequest, provider: str | None = None) -> LMResponse:
    adapter = self._adapter(request.model, provider)
    if not adapter.supports.complete:
        raise UnsupportedFeatureError(f"{adapter.provider}: complete not supported")
    run = self.middleware.wrap_complete(adapter.complete)
    return run(request)
```

Resolve the adapter. Check it supports `complete`. Wrap the adapter's `complete` method with middleware. Call it. Return the result.

The interesting part is `_adapter()`:

```python
def _adapter(self, model: str, provider: str | None = None) -> LMAdapter:
    p = provider or resolve_provider(model)
    adapter = self.adapters.get(p)
    if not adapter:
        raise ProviderError(f"no adapter registered for provider '{p}'")
    return adapter
```

If you passed `provider="anthropic"`, it uses that. Otherwise it calls `resolve_provider()` from `capabilities.py`. Let's follow that call.

## Provider Resolution

`capabilities.py` has a `CapabilityResolver` with a `REGISTRY` — a tuple of `(provider, pattern, capabilities)` entries:

```python
REGISTRY = (
    ModelCapabilities(provider="anthropic", pattern="claude", caps=...),
    ModelCapabilities(provider="gemini",    pattern="gemini", caps=...),
    ModelCapabilities(provider="openai",    pattern="gpt",    caps=...),
)
```

`resolve_provider("claude-sonnet-4-5")` lowercases the model name and checks each pattern with `startswith()`. `"claude-sonnet-4-5"` starts with `"claude"` → provider is `"anthropic"`.

That's it. Prefix matching. No regex, no mapping table, no API call. If the model name doesn't match any pattern and isn't in the hydrated catalog, you get `UnsupportedModelError`. This is why fine-tuned models with custom names need `provider="openai"` — their names don't start with a recognized prefix.

The resolver also has a `_model_index` dict that gets populated by `hydrate_with_specs()` — either from the built-in model catalog or from `models.dev`. If a model ID is in the index, it matches by exact name before falling back to prefix matching. This handles models like `o1-preview` that don't start with `gpt-`.

## The Factory

How do adapters get into the client? `build_default()` in `factory.py` wires everything:

```python
def build_default(use_pycurl=True, policy=None, api_key=None, env=None, ...):
    transport = UrlLibTransport(policy=policy or TransportPolicy())
    # try pycurl if requested ...
    
    client = UniversalLM()
    for cls in _CORE_ADAPTERS:  # (OpenAIAdapter, AnthropicAdapter, GeminiAdapter)
        manifest = cls.manifest
        key = ...  # resolve from api_key, env file, or os.environ
        if key:
            client.register(cls(api_key=key, transport=transport))
    return client
```

It iterates over the three core adapter classes, looks for an API key for each one (checking `api_key=` parameter, then `.env` file, then environment variables), and registers the adapter if a key is found. Providers without keys are silently skipped — that's why calling a model from an unconfigured provider gives "no adapter registered" instead of "missing API key."

The key resolution priority — `api_key=` > env file > `os.environ` — is implemented right here, not in a separate auth module. The `auth.py` file handles auth *strategies* (Bearer, Header, Query), but key *resolution* is a factory concern.

The `.env` file parser in `factory.py` is surprisingly thorough. It handles `KEY=VALUE`, `export KEY=VALUE`, quoted values, comments, and blank lines. It also handles `~/.bashrc` and `~/.zshrc` files by tolerating lines that don't match `KEY=VALUE` format. When you pass `env="~/.zshrc"`, lm15 scans every line, picks out the ones that look like `OPENAI_API_KEY=sk-...`, and ignores the rest.

One subtle detail: after parsing the env file for lm15's own keys, `build_default()` also calls `_push_env_file_to_environ()`, which sets all recognized keys into `os.environ`. This is for third-party plugins — they read `os.getenv()` in their own factories, and this ensures they pick up keys from the user's `.env` file.

## The Adapter Protocol

What does an adapter look like, structurally? `protocols.py` defines the `LMAdapter` protocol:

```python
class LMAdapter(Protocol):
    provider: str
    capabilities: Capabilities
    supports: EndpointSupport
    manifest: ProviderManifest
    
    def complete(self, request: LMRequest) -> LMResponse: ...
    def stream(self, request: LMRequest) -> Iterator[StreamEvent]: ...
    def file_upload(self, request: FileUploadRequest) -> FileUploadResponse: ...
    def batch_submit(self, request: BatchRequest) -> BatchResponse: ...
    # ... more optional endpoints
```

The protocol is a structural contract, not an inheritance hierarchy. An adapter doesn't need to inherit from `LMAdapter` — it just needs to have the right attributes and methods. This is Python's Protocol typing: if it walks like an adapter and quacks like an adapter, it is an adapter.

`EndpointSupport` (from `features.py`) declares which endpoints the adapter implements:

```python
@dataclass(slots=True, frozen=True)
class EndpointSupport:
    complete: bool = True
    stream: bool = True
    embeddings: bool = False
    files: bool = False
    batches: bool = False
    # ...
```

`UniversalLM` checks these flags before dispatching. If `adapter.supports.stream` is `False`, calling `stream()` raises `UnsupportedFeatureError` instead of hitting a `NotImplementedError` at the adapter level.

## What the Client Doesn't Do

The client doesn't parse model names beyond resolving the provider. It doesn't manage conversation state. It doesn't retry failed calls (that's middleware). It doesn't read API keys (that's the factory). It doesn't know JSON, HTTP, or any provider's wire format (that's the adapters and transport).

This is the Unix philosophy applied to a Python class: do one thing well. The client routes and dispatches. Everything else is someone else's job. The next chapter follows the request into the adapter — where the real translation work happens.
