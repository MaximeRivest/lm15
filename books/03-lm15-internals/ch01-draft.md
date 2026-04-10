# Chapter 1: The Surface

Here is the complete source code of `lm15.complete()`:

```python
def complete(
    model_name: str,
    prompt: str | list[str | Part] | None = None,
    *,
    messages=None,
    system: str | None = None,
    tools=None,
    reasoning=None,
    prefill: str | None = None,
    output: str | None = None,
    prompt_caching: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    top_p: float | None = None,
    stop=None,
    provider: str | None = None,
    api_key: str | dict[str, str] | None = None,
    env: str | None = None,
):
    m = model(model_name, provider=provider, prompt_caching=prompt_caching,
              system=system, api_key=api_key, env=env)
    return m(
        prompt, messages=messages, tools=tools, reasoning=reasoning,
        prefill=prefill, output=output, prompt_caching=prompt_caching,
        temperature=temperature, max_tokens=max_tokens,
        top_p=top_p, stop=stop, provider=provider,
    )
```

That's it. The function that Book 1 spent ten chapters teaching you to use — the function that talks to OpenAI, Anthropic, and Gemini, handles tools, streaming, reasoning, caching, retries, and multimodal content — is a wrapper. It creates a `Model` object and calls it. One line of construction, one line of delegation.

This is either underwhelming or illuminating, depending on how you think about software. The function looks simple because it *is* simple. The complexity isn't here — it's in the `Model` it creates, the `UniversalLM` the Model delegates to, the `ProviderAdapter` the UniversalLM dispatches to, and the `Transport` the adapter uses to make HTTP calls. Five layers, each doing one thing, each ignorant of the others' internals.

This book traces a single call — `lm15.complete("claude-sonnet-4-5", "Hello.")` — through every layer, from the moment you call it to the moment the response object lands in your hands. This chapter maps the layers. The next nine chapters walk through them.

## The Five Layers

Here's what happens when you call `complete()`, step by step, layer by layer:

```
lm15.complete("claude-sonnet-4-5", "Hello.")
    │
    │  api.py — creates a Model, calls it
    ▼
Model.__call__("Hello.")
    │
    │  model.py — builds LMRequest, manages history, auto-executes tools
    ▼
UniversalLM.complete(request)
    │
    │  client.py — resolves provider, runs middleware, dispatches
    ▼
AnthropicAdapter.complete(request)
    │
    │  providers/anthropic.py — translates LMRequest → Anthropic JSON,
    │                           sends HTTP, parses response → LMResponse
    ▼
UrlLibTransport.request(http_request)
    │
    │  transports/urllib_transport.py — sends bytes, receives bytes
    ▼
  HTTP → api.anthropic.com → HTTP response
```

Five layers. Let me describe what each one does and — more importantly — what it doesn't do.

**Layer 1: The sugar (`api.py`).** `lm15.complete()` and `lm15.stream()` live here. They accept user-friendly arguments (`prompt`, `system=`, `tools=`, `env=`), create a `Model` object, and call it. They don't manage state, don't know about providers, don't touch HTTP. They exist so you can write `lm15.complete(...)` instead of constructing a `Model` every time.

**Layer 2: The model (`model.py`).** The `Model` class manages conversation history, tool auto-execution, per-call overrides, retries, and local caching. When you call `model("Hello.")`, it builds an `LMRequest` (the normalized request type), passes it to `UniversalLM`, gets back an `LMResponse`, and records both in history. It's the statefulness layer — everything above it is stateless sugar, everything below it is stateless infrastructure.

**Layer 3: The client (`client.py`).** `UniversalLM` is the router. It holds a dict of registered adapters (`{"openai": OpenAIAdapter, "anthropic": AnthropicAdapter, ...}`), resolves which adapter handles which model, runs the middleware pipeline, and dispatches the request. It doesn't know how to talk to any provider — it just knows which adapter to ask.

**Layer 4: The adapter (`providers/*.py`).** Each adapter translates between lm15's types and one provider's wire format. `AnthropicAdapter.complete()` receives an `LMRequest`, builds the JSON body that Anthropic's API expects, sends it via the transport, parses Anthropic's JSON response, and returns an `LMResponse`. The adapter knows everything about Anthropic's API and nothing about OpenAI's. There are three adapters — one per provider — and they share no code beyond common utilities.

**Layer 5: The transport (`transports/*.py`).** The transport makes HTTP requests. That's all. It takes a URL, headers, and a body. It returns a status code, headers, and a response body. It doesn't know what an LLM is. It doesn't know what JSON is (though it can serialize it). It's `urllib.request.urlopen` with timeout handling and error mapping.

## Why Five Layers

Five feels like a lot for a library that prides itself on simplicity. Why not three? Why not one?

I can answer by showing what happens if you try to collapse them.

**Merge sugar and model.** Now `lm15.complete()` needs to manage conversation history — but it's supposed to be a stateless function. You'd need a global model registry, or you'd lose the stateless API entirely. The sugar exists so that `complete()` can be a pure function call while `model()` handles state.

**Merge model and client.** Now the `Model` class needs to know about provider routing and middleware. Every model object carries the routing table and middleware pipeline. You can't have two models sharing the same transport without duplicating the client. The separation means you can create ten model objects that all share one `UniversalLM` (and therefore one set of HTTP connections).

**Merge client and adapter.** Now the client needs to know how to talk to every provider. Adding Mistral means modifying the client. The adapter separation is what makes the plugin system possible — you can add providers without touching core.

**Merge adapter and transport.** Now the OpenAI adapter includes its own HTTP code. You can't swap urllib for pycurl without modifying every adapter. You can't add a logging proxy without touching provider code. The transport abstraction means HTTP concerns are handled once, not three times.

Each layer boundary exists because collapsing it made something worse. That's not always true in software — many libraries have unnecessary layers. But in lm15, each one pays for itself.

## The Module Map

lm15 is 30 Python files. Here's every one, grouped by layer:

**The sugar** (what users import):
- `__init__.py` — re-exports the public API, sets up version info
- `api.py` — `complete()`, `stream()`, `model()`, `upload()`, `models()`, `providers_info()`

**The model** (state and orchestration):
- `model.py` — `Model` class, tool auto-execution, history, `with_*` derivation, `callable_to_tool()`
- `stream.py` — `Stream` class, `StreamChunk`, event-to-chunk translation, response materialization

**The client** (routing and middleware):
- `client.py` — `UniversalLM`, adapter registration, provider dispatch
- `middleware.py` — `MiddlewarePipeline`, `with_cache`, `with_history`, `with_retries`
- `factory.py` — `build_default()`, transport selection, adapter wiring, env file parsing
- `capabilities.py` — `CapabilityResolver`, provider resolution from model names
- `protocols.py` — `LMAdapter` protocol, `Capabilities`, `LiveSession`

**The adapters** (provider translation):
- `providers/openai.py` — OpenAI chat completions, images, audio, streaming
- `providers/anthropic.py` — Anthropic Messages API, cache control, thinking
- `providers/gemini.py` — Gemini generateContent, cached content, files
- `providers/base.py` — shared adapter base class
- `providers/common.py` — cross-adapter utilities

**The transport** (HTTP):
- `transports/base.py` — `Transport` base, `HttpRequest`, `HttpResponse`, `TransportPolicy`
- `transports/urllib_transport.py` — stdlib HTTP transport
- `transports/pycurl_transport.py` — optional high-performance transport

**The types** (data):
- `types.py` — every dataclass: `Part`, `Message`, `Tool`, `LMRequest`, `LMResponse`, `Usage`, `StreamEvent`, and more

**Support systems** (cross-cutting):
- `errors.py` — error hierarchy, HTTP status mapping
- `auth.py` — auth strategies (`BearerAuth`, `HeaderKeyAuth`, `QueryKeyAuth`)
- `features.py` — `EndpointSupport`, `ProviderManifest`
- `discovery.py` — live model listing, provider status, merge logic
- `model_catalog.py` — `ModelSpec`, fallback catalog, models.dev fetching
- `plugins.py` — entry point discovery, plugin loading
- `sse.py` — SSE line parser
- `repl.py` — REPL error formatting

That's everything. No hidden modules, no generated code, no metaprogramming. You can read the entire library in an afternoon — and by the end of this book, you will have.

## Three Design Principles

Before we dive into the layers, three principles that explain recurring patterns throughout the codebase:

**Zero dependencies.** lm15 imports nothing outside the standard library. HTTP is `urllib`. Data types are `dataclasses`. JSON is `json`. The argument isn't that `urllib` is better than `httpx` — it's worse. The argument is that every dependency is a surface for version conflicts, security patches, and install failures, and lm15's value proposition includes "never breaks because something else changed." The cost is that HTTP operations are more verbose and less capable than they'd be with a real HTTP library. The benefit is that `pip install lm15` will work on every platform, in every environment, for the foreseeable future.

**Frozen dataclasses all the way down.** `LMRequest`, `LMResponse`, `Message`, `Part`, `Tool`, `Config`, `Usage` — every data type is a frozen dataclass. You can read fields but not modify them. This prevents action-at-a-distance bugs (nobody mutates a shared request), enables caching (frozen objects are hashable), and provides thread safety for free. The cost is verbose construction — you build new objects instead of modifying existing ones. Chapter 2 covers the types in detail.

**Nothing is hidden.** Every type in `types.py` is importable. The `provider` field on `LMResponse` contains the raw JSON from the provider. The `UniversalLM` class is public. You can instantiate adapters directly, skip the sugar entirely, and wire everything by hand. lm15 is not a black box with a narrow API — it's a transparent box with a convenient API on top. The convenience API (Book 1) is what most people use. The transparent internals (this book) are what you reach for when the convenience isn't enough.

## What You've Seen

We've mapped the territory. You know the five layers, why each one exists, and where every file fits. The next chapter opens `types.py` — the longest file in the library, and arguably the most important. It defines the data model that every layer speaks. Once you understand the types, you understand the contracts between layers, and the rest of the book falls into place.
