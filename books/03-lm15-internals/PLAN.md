# Book 3 — lm15 Internals

**Audience:** Contributors, plugin authors, curious developers who want to understand the machinery — people who open the hood
**Length:** ~45,000 words / ~180 pages
**Prerequisite:** Book 1 (user-level proficiency), comfort reading Python source

---

## Thesis

lm15 is 30 files and zero dependencies. Every design choice is visible, every tradeoff is deliberate, and every abstraction exists because removing it made things worse. This book walks through the entire codebase — not as a reference, but as a story of decisions. Why frozen dataclasses instead of Pydantic? Why urllib instead of httpx? Why five layers instead of three? Each chapter answers a question the reader didn't know they had, and by the end they can debug anything, extend anything, and argue with any design choice from an informed position.

---

## The Through-Line

The reader follows a single request — `lm15.complete("claude-sonnet-4-5", "Hello.")` — from the moment the function is called to the moment the response object is returned. Each chapter picks up where the previous one left off, tracing the request deeper into the stack. By Chapter 8, the reader has seen every layer: the sugar, the types, the client, the adapter, the transport, the wire, and back.

The second half of the book (Chapters 6-10) covers the systems that don't fit the request path: streaming, middleware, discovery, auth, and errors. These are the cross-cutting concerns — they touch every layer but belong to none.

---

## Part I — Following the Request

### Chapter 1: The Surface

**Arc:** The reader calls `lm15.complete()` and we freeze time at the moment of the call. What does this function actually do? It's seven lines in `api.py`. It creates a `Model`, calls it, and returns the response. The "simple function" is sugar over an object, which is sugar over a client, which is sugar over an adapter, which is sugar over urllib. Five layers. This chapter maps them and explains why each one exists.

**Opens with:** The source code of `lm15.complete()`. Seven lines. The reader expects complexity and finds delegation. The function creates a `Model` (from `model.py`), calls it with the prompt, and returns whatever comes back. That's it. The question becomes: what does `Model.__call__` do?

**The five-layer stack:**
```
lm15.complete() / lm15.model()       ← sugar
        │
        ▼
    Model.__call__()                  ← state, tools, history
        │
        ▼
    UniversalLM.complete()            ← routing, middleware
        │
        ▼
    ProviderAdapter.complete()        ← wire format translation
        │
        ▼
    Transport.request()               ← HTTP
```

**Why five and not three:** Every time someone tried to collapse two layers, something got worse. Merge Model and UniversalLM? Then you can't have two models sharing the same transport. Merge Adapter and Transport? Then you can't swap urllib for pycurl. Merge Sugar and Model? Then `complete()` needs state management. The layers exist because the alternative is worse. Show the "what if we merged these" thought experiment for each boundary.

**The module map:** Every `.py` file, one sentence each. Not a reference table — a narrative. "You've seen `api.py` (the sugar) and `model.py` (the state). Next is `client.py`, which knows about providers but not about models. Then `providers/openai.py`, which knows about OpenAI but not about lm15's types until it translates them. Then `transports/urllib_transport.py`, which knows about HTTP and nothing else."

**Sections:**
- Seven lines of sugar (`api.py` source, annotated)
- The five layers (diagram, one paragraph per layer)
- Why five and not three (thought experiments on merging)
- The module map (every file, one sentence)
- Design principles (zero deps, frozen types, nothing hidden)

---

### Chapter 2: Types

**Arc:** The reader opens `types.py` — the longest file in lm15 — and discovers that it's almost the entire conceptual model of the library. Every request, every response, every piece of content is a frozen dataclass defined here. This chapter explains each type, why it's shaped the way it is, and what would break if you changed it.

**Opens with:** A question: how do you represent "a message that might contain text, an image, a tool call, and a thinking block — any combination, in any order"? The answer is `Part`. A single type with a `type` discriminator field. Not a class hierarchy. Not a union. One dataclass with optional fields. The chapter explores why.

**Part — the universal content unit:**
- Why one type instead of `TextPart`, `ImagePart`, `ToolCallPart`, etc.
- The `type` field as discriminator: `"text"`, `"image"`, `"tool_call"`, `"thinking"`, etc.
- The tradeoff: simpler code, weaker type checking. Why lm15 chose simplicity.
- `DataSource` — the sub-object for media: URL vs base64 vs file reference
- Factory methods: `Part.text_part()`, `Part.image()`, `Part.tool_call()` — sugar over the constructor
- The `cache` metadata field — how `cache=True` on a `Part` becomes provider-specific instructions

**Message — role + parts:**
- Why `parts` is a tuple, not a list (immutability, hashability)
- Factory methods: `Message.user("text")`, `Message.assistant("text")`
- The `tool` role — how tool results ride on messages

**Tool — the function schema:**
- Why `parameters` is a raw dict (JSON Schema) instead of a typed structure
- The `builtin` type for provider-side tools like `"web_search"`

**LMRequest and Config:**
- `LMRequest` is the normalized request: model, messages, system, tools, config
- `Config` holds the knobs: temperature, max_tokens, reasoning, provider-specific overrides
- Why `system` is on `LMRequest`, not on `Config` — it's content, not configuration
- Frozen semantics: you build a new request, never mutate one

**LMResponse and Usage:**
- The response is a frozen snapshot — `id`, `model`, `message`, `finish_reason`, `usage`, `provider`
- Convenience properties: `.text`, `.thinking`, `.image`, `.tool_calls`, `.citations` — computed from `message.parts`
- `Usage` — the token counters, including cache and reasoning tokens
- The `provider` escape hatch — raw dict for when you need what lm15 doesn't normalize

**Why frozen dataclasses:**
- Immutability prevents action-at-a-distance bugs
- Hashability enables caching (the local response cache in `Model` uses request-as-key)
- Thread safety for free
- The cost: verbose construction (`LMRequest(model=..., messages=(...,))` instead of `req.model = ...`)
- Why not Pydantic, attrs, or plain dicts

**Sections:**
- The representation problem (how do you model multimodal content?)
- Part and DataSource (the universal content unit)
- Message (role + parts, why tuples)
- Tool (schema as raw JSON, builtin tools)
- LMRequest and Config (the normalized request)
- LMResponse and Usage (the normalized response)
- Why frozen dataclasses (tradeoffs, alternatives rejected)

---

### Chapter 3: The Client

**Arc:** `UniversalLM` is the router. It knows which provider handles which model, maintains the middleware pipeline, and dispatches requests. This chapter walks through `client.py` — the class that ties providers together without knowing anything about their wire formats.

**Opens with:** A puzzle. `lm15.complete("claude-sonnet-4-5", "Hello.")` goes to Anthropic. `lm15.complete("gpt-4.1-mini", "Hello.")` goes to OpenAI. How? Who decides? The reader expects a complex routing table and finds a prefix match.

**Provider resolution:**
- `capabilities.py` — `resolve_provider()` maps model name prefixes to provider names
- `gpt-*` → `"openai"`, `claude-*` → `"anthropic"`, `gemini-*` → `"gemini"`
- The explicit `provider=` override for fine-tuned or custom models
- What happens when no provider matches (and the error you get)

**UniversalLM:**
- `register()` — how adapters are added (at `build_default()` time)
- `complete()` — resolve provider, run middleware, call adapter
- `stream()` — same path, different adapter method, returns iterator
- Why `UniversalLM` doesn't know about `Model` — separation of concerns

**The factory — `build_default()`:**
- `factory.py` — how `build_default()` wires everything together
- Auth resolution: `api_key=` → `env=` → `os.environ`
- Transport selection: urllib by default, pycurl if requested
- Adapter instantiation: one per configured provider
- Plugin discovery: entry points for third-party adapters

**Key insight:** The client is a dispatcher, not a translator. It knows *where* to send a request but not *how* to encode it. That's the adapter's job. This separation is why adding a new provider doesn't touch the client code at all.

**Sections:**
- The routing question (how does the model name become a provider?)
- Provider resolution (`capabilities.py`, prefix matching)
- UniversalLM (register, dispatch, middleware)
- The factory (`build_default()`, wiring)
- Auth resolution (`auth.py`, the key lookup chain)

---

### Chapter 4: The Adapters

**Arc:** The reader opens `providers/openai.py` and sees the translation layer — where lm15's normalized types become OpenAI's JSON, and OpenAI's JSON becomes lm15's types. This is the longest chapter because it covers all three adapters, and the differences between them are where the real complexity lives.

**Opens with:** The same `LMRequest` — model="claude-sonnet-4-5", one user message "Hello." — and what it looks like on the wire for each provider. Three completely different JSON structures. Three different ways to encode the system prompt. Three different tool call formats. The adapter's job is to make this invisible.

**The adapter contract:**
- `complete(request) → LMResponse`
- `stream(request) → Iterator[StreamEvent]`
- `file_upload()`, `batch_submit()` — optional endpoints
- `provider`, `capabilities`, `supports`, `manifest` — identity and capability declaration

**OpenAI adapter walkthrough:**
- `build_request()`: `LMRequest` → OpenAI chat completions body
- System prompt: becomes a `{"role": "system", "content": "..."}` message
- Tool encoding: `Tool` → `{"type": "function", "function": {...}}`
- Message encoding: each `Part` type → OpenAI content blocks
- `parse_response()`: OpenAI JSON → `LMResponse`
- Streaming: SSE parsing, delta accumulation, the `[DONE]` sentinel
- Image generation: the `output="image"` path, DALL-E integration
- Audio: TTS path
- Automatic prompt caching (no-op on lm15 side)

**Anthropic adapter walkthrough:**
- The Messages API: system prompt extracted to top-level `system` field
- Tool encoding: `Tool` → `{"name": ..., "input_schema": {...}}`
- The `tool_use` / `tool_result` block format (different from OpenAI)
- `cache_control` injection: where breakpoints go, the advancing pattern
- Prefill: injecting a partial assistant message
- Extended thinking: `reasoning={"budget": N}` → Anthropic's thinking parameter
- Why Anthropic is the most complex adapter (cache control, thinking, prefill all have special handling)

**Gemini adapter walkthrough:**
- REST API (not the Python SDK): `generateContent` endpoint
- Content parts: lm15 `Part` → Gemini `parts` array
- Tool encoding: `functionDeclarations`
- `CachedContent` API for prompt caching (more complex than Anthropic's)
- File upload: Gemini Files API
- Model name normalization

**Cross-adapter comparison:**
- Where they agree (basic text, tool calls, usage)
- Where they diverge (system prompt placement, cache control, reasoning, streaming format)
- The normalization tax: what lm15 hides and what it can't
- The `provider` escape hatch: when you need raw access

**Sections:**
- The same request, three wires (JSON comparison)
- The adapter contract (protocol, methods, identity)
- OpenAI (build, parse, stream, images, audio)
- Anthropic (system, tools, cache, prefill, thinking)
- Gemini (REST, content mapping, cached content, files)
- Where they diverge (comparison table with examples)
- The normalization tax (what's hidden, what leaks)

---

### Chapter 5: The Transport

**Arc:** At the bottom of the stack, something has to make an HTTP request. lm15 uses Python's stdlib `urllib` — no requests, no httpx, no aiohttp. This chapter explains why, what it costs, and what the pycurl alternative buys you.

**Opens with:** The most controversial design decision in lm15: using `urllib` for HTTP in 2025. Every other Python HTTP library — requests, httpx, aiohttp — is better at HTTP. They have connection pooling, HTTP/2, async, retry logic, progress callbacks. urllib has none of that. So why?

**The answer:** Dependencies. requests brings in urllib3, certifi, charset-normalizer, idna. httpx brings in httpcore, h11, anyio, sniffio, certifi. Each one is a surface for version conflicts, security patches, and install failures. lm15 chose to be worse at HTTP in exchange for being impossible to break by installing.

**urllib transport (`urllib_transport.py`):**
- `request()` — build a `urllib.request.Request`, call `urlopen`, read the response
- Headers, timeouts, SSL context
- Error mapping: HTTP status codes → lm15 error types
- Why there's no connection pooling (and when it matters)

**Streaming with urllib:**
- Chunked transfer encoding: read line by line from the response body
- The SSE parser (`sse.py`): `data:` lines, `event:` lines, empty-line delimiters
- Provider differences in SSE format
- The challenge: partial chunks, multi-line data fields, keep-alive comments

**pycurl transport (`pycurl_transport.py`):**
- Optional — only loaded if pycurl is installed
- Connection reuse, HTTP/2, better streaming performance
- Same interface as urllib transport — the `Transport` protocol
- When to use it: high-throughput, long-lived connections, latency-sensitive applications

**TransportPolicy:**
- `timeout`, `connect_timeout`, `read_timeout`, `max_retries`
- How policy flows from `build_default()` through the client to the transport

**Key insight:** The transport is the most boring layer and that's the point. It does one thing — send bytes, receive bytes — and every interesting decision happens above it. The abstraction exists so you can swap urllib for pycurl (or something custom) without touching any other layer.

**Sections:**
- The urllib decision (why, what it costs, what it buys)
- urllib transport (request, response, error mapping)
- SSE parsing (the line protocol, provider differences)
- pycurl transport (when and why)
- The Transport protocol (the interface, swappability)
- TransportPolicy (timeouts, retries)

---

## Part II — Cross-Cutting Concerns

### Chapter 6: Streaming Internals

**Arc:** The reader has seen streaming from the user side (Book 1 Chapter 5) and from the transport side (Chapter 5 of this book). This chapter connects them — how raw SSE bytes become `StreamChunk` objects with typed events.

**Opens with:** The `Stream` class in `stream.py`. It's an iterator that wraps the adapter's `StreamEvent` iterator and translates low-level events into the high-level `StreamChunk` objects the user sees. The chapter walks through the translation.

**The event pipeline:**
```
Transport (bytes) → SSE parser (lines) → Adapter (StreamEvent) → Stream (StreamChunk)
```

**StreamEvent → StreamChunk translation:**
- `start` → captured (id, model), not yielded
- `delta` with `text` → `StreamChunk(type="text", text=...)`
- `delta` with `thinking` → `StreamChunk(type="thinking", text=...)`
- `delta` with `tool_call` → accumulated, yielded as `StreamChunk(type="tool_call")`
- `end` → materialized into `LMResponse`, yielded as `StreamChunk(type="finished")`

**Response materialization:**
- The `_materialize_response()` method: collect all accumulated parts, build a `Message`, build an `LMResponse`
- Why materialization happens at stream end, not incrementally
- The `stream.response` property and its lazy consumption

**Tool call accumulation:**
- Tool call deltas arrive as JSON fragments: `{"ci` then `ty":` then `"Paris"}`
- `_parse_json_best_effort()` — try to parse, fall back to partial
- The `_tool_call_raw` and `_tool_call_meta` dicts: accumulating across deltas

**The `on_finished` callback:**
- How `Model.stream()` hooks into `Stream` to record history
- Why the callback exists (the Model needs to know the final response without owning the Stream)

**Sections:**
- The event pipeline (four stages, transport to user)
- StreamEvent to StreamChunk (the translation table)
- Response materialization (how streaming becomes a response)
- Tool call accumulation (parsing JSON fragments)
- The on_finished hook (Stream ↔ Model integration)

---

### Chapter 7: Middleware

**Arc:** Between the client and the adapter, there's a pipeline of middleware that can intercept, modify, cache, or retry requests. This chapter explains the middleware system and shows how to write custom middleware.

**Opens with:** A question. Where do retries happen? Not in the adapter — it shouldn't know about retry policy. Not in the transport — it shouldn't know about requests. Not in the Model — it shouldn't know about HTTP. Retries happen in middleware — a layer that wraps the adapter call and can intercept it, retry it, cache it, or log it.

**The middleware contract:**
- Signature: `(request, next_fn) → response`
- `next_fn` calls the next middleware in the chain, or the adapter if it's the last one
- Ordering: middleware runs in list order, each wrapping `next_fn`

**Built-in middleware:**
- `with_cache(cache_dict)` — request-keyed response caching
- `with_history(history_list)` — append request/response pairs to a list
- `with_retries(max_retries, sleep_base)` — exponential backoff on transient errors
- Walk through each one's source — they're 5-15 lines each

**Writing custom middleware:**
- Example: logging middleware (model, tokens, latency per call)
- Example: token budget middleware (abort if cumulative tokens exceed limit)
- Example: model fallback (catch error, retry with different model)
- How to register middleware on `UniversalLM`

**Composition order:**
- Cache before retries? After? Why order matters.
- The general principle: transformations outside, side effects inside

**Sections:**
- The retry question (where does it belong?)
- The middleware contract (signature, chaining)
- Built-in middleware (cache, history, retries — source walkthrough)
- Writing your own (three examples with full code)
- Composition and ordering (why it matters, the general principle)

---

### Chapter 8: Discovery

**Arc:** `lm15.models()` returns a list of every model available across all configured providers. This chapter explains how that list is built — live API queries, fallback catalogs, models.dev hydration, caching, and the merge strategy.

**Opens with:** The reader calls `lm15.models()` and gets 82 results in 1.3 seconds. Where did they come from? Some from live API queries (each provider has a list-models endpoint). Some from a built-in fallback catalog (for when the API is unreachable). Some from models.dev hydration (a third-party catalog of model capabilities). The merge strategy is non-trivial.

**Live discovery:**
- Each adapter implements a list-models endpoint
- OpenAI: `GET /v1/models`
- Anthropic: hardcoded catalog (Anthropic doesn't have a list endpoint)
- Gemini: `GET /v1beta/models`
- Timeout handling: if a provider doesn't respond in 3 seconds, skip it

**The fallback catalog (`model_catalog.py`):**
- A hardcoded dict of known models with capabilities
- Used when live discovery fails or for providers without list endpoints
- Updated with each lm15 release
- Why it exists: Anthropic has no list-models API; providers go down; offline development

**ModelSpec — the normalized model metadata:**
- `id`, `provider`, `context_window`, `max_output`
- `input_modalities`, `output_modalities`
- `tool_call`, `structured_output`, `reasoning`
- `raw` — the provider's original model object

**The merge strategy:**
- Live data takes priority over fallback catalog
- Fallback fills in models that live discovery missed
- models.dev enriches with capability metadata (context window, modalities)
- Deduplication by (provider, model_id)

**Filtering:**
- `supports={"tools", "reasoning"}` — capability filters
- `input_modalities={"image"}` — modality filters
- `provider="openai"` — provider filter
- How filters compose (intersection)

**Caching and refresh:**
- Results cached in memory after first call
- `refresh=True` forces a new query
- TTL semantics (or lack thereof — manual refresh only)

**Sections:**
- The 82 models question (where do they come from?)
- Live discovery (each provider's endpoint)
- The fallback catalog (why, when, how)
- ModelSpec (the normalized type)
- The merge strategy (priority, deduplication, enrichment)
- Filtering (capabilities, modalities, provider)
- Caching and refresh

---

### Chapter 9: Auth

**Arc:** Somewhere between "the user has an API key" and "the HTTP request includes an Authorization header," a lot of small decisions happen. This chapter covers the auth resolution chain and the `.env` file parser.

**Opens with:** The three ways to provide a key — `api_key=`, `env=`, environment variables — and the priority between them. The reader expects a simple lookup and discovers a file parser that handles `.env` files, `.bashrc` files, quoted values, `export` prefixes, comments, and edge cases.

**The priority chain:**
1. `api_key=` parameter (explicit, wins always)
2. `env=` file (parsed, keys set into `os.environ`)
3. `os.environ` (ambient, checked last)

**The file parser:**
- `KEY=VALUE`, `export KEY=VALUE`, `KEY="VALUE"`, `KEY='VALUE'`
- Comments (`#`), blank lines, inline comments
- Shell config files (`~/.bashrc`, `~/.zshrc`) work because the parser tolerates non-KEY=VALUE lines
- Edge cases: keys with `=` in the value, values with spaces, values with quotes inside quotes

**Per-provider key names:**
- `ProviderManifest.env_keys` — each provider declares what variable names it uses
- OpenAI: `("OPENAI_API_KEY",)`
- Gemini: `("GEMINI_API_KEY", "GOOGLE_API_KEY")` — two names, first match wins

**The `api_key` dict form:**
- `api_key={"openai": "sk-...", "anthropic": "sk-ant-..."}` — per-provider keys
- How the dict is distributed to adapters during `build_default()`

**Sections:**
- The priority chain (explicit → file → environment)
- Parsing `.env` files (formats, edge cases)
- Per-provider key names (manifests, multiple names)
- The dict form (per-provider keys)

---

### Chapter 10: Errors

**Arc:** lm15 has a typed error hierarchy. This chapter explains why each error class exists, how HTTP status codes map to error types, and how errors propagate through the five-layer stack.

**Opens with:** A provider returns HTTP 429 with a JSON body `{"error": {"message": "Rate limit exceeded"}}`. By the time this reaches user code, it's a `RateLimitError("Rate limit exceeded")`. What happened in between?

**The error hierarchy:**
```
ULMError
├── TransportError (network failures)
└── ProviderError (API failures)
    ├── AuthError (401, 403)
    ├── BillingError (402)
    ├── RateLimitError (429)
    ├── InvalidRequestError (400, 404, 422)
    │   └── ContextLengthError
    ├── TimeoutError (408, 504)
    ├── ServerError (5xx)
    ├── UnsupportedModelError
    ├── UnsupportedFeatureError
    └── NotConfiguredError
```

**The mapping pipeline:**
- Transport catches HTTP errors, extracts status code and body
- `map_http_error(status, message)` maps status → error class
- Adapter's `normalize_error()` extracts the human-readable message from provider-specific JSON
- Error is raised through middleware, client, model, sugar — each layer can catch or propagate

**Canonical error codes:**
- `canonical_error_code(error)` → `"auth"`, `"rate_limit"`, `"timeout"`, etc.
- `error_class_for_canonical_code(code)` → the inverse
- Why these exist: middleware and plugins need provider-agnostic error classification

**Error propagation through streaming:**
- Stream errors arrive as `StreamEvent(type="error")` instead of exceptions
- The `Stream` class converts them to exceptions
- What happens when a stream fails mid-way (partial response + exception)

**Retryability:**
- `Model._is_retryable_error()` — `RateLimitError`, `TimeoutError`, `ServerError`, `TransportError`
- Why `AuthError` and `InvalidRequestError` are not retryable
- How `retries=` uses this classification

**Sections:**
- From HTTP 429 to RateLimitError (the full path)
- The error hierarchy (tree, one paragraph per class)
- The mapping pipeline (status → class → raise)
- Canonical error codes (why, how)
- Errors in streaming (partial responses)
- Retryability (which errors retry, which don't)

---

## Chapter Rhythm

| Ch | Title | Words | Through-line position |
|---|---|---|---|
| 1 | The Surface | 4,500 | The call enters: `complete()` → Model → Client → Adapter → Transport |
| 2 | Types | 5,500 | The data: what travels between layers |
| 3 | The Client | 4,000 | The router: how the request finds its provider |
| 4 | The Adapters | 6,000 | The translators: lm15 types ↔ wire JSON |
| 5 | The Transport | 4,000 | The bottom: HTTP, SSE, urllib vs pycurl |
| 6 | Streaming Internals | 4,000 | Cross-cut: bytes → events → chunks → response |
| 7 | Middleware | 4,000 | Cross-cut: intercept, retry, cache, log |
| 8 | Discovery | 4,500 | Cross-cut: what models exist, how lm15 knows |
| 9 | Auth | 3,500 | Cross-cut: keys, files, priority |
| 10 | Errors | 4,000 | Cross-cut: failures, classification, propagation |
| | **Total** | **~44,000** | |

---

## Design Principles

**Follow the request.** Part I is a single request traced through every layer. The reader never loses the thread — they always know where they are in the stack and why.

**Show the source.** This is a book about 30 Python files. The reader should see real source code — not pseudocode, not abstractions of abstractions. Quote the actual lines from `types.py`, `client.py`, `stream.py`. The code is short enough to show.

**Explain the decisions, not just the code.** "Here's what the code does" is documentation. "Here's why it does it this way instead of the three other ways we considered" is a book. Every chapter has at least one "why not X?" section.

**Honest about tradeoffs.** urllib is worse at HTTP than httpx. Frozen dataclasses are more verbose than mutable ones. One `Part` type with a discriminator is less type-safe than a class hierarchy. The book names the costs, not just the benefits.

**No forward references.** Each chapter stands on the previous one. The reader never encounters "we'll explain this in Chapter 8" — by the time they need to understand something, they already do.
