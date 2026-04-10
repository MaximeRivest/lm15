# Book 6 — Modeling LLM Interactions

**Audience:** Software designers and senior developers who think about
abstractions — people who build libraries, design APIs, or choose between
frameworks. They may never use lm15, but they think about the same problems.
**Length:** ~55,000 words / ~220 pages **Prerequisite:** Experience calling at
least one LLM API. No lm15 knowledge required. **Tone:** Design book. Think *A
Philosophy of Software Design*, *Domain-Driven Design*, or *Designing
Data-Intensive Applications* — but for LLM integration.

---

## Thesis

Every application that talks to a language model must solve the same modeling
problems: what is a message, what is a conversation, what is a tool, how does
streaming work, where do provider differences belong, and what should be hidden
from the caller. These are design problems with real tradeoffs — and most
libraries get several of them wrong, because the problem space is newer than the
solutions. This book examines each problem, maps the solution space, shows what
lm15 chose and what it gave up, and equips you to make your own decisions —
whether you're picking a library, building one, or wrapping an API for your
team.

---

## Why This Book Exists

There are hundreds of LLM wrapper libraries. They all solve the same eight
problems. They all make different tradeoffs. And almost none of them explain
*why* — why this representation, why this abstraction boundary, why this thing
is hidden and that thing is exposed. The result is that users pick libraries
based on GitHub stars and README aesthetics, not on whether the library's
modeling decisions match their needs.

This book makes the decisions visible. Each chapter is a design problem. Each
problem has a solution space with real options. lm15 is one point in that space
— a case study, not a sales pitch. The reader finishes each chapter able to
evaluate any library's answer to the same question, or to design their own.

---

## Part I — Representing Content

### Chapter 1: What Is a Message?

**The problem:** An LLM interaction is a sequence of messages. But what is a
message? It's not a string — it can contain images, audio, tool calls, thinking
traces, and citations alongside text. It's not a dict — you need type safety and
portability. It's not a class hierarchy — you need to serialize it, compare it,
and pass it between providers.

**The solution space:**
- **String-in, string-out** (the simplest model). Works until you need images.
  Every library starts here and outgrows it.
- **Role + content string** (OpenAI's original shape). `{"role": "user",
  "content": "Hello."}`. Works until content isn't just text.
- **Role + content blocks** (OpenAI/Anthropic's current shape). Content is a
  list of typed blocks — text, image, tool_use, etc. Flexible, but the block
  types are provider-specific.
- **Role + parts** (lm15's approach). A tuple of `Part` objects with a `type`
  discriminator. Provider-agnostic. One type for all content.

**The deep question:** Should a message be a *container* (role + bag of content)
or a *statement* (a typed utterance like UserTextMessage,
AssistantToolCallMessage)? Containers are flexible and composable. Statements
are precise and self-documenting. Most libraries start with statements and end
up with containers as the modalities multiply.

**The tradeoff lm15 made:** One `Part` type with a discriminator field instead
of a class hierarchy. This buys simplicity (one serialization path, one
iteration pattern, cross-provider portability) and costs type safety (nothing
prevents `Part(type="text", source=DataSource(...))` at compile time). The
`__post_init__` validation catches *some* mistakes at runtime, but it's not the
same as the compiler catching them.

**What would be different with hindsight:** The `Part` type has 12 optional
fields. Most are `None` for any given instance. This is a code smell in most
contexts — it suggests the type is doing too many jobs. But the alternative (12
separate types) creates a combinatorial explosion in adapter code. The smell is
real; the alternative is worse. This is a recurring theme in LLM modeling: the
"clean" solution doesn't survive contact with three providers.

**Sections:**
- The representation problem (what is content, really?)
- Four approaches (string, role+string, role+blocks, role+parts)
- The discriminator pattern vs class hierarchies
- The portability constraint (content must travel between providers)
- Where this breaks down (provider-specific content types that don't map)
- What other libraries chose (LangChain's message types, LiteLLM's dict
  approach, Vercel AI SDK's parts)

---

### Chapter 2: The Conversation Problem

**The problem:** LLMs have no memory. Every API call sends the full conversation — system prompt, all previous turns, the new message — and the model generates a response as if seeing everything for the first time. "Conversation" is a fiction maintained by the caller. How should this fiction be modeled?

**The solution space:**
- **Explicit messages** (you build the list yourself). Maximum control, maximum boilerplate. This is what the raw APIs give you.
- **Append-only history** (a model object that accumulates turns). Convenient, but the list grows without bound, and you lose control over what's in it.
- **Managed context** (the library decides what to keep and what to drop). Sophisticated, but opaque — you don't know what the model sees.
- **External memory** (RAG, vector stores, scratchpads). The conversation is an index, not a list.

**The deep question:** Who owns the conversation state? If the library owns it, the user can't inspect or modify it without reaching into internals. If the user owns it, the library can't optimize it (caching, summarization, truncation). The boundary between "library manages state" and "user manages state" is the most consequential design decision in an LLM wrapper.

**The tradeoff lm15 made:** The `Model` object owns conversation state (`self._conversation`), but history is inspectable (`model.history`) and clearable (`model.history.clear()`). The library doesn't manage context size — it sends everything every time, and if the context overflows, the provider returns an error. This is honest but expensive: the user pays quadratically for long conversations unless they enable prompt caching or manually clear history.

**What other libraries chose differently:**
- LangChain's memory system abstracts conversation into pluggable backends (buffer, summary, vector). Flexible but complex — the abstraction has more surface area than the problem.
- Anthropic's SDK leaves conversation entirely to the user. No history management, no accumulation. The user builds the message list.
- Vercel AI SDK uses a `messages` array that the UI framework manages. The conversation state lives in React state, not in the SDK.

**The context window as a design constraint:** Every decision about conversation modeling is shaped by the context window. A 4K window demands summarization. A 200K window makes "send everything" viable. A 1M window makes it almost free. As windows grow, the simplest approach (send everything, cache the prefix) becomes the best approach. lm15 bet on large windows; that bet is paying off.

**Sections:**
- The fiction of memory (models don't remember; callers simulate it)
- Four approaches to managing the fiction
- Who owns the state? (library vs user, the spectrum)
- The context window as a moving constraint
- Prompt caching as a solution to quadratic cost
- When "send everything" wins and when it doesn't

---

### Chapter 3: Modeling Tool Use

**The problem:** Tool calling is a three-step dance — the model requests a call, the application executes it, and the result goes back to the model. This is the most complex interaction pattern in LLM APIs, because it crosses the boundary between the model's world (text, JSON) and the application's world (functions, side effects, errors). How you model this dance determines whether tools are easy to use, safe to deploy, and composable with other features.

**The solution space:**
- **Schema-first** (you define a JSON Schema, the model fills it, you execute). This is what the raw APIs give you. Maximum control, maximum boilerplate.
- **Function-first** (you pass a Python function, the library infers the schema, handles execution). Magic, but what do you do about errors? Side effects? Async?
- **Hybrid** (auto-execute simple functions, manual control for complex ones). Two code paths, but each path is clean.

**The deep question:** Should the library execute tools, or should the caller? If the library executes, it owns the side effects — file writes, database calls, API requests. The caller loses control over when and how these happen. If the caller executes, the library returns the tool call and waits — but now the caller needs to manage the conversation state between the call and the result, which means the library needs stateful objects (model objects, not functions).

**This is why `lm15.model()` exists.** Not for conversation history — that's a secondary benefit. The model object exists because manual tool calling requires state between the tool call and the result submission. `submit_tools()` needs to know the conversation so far, the pending tool calls, and the tool definitions. A stateless function can't do this. The design of tool use drove the design of the entire Model class.

**The schema inference problem:** When you pass `tools=[get_weather]`, lm15 reads the function's name, docstring, type hints, and defaults to build a JSON Schema. This is convenient and fragile. `str` → `"string"` works. `list[dict[str, int]]` doesn't map cleanly. Complex types need explicit schemas. The inference is deliberately simple — it handles the 80% case and forces the 20% to use `Tool` objects with explicit schemas.

**The multi-hop problem:** The model might call a tool, read the result, call another tool, read that result, and then answer. With auto-execute, lm15 loops up to 8 times. Who chose 8? Why not 3? Why not unlimited? The limit is a safety mechanism — a confused model that keeps calling the same tool would loop forever without it. But 8 is arbitrary. The right number depends on the task, and there's no way for the library to know the task.

**The error boundary:** What happens when a tool function raises an exception? The tool call fails. The model gets no result. It's as if the tool doesn't exist. This is wrong — the model should learn that the tool failed and why, so it can try a different approach. lm15's guidance (return error strings, don't raise) is a convention, not an enforcement. A library that enforced this — catching exceptions and returning them as error results — would be more reliable but more opinionated.

**Sections:**
- The three-step dance (request, execute, result)
- Who executes? The control problem.
- Why tool use drove the design of Model objects
- Schema inference: the 80/20 line
- Multi-hop loops: bounding the unbounded
- Error boundaries: exceptions vs error returns
- Tools + streaming: when the tool call arrives mid-stream
- What happens when tool schemas diverge between providers

---

## Part II — Bridging Providers

### Chapter 4: The Normalization Question

**The problem:** OpenAI, Anthropic, and Gemini have different APIs. Different JSON structures, different authentication, different error formats, different streaming protocols, different tool call encodings. A multi-provider library must normalize these differences. But how much normalization? Normalize too little and the user deals with three APIs. Normalize too much and you lose provider-specific features.

**The deep question:** Is a universal LLM API even possible? Or are provider differences deep enough that normalization is a lie — a thin coat of paint over fundamentally different systems?

**The spectrum of normalization:**
- **Pass-through** (each provider has its own API in the library). No normalization. The user writes provider-specific code. This is what the raw SDKs give you.
- **Structural normalization** (common types, different behavior). Same request/response types, but features like caching work differently on each provider. This is where lm15 sits.
- **Behavioral normalization** (common types, same behavior). The library hides all differences. Features work identically regardless of provider. This is the promise of LiteLLM.
- **Lowest-common-denominator** (only expose what all providers support). Simple and honest, but you lose 60% of what modern models can do.

**Where normalization works:** Basic text completion. Tool call round-trips. Token usage reporting. Error classification. These are structurally similar across providers, and normalizing them is straightforward and valuable.

**Where normalization lies:** Prompt caching (three completely different mechanisms). Reasoning (different parameters, different semantics). Image generation (different endpoints, different capabilities). Audio (different formats, different APIs). When lm15 normalizes these, it creates the impression of uniformity while the behavior differs. `prompt_caching=True` means something different on each provider. Is that normalization or deception?

**The escape hatch pattern:** lm15's answer is the `provider` dict on both `Config` and `LMResponse`. Anything that doesn't normalize goes in the escape hatch. The universal type stays clean; provider-specific features are still accessible. The tradeoff: escape hatch access is untyped and undocumented. You need to read the provider's docs and the adapter's source to know what keys are available.

**Sections:**
- The normalization spectrum (pass-through to lowest-common-denominator)
- Where normalization is honest (text, tools, errors)
- Where normalization deceives (caching, reasoning, generation)
- The escape hatch pattern (controlled leakage)
- Case study: modeling prompt caching across three providers
- Is a universal LLM API possible? (the honest answer)

---

### Chapter 5: The Adapter Pattern

**The problem:** You need a translation layer between universal types and provider wire formats. How do you structure it? What is the contract? How do you make it extensible without making it fragile?

**The design decisions:**
- **Protocol vs inheritance.** lm15 uses a Protocol (structural typing) instead of an abstract base class. An adapter doesn't inherit from anything — it just needs the right methods and attributes. This is more Pythonic and more flexible (any object that satisfies the protocol is an adapter), but it's also less discoverable (there's no base class to subclass and fill in).
- **One adapter per provider vs one adapter per model.** lm15 has one adapter per provider (OpenAIAdapter handles all `gpt-*` models). The alternative — one adapter per model — would allow model-specific behavior but create hundreds of classes. The per-provider approach is simpler but means model-specific quirks are handled with conditionals inside the adapter.
- **Translation vs delegation.** The adapter translates `LMRequest` → JSON and JSON → `LMResponse`. It could instead delegate to the provider's SDK (call `openai.chat.completions.create()` internally). Translation is more work but avoids SDK dependencies. Delegation is easier but inherits the SDK's bugs, version constraints, and abstractions.

**The hardest translation: tool calls.** Each provider encodes tool calls differently. OpenAI uses `tool_calls` in the assistant message with `function.arguments` as a JSON string. Anthropic uses `tool_use` content blocks with `input` as a dict. Gemini uses `functionCall` parts. The adapter must translate all three into lm15's `Part(type="tool_call", id=..., name=..., input=...)` — and then translate tool *results* back into each provider's format. This round-trip is where most adapter bugs live.

**The streaming translation problem:** Streaming adds a dimension. The adapter must parse SSE events in the provider's format, accumulate partial data (especially tool call JSON fragments), and yield `StreamEvent` objects in real time. The adapter can't buffer the entire stream (that would defeat streaming) and can't yield incomplete events (that would confuse the `Stream` class). The boundary between "enough data to yield" and "need more data" is the hardest judgment call in adapter implementation.

**Sections:**
- Protocol vs abstract base class (structural vs nominal typing)
- One per provider vs one per model (granularity)
- Translation vs delegation (SDK wrapping vs raw HTTP)
- The tool call round-trip (the hardest translation)
- Streaming translation (partial data, accumulation, yield timing)
- The extension story (entry points, plugin discovery, `lm15-x-*`)

---

### Chapter 6: Streaming as a Design Problem

**The problem:** A blocking API call is simple — you send a request and get a response. Streaming is fundamentally more complex: you get partial data over time, you need to handle errors mid-stream, and you need to decide what the user sees (raw events? filtered text? both?).

**The design question that shapes everything:** Is a stream an *iterator* or a *callback*?

- **Iterator** (Python generator, `for event in stream`). The user pulls events. Backpressure is natural. The code reads like synchronous code. But you can't process events in the background while doing other work.
- **Callback** (`stream.on_text(lambda text: ...)`, `stream.on_tool_call(...)`). The library pushes events. Easy to compose with UIs. But callback hell is real, error handling is awkward, and ordering is implicit.
- **Async iterator** (`async for event in stream`). Best of both worlds — pull-based but non-blocking. But requires async throughout the call stack, which lm15 doesn't support (zero deps means no async runtime).
- **Observable** (RxPY-style). Powerful composition operators. Massive conceptual overhead. Nobody wants to learn reactive programming to print "hello."

**lm15 chose iterators.** The `Stream` class is a synchronous iterator. `for event in stream` pulls events one at a time. The `.text` property is a filtered generator that yields only text. This is the simplest model that works — no callbacks, no async, no observables. The cost is that you can't do background processing while streaming. For CLI tools and scripts, this is fine. For web servers that need to stream to browsers while handling other requests, you need threading or async, and lm15 doesn't help with either.

**The two-level API:** Most users want text. Some users want everything. lm15 solves this with two levels: `stream.text` (just strings) and `for event in stream` (typed events). This avoids forcing all users through the event interface while making it available for those who need it. The design insight: the simple path should be *default*, not *simplified*. `stream.text` isn't a dumbed-down version of the event stream — it's the right abstraction for the common case.

**Response materialization:** After streaming, you often need the complete response — for logging, for token counts, for passing to the next stage. lm15 accumulates all streamed data internally and materializes an `LMResponse` when the stream ends. This means the `Stream` object is both an iterator (for real-time consumption) and a response container (for post-hoc access). This dual nature is convenient but surprising — the object changes behavior after you iterate it.

**Error semantics in streams:** A blocking call either succeeds or fails. A stream can *partially succeed*. You receive 90% of the text and then the connection drops. What's the right behavior? lm15 raises an exception, but the partial text is accessible through `_materialize_response()`. Other libraries discard the partial data (simpler, but wasteful) or require explicit error handling in the iteration loop (more correct, but more boilerplate).

**Sections:**
- Iterator vs callback vs async vs observable
- Why iterators won (simplicity, synchronous code, backpressure)
- The two-level API (text vs events)
- Response materialization (the dual nature of Stream)
- Partial failure (the hardest streaming problem)
- Streaming + tools (tool calls arrive mid-stream, execution happens... when?)
- What async would change (and why lm15 doesn't do it)

---

## Part III — System Design

### Chapter 7: The Dependency Question

**The problem:** Every library depends on something. Most LLM libraries depend on HTTP clients, serialization libraries, validation frameworks, and async runtimes. lm15 depends on nothing. Is this wise?

**This is not an obvious choice.** The Python ecosystem's default is to depend on well-maintained packages. `requests` is battle-tested. `pydantic` catches bugs at construction time. `httpx` supports async natively. By choosing zero dependencies, lm15 chose to be worse at HTTP, worse at validation, and incapable of async — in exchange for never breaking because something else changed.

**The argument for dependencies:** They let you focus on your domain. `requests` handles HTTP/2, connection pooling, redirect following, cookie management, proxy support, certificate handling — things you don't want to write yourself. `pydantic` handles validation, serialization, schema generation — things that are tedious and error-prone to do manually. Using them isn't laziness; it's leveraging specialization.

**The argument against dependencies:** Every dependency is a coupling. `requests` pinning `urllib3>=1.21.1,<3` means your library inherits that constraint. When `urllib3` 3.0 ships with breaking changes, your users get version conflicts — not because of anything you did, but because two things they installed can't agree on a third thing neither of them wrote. The deeper your dependency tree, the more likely this becomes.

**lm15's position on the spectrum:** Zero dependencies is an extreme position. It works because lm15's HTTP needs are simple (send JSON, receive JSON/SSE), its validation needs are minimal (`__post_init__` on frozen dataclasses), and its async needs are zero (synchronous-only by design). A library with more complex needs — OAuth flows, WebSocket connections, streaming uploads, async batch processing — couldn't make this choice without reimplementing a significant fraction of the Python ecosystem.

**The import time argument:** lm15 imports in 95ms. `google-genai` takes 2,656ms. `litellm` takes 4,534ms. For CLI tools, serverless functions, and notebooks, cold-start time matters. Dependencies don't just risk breakage — they cost time on every invocation.

**What lm15 reimplements (and what it doesn't):**
- HTTP: `urllib.request.urlopen`. No connection pooling, no HTTP/2. Works.
- SSE parsing: 50 lines. Handles the spec. No edge cases in practice.
- JSON serialization: `json.dumps`. Standard library. Fine.
- Validation: `__post_init__`. Catches obvious mistakes. Misses subtle ones.
- NOT reimplemented: async (doesn't exist), streaming uploads (provider SDK territory), OAuth (not needed for API key auth).

**Sections:**
- The dependency spectrum (zero to framework)
- What dependencies buy you (specialization, maturity)
- What dependencies cost you (coupling, conflicts, cold-start)
- lm15's position and why it's viable
- What zero-deps forces you to reimplement
- The import time tax (real measurements)
- When zero-deps is wrong (complex HTTP, async, OAuth)

---

### Chapter 8: Layering and Boundaries

**The problem:** Every library has layers — whether deliberate or accidental. The question isn't whether to have layers, but where to draw the boundaries. Each boundary is a decision about what changes independently of what.

**lm15's five layers and the reasoning behind each boundary:**

**Sugar ↔ Model:** The sugar is stateless; the model is stateful. This boundary exists because some callers want `complete("model", "prompt")` — a function call with no side effects — and some callers want `model("prompt")` — an object that accumulates history. If you merge them, you lose one use case. The sugar is trivially simple (7 lines) precisely because all complexity lives below it.

**Model ↔ Client:** The model manages conversation state, tool execution, retries, and caching. The client routes requests to providers and runs middleware. These change independently — you can add retry logic without knowing about providers, and you can add providers without knowing about retries. If you merge them, every model object carries the routing table and every provider change affects the state management code.

**Client ↔ Adapter:** The client knows about providers as abstract slots. The adapter knows about one provider's wire format. This is the plugin boundary — the reason third parties can add providers without touching core. If you merge them, the client becomes a monolith that knows every provider's API.

**Adapter ↔ Transport:** The adapter knows about JSON and API contracts. The transport knows about HTTP. This lets you swap `urllib` for `pycurl` without touching any adapter, and lets you build test harnesses with mock transports.

**The cost of layers:** Each boundary is an abstraction, and abstractions leak. The `provider` escape hatch on Config and LMResponse is a leak — provider-specific features crossing a boundary that was supposed to hide them. The `TransportPolicy` is a leak — timeout configuration crossing from the transport into the factory. Every leak is evidence of a boundary that's slightly in the wrong place, or a concern that truly can't be separated.

**The alternative: fewer layers.** LiteLLM has roughly three layers: function → translation → HTTP. No model objects, no middleware, no transport abstraction. This is simpler — fewer files, fewer concepts, easier to understand on first read. The cost is that everything that lm15 puts in separate layers (retries, caching, state management, transport swapping) has to go somewhere else — either in user code or in monolithic translation functions.

**Sections:**
- Why layers exist (change isolation, the substitution test)
- lm15's five boundaries (the reasoning behind each)
- Where boundaries leak (escape hatches, cross-cutting concerns)
- The alternative: fewer layers (LiteLLM's approach)
- How to evaluate layering decisions (the "what changes independently" test)
- The accidental layer (when structure emerges from code growth, not design)

---

### Chapter 9: What to Hide

**The problem:** A library is an act of concealment. You hide complexity so the user doesn't have to see it. But every hidden detail is a detail the user can't control. Hide too much and the library is a black box — convenient until it does the wrong thing. Hide too little and the library is a thin wrapper — flexible but pointless.

**The concealment spectrum, applied to LLM libraries:**

**What lm15 hides completely (and should):**
- Wire format differences (JSON structure, header formats, endpoint URLs)
- SSE parsing (line protocol, event delimiting)
- Auth mechanics (Bearer vs header key vs query param)
- Response parsing (extracting text, tool calls, usage from provider JSON)

These are pure implementation details. No user wants to deal with them. No user's decision would change if they could see them. Hiding them is unambiguously correct.

**What lm15 hides partially (and it's debatable):**
- Prompt caching behavior (`prompt_caching=True` works differently on each provider — same parameter, different mechanism, different cost)
- Reasoning semantics (`reasoning=True` means different things to different providers — token budget vs effort level)
- Model capabilities (some models support tools, some don't — the user finds out when they get an error)
- Streaming format (the `Stream` class hides event accumulation, tool call fragment parsing, response materialization)

These are cases where the abstraction is leaky by nature. The providers genuinely differ, and no abstraction can make them the same. lm15 hides the *mechanism* but can't hide the *behavior*.

**What lm15 exposes (and should):**
- The `provider` escape hatch (raw dict for provider-specific features)
- `UniversalLM` and adapters (public classes you can instantiate directly)
- The full type system (`LMRequest`, `LMResponse`, `Part`, etc. — all importable)
- History (`model.history` — inspectable, clearable)

These are deliberate transparency points. The user who needs them can reach them without monkey-patching or subclassing.

**The test for what to hide:** Can you describe the hidden behavior in one sentence without mentioning the provider? If yes, hide it ("sends the request and parses the response" — provider-irrelevant). If no, expose it or provide an escape hatch ("caches the conversation prefix using Anthropic's ephemeral cache control markers" — provider-specific behavior that the user might need to understand).

**Sections:**
- The concealment problem (hiding is both the point and the danger)
- Three categories: fully hidden, partially hidden, deliberately exposed
- The one-sentence test (can you describe it without naming the provider?)
- Case study: prompt caching (same parameter, three different behaviors)
- Case study: error handling (same hierarchy, different meanings)
- The transparency-convenience tradeoff (escape hatches as pressure valves)
- Comparison: what LangChain hides vs what LiteLLM hides vs what lm15 hides

---

### Chapter 10: The Design of Time

**The problem:** LLM APIs change. Models are deprecated, renamed, and replaced. Providers add features, change pricing, alter rate limits. Wire formats evolve. A library that models LLM interactions must model them across time — not just how they work today, but how the design accommodates change.

**Model name instability:** `claude-sonnet-4-5` will be replaced. `gpt-4.1-mini` will be renamed. Every hardcoded model name is a ticking clock. lm15's discovery system (`lm15.models()`) is an answer to this — query what exists, pick from the result, adapt. But discovery introduces its own instabilities: live API endpoints change, capability metadata is inconsistent, fallback catalogs go stale.

**Provider API evolution:** OpenAI has changed their API three times (completions → chat completions → responses API). Anthropic has changed their message format twice. Gemini changes their endpoint paths between API versions. An adapter written for today's API might break next month. lm15 handles this by talking to the REST API directly (not through SDKs that version independently) and by versioning API calls explicitly (Anthropic's `anthropic-version` header).

**The plugin contract as a stability boundary:** lm15's adapter protocol is the stability promise — `complete(LMRequest) → LMResponse` and `stream(LMRequest) → Iterator[StreamEvent]`. As long as this contract holds, plugins don't break. But what if lm15 needs to add a field to `LMRequest`? Frozen dataclasses make this backward-compatible (new fields have defaults), but plugins that inspect request fields might miss the new one. The contract is stable; the types it references evolve.

**Feature detection vs feature assumption:** lm15 routes by model name prefix (`gpt-*` → OpenAI). This assumes the naming convention holds. A model called `o1-preview` breaks it (doesn't start with `gpt-`). The hydration system fixes individual cases, but the assumption is structural. An alternative — always requiring explicit `provider=` — would be more robust but less convenient.

**The deprecation problem:** When a model is deprecated, should the library error immediately, warn, or silently route to the successor? lm15 errors (the model isn't in the provider's list). Other libraries maintain compatibility aliases. Neither is clearly right — errors are annoying but honest; aliases are convenient but hide the change.

**Designing for the unknown:** The `provider` escape hatch on Config and LMResponse, the raw `parameters` dict on Tool, the `raw` dict on ModelSpec — these are all bets that the future will bring features lm15 can't predict. Every escape hatch is an admission of incomplete modeling and an investment in forward compatibility.

**Sections:**
- Model name instability (the deprecation clock)
- Provider API evolution (three providers, three histories)
- The plugin contract as a stability boundary
- Feature detection vs feature assumption (prefix matching, its limits)
- Designing escape hatches (the `provider` dict as a bet on the future)
- What lm15 would design differently today (with hindsight)
- The meta-question: how do you model something that's still being invented?

---

## Chapter Rhythm

| Ch | Title | Words | Core question |
|---|---|---|---|
| 1 | What Is a Message? | 5,500 | How do you represent multimodal, multi-typed content? |
| 2 | The Conversation Problem | 5,500 | Who owns the conversation state, and what does "memory" mean? |
| 3 | Modeling Tool Use | 6,000 | Who executes tools, and how does that shape the entire API? |
| 4 | The Normalization Question | 5,500 | How much should you hide provider differences? |
| 5 | The Adapter Pattern | 5,000 | How do you structure a translation layer for extensibility? |
| 6 | Streaming as a Design Problem | 5,500 | Iterator, callback, or async — and what falls out of each choice? |
| 7 | The Dependency Question | 5,000 | What do dependencies buy, what do they cost, and where's the line? |
| 8 | Layering and Boundaries | 5,500 | Where do you draw boundaries, and how do you know they're right? |
| 9 | What to Hide | 5,500 | How do you decide what the user sees and what they don't? |
| 10 | The Design of Time | 5,500 | How do you model something that's still being invented? |
| | **Total** | **~55,000** | |

---

## Design Principles for the Book

**Problems first, solutions second.** Each chapter opens with a design problem that exists independently of lm15. You could face this problem in any language, with any provider, using any library. lm15 is *one* answer, examined in detail, but never presented as the only answer.

**The solution space, not the solution.** Each chapter maps the options — string vs blocks vs parts, iterator vs callback vs async, zero deps vs full framework. The reader sees the tradeoff landscape, not just one point in it.

**Honest about costs.** Every design choice has a cost. lm15 chose simplicity over type safety (one Part type). lm15 chose zero deps over HTTP quality. lm15 chose synchronous over async. The book names the costs clearly, without apologizing for them.

**Comparative.** Each chapter references what LangChain, LiteLLM, Vercel AI SDK, and the raw provider SDKs chose for the same problem. Not to rank them, but to show that the solution space is real — different libraries made different choices, and those choices have different consequences.

**Useful to people who never use lm15.** The reader who uses LangChain should finish this book with a deeper understanding of *why* LangChain is shaped the way it is, and *what tradeoffs* it made. The problems are universal; the solutions vary.
