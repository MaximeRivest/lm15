# Book 4 — Extending lm15: Plugins and Custom Providers

**Audience:** Library authors who want to add provider support — Mistral, Cohere, Ollama, a corporate proxy, a mock for testing
**Length:** ~30,000 words / ~120 pages
**Prerequisite:** Book 3 Chapters 1-5 (the five-layer stack, types, adapters, transport)

---

## Thesis

lm15's plugin system exists so that adding a provider never requires a pull request to core. You build an adapter, package it as `lm15-x-yourprovider`, publish to PyPI, and users get it with `pip install`. This book walks you through the entire process — from "I have a provider with an API" to "users are calling `lm15.complete('my-model', ...)`" — by building a real adapter for a real provider, start to finish.

---

## The Through-Line

The reader builds an adapter for **Mistral** (or a comparable provider with a public API). Not a toy. Not a mock. A real adapter that handles complete, stream, tool calls, and error mapping. Each chapter adds one capability to the adapter, and by Chapter 7 it's published on PyPI and discoverable by `build_default(discover_plugins=True)`.

The Mistral adapter is the vehicle. The destination is understanding the adapter contract well enough to build one for any provider, including custom/internal ones.

---

## Part I — Building the Adapter

### Chapter 1: The Contract

**Arc:** Before writing code, the reader understands what an adapter must do — and, equally important, what it must not do. The adapter contract is small: implement `complete()`, implement `stream()`, declare your capabilities. Everything else is optional.

**Opens with:** The simplest possible adapter — a class with `complete()` that returns a hardcoded `LMResponse`. Fifteen lines. Register it with `UniversalLM`, call `lm15.complete("mock-model", "Hello.")`, get the hardcoded response. The reader has a working (useless) adapter in the first two pages.

**The contract:**
- Required: `provider` attribute, `complete(request) → LMResponse`
- Expected: `stream(request) → Iterator[StreamEvent]`, `manifest`, `supports`
- Optional: `file_upload()`, `batch_submit()`, `list_models()`
- What the adapter must NOT do: manage auth, handle retries, parse model names, manage conversation state — these are other layers' jobs

**ProviderManifest and EndpointSupport:**
- `ProviderManifest(provider="mistral", supports=..., env_keys=("MISTRAL_API_KEY",))`
- `EndpointSupport(complete=True, stream=True, file_upload=False, ...)`
- How lm15 uses these for routing and capability checks
- When to set `False` vs not implementing the method (both work, explicit is better)

**The build_adapter() factory:**
- The entry point function that the plugin system calls
- Returns an adapter instance (not a class)
- Can accept configuration, read env vars, construct a transport

**Key insight:** The adapter contract is intentionally minimal. lm15 asks you to translate between two formats — its types and your provider's wire format. Everything else (auth, retries, caching, routing) is handled by layers above you. This is freeing: you only need to understand your provider's API and lm15's types.

**Sections:**
- The fifteen-line mock adapter (instant gratification)
- What the contract requires (complete, stream, manifest)
- What the contract forbids (auth, retries, state)
- ProviderManifest and EndpointSupport
- The build_adapter() factory
- The adapter's place in the stack (where it sits, what it touches)

---

### Chapter 2: Building the Request

**Arc:** The reader implements `complete()` for Mistral. The job: receive an `LMRequest`, translate it into Mistral's JSON body, and send it. This chapter covers the translation — message encoding, tool encoding, system prompt placement, config mapping.

**Opens with:** A side-by-side. The `LMRequest` on the left. Mistral's expected JSON on the right. The reader's job is to write the function that turns one into the other.

**Message translation:**
- `Message` with `Part` objects → Mistral's message format
- Text parts → `{"type": "text", "text": "..."}`
- Image parts → whatever Mistral expects (URL reference, base64, etc.)
- Tool call parts → Mistral's tool call format
- Tool result parts → Mistral's tool result format

**System prompt placement:**
- Some providers want system as a top-level field (Anthropic)
- Some want it as the first message with role "system" (OpenAI, Mistral)
- Reading the provider's docs and translating accordingly

**Tool schema encoding:**
- `Tool` objects → Mistral's function declaration format
- JSON Schema passthrough (most providers accept standard JSON Schema)
- The `builtin` tool type (handle or skip, depending on provider support)

**Config mapping:**
- `temperature`, `max_tokens`, `top_p` → Mistral's parameter names
- `reasoning` → whatever the provider calls its thinking feature (or skip if unsupported)
- `provider` dict → passthrough for provider-specific parameters

**Sending the request:**
- Use the built-in `UrlLibTransport` or `PycurlTransport`
- Build the URL, headers (including auth), and JSON body
- Send and receive

**Sections:**
- The translation diagram (LMRequest → Mistral JSON)
- Message encoding (each Part type)
- System prompt placement (where does it go?)
- Tool encoding (schema → provider format)
- Config mapping (lm15 names → provider names)
- Sending the HTTP request

---

### Chapter 3: Parsing the Response

**Arc:** The provider returns JSON. The reader parses it into an `LMResponse`. This is where normalization happens — mapping the provider's response shape into lm15's universal format.

**Opens with:** Mistral's response JSON. The reader identifies the text content, the finish reason, the usage counters, and the tool calls (if any). Then writes the code to extract each one.

**Text extraction:**
- Find the text content in the response → `Part.text_part("...")`
- Handle multiple content blocks if the provider returns them
- Handle empty responses gracefully

**Tool call extraction:**
- Find tool calls in the response → `Part.tool_call(id, name, input)`
- Parse the arguments (usually a JSON string inside the response)
- Handle multiple tool calls in one response

**Usage mapping:**
- Provider's token counts → `Usage(input_tokens=..., output_tokens=..., total_tokens=...)`
- Not all providers report all fields — set missing ones to 0 or None
- Reasoning tokens, cache tokens — map if available, skip if not

**Finish reason mapping:**
- Provider's finish reason string → lm15's `FinishReason` literal
- `"stop"`, `"length"`, `"tool_call"` — the common ones
- Provider-specific values that need translation

**Building the LMResponse:**
- Assemble the message from extracted parts
- Set the response ID, model name, finish reason, usage
- The `provider` escape hatch: store the raw response dict for debugging

**Error responses:**
- Detect HTTP errors before parsing
- Use `map_http_error(status, message)` to create typed errors
- Extract the human-readable error message from provider-specific JSON

**Sections:**
- Reading the provider's response format
- Text and tool call extraction
- Usage mapping
- Finish reason mapping
- Assembling the LMResponse
- Error handling and mapping

---

### Chapter 4: Streaming

**Arc:** Streaming is the hardest part of writing an adapter. The reader implements `stream()` — opening an SSE connection, parsing deltas, accumulating tool calls, and yielding `StreamEvent` objects.

**Opens with:** The difference between `complete()` and `stream()` from the adapter's perspective. `complete()` is synchronous: send request, receive response, parse, return. `stream()` is a generator: open connection, yield events as they arrive, handle partial data, signal completion. It's fundamentally more complex.

**SSE connection:**
- Open a streaming HTTP request (chunked transfer encoding)
- Use the transport's streaming method
- Handle connection errors, timeouts, and unexpected disconnections

**Parsing SSE events:**
- The SSE line protocol: `data:` lines, `event:` lines, empty line as delimiter
- Using lm15's `sse.py` parser or writing your own
- Provider differences: some send `data: [DONE]`, some send `event: done`, some just close the connection

**Yielding StreamEvents:**
- `StreamEvent(type="start", id=..., model=...)` — first event
- `StreamEvent(type="delta", delta=PartDelta(type="text", text="..."))` — text chunk
- `StreamEvent(type="delta", delta=PartDelta(type="thinking", text="..."))` — reasoning chunk
- `StreamEvent(type="delta", delta=PartDelta(type="tool_call", input="..."))` — tool call fragment
- `StreamEvent(type="end", finish_reason=..., usage=...)` — final event

**Tool call deltas:**
- Tool calls arrive as fragments: the name first, then the arguments in JSON chunks
- Track `part_index` to accumulate the right tool call
- The adapter yields fragments; the `Stream` class (Chapter 6 of Book 3) handles accumulation

**Error events:**
- `StreamEvent(type="error", error={"code": "...", "message": "..."})` — stream-level errors
- Connection drops mid-stream — what to yield

**Testing streams:**
- Feed canned SSE bytes to your parser
- Verify the event sequence matches expectations
- Edge cases: empty deltas, multiple tool calls, thinking then text

**Sections:**
- complete() vs stream() (the complexity difference)
- Opening the SSE connection
- Parsing SSE events (the line protocol)
- Yielding StreamEvents (each type)
- Tool call fragments (accumulation, part_index)
- Error handling in streams
- Testing with canned data

---

### Chapter 5: Tool Round-Trips

**Arc:** Tool calling is a multi-step conversation. The model requests a tool call, the user executes it, and the adapter must encode the tool result back into the next request. This chapter covers the full round-trip — request → tool_call response → tool_result message → follow-up request.

**Opens with:** The tool-call dance from the adapter's perspective. The first request goes out with tool definitions. The response comes back with `finish_reason="tool_call"` and a tool call in the message. Now the Model builds a follow-up request with a `tool` role message containing the result. Your adapter receives *this* request — a conversation with user messages, assistant messages with tool calls, and tool messages with results. You need to encode all of it correctly.

**Encoding tool definitions:**
- `Tool` objects → provider's function declaration format
- This was covered in Chapter 2 but now we test it end-to-end

**Parsing tool calls from responses:**
- Extract `Part.tool_call(id, name, input)` from the provider's response
- Get the right ID format (some providers use UUIDs, some use sequential IDs)

**Encoding tool results in follow-up requests:**
- `Part.tool_result(id, content)` → provider's tool result format
- The `tool` role message → provider's expected format for tool results
- Matching result IDs to call IDs

**End-to-end test:**
- Send a request with tools defined
- Receive a tool_call response
- Build the follow-up with tool results
- Send the follow-up
- Receive the final text response
- Verify the full round-trip works

**Common bugs:**
- ID mismatch between tool call and tool result
- Tool result content encoded as string instead of content block
- Missing tool definitions on the follow-up request (some providers require them again)
- Arguments parsed as string instead of dict

**Sections:**
- The round-trip from the adapter's view
- Encoding tool definitions (review)
- Parsing tool calls from responses
- Encoding tool results
- The end-to-end test
- Common bugs and how to debug them

---

## Part II — Packaging and Shipping

### Chapter 6: Testing

**Arc:** You have an adapter. Does it work? This chapter covers testing strategies — from unit tests with mocked HTTP to integration tests against the live API to lm15's completeness framework.

**Opens with:** A mock transport that returns canned JSON responses. The reader plugs it into their adapter and runs tests without hitting any API. The tests are fast, free, and deterministic.

**Unit testing with mock transport:**
- Build a mock transport that returns predefined responses
- Test `complete()`: verify the request body your adapter built, verify the response it parsed
- Test `stream()`: feed canned SSE lines, verify the event sequence
- Test tool round-trip: verify encoding and parsing of tool calls and results
- Test error handling: return HTTP 429, 500, malformed JSON — verify the right exception

**Integration testing against live API:**
- Run real calls against the provider with a real API key
- Keep these tests separate (slow, costly, require credentials)
- CI setup: API key in GitHub secrets, run on push to main
- Test matrix: basic text, tools, streaming, multimodal (if supported)

**The completeness framework:**
- lm15's `completeness/` directory has a test suite for adapters
- Standard test cases: simple text, tool call round-trip, streaming, error handling
- Running the completeness tests against your adapter
- What a "complete" adapter means (and when partial is fine)

**The test pyramid for adapters:**
- Many unit tests (mock transport, fast, free)
- Some integration tests (live API, slow, costly)
- Few completeness tests (comprehensive, live API)

**Sections:**
- The mock transport pattern
- Unit testing (complete, stream, tools, errors)
- Integration testing (live API, CI setup)
- The completeness framework
- The test pyramid

---

### Chapter 7: Publishing

**Arc:** The adapter works and is tested. Now package it, publish it to PyPI, and make it discoverable by `build_default(discover_plugins=True)`. The reader ships a real plugin.

**Opens with:** The project structure — six files total. The reader has already written the adapter (the hard part). This chapter handles the packaging (the boring-but-necessary part).

**Project structure:**
```
lm15-x-mistral/
├── pyproject.toml
├── README.md
├── LICENSE
├── lm15_x_mistral/
│   ├── __init__.py      # build_adapter() factory
│   └── adapter.py       # MistralAdapter class
└── tests/
    ├── test_complete.py
    └── test_stream.py
```

**The entry point (`pyproject.toml`):**
```toml
[project.entry-points."lm15.providers"]
mistral = "lm15_x_mistral:build_adapter"
```
- This is the magic line. It tells lm15's plugin system where to find your adapter.
- When a user calls `build_default(discover_plugins=True)`, Python's `importlib.metadata` finds this entry point, imports `lm15_x_mistral`, calls `build_adapter()`, and registers the returned adapter.

**Naming conventions:**
- Package name: `lm15-x-{provider}` (the `x` prefix means "external")
- Module name: `lm15_x_{provider}` (underscores, not hyphens)
- Provider name: `{provider}` (the string used in `provider=` arguments)

**Versioning and compatibility:**
- Depend on `lm15 >= X.Y, < X+1.0` — pin to major version
- The adapter contract is the stability boundary — lm15 won't break it in minor versions
- What can change (internal APIs, new optional methods) vs what won't (the contract)

**Publishing:**
- Build: `python -m build`
- Upload: `twine upload dist/*`
- Verify: `pip install lm15-x-mistral` then `build_default(discover_plugins=True)`

**The README:**
- Installation instructions
- Configuration (which env var, where to get a key)
- Supported features (complete, stream, tools, etc.)
- Limitations (what's not supported yet)

**After publishing:**
- Users do `pip install lm15-x-mistral`
- Then `lm15.complete("mistral-large", "Hello.")` works
- No changes to user code, no imports, no registration — the entry point handles everything

**Sections:**
- Project structure (six files)
- The entry point (the magic line in pyproject.toml)
- Naming conventions
- Versioning and compatibility
- Building and publishing to PyPI
- The README template
- Verification (install and test)

---

### Chapter 8: Advanced Patterns

**Arc:** The reader's adapter works for basic text, streaming, and tools. This chapter covers the optional features — file upload, batch submit, custom transports, and adapting for providers with non-standard APIs.

**Opens with:** The reader's adapter handles the 80% case. This chapter covers the 20% — the features that make an adapter production-complete.

**File upload:**
- `file_upload(FileUploadRequest) → FileUploadResponse`
- Implementing the provider's file upload endpoint
- Returning a file ID that can be used in subsequent requests
- Setting `EndpointSupport(file_upload=True)`

**Batch submit:**
- `batch_submit(BatchRequest) → BatchResponse`
- Implementing async batch processing
- Provider-specific lifecycle (submit, poll, retrieve)

**Custom transports:**
- When to build one: corporate proxy, mTLS, custom HTTP client
- Implementing the `Transport` protocol
- Plugging it into your adapter: `MistralAdapter(transport=MyTransport())`
- Example: a logging transport that records all HTTP traffic

**Non-standard providers:**
- Providers that use WebSocket instead of HTTP
- Providers that require OAuth instead of API keys
- Providers that don't follow the chat completion pattern (embedding-only, etc.)
- Ollama and local models: adapting the transport for localhost

**Multiple adapters for one provider:**
- Different API versions (v1 vs v2)
- Different authentication (API key vs OAuth)
- Different endpoints (chat vs embedding vs image)

**Sections:**
- File upload (implementing, registering)
- Batch submit (implementing, lifecycle)
- Custom transports (protocol, examples)
- Non-standard providers (WebSocket, OAuth, local)
- Multiple adapters (when and how)

---

## Chapter Rhythm

| Ch | Title | Words | What the reader has at the end |
|---|---|---|---|
| 1 | The Contract | 3,500 | A mock adapter that handles `complete()` with hardcoded responses |
| 2 | Building the Request | 4,000 | An adapter that translates LMRequest → Mistral JSON and sends it |
| 3 | Parsing the Response | 4,000 | An adapter that handles complete() with real API calls |
| 4 | Streaming | 4,500 | An adapter that handles stream() with SSE parsing |
| 5 | Tool Round-Trips | 3,500 | An adapter that handles the full tool call lifecycle |
| 6 | Testing | 3,500 | A tested adapter with unit, integration, and completeness tests |
| 7 | Publishing | 3,000 | A published PyPI package that users can install and use |
| 8 | Advanced Patterns | 4,000 | Knowledge to handle file upload, batch, custom transports |
| | **Total** | **~30,000** | |

---

## Design Principles

**Build one real thing.** Not three hypothetical adapters. One real adapter for a real provider, built from scratch, tested against a live API, published to PyPI. The reader can fork it for their own provider.

**The adapter grows across chapters.** Chapter 1: hardcoded mock. Chapter 2: builds real requests. Chapter 3: parses real responses. Chapter 4: streams. Chapter 5: tool round-trips. Chapter 6: tested. Chapter 7: published. Each chapter adds one capability, same as Book 1.

**Show the provider's documentation.** The reader needs to read two things: lm15's types and the provider's API docs. The book shows both side by side — lm15's `LMRequest` next to Mistral's expected JSON. The translation is visible.

**Honest about the hard parts.** Streaming is hard. Tool call encoding is fiddly. SSE parsing has edge cases. The book names the difficulty rather than pretending everything is straightforward.

**Copy-pasteable.** The adapter code from each chapter is complete and runnable. The reader can literally copy it, change the provider name and URL, and have a working adapter for a different provider.
