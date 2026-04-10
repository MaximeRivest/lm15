# Chapter 4: The Adapters

Our request has arrived at `AnthropicAdapter.complete()`. It's an `LMRequest` — lm15's universal type. It needs to become a JSON body that Anthropic's Messages API will accept. Then the response — Anthropic's JSON — needs to become an `LMResponse`. This translation is the adapter's entire job, and it's where the complexity actually lives.

To understand why adapters are the hardest part of lm15, look at the same request encoded for each provider:

**lm15 (universal):**
```python
LMRequest(
    model="claude-sonnet-4-5",
    messages=(Message(role="user", parts=(Part.text_part("Hello."),)),),
    system="Be brief.",
)
```

**OpenAI wire format:**
```json
{"model": "claude-sonnet-4-5", "messages": [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hello."}
]}
```

**Anthropic wire format:**
```json
{"model": "claude-sonnet-4-5", "max_tokens": 8192,
 "system": "Be brief.",
 "messages": [{"role": "user", "content": "Hello."}]}
```

**Gemini wire format:**
```json
{"contents": [{"role": "user", "parts": [{"text": "Hello."}]}],
 "systemInstruction": {"parts": [{"text": "Be brief."}]}}
```

Three completely different structures. The system prompt is a message (OpenAI), a top-level field (Anthropic), or a `systemInstruction` object (Gemini). The user message is `"content": "Hello."` (OpenAI), `"content": "Hello."` (Anthropic — same shape, different context), or `"parts": [{"text": "Hello."}]` (Gemini). And this is the *simplest* possible request. Add tool calls, images, cache control, and reasoning, and the divergence compounds.

Each adapter is 400-800 lines of this translation. I'll walk through the key patterns in each one, focusing on where they diverge and what makes each one complex.

## The Common Pattern

All three adapters follow the same structure, implemented in `providers/base.py`:

1. **Build the request**: `LMRequest` → provider-specific JSON dict
2. **Send it**: construct an `HttpRequest`, call `self.transport.request()`
3. **Check for errors**: if status ≠ 200, call `map_http_error()` to raise typed error
4. **Parse the response**: provider JSON → `LMResponse`

Streaming follows the same pattern but uses `self.transport.stream()` and yields `StreamEvent` objects instead of returning a single `LMResponse`.

## OpenAI

The OpenAI adapter is the most straightforward because OpenAI's API most closely matches lm15's data model (or rather, lm15's model was partly inspired by OpenAI's).

**System prompt**: becomes a message with `role: "system"` at the start of the messages array.

**Tools**: `Tool` → `{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`. The JSON Schema in `tool.parameters` passes through almost verbatim.

**Tool results**: `Part` with `type="tool_result"` → a message with `role: "tool"` and `tool_call_id` matching the original call's ID.

**Images**: `Part` with `type="image"` → `{"type": "image_url", "image_url": {"url": ...}}` for URLs, or base64-encoded data URLs for bytes.

**The `output` modality**: When `config.provider` contains `output: "image"`, the adapter routes to DALL-E for image generation. When `output: "audio"`, it routes to the TTS endpoint. These are entirely different API endpoints masquerading as part of `complete()`.

**Streaming**: OpenAI sends SSE with `data: {...}` lines. Each chunk has a `delta` with partial text, tool call fragments, or a `finish_reason`. The `[DONE]` sentinel signals end-of-stream.

**Prompt caching**: No-op. OpenAI caches prefixes automatically server-side. lm15 doesn't need to send cache control markers.

## Anthropic

Anthropic is the most complex adapter. Three features — cache control, prefill, and extended thinking — each require special handling that doesn't exist in the other adapters.

**System prompt**: top-level `"system"` field, not a message. If the system prompt is a string, it becomes `[{"type": "text", "text": "..."}]`. If it's a tuple of Parts (multimodal system prompt), each Part is translated individually.

**Cache control**: When `prompt_caching=True`, the adapter injects `"cache_control": {"type": "ephemeral"}` markers into the request. The placement is strategic — on the system prompt and on the latest "safe" message boundary. Each turn, the breakpoint advances, caching more of the conversation prefix. The logic for where to place the breakpoint is the most complex part of the adapter — roughly 50 lines of code that decide which message gets the cache marker based on its position in the conversation.

**Prefill**: lm15's `prefill=` parameter becomes a partial assistant message appended to the messages array. Anthropic uniquely supports this — the model sees the assistant message and continues from it. The adapter adds `Message(role="assistant", content=[{"type": "text", "text": prefill_string}])` at the end.

**Extended thinking**: When `reasoning` is enabled, the adapter adds `"thinking": {"type": "enabled", "budget_tokens": N}` to the request. The response includes `thinking` content blocks before the text blocks. The adapter parses these into `Part.thinking(text=...)`.

**Tool format**: Similar to OpenAI but with different key names. `Tool` → `{"name": ..., "description": ..., "input_schema": ...}`. Note `input_schema`, not `parameters`. Tool results use Anthropic's `tool_use`/`tool_result` content block format, which nests content inside arrays differently than OpenAI.

## Gemini

Gemini's adapter talks to the REST API directly — not the Google Python SDK. This was a deliberate choice: the SDK is a 41MB dependency with 25 transitive packages. The REST API is just HTTP.

**Content structure**: Gemini doesn't have `messages` — it has `contents`, each with a `role` and `parts`. The role mapping: `"user"` → `"user"`, `"assistant"` → `"model"`. Tool results use role `"function"`.

**System prompt**: `"systemInstruction": {"parts": [{"text": "..."}]}` — its own top-level object, separate from contents.

**Tools**: `Tool` → `{"functionDeclarations": [{"name": ..., "description": ..., "parameters": ...}]}`. Tool calls in the response come as `functionCall` parts. Tool results are `functionResponse` parts.

**CachedContent**: Gemini's prompt caching is the most complex of the three. Instead of cache markers on messages, the adapter creates a `CachedContent` resource via a separate API call, then references it in the generate request. This means caching involves an extra HTTP round-trip on the first call. The adapter manages the cached content lifecycle internally.

**File upload**: Gemini's Files API uses a different endpoint and returns file URIs that are referenced in subsequent requests. The adapter's `file_upload()` implementation talks to `generativelanguage.googleapis.com/upload/v1beta/files`.

**Model name normalization**: Gemini model names in the API are prefixed with `models/` — the adapter strips this on response and adds it on request, transparently.

## The Normalization Tax

Every adapter pays a tax for normalization — there are things that don't translate cleanly between providers, and the adapter has to make choices.

**What lm15 hides successfully**: basic text messages, tool call round-trips, usage counting, finish reasons, error mapping. These work identically across providers.

**What lm15 hides imperfectly**: cache control (different mechanisms per provider), reasoning (different parameter names and semantics), streaming format (different SSE structures), image/audio generation (different endpoints entirely).

**What lm15 can't hide**: provider-specific features that have no equivalent elsewhere. Anthropic's prefill. Gemini's `CachedContent` lifecycle. OpenAI's response format constraints. These go through the `config.provider` escape hatch — a raw dict that the adapter checks for provider-specific keys.

The `provider` escape hatch on both request and response is the pressure relief valve. Without it, every provider-specific feature would require a new field on `Config` or `LMResponse`, and the universal types would slowly become a union of all providers' types. The escape hatch keeps the universal types clean at the cost of untyped access to provider-specific features.

## What's Below

The adapter builds an `HttpRequest` and hands it to `self.transport.request()`. It gets back an `HttpResponse` with status, headers, and body bytes. The adapter never opens a socket, never manages TLS, never handles chunked encoding. That's the transport's job — and it's the last layer in the stack. The next chapter goes there.
