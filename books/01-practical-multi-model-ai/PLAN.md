# Book 1 — Practical Multi-Model AI in Python with lm15

**Analog:** R for Data Science
**Audience:** Python developers with no prior LLM API experience
**Length:** ~300 pages
**Prerequisite:** Basic Python (functions, dicts, loops, pip)
**Ships with:** lm15 v1.0

---

## Thesis

You don't need three SDKs, fifty dependencies, or a framework to use LLMs in Python. One import, one function, one string to switch models. This book takes you from zero to production-grade LLM calls in the simplest possible way.

---

## Part I — First Contact

### Chapter 1: One Function, Three Providers

- What lm15 is and why it exists (zero deps, frozen dataclasses, stdlib transport)
- `pip install lm15`
- Getting API keys (OpenAI, Anthropic, Gemini) — where, how, cost expectations
- The `.env` file pattern — `env=".env"`, `.gitignore` hygiene
- `lm15.complete("gpt-4.1-mini", "Hello.")` — your first call
- Anatomy of `LMResponse`: `.text`, `.model`, `.finish_reason`, `.usage`

### Chapter 2: Switching Models

- Change the string, change the model: `gpt-4.1-mini` → `claude-sonnet-4-5` → `gemini-2.5-flash`
- Same code, same response type, same fields
- When to use `provider=` for custom/fine-tuned model names
- Cost and latency differences between providers (ballpark)

### Chapter 3: Controlling Output

- `system=` — shaping model behavior
- `temperature=` — deterministic vs creative
- `max_tokens=` — capping response length
- `top_p=`, `stop=` — advanced sampling controls
- Config-from-dict pattern: `lm15.complete(**config, prompt="...")`

---

## Part II — Conversations

### Chapter 4: The Model Object

- `lm15.model()` — configure once, call many times
- Why stateful vs stateless: when each matters
- Bound config: system prompt, temperature, max_tokens
- Per-call overrides: `gpt("Be creative.", temperature=1.5)`

### Chapter 5: Multi-Turn Conversation

- Automatic history tracking
- `gpt("My name is Max.")` → `gpt("What's my name?")`
- Inspecting history: `gpt.history`, entry structure
- Clearing history: `gpt.history.clear()`
- Explicit multi-turn with `messages=[Message.user(...), Message.assistant(...), ...]`

### Chapter 6: Derived Models and Reuse

- Immutable derivation: `with_model()`, `with_system()`, `with_tools()`, `with_provider()`
- Original unchanged, new copy returned
- Pattern: base config object + per-task derivations
- `cache=True` for local response caching

---

## Part III — Tools

### Chapter 7: Auto-Execute Tools

- Pass a Python function, get tool calling for free
- Schema inference from type hints + docstring
- Single tool: `tools=[get_weather]`
- Multiple tools: `tools=[search, calculator]`
- What happens under the hood: model calls → lm15 executes → model gets result → final text

### Chapter 8: Manual Tool Loops

- `Tool` objects with explicit JSON Schema
- The loop: `resp.finish_reason == "tool_call"` → execute → `submit_tools()` → repeat
- Why manual: async tools, approval gates, side effects, multimodal tool results
- Multi-hop: model calls multiple tools across multiple rounds

### Chapter 9: Built-In Tools

- `tools=["web_search"]` — provider-side server tools
- `resp.citations` — accessing source URLs and titles
- Provider support matrix for built-in tools

### Chapter 10: Tool Design Patterns

- Naming and description best practices (the model reads them)
- Parameter design: simple types, required vs optional
- Error handling in tool functions
- Tools that return structured data vs prose

---

## Part IV — Multimodal

### Chapter 11: Images

- `Part.image(url=...)` — image from URL
- `Part.image(data=..., media_type=...)` — image from bytes
- Vision: describe, analyze, compare images
- Image generation: `output="image"`, `resp.image`
- Cross-model: generate on GPT, describe on Claude

### Chapter 12: Documents

- `Part.document(url=...)` and `Part.document(data=..., media_type=...)`
- PDF analysis, contract review, summarization
- `lm15.upload()` — provider file API for large documents
- Upload on model objects: `claude.upload("file.pdf")`

### Chapter 13: Audio and Video

- `Part.video(url=...)` — video understanding (Gemini)
- Audio generation: `output="audio"`, `resp.audio`
- Cross-modal pipelines: generate audio → transcribe with another model
- Provider capability matrix for media types

### Chapter 14: Mixed-Media Prompts

- Lists as prompts: `["Describe this.", Part.image(...), "Focus on colors."]`
- Combining text, images, documents in one call
- When to use single calls vs pipelines

---

## Part V — Streaming

### Chapter 15: Text Streaming

- `lm15.stream(...).text` — the simple iterator
- Print-as-you-go pattern
- When streaming matters: UX, long responses, time-to-first-token

### Chapter 16: Event Streaming

- Full event loop: `text`, `thinking`, `tool_call`, `tool_result`, `finished`
- `match event.type` pattern
- `stream.response` — materialized response after consumption
- Token counts and finish reason from the finished event

### Chapter 17: Streaming with Tools and Reasoning

- Streaming + auto-execute tools: tool calls and results appear as events
- Streaming + `reasoning=True`: thinking tokens stream before text
- Streaming on model objects: `gpt.stream(...)` — records to history
- Building responsive UIs with streaming events

---

## Part VI — Production

### Chapter 18: Prompt Caching

- `prompt_caching=True` — what it does per provider
- Agent loop caching: advancing breakpoint pattern
- Per-part cache hints: `Part.document(..., cache=True)`
- Reading cache metrics: `resp.usage.cache_read_tokens`, `cache_write_tokens`
- Provider behavior differences (Anthropic explicit, OpenAI automatic, Gemini CachedContent)

### Chapter 19: Reasoning / Extended Thinking

- `reasoning=True` — enable chain-of-thought
- `resp.thinking` — reading the model's reasoning
- Fine-grained control: `reasoning={"budget": 10000}`, `reasoning={"effort": "high"}`
- When reasoning helps: math, logic, multi-step planning
- Cost implications: reasoning tokens count

### Chapter 20: Reliability

- `retries=` on model objects
- Timeout configuration
- Error types and what to catch
- Prefill: `prefill="{"` — steering output format
- Defensive patterns: fallback models, token budget guards

### Chapter 21: Discovery and Introspection

- `lm15.models()` — list all available models
- Filter by provider, capabilities, modalities
- `lm15.providers_info()` — which providers are configured
- Pattern: pick model from discovered list instead of hardcoding
- `lm15.providers()` — env var names per provider

---

## Appendices

- A: Full `complete()` / `stream()` parameter reference
- B: `LMResponse` field reference
- C: `Usage` field reference
- D: Provider support matrix
- E: Environment variable reference
