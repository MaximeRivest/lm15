# Canonical mapping rules

Normative rules for mapping provider responses into the canonical lm15
representation. Companion to `serde-rules.md` (which governs the JSON wire
format); these govern WHAT becomes a canonical part. Goldens and conformance
fixtures cite these rules by number.

## MAP-1 — Parts are what the application must act on

A canonical message part is something the application must handle or display:
text, thinking, citations, media, a client-side tool call (which obligates the
caller to execute it and return a tool_result), or a tool result.

Provider-executed builtin tool activity is **not** represented as parts. This
includes Anthropic `server_tool_use` / `code_execution_tool_result` blocks and
container metadata, OpenAI `code_interpreter_call` / `web_search_call` items,
and Gemini `executableCode` / `codeExecutionResult` parts. The user-relevant
*outputs* of such tools (answer text, citations, generated media) are mapped
to parts as usual; the execution mechanics remain available verbatim in
`provider_data`.

**Why:** a `tool_call` part is a contract — "the caller must execute this."
Agent loops iterate tool_call parts and run them. Surfacing provider-executed
calls as tool_call parts would cause every agent loop to re-execute work the
provider already performed. If canonical access to execution traces is needed
later, it must be a NEW part type (additive), never a reinterpretation of
tool_call.

## MAP-2 — A response message is never empty

When a provider response yields no canonical parts (e.g. the model spent its
entire output budget on hidden reasoning and was truncated), the canonical
message is a single empty `TextPart` (`text: ""`).

**Why:** `Message.parts` is non-empty by invariant, everywhere, for every
producer — relaxing it for one edge case would weaken a guarantee all ports
and consumers rely on. Erroring would turn a legitimate provider response into
a crash. With the empty part, `response.text == ""` plus the finish_reason
(e.g. `"length"`) reads as exactly what happened.

## MAP-3 — A stream yields exactly one end event, and it is final

An lm15 stream yields EXACTLY ONE `StreamEndEvent`, as the final event of the
stream, carrying the `finish_reason` and `usage` accumulated across all of the
provider's terminal frames.

Providers split their terminal data across multiple frames: OpenAI-compatible
servers (vLLM, SGLang, ollama, Groq) send a `finish_reason`-bearing chunk,
then — with `stream_options.include_usage` — a usage-only chunk, then
`[DONE]`; Anthropic sends `message_delta` (stop_reason + usage) followed by a
bare `message_stop`. Adapters stay stateless and may emit one per-frame end
event for each such terminal frame, but that is an internal detail: a
provider-agnostic coalescer (`lm15.result.coalesce_stream`) absorbs every
adapter end event — later non-`None` fields fill gaps, a non-`None` field is
never overwritten by `None` — and emits the single merged end event once the
underlying iterator is exhausted. The canonical event trace (goldens, the vet
shim's `replay_stream`, conformance `parse_stream`) is the POST-coalesce
trace.

**Why:** multiple end events made every consumer's merge semantics
load-bearing, and they failed in live testing. `Result` treated the first end
event as terminal (`break` on `type == "end"`), so the post-finish usage-only
chunk that vLLM/SGLang/ollama send was never applied and the materialized
`Response.usage` came out all zeros (pinned as a known-bug baseline in the
streaming_vllm/streaming_sglang draft goldens before this rule). With exactly
one final end event, "the end event" and "the stream's finish_reason and
usage" are the same thing by construction, in every port.

---

## MAP-4 — A stream opens with exactly one start event

An lm15 stream that yields any delta or end event yields EXACTLY ONE
`StreamStartEvent`, before all of them. Dialects with a real start frame
(OpenAI Responses `response.created`, Anthropic `message_start`) pass it
through with its `id` and `model`; dialects without one (chat completions,
Gemini SSE) get a synthesized start carrying the request's model, added by
the same coalescer that enforces MAP-3. Duplicate starts collapse to the
first. Error events never force a start: a stream that fails to open has no
start.

**Why:** live testing (dspy-greenfield `tests/live`, 2026-08-16) showed the
event vocabulary split by dialect — Responses API streams began with a start
event, chat-completions and Gemini streams began with a bare delta — so any
consumer that keyed on the start event worked on one provider and broke on
the next. One vocabulary means the trace shape is provider-independent.

## MAP-5 — Explicit reasoning-off reaches the wire or fails loudly

`Config(reasoning=Reasoning(effort="off"))` is an explicit instruction, not
a default (the tri-state is defined in spec/types.md §Reasoning). Every
adapter must translate it into the provider's native disable mechanism:

- OpenAI Responses dialect: `reasoning: {"effort": "none"}`
- Chat Completions `reasoning_effort` servers (incl. Groq, vLLM, SGLang):
  `reasoning_effort: "none"`
- OpenRouter: `reasoning: {"enabled": false}`
- DeepSeek: `thinking: {"type": "disabled"}`
- Qwen/DashScope, Z.AI: `enable_thinking: false`
- Gemini: `thinkingConfig: {"thinkingBudget": 0}`
- Anthropic and Claude Code: omit `thinking` — thinking is opt-in there,
  so absence IS the native off switch.
- xAI: RAISES `UnsupportedFeatureError`. Grok reasoning models have no
  off switch and api.x.ai silently ignores disable-shaped fields.

When the selected model cannot honor the disable (gpt-5-mini's floor is
`"minimal"`; gemini-2.5-pro rejects budget 0), the provider's 400 surfaces
unchanged. An adapter must never omit the field and let the model reason at
its default — that is a silent paid no-op.

**Why:** live testing (2026-09-01) showed omission was not off:
gpt-5-mini spent 64 hidden reasoning tokens, Groq gpt-oss-20b spent 45,
and grok-4.6 spent 158 while accepting `thinking: {"type": "disabled"}`
without effect. Reasoning tokens are billed output; an explicit off that
silently does nothing charges the caller for what they disabled.

## MAP-6 — Caching: one model for every provider

Every provider caches prompt prefixes in up to three tiers, measured
2026-09-01 across 13 providers (lm15-contract/research/caching/):

- **automatic** — nothing to send, best-effort, no user-visible state
  (OpenAI all classes, Gemini implicit, xAI, Groq, DeepSeek, Fireworks,
  vLLM, SGLang);
- **breakpoint** — a mark on a block, guaranteed above a per-model
  minimum, 1.25x write price (Anthropic; OpenAI gpt-5.6 and later);
- **resource** — a stored, named object with a lifetime and a storage
  price per token-hour, pinned to one model (Gemini `cachedContents`,
  Vertex; any future provider with the same shape).

`CacheConfig` names INTENTS. Each adapter maps an intent to the best tier
it has; the outcome is always visible in `Usage.cache_read_tokens` /
`cache_write_tokens`.

1. **No `config.cache`: send nothing.** Automatic tiers apply server-side.
2. **`mode="off"`: send nothing, and disable cache WRITES where a switch
   exists** — OpenAI gpt-5.6+ `prompt_cache_options: {"mode":
   "explicit"}` with no marks. Pre-5.6 OpenAI models reject the option and
   write for free, so they get nothing (option 2, ratified 2026-09-01).
3. **`mode="auto"` with no prefix: the cheapest safe instruction.**
   Anthropic marks the system block; every other provider sends nothing.
   Never the trailing marker: with a changing last message it wrote the
   full prefix at 1.25x on every call and read nothing (measured on both
   Anthropic top-level `cache_control` and OpenAI implicit mode).
4. **Prefix intents are marks where marks exist, and fall back to the
   automatic tier where they do not.** `prefix="stable"` marks the end of
   system + tools (Anthropic: the system block; OpenAI: the system prompt
   is rendered as the first developer/system message with the mark,
   because top-level `instructions` cannot carry one). `prefix="history"`
   marks the last block of the last message (Anthropic); on OpenAI 5.6+
   implicit mode already does exactly that, so nothing is sent.
   `prefix_until_index=N` marks the last block of message N (Anthropic:
   any block; OpenAI: a text block, else RAISE). Providers without marks
   (Gemini, xAI, Groq, older OpenAI, compat servers with
   `cache_control="none"`) send nothing. The fallback is permitted by two
   conditions, both required: it spends nothing, and its outcome is
   observable in usage. It must not be extended to fields that fail
   either condition.
5. **`retention="long"`** names a specific mechanism: Anthropic `ttl:
   "1h"` (2x write); OpenAI <5.6 `prompt_cache_retention: "24h"`;
   OpenAI 5.6+ (30m is the only value) and Gemini (lifetime belongs to
   the stored object) RAISE.
6. **`key`** is a best-effort affinity hint: OpenAI and OpenRouter
   `prompt_cache_key`; Anthropic and Gemini RAISE.
7. **`resource`** is a `CacheInfo.id` from the resource tier. The adapter
   references the object and sends only what the object does not hold —
   Gemini: `cachedContent` + the messages after `prefix_until_index`, no
   `systemInstruction`/`tools`/`toolConfig` (the server rejects them next
   to a cache). Providers without the tier RAISE.
8. **The resource tier is a surface**, shaped like files: `cache_create
   (prefix: Request, ttl_seconds, label)`, `cache_get`, `cache_list`,
   `cache_delete`, `cache_update(id, ttl_seconds)`, pure hooks, async
   mirrors, the `cache` harness direction, `EndpointSupport.caches`.
   `lm.cache(prefix)` returns a `CachedPrefix`: on the resource tier it
   creates the object (one explicit, billed call); elsewhere it is pure.
   `cached + messages` builds the Request with the boundary at the seam.
9. **No hidden network calls.** An adapter's `build_request` never
   creates cache state. (Removed 2026-09-02: the Gemini adapter's
   per-request `cachedContents` POST, which made a billed object per
   turn and reused none.)
10. **Docs state the fan-out trap**: on OpenAI 5.6+ with no config, one
    document and many questions writes at 1.25x every time and never
    reads; `prefix="stable"` or `lm.cache(prefix)` is the one-line fix.
    A tools change is a miss everywhere.

**Why:** the caching design pass (lm15-contract/changes/2026-09-01-caching-design.md,
research/caching/). Provider agnosticism is defined as: the same code
runs everywhere, does the best thing the provider offers, and shows the
result — not identical bytes saved everywhere.

## MAP-7 — Reasoning: one dial, two spellings, no silent drops

Measured 2026-09-02 across OpenAI, Anthropic, Gemini, xAI, Groq (134
cells) and 17 sources (lm15-contract/research/reasoning/).

1. **Absent `config.reasoning` sends nothing**: the model decides. Every
   provider's default is adaptive now; "adaptive" is not a level.
2. **`effort` is the one dial**, required, vocabulary `off, minimal, low,
   medium, high, xhigh, max`. Providers with levels get the word
   verbatim (OpenAI; Anthropic adaptive class as `output_config.effort`;
   Gemini 3.x as `thinkingLevel`; xAI and Groq as `reasoning_effort`).
   Model-unsupported words fail with the server's 400. Words with no
   native level on a provider RAISE client-side: Anthropic `minimal`,
   Gemini 3.x `xhigh`/`max`.
3. **Budget-only model classes express effort as a budget** through one
   grading table — minimal 1024, low 2048, medium 8192, high 16384,
   xhigh 24576, max 32768 — Anthropic's manual class (4.5 and earlier:
   `budget_tokens`) and Gemini 2.5 (`thinkingBudget`). The design's one
   invented mapping; stated, receipted on both.
4. **`effort="off"`** sends the native disable (OpenAI `none`; Anthropic
   omits `thinking`; Gemini 2.5 `thinkingBudget: 0`; compat disable
   forms) and RAISES where the provider cannot disable or accepts the
   disable without honouring it: xAI, Gemini 3.x (3.7 Flash took
   `thinkingBudget: 0` and spent 58 tokens). MAP-5, extended.
5. **`thinking_budget`** maps where the wire has a budget (Anthropic
   manual class; Gemini, both classes) and RAISES elsewhere (OpenAI,
   Anthropic adaptive class, xAI, the chat dialect). On budget classes
   the budget is the spelling and `effort` stays the intent; they are
   not a conflict.
6. **`total_budget` is gone.** `Config.max_tokens` is the ceiling: on
   Anthropic's manual class the adapter adds the thinking budget to it;
   on the adaptive class it is the total (provider semantics).
7. **`summary`** is visibility: `None` = provider default; `"auto"` =
   show the thinking where a knob exists (OpenAI `summary: auto`; Gemini
   `includeThoughts: true`; Groq preset `reasoning_format: parsed`) and
   is satisfied silently where thinking is always returned (Anthropic,
   xAI); `"concise"`/`"detailed"` verbatim on OpenAI Responses, RAISE
   elsewhere. Gemini gets `includeThoughts` only when asked.
8. **Replay.** Native when the continuation state is present: Anthropic
   signed blocks, Gemini signatures (required on 3.x function calls —
   400 without), OpenAI reasoning items (`openai:reasoning_item` with
   `id` and `encrypted_content`, replayed as `{"type": "reasoning",
   "summary": [...]}` — `summary` is required even when empty, 400
   without). Without state, a `ThinkingPart` is replayed as assistant
   text on every provider (decision G); the chat dialect's
   `thinking_replay` default is `as_text`.
9. **An OpenAI reasoning item with no summary is an empty `ThinkingPart`**
   carrying its replay state, never dropped. `Usage.reasoning_tokens`
   comes from every provider's exact field.
10. **Model-class detection** (Anthropic adaptive vs manual; Gemini 2.5
    vs 3.x) is by model-name table — a table that rots; the server 400s
    loudly when wrong; `extensions` overrides.

**Why:** the reference knew one Anthropic class and one Gemini class, so
every `Reasoning` on Sonnet 5 was a 400 and every one on Gemini 3.x used
a deprecated field; it silently dropped budgets, summaries, and OpenAI
reasoning items; and it downgraded `xhigh` to `high`. The design pass
record: lm15-contract/changes/2026-09-02-reasoning-design.md.

## MAP-8 — Tool choice and structured output: no silent cells, one shape

Measured 2026-09-02 (141 cells; lm15-contract/research/tool-choice/,
research/structured-output/). The 2026-09-01 kind-aware `ToolChoice`
mapping holds; three cells were silent, and `response_format` had no
canonical shape.

1. **xAI ignores allowlists.** `tool_choice.allowed` subsets RAISE on
   xAI; a single name with `mode="required"` maps to the forced-function
   form, which held.
2. **Gemini has no parallel knob.** `parallel=False` RAISES on Gemini
   (two calls came back regardless). The MAP-6 fallback exception does
   not apply: the outcome is not observable from usage.
3. **xAI drops a forced tool next to a `response_format`** (JSON text,
   no call): the pair RAISES on xAI. Elsewhere the server decides —
   Gemini and Groq 400, OpenAI and Anthropic let the call win.
4. **INV-050: `response_format` is `{"type": "json_object"}` or
   `{"type": "json_schema", "schema", "name"?, "strict"?}`**, validated at
   `Config` construction. Provider-native spellings belong in
   `extensions`. `schema` is verbatim: lm15 never rewrites a keyword to
   make a request pass (Anthropic rejects `minimum`; OpenAI and Groq
   strict mode need every property required — their 400s are the
   contract).
5. Mapping: OpenAI Responses `text.format` (`name` defaults to
   `"response"`); chat dialect, xAI, Groq, compat `response_format
   .json_schema {name, schema, strict}`; Anthropic `output_config.format
   {type: json_schema, schema}` — `json_object` RAISES (no any-JSON
   mode); Gemini `responseMimeType` + `responseJsonSchema` or
   `responseSchema` by the `additionalProperties` rule.
6. `strict` goes verbatim where the wire has it and is satisfied where
   enforcement is always on (Anthropic, Gemini). `name` is a label, not
   a control: dropped where there is no slot.

**Why:** two canonical spellings for one intent, with the wire deciding
which, violates principle 2 of types.py; and a restriction that widens
silently is the worst failure a tool-using loop can have.

---

History: MAP-1 and MAP-2 were implicit in the reference adapters; they were
ratified as written rules on 2026-06-10 after the adversarial golden review
flagged anthropic.container, openai.code_interpreter (MAP-1) and
gemini.max_output_tokens (MAP-2) — see
`lm15-contract/goldens/REVIEW-2026-06-10.md`. MAP-3 was written on 2026-06-10
after live vLLM/SGLang/ollama testing showed the multi-end merge losing usage.
MAP-5 was written on 2026-09-01 after a reasoning-off audit found four
adapters silently omitting the disable (see
`lm15-contract/changes/2026-09-01-reasoning-off.md`). MAP-6 was written on
2026-09-01/02 from the first design pass, MAP-7 and MAP-8 on 2026-09-02 from the second, third, and fourth (`lm15-contract/playbooks/design-pass.md`).
