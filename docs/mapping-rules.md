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

**Terminal frames that say nothing say nothing.** A bare terminator (`[DONE]`
on either OpenAI dialect, Anthropic `message_stop`) maps to an end event with
`finish_reason=None` and `usage=None`. It must not claim `stop`: the merge
rule lets a later non-`None` value replace an earlier one, so a `[DONE]`
carrying `stop` would overwrite the `tool_call` that `response.completed`
had already established. The event trace would then contradict the
materialized `Response`.

**`StreamEndEvent.provider_data` is the wire frame that supplied `usage`,
verbatim (the JSON object of that frame; when several frames carry usage,
the last one — the coalescer's later-fills reading); if no frame supplied
usage, the frame that supplied `finish_reason`. Bare terminators contribute nothing.
It is an escape hatch, not a canonical fact: the harness compares it for
presence and JSON type only.** Chat: the usage chunk. Anthropic: the
`message_delta` frame. Responses: `response.completed` (the event
payload's `response` object, as today). Gemini: the last chunk. xAI: same
as its wire. (Ratified 2026-09-06,
lm15-contract/changes/2026-09-06-ratification.md D9.)

**Usage counters at the wire boundary (INV-029).** An adapter never invents
`0` for a counter the provider did not send; absent stays `None` and
`Usage` auto-sums `total_tokens` only when both primaries are present. One
stated exception: Gemini's proto3-JSON wire omits zero-valued fields, so
inside a present `usageMetadata` an absent `promptTokenCount` or
`candidatesTokenCount` is a reported `0` (pinned by the reviewed golden
`gemini.max_output_tokens`: `candidatesTokenCount` absent, `totalTokenCount`
= prompt + thoughts). When `usageMetadata` itself is absent, every counter
is `None`. Secondary Gemini counters (cache, thoughts) stay verbatim: absent
is "not reported".

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

Every provider caches prompt prefixes in up to three tiers; 13 providers
were studied on 2026-09-01, six of them measured live (OpenAI both
dialects, Anthropic, Gemini, xAI, Groq) and seven from documentation only
(lm15-contract/research/caching/):

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
   any block; OpenAI: a text block, else RAISE). On the OpenAI 5.6+ class
   a placed mark travels with `prompt_cache_options: {mode: "explicit"}`:
   without the mode the warm call still wrote the volatile suffix at
   1.25x (pinned 18 tokens after reading 3066); with it the warm call
   writes 0 and the cold write shrinks to exactly the marked prefix
   (3088 → 3070). Amended 2026-09-02 on the independent review's probe
   3; no mark, no mode (explicit mode with no mark caches nothing).
   Providers without marks
   (Gemini, xAI, Groq, older OpenAI, compat servers with
   `cache_control="none"`) send nothing. The fallback is permitted by two
   conditions, both required: it spends nothing, and its outcome is
   observable in usage. It must not be extended to fields that fail
   either condition.
5. **`retention="long"`** names a specific mechanism: Anthropic `ttl:
   "1h"` (2x write); OpenAI, every class, `prompt_cache_retention:
   "24h"`; Gemini (lifetime belongs to the stored object) RAISES. The
   5.6+ class used to RAISE on a doc line about `prompt_cache_options.ttl`
   (30m only) — a different field. Every pinned 5.6 body already echoes
   `prompt_cache_retention: "24h"` as its default, and sending it answers
   200 with the same echo (review probe 2, 2026-09-02). Amended.
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
   `includeThoughts: true`; Groq preset `reasoning_format: parsed` —
   receipted 2026-09-02 on qwen3.6-27b: the default leaks a raw
   `<think>` block into `content`, `parsed` returns `message.reasoning`
   and clean content) and
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
   `thinking_replay` default is `as_text`. **`ContinuationState.provider`
   names the dialect that consumes the state (`openai`, `anthropic`,
   `gemini`, `xai` where xAI has its own wire), never the door. A Meta or
   Azure reasoning item is `openai:reasoning_item`. State replays
   verbatim on any door of that dialect; the server judges.** (Ratified
   2026-09-06, D7.)
9. **An OpenAI reasoning item with no summary is an empty `ThinkingPart`**
   carrying its replay state, never dropped. `Usage.reasoning_tokens`
   comes from every provider's exact field.
10. **Model-class detection** (Anthropic adaptive vs manual; Gemini 2.5
    vs 3.x) is by model-name table — a table that rots; the server 400s
    loudly when wrong; `extensions` overrides.
11. **Hidden thinking is a `ThinkingPart` with empty `text` and
    continuation state. There is no flag and no placeholder text.**
    Anthropic `redacted_thinking` → `ThinkingPart(text="",
    continuation=[anthropic:redacted_thinking {"data": <blob>}])`. In a
    stream: `ThinkingDelta(text="")` at `content_block_start`,
    `ContinuationDelta(part_index=i)` at `content_block_stop`. Replay is
    unchanged (the blob goes back as `redacted_thinking`). A Responses
    reasoning item with no summary is already "empty text + state" (rule
    9): one concept, not two. `[redacted]` was English presentation text,
    not provider text. (Ratified 2026-09-06, D5; `ThinkingPart.redacted`
    removed.)
12. **Thinking comes only from a typed wire field. lm15 never parses
    delimiters (`<think>`, `<reasoning>`, …) out of provider text. Where
    a server knob separates reasoning (Groq `reasoning_format: parsed`,
    rule 7), the preset sends the knob.** Bedrock runtime gpt-oss inline
    tags stay literal text; a live cell (does the runtime accept a
    parsed-reasoning knob?) is a follow-up, not done here. (Ratified
    2026-09-06, D10.)

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
   form, which held. The single cell of 2026-09-02 was repeated five
   times with fresh nonces on the review's request: 5/5 called the
   disallowed tool (`research/review-2026-09-02/`).
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

## MAP-9 — Stream assembly never invents a tool-call name

When a stream's tool-call fragments for one part index never carry a
`name`, the assembler (`lm15.result.StreamAccumulator.response`) does not
guess. It raises `StreamAssemblyError` (ErrorCode `stream_assembly`) whose
`partial` is the Response assembled from everything else that arrived —
text, thinking, media, citations, named calls, usage, the provider's finish
reason — with the unnamed call(s) left out, and whose `part_index` is the
first offending part. Wrappers surface it at the earliest point the defect
is known: `materialize_response` when called, `ResponseStream` at the end
of iteration (text already yielded stays yielded).

What lm15 does mint: a missing `id` becomes `tool_call_<index>`. That is an
lm15-owned correlator, stated, needed because Gemini sends no call ids; it
is not a guess about what the model meant.

**Assembly algorithm** (the same in every port; written down 2026-09-02
after the independent review found it lived only in code):

1. `part_index` names a **slot**, not a part. A slot may accumulate
   several kinds at once, because the chat dialect indexes text, thinking,
   and tool calls independently (a text delta and a tool-call delta both
   arrive at index 0 — pinned by `openai_chat.tool_call_unnamed` and
   `xai.streaming`).
2. Per slot, per kind, fragments concatenate in arrival order: text and
   thinking by string, audio by base64 chunk, tool-call input by string
   (parsed as JSON at the end, best-effort), tool-call `id`/`name` last
   non-`None` wins, image and citation parts replace.
3. At materialization, slots are visited in ascending index; within one
   slot the parts are emitted in this fixed kind order: **thinking, text,
   image, audio, citations, tool call**. Continuation state attached to
   the slot goes on every part emitted from it.
4. A slot that received only continuation state (no content of any kind)
   emits one empty `TextPart` carrying that state, so the state is not
   lost. A message with no parts at all emits one empty `TextPart`
   (MAP-2).
5. `finish_reason`: the end event's word wins; `None` becomes `tool_call`
   when a tool call was assembled, else `stop`; a provider `stop` next to
   an assembled tool call becomes `tool_call`.
6. The stream's `id` is the start event's id when the dialect has a start
   frame (Anthropic `message_start`, Responses `response.created`); the
   chat dialect has no start frame and its per-chunk `id` is **not**
   lifted into the Response (pinned by the reviewed `openai_chat.streaming`
   and by `xai.streaming`; a wire fact dropped by rule, stated here).
   The fields the wire withholds on the stream path (INV-051), per
   dialect, today: chat dialect — `id` (this rule) and the served `model`
   snapshot (a chunk's `model` may be the dated snapshot while the complete
   body's is the alias; the stream keeps the request's model, MAP-4);
   Gemini — `id` (no start frame carries one). Nothing else. A minted
   tool-call id is `tool_call_<index>` on both paths (parity verified
   2026-09-06 over every pinned stream body; the complete Gemini path
   used to mint `fc_<index>`, corrected the same day with the three
   frozen Gemini goldens re-reviewed). Continuation state
   is never withheld: state known at start is emitted immediately after
   `start` as a message-level `ContinuationDelta`; state known at a
   part's end is emitted at that end. No dialect emits
   `openai:response_id`, `gemini:response_id`, or `anthropic:message_id`
   continuation: `Response.id` (and `StreamStartEvent.id`) carry the id;
   server-side chaining knobs (`previous_response_id`, `conversation`)
   stay `extensions` per INV-049. (Ratified 2026-09-06, D8.)

**Why:** an unnamed call is not actionable (MAP-1), and every shipped
dialect names a call on its first fragment — pinned 2026-09-02 as four
stream cases, `<dialect>.streaming_tool_call` on OpenAI Responses, OpenAI
Chat, Anthropic, and Gemini — so a missing name is an adapter defect, not
model behaviour. The previous rule filled the name from
`Request.tools` by position — one declared tool, else the tool at the
part's rank among all parts, else the literal `"tool"`. That guess flipped
when the model emitted text before the call, and an agent loop dispatching
on the guessed name would run the wrong function with no error. That is the
silent failure MAP-8 refuses on the wire; MAP-9 refuses it in assembly.
`partial` exists so a caller who wants the turn's text can still have it,
on the same principle as the MAP-3 coalescer: never fabricate, never
discard what arrived.

Pinned by `lm15-contract/cases/openai_chat/tool_call_unnamed.json`, a
hand-built degraded OpenAI-compatible stream (arguments and id, no
`function.name`), whose golden pins the raise, the salvaged partial, and
the event trace. The `partial.finish_reason` is the provider's `tool_call`,
kept as reported even though the partial holds no call: the caller is
holding the error and knows why.

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
MAP-9 was first written on 2026-09-02 as a transcription of the
positional name guess that had lived in the accumulator since its first
version, then replaced the same day by the refusal rule after the
maintainer chose it over the guess
(`lm15-contract/changes/2026-09-02-stream-assembly-no-guess.md`).
On 2026-09-06 the ratification session added the MAP-3 `provider_data`
rule, MAP-7 rules 11–12 and the MAP-7.8 dialect sentence, and the
MAP-9.6 withheld-field list with INV-051
(`lm15-contract/changes/2026-09-06-decisions.md`,
`lm15-contract/changes/2026-09-06-ratification.md`).
