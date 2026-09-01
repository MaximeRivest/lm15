# Design draft: logprobs and builtin-tool forcing

Status: RATIFIED 2026-09-01 ("yes good, go!") and IMPLEMENTED — see
`lm15-contract/changes/2026-09-01-logprobs-and-builtin-tool-forcing.md`
for the authoritative record, including three findings the live probes
added after this draft was written:

1. **Anthropic server-tool forcing WORKS** (`tool_choice
   {"type":"tool","name":"web_search"}` — live-captured). §2.3's
   "raise pending probe" for Anthropic became real support; instead,
   Anthropic raises on proper-subset allowlists, which its wire truly
   cannot express.
2. **Gemini rejects logprobs on every currently served model**
   ("Logprobs is not enabled"). The request mapping shipped doc-based;
   rejection surfaces as the provider's own InvalidRequestError.
3. **Gemini `VALIDATED` mode** maps canonical `mode="auto"` + allowed
   exactly; `allowedFunctionNames` under `AUTO` was doc-illegal and is
   fixed.

Open question 3 (openai_chat multi-name raise) resolved as: the dialect
HAS a nested `allowed_tools` form — mapped, no raise needed. Builtin
forcing raises on the dialect.

Date: 2026-09-01
Evidence: `curl-fixtures/api-references/{openai,anthropic,gemini}/pages`,
live captures under `curl-fixtures/logprobs-toolchoice-2026-09-01/` and
`lm15-contract/bodies/{openai.tool_choice_builtin,openai.tool_choice_allowed,anthropic.tool_choice_builtin,openai_chat.logprobs}/`

This draft covers two features:

1. **Logprobs** — a canonical shape for token probabilities.
2. **Builtin-tool forcing** — forcing the model to use a provider-native
   tool (web search, code execution, ...).

---

## Part 1 — Logprobs

### 1.1 What providers offer (evidence)

**Request side**

| Provider | Wire knobs | Range |
|---|---|---|
| OpenAI Responses | `top_logprobs: int` + `include: ["message.output_text.logprobs"]` | 0–20 |
| OpenAI Chat dialect | `logprobs: true` + `top_logprobs: int` | 0–20 |
| Gemini | `generationConfig.responseLogprobs: true` + `generationConfig.logprobs: int` | 0–20 |
| Anthropic | **no logprobs support** | — |

All three supporting wires express the same two facts:
"return logprobs" and "return N top alternatives per position".

**Response side**

| Provider | Wire shape |
|---|---|
| OpenAI Responses | per `output_text` block: `logprobs: [{token, bytes, logprob, top_logprobs: [{token, bytes, logprob}]}]` |
| OpenAI Chat dialect | per choice: `logprobs.content: [{token, logprob, bytes, top_logprobs: [...]}]` (plus a separate `refusal` array) |
| Gemini | per candidate: `logprobsResult: {chosenCandidates: [{token, tokenId, logProbability}], topCandidates: [{candidates: [...]}], logProbabilitySum}` plus `avgLogprobs` |

Shared semantic core: **for each decoding step, the chosen token with its
log probability, plus a ranked list of top alternative candidates.**

Provider-only extras:

- OpenAI reports `bytes` (UTF-8 byte values of the token).
- Gemini reports `tokenId` (vocabulary id).
- Gemini reports `logProbabilitySum` and `avgLogprobs` — both are
  derivable from the per-token values.
- Neither provider guarantees that the chosen token appears in the
  top-alternatives list.

### 1.2 Canonical request knob

```python
# Config gains one field:
logprobs: int | None = None
```

- `None` — do not request logprobs (omit from wire).
- `0` — request logprobs for the chosen tokens only.
- `n > 0` — also request the top-`n` alternatives per position.

Validation: `>= 0`, float-coerced like other int knobs. lm15 does **not**
hard-code the 0–20 cap; the cap is provider-owned and may change. A
too-large value maps to the provider's own `InvalidRequestError`.

**Rejected alternative — a `LogprobsConfig` object.** The concept has
exactly one degree of freedom on every wire that supports it (the number
of alternatives). An object with one field is ceremony without meaning.
Pre-1.0, if a second knob ever appears with provider evidence, we make a
clean break then. Trade-off taken: `0` carries meaning ("chosen only"),
which must be documented; the alternative (`bool` + `int` pair) creates
two fields for one concept.

**Rejected alternative — extensions-only.** Three of four wire dialects
support the same semantics; that is a genuinely portable concept and per
project principle it must be typed.

Provider mappings:

| Provider | Mapping for `logprobs=n` |
|---|---|
| OpenAI Responses | `top_logprobs: n`, add `"message.output_text.logprobs"` to `include` |
| OpenAI Chat dialect | `logprobs: true`, `top_logprobs: n` (omit `top_logprobs` when `n == 0`) |
| Gemini | `responseLogprobs: true`, `logprobs: n` |
| Anthropic | **raise `UnsupportedFeatureError`** — never silently drop |

### 1.3 Canonical output shape

Two small value types (mirrors the OpenAI SDK precedent of
`ChatCompletionTokenLogprob` / `TopLogprob`, which ports cleanly):

```python
@dataclass(frozen=True, slots=True)
class TopLogprob:
    """One scored alternative token at a decoding step."""
    token: str
    logprob: float
    bytes: tuple[int, ...] | None = None   # OpenAI only
    token_id: int | None = None            # Gemini only

@dataclass(frozen=True, slots=True)
class TokenLogprob:
    """The chosen token at one decoding step, with alternatives."""
    token: str
    logprob: float
    bytes: tuple[int, ...] | None = None
    token_id: int | None = None
    top: tuple[TopLogprob, ...] = ()       # ranked desc; chosen not guaranteed present
```

**Rejected alternative — one recursive type.** A self-referencing `top`
field that must be empty at depth 1 is an invariant the type system
cannot state. Two flat types say exactly what the wire guarantees.

### 1.4 Placement: `Response.logprobs`, not a Part field

```python
# Response gains one field:
logprobs: tuple[TokenLogprob, ...] | None = None
```

- `None` — provider did not report logprobs (the `Usage` convention).
- Present — the full decoding sequence, in generation order.

Reasoning: logprobs are **decoding telemetry**, the same category as
`usage` and `finish_reason`, which already live on `Response` and not on
`Message`. They are never sent back to a provider in history. Two of the
three wire shapes (Chat dialect, Gemini) are message-level already.

**Trade-off stated:** OpenAI Responses attaches logprobs per
`output_text` block. When a response contains more than one text block
(rare), materializing to `Response` concatenates the block lists in
document order and loses the block boundary. The boundary survives in
`provider_data` (raw payload) and in the stream (deltas carry
`part_index`). We accept this loss because putting output-only telemetry
on `TextPart` would pollute the request-side atom vocabulary for every
port and every serializer.

Derived stats (`avgLogprobs`, `logProbabilitySum`) stay in
`provider_data`: classification "intentionally ignored — derivable".
Chat-dialect `refusal` logprobs stay in `provider_data`: no canonical
refusal-text concept exists.

### 1.5 Streaming

```python
# TextDelta gains one field:
logprobs: tuple[TokenLogprob, ...] = ()
```

- OpenAI Responses: `response.output_text.delta` events carry `logprobs`
  when requested; map per delta.
- OpenAI Chat dialect: chunk `choices[].logprobs.content` maps per delta.
- Gemini: per-chunk `logprobsResult` maps to that chunk's text delta.

Stream materialization (blocking response = fully consumed stream)
concatenates delta logprobs by arrival order into `Response.logprobs`.
Empty tuple on a delta means "none in this chunk" — cheap default, no
tri-state needed because the request knob already says whether logprobs
were asked for.

### 1.6 Serde and contract impact

- `spec/types.md`: rows for `Config.logprobs`, `Response.logprobs`,
  `TextDelta.logprobs`; new `TokenLogprob` / `TopLogprob` tables.
  Omission rule: omit-empty everywhere.
- `serde.py`: `token_logprob_to/from_dict`; wired into config, response,
  and stream-event serializers. Canonical JSON uses the same field
  names (`token`, `logprob`, `bytes`, `token_id`, `top`).
- `support-matrix.json`: `logprobs` row — openai: full; openai_chat:
  full; gemini: full; anthropic: raise.
- Conformance: request-direction fixtures (knob → 3 wire forms +
  anthropic raise); response-direction fixtures from captured bodies;
  one streaming fixture per wire.
- Live captures needed under `curl-fixtures/`: OpenAI Responses with
  `include`, Gemini with `responseLogprobs` (verify the exact shape of
  `chosenCandidates` — the reference doc reuses the name `Candidate`).

---

## Part 2 — Builtin-tool forcing

### 2.1 What providers offer (evidence)

| Provider | Force a builtin? | Wire |
|---|---|---|
| OpenAI Responses | yes | `tool_choice: {"type": "web_search_preview"}` (ToolChoiceTypes); mixed allowlist via `{"type": "allowed_tools", "mode": "auto"\|"required", "tools": [...]}` |
| Anthropic | not documented | `tool_choice {"type":"tool","name":...}` is documented for client tools; server tools "have their own behavior" |
| Gemini | no | `functionCallingConfig` / `allowedFunctionNames` apply to function declarations only |

### 2.2 Design: no new field — kind-aware name resolution

`ToolChoice` already carries everything needed:

```python
ToolChoice(mode="required", allowed=("web_search",))
```

`BuiltinTool` instances have canonical names (`web_search`,
`code_execution`, ...), and INV-031 already forces every `allowed` entry
to name a tool present in `Request.tools`. The design change is a
**semantic clarification, not a new type**:

> `ToolChoice.allowed` entries may name tools of either kind. Adapters
> resolve each name against `Request.tools` and emit the kind-correct
> wire form, or raise when the wire cannot express it.

**Rejected alternative — a `builtin:` field on ToolChoice.** It would
create two ways to say "restrict to this tool", split by a distinction
(function vs builtin) that the request's tool list already encodes.
One concept, one name.

**Rejected alternative — extensions passthrough.** OpenAI's builtin
forcing is typed, documented, and requested by users; the concept
("force this declared tool") is already canonical. Only reach for
extensions where the concept itself is provider-specific.

### 2.3 Provider mappings

**OpenAI Responses** (resolve each allowed name against `Request.tools`):

| Canonical | Wire |
|---|---|
| single allowed name → FunctionTool, mode=required | `{"type": "function", "name": ...}` |
| single allowed name → BuiltinTool, mode=required | `{"type": <mapped type>}` e.g. `web_search_preview` |
| multiple allowed names (any mix), or single with mode=auto | `{"type": "allowed_tools", "mode": <mode>, "tools": [{"type":"function","name":...}, {"type":"web_search_preview"}, ...]}` |

Builtin name → wire type reuses `_OPENAI_BUILTIN_MAP` (web_search →
`web_search_preview`, code_execution → `code_interpreter`, file_search,
computer_use → `computer_use_preview`, plus passthrough for unmapped
names such as `image_generation`).

This **fixes a live correctness bug**: today a single allowed function
with `mode="auto"` emits `{"type":"function","name":...}`, which forces
the call. `mode="auto"` means "the model may also answer in text"; the
correct wire is now expressible via `allowed_tools` and the adapter
comment claiming "no portable multi-tool allowlist in Responses" is
stale — `ToolChoiceAllowed` exists.

**Anthropic**: an allowed name that resolves to a `BuiltinTool` raises
`UnsupportedFeatureError`. The API reference documents `tool_choice:
{"type":"tool"}` for client tools and explicitly defers server-tool
behavior to per-tool docs; no capture proves forcing works. Per project
principle, absence of evidence means raise, not hope. A cheap live probe
(`tool_choice: {"type":"tool","name":"web_search"}`) should be captured;
if it succeeds, flip the matrix entry with the capture as provenance.

**Gemini**: an allowed name that resolves to a `BuiltinTool` raises
`UnsupportedFeatureError` — `allowedFunctionNames` accepts function
declaration names only, and builtins (`googleSearch`, `codeExecution`)
are not addressable by `toolConfig`.

**openai_chat dialect**: Chat Completions has no hosted-tool forcing;
builtin names in `allowed` raise. Function allowlists >1 name have no
wire form either (only single `{"type":"function"}`) — today's silent
degradation should become a raise for consistency (separate decision,
flagged here).

### 2.4 Serde and contract impact

- No serde change — `ToolChoice` shape is unchanged.
- `spec/types.md`: extend the ToolChoice section with the kind-resolution
  rule and the per-provider support note.
- `spec/support-matrix.json`: `tool_choice_builtin` row — openai: full;
  anthropic: raise (pending probe); gemini: raise; openai_chat: raise.
- Conformance fixtures: openai single-builtin forcing; openai mixed
  allowed_tools; anthropic and gemini raise cases; regression fixture for
  the mode="auto" single-function fix.
- Live capture: one OpenAI Responses run with `tool_choice:
  {"type":"web_search_preview"}`; one Anthropic probe for server-tool
  forcing.

---

## Open questions for ratification

1. `Config.logprobs: int | None` with `0` = "chosen only" — accept, or
   prefer a named object despite single degree of freedom?
2. `Response.logprobs` placement (telemetry) vs `TextPart.logprobs`
   (block association) — accept the stated multi-block trade-off?
3. Should the openai_chat dialect's silent degradation of multi-name
   allowlists become a raise in the same change?
4. Anthropic server-tool forcing: probe live before or after landing the
   raise behavior?
