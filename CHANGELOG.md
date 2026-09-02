# Changelog

## Unreleased

**One truth per support fact.** The adapter-level `Capabilities` object
(`lm.capabilities`: free-text `features`, adapter-wide modalities) is
removed. Nothing in lm15 read it, it was not in the contract, and it had
already drifted (Gemini gained reasoning under MAP-7; its `features` set
said otherwise). Endpoint support lives in `lm.supports` / `lm.manifest`,
pinned by `spec/support-matrix.json`; modalities and prices are per model
on `ModelInfo`. Callers reading `lm.capabilities` get `AttributeError`.

**Three follow-ups from the independent review.**
- `LiveServerUsageEvent` (`type: "usage"`): billed tokens of a live
  response that does not end the turn — a tool-call response (75 tokens
  were vanishing in the pinned transcript) or a cancelled one (143).
  `Turn.usage` now sums every `usage` and `turn_end` event it saw; an
  interrupted turn keeps its usage instead of `None`.
- Gemini modality breakdowns (`promptTokensDetails`,
  `candidatesTokensDetails`, `responseTokensDetails`, modality `AUDIO`)
  now fill `input_audio_tokens` / `output_audio_tokens`, as OpenAI's do.
- Usage counters are declared provider-verbatim in the contract, with a
  per-provider table of what `input_tokens` and `output_tokens` include.
  Nothing changed in the numbers; the rule that they differ is now text.

**Caching on gpt-5.6+: two amendments from live probes** (MAP-6 rules 4
and 5). `retention="long"` no longer raises on the 5.6 class; it sends
`prompt_cache_retention: "24h"` like every other class (the server
accepts and echoes it; every 5.6 body already echoed 24h as default). A
placed breakpoint (`prefix="stable"`, `prefix_until_index`) now travels
with `prompt_cache_options: {mode: "explicit"}` on 5.6+: without it the
warm call still wrote the volatile suffix at 1.25x; with it the warm call
writes 0 and the cold write is exactly the marked prefix.

**Anthropic streamed tool calls assembled an unparseable input.** The
`content_block_start.input: {}` placeholder was serialised and glued in
front of the `input_json_delta` fragments. Fixed; the first streaming
tool-call body in the corpus caught it.

**Auth by composition** (contract AUTH-10, proposed). An adapter is a
dialect bound to an `AccessPolicy` value (`lm15.access`; `ProviderManifest`
is the same class under its earlier name): credential policy, auth header,
static headers, login hint, endpoint surfaces, backend variant, system
prefix, base URL. `ClaudeCodeLM` and `OpenAICodexLM` are now names for
`AnthropicLM(access=CLAUDE_CODE)` and `OpenAILM(access=OPENAI_CODEX)` and
define nothing but constructors; `XaiLM` composes its credential path and
keeps its provider wire (images, video, refusals). Every dialect and async
mirror takes `access=` and `credentials_path=`; `api_key` is optional and a
`key` policy with no key raises `NotConfiguredError` naming the env keys
(was `TypeError`). Endpoint surfaces are gated on the bound policy in the
shared drivers, so a subscription login that lacks files/batch raises
before any hook. Wire output is byte-identical (harness 13/13).

**Stream assembly never invents a tool-call name** (MAP-9, ErrorCode
`stream_assembly`). When a streamed tool call's fragments never carried a
name, the accumulator used to guess one from `Request.tools` by position
and could dispatch the wrong function silently. It now raises
`StreamAssemblyError`, carrying `partial` (the Response assembled from
everything else) and `part_index`. `ResponseStream` raises at the end of
iteration; text already yielded stays yielded. No shipped dialect
triggers this — every one names a call on its first fragment — so the
change is visible only to code that fed hand-built events into the
accumulator.

**Honest usage counters and a silent `[DONE]`** (INV-029, MAP-3):

- Adapters no longer write `0` for `input_tokens`/`output_tokens` the
  provider did not report; absent stays `None` and `total_tokens` is
  summed only when both primaries are present. Callers that summed
  usage across calls and relied on the invented zeros will now see
  `None` where the provider said nothing. Gemini is the one stated
  exception: proto3-JSON omits zero-valued fields, so an absent primary
  inside a present `usageMetadata` is a reported `0` (pinned by the
  reviewed golden `gemini.max_output_tokens`); a missing `usageMetadata`
  is all `None`.
- The Responses dialect's bare `[DONE]` terminator no longer claims
  `finish_reason="stop"`. Before, on the Codex backend it overwrote the
  `tool_call` from `response.completed` in the coalesced end event, so
  the event trace contradicted the materialized `Response`.

**Auth hardening + login primitives + doctor** (contract
`lm15-contract/spec/auth.md`, ratified 2026-08-31; fixtures
`auth/resolution.json`):

- `lm15.auth`: credential-file writes are now atomic (temp + rename,
  0600) and serialized by a cross-process advisory lock; token refresh
  is double-checked under the lock, so a refresh completed by another
  process is used instead of repeated (repeating it loses rotated
  refresh tokens). Lock contention raises the new
  `CredentialLockTimeout` (a `TimeoutError`, deliberately not
  `AuthError`). Locks live in `$XDG_CACHE_HOME/lm15/locks`
  (`$LM15_LOCK_DIR` overrides), never inside `~/.claude`/`~/.codex`.
  Stated trade-offs: the lock is advisory and lm15-cooperative only
  (foreign CLIs do not take it; the double-checked re-read is the
  mitigation), and refresh holds the lock across the network call (a
  slow refresh can stall sibling lm15 processes; the alternative
  double-spends rotated refresh tokens).
- New `lm15.authkit`: login-flow primitives for apps that own a login
  UX — PKCE (S256 only, RFC 7636 vector pinned), the RFC 8628
  device-code polling state machine (injectable clock/sleep),
  `OAuthCallbackListener` (one-shot loopback listener, 127.0.0.1
  only), and `CredentialFileStore` (locked, atomic, 0600, keyed by
  provider, serialized `mutate`; default
  `$XDG_CONFIG_HOME/lm15/credentials.json`, `$LM15_CREDENTIALS_PATH`
  overrides).
- New `lm15.doctor.explain_auth`: rung-by-rung credential-resolution
  report (selected / shadowed / absent) mirroring the router's exact
  chain; no network, secret values never rendered. Purity trade-off vs
  `resolve()`: it tests env vars for presence, so values transit
  memory but are never retained or shown.
- Contract: `spec/auth.md` (AUTH-1..9, ratified 2026-08-31),
  language-neutral resolution fixtures (mirrored at
  `conformance/auth_resolution.json`, run by
  `tests/test_auth_resolution_contract.py`), and a corpus-wide secrecy
  CI gate (`tools/check_secrecy.py`). Ports are not yet updated; they
  are formally behind the contract on this surface until they
  implement AUTH-1..AUTH-9 against `auth/resolution.json`.

**The API review (breaking — the alpha's one-time window).** A
four-lens fresh-eyes panel reviewed the public surface; findings in
`architecture-review/api-review-2026-07-13.md`. The breaking set,
batched here so the alpha churns exactly once:

- **`Result` → `ResponseStream`**, constructor positional:
  `ResponseStream(router.stream(req), req)` — no more keyword-only
  double threading. `.events()` yields canonical `StreamEvent`s (the
  `StreamChunk` second vocabulary is gone); accessors mirror
  Response's minimal set. New: `StreamAccumulator` (the shared
  push-based engine), `AsyncResponseStream` (a true async mirror —
  the old `AsyncResult`, which could not consume async streams, is
  deleted), `amaterialize_response`. `lm15/stream.py` (which aliased
  `Result` as `Stream`) is gone; both MAP-3 coalescer twins live in
  `lm15.result`.
- **Namespace curation, 161 → 107 top-level exports.** Serde pairs →
  `lm15.serde`; adapter machinery (BaseProviderLM, transports
  protocols, Credential, HttpResponse) → `lm15.providers`; error
  machinery → `lm15.errors`; router data tables (DEFAULT_RULES,
  ADAPTERS, CHAT_PRESET_ROUTES, RouteRule, PresetRoute) →
  `lm15.router`; profile/compat/SSE machinery → their modules.
  Promoted: `RETRYABLE_ERRORS` and `tool_result` to the top level.
  `derive` is exported as `derive_tool` (collision doctrine).
- **`ProviderLM` now names the callable surface** (complete/stream/…);
  the wire-mapping protocol formerly exported under that name is
  `lm15.providers.ProviderDialect`.
- **Provider strings are hyphenated**: `openai-chat` is canonical,
  `openai_chat` remains a permanent alias everywhere;
  `Resolution.provider` reports the canonical spelling.

Additive, same review:

- `Message.tool(call_id, output, is_error=False)` positional spelling;
  every wrong shape now raises a TypeError listing the accepted forms.
- `Request.tools` accepts a bare tool (1-tuple coercion).
- Errors state their cure: the messages TypeError names
  `Message.user`; `ProviderError.__str__` appends
  `(provider, HTTP status, request id)`; `UnknownModelError` uses a
  neutral prefix example and hints near-miss provider heads
  ("did you mean…").
- **`lm15.testing`**: `FakeLM` (canonical-level double),
  `FakeTransport`/`FakeResponse` (wire-level, promoted from the test
  suite). `RouterConfig(transport=...)` injects a transport into every
  LM the router builds.
- **`Retry-After` is parsed** (delta-seconds and HTTP-date) into
  `error.retry_after` on both `complete()` and `stream()` paths;
  provider-body values win.

**One front door: credential providers + universal routing.**

- **Credential providers.** `api_key` on every adapter (sync and async)
  now accepts a zero-argument callable as well as a string, resolved at
  request-build time, once per request — Azure Entra
  `get_bearer_token_provider(...)` output plugs in verbatim; rotating
  keys never go stale in long-lived clients. Acquisition stays the
  caller's job: lm15 gains no auth dependencies. `RouterConfig.api_keys`
  values may be credential providers too.
- **Subscription freshness.** `ClaudeCodeLM`/`OpenAICodexLM` (and async
  mirrors) validate the local CLI credential at construction, then
  re-resolve it per request — tokens refreshed on disk are picked up
  without rebuilding the client.
- **Credential hygiene.** `api_key` is repr-suppressed on all adapters
  (previously the plain adapters' dataclass repr included it).
- **Router rung 0 (object attribute).** A model value carrying a
  non-empty string `provider` attribute (catalog packages ship these —
  aimo's model objects) resolves directly when the provider is
  routable; duck-typed, no package named. Bare-id ambiguity disappears
  when the model object knows its provider.
- **Router preset routes.** `groq`, `openrouter`, `ollama`, `vllm`, and
  `sglang` are now routable provider strings — prefix, catalog, object
  attribute, or rules — landing on `OpenAIChatLM(compat=<preset>)` with
  the preset's pinned base_url, the server's own env-key convention
  (`GROQ_API_KEY`, `OPENROUTER_API_KEY`), and keyless placeholders for
  local servers. `Resolution` gains a `compat` field; `describe()`
  narrates the new rungs.
- New exports: `Credential`, `resolve_credential`, `PresetRoute`,
  `CHAT_PRESET_ROUTES`.

## 1.0.0a1 — 2026-06-11

**The stability promise.** The chat core — canonical types, serde, errors,
request building, response parsing, streaming — is frozen; all future changes
to it are additive (enforced mechanically by the surface ratchet and spec
drift gate). Non-chat endpoints and live sessions remain provisional; see
`lm15-contract/spec/SCOPE.md`.

What backs the promise:

- **Four independent implementations** — Python (this package), Rust, Go,
  TypeScript — each passing the identical 304-check conformance corpus
  (`lm15-dev/lm15-contract`), each live-tested against real providers
  including the full tool-calling round-trip.
- **A written, ratified spec**: 61 types, 25 vocabularies, 49 numbered
  invariants, mapping rules MAP-1..3, one omission rule, one number rule.
- **Every fixture carries provenance**; wire fixtures change only with live
  receipts; the reference implementation holds no oracle authority.
- **Measured, regenerable benchmarks**: 0 dependencies, 0.5 MiB installed,
  171 ms cold import, and faster than raw stdlib HTTP at steady state
  (connection pooling).

Changes since 0.3.0: prompt-caching fixtures recaptured (GA, no beta
header); OpenAI file inputs send `filename` (provider drift caught by the
live sweep); `FunctionTool.parameters` always emitted, `{}` round-trips
verbatim; malformed nested config objects reject instead of silently
dropping; `Result` and live sessions no longer contain any automatic
tool-execution machinery.

## 0.3.0 — 2026-06-11

Ground-up rewrite. `lm15` is now a **low-level foundation library**: one
canonical representation, exact serde, provider adapters — and nothing
opinionated. The 0.2.x high-level API (`lm15.call()`, `Model`, `Conversation`,
cost tracking, middleware, REPL) is **gone by design**; build it (or your own
take) on top. Pin `lm15==0.2.*` if you depend on the old surface.

### The canonical core
- Typed, frozen, immutable canonical model: `Request`/`Response`, `Message`,
  typed `Part`s (text, thinking, media, tool calls/results, citations),
  `Config`, `Usage`, stream events.
- Exact canonical JSON serde with written rules: one omission rule, opaque
  payloads never mutated, declared number types (`serde-rules.md`).
- Normalized error hierarchy (`AuthError` with key/credential guidance,
  `RateLimitError.retry_after`, `ContextLengthError`, ...).
- Mapping invariants written and pinned: provider-executed tools are not
  parts (MAP-1), response messages are never empty (MAP-2), a stream yields
  exactly one end event carrying finish_reason and usage (MAP-3).

### Providers
- First-party adapters: OpenAI (Responses), Anthropic, Gemini.
- `OpenAIChatLM`: the Chat Completions dialect with compat presets for
  ollama, Groq, OpenRouter, vLLM, SGLang — live-validated against Groq,
  ollama, vLLM, and SGLang.
- Native async mirrors of every adapter (`AsyncOpenAILM`, ...): same
  constructor, same canonical types, no thread-wrapping.
- Local subscription adapters: `ClaudeCodeLM` (Claude Code OAuth) and
  `OpenAICodexLM` (Codex/ChatGPT OAuth).
- Stdlib-only HTTP/1.1 sync + async transports; `websockets` is the single
  optional extra (live sessions).

### Conformance
- Behavior is pinned by the cross-language `lm15-contract` corpus: 108
  request cases, 108 reviewed response/stream goldens, error and serde
  vectors, all live-captured or hand-authored with provenance, verified by a
  language-neutral harness (`python -m lm15.vet`).
- A written spec (types, vocabularies, 48 numbered invariants) with a
  reflection-based drift gate.

### Optional model metadata
- `ModelRegistry.discover()` hydrates advisory pricing/context metadata from
  installed catalogs (entry-point group `lm15.model_catalogs`); never affects
  what adapters send.

## 0.2.0 and earlier

The previous-generation high-level SDK, developed in the `lm15-python`
repository. See its history there.
