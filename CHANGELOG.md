# Changelog

## Unreleased

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
