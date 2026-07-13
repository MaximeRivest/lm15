# Providers & models

The first three questions with any LLM library: which providers work,
how do I name a model, and what does the library know about it. This
page answers all three; every table row comes from lm15's own
manifests, and every provider behavior is contract-tested (see
[How lm15 is specified](how-lm15-is-specified.md)).

## The provider matrix

| provider string | adapter | endpoints | auth | credential |
|---|---|---|---|---|
| `openai` | `OpenAILM` (Responses API) | chat, stream, live, embeddings, files, batch, images, audio | bearer | `OPENAI_API_KEY` |
| `openai_chat` | `OpenAIChatLM` (Chat Completions) | chat, stream | bearer | `OPENAI_API_KEY` |
| `anthropic` | `AnthropicLM` | chat, stream, files, batch | x-api-key | `ANTHROPIC_API_KEY` |
| `gemini` | `GeminiLM` | chat, stream, live, embeddings, files, batch, images, audio | x-goog-api-key | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| `claude-code` | `ClaudeCodeLM` | chat, stream | local OAuth | Claude CLI login |
| `openai-codex` | `OpenAICodexLM` | chat, stream | local OAuth | Codex CLI login |
| `groq` | `OpenAIChatLM(compat="groq")` | chat, stream | bearer | `GROQ_API_KEY` |
| `openrouter` | `OpenAIChatLM(compat="openrouter")` | chat, stream | bearer | `OPENROUTER_API_KEY` |
| `ollama` | `OpenAIChatLM(compat="ollama")` | chat, stream | keyless | — |
| `vllm` | `OpenAIChatLM(compat="vllm")` | chat, stream | keyless | — |
| `sglang` | `OpenAIChatLM(compat="sglang")` | chat, stream | keyless | — |

The chat core (types, serde, request building, response parsing,
streaming, errors) is **frozen** and additive-only; the non-chat
endpoints work and are live-tested but remain **provisional** until 1.0
stable — see the [roadmap](roadmap.md). Credentials in every form
(env, explicit, rotating, subscription) are on
[Authentication](authentication.md).

Beyond the table, **any OpenAI-compatible server** is one constructor
away — a compat policy plus your URL:

```python
lm = OpenAIChatLM(api_key="...", compat="openai", base_url="https://your-gateway/v1")
```

Presets bundle known servers' wire quirks; see
[Model profiles & compat](using-model-profiles.md). Azure OpenAI,
Bedrock, and Vertex are on the roadmap (fixtures with live receipts
first, code second — that is how every provider lands).

## Naming a model

Three forms, resolved on a fixed, explainable ladder
([the router](using-the-router.md) has the details):

```python
"anthropic:claude-haiku-4-5"      # provider prefix — always works, never ambiguous
"claude-haiku-4-5"                # bare id — built-in rules or a catalog resolve it
aimo.anthropic.claude_3_5_sonnet_20240620   # a model object that knows its provider
```

Nothing is magic; ask the router what it did:

```python
print(LMRouter().resolve("claude-haiku-4-5"))
```

```output
'claude-haiku-4-5' -> provider 'anthropic' (AnthropicLM); via built-in rule prefix='claude-' — Anthropic Claude family; wire model 'claude-haiku-4-5'; key from $ANTHROPIC_API_KEY.
```

One grammar note: strings split on the *first* `:`, so a fine-tune id
like `ft:gpt-4.1:org` needs the explicit `openai:ft:gpt-4.1:org`.

## What lm15 knows about a model

Two layers, one built in and one opt-in.

**Built in: what each adapter can do.** Every adapter declares its
`capabilities` (input/output modalities, features) and endpoint support
as inspectable data — that is where the matrix above comes from:

```python
lm = AnthropicLM(api_key="...")
print(sorted(lm.capabilities.features))
print(lm.supports.batches)
```

```output
['batch', 'files', 'reasoning', 'streaming', 'tools']
True
```

**Opt-in: per-model metadata from a catalog.** lm15's core
deliberately ships no model list, no context windows, and no price
table — they change weekly and would rot in a frozen library. Instead,
catalog packages hydrate a `ModelRegistry` through a
[specified protocol](model-hydration.md). Install one
(`pip install aimo-registry`) and you get the numbers:

```python
from lm15 import ModelRegistry

registry = ModelRegistry.discover()   # finds installed catalog packages
info = registry.resolve("claude-3-5-sonnet-20240620", provider="anthropic")

print(info.inference.context_window)
print(info.inference.pricing.input_per_million,
      info.inference.pricing.output_per_million,
      info.inference.pricing.currency)
print(info.inference.pricing.estimate(input_tokens=1200, output_tokens=350))
```

```output
200000
3.0 15.0 USD
0.00885
```

(That registry run knew 6,993 models across 209 providers.) Catalog
data is **advisory by rule**: it informs routing, cost estimates, and
your own logic, but it never changes the bytes `build_request`
produces. Wire that registry into the router
(`RouterConfig(registry=...)`) and bare ids, aliases, and
provider-carrying model objects all resolve — with typed
`AmbiguousModelError`s instead of silent guesses when two providers
offer the same id.

## What lm15 deliberately does not do

No retries, no automatic tool loop, no cost ledger, no policy routing —
lm15 is the foundation layer; those belong to your application or the
opinionated libraries built on top. Each omission has a one-page
recipe in the [cookbooks](cookbooks/index.md) and a written reason in
the [design rationale](design-rationale.md).
