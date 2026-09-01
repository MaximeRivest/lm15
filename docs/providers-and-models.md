# Providers & models

You usually arrive knowing which model you want — Claude, GPT, Gemini,
a Llama on Groq, something running on your own machine. This page gets
you from that to a working call: what a "provider" means in lm15,
which ones are supported, how to write the model string, and what the
library can tell you about a model before you spend a token.

## The mental model

Every provider speaks its own HTTP dialect: different URLs, different
JSON shapes, different auth headers, different streaming formats. lm15
hides exactly that — you write one `Request`, and a per-provider
**adapter** translates it into that provider's dialect, byte-for-byte
correctly.

So in lm15, a *provider* is just a short string naming a dialect:
`anthropic`, `openai`, `gemini`, `groq`. And the simplest way to pick
one is to put it in front of the model name:

```python
router.complete(Request(model="anthropic:claude-haiku-4-5", ...))
router.complete(Request(model="groq:llama-3.3-70b-versatile", ...))
```

That is genuinely the whole trick. Everything below is the supporting
detail: the list of providers, the shortcuts, and the metadata.

## Which providers are supported?

| provider | string | key you need | beyond chat |
|---|---|---|---|
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | files, batch |
| OpenAI | `openai` | `OPENAI_API_KEY` | live voice, files, batch, images, speech, video |
| Google Gemini | `gemini` | `GEMINI_API_KEY` | live voice, files, batch, images, speech, video |
| xAI | `xai` | `XAI_API_KEY` or Grok subscription | images, video |
| Groq | `groq` | `GROQ_API_KEY` | — |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | — |
| ollama (local) | `ollama` | none | — |
| vLLM (local) | `vllm` | none | — |
| SGLang (local) | `sglang` | none | — |
| Claude subscription | `claude-code` | your `claude` CLI login | — |
| ChatGPT subscription | `openai-codex` | your `codex` CLI login | — |

Every provider does chat with streaming and tools — that column would
be all checkmarks, so the table only lists what each offers *beyond*
it. Setting the key is one `export`; every other way to authenticate
is on [Authentication](authentication.md).

!!! note "You may also see `openai-chat`"
    OpenAI has two wire dialects: its current **Responses API** (what
    the `openai` string uses) and the older **Chat Completions**
    dialect that half the industry adopted as a de-facto standard.
    lm15 ships both. The `openai-chat` adapter speaks Chat
    Completions — and it is the same adapter that powers the Groq,
    OpenRouter, ollama, vLLM, and SGLang rows above, each via a preset
    that knows that server's URL and quirks. You rarely type
    `openai-chat` yourself; the presets do.

Chat — the part all of this rests on — is **stable**: it is frozen by
a cross-language contract and only changes additively. The
beyond-chat endpoints work and are live-tested, but their shapes may
still move before 1.0 stable ([roadmap](roadmap.md)).

## What if my provider isn't listed?

Most hosted services and gateways speak the Chat Completions dialect.
If yours does, it already works — point the adapter at it:

```python
lm = OpenAIChatLM(api_key="...", base_url="https://your-gateway/v1")
```

[Model profiles & compat](using-model-profiles.md) covers tuning the
dialect quirks if the server has any. Azure OpenAI, Bedrock, and
Vertex are on the [roadmap](roadmap.md); new providers land in lm15
fixtures-first, with live receipts, so support is never a guess.

## How do I write the model string?

Start with the prefix form — `"provider:model-id"`. It always works
and can never be ambiguous:

```python
"anthropic:claude-haiku-4-5"
"groq:llama-3.3-70b-versatile"
"ollama:qwen3:4b"                 # only the FIRST colon splits
```

For well-known families you can drop the prefix; a small built-in rule
table recognizes `claude-*`, `gpt-*`, `gemini-*`, `grok-*`,
`o1/o3/o4*`, and the video families `sora-*` and `veo-*`:

```python
"claude-haiku-4-5"                # resolves to anthropic on its own
```

And if you ever wonder what happened, ask — resolution is a lookup you
can read, not magic:

```python
print(LMRouter().resolve("claude-haiku-4-5"))
```

```output
'claude-haiku-4-5' -> provider 'anthropic' (AnthropicLM); via built-in rule prefix='claude-' — Anthropic Claude family; wire model 'claude-haiku-4-5'; key from $ANTHROPIC_API_KEY.
```

One honest gotcha: the string splits on the *first* `:`, so a
fine-tune id like `ft:gpt-4.1:org` needs the explicit
`openai:ft:gpt-4.1:org`. The full resolution ladder — including model
*objects* that carry their own provider, from catalog packages like
aimo — is in [the router guide](using-the-router.md).

## Will it work before I try it?

Adapters describe themselves. Instead of hunting through docs for
"does Anthropic support batch?", ask the object:

```python
lm = AnthropicLM(api_key="...")
print(sorted(lm.capabilities.features))
print(lm.supports.batches)
```

```output
['batch', 'files', 'reasoning', 'streaming', 'tools']
True
```

Asking for something a provider can't do raises a typed
`UnsupportedFeatureError` immediately — you never find out via a
confusing HTTP 400.

## What does it cost? What's the context window?

Here lm15 is deliberately humble: the core library ships **no** model
list, no context windows, no price table. Those numbers change weekly,
and a frozen library that pretended to know them would quietly rot.

Instead, *catalog packages* supply the numbers through a
[specified protocol](model-hydration.md), and lm15 gives them one
home. Install one — `pip install aimo-registry` — and ask:

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

That run knew 6,993 models across 209 providers. Hand the registry to
the router (`RouterConfig(registry=...)`) and bare model ids and
aliases resolve through it too — with a typed `AmbiguousModelError`
naming the fix when two providers offer the same id, never a silent
guess.

One rule keeps this trustworthy: catalog data is **advisory**. It
informs routing, cost estimates, and your own logic — it never changes
the bytes lm15 puts on the wire.

## What lm15 won't do for you (on purpose)

If you are coming from LiteLLM or LangChain, you may expect built-in
retries, an automatic tool-execution loop, a cost ledger, or fallback
routing. lm15 has none of these — deliberately. It is the foundation
layer; policy belongs to your application or to the opinionated
libraries built on top of it. Every omission has a one-page answer in
the [cookbooks](cookbooks/index.md) (retries, budgets, fallbacks are a
few lines each) and a written reason in the
[design rationale](design-rationale.md).

## Where next

- Set up credentials properly: [Authentication](authentication.md)
- The resolution ladder in full: [Using the router](using-the-router.md)
- Actually calling models: back to [Getting started](getting-started.md),
  or the [cookbooks](cookbooks/index.md)
