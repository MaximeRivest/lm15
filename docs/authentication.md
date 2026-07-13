# Authentication

Every way lm15 takes credentials, from zero-setup to rotating tokens.
The rule behind all of them: **lm15 places the credential you provide
on the wire, in the provider's dialect, and does nothing else.** It
never fetches, refreshes, or stores tokens for you, and it depends on
no auth SDKs — see
[the design rationale](design-rationale.md#why-api_key-accepts-a-callable-and-lm15-still-has-no-auth-dependencies).

## Environment variables (the default)

The router reads each provider's standard variable. The adapter
classes themselves **never** touch the environment — env pickup happens
only in the router, explicitly and inspectably.

| provider string | env var | get a key at |
|---|---|---|
| `openai`, `openai_chat` | `OPENAI_API_KEY` | platform.openai.com/api-keys |
| `anthropic` | `ANTHROPIC_API_KEY` | console.anthropic.com |
| `gemini` | `GEMINI_API_KEY`, then `GOOGLE_API_KEY` | aistudio.google.com/apikey |
| `groq` | `GROQ_API_KEY` | console.groq.com/keys |
| `openrouter` | `OPENROUTER_API_KEY` | openrouter.ai/keys |
| `ollama`, `vllm`, `sglang` | — (keyless, placeholder sent) | — |
| `claude-code`, `openai-codex` | — (local CLI credential) | — |

`resolve()` records *which* variable would be read, never the value:

```python
print(LMRouter().resolve("claude-haiku-4-5"))
```

```output
'claude-haiku-4-5' -> provider 'anthropic' (AnthropicLM); via built-in rule prefix='claude-' — Anthropic Claude family; wire model 'claude-haiku-4-5'; key from $ANTHROPIC_API_KEY.
```

## Explicit keys

Pass the key yourself and the environment is never consulted — pass
`env={}` too and the router is fully hermetic (this is how lm15's own
tests run):

```python
from lm15 import AnthropicLM, LMRouter, RouterConfig

router = LMRouter(RouterConfig(env={}, api_keys={"anthropic": "sk-ant-..."}))

lm = AnthropicLM(api_key="sk-ant-...")   # or skip the router entirely
```

`api_keys` is repr-suppressed, like every credential field in lm15.

## Rotating credentials (token providers)

Anywhere a key string goes, a **zero-argument callable returning a
string** works too. The adapter calls it at request-build time, once
per request, so a client that lives for days never holds a stale token.
Refreshing and caching are your callable's business — which means the
ecosystem's existing token providers plug in unchanged:

```python
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from lm15 import OpenAIChatLM

lm = OpenAIChatLM(
    api_key=get_bearer_token_provider(          # returns () -> str
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    ),
    base_url="https://YOUR-RESOURCE.openai.azure.com/openai/v1",
)
```

Or your own logic — anything callable:

```python
def credential() -> str:
    return read_key_from_your_vault()

router = LMRouter(RouterConfig(api_keys={"anthropic": credential}))
```

## Subscriptions (Claude Code / Codex CLI)

Already logged in to the `claude` or `codex` CLI? Those local OAuth
credentials work directly — no API key, billed to the subscription:

```python
from lm15 import ClaudeCodeLM, OpenAICodexLM

lm = ClaudeCodeLM()      # ~/.claude/.credentials.json
lm = OpenAICodexLM()     # ~/.codex/auth.json
```

The credential is validated at construction (missing or unrefreshable
credentials raise typed errors that say which CLI login to run), then
re-resolved on every request: tokens refreshed on disk — by the CLI, by
another process, or by lm15's own expiry refresh — are picked up
without rebuilding the client. Both adapters are also routable
(`claude-code:...`, `openai-codex:...`).

## Keyless local servers

`ollama:`, `vllm:`, and `sglang:` model strings need no configuration —
the router sends the placeholder these servers expect (`"ollama"` /
`"EMPTY"`). If your local server *is* locked down, set the key
explicitly via `api_keys`.

## What lm15 guarantees

- Credential material never appears in a repr, an error message, or an
  exception chain — whichever form it takes.
- Adapters read credentials only from what you passed; nothing is
  discovered behind your back.
- Wire placement is contract-tested per provider dialect
  (`Authorization: Bearer`, `x-api-key`, `x-goog-api-key`, …) — see
  [How lm15 is specified](how-lm15-is-specified.md).
- Zero auth dependencies, forever: for Azure/Bedrock/Vertex-style
  delegated auth you bring the token provider or a signing transport;
  lm15 provides the seam.

## When it goes wrong

Missing keys fail at construction, not mid-request, and the errors are
typed (`MissingCredentialError` from the router, `NotConfiguredError` /
`AuthError` from adapters — all under `LM15Error`) and self-explaining:

```output
MissingCredentialError: no API key found for provider 'anthropic'.
Set ANTHROPIC_API_KEY in the environment, or pass
RouterConfig(api_keys={'anthropic': "..."}).
```

Subscription errors carry the re-login hint instead (`run \`claude\`
and use /login`), because there is no env var to set.
