# Authentication

Every way lm15 takes credentials, from zero-setup to rotating tokens.
Find yourself first — most people need exactly one section:

- **Just trying things out?** Export one env var and you're done —
  [first section](#environment-variables-the-default).
- **Building an app or service?** Pass keys explicitly, from wherever
  you keep secrets — [explicit keys](#explicit-keys).
- **Behind Azure or an enterprise setup where tokens expire?**
  [Rotating credentials](#rotating-credentials-token-providers).
- **On a Claude or ChatGPT plan, no API account?**
  [Subscriptions](#subscriptions-claude-code-codex-cli).
- **Everything local?** [No key at all](#keyless-local-servers).

One rule sits behind all of them, and it explains everything else on
this page: **lm15 places the credential you provide on the wire, in
the provider's dialect, and does nothing else.** It never fetches,
refreshes, or stores tokens for you, and it depends on no auth SDKs —
[the design rationale](design-rationale.md#why-api_key-accepts-a-callable-and-lm15-still-has-no-auth-dependencies)
explains why that restraint is the feature.

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

When keys live in a secrets manager rather than the shell, pass them
yourself — an explicit key always beats the environment, and passing
`env={}` as well guarantees the environment is never even looked at
(useful in tests, and it is how lm15's own test suite runs):

```python
from lm15 import AnthropicLM, LMRouter, RouterConfig

router = LMRouter(RouterConfig(env={}, api_keys={"anthropic": "sk-ant-..."}))

lm = AnthropicLM(api_key="sk-ant-...")   # or skip the router entirely
```

`api_keys` is repr-suppressed, like every credential field in lm15.

## Rotating credentials (token providers)

Some credentials aren't static strings. Azure Entra tokens expire
after about an hour; enterprises rotate keys on a schedule; OAuth
tokens refresh themselves. If your process runs for days, a key read
once at startup *will* eventually be stale — and the failure shows up
as a confusing 401 at 3 a.m.

lm15's answer: anywhere a key string goes, a **zero-argument callable
returning a string** works too. The adapter calls it when it builds
each request, so whatever your callable returns *now* is what goes on
the wire *now*. Refreshing and caching stay your callable's business —
which means the ecosystem's existing token providers plug in
unchanged:

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

If you use Claude Code or the Codex CLI, you already have working
credentials on disk — no API platform account needed. These adapters
use that login directly, and usage bills to your existing subscription
plan rather than pay-per-token:

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

Worth knowing before a security review asks:

- **Your key is never printed.** `print()` an adapter, log an error,
  render a traceback — credential material appears in none of them,
  whichever form it takes.
- **Nothing is discovered behind your back.** Adapters read only what
  you passed. Env pickup happens in exactly one place (the router),
  and `resolve()` will tell you which variable it would use.
- **The right header, every time.** Where each credential goes on the
  wire (`Authorization: Bearer`, `x-api-key`, `x-goog-api-key`, …) is
  contract-tested per provider —
  [How lm15 is specified](how-lm15-is-specified.md).
- **Zero auth dependencies, forever.** For Azure/Bedrock/Vertex-style
  delegated auth you bring the token provider or a signing transport;
  lm15 provides the seam.

## When it goes wrong

Credential problems surface early and loudly: a missing key fails when
you *build* the client, not twenty minutes into a batch run. The
errors are typed (`MissingCredentialError` from the router,
`NotConfiguredError` / `AuthError` from adapters — all catchable under
`LM15Error`), and each one states its own fix:

```output
MissingCredentialError: no API key found for provider 'anthropic'.
Set ANTHROPIC_API_KEY in the environment, or pass
RouterConfig(api_keys={'anthropic': "..."}).
```

Subscription errors carry the re-login hint instead (`run \`claude\`
and use /login`), because there is no env var to set.
