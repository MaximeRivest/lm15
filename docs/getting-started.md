# Getting started

lm15 is one set of types for every LLM API. You build a `Request`, you
get back a `Response` — the same two objects whichever provider
answers. This page is the whole core loop: install, one call, switch
provider, stream, use a tool. Every output below is real captured
output.

## Install

```bash
python3 -m pip install --pre lm15
```

Zero dependencies, stdlib only. (`--pre` is needed while the current
release is a pre-release; it goes away at 1.0 stable.)

## One call

Set your provider's usual env var (`ANTHROPIC_API_KEY` here) and go:

```python
from lm15 import LMRouter, Message, Request

router = LMRouter()

response = router.complete(
    Request(model="claude-haiku-4-5",
            messages=(Message.user("Say hello in exactly three words."),))
)
print(response.text)
```

```output
Hello, world friend.
```

`response` is typed all the way down: `response.usage.input_tokens` is
`14`, `response.finish_reason` is `"stop"`, and `response.message`
slots straight into the next request's `messages` to continue the
conversation.

## Any provider: change one string

```python
for model in ("claude-haiku-4-5", "gemini-3-flash-preview",
              "groq:llama-3.3-70b-versatile"):
    r = router.complete(Request(model=model,
                                messages=(Message.user("The capital of Canada, two words max."),)))
    print(f"{model:30} {r.text.strip()}")
```

```output
claude-haiku-4-5               Ottawa.
gemini-3-flash-preview         Ottawa
groq:llama-3.3-70b-versatile   Ottawa
```

The same code covers OpenAI (`gpt-4.1-mini`), a local server
(`ollama:qwen3:4b`, `vllm:...`, `sglang:...` — no key needed), and
anything else the router knows. And nothing about it is magic — ask it
to explain itself:

```python
print(router.resolve("claude-haiku-4-5"))
```

```output
'claude-haiku-4-5' -> provider 'anthropic' (AnthropicLM); via built-in rule prefix='claude-' — Anthropic Claude family; wire model 'claude-haiku-4-5'; key from $ANTHROPIC_API_KEY.
```

Prefer no router at all? The adapter classes are equally first-class:
`AnthropicLM(api_key="...")` takes the same `Request`. See
[Using the router](using-the-router.md) and
[Using the providers](using-the-a-provider.md).

## Streaming

```python
from lm15 import Result

req = Request(model="claude-haiku-4-5",
              messages=(Message.user("One short sentence about rivers."),))
result = Result(events=router.stream(req), request=req)
for text in result:
    print(text, end="", flush=True)
```

```output
Rivers flow from high elevations to the sea, shaping landscapes and sustaining life along their paths.
```

Iterate `result` for text, `result.events()` for typed chunks (tool
calls, thinking, audio, …), and read `result.response` afterwards — it
is the same `Response` a non-streaming call returns. Full recipe:
[Streaming](cookbooks/05-streaming.md).

## Tools

Tool calls arrive as data; you run the function and send the result
back. lm15 never executes anything for you.

```python
from lm15 import tool

def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"18°C and sunny in {city}"

weather = tool(get_weather)   # FunctionTool derived from the signature

messages = (Message.user("What's the weather in Montreal right now?"),)
r = router.complete(Request(model="claude-haiku-4-5", messages=messages, tools=(weather,)))

call = r.tool_calls[0]        # ToolCallPart(name='get_weather', input={'city': 'Montreal'})
messages = (*messages, r.message, Message.tool({call.id: get_weather(**call.input)}))

final = router.complete(Request(model="claude-haiku-4-5", messages=messages, tools=(weather,)))
print(final.text)
```

```output
The weather in Montreal right now is **18°C (about 64°F) and sunny**. It's a nice day!
```

Full recipe, including hand-written schemas and parallel calls:
[Function tools](cookbooks/06-function-tools.md).

## Async

Everything has an async mirror with the same shape:

```python
from lm15 import AsyncLMRouter

router = AsyncLMRouter()
response = await router.complete(
    Request(model="groq:llama-3.3-70b-versatile",
            messages=(Message.user("Say ok."),))
)
print(response.text)
```

```output
Ok.
```

## Credentials

- **Env vars** — the router reads each provider's standard variable
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
  `GROQ_API_KEY`, …). The adapter classes themselves never touch the
  environment.
- **Explicit** — `LMRouter(RouterConfig(api_keys={"anthropic": "..."}))`,
  or pass `api_key=` when constructing an adapter directly.
- **Rotating** — anywhere a key string goes, a zero-argument callable
  works too (an Azure Entra token provider, your own refresh logic);
  it is resolved once per request, so long-lived clients never go
  stale.
- **Subscriptions** — logged into the Claude or Codex CLI?
  `ClaudeCodeLM()` / `OpenAICodexLM()` use those local credentials
  directly.
- **Local servers** — `ollama:`, `vllm:`, and `sglang:` models need no
  key at all.

## Where next

- Multimodal input, structured output, reasoning, caching, batch,
  embeddings: sixteen [cookbook recipes](cookbooks/index.md), each with
  real captured output.
- The canonical types in depth:
  [Using the type system](using-the-type-system.md).
- Why the API looks the way it does:
  [Design rationale](design-rationale.md).
