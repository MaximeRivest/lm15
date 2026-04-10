# Chapter 9: When Things Break

It's 2 AM and your research assistant is processing a batch of 500 document queries. At query 247, Anthropic returns a 429 — rate limit exceeded. Your script crashes. The first 246 results are fine, sitting in memory. The remaining 254 are lost. You restart, process 246 duplicates, hit the rate limit again at query 493, and briefly consider a career in carpentry.

This chapter is about not having that night.

Everything we've built so far works beautifully on the happy path. The API responds, the model generates useful text, the tools work, the network holds. In production, every one of those assumptions fails regularly — and the difference between a toy project and a real tool is how you handle the failures.

## Retries

The most common failures are transient. The provider is busy (429), the server hiccupped (500), or the network timed out. These fix themselves — you just need to wait and try again. `retries=` on a model object does this automatically:

```python
import lm15

assistant = lm15.model("claude-sonnet-4-5",
    system="You are a research assistant.",
    retries=3,  # ← retry up to 3 times on transient errors
    env=".env",
)

resp = assistant("What caused the 2008 financial crisis?")
```

If the call fails with a rate limit, timeout, or server error, lm15 waits (200ms, then 400ms, then 800ms — exponential backoff) and tries again. After three retries, if it's still failing, the exception is raised. Permanent errors — bad API key, malformed request — fail immediately. Retrying a 401 is pointless and lm15 knows that.

For `complete()`, retries aren't built in. You write the loop yourself:

```python
import time
import lm15
from lm15.errors import RateLimitError, ServerError, TimeoutError

def resilient_complete(max_retries=3, **kwargs):
    for attempt in range(max_retries + 1):
        try:
            return lm15.complete(**kwargs)
        except (RateLimitError, ServerError, TimeoutError):
            if attempt == max_retries:
                raise
            time.sleep(0.2 * (2 ** attempt))

resp = resilient_complete(model="gpt-4.1-mini", prompt="Hello.", env=".env")
```

Ten lines. No framework. This is the same logic that `retries=3` uses internally.

## Prefill: Forcing Output Format

Models are surprisingly bad at following format instructions consistently. Ask for JSON and you'll get JSON — most of the time. One call in fifty, the model prepends "Sure, here's the JSON:" before the actual data, and your parser crashes.

Prefill eliminates this by seeding the response with a string you provide. The model continues from where you left off:

```python
import lm15

resp = lm15.complete("claude-sonnet-4-5",
    "Extract the name and age: 'Alice is 30 years old.'",
    system="Return valid JSON with keys 'name' and 'age'. Nothing else.",
    prefill="{",
    env=".env",
)
print(resp.text)
```
```
{"name": "Alice", "age": 30}
```

The response started with `{` because you forced it. The model couldn't write a preamble — it was already inside the JSON object. This is one of those techniques that sounds minor and turns out to prevent half your production parsing failures.

Prefill works best with Anthropic. OpenAI and Gemini support it to varying degrees. Always test with your specific model.

## Error Handling

Chapter 1 introduced the error hierarchy. Let's put it to work. In a production system, different errors demand different responses:

```python
import lm15
from lm15.errors import (
    AuthError,
    RateLimitError,
    ContextLengthError,
    InvalidRequestError,
    ServerError,
    ULMError,
)

def answer_question(question: str) -> dict:
    try:
        resp = lm15.complete("gpt-4.1-mini", question, env=".env")
        return {"status": "ok", "text": resp.text, "tokens": resp.usage.total_tokens}

    except AuthError:
        # Configuration problem — nothing the user can fix at runtime
        return {"status": "error", "message": "API key invalid or expired"}

    except RateLimitError:
        # Transient — the caller should retry after a delay
        return {"status": "retry", "message": "Rate limited, try again in 30 seconds"}

    except ContextLengthError:
        # Input too long — need to truncate or use a different model
        return {"status": "error", "message": "Input too long for this model"}

    except InvalidRequestError as e:
        # Something wrong with the request — log it for debugging
        return {"status": "error", "message": f"Bad request: {e}"}

    except ServerError:
        # Provider is down — fall back or retry
        return {"status": "retry", "message": "Service temporarily unavailable"}

    except ULMError as e:
        # Catch-all for anything else lm15 can throw
        return {"status": "error", "message": f"Unexpected: {e}"}
```

The key insight: not all errors are equal. `AuthError` means your configuration is wrong — no amount of retrying will fix it. `RateLimitError` means wait and try again. `ContextLengthError` means truncate the input or pick a bigger model. Each error type maps to a different recovery strategy.

## Fallback Models

When a provider is down, switch to another. This is lm15's fundamental advantage — you're not locked into one provider's SDK, so failing over is trivial:

```python
import lm15
from lm15.errors import ULMError

MODELS = ["claude-sonnet-4-5", "gpt-4.1-mini", "gemini-2.5-flash"]

def complete_with_fallback(prompt: str, **kwargs) -> lm15.LMResponse:
    for model in MODELS:
        try:
            return lm15.complete(model, prompt, **kwargs)
        except ULMError as e:
            print(f"  {model} failed: {e}")
            continue
    raise RuntimeError(f"All models failed: {MODELS}")

resp = complete_with_fallback("What is DNA?", env=".env")
print(f"Answered by: {resp.model}")
```

Try the preferred model. If it fails, try the next one. In production, you'd log which model responded and alert on fallbacks — a silent fallback is a hidden degradation.

A more sophisticated version ranks models by capability:

```python
def smart_fallback(prompt: str, needs_reasoning=False, **kwargs):
    if needs_reasoning:
        models = ["claude-sonnet-4-5", "gpt-4.1-mini"]  # strong reasoners first
    else:
        models = ["gemini-2.5-flash", "gpt-4.1-mini", "claude-sonnet-4-5"]  # cheap first
    return complete_with_fallback(prompt, models=models, **kwargs)
```

## Token Budget Guards

In agent loops — where the model decides what to do and your code executes it — there's no natural stopping point. The model can call tools indefinitely, growing the conversation and the cost with each turn. A budget guard puts a ceiling on spending:

```python
import lm15

assistant = lm15.model("claude-sonnet-4-5",
    system="You are a research assistant.",
    tools=[search, read_url],
    prompt_caching=True,
    retries=2,
    env=".env",
)

TOKEN_BUDGET = 50_000
total = 0

resp = assistant("Research the history of the Panama Canal.")
total += resp.usage.total_tokens

while resp.finish_reason == "tool_call" and total < TOKEN_BUDGET:
    results = execute_tools(resp.tool_calls)
    resp = assistant.submit_tools(results)
    total += resp.usage.total_tokens
    print(f"  Turn {len(assistant.history)}: {resp.usage.total_tokens} tokens (total: {total})")

if total >= TOKEN_BUDGET:
    print(f"⚠️ Budget exceeded at {total} tokens")
else:
    print(resp.text)
```

Without this guard, a confused model that keeps calling `search` with variations of the same query can burn through $5 of tokens before you notice. With the guard, it stops at your limit and you get a clean exit.

## Defensive Output Handling

Never trust model output. Even with `system=`, `prefill=`, and `temperature=0`, the model can produce unexpected output. Always validate:

```python
import json
import lm15

def classify_review(review: str) -> str | None:
    """Classify a review as POSITIVE, NEGATIVE, or MIXED. Returns None on failure."""
    resp = lm15.complete("gpt-4.1-mini",
        prompt=review,
        system="Classify sentiment. Return JSON: {\"label\": \"POSITIVE|NEGATIVE|MIXED\"}",
        prefill="{",
        temperature=0,
        max_tokens=20,
        env=".env",
    )
    try:
        data = json.loads(resp.text)
        label = data.get("label", "").upper()
        if label in {"POSITIVE", "NEGATIVE", "MIXED"}:
            return label
    except (json.JSONDecodeError, AttributeError):
        pass
    return None
```

This function uses every defensive technique: `system=` for format, `prefill="{"` to prevent preamble, `temperature=0` for consistency, `max_tokens=20` to prevent runaway output, JSON parsing with error handling, and label validation against an allowed set. If anything unexpected happens — malformed JSON, unexpected label, parsing error — it returns `None` and the caller decides what to do.

`max_tokens=20` is particularly important here. Without it, a misbehaving model could generate 4,000 tokens of explanation before the JSON. With it, you lose 20 tokens in the worst case.

## A Resilient Research Session

Here's our research assistant with everything from this chapter bolted on:

```python
import lm15

def search(query: str) -> str:
    """Search the web for current information."""
    try:
        return actual_search(query)
    except Exception as e:
        return f"Search failed: {e}"  # error in return value, not exception

assistant = lm15.model("claude-sonnet-4-5",
    system="""You are a research assistant. When asked a question:
- Search the web before answering factual questions
- Cite your sources
- If uncertain, say so
- Keep answers under 3 paragraphs""",
    tools=[search],
    temperature=0,
    prompt_caching=True,
    retries=2,  # ← retry on transient failures
    env=".env",
)
```

Three additions to the assistant we've been building since Chapter 4: `retries=2` handles transient failures silently. `prompt_caching=True` keeps costs down on long conversations. And the `search` function catches its own errors instead of crashing the tool-call loop.

This assistant won't crash at 2 AM. It won't silently eat your budget. It won't blow up when Anthropic returns a 500. It will work — quietly, reliably, cheaply — while you sleep.

One chapter left. We've been hardcoding model names this whole time: `"claude-sonnet-4-5"`, `"gpt-4.1-mini"`, `"gemini-2.5-flash"`. Those names will change. Chapter 10 shows you how to discover what's available and adapt.
