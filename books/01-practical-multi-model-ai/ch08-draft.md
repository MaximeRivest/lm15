# Chapter 8: Making It Cheap

Here's something that might not be obvious yet: every time our research assistant answers a question, it re-reads the entire conversation. Not just your latest message — *everything*. The system prompt. The tool definitions. Every previous question and answer. All of it, re-sent and re-processed from scratch.

On the first turn, that's fine — there's not much to process. By the tenth turn, the model is re-reading nine complete exchanges before getting to your new question. By the twentieth turn, the input has ballooned to thousands of tokens, most of which are identical to the previous call. You're paying to process the same text over and over, and the cost grows quadratically with conversation length.

Prompt caching fixes this. The provider remembers what it's already processed and skips it on subsequent calls. The first call pays full price. Every call after that gets a discount — typically 75% to 90% off — on the repeated portion.

```python
import lm15

assistant = lm15.model("claude-sonnet-4-5",
    system="""You are a research assistant. When asked a question:
- Answer with specific facts, cite numbers and dates
- Search the web before answering factual questions
- If you're uncertain, say so explicitly""",
    tools=[search, read_url],
    prompt_caching=True,  # ← this is it
    env=".env",
)

resp = assistant("When was the Suez Canal opened?")
print(f"Cache write: {resp.usage.cache_write_tokens} tokens")

resp = assistant("How long is it?")
print(f"Cache read:  {resp.usage.cache_read_tokens} tokens")

resp = assistant("Has it been expanded recently?")
print(f"Cache read:  {resp.usage.cache_read_tokens} tokens")
```
```
Cache write: 1,250 tokens
Cache read:  1,250 tokens
Cache read:  1,890 tokens
```

The first call wrote 1,250 tokens to cache — the system prompt and tool schemas. The second call read those 1,250 tokens from cache instead of reprocessing them. The third call cached even more — the system prompt, tools, and the first two exchanges. Each turn gets faster and cheaper because the cached prefix grows.

One parameter. `prompt_caching=True`. That's the change. The rest is automatic.

## What Gets Cached

Caching works on **prefixes** — the beginning of the input that's identical across calls. On a model object with `prompt_caching=True`, lm15 places cache breakpoints that advance with the conversation:

```
Turn 1: [system ✎ | user₁]
Turn 2: [system ✓ | user₁ ✎ | assistant₁ | tool₁ | user₂]
Turn 3: [system ✓ | user₁ ✓ | assistant₁ ✓ | tool₁ ✓ | assistant₂ ✎ | tool₂ | user₃]
```

`✓` = served from cache. `✎` = the new cache breakpoint. Everything before the breakpoint was cached on a previous turn. Everything after it — the new messages — is processed at full price.

The beauty of this scheme is that it's automatic. You don't decide what to cache. lm15 caches the system prompt and tools on the first turn, then extends the cache to include each completed exchange on subsequent turns. The only new content that costs full price is whatever was added since the last call.

## The Money

Let's make this concrete. Here's a 10-turn research session without caching versus with caching, using Claude Sonnet ($3/M input tokens, cached tokens at 10% = $0.30/M):

| Turn | Without caching (input tokens) | With caching (full-price tokens) |
|---|---|---|
| 1 | 200 | 200 |
| 2 | 450 | 250 |
| 3 | 750 | 300 |
| 4 | 1,100 | 350 |
| 5 | 1,500 | 400 |
| ... | ... | ... |
| 10 | 3,800 | 650 |
| **Total** | **~13,000** | **~3,500** |

Without caching: ~13,000 tokens at $3/M = $0.039. With caching: ~3,500 full-price tokens + ~9,500 cached tokens at $0.30/M = $0.013. That's a 67% cost reduction on a 10-turn conversation. On a 50-turn agent loop — which is common for coding agents — the savings exceed 80%.

These aren't theoretical numbers. I've seen production agent loops where prompt caching cut the bill from $0.40 per task to $0.08. On a thousand tasks per day, that's $320 saved daily. `prompt_caching=True` might be the most valuable boolean in this book.

## Caching Documents

In Chapter 6, we discussed asking multiple questions about the same document. Without caching, each question re-processes the entire PDF. With per-part caching, the PDF is processed once:

```python
from lm15 import Part

contract = Part.document(
    data=open("contract.pdf", "rb").read(),
    media_type="application/pdf",
    cache=True,  # ← cache this part
)

# Question 1: processes the PDF, caches it
resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 1.", contract], env=".env")

# Question 2: PDF served from cache
resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 2.", contract], env=".env")

# Question 3: PDF served from cache
resp = lm15.complete("claude-sonnet-4-5", ["Find all liability clauses.", contract], env=".env")
```

`cache=True` on the `Part` tells the provider to cache that specific content. This works with stateless `complete()` calls — no model object needed. For a 100-page PDF that tokenizes to 50,000 tokens, the savings from three queries is roughly $0.27 on Claude (versus $0.45 without caching).

## Provider Differences

Caching exists on all three providers, but the mechanics differ:

**Anthropic** caches explicitly — lm15 adds `cache_control` markers to the request. You pay a small write cost on the first call, then read at 10% of input price. Cache entries expire after a few minutes of inactivity.

**Gemini** creates a persistent `CachedContent` resource on Google's servers. Pricing is 25% of the input rate. The cached content has a longer lifetime than Anthropic's.

**OpenAI** caches automatically — they detect repeated prefixes and cache them server-side without any action from you. `prompt_caching=True` is technically a no-op on OpenAI. Cached tokens cost 50% of the input rate.

The practical implication: `prompt_caching=True` is always safe to use. On Anthropic and Gemini, it actively reduces cost. On OpenAI, it does nothing (but doesn't hurt). Write the code once, run it on any provider.

## When to Cache

**Always cache agent loops.** An agent that makes 10+ tool-calling turns with the same system prompt and tools is the highest-ROI case. `prompt_caching=True` on the model object.

**Always cache long system prompts.** If your system prompt is more than a few hundred tokens — which it will be if it includes examples, rules, or reference material — caching pays for itself on the second call.

**Cache documents you'll query repeatedly.** `cache=True` on `Part.document(...)`. If you're asking more than one question about the same file, caching wins.

**Don't bother for one-shot calls.** If you call `complete()` once with a prompt and never repeat it, there's nothing to cache. The write cost is wasted.

The good news is that caching has essentially no downside. The write cost on the first call is small, and if there's no subsequent cache hit, you've wasted a fraction of a cent. When in doubt, enable it.

## Putting It on the Research Assistant

Our assistant now has all the pieces: tools, memory, streaming, document reading, reasoning. Adding caching is one parameter:

```python
import lm15
from lm15 import Part

def search(query: str) -> str:
    """Search the web for current information."""
    return "..."

def read_url(url: str) -> str:
    """Fetch and read a web page."""
    return "..."

assistant = lm15.model("claude-sonnet-4-5",
    system="""You are a research assistant. When asked a question:
- Search the web before answering factual questions
- Read and analyze documents the user provides
- Cite your sources with URLs
- Reason through analytical questions step by step
- Keep answers under 3 paragraphs unless asked for more""",
    tools=[search, read_url],
    temperature=0,
    prompt_caching=True,  # ← 67-80% cheaper conversations
    env=".env",
)
```

That's the same assistant from the last three chapters, plus one boolean. The system prompt, tool schemas, and conversation history are now cached across turns. A 20-turn research session that would have cost $0.15 now costs $0.04.

One more chapter. Everything we've built assumes the happy path — the network is up, the model responds, the answer makes sense. Chapter 9 handles everything else.
