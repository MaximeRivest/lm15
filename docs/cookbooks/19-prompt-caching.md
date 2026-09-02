# Prompt caching

**Problem** — You send the same long beginning many times: a document
with many questions, or a chat that grows. Every provider can reuse that
work, but each asks differently, and one of them will bill you for cache
writes you never read back. You want one line that does the right thing
everywhere and shows you what happened.

## Recipe

Make the beginning reusable once, then ask against it. `router.cache`
uses the best tier the provider has; `cached + question` builds the
request with the boundary at the seam.

```python
from lm15 import LMRouter, Request, Message

router = LMRouter()
DOCUMENT = "Field notes. " + " ".join(f"Plot {i}: soil pH {5 + i % 4}.{i % 10}, moisture {30 + i % 50}%." for i in range(300))
prefix = Request(model="claude-sonnet-4-5", system="You answer with one number.", messages=[Message.user(DOCUMENT)])

cached = router.cache(prefix)
for question in ["What is the pH of plot 7?", "What is the moisture of plot 12?"]:
    response = router.complete(cached + question)
    print(response.text, response.usage.cache_write_tokens, response.usage.cache_read_tokens)
```

```output
…The pH of plot 7 is **8.7**. 4513 0
…42 0 4513
```

The first call wrote 4513 tokens to the cache; the second read them
back. On Anthropic that is a mark on the document block.

Same code, Gemini. Here the beginning becomes a stored object with a
lifetime and an id, and `cached + question` sends only the question.

```python
prefix = Request(model="gemini-2.5-flash", system="You answer with one number.", messages=[Message.user(DOCUMENT)])
cached = router.cache(prefix, ttl_seconds=300)
print(cached.id, cached.resource.tokens, cached.expires_at)
for question in ["What is the pH of plot 7?", "What is the moisture of plot 12?"]:
    response = router.complete(cached + question)
    print(response.text, response.usage.cache_read_tokens)
router.lm("gemini:x").cache_delete(cached.id)
```

```output
cachedContents/veegzt3p… 5300 2026-09-02T12:13:40Z
8.7 5300
42 5300
```

A chat that grows is a different intent: mark everything so far each
turn, and the next turn reads it.

```python
from lm15 import Config, CacheConfig

history = [Message.user(DOCUMENT + "\nRemember these notes.")]
for turn in ["Which plot has the highest pH?", "And its moisture?"]:
    request = Request(model="claude-sonnet-4-5", messages=history, config=Config(cache=CacheConfig(prefix="history")))
    response = router.complete(request)
    print(response.text[:40], "| write", response.usage.cache_write_tokens, "read", response.usage.cache_read_tokens)
    history += [response.message, Message.user(turn)]
```

```output
I've recorded all 300 plots with their s | write 4512 read 0
The plots with the highest pH are those  | write 191 read 4512
```

## How it works

Every provider caches in up to three tiers, measured live on 2026-09-01
(the receipts are in the contract repository under `research/caching/`):

- **Automatic.** Nothing to send. Best-effort. OpenAI, Gemini, xAI, Groq
  and most compatible servers. Gemini hit 1 time in 10 in our runs.
- **A mark on a block.** Anthropic, and OpenAI from gpt-5.6 on. Reliable
  above the model's minimum (1,024 to 4,096 tokens). Writes cost 1.25×,
  reads 0.1×.
- **A stored object.** Gemini today. Reliable, billed per token-hour
  while it exists, pinned to one model, and it owns your system prompt
  and tools, so the request may not repeat them.

`CacheConfig` names an intent, and each adapter maps it to the best tier
it has. `prefix="stable"` marks the end of system and tools;
`prefix="history"` marks the last message; `prefix_until_index=N` marks
message N. Providers with no marks fall back to their automatic tier.
That fallback is allowed because it costs nothing and the outcome is
visible in `Usage.cache_read_tokens`. Fields that name a specific
mechanism raise where it does not exist: `retention="long"`, `key`,
`resource`.

```python
try:
    router.complete(Request(model="gemini-2.5-flash", messages=[Message.user("hi")], config=Config(cache=CacheConfig(key="thread-1"))))
except Exception as e:
    print(type(e).__name__, "-", str(e)[:90])
```

```output
UnsupportedFeatureError - gemini: cache.key is not supported — GenerateContent has no cache affinity key; use cache.…
```

lm15 never creates cache state behind your back. `router.cache` is the
one call that may spend money on a stored object, and it returns the
object with its token count and expiry so you can delete it. The full
rule is MAP-6 in [mapping rules](../mapping-rules.md).

## Variations

- **The fan-out trap on gpt-5.6 and later.** With no `CacheConfig`,
  OpenAI's automatic mode marks the *latest* message. A document with
  many different questions then writes the whole prompt at 1.25× on
  every call and never reads it back (measured: five calls, five full
  writes, zero hits). `router.cache(prefix)` or `prefix="stable"` is the
  one-line fix. A growing chat is the case automatic mode was built for.
- **Turning writes off.** `CacheConfig(mode="off")` sends OpenAI's real
  off switch on gpt-5.6+ and nothing elsewhere, where writes are free:

  ```python
  request = Request(model="gpt-5.6-sol", messages=[Message.user(DOCUMENT), Message.user("pH of plot 7?")], config=Config(cache=CacheConfig(mode="off")))
  r = router.complete(request); print(r.usage.cache_write_tokens, r.usage.cache_read_tokens)
  ```

  ```output
  0 0
  ```

- **Keep the beginning stable.** A hit needs the same model, tools,
  system prompt, and earlier messages. Changing the tool list is a miss
  on every provider. Put timestamps and user names at the end.
- **A second process.** Hits are server-side; nothing to share. For a
  stored object, share `cached.id` and reattach with
  `lm.cache_get(id)`.
- **Async.** `await router.cache(prefix)`, `await router.complete(cached + question)`.

## See also

- [03 — System prompts](03-system-prompts.md)
- [02 — Conversations](02-conversations.md)
- [18 — Provider passthrough](18-provider-passthrough.md) for a provider's own cache knobs
- [Mapping rules](../mapping-rules.md), MAP-6
- [Using the type system](../using-the-type-system.md)
