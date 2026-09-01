# Streaming

**Problem** — You want tokens on screen as the model produces them, but
you also want the complete `Response` at the end — without parsing SSE,
without accumulating deltas by hand, and without writing the code twice
for streaming and non-streaming paths.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

`router.stream()` yields typed `StreamEvent`s. Wrap them in
`ResponseStream` and the stream becomes something you can iterate as
text:

```python
from lm15 import AsyncLMRouter, AsyncResponseStream, LMRouter, Message, Request, ResponseStream

router = LMRouter()
req = Request(
    model="claude-haiku-4-5",
    messages=(Message.user("Name three rivers in Quebec, one per line, names only."),),
)
result = ResponseStream(router.stream(req), req)
for text in result:
    print(text, end="", flush=True)
```
```output
Saint Lawrence River
Ottawa River
Saguenay River
```

While text streamed past, `ResponseStream` was accumulating it. After
the loop, the materialized response is already there — `.text` does not
re-call the API:

```python
print(result.text)
print(result.finish_reason, result.usage)
```
```output
Saint Lawrence River
Ottawa River
Saguenay River
stop Usage(input_tokens=20, output_tokens=16, total_tokens=36, …)
```

Iterating a `ResponseStream` yields only text. For everything else —
thinking, tool calls, images, audio — iterate `.events()` instead: the
same canonical `StreamEvent`s the raw stream carries, one vocabulary
everywhere, still accumulating toward `.response` as they pass through:

```python
req = Request(
    model="claude-haiku-4-5",
    messages=(Message.user("Say 'streams are lazy' and nothing else."),),
)
result = ResponseStream(router.stream(req), req)
for event in result.events():
    if event.type == "delta":
        print("delta", event.delta.type, repr(getattr(event.delta, "text", None)))
    else:
        print(event.type)
```
```output
start
delta continuation None
delta text 'streams'
delta text ' are lazy'
end
```

(Note the discriminated union at work: deltas are typed — this
Anthropic stream carried a `continuation` delta alongside the text,
which is why the code switches on `event.delta.type` instead of
assuming `.text` exists.)

You do not have to iterate at all. Touching `.response` (or `.text`,
`.usage`, …) drains the stream and blocks until it is done. What you
get is an ordinary `Response` — the same type, the same fields, as
`router.complete()` returns:

```python
result = ResponseStream(router.stream(req), req)
response = result.response          # drains the stream, blocks
print(type(response).__name__)
print(response.message.parts)
print(response.finish_reason, response.usage)
```
```output
Response
(TextPart(text='streams are lazy', continuation=(), type='text'),)
stop Usage(input_tokens=18, output_tokens=6, total_tokens=24, …)
```

The stream contract is strict. Every lm15 stream is a `start` event,
zero or more `delta` events, and **exactly one** `end` event, last.
This is rule MAP-3, and it holds even when the provider's wire format
emits several terminal frames:

```python
events = list(router.stream(req))
print("first:", events[0].type, "last:", events[-1].type)
print("end events:", sum(e.type == "end" for e in events))
```
```output
first: start last: end
end events: 1
```

`finish_reason` and `usage` ride on that single end event, which is why
`ResponseStream` can always hand you a complete `Response`.

## How it works

`router.stream()` resolves the model string, opens the provider's SSE
connection, and translates each wire frame into a `StreamEvent` — the
same vocabulary (`TextDelta`, `ThinkingDelta`, `ToolCallDelta`,
`ImageDelta`, …) across OpenAI, Anthropic and Gemini. See
[using the router](../using-the-router.md) for resolution; the event
vocabulary lives in `lm15.types`.

Providers disagree about endings: OpenAI sends a finish-reason chunk,
then a usage-only chunk, then `[DONE]`; Anthropic sends
`message_delta` plus `message_stop`. Adapters are stateless and emit
one end event per terminal frame; `coalesce_stream()` (applied inside
every provider's `stream()`) merges them into the single final
`StreamEndEvent` you observed above. If a stream errors or is cut off
mid-flight, no end event is fabricated — absence of `end` means the
stream did not finish.

`ResponseStream` is a thin skin over `lm15.result.StreamAccumulator`,
the push-based engine that folds events into `Message` parts — the
same engine behind `materialize_response()` and the async mirror, so
every path assembles identically. It executes nothing: tool-call
deltas surface as data (recipe [06](06-function-tools.md)), and any
run-tools-and-continue loop is yours to write. There is no retry, no
timeout policy, no reconnection — lm15 hands you the events; policy is
the layer above.

## Variations

- **Async.** `AsyncResponseStream` mirrors the sync class over
  `AsyncLMRouter.stream()`; `response()` is a method there (consuming
  an async stream is awaitable work). This ran against Gemini:

  ```python
  import asyncio

  async def main():
      arouter = AsyncLMRouter()
      req = Request(
          model="gemini-3-flash-preview",
          messages=(Message.user("Count from 1 to 5, comma-separated."),),
      )
      result = AsyncResponseStream(arouter.stream(req), req)
      async for text in result:
          print(text, end="", flush=True)
      print()
      print((await result.response()).finish_reason)

  asyncio.run(main())
  ```
  ```output
  1, 2, 3, 4, 5
  stop
  ```

  The MAP-3 guarantee is identical; both coalescer twins live in
  `lm15.result`.

- **Raw events, no wrapper.** Filtering `router.stream(req)` yourself
  is fine when you only want one delta type. `ResponseStream` earns
  its keep when you want the materialized `Response` afterward.

- **One-shot.** `materialize_response(events, request)` (and
  `amaterialize_response`) skip the wrapper entirely — events in,
  `Response` out.

- **Delta granularity differs by provider.** OpenAI streams a few
  tokens per delta; Gemini sends larger sentence-sized deltas;
  Anthropic sits in between. Your code should not depend on chunk
  boundaries.

- **Replay.** `lm15.result.response_to_events(response)` converts a
  complete `Response` back into a stream — useful for testing stream
  consumers without a network (recipe
  [15](16-errors-and-testing.md)).

## See also

- [01 — Your first request](01-first-request.md) — keys and the router front door.
- [06 — Function tools](06-function-tools.md) — tool-call parts surfaced by `ResponseStream`.
- [10 — Audio, video & reasoning models](10-audio-video-reasoning.md) — thinking deltas.
- [16 — Errors, retries & testing](16-errors-and-testing.md) — stream errors, offline replay.
- [Using the router](../using-the-router.md) — resolution rules and `RouterConfig`.
