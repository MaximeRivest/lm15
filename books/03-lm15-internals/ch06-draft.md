# Chapter 6: Streaming Internals

You've seen streaming from the user side — `for text in stream.text` — and from the transport side — `readline()` yielding bytes. This chapter connects them. Between raw SSE bytes and the `StreamChunk` objects you iterate over, there are four stages, and each one transforms the data.

## The Pipeline

```
Transport.stream()    →  raw bytes (lines)
SSE parser (sse.py)   →  SSEEvent (event name + data string)
Adapter.stream()      →  StreamEvent (typed deltas, usage, errors)
Stream (stream.py)    →  StreamChunk (text, thinking, tool_call, finished)
```

The transport yields byte lines. The SSE parser groups them into events. The adapter parses the JSON in each event and yields `StreamEvent` objects. The `Stream` class translates those into the `StreamChunk` objects users see.

Each stage knows nothing about the stages above or below it. The SSE parser doesn't know about JSON. The adapter doesn't know about `StreamChunk`. The `Stream` class doesn't know about HTTP. Clean boundaries.

## Stream: The Translator

`stream.py` defines the `Stream` class — an iterator that wraps the adapter's `StreamEvent` iterator and produces `StreamChunk` objects. It's 183 lines, and the core is the `__next__` method, which is essentially a big `match` on `event.type`.

The translation:

- `StreamEvent(type="start")` → captured internally (saves `id` and `model`), nothing yielded
- `StreamEvent(type="delta", delta={type: "text", text: "..."})` → `StreamChunk(type="text", text="...")`
- `StreamEvent(type="delta", delta={type: "thinking", text: "..."})` → `StreamChunk(type="thinking", text="...")`
- `StreamEvent(type="delta", delta={type: "tool_call", input: "..."})` → `StreamChunk(type="tool_call", ...)`
- `StreamEvent(type="end")` → `StreamChunk(type="finished", response=LMResponse(...))`
- `StreamEvent(type="error")` → raises a typed exception

Along the way, the `Stream` accumulates all text chunks, thinking chunks, and tool call fragments. When the stream ends (or when someone accesses `stream.response`), it assembles these accumulated parts into a complete `LMResponse` — the same object you'd get from `complete()`.

## Tool Call Accumulation

Tool call handling is the most complex part of streaming. Unlike text (which arrives as simple string chunks), tool calls arrive as JSON fragments:

```
delta: {"type": "tool_call", "id": "call_1", "name": "search"}
delta: {"type": "tool_call", "input": "{\"quer"}
delta: {"type": "tool_call", "input": "y\": \"tcp\"}"}
```

The name and ID arrive in the first delta. The arguments arrive as JSON string fragments that need to be concatenated. The `Stream` class maintains two dicts:

- `_tool_call_raw: dict[int, str]` — accumulated JSON string per tool index
- `_tool_call_meta: dict[int, dict]` — name, ID, and parsed input per tool index

On each tool call delta, the raw JSON fragment is appended. Then `_parse_json_best_effort()` tries to parse the accumulated string. If it's valid JSON, great — the parsed dict becomes the tool call's `input`. If it's partial JSON (the common case mid-stream), the method returns `{"partial_json": raw_string}`.

This means `event.input` on a `StreamChunk(type="tool_call")` transitions from partial to complete during streaming. Early events have `{"partial_json": "{\"quer"}`. The final event has `{"query": "tcp"}`. The user sees the tool call being built in real time.

## Response Materialization

When the stream ends, `_materialize_response()` assembles all accumulated parts into an `LMResponse`:

1. If there are thinking parts, join them into one `Part.thinking(text=...)`
2. If there are text parts, join them into one `Part.text_part(text=...)`
3. For each accumulated tool call, build a `Part.tool_call(id, name, input)`
4. Wrap everything in a `Message(role="assistant", parts=...)` and an `LMResponse`

The `usage` and `finish_reason` come from the `end` event. If the stream ended abnormally (no `end` event), defaults are used — `Usage()` and `"stop"`.

## The on_finished Callback

`Model.stream()` needs to record the streamed response in conversation history. But `Model` doesn't own the `Stream` — the user iterates it. So `Model` passes an `on_finished` callback to the `Stream` constructor:

```python
def on_finished(req: LMRequest, resp: LMResponse) -> None:
    self.history.append(HistoryEntry(request=req, response=resp))
    if update_conversation:
        self._conversation = list(req.messages) + [resp.message]
```

When the stream materializes the response (either at the `end` event or when `stream.response` is accessed), it calls this callback. The `Model` gets notified, updates history, and everything is consistent — even though the `Model` never directly iterated the stream.

## The .text Property

The most-used feature of `Stream` is the simplest:

```python
@property
def text(self):
    for chunk in self:
        if chunk.type == "text" and chunk.text is not None:
            yield chunk.text
```

It iterates the stream, yields only text chunks, and filters everything else. Thinking tokens, tool calls, the finished event — all invisible. Users who call `for text in stream.text` get a clean string generator with no ceremony.

## Error Handling in Streams

When an adapter yields `StreamEvent(type="error", error={"code": "rate_limit", "message": "..."})`, the `Stream` class converts it to an exception:

```python
exc_cls = error_class_for_canonical_code(code)  # → RateLimitError
raise exc_cls(message)
```

The stream is marked as done. Any subsequent iteration raises `StopIteration`. Partial responses accumulated before the error are available through `_materialize_response()`, but `stream.response` will trigger the materialization — giving you whatever was received before the failure.

Connection drops mid-stream are different. The transport raises `TransportError`, which propagates through the adapter's generator, through the `Stream`, and to the user. There's no partial recovery — the stream is dead.

This is one area where streaming is genuinely worse than `complete()`. A blocking call either succeeds or fails. A stream can succeed for 90% and then fail — leaving you with partial text and no clean way to resume. The `stream.response` property handles this gracefully (materializes whatever was received), but the user experience is a truncated answer followed by an exception.

The next chapter covers middleware — the system that wraps adapter calls with retries, caching, and logging, and that sits between the client and the adapter in the dispatch chain.
