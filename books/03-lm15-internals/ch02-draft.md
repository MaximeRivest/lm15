# Chapter 2: Types

Open `types.py`. At 502 lines, it's the longest file in lm15, and almost every
other file imports from it. This is the library's data model — the vocabulary
that every layer speaks. If you understand this file, you understand the
contracts between layers, because every boundary in the stack is defined by
these types: sugar builds them, the model composes them, the client passes them
through, adapters translate them to wire format, and the transport never sees
them at all.

The file defines about twenty dataclasses. But the conceptual model is smaller
than that. Four ideas carry the whole thing: **Parts** (content), **Messages**
(conversation), **Requests** (what goes in), and **Responses** (what comes out).
Everything else is support structure.

## Part: The Universal Content Unit

Here's the central design question of `types.py`: how do you represent a piece
of content that might be text, an image, an audio clip, a tool call, a thinking
block, or a citation — and that might appear alongside any combination of those
other types in a single message?

The obvious approach is a class hierarchy: `TextPart`, `ImagePart`,
`ToolCallPart`, each with their own fields. Python's type system would catch
misuse — you can't access `.text` on an `ImagePart`. Clean, safe, conventional.

lm15 doesn't do this. Instead, there's one type — `Part` — with a `type`
discriminator field and a collection of optional fields:

```python
@dataclass(slots=True, frozen=True)
class Part:
    type: PartType  # "text", "image", "tool_call", "thinking", etc.
    text: str | None = None
    source: DataSource | None = None
    id: str | None = None
    name: str | None = None
    input: dict[str, Any] | None = None
    content: tuple["Part", ...] = field(default_factory=tuple)
    # ... more optional fields
```

A `Part` with `type="text"` uses the `text` field. A `Part` with `type="image"`
uses the `source` field (which holds a `DataSource` pointing to the image data).
A `Part` with `type="tool_call"` uses `id`, `name`, and `input`. The fields that
don't apply are `None`.

Why? Three reasons.

**Simpler serialization.** A single type serializes and deserializes with one
code path. A class hierarchy needs a discriminator-based deserializer that
inspects a field, picks the right class, and constructs it. lm15 avoids this
with `Part.from_dict()` — one method, not ten.

**Simpler adapter code.** Adapters iterate over `message.parts` and switch on
`part.type`. With a class hierarchy, they'd use `isinstance()` checks — which
are slower, more verbose, and break when a new subclass is added without
updating every adapter.

**Portability across providers.** A `Part` with `type="image"` from OpenAI's
response can be passed directly into a Gemini request. If there were separate
`OpenAIImagePart` and `GeminiImagePart` types, you'd need conversion logic. The
single `Part` type is the lingua franca that makes cross-model pipelines work.

The tradeoff is weak type checking. Nothing stops you from creating
`Part(type="text", source=DataSource(...))` — a text part with image data. The
`__post_init__` validation catches some of these mistakes (it requires `text`
for text parts, `source` for image parts), but it can't catch everything. lm15
chose simplicity over safety, and so far the choice has held up.

### DataSource

Media parts carry their data in a `DataSource`:

```python
@dataclass(slots=True, frozen=True)
class DataSource:
    type: DataSourceType  # "base64", "url", "file"
    media_type: str | None = None
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    detail: Literal["low", "high", "auto"] | None = None
```

Three ways to reference content: inline (`base64` with `data`), by URL, or by
file ID (after uploading through the provider's file API). The `detail` field is
an OpenAI-specific hint for image resolution. It leaks provider details into the
universal type — a small normalization failure that's cheaper to live with than
to abstract away.

### Factory Methods

The raw constructor is verbose — `Part(type="text", text="Hello")` — so `Part`
has static factory methods:

```python
Part.text_part("Hello")
Part.image(url="https://example.com/cat.jpg")
Part.tool_call(id="call_1", name="search", input={"query": "tcp"})
Part.document(data=pdf_bytes, media_type="application/pdf", cache=True)
```

These are sugar. They build the same frozen dataclass. But they enforce
constraints (exactly one of `url`, `data`, or `file_id`), handle base64 encoding
of bytes, and set sensible defaults for `media_type`. Book 1's `Part.image()`
calls are these factory methods.

## Message: Role + Parts

A `Message` is a role and a tuple of parts:

```python
@dataclass(slots=True, frozen=True)
class Message:
    role: Role  # "user", "assistant", "tool"
    parts: tuple[Part, ...]
```

Why a tuple, not a list? Immutability. A frozen dataclass's fields must be
hashable (or at least not obviously mutable) for the object to behave correctly
as a dict key — which matters for the local response cache in `Model`. Tuples
are hashable; lists aren't. This choice ripples through the codebase: you'll see
`tuple()` wrapping lists everywhere, and `message.parts + (new_part,)` instead
of `message.parts.append(new_part)`.

The `role` is a `Literal["user", "assistant", "tool"]`. Three values. `"user"`
is what you said. `"assistant"` is what the model said. `"tool"` carries tool
results back to the model. There's no `"system"` role — system prompts live on
`LMRequest`, not on messages, because some providers (Anthropic) want the system
prompt as a top-level field while others (OpenAI) want it as a message. Putting
it on the request lets each adapter place it where the provider expects.

`Message.user("Hello")` and `Message.assistant("Hi!")` are convenience methods
that wrap a string in a `Part.text_part()` and a `Message`. They save you from
writing `Message(role="user", parts=(Part.text_part("Hello"),))` every time.

## Tool: The Function Schema

```python
@dataclass(slots=True, frozen=True)
class Tool:
    name: str
    type: Literal["function", "builtin"] = "function"
    description: str | None = None
    parameters: dict[str, Any] | None = None
```

`parameters` is a raw dict — JSON Schema, not a typed Python structure. Why?
Because JSON Schema is already the standard that every provider uses. If lm15
defined its own schema type, every adapter would need to convert it to JSON
Schema anyway, and every user would need to learn a proprietary format. The raw
dict is the zero-abstraction choice: what you write is what the model sees.

The `builtin` type is for provider-side tools like `"web_search"`. These don't
have parameters — they're strings that the adapter passes through to the
provider's native tool system.

`callable_to_tool()` in `model.py` is the bridge between Book 1's
`tools=[get_weather]` and this type. It inspects a Python function's signature
and docstring, builds the JSON Schema, and returns a `Tool`. The schema
inference is simple — `str` → `"string"`, `int` → `"integer"`, `float` →
`"number"` — and deliberately doesn't try to handle complex types. Simple tools
with simple parameters are what models handle well.

## LMRequest: The Normalized Request

```python
@dataclass(slots=True, frozen=True)
class LMRequest:
    model: str
    messages: tuple[Message, ...]
    system: str | tuple[Part, ...] | None = None
    tools: tuple[Tool, ...] = ()
    config: Config = field(default_factory=Config)
```

This is the contract between the model layer and the client layer. Everything
above builds an `LMRequest`. Everything below consumes one. The request carries
the model name, the conversation messages, the system prompt, tool definitions,
and a `Config` with generation parameters.

`system` is `str | tuple[Part, ...]` — either a plain string or a sequence of
parts (for multimodal system prompts). It's separate from `messages` because
Anthropic's API wants it as a top-level field, not as a message. If it were a
message with role `"system"`, every adapter except OpenAI's would need to
extract it. Putting it on the request lets each adapter handle placement
naturally.

`Config` holds the knobs:

```python
@dataclass(slots=True, frozen=True)
class Config:
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop: tuple[str, ...] = ()
    response_format: dict[str, Any] | None = None
    tool_config: ToolConfig | None = None
    reasoning: dict[str, Any] | None = None
    provider: dict[str, Any] | None = None
```

The `provider` field is the escape hatch. Anything that doesn't fit lm15's
normalized model — provider-specific parameters, experimental features, one-off
API options — goes in this dict. Adapters check it and pass recognized keys
through to the wire format. It's the safety valve that prevents lm15 from
needing a new field every time a provider adds a feature.

## LMResponse: The Normalized Response

```python
@dataclass(slots=True, frozen=True)
class LMResponse:
    id: str
    model: str
    message: Message
    finish_reason: FinishReason
    usage: Usage
    provider: dict[str, Any] | None = None
```

The response is a frozen snapshot. The model's output is in `message` — a
`Message` with `role="assistant"` and parts that might include text, thinking,
tool calls, images, or citations. The convenience properties (`.text`,
`.thinking`, `.tool_calls`, `.image`) are computed from `message.parts` — they
filter by type and return the relevant content.

`Usage` counts tokens:

```python
@dataclass(slots=True, frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
```

Cache and reasoning tokens are `None` when not applicable — a distinction from
zero, which means "measured and the count was zero."

The `provider` dict on `LMResponse` is the read-side escape hatch. It contains
the raw JSON from the provider's response — everything lm15 didn't normalize. If
you need the provider's response ID format, their specific finish reason string,
or any field that lm15 doesn't expose, it's in this dict.

## StreamEvent: The Wire-Level Stream

Streaming has its own types. `StreamEvent` is what adapters yield during a
streaming call:

```python
@dataclass(slots=True, frozen=True)
class StreamEvent:
    type: StreamEventType  # "start", "delta", "part_start", "part_end", "end", "error"
    delta: PartDelta | dict[str, Any] | None = None
    finish_reason: FinishReason | None = None
    usage: Usage | None = None
    error: dict[str, str] | None = None
```

This is a low-level type — adapters produce it, the `Stream` class in
`stream.py` consumes it and translates it into the `StreamChunk` objects users
see. The `delta` field is either a `PartDelta` (typed) or a raw dict (for
backward compatibility and adapter flexibility). Chapter 6 covers the streaming
pipeline in detail.

## Why Frozen Dataclasses

This question deserves a direct answer, because it affects every line of code in
the library.

**Why not Pydantic?** Pydantic is a dependency — 4.5MB, plus `pydantic-core` (a
Rust binary). lm15's zero-dependency principle rules it out. But even without
that constraint: Pydantic's value is validation and serialization, and lm15 does
very little of either. Requests are built by trusted code (the Model class), not
parsed from user input. Responses are parsed from provider JSON by adapters that
do their own extraction. Pydantic would add weight without earning it.

**Why not plain dicts?** Dicts have no structure. `response["text"]` might be a
string, might be `None`, might not exist. Dataclasses give you IDE
autocompletion, type checking, and `__post_init__` validation. The cost is
verbosity; the benefit is that bugs surface at construction time, not at access
time.

**Why not mutable dataclasses?** Imagine two model objects sharing an
`LMRequest`. One modifies a field. The other's request silently changes. With
frozen dataclasses, this can't happen — you build a new request instead of
modifying the old one. The code is more verbose (`LMRequest(model=old.model,
messages=old.messages + new_msgs, ...)`) but the bugs it prevents are the kind
that take hours to find.

**Why not attrs?** Same problem as Pydantic — it's a dependency. attrs is
lightweight, but it's not zero.

The frozen dataclass pattern costs verbosity and buys safety, cacheability, and
thread-safety. For a library where requests flow through multiple layers and
might be cached, retried, or shared between model objects, the trade is worth
it.

## The Full Picture

Twenty-odd types. Four ideas: parts, messages, requests, responses. One file.
Every boundary in the stack is defined by these types — `Model` builds an
`LMRequest`, `UniversalLM` passes it through, adapters translate it, and the
response comes back as an `LMResponse`. The types are the contracts.

The next chapter follows our request into `client.py`, where `UniversalLM`
decides which adapter gets it and how middleware wraps the call. The types don't
change — they just pass through. That's the point of having a universal type
system: the request looks the same to every layer.
