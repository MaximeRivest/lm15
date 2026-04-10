# Chapter 5: Thinking Out Loud

There's a moment in every demo where the AI stops and thinks. Maybe it's searching the web. Maybe it's composing a long answer. Maybe it's reasoning through a hard problem. Whatever it's doing, the human on the other end sees nothing. A blank screen. A frozen cursor. Five seconds of silence that feel like thirty.

This is the uncanny valley of AI tools. The model is working — you're paying for tokens being generated right now — but the interface gives you no evidence of life. Is it thinking? Is it stuck? Did the network drop? You don't know, and that uncertainty is worse than waiting.

Streaming solves this. Instead of waiting for the complete response and delivering it in one block, lm15 can show you the response as it's generated — word by word, in real time. The total wait is the same. The experience is completely different.

```python
import lm15

for text in lm15.stream("gpt-4.1-mini", "Write a haiku about debugging.", env=".env").text:
    print(text, end="", flush=True)
print()
```
```
Null pointer, line five—
the bug hides in plain sight while
I blame the network.
```

The first word appeared in about 200 milliseconds. The rest followed at reading speed. Instead of waiting 1.5 seconds for the complete haiku, the user saw it unfold like someone typing it live.

This matters more than it sounds. In a CLI tool, streaming is the difference between "is this thing working?" and watching the model think in front of you. In a web app, it's the difference between a loading spinner and a live, responsive conversation. In an agent that calls tools and reasons through problems, streaming lets you see what the agent is doing — which tools it's calling, what it's thinking, when it gets stuck — instead of waiting in the dark for a final answer.

## Two Levels of Access

lm15 gives you two ways to consume a stream, depending on how much visibility you need.

### The simple path: just the text

`.text` is a generator that yields only the text chunks. Everything else — thinking tokens, tool calls, metadata — is filtered out:

```python
stream = lm15.stream("gemini-2.5-flash",
    "Explain why the sky is blue in three sentences.", env=".env")

for text in stream.text:
    print(text, end="", flush=True)
print()
```

This is the right choice most of the time. You're building a CLI that prints the response, or a web endpoint that sends chunks to the browser. You don't care about the internals — you want text, arriving incrementally.

### The full picture: events

When you iterate the stream directly, you get `StreamChunk` objects with a `type` field. This is how you build UIs that distinguish between thinking and answering, or show tool calls as they happen:

```python
stream = lm15.stream("claude-sonnet-4-5",
    "What's larger: e^π or π^e? Explain your reasoning.",
    reasoning=True, env=".env")

for event in stream:
    match event.type:
        case "thinking": print(f"💭 {event.text}", end="", flush=True)
        case "text":     print(f"\n\n✅ {event.text}", end="", flush=True)
        case "finished":
            u = event.response.usage
            print(f"\n\n📊 {u.total_tokens} tokens ({u.reasoning_tokens} reasoning)")
```
```
💭 I need to compare e^π and π^e. Taking the natural log of both: ln(e^π) = π ≈ 3.14159, and ln(π^e) = e·ln(π) ≈ 2.71828 × 1.14473 ≈ 3.11. Since π > e·ln(π), we have e^π > π^e.

✅ e^π is larger than π^e. Numerically, e^π ≈ 23.14 while π^e ≈ 22.46. You can prove this by comparing their natural logarithms: ln(e^π) = π ≈ 3.14, while ln(π^e) = e·ln(π) ≈ 3.11.

📊 198 tokens (142 reasoning)
```

The thinking stream completed before the text stream began. In a real UI, you'd render thinking in a collapsible section or a dimmed panel, and display the answer prominently. The user can see that the model actually reasoned through the comparison instead of guessing — which builds trust, and helps you debug when it gets something wrong.

The event types you'll encounter:

| `event.type` | What's happening | Key fields |
|---|---|---|
| `"text"` | A chunk of the model's text response | `event.text` |
| `"thinking"` | A chunk of chain-of-thought reasoning | `event.text` |
| `"tool_call"` | The model is calling a tool | `event.name`, `event.input` |
| `"audio"` | A chunk of generated audio | `event.audio` |
| `"finished"` | The stream is complete | `event.response` |

`"finished"` always comes last. Its `event.response` is a full `LMResponse` — the same object you'd get from `complete()`, with `.text`, `.usage`, `.finish_reason`, and everything else.

## Getting the Full Response After Streaming

Sometimes you want both: stream the text live for the user, then inspect the full response afterward for logging, analytics, or downstream processing. The `stream.response` property gives you the materialized response after the stream is consumed:

```python
stream = lm15.stream("gpt-4.1-mini", "Explain DNS in two paragraphs.", env=".env")

# Stream to the user
for text in stream.text:
    print(text, end="", flush=True)
print()

# Inspect the full response
resp = stream.response
print(f"\nModel: {resp.model}")
print(f"Tokens: {resp.usage.total_tokens}")
print(f"Finish: {resp.finish_reason}")
```

`stream.response` is always available after the stream is consumed. If you access it *before* consuming the stream, it consumes everything silently to build the response — you get the data, but you lose the streaming benefit. Stream first, then inspect.

## Streaming on Model Objects

On a `model()` object, streaming integrates with conversation history:

```python
import lm15

assistant = lm15.model("gpt-4.1-mini", env=".env",
    system="You are a research assistant. Be specific.")

# First turn: streamed
for text in assistant.stream("What was the Marshall Plan?").text:
    print(text, end="", flush=True)
print()

# Second turn: non-streamed — but it sees the first turn
resp = assistant("Who proposed it and when?")
print(resp.text)
```
```
The Marshall Plan (officially the European Recovery Program) was a U.S. initiative to provide economic aid to Western European countries after World War II...

The Marshall Plan was proposed by U.S. Secretary of State George C. Marshall in a speech at Harvard University on June 5, 1947.
```

The streamed first response was saved to history. The second call (non-streamed) has full context. You can mix streaming and non-streaming calls freely — the model object tracks everything regardless of how the response was delivered.

## Streaming with Tools

When auto-execute tools are active during streaming, tool interactions appear as events:

```python
import lm15

def search(query: str) -> str:
    """Search the web for current information."""
    return "The current population of Lagos is approximately 16.6 million (2024 estimate)."

stream = lm15.stream("gpt-4.1-mini",
    "What's the current population of Lagos?",
    tools=[search], env=".env")

for event in stream:
    match event.type:
        case "tool_call":  print(f"\n🔧 Searching: {event.input}")
        case "text":       print(event.text, end="", flush=True)
        case "finished":   print(f"\n📊 {event.response.usage.total_tokens} tokens")
```
```
🔧 Searching: {'query': 'current population of Lagos'}
The current population of Lagos, Nigeria is approximately 16.6 million people according to 2024 estimates.
📊 57 tokens
```

You see the tool call the instant the model makes it. In an agent UI, this is how you show the user what's happening — "Searching for..." — while the tool executes in the background.

## When Not to Stream

Streaming isn't always the right choice. It adds complexity — you're handling a generator instead of a return value — and there are cases where that complexity buys you nothing:

**Batch processing.** If you're classifying a thousand emails, nobody's watching. Use `complete()`. The response arrives faster because there's no per-token overhead.

**Structured output.** If you need to parse the response as JSON, you can't parse half a JSON object. Wait for the full response.

**Short responses.** A one-word classification finishes in 200ms either way. Streaming a single token is overhead, not benefit.

**Pipeline stages.** If the output of one model feeds into another, you need the complete response before starting the next stage.

Use streaming when a human is watching. Use `complete()` when a machine is processing. The token cost is identical — streaming changes *when* you see the tokens, not how many there are.

## A Streaming Research Session

Here's what our research assistant looks like when it's streaming — a real interactive session where the user sees the model searching, thinking, and answering in real time:

```python
import lm15

def search(query: str) -> str:
    """Search the web for current information."""
    return "..."  # your implementation

assistant = lm15.model("gpt-4.1-mini",
    system="You are a research assistant. Search before answering.",
    tools=[search],
    temperature=0,
    env=".env",
)

# Simulate an interactive session
questions = [
    "When was the Suez Canal opened?",
    "How long is it?",
    "Has it been expanded recently?",
]

for q in questions:
    print(f"\n> {q}")
    for event in assistant.stream(q):
        match event.type:
            case "tool_call": print(f"  🔍 searching...", flush=True)
            case "text":      print(f"  {event.text}", end="", flush=True)
    print()
```

Three questions. The model searches before each one, streams the answer live, and each follow-up has full context from previous turns. The user sees every step — search, answer, flowing text — instead of staring at a blank screen.

This is the experience we're building toward. The assistant isn't finished — it can't read your documents yet, doesn't reason through hard problems, and gets expensive on long conversations. But the interactive core is here. Next, we give it eyes.
