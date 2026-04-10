# Chapter 4: Memory

Try this with our research assistant so far:

```python
import lm15

resp = lm15.complete("gpt-4.1-mini", "What's the population of Tokyo?", env=".env")
print(resp.text)

resp = lm15.complete("gpt-4.1-mini", "And what about Osaka?", env=".env")
print(resp.text)
```
```
Tokyo has a population of approximately 13.96 million people in the city proper.

I'd be happy to help, but could you clarify what you'd like to know about Osaka?
```

The model has no idea what "and what about" refers to. Each `complete()` call is a stranger walking up to the model and asking a question — no context, no history, no relationship to anything that came before. For classification and extraction, that's fine. For anything resembling a conversation, it's useless.

`lm15.model()` solves this. It creates an object that remembers every exchange:

```python
import lm15

gpt = lm15.model("gpt-4.1-mini", env=".env")

gpt("What's the population of Tokyo?")
resp = gpt("And what about Osaka?")
print(resp.text)
```
```
Osaka has a population of approximately 2.75 million people in the city proper, and about 19.3 million in the greater Keihanshin metropolitan area.
```

"And what about" works now because the model sees both turns. The first question and answer are sent alongside the second question — the model has full context.

This might seem like a small convenience. It's not. Memory transforms what you can build. A stateless call is a calculator — put something in, get something out, done. A model with memory is a collaborator. It accumulates understanding. It refines its thinking based on your corrections. It maintains context across a complex multi-step task. Everything we build from this chapter forward depends on this capability.

## The Model Object

`lm15.model()` takes the same parameters as `complete()` — model name, system prompt, temperature, tools — but you set them once:

```python
import lm15

assistant = lm15.model("gpt-4.1-mini",
    system="""You are a research assistant. When asked a question:
- Answer with specific facts, not generalities
- Cite dates, numbers, and names when you know them
- If you're uncertain, say so explicitly
- Keep answers under 3 paragraphs""",
    temperature=0,
    max_tokens=500,
    env=".env",
)
```

Now `assistant` is an object you call like a function:

```python
resp = assistant("When was the Treaty of Versailles signed?")
print(resp.text)
```
```
The Treaty of Versailles was signed on June 28, 1919, in the Hall of Mirrors at the Palace of Versailles, France. It formally ended World War I between the Allied Powers and Germany.
```

Every parameter — system prompt, temperature, max_tokens — is bound to the object. You don't repeat them. Call `assistant(...)` ten times and each call uses the same configuration. The system prompt is applied once, not re-specified on every call.

But the real value isn't config reuse. It's what comes next.

## Conversations

```python
assistant("My name is Max and I'm researching the interwar period.")
assistant("I'm particularly interested in the economic consequences.")
resp = assistant("What should I read first?")
print(resp.text)
```
```
Given your interest in the economic consequences of the interwar period, I'd recommend starting with John Maynard Keynes' "The Economic Consequences of the Peace" (1919). Keynes attended the Versailles negotiations as a representative of the British Treasury and wrote this book as a critique of the treaty's reparations provisions. It's both a primary source and a foundational analysis.

For a broader economic history, Charles Kindleberger's "The World in Depression, 1929–1939" (1973) is the standard reference, covering how the economic instability following Versailles contributed to the Great Depression.

Max, since you're focused on economic consequences specifically, you might also look at Adam Tooze's "The Deluge: The Great War and the Remaking of Global Order" (2014) for a more recent perspective.
```

Three things happened that couldn't happen with `complete()`:

1. The model remembered the user's name — "Max" — from two turns ago
2. It remembered the research topic — "interwar period" — and the specific focus — "economic consequences"
3. It combined all three turns to give a personalized, targeted answer

This is the model object's purpose. Every call appends to a conversation, and every subsequent call sees the full history. The model accumulates context the way a human research librarian would — each question refines its understanding of what you're really looking for.

## What's Actually Happening

Under the hood, the model object maintains a list of messages. Each call adds your prompt as a user message and the model's response as an assistant message. On the next call, lm15 sends the entire conversation — all previous turns plus your new message — to the provider.

This means the context grows with every turn. After ten turns, the model receives all twenty messages (ten from you, ten from it) plus your new question. After fifty turns, it receives a hundred messages. This has a cost implication: each call processes more input tokens than the last, because the full conversation is re-sent every time. (Chapter 8 — Prompt Caching — makes this dramatically cheaper. But for now, be aware that long conversations get expensive.)

You can inspect the history directly:

```python
print(f"Turns: {len(assistant.history)}")

for entry in assistant.history:
    q = entry.request.messages[-1].parts[0].text
    a = (entry.response.text or "")[:80]
    print(f"  Q: {q[:60]}...")
    print(f"  A: {a}...")
    print()
```

And clear it when you want a fresh start:

```python
assistant.history.clear()

resp = assistant("What's my name?")
print(resp.text)
```
```
I don't know your name — you haven't told me yet. What should I call you?
```

The history is gone. The system prompt, tools, and configuration are unchanged. Only the memory was wiped. This is useful between sessions, when switching topics, or when the context has grown so long it's degrading response quality.

## Per-Call Overrides

The model object's configuration is the default, not a constraint. Override any parameter for a single call:

```python
# Default: temperature=0, max_tokens=500
resp = assistant("List five key events of 1919.")

# Just this once: more creative
resp = assistant("Write a dramatic opening line for an essay about Versailles.",
    temperature=1.2)

# Back to defaults
resp = assistant("What happened at the Paris Peace Conference?")
```

The override applies to one call and disappears. The model object is unchanged. This is useful when most of your calls need reliability (low temperature) but occasionally you want the model to be creative — a draft, a brainstorm, a different angle.

## Tools on Model Objects

In Chapter 3, we passed tools to individual `complete()` calls. On a model object, you bind them once and they're available on every turn:

```python
import lm15

def search(query: str) -> str:
    """Search the web for current information."""
    return "Tokyo population: 13.96 million (city proper), 37.4 million (metro area)"

def read_url(url: str) -> str:
    """Fetch and read a web page."""
    return "Tokyo is the capital and most populous city of Japan..."

assistant = lm15.model("gpt-4.1-mini",
    system="You are a research assistant. Always search before answering factual questions.",
    tools=[search, read_url],
    temperature=0,
    env=".env",
)

assistant("What's the population of Tokyo?")
resp = assistant("How does that compare to New York?")
print(resp.text)
```

The second call works because the model remembers the first answer (Tokyo's population) from conversation history, and searches for New York's population using the bound tool. Tools + memory is the combination that makes agents possible — the model can reason across turns and act when it needs new information.

This is also where manual tools from Chapter 3 find their natural home. `submit_tools()` requires a model object because it continues a conversation:

```python
from lm15 import Tool

write_file = Tool(
    name="write_file",
    description="Write content to a file on disk",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
)

agent = lm15.model("claude-sonnet-4-5", tools=[write_file], env=".env")
resp = agent("Create a Python script that prints 'hello world'.")

# Model requested to write a file — we approve and execute manually
for tc in resp.tool_calls:
    print(f"Write to {tc.input['path']}?")
    # ... approval logic ...

results = {tc.id: "File written." for tc in resp.tool_calls}
resp = agent.submit_tools(results)
print(resp.text)
```

## Derived Models

Sometimes you want a variation of a model object — same config, different model name, or same everything but a different system prompt. The `with_*` methods create a new object with one thing changed:

```python
# Our base research assistant
assistant = lm15.model("gpt-4.1-mini",
    system="You are a research assistant.",
    temperature=0,
    env=".env",
)

# Same config, different model
claude_assistant = assistant.with_model("claude-sonnet-4-5")

# Same config, different persona
editor = assistant.with_system("You are a writing editor. Fix grammar and improve clarity. Return only the corrected text.")

# Same config, with tools added
web_assistant = assistant.with_tools([search, read_url])
```

The original is never modified. Each `with_*` returns a fresh object with its own empty history. This is useful when you need several related agents — a researcher, a writer, a fact-checker — that share base configuration but differ in personality or capabilities.

## `complete()` vs `model()`: A Decision, Not a Preference

You now have two ways to call a model. They're not interchangeable — each is right for a different job.

`complete()` is a function. It takes inputs and produces an output. No state, no memory, no side effects. It's perfect for batch processing — classify a thousand reviews, summarize fifty documents, translate a hundred sentences. Each call is independent and can run in parallel.

`model()` is an object. It accumulates history, binds configuration, and maintains relationships between calls. It's perfect for conversation — a chatbot, an interactive research session, an agent that reads code and writes fixes across multiple files.

I've seen people use `model()` for batch work because they like the config binding. Don't. The history accumulation will slow your calls and inflate your costs as the object silently drags every previous conversation turn into every new call. Use a config dict with `complete()` for batch work. Use `model()` when you need the model to remember.

## Our Research Assistant So Far

Let's step back and see what we've built across four chapters:

```python
import lm15

def search(query: str) -> str:
    """Search the web for current information."""
    # Your search implementation here
    return "..."

def read_url(url: str) -> str:
    """Fetch and read a web page (first 5000 characters)."""
    # Your URL reader here
    return "..."

assistant = lm15.model("gpt-4.1-mini",
    system="""You are a research assistant. When asked a question:
- Search the web before answering factual questions
- Cite your sources with URLs
- If you're uncertain, say so explicitly
- Keep answers under 3 paragraphs unless asked for more""",
    tools=[search, read_url],
    temperature=0,
    max_tokens=500,
    env=".env",
)
```

This object can answer questions, search the web, read pages, cite sources, and carry on a conversation that builds understanding over time. In 15 lines. We're going to keep building on it — adding streaming, document reading, reasoning, caching — but the core is here.

What's missing right now is immediacy. When the assistant is thinking — searching, reading, composing a long answer — you stare at a blank screen until the entire response arrives. That's fine for scripts. It's terrible for anything a human is watching. Chapter 5 fixes that.
