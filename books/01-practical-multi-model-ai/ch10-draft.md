# Chapter 10: Staying Alive

Every model name in this book is a lie. Not today — today they work. But `claude-sonnet-4-5` will be replaced by `claude-sonnet-5-0`. `gpt-4.1-mini` will become `gpt-5-mini` or `gpt-5-nano` or something nobody predicted. `gemini-2.5-flash` will be `gemini-3.0-flash` before you've finished your first production deployment. The landscape moves fast enough that any hardcoded model name is a ticking clock.

The fix isn't vigilance — keeping a list of current models and updating it quarterly. The fix is asking the providers what's available right now and choosing from the answer.

## What's Running?

```python
import lm15

info = lm15.providers_info(env=".env")
for name, meta in info.items():
    status = "✅" if meta["configured"] else "❌"
    print(f"  {status} {name}: {meta['model_count']} models")
```
```
  ✅ openai: 42 models
  ✅ anthropic: 12 models
  ✅ gemini: 28 models
```

`providers_info()` tells you which providers have a valid API key and how many models each one currently offers. This is the first thing to check when something stops working — if a provider shows `configured=False`, you know the problem is your key, not your code.

## What's Available?

```python
import lm15

models = lm15.models(env=".env")
print(f"Total models: {len(models)}")

for m in models[:5]:
    print(f"  {m.provider}/{m.id}  context={m.context_window}  tools={m.tool_call}")
```
```
Total models: 82
  anthropic/claude-3-haiku-20240307  context=200000  tools=True
  anthropic/claude-sonnet-4-5  context=200000  tools=True
  openai/gpt-4.1-mini  context=1047576  tools=True
  openai/gpt-4.1  context=1047576  tools=True
  gemini/gemini-2.5-flash  context=1048576  tools=True
```

`lm15.models()` queries each provider's API and returns a list of `ModelSpec` objects. Each one tells you what the model can do — context window size, maximum output length, supported modalities, whether it supports tool calling, structured output, reasoning.

This is live data, not a hardcoded catalog. When a provider adds a model, it shows up here. When they deprecate one, it disappears.

## Finding What You Need

The real power is filtering. Instead of "give me all models," you ask for models that can do what your code needs:

```python
# Models that support tool calling AND reasoning
smart = lm15.models(supports={"tools", "reasoning"}, env=".env")
print(f"Models with tools + reasoning: {len(smart)}")
for m in smart:
    print(f"  {m.provider}/{m.id}")

# Models that can read images
vision = lm15.models(input_modalities={"image"}, env=".env")
print(f"\nVision models: {len(vision)}")

# Models that can generate audio
audio = lm15.models(output_modalities={"audio"}, env=".env")
print(f"Audio generation models: {len(audio)}")
```

This is how you write code that survives model changes. Instead of `model="claude-sonnet-4-5"`, you say "give me a model from Anthropic that supports tools" — and whatever model exists when your code runs, that's what gets used.

## Building an Adaptive Selector

Here's a practical pattern: a function that picks the best available model for a given task, with preferences and fallbacks:

```python
import lm15

PREFERENCES = {
    "research": ["claude-sonnet-4-5", "gpt-4.1-mini", "gemini-2.5-flash"],
    "cheap":    ["gemini-2.5-flash", "gpt-4.1-mini"],
    "vision":   ["gemini-2.5-flash", "gpt-4.1-mini", "claude-sonnet-4-5"],
    "reasoning": ["claude-sonnet-4-5", "gpt-4.1-mini"],
}

def pick_model(task: str = "research", needs: set[str] | None = None) -> str:
    available = {m.id for m in lm15.models(supports=needs, env=".env")}
    preferences = PREFERENCES.get(task, PREFERENCES["research"])

    for model in preferences:
        if model in available:
            return model

    if available:
        return next(iter(available))
    raise RuntimeError(f"No model available for task={task}, needs={needs}")

# Use it
model_id = pick_model("research", needs={"tools"})
resp = lm15.complete(model_id, "What is the current GDP of France?",
    tools=[search], env=".env")
print(f"Answered by {resp.model}: {resp.text[:80]}...")
```

The preference list is configuration — a YAML file, an environment variable, a database row. The discovery is live. When `claude-sonnet-4-5` is deprecated and replaced by `claude-sonnet-5-0`, you update one line in a config file instead of hunting through your codebase for hardcoded model names.

## What Our Research Assistant Becomes

Let's write the final version. Everything from every chapter, composed into one coherent tool:

```python
import lm15
from lm15 import Part

# --- Tools ---

def search(query: str) -> str:
    """Search the web for current information. Returns summaries with URLs."""
    try:
        return actual_search_implementation(query)
    except Exception as e:
        return f"Search failed: {e}"

def read_url(url: str) -> str:
    """Fetch a web page and return its text content (first 5000 characters)."""
    try:
        import urllib.request
        return urllib.request.urlopen(url).read().decode()[:5000]
    except Exception as e:
        return f"Failed to read URL: {e}"

# --- Model selection ---

def pick_research_model() -> str:
    preferences = ["claude-sonnet-4-5", "gpt-4.1-mini", "gemini-2.5-flash"]
    available = {m.id for m in lm15.models(supports={"tools"}, env=".env")}
    for model in preferences:
        if model in available:
            return model
    return next(iter(available))

# --- The assistant ---

assistant = lm15.model(
    pick_research_model(),
    system="""You are a research assistant. When asked a question:
- Search the web before answering factual questions
- Read URLs when search results need more detail
- Read and analyze documents the user provides
- Cite your sources with URLs
- Reason through analytical questions step by step
- If you're uncertain, say so explicitly
- Keep answers under 3 paragraphs unless asked for more""",
    tools=[search, read_url],
    temperature=0,
    prompt_caching=True,
    retries=2,
    env=".env",
)
```

Forty lines. Let's count what's in them:

- **Model discovery** (Chapter 10) — picks the best available model at runtime
- **System prompt** (Chapter 2) — defines behavior, constraints, and persona
- **Tools** (Chapter 3) — web search and URL reading, with error handling
- **Stateful conversation** (Chapter 4) — the model object tracks history across turns
- **Prompt caching** (Chapter 8) — 67-80% cheaper conversations
- **Retries** (Chapter 9) — survives transient API failures
- **Temperature 0** (Chapter 2) — deterministic, factual responses

Use it with streaming and reasoning:

```python
# A research session with visibility
for event in assistant.stream("Compare the economic recovery after 2008 vs 2020.",
                              reasoning=True):
    match event.type:
        case "thinking":   print(f"💭 {event.text}", end="", flush=True)
        case "tool_call":  print(f"\n🔍 searching...", flush=True)
        case "text":       print(event.text, end="", flush=True)
        case "finished":
            u = event.response.usage
            print(f"\n\n📊 {u.total_tokens} tokens"
                  f" (cached: {u.cache_read_tokens or 0},"
                  f" reasoning: {u.reasoning_tokens or 0})")
```

Upload a document and analyze it:

```python
paper = assistant.upload("economic_analysis.pdf")
assistant(["What methodology does this paper use?", paper])
resp = assistant("How do the authors' findings compare to current consensus? Search for recent data.")
print(resp.text)
```

All the chapters, composing. Tools + memory + documents + search + reasoning + streaming + caching + retries + discovery. Not as separate features awkwardly stitched together, but as parameters to the same function you learned in Chapter 1.

## What This Book Gave You

Ten chapters. One function. Here's the path we took:

| Chapter | What you learned | What it added to the assistant |
|---|---|---|
| 1. One Function | `lm15.complete()` — call any model | A brain that answers questions |
| 2. Shaping Output | `system=`, `temperature=`, `max_tokens=` | Consistent, well-formatted answers |
| 3. Giving Hands | `tools=` — the model calls your functions | Web search and URL reading |
| 4. Memory | `lm15.model()` — conversation history | Follow-up questions, accumulated context |
| 5. Thinking Out Loud | `lm15.stream()` — real-time output | Live, responsive interaction |
| 6. Eyes and Ears | `Part.*()` — images, documents, audio | Document analysis, image understanding |
| 7. Thinking First | `reasoning=True` — chain of thought | Rigorous analysis, verified answers |
| 8. Making It Cheap | `prompt_caching=True` — cached prefixes | 67-80% lower conversation costs |
| 9. When Things Break | `retries=`, error handling, fallbacks | Reliability at 2 AM |
| 10. Staying Alive | `lm15.models()` — live discovery | Survives model deprecations |

Each row is one capability. Each capability is one parameter. They all compose.

The research assistant we built is real — not a demo, not a tutorial toy. Plug in actual search and URL implementations, point it at your documents, and it works. It searches the web, reads your PDFs, reasons through hard questions, cites its sources, remembers your conversation, streams its answers live, caches its context to save money, retries on failure, and adapts to whatever models are available today.

It's also forty lines of Python with zero dependencies beyond lm15.

That's the tool. Build something with it.
