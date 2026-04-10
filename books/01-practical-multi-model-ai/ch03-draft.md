# Chapter 3: Giving the Model Hands

Here's the problem with everything we've built so far: our research assistant is a know-it-all who can't check anything. Ask it about the weather and it'll generate a plausible-sounding forecast from its training data — data that was frozen months or years ago. Ask it to read a file on your computer and it'll politely explain that it can't do that. It's a brain in a jar. All knowledge, no senses, no hands.

Tools fix this. A tool is a Python function you give to the model. The model can *request* that lm15 call it — "I need to check the current weather in Montreal" — and lm15 executes your function, sends the result back, and lets the model continue with real, live data.

Let me show you before I explain:

```python
import lm15

def search_web(query: str) -> str:
    """Search the web and return the top result."""
    # In a real application, this would call a search API
    return "The population of Tokyo is approximately 13.96 million (2023 estimate)."

resp = lm15.complete("gpt-4.1-mini",
    "What's the current population of Tokyo?",
    tools=[search_web],
    env=".env",
)
print(resp.text)
```
```
According to a 2023 estimate, the population of Tokyo is approximately 13.96 million people.
```

The model didn't make up a number. It called your function, got the result, and incorporated it into its answer. You can tell because the phrasing mirrors the tool's return value — "approximately 13.96 million" — rather than the round numbers the model would have hallucinated from training data.

Let's unpack what happened, because the mechanism is elegant and you'll use it constantly.

## The Tool-Call Dance

When you pass `tools=[search_web]` to `complete()`, lm15 does something clever before the call even reaches the provider. It inspects your function — reads the name, the docstring, the type hints, the parameter defaults — and builds a JSON Schema that describes the tool. For `search_web`, that schema looks like:

```json
{
  "name": "search_web",
  "description": "Search the web and return the top result.",
  "parameters": {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"]
  }
}
```

This schema is sent to the model alongside your prompt. The model now knows: "There's a function I can call. It's called `search_web`. It takes a `query` string. It searches the web."

The model reads your prompt — "What's the current population of Tokyo?" — and decides it should call the tool. Instead of generating text, it generates a structured tool call: `search_web(query="current population of Tokyo")`. The response comes back with `finish_reason="tool_call"` instead of `"stop"`.

lm15 intercepts this. It calls your Python function with the arguments the model specified. Your function runs and returns a string. lm15 sends that string back to the model as a "tool result." The model reads the result, incorporates it into its thinking, and *now* generates its final text response.

The whole sequence — prompt → tool call → execute → result → final answer — happened inside that single `complete()` call. You didn't write a loop or check `finish_reason`. lm15 handled the round-trip automatically because you passed a callable function.

This automatic execution is the key distinction. If you'd passed a `Tool` object (a schema without a function), lm15 would have returned the tool call to you and waited. You would have executed it yourself and called back. We'll get to that manual mode later in this chapter — it exists for when you need control over what happens between the request and the execution. But for most tools, auto-execute is what you want.

## Writing Tools the Model Can Use

The model's only documentation for your tool is the name, the docstring, and the parameter types. It can't see your implementation. It can't read comments inside the function body. It makes its decision to call the tool — and its choice of arguments — based entirely on those three things.

This means tool design is prompt engineering. The name and docstring aren't for humans; they're for the model. Write them accordingly.

Here's a bad tool:

```python
def do_stuff(x: str) -> str:
    """Process the input."""
    return get_data(x)
```

The model sees a function called `do_stuff` that "processes the input." It has no idea when to call it, what to pass, or what it'll get back. It'll either never call it or call it at random.

Here's the same tool, written for the model:

```python
def search_knowledge_base(query: str) -> str:
    """Search the company knowledge base for articles matching the query.
    Returns the title and first paragraph of the top matching article."""
    return get_data(query)
```

Same implementation. But now the model knows exactly what this tool does, when to use it, what to pass, and what the return value looks like. The description is specific enough that the model can reason about whether this tool will help answer the user's question.

A few rules I've learned the hard way:

**Return strings, not data structures.** The model reads your return value as text. If you return a dict, lm15 calls `str()` on it, and `"{'temp': 22, 'unit': 'celsius'}"` is harder for the model to parse than `"22°C, partly cloudy"`. Format the result for the model's consumption, not for a downstream program.

**Handle your own errors.** If your function raises an exception, the tool call fails and the model gets nothing — no error message, no diagnostic, just silence. Instead, catch errors and return informative strings:

```python
def read_file(path: str) -> str:
    """Read a file from disk and return its contents."""
    try:
        return open(path).read()
    except FileNotFoundError:
        return f"Error: file '{path}' not found"
    except PermissionError:
        return f"Error: no permission to read '{path}'"
```

The model can read "Error: file not found" and adapt — try a different path, ask the user, or report the problem. A Python traceback crashing through the tool-call machinery is useful to nobody.

**Keep parameters simple.** `str`, `int`, `float`, `bool`. The model handles these well because they map cleanly to JSON Schema types. Avoid complex nested objects. If your tool needs structured input, accept a JSON string and parse it yourself.

## Our Research Assistant Gets Tools

Let's give our research assistant something real. Two tools: one to search the web, one to read URLs. (In a real application, you'd use actual search and HTTP libraries. I'm using stubs so the examples run without API keys for search services, but the tool design is production-grade.)

```python
import lm15

def search(query: str) -> str:
    """Search the web for current information. Returns a summary of top results with URLs."""
    # Stub — replace with real search API
    return """1. "Tokyo Population 2024" (worldpopulationreview.com) — Tokyo's population is 13.96 million in the city proper, 37.4 million in the greater metropolitan area.
2. "Tokyo Demographics" (wikipedia.org) — Tokyo is the most populous metropolitan area in the world."""

def read_url(url: str) -> str:
    """Fetch a web page and return its text content (first 5000 characters)."""
    # Stub — replace with real HTTP fetch
    return "Tokyo (東京) is the capital of Japan and the most populous metropolitan area in the world..."

RESEARCHER = {
    "model": "gpt-4.1-mini",
    "system": """You are a research assistant with access to web search and URL reading.
- Always search before answering factual questions
- Cite your sources with URLs
- If search results are insufficient, read the most promising URL for more detail
- Be specific: cite numbers, dates, and names""",
    "tools": [search, read_url],
    "temperature": 0,
    "env": ".env",
}

resp = lm15.complete(prompt="What's the population of the Tokyo metropolitan area?", **RESEARCHER)
print(resp.text)
```
```
The Tokyo metropolitan area has a population of approximately 37.4 million people, making it the most populous metropolitan area in the world. The city proper has about 13.96 million residents.

Sources:
- [World Population Review](https://worldpopulationreview.com) — Tokyo population figures
- [Wikipedia — Tokyo Demographics](https://wikipedia.org)
```

The model called `search`, read the results, and cited its sources — because the system prompt told it to. It didn't call `read_url` because the search results were sufficient. If they hadn't been, it would have called `read_url` on the most promising link, read the page, and used that information instead.

This is the pattern you'll use in every tool-enabled application: system prompt defines *behavior*, tools define *capabilities*, and the model decides *when* to use which tool.

## Multiple Tool Calls

Models are smart about combining tools. Give them several and they'll figure out which ones to use:

```python
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, **, sqrt, etc."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

resp = lm15.complete("gpt-4.1-mini",
    "Search for the height of the Eiffel Tower in meters, then calculate how many stacked on top of each other it would take to reach the altitude of the ISS.",
    tools=[search, calculator],
    temperature=0,
    env=".env",
)
print(resp.text)
```

The model calls `search` to find the height (330 meters) and the ISS altitude (~408 km), then calls `calculator` to compute `408000 / 330`. Two tools, coordinated, in one call. You didn't write orchestration logic. The model's ability to plan and sequence tool calls is one of the most useful properties of modern language models — and one that only works if your tool names and descriptions are clear.

## Built-In Tools

Some providers offer server-side tools that run on their infrastructure. You don't supply a function — you just pass a string:

```python
resp = lm15.complete("gpt-4.1-mini", "What happened in AI news today?",
    tools=["web_search"], env=".env")
print(resp.text)

for c in resp.citations:
    print(f"  [{c.title}]({c.url})")
```

`"web_search"` is executed by OpenAI — the model searches the live web, reads results, and cites sources in `resp.citations`. No API key for a search service, no function to write, no results to parse. The tradeoff is control: you can't customize the search, filter results, or retry on failure. For quick experiments, it's great. For production, you'll usually want your own search tool.

## Manual Mode: When You Need Control

Auto-execute is the right default. But sometimes you can't let lm15 call the function automatically:

- The tool writes to a database and you need human approval
- The tool is async and you're in a sync context
- You want to log, filter, or transform tool calls before executing them
- The tool talks to a paid API and you want to check the budget first

For these cases, pass a `Tool` object instead of a callable. lm15 will tell the model the tool exists but won't execute anything:

```python
import lm15
from lm15 import Tool

write_to_db = Tool(
    name="write_record",
    description="Write a record to the production database",
    parameters={
        "type": "object",
        "properties": {
            "table": {"type": "string"},
            "data": {"type": "string"},
        },
        "required": ["table", "data"],
    },
)

agent = lm15.model("gpt-4.1-mini", env=".env")
resp = agent("Add a new user: name=Alice, role=admin", tools=[write_to_db])

# The model requested the tool call, but nothing was executed
for tc in resp.tool_calls:
    print(f"Tool: {tc.name}")
    print(f"Args: {tc.input}")
    print(f"Approve? ", end="")
    if input().strip().lower() == "y":
        # Execute manually
        print(f"Writing to {tc.input['table']}...")

# Send results back
results = {tc.id: "Record written successfully" for tc in resp.tool_calls}
resp = agent.submit_tools(results)
print(resp.text)
```

Notice that manual tools require `lm15.model()` — the model object from Chapter 4, which we haven't covered yet. That's because `submit_tools()` needs the conversation state from the original call. `complete()` is stateless — there's no conversation to continue. This is a preview of Chapter 4's territory, and I'll cover it fully there.

## The Multi-Hop Pattern

Sometimes one round-trip isn't enough. A complex question might require the model to search, read a result, search again with refined terms, and then synthesize. With auto-execute tools, lm15 handles this automatically — up to 8 round-trips per call. With manual tools, you write the loop:

```python
resp = agent("Find and compare the GDP of France and Germany in 2023.", tools=[search])

while resp.finish_reason == "tool_call":
    results = {}
    for tc in resp.tool_calls:
        results[tc.id] = execute_search(tc.input["query"])
    resp = agent.submit_tools(results)

print(resp.text)
```

This loop is the fundamental pattern of every AI agent. The model decides what to do, your code executes it, and the loop continues until the model has enough information to answer. Every agent framework in existence is a dressed-up version of this loop. We'll build a full agent with it in Chapter 10 of the agents book — but the pattern is here, in twelve lines, with nothing hidden.

## What You Can Build Now

Our research assistant went from a brain in a jar to a brain with hands. It can search, fetch, calculate, and cite its sources. It's still stateless — each call starts fresh, with no memory of previous questions. It can't look at your documents or images. It answers all at once, with no streaming. But the core capability loop — ask, search, synthesize, cite — works.

The next thing we need is memory. Right now, if you ask our assistant a follow-up question — "What about the population of the metro area?" — it has no idea what "the" refers to. Each call is independent. Chapter 4 gives the model a conversation, and that changes what's possible more than any other single feature.
