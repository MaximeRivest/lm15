# Book 2 — Building AI Agents with lm15

**Audience:** Python developers who've read Book 1 (or equivalent) and want to build agents that do real work
**Length:** ~50,000 words / ~200 pages
**Prerequisite:** Comfortable with `lm15.model()`, tools, streaming

---

## Thesis

An agent is a while loop. Everything else — frameworks, orchestrators, planners, memory systems — is decoration on a while loop. This book strips the decoration away and shows you what's underneath: a model that decides what to do, code that does it, and a loop that keeps going until the job is done. You'll build agents that write code, research topics, analyze documents, and operate across multiple models — and you'll understand every line, because there's no framework hiding the machinery.

---

## The Through-Line

The reader builds a **coding agent** from scratch in Part I, watches it fail in instructive ways, and fixes each failure. By Part II, the patterns generalize — the same loop, with different tools, becomes a research agent, a document analyst, a multi-model pipeline. Part III handles the production reality: cost, reliability, observability.

The coding agent is the vehicle, not the destination. The destination is understanding agent architecture well enough to build any agent.

---

## Part I — The Loop

### Chapter 1: Twelve Lines

**Arc:** The reader builds a working agent in twelve lines of code, runs it on a trivial task, and watches it succeed. Then runs it on a real task and watches it struggle. The chapter ends with the reader understanding that the loop is simple — the hard part is everything around it.

**Opens with:** The complete agent. Twelve lines. Three tools: `read_file`, `write_file`, `run_command`. One while loop. No framework.

**The first task:** "Add a docstring to every function in `utils.py`." The agent reads the file, edits it, and finishes. It works. The reader has a dopamine hit in the first three pages.

**The second task:** "Add input validation to the User model, then run the tests." The agent reads, edits, runs tests, sees failures, edits again, runs again. Multiple tool rounds. It works — but it took 7 turns and 12,000 tokens. The reader starts noticing cost.

**The third task:** "Refactor the auth module." The agent reads one file, edits it, breaks an import in another file, doesn't notice, and declares success. The reader's first agent failure. The chapter ends here — not with a fix, but with the problem clearly stated.

**Key idea:** The loop is trivially simple. The difficulty is in tools, prompts, and knowing when to stop.

**Sections:**
- The twelve lines (code, walkthrough, first run)
- A harder task (multi-turn, test loop)
- The first failure (broken import, false success)
- Why agents are hard (it's not the loop)

---

### Chapter 2: Tool Design

**Arc:** The reader redesigns the three tools from Chapter 1, learns why tool design is the highest-leverage activity in agent building, and sees the agent improve dramatically with better tools and no other changes.

**Opens with:** The Chapter 1 failure replayed. The agent broke an import because `read_file` showed it one file at a time and it never looked at the files that imported the module it changed. The problem isn't the model — it's the tools. The model couldn't see the dependency.

**The redesign:** Add `list_files`, `search_code` (grep), `read_file` (with line numbers). Now the agent can navigate. Re-run the refactoring task. The agent searches for imports of the module it's changing, reads the dependent files, and updates them. Same model. Same prompt. Better tools. Different outcome.

**Tool design principles — earned through examples, not listed as rules:**
- The name and docstring are the model's only documentation — write them for an AI, not a human
- Return strings formatted for the model, not data structures for a program
- Handle errors in the return value, never raise exceptions
- One action per tool — `search_code` and `read_file` are separate, not `search_and_read`
- Keep parameters simple: `str`, `int`, `bool`
- Return enough context (line numbers, file paths) for the model to act on the result
- Bound output length — don't return a 50,000-character file; truncate and tell the model

**The "tools are the API" insight:** You're not designing functions. You're designing an API for a non-deterministic client that reads documentation literally, can't see your source code, and makes its own decisions about when to call what. Good API design is the single highest-leverage skill in agent building.

**Sections:**
- Replaying the failure (the agent couldn't see dependencies)
- Adding navigation tools (list, search, read with context)
- The redesigned agent (same task, better outcome)
- Tool design principles (with before/after examples for each)
- The output formatting problem (what the model sees vs what you return)
- Error returns (the model can recover if you tell it what went wrong)

---

### Chapter 3: Memory and Its Limits

**Arc:** The reader discovers that conversation history is both the agent's greatest asset and its most dangerous liability. Too little memory and the agent forgets what it's doing. Too much and it exceeds the context window, costs a fortune, or gets confused by irrelevant old turns.

**Opens with:** A 20-turn coding session. The agent is working well — reading files, making changes, running tests. On turn 15, the context is 40,000 tokens. On turn 18, the model starts repeating edits it already made — it's losing track of what's in its own history. On turn 20, it hits the context window limit and the call fails.

**The core tension:** The model sees every prior turn. Early turns are useful context ("I already fixed the import in `auth.py`"). Late turns are noise ("Turn 3: I read `utils.py` and it had 200 lines of boilerplate"). The model can't distinguish signal from noise in its own history.

**Strategies — each one demonstrated with the coding agent:**
- **Do nothing** (small tasks where history stays manageable)
- **Clear and restart** (`history.clear()` between logical phases)
- **Sliding window** (keep only the last N turns)
- **Summarize and compress** (use a cheap model to summarize history, inject the summary)
- **External state** (files on disk, a database, a scratchpad the agent writes to and reads from)

**The scratchpad pattern:** Give the agent a `write_scratchpad` and `read_scratchpad` tool. The agent keeps its own notes — decisions made, files changed, remaining tasks — in a file it controls. The conversation history becomes disposable because the important state is externalized. This is the most practical pattern for long-running agents.

**Key idea:** Memory management is the unsexy core of agent engineering. The loop is trivial. Tools are designable. Memory is the thing that breaks at scale, and there's no single right answer — only tradeoffs.

**Sections:**
- The 20-turn wall (context overflow, repeated edits)
- What the model sees (the full conversation, every turn)
- Strategy 1: Clear and restart
- Strategy 2: Sliding window
- Strategy 3: Summarize
- Strategy 4: The scratchpad (externalized state)
- Choosing a strategy (task duration vs context needs)

---

## Part II — Real Agents

### Chapter 4: The Coding Agent

**Arc:** The reader builds a production-quality coding agent — the polished version of chapters 1-3. System prompt design for code tasks. The test-driven loop. Handling linters, formatters, type checkers. Multi-file changes. The agent that works overnight.

**Opens with:** The final version of the coding agent — maybe 40 lines — running a real task end-to-end with streaming visibility. The reader sees what they're building before they build it.

**The test-driven loop:** The most important agent pattern in the book. The agent writes code, runs tests, reads failures, fixes the code, and repeats. The loop bounds itself naturally — when tests pass, the agent stops. This is the agent equivalent of TDD, and it makes coding agents dramatically more reliable because the agent has an objective exit condition.

**System prompt design:** Not a generic "you are a coding assistant." A specific, opinionated prompt that tells the model: read before you edit, run tests after every change, never edit a file you haven't read, search for usages before renaming, announce your plan before executing it. The prompt is 300-500 tokens of distilled coding practice. Walk through each instruction and explain why it's there — what failure it prevents.

**Streaming the agent:** Building a CLI that shows the agent's work in real time — tool calls, thinking, text — so the user knows what's happening and can interrupt if needed. The streaming agent from Book 1 Chapter 5, applied to a multi-turn tool loop.

**Multi-file operations:** The agent that can navigate a codebase. `list_files` with glob patterns. `search_code` with regex. Reading multiple files before making changes. The discipline of reading imports before editing exports.

**Bounding the loop:** Max iterations. Token budget. Wall-clock timeout. Detecting stuck states (agent makes the same edit twice, agent keeps failing the same test). Graceful degradation — "I've used 80% of the budget, here's what I've done so far and what remains."

**Sections:**
- The complete coding agent (show the finished product)
- System prompt anatomy (walk through each instruction)
- The test-driven loop (write → test → read errors → fix → repeat)
- Streaming visibility (CLI that shows the agent working)
- Multi-file navigation (search, list, read before edit)
- Bounding the loop (iterations, budget, stuck detection)
- The full code (complete, copy-pasteable agent)

---

### Chapter 5: The Research Agent

**Arc:** The coding agent's tools were filesystem operations. Swap in web search and URL reading, change the system prompt, and the same loop becomes a research agent. This chapter proves the loop is general — the tools are the variable, not the architecture.

**Opens with:** A side-by-side: the coding agent's tool list and the research agent's tool list. Different tools, same `while finish_reason == "tool_call"` loop. The reader sees the structural equivalence immediately.

**Research-specific tools:**
- `search(query)` — wrapping a search API (Brave, Serper, or the built-in `"web_search"`)
- `read_url(url)` — fetching and truncating web pages
- `save_finding(topic, fact, source_url)` — the scratchpad pattern from Chapter 3, specialized for research

**The synthesis challenge:** The hardest thing for a research agent isn't finding information — it's synthesizing multiple sources without copying them verbatim, without hallucinating connections, and without losing citations. System prompt design for synthesis. The "cite your sources" instruction and why it works.

**Multi-source research:** The agent searches, reads three sources, finds conflicting information, searches again with refined terms, and produces a synthesized summary with citations. Walk through the tool calls, show the history growing, show the cache saving money.

**Document analysis:** Upload a PDF. Ask questions that require both the document and web search — "How do the findings in this paper compare to current consensus?" The agent reads the paper (from upload), searches the web (with tools), and combines both.

**Sections:**
- The structural equivalence (coding agent → research agent)
- Research tools (search, read, save)
- System prompt for research (synthesis rules, citation requirements)
- A multi-source research session (walk-through with tool calls)
- Document + web (combining uploads with search)
- When the agent gets it wrong (hallucinated citations, paraphrasing failures)

---

### Chapter 6: Multi-Model Agents

**Arc:** The reader discovers that using one model for everything is like using one tool for everything. Some models are cheap and fast. Some are expensive and brilliant. Some can see images. The right architecture uses multiple models, each doing what it's best at.

**Opens with:** A cost breakdown of the coding agent. The agent used Claude Sonnet for everything — including reading file listings and deciding which file to open next. That's a $15/M-token model doing $0.60/M-token work. The realization: 70% of the agent's turns are cheap decisions. 30% are hard edits. You're paying premium prices for routine work.

**Model routing:**
- Cheap model for planning and triage ("which files should I look at?")
- Expensive model for actual code generation and analysis
- `lm15.models(supports={"tools"})` for dynamic selection
- Fallback chains when a provider is down

**Pipeline composition:**
- Generate → critique: one model writes, another reviews
- Vision → text: Gemini describes a screenshot, Claude analyzes the description
- Draft → refine: cheap model writes a first draft, expensive model polishes

**The orchestrator-worker pattern:**
- An orchestrator model breaks the task into sub-tasks
- Worker models execute each sub-task independently
- The orchestrator synthesizes the results
- Implementation: the orchestrator's tools internally call other models

**Parallel execution:**
- `concurrent.futures` with `lm15.complete()` for independent sub-tasks
- When to parallelize (independent analysis) vs when to sequence (dependent steps)

**Key idea:** Multi-model isn't an optimization — it's an architecture. The best agents are ensembles, not monoliths.

**Sections:**
- The cost breakdown (paying $15/M for $0.60/M work)
- Model routing (cheap for triage, expensive for generation)
- Pipeline composition (generate → critique, vision → text)
- The orchestrator-worker pattern
- Parallel execution
- Cost comparison: single-model vs multi-model

---

### Chapter 7: Agents with Approval Gates

**Arc:** Auto-execute is convenient and dangerous. This chapter builds agents where a human stays in the loop — approving destructive actions, reviewing generated content before it's published, controlling what the agent can and can't do autonomously.

**Opens with:** The coding agent, running unattended, writes a migration that drops a production database column. The tests pass because the test database doesn't have real data. The agent declares success. The chapter opens with a disaster and works backward to the architecture that prevents it.

**Manual tools for dangerous actions:**
- `Tool` objects (schema without a function) for write operations
- Auto-execute for read operations (no approval needed to read a file)
- The hybrid: auto-execute `read_file` and `search_code`, manual `write_file` and `run_command`

**Approval patterns:**
- Simple y/n approval on every write
- Show the diff and ask for approval
- Whitelist safe operations, gate dangerous ones
- Batch approval: "here are 5 proposed changes, approve all / review each / reject all"

**The customer support agent:** A full case study. The agent can search the knowledge base (auto), look up orders (auto), update customer records (manual approval), and escalate to a human (manual). System prompt with policy constraints: "never promise a refund over $100 without escalation." The manual tool loop with streaming for real-time chat UX.

**Key idea:** The best agents aren't fully autonomous. They're tools that a human drives — with the human controlling the wheel on the dangerous turns and the AI handling the straightaways.

**Sections:**
- The migration disaster (opens with consequences)
- Auto vs manual: the decision per tool
- The hybrid pattern (auto-read, manual-write)
- Approval UI patterns (y/n, diff review, batch)
- Case study: customer support agent (full build)
- Designing for trust (what the human sees, what they control)

---

## Part III — Production

### Chapter 8: The Economics of Agents

**Arc:** Agents are expensive. Not conceptually — literally. A coding agent that makes 20 tool calls can cost $1-5 per task. A research agent that reads 10 URLs costs $0.50. At scale, these numbers determine whether your agent is viable. This chapter is entirely about making agents affordable.

**Opens with:** An itemized bill. A real 15-turn coding agent session, broken down by turn: input tokens, output tokens, cached tokens, reasoning tokens, cost per turn, cumulative cost. The reader sees exactly where the money goes. (Most of it goes to re-reading the conversation history on every turn.)

**Prompt caching deep dive:**
- Advancing breakpoint pattern (Chapter 8 of Book 1, but applied to real agent loops)
- Before/after cost comparison: same task, with and without caching
- The cache hit ratio across turns (first turn: 0%, second turn: 40%, tenth turn: 85%)
- Provider differences and their cost implications

**Token budgets:**
- Setting a ceiling: `TOKEN_BUDGET = 50_000`
- Tracking cumulative usage across turns
- Graceful degradation: "I've used 80% of the budget. Here's what I've done and what remains."
- Per-turn budget (max_tokens on the model) vs session budget (cumulative)

**Model routing for cost:**
- The 70/30 split: most turns are cheap, few are expensive
- Using `gemini-2.5-flash` ($0.15/M) for planning, `claude-sonnet-4-5` ($3/M) for execution
- Cost math: same 15-turn session, single-model vs routed

**Choosing your spend:**
- The cost-quality tradeoff curve (with real numbers)
- "Is this agent viable?" worksheet
- When to use agents vs simpler pipelines

**Sections:**
- The itemized bill (a real session, broken down)
- Prompt caching in agent loops (before/after, with numbers)
- Token budgets (setting, tracking, graceful degradation)
- Model routing for cost (70/30 split, real savings)
- The viability question (when agents are worth their cost)

---

### Chapter 9: Reliability

**Arc:** Agents fail in ways that single calls don't. A single bad response is a retry. A bad response in turn 12 of 20 corrupts everything that follows. This chapter covers the failure modes unique to agents and the patterns that handle them.

**Opens with:** A taxonomy of agent failures — not abstract, but with examples from the coding and research agents:
- The infinite loop (agent keeps calling the same tool with the same arguments)
- The hallucinated success ("I've completed the task" — but it hasn't)
- The cascading error (one bad edit breaks everything downstream)
- The context poisoning (a misleading early result stays in history and biases all future turns)
- The budget runaway (agent keeps searching, reading, searching, never synthesizing)

**Retries and their limits:**
- `retries=` handles transient API failures (429, 500, timeout)
- It does NOT handle model-level failures (bad output, refusal, hallucination)
- The distinction: infrastructure reliability vs output reliability

**Stuck detection:**
- Same tool, same arguments, N times in a row → stuck
- Same test failure, N fixes attempted → stuck
- Token usage growing but no progress on the task → stuck
- What to do: clear history, rephrase the task, switch models, give up gracefully

**Output validation for agents:**
- After the loop completes, verify the result
- Coding agent: run the tests one final time, run the linter, check the diff
- Research agent: verify citations are real URLs, check for self-contradiction
- The "second opinion" pattern: have a different model review the agent's work

**Error recovery:**
- Tool errors → return error string, let the model adapt
- Model refusal → rephrase and retry (once)
- Context overflow → summarize history and restart
- Provider outage → fallback to a different model

**Sections:**
- Taxonomy of agent failures (with examples)
- Infrastructure reliability (retries, timeouts, fallbacks)
- Output reliability (stuck detection, validation)
- The second-opinion pattern
- Error recovery playbook
- The resilient agent template

---

### Chapter 10: Shipping It

**Arc:** The reader takes everything from the book and builds a production-ready agent. This chapter covers the last mile: logging, observability, deployment, and the patterns that keep an agent running in the real world.

**Opens with:** The complete, production-grade coding agent — 100 lines of Python, zero dependencies beyond lm15. Every technique from every chapter, composed. The reader sees the destination before the journey.

**Observability:**
- Logging every tool call, every response, every token count
- `agent.history` as the complete audit trail
- Structured logs: turn number, tool name, arguments, result length, token usage, latency, cache hit ratio
- Post-mortem analysis: "why did the agent take 25 turns on a task that should have taken 5?"

**The agent as a CLI tool:**
- Argument parsing, streaming output, graceful interruption (Ctrl+C)
- Exit codes: 0 for success, 1 for failure, 2 for budget exceeded
- Machine-readable output (JSON) alongside human-readable streaming

**The agent as a service:**
- Wrapping in a web endpoint (FastAPI, Flask)
- Streaming responses via SSE
- Concurrency: each request gets its own model object (no shared state)
- Timeout and budget enforcement per request

**Testing agents:**
- Unit testing tools in isolation
- Integration testing the full loop with canned tool results
- The "golden transcript" pattern: record a successful run, replay it as a test
- Cost of running tests against live APIs (and how to minimize it)

**The final agent:**
- All the code, in one place, with comments explaining each decision
- Thirty lines of tools
- Ten lines of system prompt
- Fifteen lines of loop with budget, stuck detection, and streaming
- Ten lines of logging
- Works. Ships. Runs at 2 AM.

**Sections:**
- The complete agent (show it all, then explain)
- Observability (logging, metrics, post-mortem)
- CLI packaging (arguments, streaming, exit codes)
- Service packaging (web endpoint, concurrency, timeouts)
- Testing strategies (unit, integration, golden transcripts)
- The final code (copy-pasteable, production-ready)

---

## Chapter Rhythm

| Ch | Title | Words | Core idea |
|---|---|---|---|
| 1 | Twelve Lines | 5,000 | The agent loop is simple; the difficulty is elsewhere |
| 2 | Tool Design | 5,500 | Tools are the agent's API — design them for an AI client |
| 3 | Memory and Its Limits | 5,000 | Context management is the unsexy core of agent engineering |
| 4 | The Coding Agent | 6,000 | Test-driven loops, system prompt design, streaming visibility |
| 5 | The Research Agent | 5,000 | Same loop, different tools — the architecture is general |
| 6 | Multi-Model Agents | 5,000 | Use multiple models, each for what it's best at |
| 7 | Agents with Approval Gates | 5,000 | The best agents keep humans in the loop on dangerous actions |
| 8 | The Economics of Agents | 5,000 | Agents are expensive — here's how to make them viable |
| 9 | Reliability | 5,500 | Agent failure modes are unique and demand unique solutions |
| 10 | Shipping It | 5,000 | The last mile: logging, deployment, testing, production |
| | **Total** | **~52,000** | |

---

## Design Principles

**One project, many expressions.** The coding agent is built in chapters 1-4. The research agent in chapter 5 reuses the architecture with different tools. The reader sees the same pattern applied twice and understands it's general.

**Failures before fixes.** Every chapter opens with something going wrong. The reader understands the problem viscerally before seeing the solution. This is the opposite of documentation, which shows the happy path first.

**Real numbers.** Token counts, dollar amounts, latency measurements. Not "caching reduces cost" but "caching reduced this session from $0.40 to $0.08." The reader can make real decisions.

**No framework.** Everything is `while finish_reason == "tool_call"`. No orchestration library, no agent framework, no hidden abstractions. The reader owns every line and understands every decision. When they later use a framework, they'll know what it's doing underneath.

**Opinions.** Which models are better at coding vs research. When agents are overkill. When to use a simple pipeline instead. When to give up and do it yourself. The book has a point of view.
