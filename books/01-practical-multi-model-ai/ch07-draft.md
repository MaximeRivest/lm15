# Chapter 7: Thinking Before Answering

Ask our research assistant "Is 9.11 greater than 9.9?" and it might say yes. Confidently. Instantly. And wrong.

```python
import lm15

resp = lm15.complete("claude-sonnet-4-5", "Is 9.11 greater than 9.9?", env=".env")
print(resp.text)
```
```
Yes, 9.11 is greater than 9.9.
```

The model isn't doing arithmetic. It's predicting the next token — and "9.11" *looks* larger because it has more digits. The same way a human might glance at two numbers and pick the longer one, the model's pattern-matching shortcuts fail on problems where the obvious answer is a trap.

But here's what's interesting. Ask a human to actually think about it — to write the comparison out on paper — and they get it right immediately. 9.11 vs 9.90. Of course 9.9 is larger. The error wasn't a lack of knowledge. It was a lack of process. The human had the right answer buried under a wrong intuition, and writing it down surfaced the right one.

Language models can do the same thing. Reasoning mode forces the model to generate a chain of thought — a visible, step-by-step working-out — before producing its final answer:

```python
resp = lm15.complete("claude-sonnet-4-5", "Is 9.11 greater than 9.9?",
    reasoning=True, env=".env")

print("Thinking:")
print(resp.thinking)
print("\nAnswer:")
print(resp.text)
```
```
Thinking:
Let me compare 9.11 and 9.9. To compare decimals, I should align the decimal places: 9.11 vs 9.90. Since 9.90 > 9.11, 9.9 is the larger number.

Answer:
No, 9.9 is greater than 9.11.
```

One parameter — `reasoning=True` — turned a wrong answer into a correct one. The model did the same thing a human would do: it stopped, wrote down the comparison, aligned the decimal places, and got the right answer. The difference between intuition and analysis is the difference between wrong and right on this class of problem.

## When Reasoning Earns Its Keep

Reasoning costs tokens. The thinking step generates output that you pay for, and on hard problems the thinking can be ten or twenty times longer than the answer. So you shouldn't enable it on everything.

I use reasoning when the task involves:

**Math and comparison.** Anything with numbers where the obvious answer might be wrong. Percentages, ratios, decimal comparisons, word problems. The 9.11 vs 9.9 case is a classic, but there are subtler ones — "which is a better deal, 30% off or buy-two-get-one-free?" requires actual calculation.

**Logic and constraints.** "Given these five rules, which option is valid?" The model needs to check each option against each rule, and without reasoning it'll pattern-match instead of checking.

**Multi-step problems.** "Read this code, find the bug, explain why it occurs, and suggest a fix." Each step depends on the previous one. Without reasoning, the model skips to the fix and sometimes fixes the wrong thing.

**High-stakes tasks.** If being wrong is expensive — legal analysis, medical triage, financial decisions — the extra tokens for reasoning are insurance.

I don't use reasoning for:

**Simple factual recall.** "What's the capital of France?" Reasoning won't make "Paris" more correct.

**Text generation.** "Write a poem about autumn." The model doesn't need to reason about poetry — it needs to create.

**Classification with clear labels.** "Is this email spam?" The model's pattern matching is fine here.

The rule of thumb I keep coming back to: if you could solve it in your head without writing anything down, the model doesn't need reasoning either. If you'd reach for paper and pen, turn on reasoning.

## Reading the Model's Work

`resp.thinking` contains the chain of thought. It's often more interesting than the answer — it's where you see the model's strategy, its false starts, its self-corrections:

```python
resp = lm15.complete("claude-sonnet-4-5",
    "A bat and a ball together cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
    reasoning=True, env=".env")

print(resp.thinking)
```
```
The intuitive answer is $0.10, but let me check. If the ball costs $0.10, then the bat costs $1.00 more, which is $1.10. Together that's $1.20, not $1.10. So $0.10 is wrong.

Let me set up the equation. Let b = ball price. Bat = b + $1.00. Total: b + (b + $1.00) = $1.10. So 2b + $1.00 = $1.10. 2b = $0.10. b = $0.05.

Check: ball = $0.05, bat = $1.05, total = $1.10. ✓
```

The model caught the intuitive wrong answer ($0.10), proved it wrong, set up the algebra, solved it, and verified. This is the cognitive behavioral therapy version of a language model — it's arguing with its own intuitions and winning.

For debugging, `resp.thinking` is invaluable. When a model gives a wrong answer with reasoning enabled, the chain of thought shows you *where* the reasoning went wrong. Was the setup correct but the arithmetic bad? Did it misinterpret the question? Did it forget a constraint? You can diagnose the failure mode and fix your prompt accordingly.

## Controlling the Effort

Different providers give you different controls over how hard the model thinks. The simplest approach — `reasoning=True` — works everywhere and uses sensible defaults. When you need finer control:

**Anthropic: token budget.** Set a maximum number of tokens the model can spend thinking:

```python
# Hard problem — let it think longer
resp = lm15.complete("claude-sonnet-4-5", "Prove that √2 is irrational.",
    reasoning={"budget": 10000}, env=".env")

# Easy problem — don't waste tokens
resp = lm15.complete("claude-sonnet-4-5", "What's 17 * 16?",
    reasoning={"budget": 1000}, env=".env")
```

**OpenAI: effort level.** A qualitative setting:

```python
resp = lm15.complete("gpt-4.1-mini", "Solve this logic puzzle...",
    reasoning={"effort": "high"}, env=".env")
```

Valid levels: `"low"`, `"medium"`, `"high"`. Higher effort means more thinking tokens.

For our research assistant, `reasoning=True` on demand is the right approach. Most questions — "What was the Marshall Plan?" — don't need reasoning. But when someone asks the assistant to compare two analyses, evaluate an argument, or work through a calculation, reasoning makes the difference between a plausible guess and a verified answer.

## The Cost of Thinking

Reasoning tokens aren't free. They count toward your bill, and on hard problems they can dominate the cost:

```python
resp = lm15.complete("claude-sonnet-4-5",
    "Prove that the set of prime numbers is infinite.",
    reasoning=True, env=".env")

u = resp.usage
print(f"Input:     {u.input_tokens:>6} tokens")
print(f"Output:    {u.output_tokens:>6} tokens")
print(f"Reasoning: {u.reasoning_tokens:>6} tokens")
print(f"Total:     {u.total_tokens:>6} tokens")
```
```
Input:        12 tokens
Output:       85 tokens
Reasoning:   340 tokens
Total:       437 tokens
```

Reasoning consumed 340 tokens — four times the visible output. For a hard math proof, the ratio can be 20:1. This is why you don't enable reasoning on a batch of 10,000 classification calls — each classification uses 5 output tokens, and 100 reasoning tokens per call would add $0.30 of pure overhead at Claude's pricing.

The mental model: reasoning is insurance. Cheap insurance on hard problems where being wrong costs more than thinking. Expensive insurance on easy problems where the model would've been right anyway.

## Reasoning in Streams

In Chapter 5, we built a streaming research session. Adding reasoning gives the user visibility into the model's thought process:

```python
import lm15

assistant = lm15.model("claude-sonnet-4-5", env=".env",
    system="You are a research assistant. Show your work on analytical questions.")

stream = assistant.stream(
    "Company A grew revenue 40% but profit dropped 10%. Company B grew revenue 15% with profit up 25%. Which is in a stronger position and why?",
    reasoning=True,
)

for event in stream:
    match event.type:
        case "thinking": print(f"💭 {event.text}", end="", flush=True)
        case "text":     print(f"\n\n{event.text}", end="", flush=True)
        case "finished":
            u = event.response.usage
            print(f"\n\n[{u.reasoning_tokens} reasoning tokens]")
```

The user watches the model think — analyzing revenue growth rates, evaluating profit margins, considering sustainability — before seeing the conclusion. In a research tool, this transparency is the feature. The user can evaluate the reasoning, not just the answer.

## Combining Everything

Here's what happens when reasoning meets tools and conversation history — a research assistant that can think, search, and build on prior context:

```python
import lm15

def search(query: str) -> str:
    """Search the web for current information."""
    return "..."

assistant = lm15.model("claude-sonnet-4-5",
    system="You are a research assistant. Search for facts. Reason through analytical questions.",
    tools=[search],
    temperature=0,
    env=".env",
)

# Turn 1: factual question (no reasoning needed)
assistant("What was Japan's GDP in 2023?")

# Turn 2: factual question (no reasoning needed)
assistant("What about South Korea's?")

# Turn 3: analytical question (reasoning helps)
resp = assistant(
    "Given the population difference, which country has higher GDP per capita? Show your work.",
    reasoning=True,
)
print(resp.thinking)
print(resp.text)
```

The model searches for GDP data on turns 1 and 2, stores the results in conversation history, and then reasons through the per-capita comparison on turn 3 — dividing the GDP figures by population estimates, showing each step. Tools for data. Memory for context. Reasoning for analysis. Three features, composed.

Every chapter in this book has added one capability to the same `complete()` function. System prompts shape behavior. Tools give the model hands. Memory gives it context. Streaming gives it responsiveness. Multimodal gives it senses. Reasoning gives it rigor. They all compose because they're all parameters to the same function.

Two chapters remain. The next one — prompt caching — makes all of this cheaper. The last one makes it reliable.
