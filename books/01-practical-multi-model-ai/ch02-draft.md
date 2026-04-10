# Chapter 2: Shaping the Response

Ask a model "Summarize DNA" and you'll get a perfectly fine answer. Ask it again and you'll get a different one. Ask it ten times and you'll get ten different answers — different lengths, different styles, different levels of detail, all of them technically correct and none of them what you wanted.

This is the fundamental tension of working with language models: they're capable of almost anything, which means they'll produce almost anything unless you tell them not to. An unconstrained model is like an enthusiastic intern — smart, eager, and liable to write a five-paragraph essay when you needed a one-line answer. The fix isn't better models. It's better instructions.

lm15 gives you five parameters to shape a model's output. This chapter covers all five, and by the end you'll have a pattern you'll use in every chapter that follows — a config dict that pins down exactly what you want and gets the same result every time.

## System Prompts: The Most Important Parameter

Every parameter in this chapter matters. But if you remember only one, make it `system=`. A system prompt is an instruction the model reads before your actual prompt — a set of standing orders that shape everything it produces.

Watch the difference:

```python
import lm15

resp = lm15.complete("gemini-2.5-flash", "Summarize DNA.", env=".env")
print(resp.text)
```
```
DNA, or deoxyribonucleic acid, is the hereditary material in almost all living organisms. It carries the genetic instructions for development, functioning, growth, and reproduction. DNA is structured as a double helix, composed of two strands of nucleotides, each containing a sugar, a phosphate group, and one of four nitrogenous bases: adenine (A), thymine (T), guanine (G), and cytosine (C). The specific pairing of these bases (A with T, G with C) enables DNA to replicate and transmit genetic information accurately.
```

Fine. Accurate. Also four sentences when I needed one, written for a biology textbook when I needed a cocktail-party answer. Now with a system prompt:

```python
resp = lm15.complete("gemini-2.5-flash", "Summarize DNA.",
    system="One sentence. No jargon. A smart ten-year-old should understand it.",
    env=".env",
)
print(resp.text)
```
```
DNA is a tiny instruction manual inside every cell in your body that tells it how to grow and work.
```

Same model. Same prompt. Completely different output. The system prompt didn't add information — the model already knew everything about DNA. It changed *how* the model expressed that knowledge. It's the difference between "tell me about X" and "explain X to a child in one sentence."

This is what makes system prompts the most powerful tool you have. They control persona, tone, format, length, audience, and constraints — all in plain English.

Here's a system prompt for the research assistant we're going to build:

```python
SYSTEM = """You are a research assistant. When asked a question:
- Answer with specific facts, not generalities
- Cite dates, numbers, and names when you know them
- If you're uncertain, say so explicitly
- Keep answers under 3 paragraphs unless asked for more
- Never invent citations or references"""
```

This is a good system prompt because it's specific, testable, and constraining. "Be helpful" is a bad system prompt — it doesn't change anything. "Answer with specific facts, not generalities" changes everything. The model will still get things wrong sometimes, but it will get them wrong in a predictable, identifiable way.

A few more patterns worth having in your back pocket:

```python
# Strict format control
system = "Output valid JSON only. No markdown, no explanation, no preamble."

# Domain expert persona
system = "You are a senior security engineer. Analyze code for vulnerabilities. Rate each finding as LOW, MEDIUM, HIGH, or CRITICAL."

# Behavioral constraints
system = "You are a customer service agent for Acme Corp. Never discuss competitors. Never promise features that don't exist. If unsure, say 'Let me check on that.'"
```

Notice how each one is a list of rules, not a paragraph of aspiration. Models follow instructions better when the instructions are concrete and enumerable.

## Temperature: Controlling Randomness

Language models don't calculate the right answer. They predict the most likely next word — or, more precisely, they assign a probability to every possible next word and then sample from that distribution. Temperature controls how they sample.

```python
# temperature=0 — always picks the most probable next token
resp = lm15.complete("gpt-4.1-mini", "Name a color.", temperature=0, env=".env")
print(resp.text)  # "Blue" — every time
```

```python
# temperature=1.0 — samples proportionally from the distribution
resp = lm15.complete("gpt-4.1-mini", "Name a color.", temperature=1.0, env=".env")
print(resp.text)  # "Magenta" — different every time
```

```python
# temperature=1.8 — flattens the distribution, makes unlikely tokens more probable
resp = lm15.complete("gpt-4.1-mini", "Name a color.", temperature=1.8, env=".env")
print(resp.text)  # "Cerulean-saffron" — creative, unpredictable, possibly nonsensical
```

At `temperature=0`, the model is deterministic. Same input, same output, every time. This is what you want for classification, extraction, and anything where creativity is a bug. At `temperature=1.0`, the model is balanced — likely tokens are still favored, but there's room for surprise. Above `1.5`, you're in exploration territory — useful for brainstorming, dangerous for production.

The mental model I use: temperature is a knob between "reliable" and "interesting." Turn it all the way left and the model is a boring, consistent machine. Turn it all the way right and it's an unreliable poet. Most production work lives between 0 and 0.5.

For our research assistant, we want `temperature=0`. Research is not the place for creative interpretation. When I ask "What's the population of Tokyo?", I want the same answer every time — not a creative take.

## Max Tokens: Capping Output

`max_tokens=` sets a hard ceiling on how many tokens the model can generate. It's not a target — it's a wall. The model doesn't aim for that length; it stops when it hits the limit, even mid-sentence:

```python
resp = lm15.complete("gemini-2.5-flash", "Explain general relativity.",
    max_tokens=30, env=".env")
print(resp.text)
print(f"finish_reason: {resp.finish_reason}")
```
```
General relativity, proposed by Albert Einstein in 1915, is a theory of gravity that describes it not as a force, but as
finish_reason: length
```

The model was cut off. `finish_reason` is `"length"` instead of `"stop"` — a signal that the response is incomplete. This is important: if you see `"length"` in production, the model had more to say and was silenced. That's sometimes a bug.

The right way to control response length is to combine `system=` (which tells the model your *intent*) with `max_tokens=` (which enforces a *budget*):

```python
resp = lm15.complete("gemini-2.5-flash", "Explain general relativity.",
    system="Two sentences maximum.",
    max_tokens=200,
    env=".env",
)
print(resp.text)
print(f"finish_reason: {resp.finish_reason}")
```
```
General relativity is Einstein's theory describing gravity as the curvature of spacetime caused by mass and energy. Objects follow the straightest possible paths through this curved spacetime, which we perceive as gravitational attraction.
finish_reason: stop
```

Two sentences, finished naturally, well within the 200-token budget. The system prompt controlled the intent; `max_tokens` was a safety net. This distinction matters a lot in production. If you're processing a thousand documents and each summary should be two sentences, you want `system=` doing the steering and `max_tokens=100` catching the rare case where the model ignores the instruction and writes an essay. Without the safety net, a misbehaving model can burn through your token budget in one runaway response.

## Top-p and Stop Sequences

Two more parameters exist. I'll cover them briefly because you'll use them rarely, but knowing they exist saves you from reinventing them.

**`top_p=`** (nucleus sampling) is an alternative way to control randomness. Instead of scaling the probability distribution like temperature does, it truncates it — the model only considers tokens that fit within the top `p` cumulative probability mass. At `top_p=0.1`, only the most likely tokens are candidates. At `top_p=0.95`, nearly everything is.

In practice: adjust temperature *or* top_p, never both simultaneously. They're two controls for the same thing. Most people use temperature because it's more intuitive. I've never needed top_p in production.

**`stop=`** takes a list of strings. When the model generates one of them, it stops immediately — the stop string itself isn't included in the output:

```python
resp = lm15.complete("gpt-4.1-mini", "List five fruits, one per line.",
    stop=["\n4."], env=".env")
print(resp.text)
```
```
1. Apple
2. Banana
3. Cherry
```

The model was about to write the fourth item and was stopped. I've found this useful exactly twice in real projects — once to extract just the first paragraph of a multi-paragraph response, and once to stop a code generator before it produced test cases I didn't want. It's a niche tool.

## The Config Dict Pattern

Here's where all five parameters pay off together. You're going to type the same system prompt, temperature, and max_tokens on many calls. That repetition is a signal — it should be data, not code:

```python
import lm15

researcher = {
    "model": "gemini-2.5-flash",
    "system": """You are a research assistant. When asked a question:
- Answer with specific facts, not generalities
- Cite dates, numbers, and names when you know them
- If you're uncertain, say so explicitly
- Keep answers under 3 paragraphs unless asked for more
- Never invent citations or references""",
    "temperature": 0,
    "max_tokens": 500,
    "env": ".env",
}

resp = lm15.complete(prompt="What caused the 2008 financial crisis?", **researcher)
print(resp.text)
```
```
The 2008 financial crisis was primarily caused by the collapse of the U.S. housing bubble and the proliferation of mortgage-backed securities (MBS) and collateralized debt obligations (CDOs) that were tied to subprime mortgages.

Starting in the early 2000s, low interest rates and relaxed lending standards led to a surge in home loans to borrowers with poor credit. These mortgages were bundled into complex financial instruments and sold to investors worldwide, with credit rating agencies like Moody's and S&P assigning them misleadingly high ratings. When housing prices began to fall in 2006–2007, defaults spiked, and the securities became worthless.

The crisis escalated in September 2008 when Lehman Brothers filed for bankruptcy on September 15, and AIG required an $85 billion Federal Reserve bailout. The resulting credit freeze spread globally, triggering the deepest recession since the 1930s.
```

That's our research assistant's first answer — factual, specific, dated, within the length constraint. And the config that produced it is a Python dict we can version, share, and reuse.

Try swapping the model:

```python
resp = lm15.complete(prompt="What caused the 2008 financial crisis?",
    **{**researcher, "model": "claude-sonnet-4-5"})
print(resp.text)
```

Same instructions, different model, different prose style — but the same factual constraints. The dict is your research assistant's personality. The model is the engine. You can swap engines without changing the personality.

This is the pattern we'll use for the rest of the book. In Chapter 4, we'll upgrade from a dict to a model object that remembers conversation history. But the config dict never stops being useful — it's how you test configs, share them with teammates, and load them from YAML files.

## What You Can Do Now

Two chapters in and you have the complete toolkit for single-shot calls. Let me be specific about what that means — because "single-shot" covers more real work than most people realize.

**Classification.** Is this email spam? Is this review positive? What's the severity of this bug report? These are all one prompt, one answer, temperature zero.

**Extraction.** Pull the person's name and age from this sentence. Find all dates mentioned in this paragraph. Extract the dollar amounts from this invoice. System prompt defines the format, prefill (Chapter 9) forces JSON, max_tokens caps the cost.

**Summarization.** Condense this article. Summarize this meeting transcript. Give me the three key takeaways from this paper. System prompt controls length and audience.

**Translation.** "Translate to French" is a system prompt. "Output only the translation, no explanation" makes it reliable.

**Rewriting.** Make this email more professional. Simplify this legal clause. Rewrite this error message for end users.

All of these are stateless. Each call is independent. You don't need conversation history, tools, or streaming. You need `complete()` with a good system prompt and `temperature=0`.

Our research assistant can already answer questions. What it can't do yet is *act* — look things up, read files, search the web. That requires tools, which is where we're going next. The model is about to get hands.
