# Chapter 1: One Function, Three Providers

This is a complete program that calls a language model:

```python
import lm15

resp = lm15.complete("gemini-2.5-flash", "Say hello.", env=".env")
print(resp.text)
```
```
Hello! How can I help you today?
```

Four lines. One of them is a blank. And if you've never called a language model from code before — if you've only ever used ChatGPT in a browser, typed into a box, and watched the reply appear — this is the moment where everything changes. That chat box is a toy. This is a tool. The difference is that you can put it inside a `for` loop.

We're going to build something with this tool. Over the course of this book, you'll construct a research assistant — a program that can search the web, read your documents, analyze images, reason through hard problems, and carry on a conversation while remembering everything you've discussed. It will run on whichever AI provider you want, switch between them on command, and cost you pennies.

But that's later. Right now, you need to get from zero to a working call. This chapter handles the plumbing — install, credentials, first call — and then takes the response apart so you know exactly what came back. It's not glamorous. It's the foundation everything else stands on.

## Install

```bash
pip install lm15
```

That finishes in under a second. I want to dwell on why, because it tells you something about the library you're about to use.

lm15 has zero dependencies. Not "a few small ones." Zero. It doesn't install `requests`, `httpx`, `aiohttp`, `pydantic`, `protobuf`, or any of the other packages that most AI libraries drag in. The HTTP calls go through Python's built-in `urllib`. The data types are frozen dataclasses from the standard library. The whole thing is 408 kilobytes on disk.

This is a deliberate design choice, not a limitation. Every dependency is a surface for breakage — version conflicts, security patches, install failures on weird platforms. A library with 55 dependencies (litellm) or 25 (google-genai) is 55 or 25 things that can go wrong before your code even runs. lm15 has you and the standard library. That's the full trust chain.

The cost of this choice is that lm15 won't give you async, won't give you connection pooling, won't give you a pretty progress bar. If you need those things, you'll reach for other tools later. But you'll be surprised how far you get without them.

## Get an API Key

Language model APIs are paid services — you need an account and a key. If you've never done this before, it feels like a bigger deal than it is. You go to a website, sign up, click "create API key," and copy a string. The whole process takes about two minutes.

You need a key from at least one of these three providers:

| Provider | Environment variable | Sign up |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) |
| Anthropic | `ANTHROPIC_API_KEY` | [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) |
| Google Gemini | `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |

Just one. If you have no existing account and no preference, start with Gemini — Google offers a free tier that's generous enough to follow this entire book without paying.

If you have keys for multiple providers, that's even better. lm15 can use all three simultaneously, and you'll want that flexibility later when we start picking models based on what they're good at.

### Storing your key

Create a file called `.env` in your project folder:

```
GEMINI_API_KEY=AIza...
```

Or, if you have several:

```
OPENAI_API_KEY=sk-proj-abc123...
ANTHROPIC_API_KEY=sk-ant-abc123...
GEMINI_API_KEY=AIza...
```

When you pass `env=".env"` to lm15, it reads this file and picks up every key it recognizes. The keys coexist peacefully — lm15 uses the right one based on which model you call.

Now: add `.env` to your `.gitignore`. Do it now. Not after you finish the chapter, not when you push to GitHub for the first time. Now.

```bash
echo ".env" >> .gitignore
```

API keys that leak to public repositories get scraped by bots within minutes. The resulting bill is not theoretical — it's happened to enough developers that GitHub now scans for exposed keys automatically. Your `.env` file is a wallet. Don't put it on the internet.

> **If you're working in a Jupyter notebook:** `env=".env"` resolves relative to your working directory, not the notebook file. If your notebook lives in `notebooks/` and your `.env` is at the project root, use `env="../.env"`. Or skip the file entirely: run `%env GEMINI_API_KEY=AIza...` in your first cell and omit `env=` from all calls. lm15 checks environment variables automatically.

Every code example in the rest of this book uses `env=".env"`. I'll stop mentioning it after this chapter.

## Your First Call

Here's that four-line program again, but this time, slow down and look at it:

```python
import lm15

resp = lm15.complete("gemini-2.5-flash", "Say hello.", env=".env")
print(resp.text)
```
```
Hello! How can I help you today?
```

`lm15.complete()` is a function call, not a method on some client object. There's no setup step. You don't create an instance of anything. The first argument is the model name — a plain string. The second is your prompt — also a string. `env=` points to your credentials. That's everything.

When you called this function, here's what lm15 did behind the scenes: it read your `.env` file, found your Gemini key, recognized from the prefix `"gemini-"` that this request goes to Google, constructed an HTTP request in Google's wire format, sent it to Google's `generateContent` endpoint, received the response, parsed it out of Google's JSON structure, and wrapped it in a clean `LMResponse` object. You saw none of that. You called a function and got text back.

Now try something more interesting:

```python
resp = lm15.complete("gemini-2.5-flash",
    "What are the three most important things to understand about TCP?",
    env=".env",
)
print(resp.text)
```
```
1. **Reliable delivery** — TCP guarantees that data arrives completely, in order, and without corruption, using acknowledgments and retransmission.

2. **Connection-oriented** — Before any data flows, TCP establishes a connection between sender and receiver through a three-way handshake (SYN, SYN-ACK, ACK).

3. **Flow and congestion control** — TCP dynamically adjusts transmission rate to avoid overwhelming the receiver or the network, using mechanisms like sliding windows and slow start.
```

That response took about a second. The model didn't look anything up — it generated the answer from patterns in its training data, which is both the magic and the limitation of language models. It knows a lot, but it knows nothing that happened after its training cutoff, and it can't check whether what it's saying is true. Keep that in mind. We'll deal with it in Chapter 3, when we give the model tools to actually look things up.

## Taking the Response Apart

Let's look at what `complete()` actually returned. It's not just text — it's an `LMResponse` object with several fields, and understanding them now will save you confusion in every subsequent chapter.

```python
resp = lm15.complete("gemini-2.5-flash",
    "What is the capital of France?", env=".env")

print(resp.text)
print(resp.model)
print(resp.finish_reason)
print(resp.usage)
```
```
The capital of France is Paris.
gemini-2.5-flash
stop
Usage(input_tokens=8, output_tokens=9, total_tokens=17)
```

**`resp.text`** is the part you'll use most — the model's text output as a single string. For a normal question-and-answer call, it's always populated. It returns `None` only in special cases — like when you ask a model to generate an image instead of text — which won't happen until Chapter 6.

**`resp.model`** is the model that actually responded. This sounds redundant — you just told it which model to use. But providers sometimes silently route you to a different version than the one you requested. OpenAI might point `gpt-4.1-mini` at a slightly updated variant. Gemini might resolve a model alias. If you're logging results or comparing models, always read this field rather than assuming the response came from exactly the string you passed in.

**`resp.finish_reason`** tells you why the model stopped talking. This is more important than it looks. `"stop"` means the model finished its thought naturally. `"length"` means it hit a token limit and was cut off mid-sentence — the response is incomplete. `"tool_call"` means the model wants to call a function, which is the entire mechanism behind Chapter 3. For now, you'll only see `"stop"`, but get in the habit of noticing it. A `"length"` hiding in production output is a silent bug.

**`resp.usage`** counts tokens. Tokens are the currency of language models — roughly three-quarters of a word in English, though the exact mapping depends on the model's tokenizer. Every call costs money, and `usage` tells you exactly how much work was done: `input_tokens` is what you sent, `output_tokens` is what the model generated, and `total_tokens` is the sum.

Here's what those tokens cost, in real money:

| Model | Input (per 1M tokens) | Output (per 1M tokens) | This call cost |
|---|---|---|---|
| `gemini-2.5-flash` | $0.15 | $0.60 | $0.000007 |
| `gpt-4.1-mini` | $0.40 | $1.60 | $0.000018 |
| `claude-sonnet-4-5` | $3.00 | $15.00 | $0.000159 |

That last column is for our 17-token call. Seven millionths of a dollar on Gemini. Even Claude — the most expensive — costs less than two hundredths of a cent. The cost only becomes meaningful at scale: a thousand calls, long prompts, large documents. But scale arrives faster than you expect, so I'll flag the cost implications throughout the book rather than ignoring them until the bill arrives.

## Switching Models

Now here's the thing that makes lm15 different from using a provider's SDK directly. Change the model name:

```python
for name in ["gemini-2.5-flash", "gpt-4.1-mini", "claude-sonnet-4-5"]:
    resp = lm15.complete(name, "What is TCP in one sentence?", env=".env")
    print(f"{resp.model:>25}  {resp.text}")
```
```
       gemini-2.5-flash  TCP is a reliable, connection-oriented transport protocol that ensures ordered, error-checked delivery of data between applications over a network.
          gpt-4.1-mini  TCP (Transmission Control Protocol) is a core Internet protocol that provides reliable, ordered, and error-checked delivery of data between applications over a network.
     claude-sonnet-4-5  TCP is a reliable, connection-oriented network protocol that ensures data is delivered accurately and in order between two devices by establishing a connection, breaking data into packets, and verifying their receipt.
```

Three companies. Three APIs with completely different wire formats, authentication schemes, and response structures. And you switched between all three by changing a string.

This is not a small thing. If you'd done this with provider SDKs directly, you'd have three separate code paths — three imports, three client objects, three different ways to construct a message, three different response shapes to parse. You'd maintain that code forever. Every time a provider updates their SDK, you'd update your parser. lm15 absorbs all of that. You see one function, one response type, one set of fields.

The routing works by prefix matching: `gpt-*` goes to OpenAI, `claude-*` to Anthropic, `gemini-*` to Google. If you have a custom model name that doesn't follow this pattern — a fine-tune, a self-hosted model — you can pass `provider="openai"` explicitly. But for every model in this book, the name is enough.

## What Went In and What Came Out

Before we move on, I want to make sure the mental model is clear, because everything in this book builds on it.

You called a function. You passed in a model name and a prompt. You got back a frozen object with text, usage data, and metadata. The call was **stateless** — lm15 doesn't remember it, the model doesn't remember it, nothing was saved anywhere. If you call `complete()` again with a different prompt, the model has no idea you ever asked the first question.

That statelesness is a feature when you're doing one-off tasks — classify this, summarize that, translate this. It becomes a limitation when you want a conversation. We'll deal with that in Chapter 4.

For now, you have the foundation: one function that talks to any provider and returns a clean, consistent response. Every chapter from here adds one capability to this foundation. System prompts that shape behavior. Tools that let the model act. Memory that lets it remember. Streaming that makes it responsive. Each one is the same function with one more argument.

Let's start shaping the output. The model will do what you tell it — but you have to learn how to tell it.
