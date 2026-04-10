# Chapter 1: One Function, Three Providers — Skeleton

## Chapter Arc

**Reader starts:** Knows Python, has never called an LLM API from code, may have used ChatGPT in a browser.
**Reader ends:** Has installed lm15, configured one API key, made a real call, and can read every field on the response. Knows it works the same across OpenAI, Anthropic, and Gemini.

---

## Section 1: What You're About to Do

**Point:** Set expectations — by the end of this chapter you will have made a real LLM call in four lines of Python, and understood what came back.

> **¶1 — The hook**
> *First sentence:* Most LLM libraries ask you to learn a provider's SDK before you can print "hello."
> *Claim:* lm15 doesn't — one function, one string, one response type, any provider.
> *Last sentence:* By the end of this chapter you'll have called a model, read the response, and switched providers by changing a single word.

---

## Section 2: Install

**Point:** lm15 is trivially small to install — no dependency tree, no surprises.

> **¶1 — The install command**
> *First sentence:* Install lm15 with pip.
> *Claim:* It installs in under a second because it has zero dependencies.
> *Last sentence:* That's everything — no extras, no optional bundles, no C extensions.

```bash
pip install lm15
```

---

## Section 3: Get an API Key

**Point:** You need exactly one key from one provider to start. Here's how to get it.

> **¶1 — Which key**
> *First sentence:* lm15 talks to OpenAI, Anthropic, and Google Gemini — you need a key from at least one.
> *Claim:* One key is enough. Pick whichever provider you already have an account with, or whichever is cheapest to try.
> *Last sentence:* If you have no preference, start with Gemini — the free tier is generous.

> **¶2 — The key table**
> *First sentence:* Here's where to get each key.
> *Claim:* Three rows, three links — this is the complete list.
> *Last sentence:* Copy the key; you'll need it in the next section.

| Provider | Variable | Where |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | platform.openai.com/api-keys |
| Anthropic | `ANTHROPIC_API_KEY` | console.anthropic.com/settings/keys |
| Google Gemini | `GEMINI_API_KEY` | aistudio.google.com/apikey |

> **¶3 — Store it safely**
> *First sentence:* Create a file called `.env` in your project folder and paste the key there.
> *Claim:* The `.env` file keeps secrets out of your code and out of git.
> *Last sentence:* Add `.env` to your `.gitignore` now — before you forget.

```
GEMINI_API_KEY=AIza...
```

```bash
echo ".env" >> .gitignore
```

---

## Section 4: Your First Call

**Point:** A complete LLM call is one import and one function call. It works right now.

> **¶1 — The code**
> *First sentence:* Open a Python file or REPL and type this.
> *Claim:* `lm15.complete()` is the entire API for single-shot calls — model name, prompt, done.
> *Last sentence:* Run it.

```python
import lm15

resp = lm15.complete("gemini-2.5-flash", "Say hello.", env=".env")
print(resp.text)
```

> **¶2 — What just happened**
> *First sentence:* lm15 read your `.env` file, found your Gemini key, sent a request to Google's API, and gave you back a response object.
> *Claim:* You didn't pick a provider, build a client, or configure a transport — `complete()` inferred everything from the model name.
> *Last sentence:* The string `"gemini-2.5-flash"` was enough for lm15 to know where to send the request.

---

## Section 5: Reading the Response

**Point:** The response is a frozen dataclass with six fields you'll use constantly. Learn them now.

> **¶1 — The fields**
> *First sentence:* `resp` is an `LMResponse` — a frozen dataclass, meaning you can read it but never accidentally mutate it.
> *Claim:* Six properties cover 95% of what you'll ever need from a response.
> *Last sentence:* Here they are.

```python
resp.text            # "Hello! How can I help you today?"
resp.model           # "gemini-2.5-flash"
resp.finish_reason   # "stop"
resp.usage.input_tokens   # 4
resp.usage.output_tokens  # 9
resp.usage.total_tokens   # 13
```

> **¶2 — text**
> *First sentence:* `resp.text` is the model's text output — all text parts joined together.
> *Claim:* It's `None` when the response contains no text (e.g., a pure image generation), but for normal calls it's always a string.
> *Last sentence:* This is the field you'll print, parse, or pass downstream in almost every program.

> **¶3 — finish_reason**
> *First sentence:* `resp.finish_reason` tells you *why* the model stopped generating.
> *Claim:* `"stop"` means it finished naturally; `"length"` means it hit the token limit; `"tool_call"` means it wants to call a function — you'll meet this in Chapter 7.
> *Last sentence:* For now, expect `"stop"`.

> **¶4 — usage**
> *First sentence:* `resp.usage` counts tokens — the unit of LLM billing.
> *Claim:* `input_tokens` is what you sent, `output_tokens` is what you got back, `total_tokens` is the sum and what you pay for.
> *Last sentence:* Checking usage after calls is the simplest way to predict your costs.

---

## Section 6: Switching Models

**Point:** Changing providers is changing a string. The code, the response type, the fields — all identical.

> **¶1 — The demonstration**
> *First sentence:* Replace the model name and run the same code.
> *Claim:* Three providers, three model names, same `resp.text`, same `resp.usage`, same everything.
> *Last sentence:* This is the entire value proposition of lm15 in three lines of code.

```python
resp = lm15.complete("gemini-2.5-flash",  "Say hello.", env=".env")  # Google
resp = lm15.complete("gpt-4.1-mini",      "Say hello.", env=".env")  # OpenAI
resp = lm15.complete("claude-sonnet-4-5",  "Say hello.", env=".env")  # Anthropic
```

> **¶2 — How routing works (one sentence)**
> *First sentence:* lm15 infers the provider from the model name prefix — `gpt-*` goes to OpenAI, `claude-*` to Anthropic, `gemini-*` to Google.
> *Claim:* You never construct a client object or pick an endpoint — the string does it.
> *Last sentence:* For custom or fine-tuned model names that don't follow this convention, you can pass `provider="openai"` explicitly — but you won't need that yet.

---

## Section 7: When Things Go Wrong

**Point:** Errors are typed and readable. The three you'll hit first are: missing key, bad model name, network timeout.

> **¶1 — Missing key**
> *First sentence:* If you call a model whose provider has no key configured, lm15 raises `NotConfiguredError`.
> *Claim:* The error message names the provider and the environment variable it looked for.
> *Last sentence:* Fix: add the key to your `.env` file.

```python
# No OPENAI_API_KEY in .env
lm15.complete("gpt-4.1-mini", "Hello.", env=".env")
# → NotConfiguredError: no API key for provider 'openai' (looked for OPENAI_API_KEY)
```

> **¶2 — Bad key or invalid request**
> *First sentence:* A wrong or expired key raises `AuthError`; a malformed request raises `InvalidRequestError`.
> *Claim:* Both are subclasses of `ProviderError`, so you can catch them together or separately.
> *Last sentence:* The error message includes the provider's original response, so you're never guessing.

> **¶3 — The hierarchy (brief)**
> *First sentence:* All lm15 errors descend from `ULMError`.
> *Claim:* `except lm15.ULMError` catches everything lm15 can throw; narrower catches let you handle auth, rate limits, and timeouts differently.
> *Last sentence:* You don't need to memorize this now — just know the tree exists and errors are specific, not generic.

---

## Section 8: What's Next

**Point:** You now have the complete foundation — install, key, call, response, switch, errors. The rest of the book builds on exactly this.

> **¶1 — Forward pointers**
> *First sentence:* You've made stateless, one-shot calls — every call independent, no memory.
> *Claim:* Chapter 2 adds control (system prompts, temperature, output limits); Chapter 4 introduces `model()` for conversation and reuse.
> *Last sentence:* But first, Chapter 2 — because the model will do what you say, if you know how to say it.

---

## Chapter Rhythm Notes

- **Total sections:** 8
- **Code blocks:** 6 (install, .env, first call, response fields, three-provider switch, error example)
- **Tables:** 1 (API key sources)
- **Tone checkpoint:** Peer-to-peer, not tutorial. "Here's how it works" not "Let's learn together." No congratulations, no "great job."
- **Length target:** ~2,500 words / ~8 pages
- **Every section earns its place by teaching one thing the reader will use in every subsequent chapter.**
