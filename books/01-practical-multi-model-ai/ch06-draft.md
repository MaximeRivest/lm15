# Chapter 6: Eyes and Ears

Our research assistant can search the web, carry a conversation, and stream its responses in real time. But it's text-in, text-out. Hand it a photograph and it has no idea what to do. Send it a PDF and it can't read it. Play it an audio recording and it hears nothing.

That limitation is falling away faster than most people realize. The big models from OpenAI, Anthropic, and Google can now see images, read documents, listen to audio, and — in Google's case — watch video. The same `complete()` function handles all of it. You just stop passing a string and start passing a list:

```python
import lm15
from lm15 import Part

resp = lm15.complete("gemini-2.5-flash", [
    "What does this chart show?",
    Part.image(url="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/CO2_Emissions_by_Source_Since_1880.svg/800px-CO2_Emissions_by_Source_Since_1880.svg.png"),
], env=".env")
print(resp.text)
```
```
This chart shows global CO₂ emissions by source from 1880 to approximately 2020. The largest contributor is coal (shown in dark gray), followed by oil and natural gas. Total emissions have increased roughly exponentially, with a particularly steep rise after 1950. The chart also shows smaller contributions from cement production and gas flaring.
```

A list instead of a string. Text parts and media parts, interleaved. That's the entire API change. The model saw the chart, read its axes and legends, identified the data sources, and described the trend — all from a URL pointing to an image.

## Images

Images are the most broadly supported modality — all three providers handle them. You can send them from a URL, from bytes in memory, or from a file on disk.

**From a URL:**

```python
resp = lm15.complete("claude-sonnet-4-5", [
    "Describe the composition of this photograph.",
    Part.image(url="https://example.com/photo.jpg"),
], env=".env")
```

**From a file:**

```python
data = open("screenshot.png", "rb").read()
resp = lm15.complete("gpt-4.1-mini", [
    "What does this UI show? Are there any usability issues?",
    Part.image(data=data, media_type="image/png"),
], env=".env")
```

When you pass bytes, `media_type=` is required — lm15 can't guess the format from raw data.

**Comparing images:**

```python
before = Part.image(url="https://example.com/room_before.jpg")
after = Part.image(url="https://example.com/room_after.jpg")

resp = lm15.complete("gemini-2.5-flash", [
    "These are before and after photos of the same room. What changed?",
    before,
    after,
], env=".env")
```

The model sees both images in order and can compare them. I've used this for UI regression testing — screenshot before, screenshot after, "tell me what changed" — and it catches layout shifts, color changes, and missing elements that pixel-diff tools miss because it understands *what* the elements are, not just that pixels moved.

### Generating images

Some models can produce images too. Pass `output="image"`:

```python
resp = lm15.complete("gpt-4.1-mini", "Draw a cat wearing a top hat, watercolor style.",
    output="image", env=".env")
```

`resp.image` is a `Part` containing the generated image. The interesting trick is that you can pass this `Part` directly to another model:

```python
# Generate on OpenAI
resp = lm15.complete("gpt-4.1-mini", "Draw a sunset over mountains.", output="image", env=".env")

# Critique on Claude
resp2 = lm15.complete("claude-sonnet-4-5", [
    "Rate this image on composition and color. Be specific.", resp.image
], env=".env")
print(resp2.text)
```

The `Part` is portable across providers. No file saving, no format conversion, no re-encoding. One model generates, another evaluates. This is the start of multi-model pipelines, and it works because lm15 normalizes the media format under the hood.

## Documents

Documents — especially PDFs — are where multimodal gets practical. Our research assistant can already search the web. Now we can hand it actual files.

```python
from lm15 import Part

pdf_data = open("quarterly_report.pdf", "rb").read()
resp = lm15.complete("claude-sonnet-4-5", [
    "What were the key findings in this report? List the top 3.",
    Part.document(data=pdf_data, media_type="application/pdf"),
], env=".env")
print(resp.text)
```

The model reads the PDF — tables, charts, headers, footnotes, everything — and responds based on its contents. This is not OCR. The model understands the document's structure and meaning, not just the pixels on the page.

### Uploading large files

For big documents — hundreds of pages, large media files — embedding the bytes in the request body is fragile and slow. The provider's file API is more reliable:

```python
import lm15

doc = lm15.upload("claude-sonnet-4-5", "contract.pdf", env=".env")
```

`lm15.upload()` sends the file to the provider's storage and returns a `Part` with a reference ID. You pass that `Part` in your prompt like any other:

```python
resp = lm15.complete("claude-sonnet-4-5", ["Find all indemnification clauses.", doc], env=".env")
```

The file lives on the provider's side. You don't re-upload it on every call. This is especially valuable on model objects, where you can ask many questions about the same document:

```python
claude = lm15.model("claude-sonnet-4-5", env=".env")
doc = claude.upload("contract.pdf")

claude(["Summarize the payment terms.", doc])
claude(["What are the termination conditions?", doc])
claude(["Are there any unusual liability clauses?", doc])
```

Each call sends the file reference, not the file bytes. Combined with conversation history, the model accumulates understanding across turns — "unusual" in the third question is informed by the model's knowledge of what it found in the first two.

### Caching documents for repeated queries

If you're making many stateless calls against the same document (no conversation, just different questions), per-part caching avoids reprocessing the document every time:

```python
contract = Part.document(
    data=open("contract.pdf", "rb").read(),
    media_type="application/pdf",
    cache=True,
)

resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 1.", contract], env=".env")
resp = lm15.complete("claude-sonnet-4-5", ["Summarize section 2.", contract], env=".env")
resp = lm15.complete("claude-sonnet-4-5", ["Find liability clauses.", contract], env=".env")
```

The first call processes the PDF and caches it. The next two reuse the cached version — faster and cheaper. This is a preview of Chapter 8, where we'll cover caching comprehensively.

## Audio and Video

Audio and video follow the same pattern — `Part.audio()` and `Part.video()` — but with narrower provider support.

**Audio generation:**

```python
resp = lm15.complete("gpt-4o-mini-tts", "Say 'hello world' in a warm, friendly tone.",
    output="audio", env=".env")
# resp.audio is a Part containing the generated audio
```

**Cross-modal pipeline:**

```python
# Generate speech on OpenAI, transcribe on Gemini
resp = lm15.complete("gpt-4o-mini-tts", "Say: the quick brown fox.", output="audio", env=".env")
resp2 = lm15.complete("gemini-2.5-flash", ["Transcribe this audio.", resp.audio], env=".env")
print(resp2.text)  # "The quick brown fox."
```

**Video understanding** (Gemini only):

```python
resp = lm15.complete("gemini-2.5-flash", [
    "What happens in this video? List the key events with timestamps.",
    Part.video(url="https://example.com/clip.mp4"),
], env=".env")
```

### What works where

Not every provider supports every modality. This table saves you from `InvalidRequestError`:

| Capability | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| Image input (vision) | ✅ | ✅ | ✅ |
| Image generation | ✅ | — | ✅ |
| Document input (PDF) | ✅ | ✅ | ✅ |
| Audio input | ✅ | — | ✅ |
| Audio generation | ✅ | — | — |
| Video input | — | — | ✅ |
| File upload | ✅ | ✅ | ✅ |

When building cross-model pipelines, pick the right provider for each step. Gemini for video and audio understanding. OpenAI for audio generation. Any of the three for vision and documents.

## Mixed-Media Prompts

A prompt list can contain any combination of text and media, in any order:

```python
resp = lm15.complete("gemini-2.5-flash", [
    "I'm redecorating my living room.",
    Part.image(url="https://example.com/current_room.jpg"),
    "Here's the current state. And here's a furniture catalog:",
    Part.document(url="https://example.com/catalog.pdf"),
    "Suggest three pieces from the catalog that would fit this room. Explain why.",
], env=".env")
```

Text strings and `Part.*()` calls, interleaved in the order you want the model to see them. This works with both `complete()` and `model()`.

## Our Research Assistant Gets Eyes

Let's add document analysis to the research assistant:

```python
import lm15
from lm15 import Part

def search(query: str) -> str:
    """Search the web for current information."""
    return "..."

assistant = lm15.model("gpt-4.1-mini",
    system="""You are a research assistant. You can:
- Search the web for current information
- Read and analyze documents the user provides
When given a document, read it thoroughly before answering.
Cite specific sections, pages, or figures when relevant.""",
    tools=[search],
    temperature=0,
    env=".env",
)

# Upload a paper
paper = assistant.upload("research_paper.pdf")

# Ask questions about it
assistant(["What methodology did the authors use?", paper])
assistant(["Were there any limitations they acknowledged?"])
resp = assistant("How do their findings compare to the current consensus? Search for recent work on this topic.")
print(resp.text)
```

The third question is where the magic happens. The model has read the paper (from the upload), discussed its methodology and limitations (from conversation history), and now combines that understanding with a live web search to compare the paper's findings to recent work. Document knowledge and web search, connected through conversation memory.

This is what a research assistant should be. Not a chatbot that generates plausible text. A tool that reads your documents, searches for context, and synthesizes both into a reasoned answer. We're getting close. The next chapter adds reasoning — the ability to think through hard problems step by step instead of guessing.
