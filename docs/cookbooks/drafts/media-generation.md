# Media generation — DESIGN DRAFT

> **This recipe is a design artifact, not documentation.** The current
> `image_generate` / `audio_generate` code is a prototype that predates
> the contract discipline; this page decides the shape it should have
> before that code is rebuilt. Unlike the batch draft, every `output`
> block here is **real**: captured live on 2026-09-01 against OpenAI
> and Gemini (transcripts in `curl-fixtures/genmedia-2026-09-01/`).
> The design appendix desk-checks every field against those captures
> and lists the ratifiable decisions. Do not link this from the
> cookbook index until the rebuild lands.

---

# Media generation

**Problem** — Sometimes the thing you want back is not text. You want
a picture of the diagram you described, or your paragraph read aloud.
Both OpenAI and Gemini sell this; Anthropic does not. The wires could
not look more different — OpenAI has a dedicated `/images` endpoint
and a raw-bytes `/audio/speech` endpoint, while Gemini does both
through the same chat call it uses for everything — but the *shape* of
the exchange is identical: a prompt goes in, media parts come out.

```text
image_generate(request)  -> ImageGenerationResponse   images as ImagePart
speech_generate(request) -> SpeechGenerationResponse  audio as AudioPart
```

The outputs are the **same Part types you already send as inputs**
([recipe 04](04-images-and-documents.md)). A generated image is an
`ImagePart` you can save, or feed straight back into a chat request.
Media generation is not a new world — it is the part system running in
reverse.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

### Generate an image

```python
from lm15 import LMRouter, ImageGenerationRequest

router = LMRouter()
lm = router.lm("gpt-image-1-mini")

resp = lm.image_generate(ImageGenerationRequest(
    model="gpt-image-1-mini",
    prompt="A simple flat red circle centered on a white background",
    size="1024x1024",
))
img = resp.images[0]
print(img.media_type, len(img.data), resp.usage.output_tokens)
```
```output
image/png 1422420 272
```

`media_type` is not a guess: OpenAI's response states
`"output_format": "png"` and lm15 reads it. The tokens are real too —
image models bill in tokens, and `usage` carries them.

Save it like any inline part:

```python
import base64
from pathlib import Path

Path("circle.png").write_bytes(base64.b64decode(img.data))
```

### The same call on Gemini

```python
lm = router.lm("gemini-2.5-flash-image")

resp = lm.image_generate(ImageGenerationRequest(
    model="gemini-2.5-flash-image",
    prompt="A simple flat red circle centered on a white background",
))
print(resp.images[0].media_type, resp.text)
```
```output
image/png Here's a simple flat red circle centered on a white background: 
```

Two things happened. First: under the hood this was an ordinary chat
call — Gemini has no separate image endpoint; its image models answer
`generateContent` with an image part inline. Second: Gemini often
**talks while it draws**. That commentary arrives in `resp.text`,
because throwing away words the model said would be silent data loss.
On OpenAI, `text` is `None` — the wire has no such thing.

`size` takes each provider's own sizing vocabulary — exactly like
`model` and `voice`, the field is portable but the values are not.
OpenAI speaks pixels (`"1024x1024"`, `"1536x1024"`); Gemini speaks
aspect ratios (`"16:9"`, `"1:1"`):

```python
resp = lm.image_generate(ImageGenerationRequest(
    model="gemini-2.5-flash-image",
    prompt="A wide banner of rolling hills",
    size="16:9",
))
```

### Generate speech

```python
from lm15 import SpeechGenerationRequest

lm = router.lm("gpt-4o-mini-tts")

resp = lm.speech_generate(SpeechGenerationRequest(
    model="gpt-4o-mini-tts",
    prompt="Hello from the capture campaign.",
    voice="alloy",
    format="wav",
))
print(resp.audio.media_type)
Path("hello.wav").write_bytes(base64.b64decode(resp.audio.data))
```
```output
audio/wav
```

Every field except `model` and `prompt` is optional, and **omitting a
field means the server decides, not lm15**. Leave out `format` and
OpenAI sends you MP3 (`audio/mpeg`) — its default, honestly reported
in `media_type`, never silently rewritten to something lm15 prefers.
Leave out `voice` and you get the provider's default voice.

On Gemini, speech is again just a chat call to a TTS model, and the
audio comes back as raw 24 kHz PCM with the most honest media type on
any wire:

```python
lm = router.lm("gemini-2.5-flash-preview-tts")

resp = lm.speech_generate(SpeechGenerationRequest(
    model="gemini-2.5-flash-preview-tts",
    prompt="Hello from the capture campaign.",
    voice="Kore",
))
print(resp.audio.media_type)
```
```output
audio/L16;codec=pcm;rate=24000
```

Gemini has no format knob — it always speaks PCM. So `format="wav"`
on Gemini **raises** instead of being dropped: lm15 never accepts a
field it cannot deliver (`format` rides `extensions`-free only where
the wire has a slot for it, same rule as batch labels on Anthropic).

### Providers that cannot draw

```python
router.lm("claude-opus-5").image_generate(...)
```
```output
UnsupportedFeatureError: anthropic: image generation not supported
```

Anthropic has no generation endpoint of any kind. lm15 says so
instead of routing your prompt somewhere you did not choose.

### What about video?

Sora (OpenAI) and Veo (Gemini) are **jobs**, not calls — you submit,
wait minutes, then download. That is the batch ticket pattern
([recipe 12](12-batch-jobs.md)) applied to a different factory, and it
gets its own design pass. This page deliberately covers only the
synchronous pair.

---

## Design appendix — wire truth and ratifiable decisions

Everything below is desk-checked against live captures from
2026-09-01 (`curl-fixtures/genmedia-2026-09-01/`), not against
documentation.

### What the wires actually do

| Fact | OpenAI | Gemini |
|---|---|---|
| Image endpoint | `POST /v1/images/generations` | `generateContent` on `*-image` models |
| Image encoding | `data[n].b64_json`, format in `output_format` | `inlineData` part, format in `mimeType` |
| Text alongside image | never | routinely (captured) |
| Image usage | token counts, image/text split | `usageMetadata`, modality split |
| Image `id` / `model` echo | **absent** from response | `responseId`, `modelVersion` |
| Sizing vocabulary | pixels (`1024x1024`) | aspect ratio via `imageConfig` (`16:9` → 1344×768 captured) |
| Speech endpoint | `POST /v1/audio/speech` | `generateContent` + `speechConfig` on `*-tts` models |
| Speech response | **raw bytes**, truth in `content-type` header | `inlineData` part with full MIME (`audio/L16;codec=pcm;rate=24000`) |
| Speech default format | `audio/mpeg` (captured) | PCM only, no knob |
| `voice` required? | no (captured: server default exists) | no (captured) |
| Speech usage | none (raw body) | `usageMetadata` |
| DALL·E / Imagen | **gone from the model list** | **absent from the API** |

### Decisions to ratify

1. **Rename the audio pair to speech.** `AudioGenerationRequest` →
   `SpeechGenerationRequest`, `audio_generate` → `speech_generate`.
   Both wires sell text-to-speech; "audio generation" promises music
   and sound effects that neither wire offers. An expert names the
   endpoint what it does. Cheap now (the types have no serde kind yet,
   nothing is frozen); a lie forever after 1.0.
2. **Media types come from the wire, never from code.** OpenAI images:
   `output_format`. OpenAI speech: the `content-type` header. Gemini:
   `inlineData.mimeType` verbatim — including the parameterized
   `audio/L16;codec=pcm;rate=24000`, which `AudioPart` already accepts.
   The prototype hardcodes `image/png`; captured truth made that a
   coincidence, not a fact.
3. **Kill the client-side defaults.** The prototype injects
   `voice="alloy"` and `format="wav"`. Both captured as optional on
   the wire with server defaults. lm15 sending its own defaults
   misreports what the provider does; omitted means omitted.
4. **`format` raises where the wire has no slot** (Gemini). Same rule
   as batch `label` on Anthropic: no silent drops.
5. **`ImageGenerationResponse` gains `text: str | None`.** Captured:
   Gemini returns commentary text next to the image. Without the
   field, canonical parsing silently discards model output — the one
   sin the contract exists to prevent. OpenAI sets it to `None`.
6. **Parse usage.** Both image wires bill real tokens and say so; the
   prototype returns an empty `Usage`. OpenAI speech genuinely has
   none — empty `Usage` is honest there.
7. **`size` is canonical, vocabulary is provider's.** Precedent:
   `model` and `voice` already work this way. OpenAI gets it verbatim;
   Gemini gets it as `imageConfig.aspectRatio`. Trade-off stated: a
   `size` string that works on one provider fails on the other —
   exactly like a model name, and honestly.
8. **`id` and `model` stay optional in responses.** Captured: OpenAI
   images return neither; Gemini returns both (`responseId`,
   `modelVersion`).
9. **Multi-image (`n`), quality, background ride `extensions`.**
   OpenAI-only knobs today; promote later if a second wire grows them.
10. **Serde kinds** for the four request/response types, with media
    payloads as base64 fields exactly as parts already serialize.
    This closes most of the "types without kinds" debt.
11. **Video is out of scope here** and reuses the ticket pattern when
    it comes. Named reason for not bundling it: sync-in, job-out are
    different teaching shapes, and this page teaches the sync pair.
12. **Chat-door generation stays chat.** Gemini image models used in
    ordinary `complete()` calls return image parts inside a normal
    `Response` — that already works and stays untouched. The dedicated
    verbs exist for the dedicated intent, not as the only door.
