# Video generation — DESIGN DRAFT

> **This recipe is a design artifact, not documentation.** Nothing on
> this page is implemented. Output blocks marked ⟨captured⟩ are real
> (grok-imagine, live 2026-09-01, transcripts in
> `curl-fixtures/xai-2026-09-01/`); the Sora and Veo blocks are
> invented pending their capture campaign — their wire *shapes* are
> desk-checked below via free probes (list endpoints, validation
> errors), but no money has been spent on their jobs yet. The design
> appendix lists the ratifiable decisions. Do not link this from the
> cookbook index until it is implemented and every output is
> re-captured live.

---

# Video generation

**Problem** — You describe eight seconds of motion and want an MP4.
Video is the one modality that is a **job on every wire that sells
it**: Sora (OpenAI), Veo (Gemini), grok-imagine (xAI) all take your
prompt, return a ticket, and cook for seconds to minutes. lm15 already
has the ticket pattern — batch jobs ([recipe 12](../12-batch-jobs.md))
— and video reuses it instead of inventing a second waiting idiom:

```text
image_generate(request)  -> ImageGenerationResponse   now
video_generate(request)  -> VideoJob                  later (a ticket)
```

## Recipe (aspirational)

### Submit, wait, download

```python
from lm15 import LMRouter, VideoGenerationRequest

router = LMRouter()
lm = router.lm("grok-imagine-video")

job = lm.video_generate(VideoGenerationRequest(
    model="grok-imagine-video",
    prompt="A red ball bounces once on a white floor",
))
print(job.id, job.status)
```
```output ⟨captured wire facts: id + pending⟩
0d309c8f-7733-990b-86d2-45afac7672a2 queued
```

The ticket works exactly like a batch ticket: `job.id` is a plain
string, `wait()` polls to a terminal snapshot, and your process can die
and re-attach with `lm.video_job(id)`.

```python
info = job.wait()
print(info.status, info.progress)

video = job.result()
print(video.media_type, video.url)
```
```output ⟨captured: done in ~30 s, public MP4 URL, duration 8⟩
completed 100
video/mp4 https://vidgen.x.ai/xai-vidgen-bucket/xai-video-0d309c8f-….mp4
```

`result()` returns a **`VideoPart`** — the same part type video inputs
already use, addressed the way the provider delivers it: a URL (xAI's
public bucket, Veo's file URI) or inline bytes (Sora's content
endpoint, downloaded for you). `video.bytes` fetches URL-addressed
results when you want the file.

### Image-to-video

Input images are `ImagePart`s in `request.images`, the media-generation
precedent verbatim — first frame in, motion out. All three wires sell
it (xAI's catalog: text+image in; Sora: `input_reference`; Veo: image
instances).

### Lost the ticket?

```python
for job in lm.video_jobs(limit=5):
    print(job.id, job.status, job.created_at)
```

Works on OpenAI (probed live: `GET /v1/videos` is a real list
endpoint). **xAI has no list endpoint** (probed live: 404) — there,
`video_jobs()` raises and the recipe teaches: store the ticket, it is
the only copy. lm15 does not paper over the asymmetry with a hidden
local journal (rejected for batch, same reasons here).

## Design appendix — desk-checked wire facts

| Fact | OpenAI (Sora) | Gemini (Veo) | xAI (grok-imagine) |
|---|---|---|---|
| Submit | `POST /v1/videos` (validates `prompt`, probed) | `models/veo-*:predictLongRunning` (validates `instances`, probed) | `POST /v1/videos/generations` (captured) |
| Ticket | video object id | operation name | `request_id` (captured) |
| Poll | `GET /v1/videos/{id}` | `GET /v1beta/{operation}` | `GET /v1/videos/{id}` → `pending` + progress % (captured) |
| Terminal | `completed`/`failed` | `done: true` (+ error) | `done` (captured, ~30 s) |
| Result | `/v1/videos/{id}/content` (bytes) | file URI in the operation | public MP4 URL (captured, no auth) |
| List | yes (probed: real endpoint) | operations list | **no** (probed: 404) |
| Duration knob | `seconds` (string enum) | `durationSeconds` | none seen |
| Cancel | documented; unverified | operation cancel; unverified | unknown |

### Decisions to ratify

1. **`video_generate` returns a `VideoJob` ticket** — batch's handle
   pattern verbatim: `wait()`, `result()`, mutate-in-place snapshots,
   re-attach via `lm.video_job(id)`, enumerate via `lm.video_jobs()`
   where the wire lists (raises on xAI — no silent local journal).
2. **Status vocabulary reuses batch's words** where meanings match:
   `queued`, `running`, `completed`, `failed`, `cancelled`. One waiting
   vocabulary across the library; provider words (`pending`, `done`,
   `in_progress`) stay in `provider_data`.
3. **`VideoJobInfo`**: `id`, `status`, `progress` (int percent | None —
   xAI reports it, captured), `created_at?`, `model?`, `provider_data`.
4. **`result()` returns a `VideoPart`** in the provider's own delivery
   mode (URL or bytes) — Part symmetry, no re-hosting, no silent
   download of gigabytes. `part.bytes` fetches on demand.
5. **`VideoGenerationRequest`**: `model`, `prompt`, `images` (input
   frames, ImageParts), `seconds: int | None` (maps Sora `seconds` and
   Veo `durationSeconds`; raises on xAI — no wire slot), `extensions`.
   Audio input (grok-imagine-video-1.5) rides `extensions` until a
   second wire grows the slot.
6. **Pure hooks + shared drivers + async twins**, the files/batch/
   generation architecture; harness direction (`video`) lands with the
   implementation using the same `*_op_build`/`*_op_parse` recipe.
7. **Capture before code**: Sora and Veo submit/poll/result bodies get
   captured at implementation time (they cost real money per job —
   probes above were free); grok-imagine is already fully captured.
   Fixture rule unchanged: no case lands without its live body.
