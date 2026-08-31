a# Batch jobs — DESIGN DRAFT (aspirational)

> **This recipe is a design artifact, not documentation.** Nothing on
> this page is implemented. The `output` blocks are invented, not
> captured. It exists to answer one question first — *what is the best
> shape to teach?* — before any code decides for us. The design
> appendix at the bottom desk-checks every call against the real
> provider wire APIs. Do not link this from the cookbook index until
> it is implemented and every output is re-captured live.

---

# Batch jobs

**Problem** — You have five hundred requests that nobody is waiting
on. Running them through `complete()` costs full price and busies your
process; what you want is to hand the whole stack to the provider,
walk away, and pick up the answers later at half price. Every provider
has this — behind three different wire shapes, two file formats, and a
status zoo. lm15 gives you a third execution mode next to the two you
know:

```text
complete(request)  -> Response            now,    full price
stream(request)    -> events -> Response  now,    full price, incremental
batch(requests)    -> BatchJob            later,  ~half price
```

Same canonical `Request` in all three. Batch is not a different kind
of asking — it is the same conversation on a different clock.

Keys loaded as in [recipe 01](01-first-request.md).

## Recipe

### Submit

`batch()` takes ordinary `Request` objects — the same ones you would
pass to `complete()`, each carrying its own model:

```python
from lm15 import LMRouter, Message, Request

router = LMRouter()
lm = router.lm("claude-sonnet-4-5")

capitals = ["France", "Japan", "Brazil", "Kenya", "Norway"]
job = lm.batch([
    Request(model="claude-sonnet-4-5",
            messages=(Message.user(f"Capital of {country}? One word."),))
    for country in capitals
])
print(job.id, job.status)
```
```output
msgbatch_01XYZ… queued
```

That is the whole submission. A batch job is a **ticket**: the provider
holds your requests, `job.id` is the ticket number, and the id is a
plain string — store it in a file, a database, an env var. Your process
can exit now.

### Re-attach and wait

Hours later, in a different process, the id is all you need:

```python
job = lm.batch_job("msgbatch_01XYZ…")   # one GET; fails fast on a bad id
print(job.status, job.done)
```
```output
running False
```

For small batches (providers often finish in minutes, not hours), you
can just block:

```python
job.wait(poll_every=30.0)   # polls until terminal; returns the job
print(job.status)
```
```output
completed
```

`wait()` is a convenience for notebooks and small jobs. The primary
pattern for real workloads is store-the-id + re-attach — a batch
outlives your process by design.

### Lost the ticket?

You will lose one eventually — a notebook restart, a crash between
submit and save. That is fine, because **the provider is the system of
record, not your memory**. Every provider can enumerate your jobs:

```python
for job in lm.batches(limit=5):        # newest first
    print(job.id, job.status, job.created_at, job.label)
```
```output
msgbatch_01XYZ… running   2026-08-31T14:02:11Z nightly-eval
msgbatch_01ABC… completed 2026-08-30T02:00:09Z nightly-eval
msgbatch_01DEF… completed 2026-08-29T02:00:12Z None
```

This is the same lesson Unix `atq`, printer queues, and HPC `squeue`
taught decades ago: submitters forget, queues remember. Losing the id
is an inconvenience, not a loss.

To make your job findable among many, label it at submit time:

```python
job = lm.batch(reqs, label="nightly-eval-2026-08-31")
```

The label rides the provider's metadata field. If a provider's wire
has nowhere to carry it, submitting with a label raises
`UnsupportedFeatureError` — lm15 never silently drops what you asked
for. Retrying a crashed submitter? List first, filter by label,
submit only if absent — that pattern also protects you from paying
twice for the same batch.

### Results

Results come back **in submission order**, one entry per request, and
partial failure is a first-class outcome, not an exception:

```python
for entry in job.results():
    if entry.ok:
        print(entry.index, entry.response.text)
    else:
        print(entry.index, "FAILED:", entry.error.code, entry.error.message)
```
```output
0 Paris
1 Tokyo
2 Brasília
3 Nairobi
4 Oslo
```

`entry.response` is a full canonical `Response` — parts, usage,
finish_reason, everything `complete()` would have given you. Batch
entries are parsed by the **same frozen response mapping** as chat;
there is no second parser to trust.

Each entry's `outcome` is one of `succeeded`, `errored`, `cancelled`,
`expired`. `entry.ok` is sugar for `outcome == "succeeded"`.

### Cancel

```python
job = lm.batch([...])
job.cancel()
print(job.status)
```
```output
cancelling
```

Cancellation is a request, not a guarantee: entries already processed
still bill and still return results (`outcome="succeeded"`), the rest
come back `cancelled`. The job ends in `cancelled` (or `completed` if
it beat you to the finish line).

## How it works

One mental model, seven states, two vocabularies:

- **Job status** (the lifecycle of the ticket):
  `queued → running → completed`, with exits to `failed`,
  `cancelling → cancelled`, and `expired`. `job.done` is true in the
  four terminal states.
- **Entry outcome** (the fate of each request): `succeeded`,
  `errored`, `cancelled`, `expired`. A `completed` job can contain
  errored entries — that is normal, check your entries.

These are separate dimensions on purpose. "The job ended" and "your
third request had a bad model name" are different facts; conflating
them is how batch APIs usually hurt people.

On the wire, `batch()` maps to each provider's native queue —
Anthropic Message Batches, the OpenAI Batch API, Gemini Batch Mode —
all ~50% of interactive price, all with a 24-hour completion target
(usually much faster). Provider quirks stay under the hood: OpenAI
requires the requests as an uploaded JSONL file, so lm15 uploads one
for you; the file id is visible in `job.provider_data` if you want it.
lm15 assigns positional custom ids internally and re-sorts provider
results so `results()` order always equals submission order.

There is **no silent fallback**. If a provider has no batch queue
(ollama, OpenRouter, the subscription adapters), `batch()` raises
`UnsupportedFeatureError` — it never quietly loops `complete()` at
full price while you believe you are getting the discount. If you want
a client-side fan-out, write the loop; the loop is honest.

## Variations

- **Async twin.** `AsyncLMRouter().lm(...)` gives `await lm.batch()`,
  `await job.wait()`, `await job.results()` with identical types.
- **Mixed models, one batch.** Entries carry their own model; batch a
  cheap model and an expensive one together if the provider allows it
  (Anthropic does; OpenAI batches are per-endpoint, not per-model).
- **Provider knobs** ride `extensions` on submit, as everywhere:
  `lm.batch(reqs, extensions={"completion_window": "24h"})`.
- **Retention.** Results do not live forever (Anthropic ~29 days;
  OpenAI until you delete the file). Fetch and persist what you need.
- **Cost check.** Each entry's `response.usage` is real usage; sum it
  and apply your pricing table — the discount shows up on the invoice,
  not in the usage numbers.

## See also

- [12 — Batch & media generation](../12-batch-media-generation.md) —
  today's submit-only surface this design replaces
- [15 — Errors, retries & testing](../15-errors-and-testing.md)
- [16 — Provider passthrough](../16-provider-passthrough.md)

---

# Design appendix (not part of the recipe)

## Desk check: can the ideal shape actually be built?

Yes. Every call above decomposes into four **pure operations** per
provider — exactly the build/parse pattern the contract already pins
for chat and models:

| operation | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| submit | `POST /v1/messages/batches` (inline `requests[]`, each params = a Messages body) | upload JSONL file (`purpose=batch`) → `POST /v1/batches` (`endpoint=/v1/responses`) | `POST …:batchGenerateContent` (inline or file-based) |
| status | `GET /v1/messages/batches/{id}` | `GET /v1/batches/{id}` | poll the long-running operation |
| results | `GET results_url` → JSONL, one Messages response per line | `GET /v1/files/{output_file_id}/content` (+ `error_file_id`) → JSONL | operation response inline, or output file |
| cancel | `POST …/cancel` | `POST …/cancel` | batch cancel |
| list | `GET /v1/messages/batches` (paginated) | `GET /v1/batches` (paginated) | `batches.list` |

Capture-campaign verification items: which providers carry a
job-level label (OpenAI `metadata` — yes; Gemini `displayName` —
yes; Anthropic — verify at capture time) and whether any batch-create
endpoint accepts a true idempotency key (map through `extensions` if
so).

The load-bearing fact: **every per-entry result body is a normal chat
response in that provider's wire format.** Batch build reuses the
frozen request mapping; batch results reuse the frozen response
mapping. The contract's new surface is thin: four op pairs plus two
vocabularies. This is why the three-mode symmetry is not a marketing
line — it is literally the same serde underneath.

Status mapping (canonical ← provider):

| canonical | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| queued | — | `validating` | `JOB_STATE_PENDING` |
| running | `in_progress` | `in_progress`, `finalizing` | `JOB_STATE_RUNNING` |
| cancelling | `canceling` | `cancelling` | — |
| completed | `ended` | `completed` | `JOB_STATE_SUCCEEDED` |
| failed | — (job-level failure is per-entry) | `failed` | `JOB_STATE_FAILED` |
| cancelled | `ended` + all-cancelled counts | `cancelled` | `JOB_STATE_CANCELLED` |
| expired | — (per-entry only) | `expired` | `JOB_STATE_EXPIRED` |

Entry outcomes: Anthropic's `succeeded/errored/canceled/expired` is
adopted verbatim (US spelling normalized to `cancelled`); OpenAI maps
from per-line `status_code`/`error`; Gemini from per-item response vs
status.

## Decisions taken in this draft (each reversible until ratified)

1. **Kill the silent local fan-out.** Today OpenAI/Gemini `batch_submit`
   without extensions quietly loops `complete()` at full price with
   `status="completed"`. An expert API reviewer vetoes this hardest:
   it fakes the one property (price/queue) the user came for.
   Trade-off: users who relied on it must write a three-line loop.
2. **Hide OpenAI's file plumbing.** The JSONL upload is wire syntax,
   not semantics; `batch()` owns it. Trade-off: lm15 creates a file on
   the user's account per batch and does not delete it (the batch
   references it); the id stays visible in `provider_data`.
3. **Vocabulary change.** `BatchStatus` drops `submitted` (merged into
   `queued`), gains `cancelling` and `expired`. Pre-1.0, so a cut, not
   a break — but it touches the frozen enum list, hence ratification.
4. **Two-layer API.** Canonical, pure, port-friendly layer:
   `batch_submit`, `batch_status(id)`, `batch_results(id)`,
   `batch_cancel(id)` returning frozen types (`BatchJobInfo`,
   `BatchEntry`). Ergonomic layer: the `BatchJob` handle the cookbook
   teaches, pure sugar over the four ops. Ports must implement the
   ops; the handle is per-language idiom. The contract pins the ops.
5. **`refresh()`/`wait()`/`cancel()` update the handle in place** and
   return it. The frozen-types culture applies to canonical types, not
   client handles; a mutating refresh avoids the
   forgot-to-reassign-stale-status bug. Named alternative: immutable
   handles (`job = job.refresh()`); rejected for that footgun, but
   cheap to swap before implementation.
6. **Positional identity, no user-facing custom ids** in v1. lm15
   numbers entries internally and re-sorts results; `entry.index` is
   the correlation key. Additive later: a `custom_ids=` kwarg if a
   real need appears.
7. **`results()` on a non-terminal job raises** (with the current
   status in the message) instead of implicitly waiting. Explicit
   beats magic; `wait()` is one line away.
8. **Job-level `request_counts` stay in `provider_data`** for v1.
   Providers disagree on the buckets; normalizing them is additive
   later. Trade-off: progress bars need provider-specific code for
   now.
9. **Enumerability is a core operation, not a nicety.** `lm.batches()`
   (pure op: `batch_list`) exists because forty years of job systems
   (`atq`, `lpq`, `squeue`, cloud ListJobs) converged on the same
   truth: submitters forget, queues remember. All three providers
   ship a list endpoint, so the coat-check attendant is free.
   Recovery from a lost id must never depend on the user having been
   careful.
10. **Optional `label=` on submit**, surfaced as `job.label` and
    filterable client-side from `batches()`. Mapped to provider
    metadata; where the wire cannot carry it, raise — the no-silent-
    drop invariant applies to labels like everything else. Trade-off:
    cross-provider code that labels must handle the raise or skip the
    label.
11. **No hidden local journal** (rejected: auto-writing
    `~/.local/state/lm15/batches.jsonl` on submit). Named reason: a
    journal is prior art for tools that ARE the system of record
    (git, terraform); here the provider is, and it is enumerable.
    lm15's stated culture is "reads nothing implicitly" — a library
    that secretly writes state breaks the same promise, misbehaves in
    containers and multi-tenant apps, and creates a second truth that
    drifts from the first. Users who want a journal write one line in
    their own storage.
12. **No `get_or_submit(label=...)`** (rejected: Kubernetes-style
    idempotent named create). Named reason: no provider enforces
    label uniqueness server-side, so the API would imply a race-free
    guarantee lm15 cannot keep — two processes can still double-
    submit. The honest form is the documented pattern (list → filter
    by label → submit if absent) plus real idempotency keys via
    `extensions` wherever a provider ever offers one.

## What ratification would unlock, in order

1. Type/vocab changes (`BatchStatus`, `BatchJobInfo`, `BatchEntry`,
   serde kinds) — offline.
2. Reference implementation: the five ops × three providers + the
   `BatchJob` handle + async twins, tests against synthetic bodies.
3. Live capture campaign (one tiny real batch per provider — costs
   cents) → pinned bodies, cases, goldens.
4. Harness direction `batch` (build/parse per op, results comparison
   reusing the response comparator) + selftest mutations.
5. This draft graduates: outputs re-captured live, file moved out of
   `drafts/`, recipe 12 updated to point at it.
