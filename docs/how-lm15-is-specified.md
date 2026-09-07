# How lm15 is specified

Most libraries are defined by their code: whatever the implementation does
is, by definition, what the library does. lm15 is built the other way
around. **The behavior is the spec, the spec is machine-checked, and the
implementation — this package included — holds no authority over it.**

This page explains how that works in practice. The normative source for
everything below is the
[lm15-contract](https://github.com/lm15-dev/lm15-contract) repository and
its constitution,
[AUTHORITY.md](https://github.com/lm15-dev/lm15-contract/blob/main/AUTHORITY.md).

## One contract, many implementations

`lm15-contract` is a repository containing no product code. It holds:

- **A written spec** — `spec/types.md` (every canonical type, field by
  field: JSON type, default, omission behavior, constraints),
  `spec/vocabularies.md` (the closed value sets: roles, part types, finish
  reasons, error codes…), `spec/invariants.md` (49 numbered
  construction-time rules, INV-001..INV-049), and `spec/SCOPE.md` (what is
  frozen for 1.0 versus provisional). Serde and response-mapping rules are
  normative in the contract repository (`lm15-contract/docs/serde-rules.md`,
  `lm15-contract/docs/mapping-rules.md`); this repo keeps forwarding stubs at
  [serde-rules](serde-rules.md) and [mapping-rules](mapping-rules.md).
- **A fixture corpus** — canonical requests paired with the exact HTTP
  request each provider must receive, verbatim captured provider response
  and SSE bodies, provider error vectors, and canonical-JSON round-trip
  cases.
- **A language-neutral harness** — `harness/check.py` drives each
  implementation through a small JSONL protocol (build a request, parse a
  response, replay a stream, normalize an error, round-trip serde…) and
  performs **all comparison itself**, with strict typed equality:
  `true ≠ 1`, `1 ≠ 1.0`, absent ≠ `null` ≠ `""` ≠ `[]` ≠ `{}`. It never
  trusts the implementation under test.

Every implementation must pass the identical corpus: **477 checks**
(117 request, 108 response, 9 stream, 18 error, 94 serde, 9 auth,
14 models, 21 live, 23 files, 26 batch, 14 generation, 24 video),
zero failures. Python — this package — is currently the only complete
implementation: earlier Rust, Go, TypeScript, and Julia ports drifted
from the contract and were deliberately deleted rather than left
misleading; they are being rebuilt from the contract, module by module,
and each module ships only when it passes its slice of the corpus. The
same canonical request must produce byte-identical wire requests in
every language.

## Who wins when things disagree

The contract's constitution defines two precedence tracks.

**Wire facts** — what an HTTP request or provider response actually looks
like:

1. Live provider behavior, captured verbatim with a receipt
2. Provider documentation snapshots
3. The contract fixture
4. Any implementation (including this one)

If live behavior contradicts a fixture, the fixture is wrong: re-capture
it, with the receipt attached. If an implementation contradicts a fixture
with no new live evidence, the implementation is wrong: fix the code,
never the fixture.

**Canonical facts** — what the lm15 representation itself is:

1. The written, ratified spec
2. The contract fixture
3. lm15-python — the *reference* implementation: changes land here first,
   but it holds **no oracle authority**
4. The other implementations

## Evidence discipline

No fixture changes without evidence of the right kind, and CI enforces it:

- A **wire fixture** changes only with a live-validation receipt — the
  captured request and response, timestamp, and model.
- A **canonical fixture** changes only with a citation of the normative
  rule that justifies it.
- Every change is recorded as a dated entry in the contract's `changes/`
  ledger, committed together with the edit.
- Every fixture carries a **provenance block** (source, date, evidence);
  a CI gate fails the build if one is missing.
- A **spec drift gate** reflects over this package's public surface and
  fails if it diverges from the spec tables — the documentation cannot
  quietly fall out of sync with the code, in either direction.
- The harness itself is tested by **mutation self-tests**: known bugs are
  injected into the comparator, and the build fails unless every one is
  caught.

## How this package relates to the contract

The file `CONTRACT_PIN` at the root of this repository records the exact
contract commit this code is verified against. On every push, CI checks
out the contract at that pin and runs the full harness against this
package. The pin is never bumped just to make CI green — the diff between
the old and new pin is part of the review.

## What this buys you

- **Cross-provider semantics you can rely on.** "Tool call arguments are
  always named `input`", "`usage.total_tokens: null` means the provider
  didn't report it, never a silent 0" — these aren't conventions of one
  codebase; they are spec, tested on every commit.
- **A frozen core.** The chat-core surface (types, serde, errors, request
  building, response parsing, streaming) is frozen for 1.0; all future
  change to it is additive, and the ratchet that enforces this is
  mechanical, not a promise.
- **Polyglot consistency, by construction.** The contract — not the
  Python code — is the portable artifact. A port earns each capability
  by passing the same checks Python passes, so when the Rust, Go,
  TypeScript, and Julia rebuilds ship, a request serialized in one
  language parses identically in the others because all of them are
  held to the same corpus, not because they share code.
