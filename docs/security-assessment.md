# Security assessment

*Last performed: 2026-09-01, against the 1.0.0 release candidate.
Re-run this assessment for any release that adds a new transport,
provider, credential flow, or parser.*

This page records what could realistically go wrong when you use lm15,
what the library does about it, and what it deliberately leaves to you.
The vulnerability reporting process lives in
[SECURITY.md](https://github.com/lm15-dev/lm15-python/blob/main/SECURITY.md).

## Trust model

- **You choose the endpoints.** lm15 sends requests to the base URLs you
  (or the built-in provider defaults) configure. It never follows
  redirects and never rewrites a host behind your back.
- **Providers are semi-trusted.** We trust them with the request content
  you send, but we parse their responses defensively: a malicious or
  compromised server should produce a clean `ProtocolError` or
  `TransportError`, not memory exhaustion or code execution.
- **Your machine is trusted.** Credentials read from environment
  variables or local credential files are assumed to be yours. lm15
  protects them from *accidental* disclosure (logs, reprs, world-readable
  files), not from an attacker who already controls your machine.

## What an attacker might try, and what stands in the way

### Steal API keys or OAuth tokens

- All provider `api_key` fields are declared `repr=False`; the OAuth
  adapters (Claude Code, Codex) override `__repr__` explicitly. Printing
  or logging a provider object does not disclose secrets.
- `lm15.doctor` explains credential resolution without ever returning
  secret material, by construction.
- Refreshed OAuth credentials are written atomically with `0o600`
  permissions (owner-only) under a lock file (`lm15/_authlock.py`).
- OAuth login flows use PKCE (S256, 64 random bytes from `secrets`), a
  callback listener bound to `127.0.0.1` only, and `state` checking.
  Token refresh goes to hardcoded HTTPS endpoints.
- Known trade-off: the Gemini Live websocket authenticates with
  `?key=` in the URL, as the API requires. The connection is `wss://`
  (encrypted), so the exposure is client-side URL logging only. If
  Google's live API gains handshake-header auth, we should switch.

### Intercept traffic (adversary in the middle)

- TLS certificate verification is **on by default**, using the system
  trust store via `ssl.create_default_context()`.
- `verify=False` exists for local development against self-signed
  servers. It is never a default and must be typed explicitly. Prefer
  `ca_bundle=` for private test CAs.
- Proxies: only plain-HTTP proxies are supported; HTTPS targets are
  tunneled with CONNECT and the TLS handshake runs end-to-end to the
  origin — the proxy never sees inside the tunnel.

### Smuggle or inject through the HTTP layer

The stdlib-only HTTP/1.1 codec (`lm15/transports/_http11.py`) is the
most exposed code and is written accordingly:

- CR/LF/NUL bytes are rejected in the method, target, host, and every
  header name and value (CRLF-injection screen).
- Response heads are capped at 64 KB; chunk-size lines at 16 KB.
- `Transfer-Encoding` stacking (`gzip, chunked`) is rejected;
  malformed or negative `Content-Length` is rejected; data past the
  declared length is a protocol error. These are the classic
  request-smuggling and desync levers.
- SSE streams enforce a 64 KB per-line and 1 MB per-event limit.
- Connect, read, and write timeouts all have non-infinite defaults.

### Trick the library into running code

- The codebase contains no `eval`, `exec`, `pickle`, subprocess, or
  shell invocation. All parsing of provider data is `json` +
  hand-written byte parsers.
- There is **no automatic tool-execution loop**: when a model requests a
  tool call, lm15 hands the parsed request to your code and does
  nothing on its own. Executing model-chosen actions is always an
  explicit decision of the application.

### Poison the supply chain

- Runtime dependencies: **none** (Python stdlib only). The single
  optional extra is `websockets`, needed only for live sessions.
- Behavior is pinned by the cross-language conformance corpus in
  `lm15-contract` at an exact SHA (`CONTRACT_PIN`); CI fails on drift.

## Accepted risks (known and deliberate)

1. **Unbounded non-streaming response bodies.** A hostile server could
   send an enormous body to a non-streaming call and exhaust memory.
   You choose your endpoints, so we accept this; a `max_body` cap is a
   candidate hardening if untrusted-endpoint use cases appear.
2. **`verify=False` exists.** It is an explicit, documented,
   local-development-only escape hatch.
3. **Gemini Live key-in-URL** (see above) — required by the API today.

## Re-assessment triggers

Run this assessment again (and update this page) when any of these
change: a new transport or protocol, a new credential flow, a new
parser over untrusted bytes, redirect support, or a new dependency.
