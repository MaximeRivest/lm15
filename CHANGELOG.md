# Changelog

## 0.1.0 - 2026-04-08

Initial public release.

### Added

- Universal normalized contract for requests/responses/messages/parts/stream events.
- Provider adapters for OpenAI, Anthropic, and Gemini.
- Transport layer with urllib and optional pycurl implementations.
- Middleware chain (cache/history/retries).
- Completeness harness with fixture and live probe modes.
- External plugin discovery via entry-points (`lm15.providers`).
- Documentation set and cookbook series.
