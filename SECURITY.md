# Security Policy

## Supported versions

Only the latest release published on PyPI receives security fixes.
The legacy `0.2.x` line (branch `legacy-0.2`) is unsupported and will
not receive fixes.

## Reporting a vulnerability

Do not open a public issue for a security problem.

Report privately, either way works:

1. **GitHub private vulnerability reporting** (preferred):
   [github.com/lm15-dev/lm15-python/security/advisories/new](https://github.com/lm15-dev/lm15-python/security/advisories/new)
2. **Email** the security contact: Maxime Rivest <mrive052@gmail.com>
   with a subject starting with `[lm15 security]`.

Include what you can: affected version, provider/endpoint involved, a
minimal reproduction, and the impact you believe it has.

## What to expect (coordinated disclosure)

- Acknowledgement of your report within **7 days**.
- An assessment (accepted, declined, or needs more info) within **14 days**.
- A coordinated fix and public disclosure within **90 days** of the
  report, sooner when a fix is ready. If more time is genuinely needed,
  we will agree on a new date with the reporter.
- Fixed vulnerabilities are published as
  [GitHub Security Advisories](https://github.com/lm15-dev/lm15-python/security/advisories)
  with affected versions and upgrade instructions.

## Scope notes

lm15 is a low-level client library. Things we consider security bugs
include, but are not limited to:

- API keys or other credentials being logged, serialized, or sent to a
  host other than the intended provider endpoint.
- Requests being routed to a different endpoint than the one the code
  specifies (for example through URL construction bugs).
- Unsafe handling of untrusted provider responses (streaming/SSE
  parsing, JSON deserialization) that leads to more than a clean error.

Bugs without a security impact belong in the
[issue tracker](https://github.com/lm15-dev/lm15-python/issues).
