# Maintainers

## Current maintainers

| Name | GitHub | Role |
| --- | --- | --- |
| Maxime Rivest | [@MaximeRivest](https://github.com/MaximeRivest) | Lead maintainer |

## Roles and responsibilities

**Lead maintainer** — reviews and merges changes, triages issues and
vulnerability reports (see [SECURITY.md](SECURITY.md)), cuts releases,
maintains CI, and is the final authority on design questions. The lead
maintainer is accountable for keeping `lm15-python` conformant to
`lm15-contract` at the pinned SHA.

**Contributor** — anyone submitting issues or pull requests. No standing
access. Expectations are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Access to sensitive resources

| Resource | Who has access |
| --- | --- |
| GitHub organization `lm15-dev` (owner) | Maxime Rivest |
| This repository (admin: settings, branch protection, secrets) | Maxime Rivest |
| PyPI project [`lm15`](https://pypi.org/project/lm15/) (owner) | Maxime Rivest |
| Documentation deployment (GitHub Pages, via CI) | Maxime Rivest |

There are no shared credentials. Releases are published from CI via
PyPI Trusted Publishing (OIDC), not with long-lived tokens.

## Becoming a maintainer

The project currently has a single maintainer. Sustained, high-quality
contributions are the path to co-maintainership; candidates are vetted
and granted access explicitly by the lead maintainer. Adding a
maintainer updates this file in the same change that grants access.
