"""Where the conformance checks find their single-source inputs.

Two inputs used to be mirrored into this directory and drifted (recorded in
lm15-contract/changes/2026-09-01-provider-refresh.md):

- the canonical serde vector, whose home is ``lm15-contract/serde/canonical.json``;
- the scraped provider reference pages, whose home is
  ``lm15-contract/scrapes/<provider>/pages/`` next to the scrapers that
  refresh them (moved there from the archived curl-fixtures repository on
  2026-09-02).

The contract repository is expected as a sibling of this one (the layout
the contract harness and ``tests/test_vet_shim.py`` already assume).
``LM15_CONTRACT_DIR`` overrides the checkout; ``LM15_API_REFERENCES``
overrides the scrape directory alone.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = REPO_ROOT.parent


def contract_root() -> Path:
    override = os.environ.get("LM15_CONTRACT_DIR")
    return Path(override).expanduser().resolve() if override else WORKSPACE / "lm15-contract"


def contract_path(*parts: str) -> Path:
    return contract_root().joinpath(*parts)


def api_references_path() -> Path:
    override = os.environ.get("LM15_API_REFERENCES")
    return Path(override).expanduser().resolve() if override else contract_path("scrapes")
