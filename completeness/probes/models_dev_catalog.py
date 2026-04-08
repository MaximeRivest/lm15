from __future__ import annotations

from pathlib import Path

from lm15.model_catalog import fetch_models_dev

from ._helpers import ProbeResult


def run(test: dict, root: Path) -> ProbeResult:
    specs = fetch_models_dev(timeout=15.0)
    if not specs:
        return ProbeResult(status="fail", details="empty catalog")
    providers = {s.provider for s in specs}
    if not {"openai", "anthropic", "google", "gemini"}.intersection(providers):
        return ProbeResult(status="fail", details="expected providers not found")
    return ProbeResult(status="pass", details=f"models={len(specs)} providers={len(providers)}")
