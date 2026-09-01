"""Run the contract auth-resolution fixtures against explain_auth.

Fixture source of truth: lm15-contract/auth/resolution.json
(spec/auth.md AUTH-1 and AUTH-7, ratified 2026-08-31); this repo's copy lives at
conformance/auth_resolution.json per the dual-landing rule. Divergence
between explain_auth and these fixtures is an implementation bug, never a
reason to edit the fixture (AUTHORITY.md).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from lm15.doctor import explain_auth

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "conformance" / "auth_resolution.json"
FIXTURE = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
SENTINEL = FIXTURE["sentinel"]
CASES = FIXTURE["cases"]


def _materialize_borrowed_file(tmp_path: Path, provider: str, state: str) -> str | None:
    """Write the provider's local OAuth credential file in its AUTH-8 wire
    format: the borrowed Claude Code store, or the lm15-owned store (xai)."""
    if state == "missing":
        return str(tmp_path / "does-not-exist.json")
    if state not in ("fresh", "expired-with-refresh", "expired-no-refresh"):  # pragma: no cover - fixture schema guard
        raise ValueError(f"unknown borrowed_file state {state!r}")
    now_ms = int(time.time() * 1000)
    expiry_ms = now_ms + 3_600_000 if state == "fresh" else 1
    has_refresh = state in ("fresh", "expired-with-refresh")
    if provider == "claude-code":
        oauth: dict = {"accessToken": SENTINEL, "expiresAt": expiry_ms}
        if has_refresh:
            oauth["refreshToken"] = SENTINEL
        body: dict = {"claudeAiOauth": oauth}
    elif provider == "xai":
        entry: dict = {"type": "oauth", "access": SENTINEL, "expires": expiry_ms}
        if has_refresh:
            entry["refresh"] = SENTINEL
        body = {"xai": entry}
    else:  # pragma: no cover - fixture schema guard
        raise ValueError(f"no materializer for provider {provider!r}")
    path = tmp_path / "credentials.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_auth_resolution_contract_case(case: dict, tmp_path: Path) -> None:
    kwargs: dict = {"env": case.get("env", {})}
    if case.get("api_keys_providers"):
        kwargs["api_keys"] = {provider: SENTINEL for provider in case["api_keys_providers"]}
    borrowed = case.get("borrowed_file")
    if borrowed is not None:
        provider = case["provider"]
        assert provider in ("claude-code", "xai"), "fixture oauth cases cover claude-code and xai"
        path = _materialize_borrowed_file(tmp_path, provider, borrowed["state"])
        kwarg = "claude_credentials_path" if provider == "claude-code" else "xai_credentials_path"
        kwargs[kwarg] = path

    report = explain_auth(case["provider"], **kwargs)

    expect = case["expect"]
    assert report.configured is expect["configured"], case["id"]
    actual_steps = [{"kind": step.kind, "state": step.state} for step in report.steps]
    assert actual_steps == expect["steps"], case["id"]

    # AUTH-5: the planted sentinel must never surface in any rendering.
    assert SENTINEL not in report.describe()
    assert SENTINEL not in repr(report)
    assert SENTINEL not in str(report)
