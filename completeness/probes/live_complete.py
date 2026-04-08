from __future__ import annotations

import os
from pathlib import Path

from lm15 import LMRequest, Message, build_default

from ._helpers import ProbeResult


KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def run(test: dict, root: Path) -> ProbeResult:
    provider = test["provider"]
    key_env = KEY_ENV.get(provider)
    if provider == "gemini" and not os.getenv("GEMINI_API_KEY"):
        key_env = "GOOGLE_API_KEY"

    if not key_env or not os.getenv(key_env):
        return ProbeResult(status="skip", details=f"missing {key_env}")

    lm = build_default(use_pycurl=True)
    req = LMRequest(model=test["model"], messages=(Message.user(test.get("prompt", "Reply with exactly: ok")),))
    resp = lm.complete(req, provider=provider)
    text = "\n".join(p.text or "" for p in resp.message.parts if p.type == "text").strip().lower()
    if not text:
        return ProbeResult(status="fail", details="empty text response")
    return ProbeResult(status="pass", details=f"text_prefix={text[:40]!r}")
