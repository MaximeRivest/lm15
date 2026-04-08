from __future__ import annotations

from pathlib import Path

from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.types import LMRequest, Message

from ._helpers import FakeTransport, ProbeResult, load_json_fixture


def run(test: dict, root: Path) -> ProbeResult:
    provider = test["provider"]
    if provider == "openai":
        payload = load_json_fixture(root, "openai_response.json")
        adapter = OpenAIAdapter(api_key="k", transport=FakeTransport(payload=payload))
        resp = adapter.complete(LMRequest(model="gpt-4.1-mini", messages=(Message.user("hi"),)))
    elif provider == "anthropic":
        payload = load_json_fixture(root, "anthropic_response.json")
        adapter = AnthropicAdapter(api_key="k", transport=FakeTransport(payload=payload))
        resp = adapter.complete(LMRequest(model="claude-sonnet-4-5", messages=(Message.user("hi"),)))
    elif provider == "gemini":
        payload = load_json_fixture(root, "gemini_response.json")
        adapter = GeminiAdapter(api_key="k", transport=FakeTransport(payload=payload))
        resp = adapter.complete(LMRequest(model="gemini-2.0-flash-lite", messages=(Message.user("hi"),)))
    else:
        return ProbeResult(status="skip", details=f"unsupported provider: {provider}")

    has_output = any((p.text or "").strip() for p in resp.message.parts if p.type in {"text", "refusal"})
    if has_output:
        return ProbeResult(status="pass", details="normalized response parsed")
    return ProbeResult(status="fail", details="no normalized text/refusal output")
