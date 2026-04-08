from __future__ import annotations

from pathlib import Path

from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.types import LMRequest, Message

from ._helpers import FakeTransport, ProbeResult, load_json_fixture


def run(test: dict, root: Path) -> ProbeResult:
    provider = test["provider"]
    req = LMRequest(model="x", messages=(Message.user("hi"),))

    if provider == "openai":
        payload = load_json_fixture(root, "openai_tool_response.json")
        adapter = OpenAIAdapter(api_key="k", transport=FakeTransport(payload=payload))
    elif provider == "anthropic":
        payload = load_json_fixture(root, "anthropic_tool_response.json")
        adapter = AnthropicAdapter(api_key="k", transport=FakeTransport(payload=payload))
    elif provider == "gemini":
        payload = load_json_fixture(root, "gemini_tool_response.json")
        adapter = GeminiAdapter(api_key="k", transport=FakeTransport(payload=payload))
    else:
        return ProbeResult(status="skip", details=f"unsupported provider: {provider}")

    resp = adapter.complete(req)
    has_tool = any(p.type == "tool_call" for p in resp.message.parts)
    if not has_tool:
        return ProbeResult(status="fail", details="tool call not normalized")
    return ProbeResult(status="pass", details="tool call normalized")
