from __future__ import annotations

from pathlib import Path

from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.types import LMRequest, Message

from ._helpers import FakeTransport, ProbeResult


def _openai_lines() -> list[bytes]:
    return [
        b'data: {"type":"response.created","response":{"id":"resp_1"}}\n',
        b"\n",
        b'data: {"type":"response.output_text.delta","delta":"ok"}\n',
        b"\n",
        b"data: [DONE]\n",
        b"\n",
    ]


def _anthropic_lines() -> list[bytes]:
    return [
        b'data: {"type":"message_start","message":{"id":"m1","model":"claude"}}\n',
        b"\n",
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n',
        b"\n",
        b'data: {"type":"message_stop"}\n',
        b"\n",
    ]


def _gemini_lines() -> list[bytes]:
    return [
        b'data: {"candidates":[{"content":{"parts":[{"text":"ok"}]}}]}\n',
        b"\n",
    ]


def run(test: dict, root: Path) -> ProbeResult:
    provider = test["provider"]
    req = LMRequest(model="x", messages=(Message.user("hi"),))

    if provider == "openai":
        adapter = OpenAIAdapter(api_key="k", transport=FakeTransport(stream_lines=_openai_lines()))
    elif provider == "anthropic":
        adapter = AnthropicAdapter(api_key="k", transport=FakeTransport(stream_lines=_anthropic_lines()))
    elif provider == "gemini":
        adapter = GeminiAdapter(api_key="k", transport=FakeTransport(stream_lines=_gemini_lines()))
    else:
        return ProbeResult(status="skip", details=f"unsupported provider: {provider}")

    events = list(adapter.stream(req))
    has_delta = any(e.type == "delta" for e in events)
    if not has_delta:
        return ProbeResult(status="fail", details="no delta events")

    return ProbeResult(status="pass", details=f"events={','.join(e.type for e in events)}")
