from __future__ import annotations

from pathlib import Path

from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.transports.urllib_transport import UrlLibTransport

from ._helpers import ProbeResult


REQUIRED_METHODS = {
    "complete",
    "stream",
    "live",
    "embeddings",
    "file_upload",
    "batch_submit",
    "image_generate",
    "audio_generate",
}


def run(test: dict, root: Path) -> ProbeResult:
    t = UrlLibTransport()
    adapters = [
        OpenAIAdapter(api_key="k", transport=t),
        AnthropicAdapter(api_key="k", transport=t),
        GeminiAdapter(api_key="k", transport=t),
    ]
    for a in adapters:
        missing = [m for m in REQUIRED_METHODS if not hasattr(a, m)]
        if missing:
            return ProbeResult(status="fail", details=f"{a.provider} missing methods: {missing}")
        if not hasattr(a, "supports") or not hasattr(a, "manifest"):
            return ProbeResult(status="fail", details=f"{a.provider} missing supports/manifest")
    return ProbeResult(status="pass", details="all adapters expose full plugin contract")
