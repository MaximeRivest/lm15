from __future__ import annotations

import os

from .capabilities import hydrate_with_specs
from .client import UniversalLM
from .model_catalog import fetch_models_dev
from .plugins import load_plugins
from .providers.anthropic import AnthropicAdapter
from .providers.gemini import GeminiAdapter
from .providers.openai import OpenAIAdapter
from .transports.base import TransportPolicy
from .transports.pycurl_transport import PyCurlTransport
from .transports.urllib_transport import UrlLibTransport


def build_default(
    use_pycurl: bool = True,
    policy: TransportPolicy | None = None,
    hydrate_models_dev: bool = False,
    discover_plugins: bool = True,
) -> UniversalLM:
    policy = policy or TransportPolicy()
    transport = UrlLibTransport(policy=policy)
    if use_pycurl:
        try:
            import pycurl  # noqa: F401

            transport = PyCurlTransport(policy=policy)
        except Exception:
            transport = UrlLibTransport(policy=policy)

    client = UniversalLM()

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        client.register(OpenAIAdapter(api_key=openai_key, transport=transport))

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        client.register(AnthropicAdapter(api_key=anthropic_key, transport=transport))

    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        client.register(GeminiAdapter(api_key=gemini_key, transport=transport))

    if hydrate_models_dev:
        try:
            hydrate_with_specs(fetch_models_dev())
        except Exception:
            pass

    if discover_plugins:
        load_plugins(client)

    return client
