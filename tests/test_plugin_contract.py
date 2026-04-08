from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.client import UniversalLM
from lm15.errors import UnsupportedFeatureError
from lm15.providers.anthropic import AnthropicAdapter
from lm15.providers.gemini import GeminiAdapter
from lm15.providers.openai import OpenAIAdapter
from lm15.transports.urllib_transport import UrlLibTransport
from lm15.types import EmbeddingRequest


class PluginContractTests(unittest.TestCase):
    def test_support_manifest_present(self):
        t = UrlLibTransport()
        adapters = [
            OpenAIAdapter(api_key="k", transport=t),
            AnthropicAdapter(api_key="k", transport=t),
            GeminiAdapter(api_key="k", transport=t),
        ]
        for a in adapters:
            self.assertIsNotNone(a.supports)
            self.assertIsNotNone(a.manifest)
            self.assertEqual(a.manifest.provider, a.provider)

    def test_dispatch_guard_for_unsupported_embeddings(self):
        lm = UniversalLM()
        lm.register(AnthropicAdapter(api_key="k", transport=UrlLibTransport()))
        with self.assertRaises(UnsupportedFeatureError):
            lm.embeddings(EmbeddingRequest(model="claude-sonnet-4-5", inputs=("hi",)), provider="anthropic")


if __name__ == "__main__":
    unittest.main()
