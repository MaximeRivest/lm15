from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.discovery import models, providers_info
from lm15.model_catalog import ModelSpec


class DiscoveryTests(unittest.TestCase):
    @patch("lm15.discovery.fetch_models_dev")
    @patch("lm15.discovery._fetch_live_models_for_provider")
    def test_models_live_prefers_provider_fetch(self, fetch_live, fetch_dev):
        fetch_live.side_effect = lambda provider, api_key, timeout: [
            ModelSpec(
                id="gpt-live",
                provider="openai",
                context_window=None,
                max_output=None,
                input_modalities=(),
                output_modalities=(),
                tool_call=False,
                structured_output=False,
                reasoning=False,
                raw={},
            )
        ] if provider == "openai" else []
        fetch_dev.return_value = []

        out = models(provider="openai", live=True, refresh=True, api_key={"openai": "k"})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "gpt-live")
        self.assertEqual(out[0].provider, "openai")

    @patch("lm15.discovery.fetch_models_dev")
    @patch("lm15.discovery._fetch_live_models_for_provider")
    def test_models_fallback_to_models_dev(self, fetch_live, fetch_dev):
        fetch_live.return_value = []
        fetch_dev.return_value = [
            ModelSpec(
                id="gemini-2.5-flash",
                provider="gemini",
                context_window=100,
                max_output=50,
                input_modalities=("text",),
                output_modalities=("text",),
                tool_call=True,
                structured_output=False,
                reasoning=False,
                raw={},
            )
        ]

        out = models(provider="gemini", live=True, refresh=True)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].id, "gemini-2.5-flash")

    @patch("lm15.discovery.models")
    def test_providers_info_reports_configuration_and_counts(self, models_fn):
        models_fn.return_value = [
            ModelSpec(
                id="gpt-4.1-mini",
                provider="openai",
                context_window=None,
                max_output=None,
                input_modalities=(),
                output_modalities=(),
                tool_call=False,
                structured_output=False,
                reasoning=False,
                raw={},
            ),
            ModelSpec(
                id="claude-sonnet-4-5",
                provider="anthropic",
                context_window=None,
                max_output=None,
                input_modalities=(),
                output_modalities=(),
                tool_call=False,
                structured_output=False,
                reasoning=False,
                raw={},
            ),
        ]
        old = os.environ.get("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = "x"
        try:
            info = providers_info(live=False)
        finally:
            if old is None:
                del os.environ["OPENAI_API_KEY"]
            else:
                os.environ["OPENAI_API_KEY"] = old

        self.assertTrue(info["openai"]["configured"])
        self.assertEqual(info["openai"]["model_count"], 1)
        self.assertEqual(info["anthropic"]["model_count"], 1)


if __name__ == "__main__":
    unittest.main()
