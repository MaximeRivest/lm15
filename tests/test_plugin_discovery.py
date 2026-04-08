from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.client import UniversalLM
from lm15.features import EndpointSupport, ProviderManifest
from lm15.plugins import discover_provider_entry_points, load_plugins


class _Adapter:
    provider = "mockp"
    capabilities = None
    supports = EndpointSupport(complete=True, stream=True)
    manifest = ProviderManifest(provider="mockp", supports=supports)

    def complete(self, request):
        raise NotImplementedError

    def stream(self, request):
        yield from ()


class _EP:
    def __init__(self, name: str, value):
        self.name = name
        self._value = value

    def load(self):
        return self._value


class _EPS:
    def __init__(self, items):
        self._items = items

    def select(self, **kwargs):
        if kwargs.get("group") == "lm15.providers":
            return self._items
        return ()


class PluginDiscoveryTests(unittest.TestCase):
    def test_discover_and_load(self):
        eps = _EPS((_EP("mockp", _Adapter),))
        with patch("lm15.plugins.metadata.entry_points", return_value=eps):
            discovered = discover_provider_entry_points()
            self.assertEqual(len(discovered), 1)

            lm = UniversalLM()
            result = load_plugins(lm)
            self.assertEqual(result.loaded, ("mockp",))
            self.assertEqual(result.failed, ())
            self.assertIn("mockp", lm.adapters)

    def test_continue_on_error(self):
        class _Bad:
            def __call__(self):
                raise RuntimeError("boom")

        eps = _EPS((_EP("bad", _Bad()),))
        with patch("lm15.plugins.metadata.entry_points", return_value=eps):
            lm = UniversalLM()
            result = load_plugins(lm, continue_on_error=True)
            self.assertEqual(result.loaded, ())
            self.assertEqual(result.failed, ("bad",))


if __name__ == "__main__":
    unittest.main()
