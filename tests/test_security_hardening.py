from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.capabilities import resolve_provider
from lm15.errors import TransportError, UnsupportedModelError
from lm15.factory import build_default
from lm15.sse import parse_sse


class SecurityHardeningTests(unittest.TestCase):
    def test_unknown_model_does_not_silently_fallback(self):
        with self.assertRaises(UnsupportedModelError):
            resolve_provider("totally-unknown-model")

    def test_build_default_does_not_auto_load_plugins_by_default(self):
        with patch("lm15.factory.load_plugins", side_effect=AssertionError("should not load plugins")):
            build_default(use_pycurl=False)

    def test_env_import_is_allowlisted(self):
        old_openai = os.environ.pop("OPENAI_API_KEY", None)
        old_bad = os.environ.pop("SHOULD_NOT_IMPORT", None)
        try:
            with tempfile.NamedTemporaryFile("w", delete=False) as f:
                f.write("OPENAI_API_KEY=sk-test\n")
                f.write("SHOULD_NOT_IMPORT=yes\n")
                env_path = f.name

            build_default(use_pycurl=False, env=env_path)
            self.assertEqual(os.environ.get("OPENAI_API_KEY"), "sk-test")
            self.assertIsNone(os.environ.get("SHOULD_NOT_IMPORT"))
        finally:
            if old_openai is not None:
                os.environ["OPENAI_API_KEY"] = old_openai
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            if old_bad is not None:
                os.environ["SHOULD_NOT_IMPORT"] = old_bad
            else:
                os.environ.pop("SHOULD_NOT_IMPORT", None)
            try:
                os.unlink(env_path)
            except Exception:
                pass

    def test_sse_parser_enforces_event_size_limit(self):
        lines = iter([b"data: " + (b"a" * 100) + b"\n", b"\n"])
        with self.assertRaises(TransportError):
            list(parse_sse(lines, max_event_bytes=32))


if __name__ == "__main__":
    unittest.main()
