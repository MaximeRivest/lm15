from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lm15.types import DataSource, LMRequest, Message, Part


class ContractTests(unittest.TestCase):
    def test_datasource_validation(self):
        with self.assertRaises(ValueError):
            DataSource(type="url")
        with self.assertRaises(ValueError):
            DataSource(type="base64", media_type="image/png")

    def test_part_factory(self):
        p = Part.from_dict({"type": "tool_call", "id": "c1", "name": "search", "input": {"q": "x"}})
        self.assertEqual(p.type, "tool_call")
        self.assertEqual(p.input["q"], "x")

    def test_request_requires_messages(self):
        with self.assertRaises(ValueError):
            LMRequest(model="gpt-4.1-mini", messages=())

    def test_message_requires_parts(self):
        with self.assertRaises(ValueError):
            Message(role="user", parts=())


if __name__ == "__main__":
    unittest.main()
