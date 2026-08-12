from __future__ import annotations

import unittest

from paicli.context import ContextProfile, estimate_message_tokens, estimate_text_tokens
from paicli.models import Message, ToolCall


class ContextProfileTest(unittest.TestCase):
    def test_trigger_formula_matches_reference_windows(self) -> None:
        self.assertEqual(5_000, ContextProfile(8_000).compression_trigger_tokens)
        self.assertEqual(95_000, ContextProfile(128_000).compression_trigger_tokens)
        self.assertEqual(167_000, ContextProfile(200_000).compression_trigger_tokens)
        self.assertEqual(967_000, ContextProfile(1_000_000).compression_trigger_tokens)

    def test_message_estimate_includes_tool_arguments_and_overhead(self) -> None:
        messages = [
            Message.user("abcd"),
            Message.assistant(None, [ToolCall("c1", "read_file", '{"path":"README.md"}')]),
        ]
        expected = estimate_text_tokens("abcd") + estimate_text_tokens('{"path":"README.md"}') + 8
        self.assertEqual(expected, estimate_message_tokens(messages))


if __name__ == "__main__":
    unittest.main()
