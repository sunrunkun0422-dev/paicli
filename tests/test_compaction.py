from __future__ import annotations

import unittest

from paicli.compaction import ConversationCompactor
from paicli.models import ChatResponse, Message, ToolCall

from .fakes import FakeLlmClient


class ConversationCompactorTest(unittest.TestCase):
    def test_compacts_old_rounds_and_keeps_tool_protocol_tail(self) -> None:
        client = FakeLlmClient([ChatResponse(content="旧对话摘要")])
        compactor = ConversationCompactor(client, retain_recent_rounds=2)
        history = [Message.system("system")]
        for index in range(4):
            history.append(Message.user(f"q{index}"))
            if index == 2:
                history.append(Message.assistant(None, [ToolCall("c1", "read_file", "{}")]))
                history.append(Message.tool("c1", "result"))
            history.append(Message.assistant(f"a{index}"))

        outcome = compactor.compact_if_needed(history, trigger_tokens=1)

        self.assertTrue(outcome.compacted)
        self.assertEqual("system", history[0].role)
        self.assertIn("旧对话摘要", history[1].content or "")
        first_tail = next(index for index, message in enumerate(history) if message.content == "q2")
        self.assertEqual("assistant", history[first_tail + 1].role)
        self.assertEqual("tool", history[first_tail + 2].role)

    def test_manual_compaction_keeps_one_user_round(self) -> None:
        client = FakeLlmClient([ChatResponse(content="summary")])
        history = [Message.system("s")]
        for index in range(3):
            history.extend([Message.user(f"q{index}"), Message.assistant(f"a{index}")])
        outcome = ConversationCompactor(client).compact_now(history)
        self.assertTrue(outcome.compacted)
        self.assertEqual("q2", history[-2].content)

    def test_failure_is_atomic(self) -> None:
        client = FakeLlmClient([])
        history = [Message.system("s"), Message.user("q1"), Message.assistant("a1"), Message.user("q2")]
        before = list(history)
        outcome = ConversationCompactor(client).compact_now(history)
        self.assertFalse(outcome.compacted)
        self.assertIsNotNone(outcome.error)
        self.assertEqual(before, history)


if __name__ == "__main__":
    unittest.main()
