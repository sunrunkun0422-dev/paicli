from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paicli.agent import Agent
from paicli.memory import LongTermMemory
from paicli.models import ChatResponse, ToolCall

from .fakes import FakeLlmClient


class AgentTest(unittest.TestCase):
    def test_react_tool_loop(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hello.txt").write_text("hello", encoding="utf-8")
            client = FakeLlmClient([
                ChatResponse(tool_calls=(ToolCall("c1", "read_file", '{"path":"hello.txt"}'),)),
                ChatResponse(content="文件内容是 hello", input_tokens=10, output_tokens=5),
            ])
            memory = LongTermMemory(root, root / ".memory")
            agent = Agent(client, root, memory=memory, approval_mode="never")

            answer = agent.run("读取 hello.txt")

            self.assertEqual("文件内容是 hello", answer)
            self.assertEqual(["system", "user", "assistant", "tool", "assistant"], [m.role for m in agent.history])
            self.assertIn("hello", agent.history[3].content or "")
            self.assertEqual(10, agent.last_input_tokens)

    def test_stagnation_guard(self) -> None:
        repeated = ChatResponse(tool_calls=(ToolCall("c", "list_dir", '{"path":"."}'),))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeLlmClient([repeated, repeated, repeated])
            agent = Agent(client, root, memory=LongTermMemory(root, root / ".memory"), approval_mode="never")
            answer = agent.run("循环")
            self.assertIn("避免死循环", answer)

    def test_refuses_request_that_still_exceeds_window_after_compaction(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            client = FakeLlmClient([])
            client.max_context_window = 8_000
            agent = Agent(client, root, memory=LongTermMemory(root, root / ".memory"), approval_mode="never")
            answer = agent.run("x" * 40_000)
            self.assertIn("已达到模型窗口", answer)
            self.assertEqual(0, len(client.requests))


if __name__ == "__main__":
    unittest.main()
