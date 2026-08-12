from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paicli.memory import LongTermMemory
from paicli.models import ToolCall
from paicli.tools import ToolRegistry


class ToolRegistryTest(unittest.TestCase):
    def test_file_tools_and_path_guard(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            memory = LongTermMemory(root, Path(directory) / "memory")
            registry = ToolRegistry(root, memory, approval_mode="auto")

            written = registry.execute(ToolCall("1", "write_file", '{"path":"a.txt","content":"hello"}'))
            self.assertFalse(written.error)
            read = registry.execute(ToolCall("2", "read_file", '{"path":"a.txt"}'))
            self.assertIn("hello", read.content)
            escaped = registry.execute(ToolCall("3", "read_file", '{"path":"../secret"}'))
            self.assertTrue(escaped.error)
            self.assertIn("策略拒绝", escaped.content)

    def test_parallel_results_preserve_call_order(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.txt").write_text("a", encoding="utf-8")
            (root / "b.txt").write_text("b", encoding="utf-8")
            registry = ToolRegistry(root, LongTermMemory(root, root / ".memory"), approval_mode="never")
            calls = [
                ToolCall("a", "read_file", '{"path":"a.txt"}'),
                ToolCall("b", "read_file", '{"path":"b.txt"}'),
            ]
            self.assertEqual(["a", "b"], [result.call_id for result in registry.execute_many(calls)])

    def test_command_policy_runs_before_shell(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            registry = ToolRegistry(root, LongTermMemory(root, root / ".memory"), approval_mode="auto")
            result = registry.execute(ToolCall("1", "execute_command", '{"command":"sudo true"}'))
            self.assertTrue(result.error)
            self.assertIn("禁止 sudo", result.content)


if __name__ == "__main__":
    unittest.main()
