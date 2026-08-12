from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from paicli.memory import LongTermMemory


class LongTermMemoryTest(unittest.TestCase):
    def test_scopes_deduplicates_and_persists_java_compatible_json(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            root.mkdir()
            storage = Path(directory) / "memory"
            memory = LongTermMemory(root, storage)
            first = memory.save("项目使用 Python 3.10", "project")
            duplicate = memory.save("项目使用 Python 3.10", "project")
            memory.save("默认使用中文", "global")

            self.assertEqual(first.id, duplicate.id)
            self.assertEqual(2, len(memory.list()))
            self.assertEqual(1, len(memory.search("Python")))
            raw = json.loads((storage / "long_term_memory.json").read_text(encoding="utf-8"))
            self.assertIn("tokenCount", raw[0])
            self.assertIn("metadata", raw[0])
            reloaded = LongTermMemory(root, storage)
            self.assertEqual(2, len(reloaded.list()))


if __name__ == "__main__":
    unittest.main()
