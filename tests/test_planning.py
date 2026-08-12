from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from paicli.agent import Agent
from paicli.memory import LongTermMemory
from paicli.models import ChatResponse
from paicli.planning import ExecutionPlan, PlanExecutor, Task

from .fakes import FakeLlmClient


class ExecutionPlanTest(unittest.TestCase):
    def test_builds_dependency_batches(self) -> None:
        plan = ExecutionPlan("goal", (
            Task("a", "A"),
            Task("b", "B"),
            Task("c", "C", ("a", "b")),
        ))
        self.assertEqual([["a", "b"], ["c"]], [[task.id for task in batch] for batch in plan.batches()])

    def test_rejects_cycle(self) -> None:
        plan = ExecutionPlan("goal", (Task("a", "A", ("b",)), Task("b", "B", ("a",))))
        with self.assertRaisesRegex(ValueError, "环"):
            plan.batches()

    def test_plan_executor_runs_dependency_chain_and_synthesizes(self) -> None:
        planner_json = '{"tasks":[{"id":"a","description":"first","dependencies":[]},{"id":"b","description":"second","dependencies":["a"]}]}'
        client = FakeLlmClient([
            ChatResponse(content=planner_json),
            ChatResponse(content="result-a"),
            ChatResponse(content="result-b"),
            ChatResponse(content="final-result"),
        ])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            agent = Agent(client, root, memory=LongTermMemory(root, root / ".memory"), approval_mode="never")
            answer = PlanExecutor(agent).run("goal")
        self.assertEqual("final-result", answer)
        dependency_prompt = client.requests[2][0][1].content or ""
        self.assertIn("result-a", dependency_prompt)


if __name__ == "__main__":
    unittest.main()
