"""Compact Plan-and-Execute support built on the same Python Agent core."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import re
from typing import Any, Callable, Mapping

from .agent import Agent
from .models import LlmClient, Message


@dataclass(frozen=True)
class Task:
    id: str
    description: str
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    goal: str
    tasks: tuple[Task, ...]

    def batches(self) -> list[list[Task]]:
        by_id = {task.id: task for task in self.tasks}
        if len(by_id) != len(self.tasks):
            raise ValueError("计划包含重复 task id")
        for task in self.tasks:
            missing = set(task.dependencies) - by_id.keys()
            if missing:
                raise ValueError(f"任务 {task.id} 引用了未知依赖: {sorted(missing)}")
            if task.id in task.dependencies:
                raise ValueError(f"任务 {task.id} 不能依赖自身")

        remaining = dict(by_id)
        completed: set[str] = set()
        result: list[list[Task]] = []
        while remaining:
            ready = [task for task in remaining.values() if set(task.dependencies) <= completed]
            if not ready:
                raise ValueError("计划依赖存在环")
            ready.sort(key=lambda task: task.id)
            result.append(ready)
            for task in ready:
                completed.add(task.id)
                del remaining[task.id]
        return result


class Planner:
    def __init__(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client

    def create(self, goal: str) -> ExecutionPlan:
        response = self.llm_client.chat(
            (
                Message.system(
                    "你是任务规划器。只输出 JSON："
                    '{"tasks":[{"id":"task_1","description":"...","dependencies":[]}]}。'
                    "任务要可独立执行，依赖必须引用前面的 id，任务数量控制在 1-8。"
                ),
                Message.user(goal),
            ),
            None,
        )
        try:
            value = _parse_json_object(response.content)
            tasks = tuple(_parse_task(item, index) for index, item in enumerate(value.get("tasks") or []))
            if not tasks:
                raise ValueError("规划结果没有任务")
            if len(tasks) > 8:
                tasks = tasks[:8]
            plan = ExecutionPlan(goal, tasks)
            plan.batches()
            return plan
        except (ValueError, TypeError, json.JSONDecodeError):
            return ExecutionPlan(goal, (Task("task_1", goal),))


class PlanExecutor:
    def __init__(
        self,
        base_agent: Agent,
        agent_factory: Callable[[], Agent] | None = None,
        max_parallel: int = 4,
    ) -> None:
        self.base_agent = base_agent
        self.planner = Planner(base_agent.llm_client)
        self.agent_factory = agent_factory or self._new_agent
        self.max_parallel = max(1, min(4, max_parallel))

    def run(self, goal: str) -> str:
        plan = self.planner.create(goal)
        results: dict[str, str] = {}
        for batch in plan.batches():
            with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(batch))) as pool:
                futures = [pool.submit(self._execute_task, goal, task, dict(results)) for task in batch]
                for task, future in zip(batch, futures):
                    results[task.id] = future.result()
        return self._synthesize(goal, plan, results)

    def _execute_task(self, goal: str, task: Task, completed: Mapping[str, str]) -> str:
        dependencies = "\n\n".join(
            f"[{dependency}]\n{completed.get(dependency, '')}" for dependency in task.dependencies
        )
        prompt = (
            f"总体目标：{goal}\n\n当前任务 [{task.id}]：{task.description}"
            + (f"\n\n依赖任务结果：\n{dependencies}" if dependencies else "")
            + "\n\n请完成当前任务；需要查看或修改项目时使用真实工具。"
        )
        return self.agent_factory().run(prompt)

    def _synthesize(self, goal: str, plan: ExecutionPlan, results: Mapping[str, str]) -> str:
        body = "\n\n".join(
            f"## {task.id}: {task.description}\n{results.get(task.id, '')}" for task in plan.tasks
        )
        response = self.base_agent.llm_client.chat(
            (
                Message.system("你是结果整合者。基于各任务真实结果，用中文给出最终答复；不要虚构未完成事项。"),
                Message.user(f"总体目标：{goal}\n\n任务结果：\n{body}"),
            ),
            None,
        )
        return response.content or body

    def _new_agent(self) -> Agent:
        return Agent(
            self.base_agent.llm_client,
            self.base_agent.project_root,
            memory=self.base_agent.memory,
            tools=self.base_agent.tools,
            approval_mode=self.base_agent.approval_mode,
            max_iterations=self.base_agent.max_iterations,
            stagnation_window=self.base_agent.stagnation_window,
        )


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("规划结果不包含 JSON object")
    value = json.loads(cleaned[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("规划结果必须是 JSON object")
    return value


def _parse_task(value: Any, index: int) -> Task:
    if not isinstance(value, dict):
        raise ValueError("task 必须是 JSON object")
    task_id = str(value.get("id") or f"task_{index + 1}").strip()
    description = str(value.get("description") or "").strip()
    if not description:
        raise ValueError(f"任务 {task_id} 缺少 description")
    raw_dependencies = value.get("dependencies") or []
    if not isinstance(raw_dependencies, list):
        raise ValueError(f"任务 {task_id} dependencies 必须是数组")
    return Task(task_id, description, tuple(str(item) for item in raw_dependencies))


__all__ = ["ExecutionPlan", "PlanExecutor", "Planner", "Task"]
