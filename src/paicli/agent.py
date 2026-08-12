"""The canonical Python ReAct agent loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Sequence

from .compaction import CompactionOutcome, ConversationCompactor
from .context import ContextProfile, estimate_message_tokens, estimate_tool_tokens
from .memory import LongTermMemory
from .models import LlmClient, Message, StreamListener, ToolCall
from .prompts import assemble_system_prompt
from .tools import ApprovalHandler, ToolRegistry


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactionResult:
    compacted: bool
    before_tokens: int
    after_tokens: int
    error: str | None = None


class Agent:
    def __init__(
        self,
        llm_client: LlmClient,
        project_root: Path | str,
        *,
        memory: LongTermMemory | None = None,
        tools: ToolRegistry | None = None,
        approval_mode: str = "suggest",
        approval_handler: ApprovalHandler | None = None,
        command_timeout_seconds: int = 60,
        max_iterations: int = 50,
        stagnation_window: int = 3,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.llm_client = llm_client
        self.memory = memory or LongTermMemory(self.project_root)
        self.tools = tools or ToolRegistry(
            self.project_root,
            self.memory,
            approval_mode,
            approval_handler,
            command_timeout_seconds,
        )
        self.approval_mode = approval_mode
        self.profile = ContextProfile(llm_client.max_context_window)
        self.compactor = ConversationCompactor(llm_client)
        self.max_iterations = max(1, max_iterations)
        self.stagnation_window = max(2, stagnation_window)
        self.history: list[Message] = [Message.system(self._system_prompt(""))]
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_cached_input_tokens = 0

    def run(self, user_input: str, listener: StreamListener | None = None) -> str:
        if not user_input or not user_input.strip():
            return ""
        self._prune_historical_images()
        memory_context = self.memory.context_for(user_input, self.profile.memory_context_tokens)
        self.history[0] = Message.system(self._system_prompt(memory_context))
        self.history.append(Message.user(user_input))
        recent_signatures: list[str] = []
        totals = [0, 0, 0]

        for iteration in range(1, self.max_iterations + 1):
            self._compact_if_needed()
            current_context = estimate_message_tokens(self.history) + self._tool_schema_tokens()
            if current_context >= self.profile.max_context_window:
                return (
                    f"当前上下文估算为 {current_context:,} tokens，已达到模型窗口 "
                    f"{self.profile.max_context_window:,}。请缩短本轮输入或执行 /clear 后重试。"
                )
            response = self.llm_client.chat(
                tuple(self.history),
                self.tools.definitions if self.llm_client.supports_tools else None,
                listener,
            )
            totals[0] += max(0, response.input_tokens)
            totals[1] += max(0, response.output_tokens)
            totals[2] += max(0, response.cached_input_tokens)

            if not response.has_tool_calls:
                answer = response.content or ""
                self.history.append(
                    Message.assistant(answer, reasoning_content=response.reasoning_content)
                )
                self.last_input_tokens, self.last_output_tokens, self.last_cached_input_tokens = totals
                return answer

            signature = self._tool_signature(response.tool_calls)
            recent_signatures.append(signature)
            recent_signatures = recent_signatures[-self.stagnation_window :]
            if len(recent_signatures) == self.stagnation_window and len(set(recent_signatures)) == 1:
                self.last_input_tokens, self.last_output_tokens, self.last_cached_input_tokens = totals
                return f"检测到连续 {self.stagnation_window} 轮相同工具调用，已停止以避免死循环。"

            self.history.append(
                Message.assistant(
                    response.content,
                    response.tool_calls,
                    response.reasoning_content,
                )
            )
            results = self.tools.execute_many(response.tool_calls)
            for result in results:
                self.history.append(Message.tool(result.call_id, result.content))
            log.debug("react iteration=%d tools=%s", iteration, [call.name for call in response.tool_calls])

        self.last_input_tokens, self.last_output_tokens, self.last_cached_input_tokens = totals
        return f"达到 Agent 硬轮数上限（{self.max_iterations}），任务已停止。"

    def clear(self) -> None:
        self.history[:] = [Message.system(self._system_prompt(""))]
        self.last_input_tokens = self.last_output_tokens = self.last_cached_input_tokens = 0

    def compact_now(self) -> CompactionResult:
        outcome = self.compactor.compact_now(self.history, self._tool_schema_tokens())
        return _compaction_result(outcome)

    def context_status(self) -> str:
        message_tokens = estimate_message_tokens(self.history)
        tool_tokens = self._tool_schema_tokens()
        return (
            self.profile.status(message_tokens + tool_tokens)
            + f"\n消息: {message_tokens:,} | 工具 schema: {tool_tokens:,}"
            + f"\n最近调用 in/out/cache: {self.last_input_tokens:,}/"
            f"{self.last_output_tokens:,}/{self.last_cached_input_tokens:,}"
        )

    def set_llm_client(self, llm_client: LlmClient) -> None:
        self.llm_client = llm_client
        self.profile = ContextProfile(llm_client.max_context_window)
        self.compactor.llm_client = llm_client
        self.history[0] = Message.system(self._system_prompt(""))

    def _compact_if_needed(self) -> CompactionResult:
        outcome = self.compactor.compact_if_needed(
            self.history,
            self.profile.compression_trigger_tokens,
            self._tool_schema_tokens(),
        )
        if outcome.compacted:
            log.info("conversation compacted: %d -> %d", outcome.before_tokens, outcome.after_tokens)
        return _compaction_result(outcome)

    def _tool_schema_tokens(self) -> int:
        return estimate_tool_tokens(self.tools.definitions) if self.llm_client.supports_tools else 0

    def _system_prompt(self, memory_context: str) -> str:
        return assemble_system_prompt(
            self.project_root,
            self.tools.definitions,
            memory_context,
            self.approval_mode,
        )

    def _prune_historical_images(self) -> None:
        self.history[:] = [message.without_images() for message in self.history]

    @staticmethod
    def _tool_signature(calls: Sequence[ToolCall]) -> str:
        return json.dumps(
            [(call.name, call.arguments) for call in calls],
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _compaction_result(outcome: CompactionOutcome) -> CompactionResult:
    return CompactionResult(
        outcome.compacted,
        outcome.before_tokens,
        outcome.after_tokens,
        outcome.error,
    )


__all__ = ["Agent", "CompactionResult"]
