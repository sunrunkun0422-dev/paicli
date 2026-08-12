"""Protocol-safe rolling compaction for the messages actually sent to an LLM."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import MutableSequence, Sequence

from .context import estimate_message_tokens
from .models import LlmClient, Message


log = logging.getLogger(__name__)


SUMMARY_PROMPT = """请把下面的对话历史压缩成简明摘要，保留：
1. 用户提出的关键诉求与目标
2. Agent 已完成的关键操作、文件改动、命令和核心结果
3. 已经达成的共识、约束或结论
4. 仍未解决的问题和下一步

不要逐条复述，不要保留无关闲聊。输出简洁中文摘要，不要加元描述。

=== 待压缩的对话 ===
{history}
=== 待压缩的对话（结束）===
"""


@dataclass(frozen=True)
class CompactionOutcome:
    compacted: bool
    before_tokens: int
    after_tokens: int
    error: str | None = None


class ConversationCompactor:
    def __init__(
        self,
        llm_client: LlmClient,
        retain_recent_rounds: int = 3,
        max_summary_input_chars: int = 60_000,
    ) -> None:
        self.llm_client = llm_client
        self.retain_recent_rounds = max(1, retain_recent_rounds)
        self.max_summary_input_chars = max(1_000, max_summary_input_chars)

    def compact_if_needed(
        self,
        history: MutableSequence[Message],
        trigger_tokens: int,
        extra_tokens: int = 0,
    ) -> CompactionOutcome:
        before = estimate_message_tokens(history) + max(0, extra_tokens)
        if before < trigger_tokens:
            return CompactionOutcome(False, before, before)
        return self._compact(history, self.retain_recent_rounds, before, extra_tokens)

    def compact_now(
        self,
        history: MutableSequence[Message],
        extra_tokens: int = 0,
    ) -> CompactionOutcome:
        before = estimate_message_tokens(history) + max(0, extra_tokens)
        return self._compact(history, 1, before, extra_tokens)

    def _compact(
        self,
        history: MutableSequence[Message],
        retain_rounds: int,
        before: int,
        extra_tokens: int,
    ) -> CompactionOutcome:
        if not history:
            return CompactionOutcome(False, before, before)
        system_end = 1 if history[0].role == "system" else 0
        user_indices = [
            index
            for index, message in enumerate(history[system_end:], start=system_end)
            if message.role == "user"
        ]
        if len(user_indices) <= retain_rounds:
            return CompactionOutcome(False, before, before)
        split = user_indices[-retain_rounds]
        old_messages = list(history[system_end:split])
        if not old_messages:
            return CompactionOutcome(False, before, before)

        try:
            summary = self.summarize(old_messages).strip()
        except Exception as exc:  # Keep the original history atomically on failure.
            log.info("conversation compaction failed: %s", exc)
            return CompactionOutcome(False, before, before, str(exc))
        if not summary:
            return CompactionOutcome(False, before, before, "摘要为空")

        rebuilt = list(history[:system_end])
        rebuilt.append(Message.user("[已压缩的历史对话摘要]\n" + summary))
        rebuilt.append(Message.assistant("好的，我已了解之前的上下文，请继续。"))
        rebuilt.extend(history[split:])
        history[:] = rebuilt
        after = estimate_message_tokens(history) + max(0, extra_tokens)
        return CompactionOutcome(True, before, after)

    def summarize(self, messages: Sequence[Message]) -> str:
        body_parts: list[str] = []
        used = 0
        for message in messages:
            lines = [f"{message.role.upper()}: {message.content or ''}"]
            for call in message.tool_calls:
                lines.append(f"  TOOL_CALL {call.name}: {call.arguments}")
            block = "\n".join(lines) + "\n\n"
            remaining = self.max_summary_input_chars - used
            if remaining <= 0:
                break
            body_parts.append(block[:remaining])
            used += min(len(block), remaining)
        if used >= self.max_summary_input_chars:
            body_parts.append("\n...(超长内容已截断)\n")
        request = (
            Message.system("你是对话摘要助手，只输出摘要本身。"),
            Message.user(SUMMARY_PROMPT.format(history="".join(body_parts))),
        )
        return self.llm_client.chat(request, None).content
