"""Context-window policy and deterministic token estimation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Sequence

from .models import Message, ToolSpec


def estimate_text_tokens(text: str | None) -> int:
    """Match the Java implementation's inexpensive mixed Chinese/ASCII estimate."""

    if not text:
        return 0
    chinese = sum(1 for char in text if "\u4e00" < char < "\u9fff")
    other = len(text) - chinese
    return math.ceil(chinese / 1.5 + other / 4.0)


def estimate_message_tokens(messages: Sequence[Message]) -> int:
    total = 0
    for message in messages:
        if message.content_parts:
            for part in message.content_parts:
                if part.type == "text":
                    total += estimate_text_tokens(part.text)
                elif part.image_base64:
                    byte_count = len(part.image_base64) * 3 // 4
                    total += max(256, min(4096, byte_count // 768))
                elif part.is_image:
                    total += 1024
        else:
            total += estimate_text_tokens(message.content)
        for call in message.tool_calls:
            total += estimate_text_tokens(call.arguments)
    return total + len(messages) * 4


def estimate_tool_tokens(tools: Sequence[ToolSpec]) -> int:
    body = json.dumps([tool.to_openai() for tool in tools], ensure_ascii=False)
    return estimate_text_tokens(body)


@dataclass(frozen=True)
class ContextProfile:
    max_context_window: int

    MAX_SUMMARY_OUTPUT_RESERVE_TOKENS = 20_000
    AUTOCOMPACT_BUFFER_TOKENS = 13_000
    MIN_WINDOW = 8_000

    @property
    def compression_trigger_tokens(self) -> int:
        window = max(self.MIN_WINDOW, self.max_context_window)
        summary_reserve = min(
            self.MAX_SUMMARY_OUTPUT_RESERVE_TOKENS,
            max(1_000, window // 4),
        )
        buffer = min(self.AUTOCOMPACT_BUFFER_TOKENS, max(1_000, window // 8))
        return max(1_000, min(window - 1, window - summary_reserve - buffer))

    @property
    def compression_trigger_ratio(self) -> float:
        ratio = self.compression_trigger_tokens / self.max_context_window
        return max(0.50, min(0.99, ratio))

    @property
    def memory_context_tokens(self) -> int:
        return max(500, min(5_000, self.max_context_window // 200))

    def status(self, used_tokens: int) -> str:
        ratio = used_tokens / self.max_context_window if self.max_context_window else 0
        remaining = max(0, self.compression_trigger_tokens - used_tokens)
        return (
            f"上下文: {used_tokens:,}/{self.max_context_window:,} tokens "
            f"({ratio:.1%}) | 压缩阈值: {self.compression_trigger_tokens:,} "
            f"| 距压缩: {remaining:,}"
        )
