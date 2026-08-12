from __future__ import annotations

from collections import deque
from typing import Sequence

from paicli.models import ChatResponse, Message, StreamListener, ToolSpec


class FakeLlmClient:
    provider_name = "fake"
    model_name = "fake-model"
    max_context_window = 128_000
    supports_tools = True
    supports_image_input = False

    def __init__(self, responses: Sequence[ChatResponse]) -> None:
        self.responses = deque(responses)
        self.requests: list[tuple[tuple[Message, ...], tuple[ToolSpec, ...]]] = []

    def chat(self, messages, tools=None, listener: StreamListener | None = None) -> ChatResponse:
        self.requests.append((tuple(messages), tuple(tools or ())))
        if not self.responses:
            raise AssertionError("FakeLlmClient 没有剩余响应")
        response = self.responses.popleft()
        if listener is not None:
            if response.reasoning_content:
                listener.on_reasoning(response.reasoning_content)
            if response.content:
                listener.on_content(response.content)
        return response
