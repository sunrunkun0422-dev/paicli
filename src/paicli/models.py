"""Provider-neutral message and tool models.

The dataclasses intentionally mirror ``LlmClient`` in the Java implementation so
the migration can be verified field by field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str = "{}"

    def parsed_arguments(self) -> dict[str, Any]:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"工具参数不是合法 JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("工具参数必须是 JSON object")
        return value

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(frozen=True)
class ContentPart:
    type: str
    text: str | None = None
    image_base64: str | None = None
    image_url: str | None = None
    mime_type: str | None = None

    @classmethod
    def text_part(cls, text: str) -> "ContentPart":
        return cls("text", text=text)

    @classmethod
    def image_data(cls, data: str, mime_type: str = "image/png") -> "ContentPart":
        return cls("image_base64", image_base64=data, mime_type=mime_type)

    @classmethod
    def remote_image(cls, url: str) -> "ContentPart":
        return cls("image_url", image_url=url)

    @property
    def is_image(self) -> bool:
        return self.type in {"image_base64", "image_url"}


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    content_parts: tuple[ContentPart, ...] = ()

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(
        cls,
        content: str | None,
        tool_calls: Sequence[ToolCall] = (),
        reasoning_content: str | None = None,
    ) -> "Message":
        return cls("assistant", content, reasoning_content, tuple(tool_calls))

    @classmethod
    def tool(cls, tool_call_id: str, content: str) -> "Message":
        return cls("tool", content, tool_call_id=tool_call_id)

    def without_images(self) -> "Message":
        count = sum(1 for part in self.content_parts if part.is_image)
        if count == 0:
            return self
        parts = [part for part in self.content_parts if not part.is_image]
        notice = f"[历史图片附件已省略 {count} 张；如需重新查看，请重新附加图片。]"
        parts.append(ContentPart.text_part(notice))
        text = "\n\n".join(part.text or "" for part in parts if part.text)
        return Message(
            self.role,
            text,
            self.reasoning_content,
            self.tool_calls,
            self.tool_call_id,
            tuple(parts),
        )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }


@dataclass(frozen=True)
class ChatResponse:
    content: str = ""
    reasoning_content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class StreamListener(Protocol):
    def on_reasoning(self, text: str) -> None: ...

    def on_content(self, text: str) -> None: ...


class NullStreamListener:
    def on_reasoning(self, text: str) -> None:
        del text

    def on_content(self, text: str) -> None:
        del text


class LlmClient(Protocol):
    provider_name: str
    model_name: str
    max_context_window: int
    supports_tools: bool
    supports_image_input: bool

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        listener: StreamListener | None = None,
    ) -> ChatResponse: ...


ToolExecutor = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class RegisteredTool:
    spec: ToolSpec
    executor: ToolExecutor = field(repr=False, compare=False)
    dangerous: bool = False
