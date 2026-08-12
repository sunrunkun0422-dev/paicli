"""Dependency-free OpenAI-compatible provider clients."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings
from .models import (
    ChatResponse,
    LlmClient,
    Message,
    NullStreamListener,
    StreamListener,
    ToolCall,
    ToolSpec,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_env: tuple[str, ...]
    model_env: tuple[str, ...]
    default_model: str
    default_url: str
    base_url_env: tuple[str, ...] = ()
    max_context_window: int = 128_000
    supports_tools: bool = True
    supports_image_input: bool = True
    send_reasoning_history: bool = False


PROVIDERS: dict[str, ProviderSpec] = {
    "glm": ProviderSpec(
        "glm", ("GLM_API_KEY",), ("GLM_MODEL",), "glm-5.1",
        "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions",
        ("GLM_BASE_URL",), 200_000,
    ),
    "deepseek": ProviderSpec(
        "deepseek", ("DEEPSEEK_API_KEY",), ("DEEPSEEK_MODEL",), "deepseek-v4-flash",
        "https://api.deepseek.com/chat/completions",
        ("DEEPSEEK_BASE_URL",), 1_000_000, True, False, True,
    ),
    "step": ProviderSpec(
        "step", ("STEP_API_KEY",), ("STEP_MODEL",), "step-3.5-flash",
        "https://api.stepfun.com/v1/chat/completions",
        ("STEP_BASE_URL",), 256_000,
    ),
    "kimi": ProviderSpec(
        "kimi", ("KIMI_API_KEY", "MOONSHOT_API_KEY"), ("KIMI_MODEL", "MOONSHOT_MODEL"),
        "kimi-k2.6", "https://api.moonshot.ai/v1/chat/completions",
        ("KIMI_BASE_URL", "MOONSHOT_BASE_URL"), 256_000, True, True, True,
    ),
    "freellmapi": ProviderSpec(
        "freellmapi", ("FREELLMAPI_API_KEY",), ("FREELLMAPI_MODEL",), "auto",
        "http://localhost:5173/v1/chat/completions", ("FREELLMAPI_BASE_URL",), 128_000,
    ),
    "xfyun": ProviderSpec(
        "xfyun", ("XFYUN_MAAS_API_KEY",), ("XFYUN_MAAS_MODEL", "XFYUN_MODEL"),
        "Qwen3.6-35B-A3B", "https://maas-api.cn-huabei-1.xf-yun.com/v2/chat/completions",
        ("XFYUN_MAAS_BASE_URL", "XFYUN_BASE_URL"), 128_000, False,
    ),
    "agnes": ProviderSpec(
        "agnes", ("AGNES_API_KEY",), ("AGNES_MODEL",), "agnes-2.0-flash",
        "https://apihub.agnes-ai.com/v1/chat/completions", ("AGNES_BASE_URL",), 1_000_000,
    ),
}


ALIASES = {
    "zhipu": "glm",
    "moonshot": "kimi",
    "moonshot-ai": "kimi",
    "free-llm-api": "freellmapi",
    "freellm": "freellmapi",
    "maas": "xfyun",
    "iflytek": "xfyun",
    "agnes-ai": "agnes",
}


class OpenAICompatibleClient:
    def __init__(
        self,
        spec: ProviderSpec,
        api_key: str,
        model_name: str | None = None,
        api_url: str | None = None,
        timeout_seconds: int = 600,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(f"缺少 {spec.name} API Key")
        self.spec = spec
        self.api_key = api_key
        self.provider_name = spec.name
        self.model_name = model_name or spec.default_model
        self.api_url = _chat_url(api_url or spec.default_url)
        if spec.name == "glm" and self.model_name.lower().startswith("glm-5v") and not api_url:
            self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.timeout_seconds = timeout_seconds
        self.max_context_window = spec.max_context_window
        self.supports_tools = spec.supports_tools
        self.supports_image_input = spec.supports_image_input
        self.extra_headers = dict(extra_headers or {})

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec] | None = None,
        listener: StreamListener | None = None,
    ) -> ChatResponse:
        stream_listener = listener or NullStreamListener()
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [self._message_payload(message) for message in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools and self.supports_tools:
            payload["tools"] = [tool.to_openai() for tool in tools]
        if self.provider_name == "step":
            payload["reasoning_format"] = "deepseek-style"
            if "2603" in self.model_name:
                payload["reasoning_effort"] = "high"

        request = Request(
            self.api_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                **self.extra_headers,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                content_type = response.headers.get("Content-Type", "")
                if "text/event-stream" not in content_type.lower():
                    return self._parse_json_response(json.load(response), stream_listener)
                return self._parse_stream(response, stream_listener)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:4000]
            raise OSError(f"{self.provider_name} HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise OSError(f"无法连接 {self.provider_name}: {exc.reason}") from exc

    def _message_payload(self, message: Message) -> dict[str, Any]:
        item: dict[str, Any] = {"role": message.role}
        if message.content_parts and message.role == "user":
            parts: list[dict[str, Any]] = []
            for part in message.content_parts:
                if part.type == "text":
                    parts.append({"type": "text", "text": part.text or ""})
                elif self.supports_image_input:
                    url = part.image_url
                    if part.image_base64:
                        mime = part.mime_type or "image/png"
                        url = f"data:{mime};base64,{part.image_base64}"
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
                else:
                    parts.append({"type": "text", "text": "[当前 provider 不支持图片输入]"})
            item["content"] = parts
        else:
            item["content"] = message.content
        if message.role == "assistant" and message.tool_calls:
            item["tool_calls"] = [call.to_openai() for call in message.tool_calls]
        if message.role == "tool":
            item["tool_call_id"] = message.tool_call_id
        if self.spec.send_reasoning_history and message.reasoning_content:
            item["reasoning_content"] = message.reasoning_content
        return item

    def _parse_stream(self, response: Any, listener: StreamListener) -> ChatResponse:
        content: list[str] = []
        reasoning: list[str] = []
        tool_fragments: dict[int, dict[str, str]] = {}
        usage: dict[str, Any] = {}
        for raw in response:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or line.startswith(":") or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                log.debug("ignored invalid SSE line: %s", data[:200])
                continue
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
            content_delta = delta.get("content") or ""
            if reasoning_delta:
                reasoning.append(reasoning_delta)
                listener.on_reasoning(reasoning_delta)
            if content_delta:
                content.append(content_delta)
                listener.on_content(content_delta)
            for fragment in delta.get("tool_calls") or []:
                index = int(fragment.get("index", 0))
                current = tool_fragments.setdefault(index, {"id": "", "name": "", "arguments": ""})
                current["id"] += fragment.get("id") or ""
                function = fragment.get("function") or {}
                current["name"] += function.get("name") or ""
                current["arguments"] += function.get("arguments") or ""
        calls = tuple(
            ToolCall(value["id"] or f"call_{index}", value["name"], value["arguments"] or "{}")
            for index, value in sorted(tool_fragments.items())
        )
        prompt_details = usage.get("prompt_tokens_details") or {}
        return ChatResponse(
            "".join(content),
            "".join(reasoning),
            calls,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            int(prompt_details.get("cached_tokens") or usage.get("cached_input_tokens") or 0),
        )

    def _parse_json_response(self, data: Mapping[str, Any], listener: StreamListener) -> ChatResponse:
        choices = data.get("choices") or []
        message = choices[0].get("message", {}) if choices else {}
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        if reasoning:
            listener.on_reasoning(reasoning)
        if content:
            listener.on_content(content)
        calls = tuple(
            ToolCall(
                call.get("id") or f"call_{index}",
                (call.get("function") or {}).get("name") or "",
                (call.get("function") or {}).get("arguments") or "{}",
            )
            for index, call in enumerate(message.get("tool_calls") or [])
        )
        usage = data.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        return ChatResponse(
            content,
            reasoning,
            calls,
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            int(details.get("cached_tokens") or usage.get("cached_input_tokens") or 0),
        )


def create_client(settings: Settings, provider: str | None = None) -> OpenAICompatibleClient:
    selected = (provider or settings.provider or "").strip().lower()
    selected = ALIASES.get(selected, selected)
    if not selected:
        raise ValueError("未配置 LLM provider；请在 .env 中设置至少一个 API Key")
    if selected not in PROVIDERS:
        raise ValueError(f"不支持的 provider: {selected}")
    spec = PROVIDERS[selected]
    api_key = _first(settings.env, spec.key_env)
    model = _first(settings.env, spec.model_env) or spec.default_model
    configured_url = _first(settings.env, spec.base_url_env)
    headers: dict[str, str] = {}
    if selected == "xfyun":
        lora = settings.env.get("XFYUN_MAAS_LORA_ID") or settings.env.get("XFYUN_LORA_ID")
        if lora:
            headers["lora_id"] = lora
    return OpenAICompatibleClient(spec, api_key, model, configured_url, extra_headers=headers)


def _first(env: Mapping[str, str], keys: Sequence[str]) -> str:
    return next((env[key].strip() for key in keys if env.get(key, "").strip()), "")


def _chat_url(value: str) -> str:
    normalized = value.rstrip("/")
    return normalized if normalized.endswith("/chat/completions") else normalized + "/chat/completions"


__all__ = ["OpenAICompatibleClient", "ProviderSpec", "PROVIDERS", "create_client"]
