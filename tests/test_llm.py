from __future__ import annotations

import unittest

from paicli.llm import OpenAICompatibleClient, PROVIDERS
from paicli.models import Message, NullStreamListener, ToolCall


class LlmClientTest(unittest.TestCase):
    def test_deepseek_serialization_keeps_reasoning_and_tool_protocol(self) -> None:
        client = OpenAICompatibleClient(PROVIDERS["deepseek"], "key")
        assistant = Message.assistant(
            None,
            [ToolCall("call-1", "read_file", '{"path":"a.py"}')],
            reasoning_content="先读取文件",
        )
        payload = client._message_payload(assistant)
        self.assertEqual("先读取文件", payload["reasoning_content"])
        self.assertEqual("call-1", payload["tool_calls"][0]["id"])
        tool = client._message_payload(Message.tool("call-1", "ok"))
        self.assertEqual("call-1", tool["tool_call_id"])

    def test_stream_assembles_fragmented_tool_calls_and_usage(self) -> None:
        client = OpenAICompatibleClient(PROVIDERS["glm"], "key")
        chunks = [
            b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_","function":{"name":"read_","arguments":"{\\\"path\\\":"}}]}}]}\n',
            b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"1","function":{"name":"file","arguments":"\\\"a.py\\\"}"}}]}}]}\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":3,"prompt_tokens_details":{"cached_tokens":4}}}\n',
            b'data: [DONE]\n',
        ]
        response = client._parse_stream(chunks, NullStreamListener())
        self.assertEqual("think", response.reasoning_content)
        self.assertEqual("call_1", response.tool_calls[0].id)
        self.assertEqual("read_file", response.tool_calls[0].name)
        self.assertEqual('{"path":"a.py"}', response.tool_calls[0].arguments)
        self.assertEqual((12, 3, 4), (response.input_tokens, response.output_tokens, response.cached_input_tokens))

    def test_base_url_is_normalized(self) -> None:
        client = OpenAICompatibleClient(PROVIDERS["step"], "key", api_url="https://example.test/v1/")
        self.assertEqual("https://example.test/v1/chat/completions", client.api_url)


if __name__ == "__main__":
    unittest.main()
