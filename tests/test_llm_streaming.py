"""Streaming behavior tests for LLM clients."""

from types import SimpleNamespace

import pytest

from clavi_agent.llm.anthropic_client import AnthropicClient
from clavi_agent.llm.openai_client import OpenAIClient
from clavi_agent.schema import Message


class AsyncListStream:
    """Simple async iterator over a fixed sequence."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __aiter__(self):
        self._index = 0
        return self

    async def __anext__(self):
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


@pytest.mark.asyncio
async def test_openai_generate_stream_emits_deltas_and_final_response():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://example.com/v1",
        model="test-model",
        reasoning_enabled=True,
    )

    stream = AsyncListStream(
        [
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="Hel", tool_calls=[]),
                        finish_reason=None,
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="lo",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="call-1",
                                    function=SimpleNamespace(name="weather", arguments='{"city":"'),
                                )
                            ],
                        ),
                        finish_reason=None,
                    )
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content="!",
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id=None,
                                    function=SimpleNamespace(name="", arguments='Shanghai"}'),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ]
            ),
        ]
    )

    async def fake_create(**kwargs):  # noqa: ANN003
        assert kwargs.get("stream") is True
        assert kwargs.get("extra_body") == {
            "reasoning_split": True,
            "reasoning": {"enabled": True},
        }
        return stream

    client.client.chat.completions.create = fake_create

    events = []
    async for event in client.generate_stream(
        messages=[Message(role="user", content="hello")],
        tools=None,
    ):
        events.append(event)

    assert [event["type"] for event in events[:3]] == [
        "content_delta",
        "content_delta",
        "content_delta",
    ]
    final_event = events[-1]
    assert final_event["type"] == "final_response"

    response = final_event["data"]["response"]
    assert response.content == "Hello!"
    assert response.finish_reason == "tool_calls"
    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "call-1"
    assert response.tool_calls[0].function.name == "weather"
    assert response.tool_calls[0].function.arguments == {"city": "Shanghai"}


@pytest.mark.asyncio
async def test_openai_generate_passes_reasoning_flag_in_extra_body():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://example.com/v1",
        model="test-model",
        reasoning_enabled=False,
    )

    async def fake_create(**kwargs):  # noqa: ANN003
        assert kwargs.get("extra_body") == {
            "reasoning_split": True,
            "reasoning": {"enabled": False},
        }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="Hello",
                        tool_calls=None,
                        reasoning_details=None,
                        model_extra={},
                    )
                )
            ]
        )

    client.client.chat.completions.create = fake_create

    response = await client.generate(messages=[Message(role="user", content="hello")])

    assert response.content == "Hello"


@pytest.mark.asyncio
async def test_anthropic_generate_stream_emits_deltas_and_final_response():
    client = AnthropicClient(
        api_key="test-key",
        api_base="https://example.com/anthropic",
        model="test-model",
    )

    stream = AsyncListStream(
        [
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="thinking", thinking="first-"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="thought"),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(type="text", text="Hi"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="text_delta", text=" there"),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=2,
                content_block=SimpleNamespace(type="tool_use", id="tool-1", name="lookup", input={}),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=2,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"query":"'),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=2,
                delta=SimpleNamespace(type="input_json_delta", partial_json='test"}'),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
            ),
        ]
    )

    async def fake_create(**kwargs):  # noqa: ANN003
        assert kwargs.get("stream") is True
        return stream

    client.client.messages.create = fake_create

    events = []
    async for event in client.generate_stream(
        messages=[Message(role="user", content="hello")],
        tools=None,
    ):
        events.append(event)

    assert [event["type"] for event in events[:-1]] == [
        "thinking_delta",
        "thinking_delta",
        "content_delta",
        "content_delta",
    ]
    final_event = events[-1]
    assert final_event["type"] == "final_response"

    response = final_event["data"]["response"]
    assert response.content == "Hi there"
    assert response.thinking == "first-thought"
    assert response.finish_reason == "tool_use"
    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "tool-1"
    assert response.tool_calls[0].function.name == "lookup"
    assert response.tool_calls[0].function.arguments == {"query": "test"}

