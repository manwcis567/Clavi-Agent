"""Anthropic LLM client implementation."""

import logging
from typing import Any, AsyncGenerator

import anthropic

from ..retry import RetryConfig, async_retry
from ..schema import (
    FunctionCall,
    LLMResponse,
    Message,
    ToolCall,
    render_message_content_for_model,
)
from .base import LLMClientBase

logger = logging.getLogger(__name__)


class AnthropicClient(LLMClientBase):
    """LLM client using Anthropic's protocol.

    This client uses the official Anthropic SDK and supports:
    - Extended thinking content
    - Tool calling
    - Retry logic
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/anthropic",
        model: str = "MiniMax-M2",
        retry_config: RetryConfig | None = None,
    ):
        """Initialize Anthropic client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax Anthropic endpoint)
            model: Model name to use (default: MiniMax-M2)
            retry_config: Optional retry configuration
        """
        super().__init__(api_key, api_base, model, retry_config)

        # Initialize Anthropic async client
        self.client = anthropic.AsyncAnthropic(
            base_url=api_base,
            api_key=api_key,
        )

    async def _make_api_request(
        self,
        system_message: str | None,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> anthropic.types.Message:
        """Execute API request (core method that can be retried).

        Args:
            system_message: Optional system message
            api_messages: List of messages in Anthropic format
            tools: Optional list of tools

        Returns:
            Anthropic Message response

        Raises:
            Exception: API call failed
        """
        params = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": api_messages,
        }

        if system_message:
            params["system"] = system_message

        if tools:
            params["tools"] = self._convert_tools(tools)

        # Use Anthropic SDK's async messages.create
        response = await self.client.messages.create(**params)
        return response

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format.

        Anthropic tool format:
        {
            "name": "tool_name",
            "description": "Tool description",
            "input_schema": {
                "type": "object",
                "properties": {...},
                "required": [...]
            }
        }

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in Anthropic dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                result.append(tool)
            elif hasattr(tool, "to_schema"):
                # Tool object with to_schema method
                result.append(tool.to_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to Anthropic format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
        """
        system_message = None
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = render_message_content_for_model(msg.content)
                continue

            # For user and assistant messages
            if msg.role in ["user", "assistant"]:
                # Handle assistant messages with thinking or tool calls
                if msg.role == "assistant" and (msg.thinking or msg.tool_calls):
                    # Build content blocks for assistant with thinking and/or tool calls
                    content_blocks = []

                    # Add thinking block if present
                    if msg.thinking:
                        content_blocks.append({"type": "thinking", "thinking": msg.thinking})

                    # Add text content if present
                    if msg.content:
                        content_blocks.append(
                            {
                                "type": "text",
                                "text": render_message_content_for_model(msg.content),
                            }
                        )

                    # Add tool use blocks
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            content_blocks.append(
                                {
                                    "type": "tool_use",
                                    "id": tool_call.id,
                                    "name": tool_call.function.name,
                                    "input": tool_call.function.arguments,
                                }
                            )

                    api_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    api_messages.append(
                        {
                            "role": msg.role,
                            "content": render_message_content_for_model(msg.content),
                        }
                    )

            # For tool result messages
            elif msg.role == "tool":
                # Anthropic uses user role with tool_result content blocks
                api_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": render_message_content_for_model(msg.content),
                            }
                        ],
                    }
                )

        return system_message, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for Anthropic API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        system_message, api_messages = self._convert_messages(messages)

        return {
            "system_message": system_message,
            "api_messages": api_messages,
            "tools": tools,
        }

    def _parse_response(self, response: anthropic.types.Message) -> LLMResponse:
        """Parse Anthropic response into LLMResponse.

        Args:
            response: Anthropic Message response

        Returns:
            LLMResponse object
        """
        # Extract text content, thinking, and tool calls
        text_content = ""
        thinking_content = ""
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_content += block.text
            elif block.type == "thinking":
                thinking_content += block.thinking
            elif block.type == "tool_use":
                # Parse Anthropic tool_use block
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function=FunctionCall(
                            name=block.name,
                            arguments=block.input,
                        ),
                    )
                )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=response.stop_reason or "stop",
        )

    @staticmethod
    def _safe_json_loads(value: str) -> dict[str, Any]:
        """Parse JSON object with fallback for partial/invalid data."""
        import json

        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate response from Anthropic LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            LLMResponse containing the generated content
        """
        # Prepare request
        request_params = self._prepare_request(messages, tools)

        # Make API request with retry logic
        if self.retry_config.enabled:
            # Apply retry logic
            retry_decorator = async_retry(config=self.retry_config, on_retry=self.retry_callback)
            api_call = retry_decorator(self._make_api_request)
            response = await api_call(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
            )
        else:
            # Don't use retry
            response = await self._make_api_request(
                request_params["system_message"],
                request_params["api_messages"],
                request_params["tools"],
            )

        # Parse and return response
        return self._parse_response(response)

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate streamed events from Anthropic-compatible API."""
        request_params = self._prepare_request(messages, tools)
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16384,
            "messages": request_params["api_messages"],
            "stream": True,
        }
        if request_params["system_message"]:
            params["system"] = request_params["system_message"]
        if request_params["tools"]:
            params["tools"] = self._convert_tools(request_params["tools"])

        content_chunks: list[str] = []
        thinking_chunks: list[str] = []
        tool_call_builders: dict[int, dict[str, str]] = {}
        finish_reason = "stop"

        stream = await self.client.messages.create(**params)
        async for event in stream:
            event_type = getattr(event, "type", "")

            if event_type == "content_block_start":
                block = getattr(event, "content_block", None)
                block_type = getattr(block, "type", "")
                block_index = int(getattr(event, "index", 0))

                if block_type == "text":
                    text = getattr(block, "text", "")
                    if isinstance(text, str) and text:
                        content_chunks.append(text)
                        yield {"type": "content_delta", "data": {"delta": text}}
                elif block_type == "thinking":
                    thinking = getattr(block, "thinking", "")
                    if isinstance(thinking, str) and thinking:
                        thinking_chunks.append(thinking)
                        yield {"type": "thinking_delta", "data": {"delta": thinking}}
                elif block_type == "tool_use":
                    builder = tool_call_builders.setdefault(
                        block_index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    block_id = getattr(block, "id", None)
                    if isinstance(block_id, str) and block_id:
                        builder["id"] = block_id
                    name = getattr(block, "name", None)
                    if isinstance(name, str) and name:
                        builder["name"] = name

                    initial_input = getattr(block, "input", None)
                    if isinstance(initial_input, dict) and initial_input:
                        import json

                        builder["arguments"] = json.dumps(initial_input, ensure_ascii=False)
                    elif isinstance(initial_input, str) and initial_input:
                        builder["arguments"] += initial_input
                continue

            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None)
                delta_type = getattr(delta, "type", "")
                block_index = int(getattr(event, "index", 0))

                if delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if isinstance(text, str) and text:
                        content_chunks.append(text)
                        yield {"type": "content_delta", "data": {"delta": text}}
                elif delta_type == "thinking_delta":
                    thinking = getattr(delta, "thinking", "")
                    if isinstance(thinking, str) and thinking:
                        thinking_chunks.append(thinking)
                        yield {"type": "thinking_delta", "data": {"delta": thinking}}
                elif delta_type == "input_json_delta":
                    partial_json = getattr(delta, "partial_json", "")
                    if isinstance(partial_json, str) and partial_json:
                        builder = tool_call_builders.setdefault(
                            block_index,
                            {"id": "", "name": "", "arguments": ""},
                        )
                        builder["arguments"] += partial_json
                continue

            if event_type == "message_delta":
                delta = getattr(event, "delta", None)
                stop_reason = getattr(delta, "stop_reason", None)
                if isinstance(stop_reason, str) and stop_reason:
                    finish_reason = stop_reason

        sorted_indexes = sorted(tool_call_builders.keys())
        tool_calls: list[ToolCall] = []
        for index in sorted_indexes:
            builder = tool_call_builders[index]
            tool_name = builder["name"] or "unknown_tool"
            arguments = self._safe_json_loads(builder["arguments"])
            tool_calls.append(
                ToolCall(
                    id=builder["id"] or f"tool_use_{index}",
                    type="function",
                    function=FunctionCall(
                        name=tool_name,
                        arguments=arguments,
                    ),
                )
            )

        final_response = LLMResponse(
            content="".join(content_chunks),
            thinking="".join(thinking_chunks) or None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason or "stop",
        )
        yield {"type": "final_response", "data": {"response": final_response}}
