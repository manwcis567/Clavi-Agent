"""OpenAI LLM client implementation."""

import json
import logging
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI

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


class OpenAIClient(LLMClientBase):
    """LLM client using OpenAI's protocol.

    This client uses the official OpenAI SDK and supports:
    - Reasoning content (via reasoning_split=True)
    - Tool calling
    - Retry logic
    """

    def __init__(
        self,
        api_key: str,
        api_base: str = "https://api.minimaxi.com/v1",
        model: str = "MiniMax-M2",
        reasoning_enabled: bool = False,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize OpenAI client.

        Args:
            api_key: API key for authentication
            api_base: Base URL for the API (default: MiniMax OpenAI endpoint)
            model: Model name to use (default: MiniMax-M2)
            reasoning_enabled: Whether to enable reasoning mode in extra_body
            retry_config: Optional retry configuration
        """
        super().__init__(api_key, api_base, model, retry_config)
        self.reasoning_enabled = reasoning_enabled

        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
        )

    def _build_extra_body(self) -> dict[str, Any]:
        """Build provider-specific extra_body parameters."""
        return {
            "reasoning_split": True,
            "reasoning": {"enabled": self.reasoning_enabled},
        }

    async def _make_api_request(
        self,
        api_messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
    ) -> Any:
        """Execute API request (core method that can be retried).

        Args:
            api_messages: List of messages in OpenAI format
            tools: Optional list of tools

        Returns:
            OpenAI ChatCompletion message

        Raises:
            Exception: API call failed
        """
        params = {
            "model": self.model,
            "messages": api_messages,
            "extra_body": self._build_extra_body(),
        }

        if tools:
            params["tools"] = self._convert_tools(tools)

        # Use OpenAI SDK's chat.completions.create
        response = await self.client.chat.completions.create(**params)
        return response.choices[0].message

    def _convert_tools(self, tools: list[Any]) -> list[dict[str, Any]]:
        """Convert tools to OpenAI format.

        Args:
            tools: List of Tool objects or dicts

        Returns:
            List of tools in OpenAI dict format
        """
        result = []
        for tool in tools:
            if isinstance(tool, dict):
                # If already a dict, check if it's in OpenAI format
                if "type" in tool and tool["type"] == "function":
                    result.append(tool)
                else:
                    # Assume it's in Anthropic format, convert to OpenAI
                    result.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool["name"],
                                "description": tool["description"],
                                "parameters": tool["input_schema"],
                            },
                        }
                    )
            elif hasattr(tool, "to_openai_schema"):
                # Tool object with to_openai_schema method
                result.append(tool.to_openai_schema())
            else:
                raise TypeError(f"Unsupported tool type: {type(tool)}")
        return result

    def _convert_messages(self, messages: list[Message]) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert internal messages to OpenAI format.

        Args:
            messages: List of internal Message objects

        Returns:
            Tuple of (system_message, api_messages)
            Note: OpenAI includes system message in the messages array
        """
        api_messages = []

        for msg in messages:
            if msg.role == "system":
                # OpenAI includes system message in messages array
                api_messages.append(
                    {
                        "role": "system",
                        "content": render_message_content_for_model(msg.content),
                    }
                )
                continue

            # For user messages
            if msg.role == "user":
                api_messages.append(
                    {
                        "role": "user",
                        "content": render_message_content_for_model(msg.content),
                    }
                )

            # For assistant messages
            elif msg.role == "assistant":
                assistant_msg = {"role": "assistant"}

                # Add content if present
                if msg.content:
                    assistant_msg["content"] = render_message_content_for_model(msg.content)

                # Add tool calls if present
                if msg.tool_calls:
                    tool_calls_list = []
                    for tool_call in msg.tool_calls:
                        tool_calls_list.append(
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.function.name,
                                    "arguments": json.dumps(tool_call.function.arguments),
                                },
                            }
                        )
                    assistant_msg["tool_calls"] = tool_calls_list

                # IMPORTANT: Add reasoning_details if thinking is present
                # This is CRITICAL for Interleaved Thinking to work properly!
                # The complete response_message (including reasoning_details) must be
                # preserved in Message History and passed back to the model in the next turn.
                # This ensures the model's chain of thought is not interrupted.
                if msg.thinking:
                    assistant_msg["reasoning_details"] = [{"text": msg.thinking}]

                api_messages.append(assistant_msg)

            # For tool result messages
            elif msg.role == "tool":
                api_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": render_message_content_for_model(msg.content),
                    }
                )

        return None, api_messages

    def _prepare_request(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare the request for OpenAI API.

        Args:
            messages: List of conversation messages
            tools: Optional list of available tools

        Returns:
            Dictionary containing request parameters
        """
        _, api_messages = self._convert_messages(messages)

        return {
            "api_messages": api_messages,
            "tools": tools,
        }

    @staticmethod
    def _safe_json_loads(value: str) -> dict[str, Any]:
        """Parse JSON object with a resilient fallback."""
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _extract_reasoning_text(details: Any) -> str:
        """Extract reasoning text from OpenAI/MiniMax reasoning_details payloads."""
        if not details:
            return ""

        chunks: list[str] = []
        for detail in details:
            if isinstance(detail, dict):
                text = detail.get("text")
            else:
                text = getattr(detail, "text", None)
            if isinstance(text, str) and text:
                chunks.append(text)
        return "".join(chunks)

    @classmethod
    def _extract_chunk_thinking_delta(cls, choice: Any, delta: Any) -> str:
        """Best-effort extraction of thinking delta from streamed chunk payload."""
        candidate_sources = [
            getattr(delta, "reasoning_details", None),
            getattr(choice, "reasoning_details", None),
            getattr(delta, "reasoning", None),
            getattr(choice, "reasoning", None),
            getattr(delta, "reasoning_content", None),
            getattr(choice, "reasoning_content", None),
        ]
        for candidate in candidate_sources:
            if isinstance(candidate, str) and candidate:
                return candidate
            extracted = cls._extract_reasoning_text(candidate)
            if extracted:
                return extracted

        # Unknown fields can live in model_extra on SDK models.
        for obj in (delta, choice):
            extras = getattr(obj, "model_extra", None)
            if not isinstance(extras, dict):
                continue
            for key in ("reasoning", "reasoning_content"):
                extra_text = extras.get(key)
                if isinstance(extra_text, str) and extra_text:
                    return extra_text
            extra_details = extras.get("reasoning_details")
            extracted = cls._extract_reasoning_text(extra_details)
            if extracted:
                return extracted
        return ""

    def _parse_response(self, response: Any) -> LLMResponse:
        """Parse OpenAI response into LLMResponse.

        Args:
            response: OpenAI ChatCompletionMessage response

        Returns:
            LLMResponse object
        """
        # Extract text content
        text_content = response.content or ""

        # Extract thinking content from reasoning_details
        thinking_content = self._extract_reasoning_text(getattr(response, "reasoning_details", None))
        if not thinking_content:
            extras = getattr(response, "model_extra", None)
            if isinstance(extras, dict):
                maybe_text = extras.get("reasoning_content") or extras.get("reasoning")
                if isinstance(maybe_text, str):
                    thinking_content = maybe_text

        # Extract tool calls
        tool_calls = []
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # Parse arguments from JSON string
                arguments = self._safe_json_loads(tool_call.function.arguments)

                tool_calls.append(
                    ToolCall(
                        id=tool_call.id,
                        type="function",
                        function=FunctionCall(
                            name=tool_call.function.name,
                            arguments=arguments,
                        ),
                    )
                )

        return LLMResponse(
            content=text_content,
            thinking=thinking_content if thinking_content else None,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason="stop",  # OpenAI doesn't provide finish_reason in the message
        )

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate streamed events from OpenAI-compatible chat completions."""
        request_params = self._prepare_request(messages, tools)
        params = {
            "model": self.model,
            "messages": request_params["api_messages"],
            "extra_body": self._build_extra_body(),
            "stream": True,
        }
        if request_params["tools"]:
            params["tools"] = self._convert_tools(request_params["tools"])

        content_chunks: list[str] = []
        thinking_chunks: list[str] = []
        tool_call_builders: dict[int, dict[str, str]] = {}
        finish_reason = "stop"

        stream = await self.client.chat.completions.create(**params)
        async for chunk in stream:
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue

            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            if getattr(choice, "finish_reason", None):
                finish_reason = choice.finish_reason

            content_delta = getattr(delta, "content", None)
            if isinstance(content_delta, str) and content_delta:
                content_chunks.append(content_delta)
                yield {"type": "content_delta", "data": {"delta": content_delta}}

            thinking_delta = self._extract_chunk_thinking_delta(choice, delta)
            if thinking_delta:
                thinking_chunks.append(thinking_delta)
                yield {"type": "thinking_delta", "data": {"delta": thinking_delta}}

            tool_deltas = getattr(delta, "tool_calls", None) or []
            for tool_delta in tool_deltas:
                raw_index = getattr(tool_delta, "index", None)
                index = int(raw_index) if isinstance(raw_index, int) else 0
                builder = tool_call_builders.setdefault(
                    index,
                    {"id": "", "name": "", "arguments": ""},
                )

                tool_id = getattr(tool_delta, "id", None)
                if isinstance(tool_id, str) and tool_id:
                    builder["id"] = tool_id

                function_delta = getattr(tool_delta, "function", None)
                if function_delta is None:
                    continue
                name_delta = getattr(function_delta, "name", None)
                if isinstance(name_delta, str) and name_delta:
                    builder["name"] += name_delta
                args_delta = getattr(function_delta, "arguments", None)
                if isinstance(args_delta, str) and args_delta:
                    builder["arguments"] += args_delta

        sorted_indexes = sorted(tool_call_builders.keys())
        tool_calls: list[ToolCall] = []
        for index in sorted_indexes:
            builder = tool_call_builders[index]
            tool_name = builder["name"] or "unknown_tool"
            arguments = self._safe_json_loads(builder["arguments"])
            tool_calls.append(
                ToolCall(
                    id=builder["id"] or f"tool_call_{index}",
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

    async def generate(
        self,
        messages: list[Message],
        tools: list[Any] | None = None,
    ) -> LLMResponse:
        """Generate response from OpenAI LLM.

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
                request_params["api_messages"],
                request_params["tools"],
            )
        else:
            # Don't use retry
            response = await self._make_api_request(
                request_params["api_messages"],
                request_params["tools"],
            )

        # Parse and return response
        return self._parse_response(response)
