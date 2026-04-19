"""LLM client wrapper that supports multiple providers.

This module provides a unified interface for different LLM providers
(Anthropic and OpenAI) through a single LLMClient class.
"""

import logging
from typing import Any, AsyncGenerator

from ..retry import RetryConfig
from ..schema import LLMProvider, LLMResponse, Message
from .base import LLMClientBase
from .openai_client import OpenAIClient

try:
    from .anthropic_client import AnthropicClient
except ModuleNotFoundError:
    AnthropicClient = None

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM Client wrapper supporting multiple providers.

    This class provides a unified interface for different LLM providers.
    It automatically instantiates the correct underlying client based on
    the provider parameter and appends the appropriate API endpoint suffix.

    Supported providers:
    - anthropic: Appends /anthropic to api_base
    - openai: Appends /v1 to api_base
    """

    @staticmethod
    def _resolve_api_base(api_base: str, provider: LLMProvider) -> str:
        """Normalize provider-specific base URLs without double-appending suffixes."""
        normalized = api_base.rstrip("/")

        if provider == LLMProvider.ANTHROPIC:
            return normalized if normalized.endswith("/anthropic") else f"{normalized}/anthropic"
        if provider == LLMProvider.OPENAI:
            return normalized if normalized.endswith("/v1") else f"{normalized}/v1"
        raise ValueError(f"Unsupported provider: {provider}")

    def __init__(
        self,
        api_key: str,
        provider: LLMProvider = LLMProvider.ANTHROPIC,
        api_base: str = "https://api.minimaxi.com",
        model: str = "MiniMax-M2",
        reasoning_enabled: bool = False,
        retry_config: RetryConfig | None = None,
    ):
        """Initialize LLM client with specified provider.

        Args:
            api_key: API key for authentication
            provider: LLM provider (anthropic or openai)
            api_base: Base URL for the API (default: https://api.minimaxi.com)
                     Will be automatically suffixed with /anthropic or /v1 based on provider
            model: Model name to use
            reasoning_enabled: Whether to enable provider reasoning mode when supported
            retry_config: Optional retry configuration
        """
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.reasoning_enabled = reasoning_enabled
        self.retry_config = retry_config or RetryConfig()

        full_api_base = self._resolve_api_base(api_base, provider)

        self.api_base = full_api_base

        # Instantiate the appropriate client
        self._client: LLMClientBase
        if provider == LLMProvider.ANTHROPIC:
            if AnthropicClient is None:
                raise ModuleNotFoundError(
                    "Anthropic provider requires the optional 'anthropic' package."
                )
            self._client = AnthropicClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                retry_config=retry_config,
            )
        elif provider == LLMProvider.OPENAI:
            self._client = OpenAIClient(
                api_key=api_key,
                api_base=full_api_base,
                model=model,
                reasoning_enabled=reasoning_enabled,
                retry_config=retry_config,
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info("Initialized LLM client with provider: %s, api_base: %s", provider, full_api_base)

    @property
    def retry_callback(self):
        """Get retry callback."""
        return self._client.retry_callback

    @retry_callback.setter
    def retry_callback(self, value):
        """Set retry callback."""
        self._client.retry_callback = value

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
    ) -> LLMResponse:
        """Generate response from LLM.

        Args:
            messages: List of conversation messages
            tools: Optional list of Tool objects or dicts

        Returns:
            LLMResponse containing the generated content
        """
        return await self._client.generate(messages, tools)

    async def generate_stream(
        self,
        messages: list[Message],
        tools: list | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Generate streamed events from LLM."""
        async for event in self._client.generate_stream(messages, tools):
            yield event
