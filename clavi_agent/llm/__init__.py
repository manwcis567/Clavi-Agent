"""LLM clients package supporting both Anthropic and OpenAI protocols."""

from .base import LLMClientBase
from .llm_wrapper import LLMClient
from .openai_client import OpenAIClient

try:
    from .anthropic_client import AnthropicClient
except ModuleNotFoundError:
    AnthropicClient = None

__all__ = ["LLMClientBase", "AnthropicClient", "OpenAIClient", "LLMClient"]

