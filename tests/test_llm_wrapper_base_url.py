"""Unit tests for LLM wrapper base URL normalization."""

from unittest.mock import patch

from clavi_agent.llm.llm_wrapper import LLMClient
from clavi_agent.schema import LLMProvider


def test_openai_provider_keeps_existing_v1_suffix():
    client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.OPENAI,
        api_base="https://openrouter.ai/api/v1",
        model="test-model",
    )

    assert client.api_base == "https://openrouter.ai/api/v1"


def test_openai_provider_appends_v1_when_missing():
    client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.OPENAI,
        api_base="https://api.minimax.io",
        model="test-model",
    )

    assert client.api_base == "https://api.minimax.io/v1"


def test_anthropic_provider_keeps_existing_suffix():
    client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.ANTHROPIC,
        api_base="https://api.minimaxi.com/anthropic",
        model="test-model",
    )

    assert client.api_base == "https://api.minimaxi.com/anthropic"


def test_anthropic_provider_appends_suffix_when_missing():
    client = LLMClient(
        api_key="test-key",
        provider=LLMProvider.ANTHROPIC,
        api_base="https://api.minimaxi.com",
        model="test-model",
    )

    assert client.api_base == "https://api.minimaxi.com/anthropic"


def test_openai_provider_forwards_reasoning_enabled():
    with patch("clavi_agent.llm.llm_wrapper.OpenAIClient") as mock_openai_client:
        LLMClient(
            api_key="test-key",
            provider=LLMProvider.OPENAI,
            api_base="https://openrouter.ai/api/v1",
            model="test-model",
            reasoning_enabled=True,
        )

    assert mock_openai_client.call_args.kwargs["reasoning_enabled"] is True

