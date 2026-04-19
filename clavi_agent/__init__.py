"""Clavi Agent - Minimal single agent with basic tools and MCP support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .agent import Agent
    from .llm import LLMClient
    from .schema import FunctionCall, LLMProvider, LLMResponse, Message, ToolCall

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "LLMClient",
    "LLMProvider",
    "Message",
    "LLMResponse",
    "ToolCall",
    "FunctionCall",
]


def __getattr__(name: str) -> Any:
    if name == "Agent":
        from .agent import Agent

        return Agent
    if name == "LLMClient":
        from .llm import LLMClient

        return LLMClient
    if name in {"LLMProvider", "Message", "LLMResponse", "ToolCall", "FunctionCall"}:
        from .schema import FunctionCall, LLMProvider, LLMResponse, Message, ToolCall

        return {
            "LLMProvider": LLMProvider,
            "Message": Message,
            "LLMResponse": LLMResponse,
            "ToolCall": ToolCall,
            "FunctionCall": FunctionCall,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

