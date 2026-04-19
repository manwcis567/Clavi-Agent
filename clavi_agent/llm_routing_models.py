"""Shared LLM profile routing models and merge helpers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


LLMProviderLiteral = Literal["anthropic", "openai"]


class LLMProfile(BaseModel):
    """One fully resolved runtime LLM profile."""

    provider: LLMProviderLiteral = "anthropic"
    api_base: str = "https://api.minimax.io"
    model: str = "MiniMax-M2"
    reasoning_enabled: bool = False


class LLMProfileOverride(BaseModel):
    """One partial role-specific LLM profile override."""

    provider: LLMProviderLiteral | None = None
    api_base: str | None = None
    model: str | None = None
    reasoning_enabled: bool | None = None

    def has_overrides(self) -> bool:
        """Return whether any role-specific field is explicitly set."""
        return bool(self.model_dump(exclude_none=True))


class LLMRoutingPolicy(BaseModel):
    """Per-template or per-account role routing overrides."""

    planner_api_config_id: str | None = None
    worker_api_config_id: str | None = None
    planner_profile: LLMProfileOverride | None = None
    worker_profile: LLMProfileOverride | None = None

    def has_overrides(self) -> bool:
        """Return whether either planner or worker profile is overridden."""
        return bool(
            self.planner_api_config_id
            or self.worker_api_config_id
            or (self.planner_profile is not None and self.planner_profile.has_overrides())
            or (self.worker_profile is not None and self.worker_profile.has_overrides())
        )


def merge_llm_profile(
    base_profile: LLMProfile,
    *overrides: LLMProfileOverride | dict | None,
) -> LLMProfile:
    """Merge multiple partial overrides onto one resolved base profile."""
    payload = base_profile.model_dump(mode="python")
    for override in overrides:
        if override is None:
            continue
        if isinstance(override, LLMProfileOverride):
            override_payload = override.model_dump(exclude_none=True)
        else:
            override_payload = LLMProfileOverride.model_validate(override).model_dump(
                exclude_none=True
            )
        payload.update(override_payload)
    return LLMProfile.model_validate(payload)
