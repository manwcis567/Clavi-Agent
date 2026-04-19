"""Domain models for reviewable skill-improvement proposals."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from .account_constants import ROOT_ACCOUNT_ID

SkillImprovementProposalStatus = Literal[
    "pending_review",
    "approved",
    "rejected",
    "applied",
]

SkillImprovementSignal = Literal[
    "repeated_user_corrections",
    "repeated_run_failures",
    "manual_successful_refinement",
]


class SkillImprovementProposalRecord(BaseModel):
    """Persisted proposal to improve one installed skill."""

    id: str
    account_id: str = ROOT_ACCOUNT_ID
    run_id: str
    session_id: str
    agent_template_id: str
    skill_name: str
    target_skill_path: str
    status: SkillImprovementProposalStatus = "pending_review"
    title: str
    summary: str = ""
    signal_types: list[SkillImprovementSignal] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)
    base_version: int = Field(default=1, ge=1)
    proposed_version: int = Field(default=2, ge=1)
    current_skill_markdown: str = ""
    proposed_skill_markdown: str = ""
    changelog_entry: str = ""
    review_notes: str = ""
    applied_skill_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    approved_at: str | None = None
    rejected_at: str | None = None
    applied_at: str | None = None
