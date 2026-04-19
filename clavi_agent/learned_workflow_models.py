"""Domain models for learned workflow candidates and promoted skills."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field

from .account_constants import ROOT_ACCOUNT_ID

WorkflowCandidateStatus = Literal[
    "pending_review",
    "approved",
    "rejected",
    "installed",
]

WorkflowCandidateSignal = Literal[
    "repeated_task_pattern",
    "successful_complex_run",
    "user_endorsed_solution",
]


class LearnedWorkflowCandidateRecord(BaseModel):
    """Persisted learned-workflow candidate derived from successful runs."""

    id: str
    account_id: str = ROOT_ACCOUNT_ID
    run_id: str
    session_id: str
    agent_template_id: str
    status: WorkflowCandidateStatus = "pending_review"
    title: str
    summary: str = ""
    description: str = ""
    signal_types: list[WorkflowCandidateSignal] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    step_titles: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    suggested_skill_name: str = ""
    generated_skill_markdown: str = ""
    review_notes: str = ""
    installed_agent_id: str | None = None
    installed_skill_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    approved_at: str | None = None
    rejected_at: str | None = None
    installed_at: str | None = None
