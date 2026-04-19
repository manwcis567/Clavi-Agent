"""Typed run and run-step domain models."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .account_constants import ROOT_ACCOUNT_ID
from .agent_template_models import AgentTemplateSnapshot

RunStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "interrupted",
    "timed_out",
    "completed",
    "failed",
    "cancelled",
]

RunStepStatus = Literal["pending", "running", "completed", "failed", "skipped"]
ApprovalStatus = Literal["pending", "granted", "denied"]
ApprovalDecisionScope = Literal["once", "run", "template"]
ArtifactSource = Literal[
    "agent_generated",
    "user_uploaded",
    "agent_revised",
    "system_generated",
]
CheckpointTrigger = Literal[
    "llm_response",
    "tool_completed",
    "delegate_completed",
    "approval_wait",
    "run_finalizing",
]

RunStepType = Literal[
    "llm_call",
    "tool_call",
    "delegate",
    "delegate_review",
    "approval_wait",
    "checkpoint",
    "completion",
    "failure",
]

RUN_STATUS_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    "queued": {"running", "cancelled"},
    "running": {
        "waiting_approval",
        "interrupted",
        "timed_out",
        "completed",
        "failed",
        "cancelled",
    },
    "waiting_approval": {"running", "interrupted", "timed_out", "cancelled"},
    "interrupted": {"running", "cancelled"},
    "timed_out": set(),
    "completed": set(),
    "failed": set(),
    "cancelled": set(),
}

RUN_STEP_STATUS_TRANSITIONS: dict[RunStepStatus, set[RunStepStatus]] = {
    "pending": {"running", "skipped"},
    "running": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
    "skipped": set(),
}

TERMINAL_RUN_STATUSES: set[RunStatus] = {"timed_out", "completed", "failed", "cancelled"}
TERMINAL_RUN_STEP_STATUSES: set[RunStepStatus] = {"completed", "failed", "skipped"}


class TriggerMessageRef(BaseModel):
    """Reference to the user message that started a run."""

    message_id: str
    role: Literal["user"] = "user"


class RunRecord(BaseModel):
    """Run-scoped domain object for one durable execution lifecycle."""

    id: str
    session_id: str
    account_id: str = ROOT_ACCOUNT_ID
    agent_template_id: str
    agent_template_snapshot: AgentTemplateSnapshot
    status: RunStatus = "queued"
    goal: str
    trigger_message_ref: TriggerMessageRef | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    current_step_index: int = Field(default=0, ge=0)
    last_checkpoint_at: str | None = None
    error_summary: str = ""
    parent_run_id: str | None = None
    run_metadata: dict[str, Any] = Field(default_factory=dict)
    deliverable_manifest: "RunDeliverableManifest" = Field(
        default_factory=lambda: RunDeliverableManifest()
    )

    @property
    def is_terminal(self) -> bool:
        """Whether the run has reached a terminal state."""
        return self.status in TERMINAL_RUN_STATUSES

    def can_transition_to(self, next_status: RunStatus) -> bool:
        """Whether the run can move to the next status."""
        return next_status in RUN_STATUS_TRANSITIONS[self.status]

    def transition_to(
        self,
        next_status: RunStatus,
        *,
        changed_at: str | None = None,
        current_step_index: int | None = None,
        error_summary: str | None = None,
        last_checkpoint_at: str | None = None,
    ) -> "RunRecord":
        """Return a new run record after validating the status transition."""
        if not self.can_transition_to(next_status):
            raise ValueError(f"Invalid run status transition: {self.status} -> {next_status}")

        updates: dict[str, object] = {"status": next_status}

        if changed_at:
            if next_status == "running" and self.started_at is None:
                updates["started_at"] = changed_at
            if next_status in TERMINAL_RUN_STATUSES:
                updates["finished_at"] = changed_at

        if current_step_index is not None:
            updates["current_step_index"] = current_step_index
        if error_summary is not None:
            updates["error_summary"] = error_summary
        if last_checkpoint_at is not None:
            updates["last_checkpoint_at"] = last_checkpoint_at

        return self.model_copy(update=updates)


class RunStepRecord(BaseModel):
    """Traceable execution step that belongs to one run."""

    id: str
    run_id: str
    sequence: int = Field(ge=0)
    step_type: RunStepType
    status: RunStepStatus = "pending"
    title: str
    input_summary: str = ""
    output_summary: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error_summary: str = ""

    @property
    def is_terminal(self) -> bool:
        """Whether the run step has reached a terminal state."""
        return self.status in TERMINAL_RUN_STEP_STATUSES

    def can_transition_to(self, next_status: RunStepStatus) -> bool:
        """Whether the run step can move to the next status."""
        return next_status in RUN_STEP_STATUS_TRANSITIONS[self.status]

    def transition_to(
        self,
        next_status: RunStepStatus,
        *,
        changed_at: str | None = None,
        output_summary: str | None = None,
        error_summary: str | None = None,
    ) -> "RunStepRecord":
        """Return a new run step after validating the status transition."""
        if not self.can_transition_to(next_status):
            raise ValueError(
                f"Invalid run step status transition: {self.status} -> {next_status}"
            )

        updates: dict[str, object] = {"status": next_status}

        if changed_at:
            if next_status == "running" and self.started_at is None:
                updates["started_at"] = changed_at
            if next_status in TERMINAL_RUN_STEP_STATUSES:
                updates["finished_at"] = changed_at

        if output_summary is not None:
            updates["output_summary"] = output_summary
        if error_summary is not None:
            updates["error_summary"] = error_summary

        return self.model_copy(update=updates)


class CheckpointMessageSnapshot(BaseModel):
    """Compact message snapshot captured inside one run checkpoint."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[dict[str, Any]] = ""
    thinking: str = ""
    tool_call_id: str | None = None
    name: str | None = None
    tool_call_names: list[str] = Field(default_factory=list)


class PendingToolCallSnapshot(BaseModel):
    """Summary of a tool call that has not yet produced a final tool result."""

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    issued_in_step_sequence: int | None = Field(default=None, ge=0)


class SubAgentCheckpointSummary(BaseModel):
    """Minimal state summary for one delegated sub-agent at checkpoint time."""

    run_id: str | None = None
    agent_name: str = ""
    status: RunStatus
    current_step_index: int = Field(default=0, ge=0)
    summary: str = ""
    last_checkpoint_at: str | None = None


class SharedContextRef(BaseModel):
    """Reference to shared context that should be available during resume."""

    kind: str = "shared_context"
    uri: str
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunCheckpointPayload(BaseModel):
    """Structured checkpoint contents needed for recovery and diagnostics."""

    message_snapshot: list[CheckpointMessageSnapshot] = Field(default_factory=list)
    current_step_index: int = Field(default=0, ge=0)
    active_step_id: str | None = None
    incomplete_tool_calls: list[PendingToolCallSnapshot] = Field(default_factory=list)
    sub_agent_states: list[SubAgentCheckpointSummary] = Field(default_factory=list)
    shared_context_refs: list[SharedContextRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_payload(cls, data: Any) -> Any:
        """Keep compatibility with early checkpoints that stored arbitrary JSON."""
        if not isinstance(data, dict):
            return data

        known_keys = {
            "message_snapshot",
            "current_step_index",
            "active_step_id",
            "incomplete_tool_calls",
            "sub_agent_states",
            "shared_context_refs",
            "metadata",
        }
        extras = {key: value for key, value in data.items() if key not in known_keys}
        if not extras:
            return data

        normalized = {key: value for key, value in data.items() if key in known_keys}
        existing_metadata = normalized.get("metadata", {})
        if not isinstance(existing_metadata, dict):
            existing_metadata = {"value": existing_metadata}
        normalized["metadata"] = {**extras, **existing_metadata}
        return normalized


class RunCheckpointRecord(BaseModel):
    """Persisted checkpoint payload for one run."""

    id: str
    run_id: str
    step_sequence: int = Field(default=0, ge=0)
    trigger: CheckpointTrigger = "llm_response"
    payload: RunCheckpointPayload = Field(default_factory=RunCheckpointPayload)
    created_at: str


class RunDeliverableRef(BaseModel):
    artifact_id: str
    uri: str
    display_name: str = ""
    format: str = ""
    mime_type: str = ""
    role: str = "final_deliverable"
    is_primary: bool = False


class RunDeliverableManifest(BaseModel):
    primary_artifact_id: str | None = None
    items: list[RunDeliverableRef] = Field(default_factory=list)


class ArtifactRecord(BaseModel):
    """Materialized artifact produced by one run or one step."""

    id: str
    run_id: str
    step_id: str | None = None
    artifact_type: str
    uri: str
    display_name: str = ""
    role: str = "intermediate_file"
    format: str = ""
    mime_type: str = ""
    size_bytes: int | None = Field(default=None, ge=0)
    source: ArtifactSource = "agent_generated"
    is_final: bool = False
    preview_kind: str = "none"
    parent_artifact_id: str | None = None
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class ApprovalRequestRecord(BaseModel):
    """Approval request for a high-risk action."""

    id: str
    run_id: str
    account_id: str = ROOT_ACCOUNT_ID
    step_id: str | None = None
    tool_name: str
    risk_level: str
    status: ApprovalStatus = "pending"
    parameter_summary: str = ""
    impact_summary: str = ""
    requested_at: str
    resolved_at: str | None = None
    decision_notes: str = ""
    decision_scope: ApprovalDecisionScope | None = None


class TraceEventRecord(BaseModel):
    """Timeline event emitted during a run lifecycle."""

    id: str
    run_id: str
    account_id: str = ROOT_ACCOUNT_ID
    parent_run_id: str | None = None
    step_id: str | None = None
    sequence: int = Field(default=0, ge=0)
    event_type: str
    status: str = ""
    payload_summary: str = ""
    duration_ms: int | None = Field(default=None, ge=0)
    created_at: str
