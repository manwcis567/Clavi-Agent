"""Domain models for scheduled task configuration and execution history."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .account_constants import ROOT_ACCOUNT_ID

ScheduledTaskTriggerKind = Literal["schedule", "manual"]
ScheduledTaskExecutionStatus = Literal[
    "queued",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "cancelled",
    "timed_out",
    "dispatch_failed",
]


class ScheduledTaskRecord(BaseModel):
    """Persisted scheduled task definition."""

    id: str
    account_id: str = ROOT_ACCOUNT_ID
    name: str
    cron_expression: str
    timezone: str = "server_local"
    agent_id: str
    prompt: str
    integration_id: str | None = None
    target_chat_id: str = ""
    target_thread_id: str = ""
    reply_to_message_id: str = ""
    enabled: bool = True
    session_id: str | None = None
    next_run_at: str | None = None
    last_scheduled_for: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class ScheduledTaskExecutionRecord(BaseModel):
    """Persisted execution dispatch record for one scheduled task."""

    id: str
    task_id: str
    account_id: str = ROOT_ACCOUNT_ID
    trigger_kind: ScheduledTaskTriggerKind = "manual"
    scheduled_for: str | None = None
    run_id: str | None = None
    status: ScheduledTaskExecutionStatus = "queued"
    error_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
