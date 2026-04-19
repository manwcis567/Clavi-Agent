"""Typed models for persisted user profiles and user-scoped memories."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


UserMemoryType = Literal[
    "preference",
    "communication_style",
    "goal",
    "constraint",
    "project_fact",
    "workflow_fact",
    "correction",
]
UserProfileFieldSource = Literal["explicit", "inferred", "hypothesis"]


class UserProfileFieldMeta(BaseModel):
    """用户画像字段级元数据。"""

    source: UserProfileFieldSource = "explicit"
    confidence: float = 1.0
    source_session_id: str | None = None
    source_run_id: str | None = None
    writer_type: str = "system"
    writer_id: str = ""
    updated_at: str = ""


class NormalizedUserProfile(BaseModel):
    """规范化后的用户画像字段集合。"""

    preferred_language: str | None = None
    response_length: str | None = None
    technical_depth: str | None = None
    recurring_projects: list[str] = Field(default_factory=list)
    dislikes_avoidances: list[str] = Field(default_factory=list)
    approval_risk_preference: str | None = None
    timezone: str | None = None
    locale: str | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)


class UserProfileRecord(BaseModel):
    """Persisted user-level profile summary and structured fields."""

    user_id: str
    profile: dict[str, Any] = Field(default_factory=dict)
    field_meta: dict[str, UserProfileFieldMeta] = Field(default_factory=dict)
    summary: str = ""
    writer_type: str = "system"
    writer_id: str = ""
    created_at: str
    updated_at: str

    def to_normalized_profile(self) -> NormalizedUserProfile:
        """返回规范化后的用户画像视图。"""
        known_fields = set(NormalizedUserProfile.model_fields) - {"extra_fields"}
        normalized_payload: dict[str, Any] = {}
        extra_fields: dict[str, Any] = {}
        for key, value in self.profile.items():
            if key in known_fields:
                normalized_payload[key] = value
            else:
                extra_fields[key] = value
        normalized_payload["extra_fields"] = extra_fields
        return NormalizedUserProfile(**normalized_payload)


class UserMemoryEntryRecord(BaseModel):
    """One persisted user-scoped long-term memory entry."""

    id: str
    user_id: str
    memory_type: UserMemoryType
    content: str
    summary: str = ""
    source_session_id: str | None = None
    source_run_id: str | None = None
    writer_type: str = "system"
    writer_id: str = ""
    confidence: float = 0.5
    created_at: str
    updated_at: str
    superseded_by: str | None = None
    is_deleted: bool = False
    deleted_at: str | None = None
    deleted_reason: str = ""


class MemoryAuditEventRecord(BaseModel):
    """Append-only audit event for memory/profile writes."""

    id: str
    user_id: str
    target_scope: str
    target_id: str
    action: str
    writer_type: str = "system"
    writer_id: str = ""
    session_id: str | None = None
    run_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class UserMemoryCompactionResult(BaseModel):
    """Summary for one duplicate-compaction pass."""

    user_id: str
    merged_group_count: int = 0
    canonical_entry_ids: list[str] = Field(default_factory=list)
    superseded_entry_ids: list[str] = Field(default_factory=list)
