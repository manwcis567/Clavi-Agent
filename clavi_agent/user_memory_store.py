"""Persistent storage for user profiles and long-term memory backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .sqlite_schema import configure_connection, ensure_agent_db_schema, utc_now_iso
from .user_memory_models import (
    MemoryAuditEventRecord,
    UserProfileFieldMeta,
    UserProfileFieldSource,
    UserMemoryCompactionResult,
    UserMemoryEntryRecord,
    UserMemoryType,
    UserProfileRecord,
)


_UNSET = object()
_ALLOWED_MEMORY_TYPES = {
    "preference",
    "communication_style",
    "goal",
    "constraint",
    "project_fact",
    "workflow_fact",
    "correction",
}
_PROFILE_FIELD_META_KEY = "_field_meta"
_PROFILE_SOURCE_PRIORITY = {
    "hypothesis": 0,
    "inferred": 1,
    "explicit": 2,
}
_PROFILE_FIELD_ALIASES = {
    "preferred_response_length": "response_length",
    "language": "preferred_language",
    "dislikes": "dislikes_avoidances",
    "avoidances": "dislikes_avoidances",
    "approval_preference": "approval_risk_preference",
    "risk_preference": "approval_risk_preference",
}
_PROFILE_LIST_FIELDS = {"recurring_projects", "dislikes_avoidances"}


class UserMemoryStore:
    """SQLite repository for user-scoped long-term memory."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return configure_connection(sqlite3.connect(self.db_path))

    def _initialize(self) -> None:
        with self._connect() as conn:
            ensure_agent_db_schema(conn)

    @staticmethod
    def _normalize_user_id(user_id: str) -> str:
        normalized = str(user_id or "").strip()
        if not normalized:
            raise ValueError("user_id is required.")
        return normalized

    @staticmethod
    def _normalize_text(value: str | None, *, field_name: str, required: bool) -> str:
        normalized = str(value or "").strip()
        if required and not normalized:
            raise ValueError(f"{field_name} is required.")
        return normalized

    @classmethod
    def _normalize_memory_type(cls, memory_type: str) -> UserMemoryType:
        normalized = str(memory_type or "").strip()
        if normalized not in _ALLOWED_MEMORY_TYPES:
            raise ValueError(f"Unsupported memory_type: {memory_type}")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _normalize_confidence(confidence: float | int) -> float:
        normalized = float(confidence)
        if normalized < 0 or normalized > 1:
            raise ValueError("confidence must be between 0 and 1.")
        return normalized

    @staticmethod
    def _normalize_profile_source(
        profile_source: str | None,
    ) -> UserProfileFieldSource:
        normalized = str(profile_source or "explicit").strip().lower()
        if normalized not in _PROFILE_SOURCE_PRIORITY:
            raise ValueError(f"Unsupported profile_source: {profile_source}")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _normalize_profile_field_name(field_name: str) -> str:
        normalized = str(field_name or "").strip()
        if not normalized:
            raise ValueError("profile field name is required.")
        return _PROFILE_FIELD_ALIASES.get(normalized, normalized)

    @classmethod
    def _normalize_profile_field_value(cls, field_name: str, value: Any) -> Any:
        normalized_field = cls._normalize_profile_field_name(field_name)
        if normalized_field in _PROFILE_LIST_FIELDS:
            if value is None:
                return []
            if isinstance(value, str):
                item = value.strip()
                return [item] if item else []
            if isinstance(value, (list, tuple, set)):
                normalized_items: list[str] = []
                for item in value:
                    text = str(item or "").strip()
                    if text and text not in normalized_items:
                        normalized_items.append(text)
                return normalized_items
            raise ValueError(f"{normalized_field} must be a list of strings.")
        return value

    @staticmethod
    def _extract_profile_meta(
        payload: dict[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        if not isinstance(payload, dict):
            return {}
        raw_meta = payload.get(_PROFILE_FIELD_META_KEY)
        if not isinstance(raw_meta, dict):
            return {}
        return {
            str(key): dict(value)
            for key, value in raw_meta.items()
            if isinstance(value, dict)
        }

    @classmethod
    def _split_profile_payload(
        cls,
        payload: dict[str, Any] | None,
        *,
        fallback_updated_at: str,
        fallback_writer_type: str,
        fallback_writer_id: str,
    ) -> tuple[dict[str, Any], dict[str, UserProfileFieldMeta]]:
        if not isinstance(payload, dict):
            return {}, {}

        raw_meta = cls._extract_profile_meta(payload)
        profile: dict[str, Any] = {}
        field_meta: dict[str, UserProfileFieldMeta] = {}
        for key, value in payload.items():
            if key == _PROFILE_FIELD_META_KEY:
                continue
            normalized_key = cls._normalize_profile_field_name(key)
            normalized_value = cls._normalize_profile_field_value(normalized_key, value)
            profile[normalized_key] = normalized_value
            meta_payload = raw_meta.get(key) or raw_meta.get(normalized_key) or {}
            field_meta[normalized_key] = UserProfileFieldMeta(
                source=cls._normalize_profile_source(meta_payload.get("source")),
                confidence=cls._normalize_confidence(
                    meta_payload.get("confidence", 1.0)
                ),
                source_session_id=(
                    str(meta_payload.get("source_session_id") or "").strip() or None
                ),
                source_run_id=(
                    str(meta_payload.get("source_run_id") or "").strip() or None
                ),
                writer_type=cls._normalize_writer_type(
                    meta_payload.get("writer_type", fallback_writer_type)
                ),
                writer_id=cls._normalize_writer_id(
                    meta_payload.get("writer_id", fallback_writer_id)
                ),
                updated_at=str(meta_payload.get("updated_at") or fallback_updated_at),
            )
        return profile, field_meta

    @staticmethod
    def _serialize_profile_payload(
        profile: dict[str, Any],
        field_meta: dict[str, UserProfileFieldMeta],
    ) -> dict[str, Any]:
        payload = dict(profile)
        if field_meta:
            payload[_PROFILE_FIELD_META_KEY] = {
                key: value.model_dump(mode="python")
                for key, value in field_meta.items()
            }
        return payload

    @staticmethod
    def _should_replace_profile_field(
        existing_meta: UserProfileFieldMeta | None,
        incoming_meta: UserProfileFieldMeta,
    ) -> bool:
        if existing_meta is None:
            return True
        existing_priority = _PROFILE_SOURCE_PRIORITY.get(existing_meta.source, 0)
        incoming_priority = _PROFILE_SOURCE_PRIORITY.get(incoming_meta.source, 0)
        if incoming_priority != existing_priority:
            return incoming_priority > existing_priority
        return str(incoming_meta.updated_at) >= str(existing_meta.updated_at)

    @staticmethod
    def _json_loads(raw: str | None, fallback: Any) -> Any:
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def generate_memory_entry_id() -> str:
        """Generate a new memory entry identifier."""
        return str(uuid.uuid4())

    @staticmethod
    def generate_audit_event_id() -> str:
        """Generate a new audit-event identifier."""
        return str(uuid.uuid4())

    @staticmethod
    def _normalize_writer_type(writer_type: str | None) -> str:
        normalized = str(writer_type or "").strip()
        return normalized or "system"

    @staticmethod
    def _normalize_writer_id(writer_id: str | None) -> str:
        normalized = str(writer_id or "").strip()
        return normalized or "user_memory_store"

    def _ensure_user_exists(self, conn: sqlite3.Connection, user_id: str) -> None:
        row = conn.execute(
            "SELECT 1 FROM accounts WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"User not found: {user_id}")

    def _profile_from_row(self, row: sqlite3.Row) -> UserProfileRecord:
        profile, field_meta = self._split_profile_payload(
            self._json_loads(row["profile_json"], {}),
            fallback_updated_at=row["updated_at"],
            fallback_writer_type=row["writer_type"],
            fallback_writer_id=row["writer_id"],
        )
        return UserProfileRecord(
            user_id=row["user_id"],
            profile=profile,
            field_meta=field_meta,
            summary=row["summary"],
            writer_type=row["writer_type"],
            writer_id=row["writer_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _entry_from_row(row: sqlite3.Row) -> UserMemoryEntryRecord:
        return UserMemoryEntryRecord(
            id=row["id"],
            user_id=row["user_id"],
            memory_type=row["memory_type"],
            content=row["content"],
            summary=row["summary"],
            source_session_id=row["source_session_id"],
            source_run_id=row["source_run_id"],
            writer_type=row["writer_type"],
            writer_id=row["writer_id"],
            confidence=float(row["confidence"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            superseded_by=row["superseded_by"],
            is_deleted=bool(row["is_deleted"]),
            deleted_at=row["deleted_at"],
            deleted_reason=row["deleted_reason"] or "",
        )

    def _audit_from_row(self, row: sqlite3.Row) -> MemoryAuditEventRecord:
        return MemoryAuditEventRecord(
            id=row["id"],
            user_id=row["user_id"],
            target_scope=row["target_scope"],
            target_id=row["target_id"],
            action=row["action"],
            writer_type=row["writer_type"],
            writer_id=row["writer_id"],
            session_id=row["session_id"],
            run_id=row["run_id"],
            payload=self._json_loads(row["payload_json"], {}),
            created_at=row["created_at"],
        )

    @staticmethod
    def _normalized_content_key(content: str) -> str:
        return " ".join(str(content or "").split()).casefold()

    def _append_audit_event(
        self,
        conn: sqlite3.Connection,
        *,
        user_id: str,
        target_scope: str,
        target_id: str,
        action: str,
        writer_type: str | None = None,
        writer_id: str | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
        payload: dict[str, Any] | None = None,
        created_at: str | None = None,
    ) -> None:
        normalized_created_at = created_at or utc_now_iso()
        conn.execute(
            """
            INSERT INTO memory_audit_events (
                id, user_id, target_scope, target_id, action,
                writer_type, writer_id, session_id, run_id,
                payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.generate_audit_event_id(),
                self._normalize_user_id(user_id),
                self._normalize_text(target_scope, field_name="target_scope", required=True),
                self._normalize_text(target_id, field_name="target_id", required=True),
                self._normalize_text(action, field_name="action", required=True),
                self._normalize_writer_type(writer_type),
                self._normalize_writer_id(writer_id),
                self._normalize_text(session_id, field_name="session_id", required=False) or None,
                self._normalize_text(run_id, field_name="run_id", required=False) or None,
                json.dumps(payload or {}, ensure_ascii=False),
                normalized_created_at,
            ),
        )

    def get_user_profile_record(self, user_id: str) -> UserProfileRecord | None:
        """Get one user profile by user id."""
        normalized_user_id = self._normalize_user_id(user_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (normalized_user_id,),
            ).fetchone()
        if row is None:
            return None
        return self._profile_from_row(row)

    def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Get one user profile as a dict payload."""
        record = self.get_user_profile_record(user_id)
        if record is None:
            return None
        return record.model_dump(mode="python")

    def inspect_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """返回包含字段级元数据的用户画像检查视图。"""
        record = self.get_user_profile_record(user_id)
        if record is None:
            return None
        payload = record.model_dump(mode="python")
        payload["normalized_profile"] = record.to_normalized_profile().model_dump(mode="python")
        return payload

    def upsert_user_profile(
        self,
        user_id: str,
        *,
        profile: dict[str, Any] | None = None,
        summary: str | None = None,
        merge: bool = True,
        profile_source: str | None = None,
        profile_confidence: float | int = 1.0,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        writer_type: str | None = None,
        writer_id: str | None = None,
    ) -> dict[str, Any]:
        """Create or update one structured user profile."""
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_profile_source = self._normalize_profile_source(profile_source)
        normalized_profile_confidence = self._normalize_confidence(profile_confidence)
        incoming_profile: dict[str, Any] = {}
        incoming_field_meta: dict[str, UserProfileFieldMeta] = {}
        normalized_session_id = self._normalize_text(
            source_session_id,
            field_name="source_session_id",
            required=False,
        ) or None
        normalized_run_id = self._normalize_text(
            source_run_id,
            field_name="source_run_id",
            required=False,
        ) or None
        normalized_writer_type = self._normalize_writer_type(writer_type)
        normalized_writer_id = self._normalize_writer_id(writer_id)
        now = utc_now_iso()

        for raw_key, raw_value in dict(profile or {}).items():
            normalized_key = self._normalize_profile_field_name(raw_key)
            incoming_profile[normalized_key] = self._normalize_profile_field_value(
                normalized_key,
                raw_value,
            )
            incoming_field_meta[normalized_key] = UserProfileFieldMeta(
                source=normalized_profile_source,
                confidence=normalized_profile_confidence,
                source_session_id=normalized_session_id,
                source_run_id=normalized_run_id,
                writer_type=normalized_writer_type,
                writer_id=normalized_writer_id,
                updated_at=now,
            )

        with self._connect() as conn:
            self._ensure_user_exists(conn, normalized_user_id)
            existing = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (normalized_user_id,),
            ).fetchone()
            if existing is not None:
                existing_profile, existing_field_meta = self._split_profile_payload(
                    self._json_loads(existing["profile_json"], {}),
                    fallback_updated_at=existing["updated_at"],
                    fallback_writer_type=existing["writer_type"],
                    fallback_writer_id=existing["writer_id"],
                )
            else:
                existing_profile, existing_field_meta = {}, {}

            next_profile = dict(existing_profile) if merge else {}
            next_field_meta = dict(existing_field_meta) if merge else {}
            for key, value in incoming_profile.items():
                if self._should_replace_profile_field(
                    next_field_meta.get(key),
                    incoming_field_meta[key],
                ):
                    next_profile[key] = value
                    next_field_meta[key] = incoming_field_meta[key]

            next_summary = (
                existing["summary"]
                if existing is not None and summary is None
                else self._normalize_text(summary, field_name="summary", required=False)
            )
            created_at = existing["created_at"] if existing is not None else now

            conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, profile_json, summary, writer_type, writer_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    summary = excluded.summary,
                    writer_type = excluded.writer_type,
                    writer_id = excluded.writer_id,
                    updated_at = excluded.updated_at
                """,
                (
                    normalized_user_id,
                    json.dumps(
                        self._serialize_profile_payload(next_profile, next_field_meta),
                        ensure_ascii=False,
                    ),
                    next_summary,
                    normalized_writer_type,
                    normalized_writer_id,
                    created_at,
                    now,
                ),
            )
            self._append_audit_event(
                conn,
                user_id=normalized_user_id,
                target_scope="user_profile",
                target_id=normalized_user_id,
                action="profile_upsert" if existing is not None else "profile_create",
                writer_type=normalized_writer_type,
                writer_id=normalized_writer_id,
                session_id=normalized_session_id,
                run_id=normalized_run_id,
                payload={
                    "profile_keys": sorted(next_profile),
                    "summary": next_summary,
                    "merge": bool(merge),
                    "profile_source": normalized_profile_source,
                },
                created_at=now,
            )

        profile_record = self.get_user_profile_record(normalized_user_id)
        if profile_record is None:
            raise RuntimeError(f"Failed to upsert user profile: {normalized_user_id}")
        return profile_record.model_dump(mode="python")

    def update_user_profile(
        self,
        user_id: str,
        *,
        profile_updates: dict[str, Any] | None = None,
        remove_fields: list[str] | None = None,
        summary: str | object = _UNSET,
        profile_source: str | None = None,
        profile_confidence: float | int = 1.0,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        writer_type: str | None = None,
        writer_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Update one structured user profile, optionally removing specific fields."""
        normalized_user_id = self._normalize_user_id(user_id)
        updates = dict(profile_updates or {})
        removals = [
            self._normalize_profile_field_name(field_name)
            for field_name in (remove_fields or [])
            if str(field_name or "").strip()
        ]
        if not updates and not removals and summary is _UNSET:
            raise ValueError("At least one profile update, removal, or summary change is required.")

        normalized_profile_source = self._normalize_profile_source(profile_source)
        normalized_profile_confidence = self._normalize_confidence(profile_confidence)
        normalized_session_id = self._normalize_text(
            source_session_id,
            field_name="source_session_id",
            required=False,
        ) or None
        normalized_run_id = self._normalize_text(
            source_run_id,
            field_name="source_run_id",
            required=False,
        ) or None
        normalized_writer_type = self._normalize_writer_type(writer_type)
        normalized_writer_id = self._normalize_writer_id(writer_id)
        now = utc_now_iso()

        incoming_profile: dict[str, Any] = {}
        incoming_field_meta: dict[str, UserProfileFieldMeta] = {}
        for raw_key, raw_value in updates.items():
            normalized_key = self._normalize_profile_field_name(raw_key)
            incoming_profile[normalized_key] = self._normalize_profile_field_value(
                normalized_key,
                raw_value,
            )
            incoming_field_meta[normalized_key] = UserProfileFieldMeta(
                source=normalized_profile_source,
                confidence=normalized_profile_confidence,
                source_session_id=normalized_session_id,
                source_run_id=normalized_run_id,
                writer_type=normalized_writer_type,
                writer_id=normalized_writer_id,
                updated_at=now,
            )

        with self._connect() as conn:
            self._ensure_user_exists(conn, normalized_user_id)
            existing = conn.execute(
                "SELECT * FROM user_profiles WHERE user_id = ?",
                (normalized_user_id,),
            ).fetchone()
            if existing is None:
                if not incoming_profile:
                    return None
                return self.upsert_user_profile(
                    normalized_user_id,
                    profile=incoming_profile,
                    summary=(
                        None
                        if summary is _UNSET
                        else self._normalize_text(summary, field_name="summary", required=False)
                    ),
                    merge=True,
                    profile_source=normalized_profile_source,
                    profile_confidence=normalized_profile_confidence,
                    source_session_id=normalized_session_id,
                    source_run_id=normalized_run_id,
                    writer_type=normalized_writer_type,
                    writer_id=normalized_writer_id,
                )

            existing_profile, existing_field_meta = self._split_profile_payload(
                self._json_loads(existing["profile_json"], {}),
                fallback_updated_at=existing["updated_at"],
                fallback_writer_type=existing["writer_type"],
                fallback_writer_id=existing["writer_id"],
            )
            next_profile = dict(existing_profile)
            next_field_meta = dict(existing_field_meta)

            removed_fields: list[str] = []
            for field_name in removals:
                if field_name in next_profile:
                    next_profile.pop(field_name, None)
                    next_field_meta.pop(field_name, None)
                    removed_fields.append(field_name)

            updated_fields: list[str] = []
            for key, value in incoming_profile.items():
                if self._should_replace_profile_field(
                    next_field_meta.get(key),
                    incoming_field_meta[key],
                ):
                    next_profile[key] = value
                    next_field_meta[key] = incoming_field_meta[key]
                    updated_fields.append(key)

            next_summary = (
                existing["summary"]
                if summary is _UNSET
                else self._normalize_text(summary, field_name="summary", required=False)
            )

            conn.execute(
                """
                UPDATE user_profiles
                SET profile_json = ?, summary = ?, writer_type = ?, writer_id = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    json.dumps(
                        self._serialize_profile_payload(next_profile, next_field_meta),
                        ensure_ascii=False,
                    ),
                    next_summary,
                    normalized_writer_type,
                    normalized_writer_id,
                    now,
                    normalized_user_id,
                ),
            )
            self._append_audit_event(
                conn,
                user_id=normalized_user_id,
                target_scope="user_profile",
                target_id=normalized_user_id,
                action="profile_update",
                writer_type=normalized_writer_type,
                writer_id=normalized_writer_id,
                session_id=normalized_session_id,
                run_id=normalized_run_id,
                payload={
                    "updated_fields": sorted(updated_fields),
                    "removed_fields": sorted(removed_fields),
                    "summary": next_summary,
                    "profile_source": normalized_profile_source,
                },
                created_at=now,
            )

        updated = self.get_user_profile_record(normalized_user_id)
        if updated is None:
            raise RuntimeError(f"Failed to update user profile: {normalized_user_id}")
        return updated.model_dump(mode="python")

    def create_memory_entry(
        self,
        *,
        user_id: str,
        memory_type: str,
        content: str,
        summary: str = "",
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        confidence: float = 0.5,
        entry_id: str | None = None,
        writer_type: str | None = None,
        writer_id: str | None = None,
    ) -> dict[str, Any]:
        """Create one user memory entry."""
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_memory_type = self._normalize_memory_type(memory_type)
        normalized_content = self._normalize_text(
            content,
            field_name="content",
            required=True,
        )
        normalized_summary = self._normalize_text(
            summary,
            field_name="summary",
            required=False,
        )
        normalized_session_id = self._normalize_text(
            source_session_id,
            field_name="source_session_id",
            required=False,
        ) or None
        normalized_run_id = self._normalize_text(
            source_run_id,
            field_name="source_run_id",
            required=False,
        ) or None
        normalized_confidence = self._normalize_confidence(confidence)
        normalized_writer_type = self._normalize_writer_type(writer_type)
        normalized_writer_id = self._normalize_writer_id(writer_id)
        resolved_entry_id = self._normalize_text(
            entry_id or self.generate_memory_entry_id(),
            field_name="entry_id",
            required=True,
        )
        now = utc_now_iso()

        with self._connect() as conn:
            self._ensure_user_exists(conn, normalized_user_id)
            conn.execute(
                """
                INSERT INTO user_memory_entries (
                    id, user_id, memory_type, content, summary,
                    source_session_id, source_run_id, writer_type, writer_id, confidence,
                    created_at, updated_at, superseded_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    resolved_entry_id,
                    normalized_user_id,
                    normalized_memory_type,
                    normalized_content,
                    normalized_summary,
                    normalized_session_id,
                    normalized_run_id,
                    normalized_writer_type,
                    normalized_writer_id,
                    normalized_confidence,
                    now,
                    now,
                ),
            )
            self._append_audit_event(
                conn,
                user_id=normalized_user_id,
                target_scope="user_memory",
                target_id=resolved_entry_id,
                action="memory_create",
                writer_type=normalized_writer_type,
                writer_id=normalized_writer_id,
                session_id=normalized_session_id,
                run_id=normalized_run_id,
                payload={
                    "memory_type": normalized_memory_type,
                    "summary": normalized_summary,
                },
                created_at=now,
            )

        created = self.get_memory_entry_record(resolved_entry_id, user_id=normalized_user_id)
        if created is None:
            raise RuntimeError(f"Failed to create memory entry: {resolved_entry_id}")
        return created.model_dump(mode="python")

    def get_memory_entry_record(
        self,
        entry_id: str,
        *,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> UserMemoryEntryRecord | None:
        """Get one user memory entry by id."""
        normalized_entry_id = self._normalize_text(
            entry_id,
            field_name="entry_id",
            required=True,
        )
        params: list[Any] = [normalized_entry_id]
        sql = "SELECT * FROM user_memory_entries WHERE id = ?"
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(self._normalize_user_id(user_id))
        if not include_deleted:
            sql += " AND is_deleted = 0"
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._entry_from_row(row)

    def get_memory_entry(
        self,
        entry_id: str,
        *,
        user_id: str | None = None,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        """Get one user memory entry as a dict payload."""
        record = self.get_memory_entry_record(
            entry_id,
            user_id=user_id,
            include_deleted=include_deleted,
        )
        if record is None:
            return None
        return record.model_dump(mode="python")

    def list_audit_event_records(
        self,
        user_id: str,
        *,
        target_scope: str | None = None,
        target_id: str | None = None,
        limit: int = 50,
    ) -> list[MemoryAuditEventRecord]:
        """List recent audit events for one user and optional memory target."""
        normalized_user_id = self._normalize_user_id(user_id)
        params: list[Any] = [normalized_user_id]
        sql = """
            SELECT *
            FROM memory_audit_events
            WHERE user_id = ?
        """
        if target_scope is not None:
            sql += " AND target_scope = ?"
            params.append(self._normalize_text(target_scope, field_name="target_scope", required=True))
        if target_id is not None:
            sql += " AND target_id = ?"
            params.append(self._normalize_text(target_id, field_name="target_id", required=True))
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._audit_from_row(row) for row in rows]

    def list_audit_events(
        self,
        user_id: str,
        *,
        target_scope: str | None = None,
        target_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List recent audit events as dict payloads."""
        return [
            record.model_dump(mode="python")
            for record in self.list_audit_event_records(
                user_id,
                target_scope=target_scope,
                target_id=target_id,
                limit=limit,
            )
        ]

    def list_memory_entry_records(
        self,
        user_id: str,
        *,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
        limit: int | None = None,
    ) -> list[UserMemoryEntryRecord]:
        """List user memory entries ordered by recent updates."""
        normalized_user_id = self._normalize_user_id(user_id)
        params: list[Any] = [normalized_user_id]
        sql = """
            SELECT *
            FROM user_memory_entries
            WHERE user_id = ?
              AND is_deleted = 0
        """
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        if memory_types:
            normalized_types = [self._normalize_memory_type(item) for item in memory_types]
            placeholders = ", ".join("?" for _ in normalized_types)
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(normalized_types)
        sql += " ORDER BY updated_at DESC, created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def list_memory_entries(
        self,
        user_id: str,
        *,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List user memory entries as dict payloads."""
        return [
            record.model_dump(mode="python")
            for record in self.list_memory_entry_records(
                user_id,
                memory_types=memory_types,
                include_superseded=include_superseded,
                limit=limit,
            )
        ]

    def search_memory_entry_records(
        self,
        user_id: str,
        *,
        query: str,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
        limit: int = 20,
    ) -> list[UserMemoryEntryRecord]:
        """Search user memory entries with simple keyword matching."""
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_query = self._normalize_text(query, field_name="query", required=False)
        if not normalized_query:
            return self.list_memory_entry_records(
                normalized_user_id,
                memory_types=memory_types,
                include_superseded=include_superseded,
                limit=limit,
            )

        params: list[Any] = [normalized_user_id]
        sql = """
            SELECT *
            FROM user_memory_entries
            WHERE user_id = ?
              AND is_deleted = 0
        """
        if not include_superseded:
            sql += " AND superseded_by IS NULL"
        if memory_types:
            normalized_types = [self._normalize_memory_type(item) for item in memory_types]
            placeholders = ", ".join("?" for _ in normalized_types)
            sql += f" AND memory_type IN ({placeholders})"
            params.extend(normalized_types)

        query_terms = [term.casefold() for term in normalized_query.split() if term]
        for term in query_terms:
            sql += " AND (LOWER(content) LIKE ? OR LOWER(summary) LIKE ?)"
            like_term = f"%{term}%"
            params.extend([like_term, like_term])

        sql += " ORDER BY confidence DESC, updated_at DESC, created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._entry_from_row(row) for row in rows]

    def search_memory_entries(
        self,
        user_id: str,
        *,
        query: str,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search user memory entries as dict payloads."""
        return [
            record.model_dump(mode="python")
            for record in self.search_memory_entry_records(
                user_id,
                query=query,
                memory_types=memory_types,
                include_superseded=include_superseded,
                limit=limit,
            )
        ]

    def update_memory_entry(
        self,
        entry_id: str,
        *,
        user_id: str | None = None,
        memory_type: str | object = _UNSET,
        content: str | object = _UNSET,
        summary: str | object = _UNSET,
        source_session_id: str | None | object = _UNSET,
        source_run_id: str | None | object = _UNSET,
        writer_type: str | object = _UNSET,
        writer_id: str | object = _UNSET,
        confidence: float | int | object = _UNSET,
    ) -> dict[str, Any] | None:
        """Update mutable fields for one user memory entry."""
        record = self.get_memory_entry_record(entry_id, user_id=user_id)
        if record is None:
            return None

        next_memory_type = (
            self._normalize_memory_type(memory_type)
            if memory_type is not _UNSET
            else record.memory_type
        )
        next_content = (
            self._normalize_text(content, field_name="content", required=True)
            if content is not _UNSET
            else record.content
        )
        next_summary = (
            self._normalize_text(summary, field_name="summary", required=False)
            if summary is not _UNSET
            else record.summary
        )
        next_session_id = (
            self._normalize_text(
                source_session_id,
                field_name="source_session_id",
                required=False,
            )
            if source_session_id is not _UNSET
            else record.source_session_id
        )
        next_run_id = (
            self._normalize_text(
                source_run_id,
                field_name="source_run_id",
                required=False,
            )
            if source_run_id is not _UNSET
            else record.source_run_id
        )
        next_writer_type = (
            self._normalize_writer_type(writer_type)
            if writer_type is not _UNSET
            else record.writer_type
        )
        next_writer_id = (
            self._normalize_writer_id(writer_id)
            if writer_id is not _UNSET
            else record.writer_id
        )
        next_confidence = (
            self._normalize_confidence(confidence)
            if confidence is not _UNSET
            else record.confidence
        )
        now = utc_now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE user_memory_entries
                SET memory_type = ?, content = ?, summary = ?,
                    source_session_id = ?, source_run_id = ?,
                    writer_type = ?, writer_id = ?,
                    confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    next_memory_type,
                    next_content,
                    next_summary,
                    next_session_id,
                    next_run_id,
                    next_writer_type,
                    next_writer_id,
                    next_confidence,
                    now,
                    record.id,
                ),
            )
            self._append_audit_event(
                conn,
                user_id=record.user_id,
                target_scope="user_memory",
                target_id=record.id,
                action="memory_update",
                writer_type=next_writer_type,
                writer_id=next_writer_id,
                session_id=next_session_id,
                run_id=next_run_id,
                payload={
                    "memory_type": next_memory_type,
                    "summary": next_summary,
                },
                created_at=now,
            )

        updated = self.get_memory_entry_record(record.id, user_id=record.user_id)
        if updated is None:
            raise RuntimeError(f"Failed to update memory entry: {record.id}")
        return updated.model_dump(mode="python")

    def supersede_memory_entry(
        self,
        entry_id: str,
        *,
        superseded_by: str,
        user_id: str | None = None,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        writer_type: str | None = None,
        writer_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Mark one memory entry as superseded by a newer entry."""
        record = self.get_memory_entry_record(entry_id, user_id=user_id)
        if record is None:
            return None

        replacement = self.get_memory_entry_record(superseded_by, user_id=record.user_id)
        if replacement is None:
            raise KeyError(f"Replacement memory entry not found: {superseded_by}")
        if replacement.id == record.id:
            raise ValueError("A memory entry cannot supersede itself.")
        if replacement.superseded_by is not None:
            raise ValueError("Replacement memory entry is already superseded.")

        now = utc_now_iso()
        normalized_session_id = self._normalize_text(
            source_session_id,
            field_name="source_session_id",
            required=False,
        ) or None
        normalized_run_id = self._normalize_text(
            source_run_id,
            field_name="source_run_id",
            required=False,
        ) or None
        normalized_writer_type = self._normalize_writer_type(writer_type)
        normalized_writer_id = self._normalize_writer_id(writer_id)
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE user_memory_entries
                SET superseded_by = ?, writer_type = ?, writer_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    replacement.id,
                    normalized_writer_type,
                    normalized_writer_id,
                    now,
                    record.id,
                ),
            )
            self._append_audit_event(
                conn,
                user_id=record.user_id,
                target_scope="user_memory",
                target_id=record.id,
                action="memory_supersede",
                writer_type=normalized_writer_type,
                writer_id=normalized_writer_id,
                session_id=normalized_session_id,
                run_id=normalized_run_id,
                payload={"superseded_by": replacement.id},
                created_at=now,
            )

        updated = self.get_memory_entry_record(record.id, user_id=record.user_id)
        if updated is None:
            raise RuntimeError(f"Failed to supersede memory entry: {record.id}")
        return updated.model_dump(mode="python")

    def delete_memory_entry(
        self,
        entry_id: str,
        *,
        user_id: str | None = None,
        reason: str | None = None,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        writer_type: str | None = None,
        writer_id: str | None = None,
    ) -> bool:
        """Soft-delete one memory entry while preserving audit history."""
        record = self.get_memory_entry_record(
            entry_id,
            user_id=user_id,
            include_deleted=True,
        )
        if record is None or record.is_deleted:
            return False

        normalized_reason = self._normalize_text(
            reason,
            field_name="reason",
            required=False,
        )
        normalized_session_id = self._normalize_text(
            source_session_id,
            field_name="source_session_id",
            required=False,
        ) or None
        normalized_run_id = self._normalize_text(
            source_run_id,
            field_name="source_run_id",
            required=False,
        ) or None
        normalized_writer_type = self._normalize_writer_type(writer_type)
        normalized_writer_id = self._normalize_writer_id(writer_id)
        now = utc_now_iso()

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE user_memory_entries
                SET is_deleted = 1,
                    deleted_at = ?,
                    deleted_reason = ?,
                    writer_type = ?,
                    writer_id = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    now,
                    normalized_reason,
                    normalized_writer_type,
                    normalized_writer_id,
                    now,
                    record.id,
                ),
            )
            self._append_audit_event(
                conn,
                user_id=record.user_id,
                target_scope="user_memory",
                target_id=record.id,
                action="memory_delete",
                writer_type=normalized_writer_type,
                writer_id=normalized_writer_id,
                session_id=normalized_session_id,
                run_id=normalized_run_id,
                payload={"reason": normalized_reason},
                created_at=now,
            )
        return True

    def enforce_memory_capacity(
        self,
        user_id: str,
        *,
        memory_type: str,
        max_active: int,
        preferred_entry_id: str | None = None,
        source_session_id: str | None = None,
        source_run_id: str | None = None,
        writer_type: str | None = None,
        writer_id: str | None = None,
    ) -> list[str]:
        """Keep only the strongest bounded set of active memories for one type."""
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_memory_type = self._normalize_memory_type(memory_type)
        normalized_limit = max(1, int(max_active))
        normalized_preferred_entry_id = self._normalize_text(
            preferred_entry_id,
            field_name="preferred_entry_id",
            required=False,
        ) or None
        normalized_session_id = self._normalize_text(
            source_session_id,
            field_name="source_session_id",
            required=False,
        ) or None
        normalized_run_id = self._normalize_text(
            source_run_id,
            field_name="source_run_id",
            required=False,
        ) or None
        normalized_writer_type = self._normalize_writer_type(writer_type)
        normalized_writer_id = self._normalize_writer_id(writer_id)

        with self._connect() as conn:
            self._ensure_user_exists(conn, normalized_user_id)
            rows = conn.execute(
                """
                SELECT *
                FROM user_memory_entries
                WHERE user_id = ?
                  AND memory_type = ?
                  AND superseded_by IS NULL
                  AND is_deleted = 0
                ORDER BY confidence DESC, updated_at DESC, created_at DESC
                """,
                (
                    normalized_user_id,
                    normalized_memory_type,
                ),
            ).fetchall()
            if len(rows) <= normalized_limit:
                return []

            ordered_rows = list(rows)
            ordered_rows.sort(key=lambda row: str(row["created_at"]), reverse=True)
            ordered_rows.sort(key=lambda row: str(row["updated_at"]), reverse=True)
            ordered_rows.sort(key=lambda row: len(str(row["content"] or "")), reverse=True)
            ordered_rows.sort(key=lambda row: len(str(row["summary"] or "")), reverse=True)
            ordered_rows.sort(key=lambda row: float(row["confidence"]), reverse=True)
            if normalized_preferred_entry_id:
                ordered_rows.sort(
                    key=lambda row: row["id"] != normalized_preferred_entry_id
                )
            kept_rows = ordered_rows[:normalized_limit]
            overflow_rows = ordered_rows[normalized_limit:]
            canonical_id = (
                normalized_preferred_entry_id
                if normalized_preferred_entry_id and any(row["id"] == normalized_preferred_entry_id for row in kept_rows)
                else kept_rows[0]["id"]
            )
            now = utc_now_iso()
            compacted_entry_ids: list[str] = []

            for row in overflow_rows:
                conn.execute(
                    """
                    UPDATE user_memory_entries
                    SET superseded_by = ?, writer_type = ?, writer_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        canonical_id,
                        normalized_writer_type,
                        normalized_writer_id,
                        now,
                        row["id"],
                    ),
                )
                self._append_audit_event(
                    conn,
                    user_id=normalized_user_id,
                    target_scope="user_memory",
                    target_id=row["id"],
                    action="memory_capacity_compact",
                    writer_type=normalized_writer_type,
                    writer_id=normalized_writer_id,
                    session_id=normalized_session_id,
                    run_id=normalized_run_id,
                    payload={
                        "memory_type": normalized_memory_type,
                        "retained_entry_id": canonical_id,
                        "max_active": normalized_limit,
                    },
                    created_at=now,
                )
                compacted_entry_ids.append(row["id"])

        return compacted_entry_ids

    def compact_memory_entries(
        self,
        user_id: str,
        *,
        memory_type: str | None = None,
        writer_type: str | None = None,
        writer_id: str | None = None,
    ) -> UserMemoryCompactionResult:
        """Merge exact-duplicate active memories for one user."""
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_memory_type = (
            self._normalize_memory_type(memory_type)
            if memory_type is not None
            else None
        )
        normalized_writer_type = self._normalize_writer_type(writer_type)
        normalized_writer_id = self._normalize_writer_id(writer_id)
        params: list[Any] = [normalized_user_id]
        sql = """
            SELECT *
            FROM user_memory_entries
            WHERE user_id = ?
              AND superseded_by IS NULL
              AND is_deleted = 0
        """
        if normalized_memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(normalized_memory_type)
        sql += " ORDER BY confidence DESC, updated_at DESC, created_at ASC"

        with self._connect() as conn:
            self._ensure_user_exists(conn, normalized_user_id)
            rows = conn.execute(sql, tuple(params)).fetchall()

            grouped_rows: dict[tuple[str, str], list[sqlite3.Row]] = {}
            for row in rows:
                key = (
                    row["memory_type"],
                    self._normalized_content_key(row["content"]),
                )
                if not key[1]:
                    continue
                grouped_rows.setdefault(key, []).append(row)

            canonical_entry_ids: list[str] = []
            superseded_entry_ids: list[str] = []
            now = utc_now_iso()

            for duplicates in grouped_rows.values():
                if len(duplicates) < 2:
                    continue

                canonical = duplicates[0]
                merged_summary = max(
                    (str(row["summary"] or "") for row in duplicates),
                    key=len,
                )
                merged_confidence = max(float(row["confidence"]) for row in duplicates)
                merged_session_id = canonical["source_session_id"] or next(
                    (row["source_session_id"] for row in duplicates if row["source_session_id"]),
                    None,
                )
                merged_run_id = canonical["source_run_id"] or next(
                    (row["source_run_id"] for row in duplicates if row["source_run_id"]),
                    None,
                )

                conn.execute(
                    """
                    UPDATE user_memory_entries
                    SET summary = ?, confidence = ?, source_session_id = ?, source_run_id = ?,
                        writer_type = ?, writer_id = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        merged_summary,
                        merged_confidence,
                        merged_session_id,
                        merged_run_id,
                        normalized_writer_type,
                        normalized_writer_id,
                        now,
                        canonical["id"],
                    ),
                )
                self._append_audit_event(
                    conn,
                    user_id=normalized_user_id,
                    target_scope="user_memory",
                    target_id=canonical["id"],
                    action="memory_compact_merge",
                    writer_type=normalized_writer_type,
                    writer_id=normalized_writer_id,
                    session_id=merged_session_id,
                    run_id=merged_run_id,
                    payload={
                        "merged_summary": merged_summary,
                        "superseded_entry_ids": [row["id"] for row in duplicates[1:]],
                    },
                    created_at=now,
                )

                for duplicate in duplicates[1:]:
                    conn.execute(
                        """
                        UPDATE user_memory_entries
                        SET superseded_by = ?, writer_type = ?, writer_id = ?, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            canonical["id"],
                            normalized_writer_type,
                            normalized_writer_id,
                            now,
                            duplicate["id"],
                        ),
                    )
                    superseded_entry_ids.append(duplicate["id"])

                canonical_entry_ids.append(canonical["id"])

        return UserMemoryCompactionResult(
            user_id=normalized_user_id,
            merged_group_count=len(canonical_entry_ids),
            canonical_entry_ids=canonical_entry_ids,
            superseded_entry_ids=superseded_entry_ids,
        )
