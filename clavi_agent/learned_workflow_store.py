"""SQLite persistence for learned workflow candidates."""

from __future__ import annotations

import json
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .learned_workflow_models import (
    LearnedWorkflowCandidateRecord,
    WorkflowCandidateSignal,
    WorkflowCandidateStatus,
)
from .sqlite_schema import configure_connection, ensure_session_db_schema, utc_now_iso

_VALID_STATUSES = {"pending_review", "approved", "rejected", "installed"}
_VALID_SIGNALS = {
    "repeated_task_pattern",
    "successful_complex_run",
    "user_endorsed_solution",
}


class LearnedWorkflowStore:
    """Repository for reviewable learned-workflow candidates."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return configure_connection(sqlite3.connect(self.db_path))

    def _initialize(self) -> None:
        with self._connect() as conn:
            ensure_session_db_schema(conn)

    @staticmethod
    def generate_candidate_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def _json_load(payload: str | None, fallback: Any) -> Any:
        if not payload:
            return fallback
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _json_dump(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _normalize_text(value: str | None, *, required: bool = False) -> str:
        normalized = str(value or "").strip()
        if required and not normalized:
            raise ValueError("required text field is blank")
        return normalized

    @classmethod
    def _normalize_status(cls, status: str | None) -> WorkflowCandidateStatus:
        normalized = str(status or "pending_review").strip().lower()
        if normalized not in _VALID_STATUSES:
            raise ValueError(f"Unsupported workflow candidate status: {status}")
        return normalized  # type: ignore[return-value]

    @classmethod
    def _normalize_signals(
        cls,
        signal_types: list[str] | tuple[str, ...] | None,
    ) -> list[WorkflowCandidateSignal]:
        normalized: list[WorkflowCandidateSignal] = []
        seen: set[str] = set()
        for item in signal_types or []:
            signal = str(item or "").strip()
            if not signal or signal in seen:
                continue
            if signal not in _VALID_SIGNALS:
                raise ValueError(f"Unsupported workflow candidate signal: {item}")
            normalized.append(signal)  # type: ignore[arg-type]
            seen.add(signal)
        return normalized

    @staticmethod
    def _normalize_string_list(values: list[object] | tuple[object, ...] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in values or []:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return normalized

    @classmethod
    def normalize_skill_name(cls, value: str | None, *, fallback: str) -> str:
        raw = str(value or "").strip().lower()
        slug = re.sub(r"[^a-z0-9._-]+", "-", raw).strip("-")
        return slug or fallback

    def _candidate_from_row(self, row: sqlite3.Row) -> LearnedWorkflowCandidateRecord:
        return LearnedWorkflowCandidateRecord(
            id=row["id"],
            account_id=row["account_id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            agent_template_id=row["agent_template_id"],
            status=self._normalize_status(row["status"]),
            title=row["title"],
            summary=row["summary"],
            description=row["description"],
            signal_types=self._normalize_signals(
                self._json_load(row["signal_types_json"], []),
            ),
            source_run_ids=self._normalize_string_list(
                self._json_load(row["source_run_ids_json"], []),
            ),
            tool_names=self._normalize_string_list(
                self._json_load(row["tool_names_json"], []),
            ),
            step_titles=self._normalize_string_list(
                self._json_load(row["step_titles_json"], []),
            ),
            artifact_ids=self._normalize_string_list(
                self._json_load(row["artifact_ids_json"], []),
            ),
            suggested_skill_name=row["suggested_skill_name"],
            generated_skill_markdown=row["generated_skill_markdown"],
            review_notes=row["review_notes"],
            installed_agent_id=row["installed_agent_id"],
            installed_skill_path=row["installed_skill_path"],
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            approved_at=row["approved_at"],
            rejected_at=row["rejected_at"],
            installed_at=row["installed_at"],
        )

    def get_candidate_record(
        self,
        candidate_id: str,
        *,
        account_id: str | None = None,
    ) -> LearnedWorkflowCandidateRecord | None:
        params: list[Any] = [self._normalize_text(candidate_id, required=True)]
        sql = "SELECT * FROM learned_workflow_candidates WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(self._normalize_text(account_id, required=True))
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._candidate_from_row(row)

    def get_candidate_by_run(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
    ) -> LearnedWorkflowCandidateRecord | None:
        params: list[Any] = [self._normalize_text(run_id, required=True)]
        sql = "SELECT * FROM learned_workflow_candidates WHERE run_id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(self._normalize_text(account_id, required=True))
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._candidate_from_row(row)

    def create_candidate(
        self,
        candidate: LearnedWorkflowCandidateRecord,
    ) -> LearnedWorkflowCandidateRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learned_workflow_candidates (
                    id, account_id, run_id, session_id, agent_template_id, status,
                    title, summary, description, signal_types_json, source_run_ids_json,
                    tool_names_json, step_titles_json, artifact_ids_json,
                    suggested_skill_name, generated_skill_markdown, review_notes,
                    installed_agent_id, installed_skill_path, metadata_json,
                    created_at, updated_at, approved_at, rejected_at, installed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    candidate.id,
                    candidate.account_id,
                    candidate.run_id,
                    candidate.session_id,
                    candidate.agent_template_id,
                    candidate.status,
                    candidate.title,
                    candidate.summary,
                    candidate.description,
                    self._json_dump(candidate.signal_types),
                    self._json_dump(candidate.source_run_ids),
                    self._json_dump(candidate.tool_names),
                    self._json_dump(candidate.step_titles),
                    self._json_dump(candidate.artifact_ids),
                    candidate.suggested_skill_name,
                    candidate.generated_skill_markdown,
                    candidate.review_notes,
                    candidate.installed_agent_id,
                    candidate.installed_skill_path,
                    self._json_dump(candidate.metadata),
                    candidate.created_at,
                    candidate.updated_at,
                    candidate.approved_at,
                    candidate.rejected_at,
                    candidate.installed_at,
                ),
            )
        return candidate

    def update_candidate(
        self,
        candidate: LearnedWorkflowCandidateRecord,
    ) -> LearnedWorkflowCandidateRecord:
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE learned_workflow_candidates
                SET account_id = ?, run_id = ?, session_id = ?, agent_template_id = ?, status = ?,
                    title = ?, summary = ?, description = ?, signal_types_json = ?,
                    source_run_ids_json = ?, tool_names_json = ?, step_titles_json = ?,
                    artifact_ids_json = ?, suggested_skill_name = ?, generated_skill_markdown = ?,
                    review_notes = ?, installed_agent_id = ?, installed_skill_path = ?,
                    metadata_json = ?, created_at = ?, updated_at = ?, approved_at = ?,
                    rejected_at = ?, installed_at = ?
                WHERE id = ?
                """,
                (
                    candidate.account_id,
                    candidate.run_id,
                    candidate.session_id,
                    candidate.agent_template_id,
                    candidate.status,
                    candidate.title,
                    candidate.summary,
                    candidate.description,
                    self._json_dump(candidate.signal_types),
                    self._json_dump(candidate.source_run_ids),
                    self._json_dump(candidate.tool_names),
                    self._json_dump(candidate.step_titles),
                    self._json_dump(candidate.artifact_ids),
                    candidate.suggested_skill_name,
                    candidate.generated_skill_markdown,
                    candidate.review_notes,
                    candidate.installed_agent_id,
                    candidate.installed_skill_path,
                    self._json_dump(candidate.metadata),
                    candidate.created_at,
                    candidate.updated_at,
                    candidate.approved_at,
                    candidate.rejected_at,
                    candidate.installed_at,
                    candidate.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Learned workflow candidate not found: {candidate.id}")
        return candidate

    def upsert_candidate_for_run(
        self,
        *,
        account_id: str,
        run_id: str,
        session_id: str,
        agent_template_id: str,
        title: str,
        summary: str = "",
        description: str = "",
        signal_types: list[str] | None = None,
        source_run_ids: list[str] | None = None,
        tool_names: list[str] | None = None,
        step_titles: list[str] | None = None,
        artifact_ids: list[str] | None = None,
        suggested_skill_name: str = "",
        generated_skill_markdown: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> LearnedWorkflowCandidateRecord:
        existing = self.get_candidate_by_run(run_id, account_id=account_id)
        now = utc_now_iso()
        fallback_skill_name = f"learned-workflow-{self._normalize_text(run_id, required=True)[:8]}"
        candidate = LearnedWorkflowCandidateRecord(
            id=existing.id if existing is not None else self.generate_candidate_id(),
            account_id=self._normalize_text(account_id, required=True),
            run_id=self._normalize_text(run_id, required=True),
            session_id=self._normalize_text(session_id, required=True),
            agent_template_id=self._normalize_text(agent_template_id, required=True),
            status="pending_review" if existing is None else existing.status,
            title=self._normalize_text(title, required=True),
            summary=self._normalize_text(summary),
            description=self._normalize_text(description),
            signal_types=self._normalize_signals(signal_types),
            source_run_ids=self._normalize_string_list(source_run_ids),
            tool_names=self._normalize_string_list(tool_names),
            step_titles=self._normalize_string_list(step_titles),
            artifact_ids=self._normalize_string_list(artifact_ids),
            suggested_skill_name=self.normalize_skill_name(
                suggested_skill_name,
                fallback=fallback_skill_name,
            ),
            generated_skill_markdown=str(generated_skill_markdown or ""),
            review_notes=existing.review_notes if existing is not None else "",
            installed_agent_id=existing.installed_agent_id if existing is not None else None,
            installed_skill_path=existing.installed_skill_path if existing is not None else None,
            metadata=dict(metadata or {}),
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            approved_at=existing.approved_at if existing is not None else None,
            rejected_at=existing.rejected_at if existing is not None else None,
            installed_at=existing.installed_at if existing is not None else None,
        )
        if existing is None:
            return self.create_candidate(candidate)
        return self.update_candidate(candidate)

    def list_candidate_records(
        self,
        *,
        account_id: str | None = None,
        status: str | None = None,
        agent_template_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[LearnedWorkflowCandidateRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(self._normalize_text(account_id, required=True))
        if status is not None:
            clauses.append("status = ?")
            params.append(self._normalize_status(status))
        if agent_template_id is not None:
            clauses.append("agent_template_id = ?")
            params.append(self._normalize_text(agent_template_id, required=True))
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(self._normalize_text(run_id, required=True))

        sql = "SELECT * FROM learned_workflow_candidates"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY updated_at DESC, created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._candidate_from_row(row) for row in rows]

    def update_candidate_status(
        self,
        candidate_id: str,
        *,
        account_id: str | None = None,
        status: str,
        review_notes: str | None = None,
        installed_agent_id: str | None = None,
        installed_skill_path: str | None = None,
    ) -> LearnedWorkflowCandidateRecord:
        existing = self.get_candidate_record(candidate_id, account_id=account_id)
        if existing is None:
            raise KeyError(f"Learned workflow candidate not found: {candidate_id}")

        normalized_status = self._normalize_status(status)
        now = utc_now_iso()
        updated = existing.model_copy(
            update={
                "status": normalized_status,
                "review_notes": (
                    existing.review_notes
                    if review_notes is None
                    else self._normalize_text(review_notes)
                ),
                "installed_agent_id": (
                    existing.installed_agent_id
                    if installed_agent_id is None
                    else self._normalize_text(installed_agent_id) or None
                ),
                "installed_skill_path": (
                    existing.installed_skill_path
                    if installed_skill_path is None
                    else self._normalize_text(installed_skill_path) or None
                ),
                "updated_at": now,
                "approved_at": (
                    now if normalized_status == "approved" else existing.approved_at
                ),
                "rejected_at": (
                    now if normalized_status == "rejected" else existing.rejected_at
                ),
                "installed_at": (
                    now if normalized_status == "installed" else existing.installed_at
                ),
            }
        )
        return self.update_candidate(updated)
