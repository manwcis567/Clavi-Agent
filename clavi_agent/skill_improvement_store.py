"""SQLite persistence for reviewable skill-improvement proposals."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .skill_improvement_models import (
    SkillImprovementProposalRecord,
    SkillImprovementProposalStatus,
    SkillImprovementSignal,
)
from .sqlite_schema import configure_connection, ensure_session_db_schema, utc_now_iso

_VALID_STATUSES = {"pending_review", "approved", "rejected", "applied"}
_VALID_SIGNALS = {
    "repeated_user_corrections",
    "repeated_run_failures",
    "manual_successful_refinement",
}


class SkillImprovementStore:
    """Repository for skill-improvement proposals."""

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
    def generate_proposal_id() -> str:
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
    def _normalize_status(
        cls,
        status: str | None,
    ) -> SkillImprovementProposalStatus:
        normalized = str(status or "pending_review").strip().lower()
        if normalized not in _VALID_STATUSES:
            raise ValueError(f"Unsupported skill improvement status: {status}")
        return normalized  # type: ignore[return-value]

    @classmethod
    def _normalize_signals(
        cls,
        signal_types: list[str] | tuple[str, ...] | None,
    ) -> list[SkillImprovementSignal]:
        normalized: list[SkillImprovementSignal] = []
        seen: set[str] = set()
        for item in signal_types or []:
            signal = str(item or "").strip()
            if not signal or signal in seen:
                continue
            if signal not in _VALID_SIGNALS:
                raise ValueError(f"Unsupported skill improvement signal: {item}")
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

    def _proposal_from_row(self, row: sqlite3.Row) -> SkillImprovementProposalRecord:
        return SkillImprovementProposalRecord(
            id=row["id"],
            account_id=row["account_id"],
            run_id=row["run_id"],
            session_id=row["session_id"],
            agent_template_id=row["agent_template_id"],
            skill_name=row["skill_name"],
            target_skill_path=row["target_skill_path"],
            status=self._normalize_status(row["status"]),
            title=row["title"],
            summary=row["summary"],
            signal_types=self._normalize_signals(
                self._json_load(row["signal_types_json"], []),
            ),
            source_run_ids=self._normalize_string_list(
                self._json_load(row["source_run_ids_json"], []),
            ),
            base_version=max(1, int(row["base_version"] or 1)),
            proposed_version=max(1, int(row["proposed_version"] or 1)),
            current_skill_markdown=row["current_skill_markdown"],
            proposed_skill_markdown=row["proposed_skill_markdown"],
            changelog_entry=row["changelog_entry"],
            review_notes=row["review_notes"],
            applied_skill_path=row["applied_skill_path"],
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            approved_at=row["approved_at"],
            rejected_at=row["rejected_at"],
            applied_at=row["applied_at"],
        )

    def get_proposal_record(
        self,
        proposal_id: str,
        *,
        account_id: str | None = None,
    ) -> SkillImprovementProposalRecord | None:
        params: list[Any] = [self._normalize_text(proposal_id, required=True)]
        sql = "SELECT * FROM skill_improvement_proposals WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(self._normalize_text(account_id, required=True))
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._proposal_from_row(row)

    def get_proposal_by_run_and_skill(
        self,
        run_id: str,
        skill_name: str,
        *,
        account_id: str | None = None,
    ) -> SkillImprovementProposalRecord | None:
        params: list[Any] = [
            self._normalize_text(run_id, required=True),
            self._normalize_text(skill_name, required=True),
        ]
        sql = "SELECT * FROM skill_improvement_proposals WHERE run_id = ? AND skill_name = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(self._normalize_text(account_id, required=True))
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._proposal_from_row(row)

    def create_proposal(
        self,
        proposal: SkillImprovementProposalRecord,
    ) -> SkillImprovementProposalRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO skill_improvement_proposals (
                    id, account_id, run_id, session_id, agent_template_id, skill_name,
                    target_skill_path, status, title, summary, signal_types_json,
                    source_run_ids_json, base_version, proposed_version,
                    current_skill_markdown, proposed_skill_markdown, changelog_entry,
                    review_notes, applied_skill_path, metadata_json,
                    created_at, updated_at, approved_at, rejected_at, applied_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.id,
                    proposal.account_id,
                    proposal.run_id,
                    proposal.session_id,
                    proposal.agent_template_id,
                    proposal.skill_name,
                    proposal.target_skill_path,
                    proposal.status,
                    proposal.title,
                    proposal.summary,
                    self._json_dump(proposal.signal_types),
                    self._json_dump(proposal.source_run_ids),
                    proposal.base_version,
                    proposal.proposed_version,
                    proposal.current_skill_markdown,
                    proposal.proposed_skill_markdown,
                    proposal.changelog_entry,
                    proposal.review_notes,
                    proposal.applied_skill_path,
                    self._json_dump(proposal.metadata),
                    proposal.created_at,
                    proposal.updated_at,
                    proposal.approved_at,
                    proposal.rejected_at,
                    proposal.applied_at,
                ),
            )
        return proposal

    def update_proposal(
        self,
        proposal: SkillImprovementProposalRecord,
    ) -> SkillImprovementProposalRecord:
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE skill_improvement_proposals
                SET account_id = ?, run_id = ?, session_id = ?, agent_template_id = ?, skill_name = ?,
                    target_skill_path = ?, status = ?, title = ?, summary = ?, signal_types_json = ?,
                    source_run_ids_json = ?, base_version = ?, proposed_version = ?,
                    current_skill_markdown = ?, proposed_skill_markdown = ?, changelog_entry = ?,
                    review_notes = ?, applied_skill_path = ?, metadata_json = ?, created_at = ?,
                    updated_at = ?, approved_at = ?, rejected_at = ?, applied_at = ?
                WHERE id = ?
                """,
                (
                    proposal.account_id,
                    proposal.run_id,
                    proposal.session_id,
                    proposal.agent_template_id,
                    proposal.skill_name,
                    proposal.target_skill_path,
                    proposal.status,
                    proposal.title,
                    proposal.summary,
                    self._json_dump(proposal.signal_types),
                    self._json_dump(proposal.source_run_ids),
                    proposal.base_version,
                    proposal.proposed_version,
                    proposal.current_skill_markdown,
                    proposal.proposed_skill_markdown,
                    proposal.changelog_entry,
                    proposal.review_notes,
                    proposal.applied_skill_path,
                    self._json_dump(proposal.metadata),
                    proposal.created_at,
                    proposal.updated_at,
                    proposal.approved_at,
                    proposal.rejected_at,
                    proposal.applied_at,
                    proposal.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Skill improvement proposal not found: {proposal.id}")
        return proposal

    def upsert_proposal_for_skill(
        self,
        *,
        account_id: str,
        run_id: str,
        session_id: str,
        agent_template_id: str,
        skill_name: str,
        target_skill_path: str,
        title: str,
        summary: str = "",
        signal_types: list[str] | None = None,
        source_run_ids: list[str] | None = None,
        base_version: int = 1,
        proposed_version: int = 2,
        current_skill_markdown: str = "",
        proposed_skill_markdown: str = "",
        changelog_entry: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> SkillImprovementProposalRecord:
        existing = self.get_proposal_by_run_and_skill(
            run_id,
            skill_name,
            account_id=account_id,
        )
        now = utc_now_iso()
        proposal = SkillImprovementProposalRecord(
            id=existing.id if existing is not None else self.generate_proposal_id(),
            account_id=self._normalize_text(account_id, required=True),
            run_id=self._normalize_text(run_id, required=True),
            session_id=self._normalize_text(session_id, required=True),
            agent_template_id=self._normalize_text(agent_template_id, required=True),
            skill_name=self._normalize_text(skill_name, required=True),
            target_skill_path=self._normalize_text(target_skill_path, required=True),
            status="pending_review" if existing is None else existing.status,
            title=self._normalize_text(title, required=True),
            summary=self._normalize_text(summary),
            signal_types=self._normalize_signals(signal_types),
            source_run_ids=self._normalize_string_list(source_run_ids),
            base_version=max(1, int(base_version)),
            proposed_version=max(1, int(proposed_version)),
            current_skill_markdown=str(current_skill_markdown or ""),
            proposed_skill_markdown=str(proposed_skill_markdown or ""),
            changelog_entry=self._normalize_text(changelog_entry),
            review_notes=existing.review_notes if existing is not None else "",
            applied_skill_path=existing.applied_skill_path if existing is not None else None,
            metadata=dict(metadata or {}),
            created_at=existing.created_at if existing is not None else now,
            updated_at=now,
            approved_at=existing.approved_at if existing is not None else None,
            rejected_at=existing.rejected_at if existing is not None else None,
            applied_at=existing.applied_at if existing is not None else None,
        )
        if existing is None:
            return self.create_proposal(proposal)
        return self.update_proposal(proposal)

    def list_proposal_records(
        self,
        *,
        account_id: str | None = None,
        status: str | None = None,
        agent_template_id: str | None = None,
        skill_name: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[SkillImprovementProposalRecord]:
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
        if skill_name is not None:
            clauses.append("skill_name = ?")
            params.append(self._normalize_text(skill_name, required=True))
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(self._normalize_text(run_id, required=True))

        sql = "SELECT * FROM skill_improvement_proposals"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY updated_at DESC, created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._proposal_from_row(row) for row in rows]

    def update_proposal_status(
        self,
        proposal_id: str,
        *,
        account_id: str | None = None,
        status: str,
        review_notes: str | None = None,
        applied_skill_path: str | None = None,
    ) -> SkillImprovementProposalRecord:
        existing = self.get_proposal_record(proposal_id, account_id=account_id)
        if existing is None:
            raise KeyError(f"Skill improvement proposal not found: {proposal_id}")

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
                "applied_skill_path": (
                    existing.applied_skill_path
                    if applied_skill_path is None
                    else self._normalize_text(applied_skill_path) or None
                ),
                "updated_at": now,
                "approved_at": (
                    now if normalized_status == "approved" else existing.approved_at
                ),
                "rejected_at": (
                    now if normalized_status == "rejected" else existing.rejected_at
                ),
                "applied_at": (
                    now if normalized_status == "applied" else existing.applied_at
                ),
            }
        )
        return self.update_proposal(updated)
