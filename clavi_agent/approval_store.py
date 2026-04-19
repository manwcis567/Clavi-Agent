"""Persistent ApprovalStore backed by the shared session SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from .account_constants import ROOT_ACCOUNT_ID
from .run_models import ApprovalRequestRecord
from .sqlite_schema import configure_connection, ensure_session_db_schema


class ApprovalStore:
    """SQLite repository for approval requests."""

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
    def _resolve_run_account_id(
        conn: sqlite3.Connection,
        run_id: str,
        fallback: str = ROOT_ACCOUNT_ID,
    ) -> str:
        row = conn.execute(
            "SELECT account_id FROM runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            return fallback
        return str(row["account_id"] or fallback)

    @staticmethod
    def _request_from_row(row: sqlite3.Row) -> ApprovalRequestRecord:
        return ApprovalRequestRecord.model_validate(dict(row))

    def create_request(self, request: ApprovalRequestRecord) -> ApprovalRequestRecord:
        """Persist one approval request."""
        with self._connect() as conn:
            resolved_account_id = self._resolve_run_account_id(
                conn,
                request.run_id,
                request.account_id,
            )
            conn.execute(
                """
                INSERT INTO approval_requests (
                    id, run_id, account_id, step_id, tool_name, risk_level, status, parameter_summary,
                    impact_summary, requested_at, resolved_at, decision_notes, decision_scope
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.id,
                    request.run_id,
                    resolved_account_id,
                    request.step_id,
                    request.tool_name,
                    request.risk_level,
                    request.status,
                    request.parameter_summary,
                    request.impact_summary,
                    request.requested_at,
                    request.resolved_at,
                    request.decision_notes,
                    request.decision_scope,
                ),
            )
        return request.model_copy(update={"account_id": resolved_account_id})

    def get_request(
        self,
        request_id: str,
        *,
        account_id: str | None = None,
    ) -> ApprovalRequestRecord | None:
        """Return one approval request by id."""
        params: list[Any] = [request_id]
        sql = "SELECT * FROM approval_requests WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()
        if row is None:
            return None
        return self._request_from_row(row)

    def list_requests(
        self,
        *,
        account_id: str | None = None,
        status: str | None = None,
        run_id: str | None = None,
    ) -> list[ApprovalRequestRecord]:
        """List approval requests with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)

        sql = "SELECT * FROM approval_requests"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY requested_at DESC"

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._request_from_row(row) for row in rows]

    def update_request(self, request: ApprovalRequestRecord) -> ApprovalRequestRecord:
        """Persist the latest request state."""
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE approval_requests
                SET run_id = ?, account_id = ?, step_id = ?, tool_name = ?, risk_level = ?, status = ?,
                    parameter_summary = ?, impact_summary = ?, requested_at = ?, resolved_at = ?,
                    decision_notes = ?, decision_scope = ?
                WHERE id = ?
                """,
                (
                    request.run_id,
                    request.account_id,
                    request.step_id,
                    request.tool_name,
                    request.risk_level,
                    request.status,
                    request.parameter_summary,
                    request.impact_summary,
                    request.requested_at,
                    request.resolved_at,
                    request.decision_notes,
                    request.decision_scope,
                    request.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Approval request not found: {request.id}")
        return request
