"""Persistent TraceStore backed by the shared session SQLite database."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .account_constants import ROOT_ACCOUNT_ID
from .run_models import TraceEventRecord
from .sqlite_schema import configure_connection, ensure_session_db_schema


class TraceStore:
    """SQLite repository for trace events."""

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
    def _event_from_row(row: sqlite3.Row) -> TraceEventRecord:
        return TraceEventRecord.model_validate(dict(row))

    def create_event(self, event: TraceEventRecord) -> TraceEventRecord:
        """Persist one trace event."""
        with self._connect() as conn:
            resolved_account_id = self._resolve_run_account_id(
                conn,
                event.run_id,
                event.account_id,
            )
            conn.execute(
                """
                INSERT INTO trace_events (
                    id, run_id, account_id, parent_run_id, step_id, sequence, event_type,
                    status, payload_summary, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.run_id,
                    resolved_account_id,
                    event.parent_run_id,
                    event.step_id,
                    event.sequence,
                    event.event_type,
                    event.status,
                    event.payload_summary,
                    event.duration_ms,
                    event.created_at,
                ),
            )
        return event.model_copy(update={"account_id": resolved_account_id})

    def list_events(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        offset: int | None = None,
    ) -> list[TraceEventRecord]:
        """Return trace events for one run in timeline order."""
        params: list[object] = [run_id]
        where_clause = "WHERE run_id = ?"
        if account_id is not None:
            where_clause += " AND account_id = ?"
            params.append(account_id)
        if offset is not None:
            where_clause += " AND sequence >= ?"
            params.append(max(0, int(offset)))

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM trace_events
                {where_clause}
                ORDER BY created_at ASC, sequence ASC
                """,
                tuple(params),
            ).fetchall()
        return [self._event_from_row(row) for row in rows]
