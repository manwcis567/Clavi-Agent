"""SQLite repository for scheduled task definitions and executions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .account_constants import ROOT_ACCOUNT_ID
from .scheduled_task_models import ScheduledTaskExecutionRecord, ScheduledTaskRecord
from .sqlite_schema import configure_connection, ensure_session_db_schema


class ScheduledTaskStore:
    """Persist scheduled task configuration and execution history."""

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
    def _json_dump(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _json_load(payload: str | None, fallback: Any) -> Any:
        if not payload:
            return fallback
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _resolve_task_account_id(
        conn: sqlite3.Connection,
        task_id: str,
        fallback: str = ROOT_ACCOUNT_ID,
    ) -> str:
        row = conn.execute(
            "SELECT account_id FROM scheduled_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            return fallback
        return str(row["account_id"] or fallback)

    def _task_from_row(self, row: sqlite3.Row) -> ScheduledTaskRecord:
        return ScheduledTaskRecord(
            id=row["id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            name=row["name"],
            cron_expression=row["cron_expression"],
            timezone=row["timezone"],
            agent_id=row["agent_id"],
            prompt=row["prompt"],
            integration_id=row["integration_id"],
            target_chat_id=row["target_chat_id"],
            target_thread_id=row["target_thread_id"],
            reply_to_message_id=row["reply_to_message_id"],
            enabled=bool(row["enabled"]),
            session_id=row["session_id"],
            next_run_at=row["next_run_at"],
            last_scheduled_for=row["last_scheduled_for"],
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _execution_from_row(self, row: sqlite3.Row) -> ScheduledTaskExecutionRecord:
        return ScheduledTaskExecutionRecord(
            id=row["id"],
            task_id=row["task_id"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            trigger_kind=row["trigger_kind"],
            scheduled_for=row["scheduled_for"],
            run_id=row["run_id"],
            status=row["status"],
            error_summary=row["error_summary"],
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def create_task(self, record: ScheduledTaskRecord) -> ScheduledTaskRecord:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO scheduled_tasks (
                    id, account_id, name, cron_expression, timezone, agent_id, prompt, integration_id,
                    target_chat_id, target_thread_id, reply_to_message_id, enabled, session_id,
                    next_run_at, last_scheduled_for, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.account_id,
                    record.name,
                    record.cron_expression,
                    record.timezone,
                    record.agent_id,
                    record.prompt,
                    record.integration_id,
                    record.target_chat_id,
                    record.target_thread_id,
                    record.reply_to_message_id,
                    int(record.enabled),
                    record.session_id,
                    record.next_run_at,
                    record.last_scheduled_for,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record

    def get_task(self, task_id: str, *, account_id: str | None = None) -> ScheduledTaskRecord | None:
        params: list[Any] = [task_id]
        sql = "SELECT * FROM scheduled_tasks WHERE id = ?"
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
        return self._task_from_row(row)

    def list_tasks(
        self,
        *,
        account_id: str | None = None,
        enabled: bool | None = None,
        agent_id: str | None = None,
        integration_id: str | None = None,
        limit: int | None = None,
    ) -> list[ScheduledTaskRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if enabled is not None:
            clauses.append("enabled = ?")
            params.append(int(enabled))
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if integration_id is not None:
            clauses.append("integration_id = ?")
            params.append(integration_id)

        sql = "SELECT * FROM scheduled_tasks"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY updated_at DESC, created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._task_from_row(row) for row in rows]

    def list_due_tasks(
        self,
        *,
        account_id: str | None = None,
        due_before: str,
        limit: int | None = None,
    ) -> list[ScheduledTaskRecord]:
        sql = """
            SELECT *
            FROM scheduled_tasks
            WHERE enabled = 1
              AND next_run_at IS NOT NULL
              AND next_run_at <= ?
        """
        params: list[Any] = [due_before]
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += " ORDER BY next_run_at ASC, updated_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._task_from_row(row) for row in rows]

    def update_task(self, record: ScheduledTaskRecord) -> ScheduledTaskRecord:
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE scheduled_tasks
                SET account_id = ?, name = ?, cron_expression = ?, timezone = ?, agent_id = ?, prompt = ?,
                    integration_id = ?, target_chat_id = ?, target_thread_id = ?,
                    reply_to_message_id = ?, enabled = ?, session_id = ?, next_run_at = ?,
                    last_scheduled_for = ?, metadata_json = ?, created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    record.account_id,
                    record.name,
                    record.cron_expression,
                    record.timezone,
                    record.agent_id,
                    record.prompt,
                    record.integration_id,
                    record.target_chat_id,
                    record.target_thread_id,
                    record.reply_to_message_id,
                    int(record.enabled),
                    record.session_id,
                    record.next_run_at,
                    record.last_scheduled_for,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Scheduled task not found: {record.id}")
        return record

    def delete_task(self, task_id: str, *, account_id: str | None = None) -> bool:
        params: list[Any] = [task_id]
        sql = "DELETE FROM scheduled_tasks WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            deleted = conn.execute(
                sql,
                tuple(params),
            ).rowcount
        return deleted > 0

    def create_execution(
        self,
        record: ScheduledTaskExecutionRecord,
    ) -> ScheduledTaskExecutionRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_task_account_id(
                conn,
                record.task_id,
                record.account_id,
            )
            conn.execute(
                """
                INSERT INTO scheduled_task_executions (
                    id, task_id, account_id, trigger_kind, scheduled_for, run_id, status, error_summary,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.task_id,
                    resolved_account_id,
                    record.trigger_kind,
                    record.scheduled_for,
                    record.run_id,
                    record.status,
                    record.error_summary,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                ),
            )
        return record.model_copy(update={"account_id": resolved_account_id})

    def get_execution(
        self,
        execution_id: str,
        *,
        account_id: str | None = None,
    ) -> ScheduledTaskExecutionRecord | None:
        params: list[Any] = [execution_id]
        sql = "SELECT * FROM scheduled_task_executions WHERE id = ?"
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
        return self._execution_from_row(row)

    def list_executions(
        self,
        *,
        account_id: str | None = None,
        task_id: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[ScheduledTaskExecutionRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        sql = "SELECT * FROM scheduled_task_executions"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._execution_from_row(row) for row in rows]

    def get_latest_execution(
        self,
        task_id: str,
        *,
        account_id: str | None = None,
    ) -> ScheduledTaskExecutionRecord | None:
        params: list[Any] = [task_id]
        sql = """
            SELECT *
            FROM scheduled_task_executions
            WHERE task_id = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        sql += """
            ORDER BY created_at DESC
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._execution_from_row(row)

    def update_execution(
        self,
        record: ScheduledTaskExecutionRecord,
    ) -> ScheduledTaskExecutionRecord:
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE scheduled_task_executions
                SET task_id = ?, account_id = ?, trigger_kind = ?, scheduled_for = ?, run_id = ?, status = ?,
                    error_summary = ?, metadata_json = ?, created_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    record.task_id,
                    record.account_id,
                    record.trigger_kind,
                    record.scheduled_for,
                    record.run_id,
                    record.status,
                    record.error_summary,
                    self._json_dump(record.metadata),
                    record.created_at,
                    record.updated_at,
                    record.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Scheduled task execution not found: {record.id}")
        return record
