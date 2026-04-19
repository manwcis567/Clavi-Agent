"""Persistent RunStore backed by the shared session SQLite database."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .account_constants import ROOT_ACCOUNT_ID
from .run_models import (
    ArtifactRecord,
    RunCheckpointRecord,
    RunDeliverableManifest,
    RunRecord,
    RunStepRecord,
    TriggerMessageRef,
)
from .sqlite_schema import (
    configure_connection,
    ensure_session_db_schema,
    rebuild_session_history_fts,
)


class RunStore:
    """SQLite repository for runs, steps, checkpoints, and artifacts."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return configure_connection(sqlite3.connect(self.db_path))

    def _initialize(self) -> None:
        with self._connect() as conn:
            ensure_session_db_schema(conn)
            rebuild_session_history_fts(conn)

    @staticmethod
    def _json_dump(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _json_load(payload: str | None, fallback: Any) -> Any:
        if not payload:
            return fallback
        try:
            return json.loads(payload)
        except (TypeError, json.JSONDecodeError):
            return fallback

    @staticmethod
    def _resolve_session_account_id(
        conn: sqlite3.Connection,
        session_id: str,
        fallback: str = ROOT_ACCOUNT_ID,
    ) -> str:
        row = conn.execute(
            "SELECT account_id FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return fallback
        return str(row["account_id"] or fallback)

    def _run_from_row(self, row: sqlite3.Row) -> RunRecord:
        trigger_payload = self._json_load(row["trigger_message_ref_json"], None)
        return RunRecord.model_validate(
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "account_id": row["account_id"] or ROOT_ACCOUNT_ID,
                "agent_template_id": row["agent_template_id"],
                "agent_template_snapshot": self._json_load(
                    row["agent_template_snapshot_json"],
                    {},
                ),
                "status": row["status"],
                "goal": row["goal"],
                "trigger_message_ref": (
                    TriggerMessageRef.model_validate(trigger_payload)
                    if trigger_payload is not None
                    else None
                ),
                "parent_run_id": row["parent_run_id"],
                "run_metadata": self._json_load(row["run_metadata_json"], {}),
                "deliverable_manifest": self._json_load(
                    row["deliverable_manifest_json"],
                    RunDeliverableManifest().model_dump(mode="python"),
                ),
                "created_at": row["created_at"],
                "started_at": row["started_at"],
                "finished_at": row["finished_at"],
                "current_step_index": row["current_step_index"],
                "last_checkpoint_at": row["last_checkpoint_at"],
                "error_summary": row["error_summary"],
            }
        )

    @staticmethod
    def _step_from_row(row: sqlite3.Row) -> RunStepRecord:
        return RunStepRecord.model_validate(dict(row))

    def _checkpoint_from_row(self, row: sqlite3.Row) -> RunCheckpointRecord:
        return RunCheckpointRecord.model_validate(
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "step_sequence": row["step_sequence"],
                "trigger": row["trigger"],
                "payload": self._json_load(row["payload_json"], {}),
                "created_at": row["created_at"],
            }
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            id=row["id"],
            run_id=row["run_id"],
            step_id=row["step_id"],
            artifact_type=row["artifact_type"],
            uri=row["uri"],
            display_name=row["display_name"],
            role=row["role"],
            format=row["format"],
            mime_type=row["mime_type"],
            size_bytes=row["size_bytes"],
            source=row["source"],
            is_final=bool(row["is_final"]),
            preview_kind=row["preview_kind"],
            parent_artifact_id=row["parent_artifact_id"],
            summary=row["summary"],
            metadata=self._json_load(row["metadata_json"], {}),
            created_at=row["created_at"],
        )

    def create_run(self, run: RunRecord) -> RunRecord:
        """Persist a new run."""
        with self._connect() as conn:
            resolved_account_id = self._resolve_session_account_id(
                conn,
                run.session_id,
                run.account_id,
            )
            conn.execute(
                """
                INSERT INTO runs (
                    id, session_id, account_id, agent_template_id, agent_template_snapshot_json, status, goal,
                    trigger_message_ref_json, parent_run_id, run_metadata_json, deliverable_manifest_json,
                    created_at, started_at, finished_at, current_step_index, last_checkpoint_at, error_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.session_id,
                    resolved_account_id,
                    run.agent_template_id,
                    self._json_dump(run.agent_template_snapshot.model_dump(mode="python")),
                    run.status,
                    run.goal,
                    (
                        self._json_dump(run.trigger_message_ref.model_dump(mode="python"))
                        if run.trigger_message_ref is not None
                        else None
                    ),
                    run.parent_run_id,
                    self._json_dump(run.run_metadata),
                    self._json_dump(run.deliverable_manifest.model_dump(mode="python")),
                    run.created_at,
                    run.started_at,
                    run.finished_at,
                    run.current_step_index,
                    run.last_checkpoint_at,
                    run.error_summary,
                ),
            )
        return run.model_copy(update={"account_id": resolved_account_id})

    def get_run(self, run_id: str, *, account_id: str | None = None) -> RunRecord | None:
        """Return one run by id."""
        params: list[Any] = [run_id]
        sql = "SELECT * FROM runs WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._run_from_row(row)

    def list_runs(
        self,
        *,
        account_id: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
        parent_run_id: str | None = None,
        limit: int | None = None,
    ) -> list[RunRecord]:
        """List runs with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if account_id is not None:
            clauses.append("account_id = ?")
            params.append(account_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if parent_run_id is not None:
            clauses.append("parent_run_id = ?")
            params.append(parent_run_id)

        sql = "SELECT * FROM runs"
        if clauses:
            sql += f" WHERE {' AND '.join(clauses)}"
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._run_from_row(row) for row in rows]

    def update_run(self, run: RunRecord) -> RunRecord:
        """Persist the latest state for an existing run."""
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE runs
                SET session_id = ?, account_id = ?, agent_template_id = ?, agent_template_snapshot_json = ?, status = ?,
                    goal = ?, trigger_message_ref_json = ?, parent_run_id = ?, run_metadata_json = ?,
                    deliverable_manifest_json = ?, created_at = ?, started_at = ?, finished_at = ?,
                    current_step_index = ?, last_checkpoint_at = ?, error_summary = ?
                WHERE id = ?
                """,
                (
                    run.session_id,
                    run.account_id,
                    run.agent_template_id,
                    self._json_dump(run.agent_template_snapshot.model_dump(mode="python")),
                    run.status,
                    run.goal,
                    (
                        self._json_dump(run.trigger_message_ref.model_dump(mode="python"))
                        if run.trigger_message_ref is not None
                        else None
                    ),
                    run.parent_run_id,
                    self._json_dump(run.run_metadata),
                    self._json_dump(run.deliverable_manifest.model_dump(mode="python")),
                    run.created_at,
                    run.started_at,
                    run.finished_at,
                    run.current_step_index,
                    run.last_checkpoint_at,
                    run.error_summary,
                    run.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Run not found: {run.id}")
        return run

    def create_step(self, step: RunStepRecord) -> RunStepRecord:
        """Persist a run step."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_steps (
                    id, run_id, sequence, step_type, status, title, input_summary,
                    output_summary, started_at, finished_at, error_summary
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.id,
                    step.run_id,
                    step.sequence,
                    step.step_type,
                    step.status,
                    step.title,
                    step.input_summary,
                    step.output_summary,
                    step.started_at,
                    step.finished_at,
                    step.error_summary,
                ),
            )
        return step

    def update_step(self, step: RunStepRecord) -> RunStepRecord:
        """Persist the latest state for one step."""
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE run_steps
                SET run_id = ?, sequence = ?, step_type = ?, status = ?, title = ?, input_summary = ?,
                    output_summary = ?, started_at = ?, finished_at = ?, error_summary = ?
                WHERE id = ?
                """,
                (
                    step.run_id,
                    step.sequence,
                    step.step_type,
                    step.status,
                    step.title,
                    step.input_summary,
                    step.output_summary,
                    step.started_at,
                    step.finished_at,
                    step.error_summary,
                    step.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Run step not found: {step.id}")
        return step

    def list_steps(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
    ) -> list[RunStepRecord]:
        """Return all steps for one run."""
        params: list[Any] = [run_id]
        sql = """
            SELECT *
            FROM run_steps
            WHERE run_id = ?
        """
        if account_id is not None:
            sql += """
              AND EXISTS (
                    SELECT 1
                    FROM runs
                    WHERE runs.id = run_steps.run_id
                      AND runs.account_id = ?
              )
            """
            params.append(account_id)
        sql += " ORDER BY sequence ASC"
        with self._connect() as conn:
            rows = conn.execute(
                sql,
                tuple(params),
            ).fetchall()
        return [self._step_from_row(row) for row in rows]

    def save_checkpoint(self, checkpoint: RunCheckpointRecord) -> RunCheckpointRecord:
        """Persist one checkpoint snapshot."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO run_checkpoints (id, run_id, step_sequence, trigger, payload_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    checkpoint.id,
                    checkpoint.run_id,
                    checkpoint.step_sequence,
                    checkpoint.trigger,
                    self._json_dump(checkpoint.payload.model_dump(mode="python")),
                    checkpoint.created_at,
                ),
            )
            conn.execute(
                """
                UPDATE runs
                SET last_checkpoint_at = ?
                WHERE id = ?
                """,
                (checkpoint.created_at, checkpoint.run_id),
            )
        return checkpoint

    def get_latest_checkpoint(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
    ) -> RunCheckpointRecord | None:
        """Return the newest checkpoint for one run if present."""
        params: list[Any] = [run_id]
        sql = """
            SELECT *
            FROM run_checkpoints
            WHERE run_id = ?
        """
        if account_id is not None:
            sql += """
              AND EXISTS (
                    SELECT 1
                    FROM runs
                    WHERE runs.id = run_checkpoints.run_id
                      AND runs.account_id = ?
              )
            """
            params.append(account_id)
        sql += """
            ORDER BY created_at DESC, step_sequence DESC
            LIMIT 1
        """
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._checkpoint_from_row(row)

    def list_checkpoints(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        trigger: str | None = None,
    ) -> list[RunCheckpointRecord]:
        """Return checkpoints for one run ordered newest-first."""
        params: list[Any] = [run_id]
        where_clause = "WHERE run_id = ?"
        if account_id is not None:
            where_clause += """
                AND EXISTS (
                    SELECT 1
                    FROM runs
                    WHERE runs.id = run_checkpoints.run_id
                      AND runs.account_id = ?
                )
            """
            params.append(account_id)
        if trigger is not None:
            where_clause += " AND trigger = ?"
            params.append(trigger)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM run_checkpoints
                {where_clause}
                ORDER BY created_at DESC, step_sequence DESC
                """,
                tuple(params),
            ).fetchall()
        return [self._checkpoint_from_row(row) for row in rows]

    def create_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        """Persist one artifact."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO artifacts (
                    id, run_id, step_id, artifact_type, uri, display_name, role, format, mime_type,
                    size_bytes, source, is_final, preview_kind, parent_artifact_id, summary,
                    metadata_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.run_id,
                    artifact.step_id,
                    artifact.artifact_type,
                    artifact.uri,
                    artifact.display_name,
                    artifact.role,
                    artifact.format,
                    artifact.mime_type,
                    artifact.size_bytes,
                    artifact.source,
                    int(artifact.is_final),
                    artifact.preview_kind,
                    artifact.parent_artifact_id,
                    artifact.summary,
                    self._json_dump(artifact.metadata),
                    artifact.created_at,
                ),
            )
        return artifact

    def update_artifact(self, artifact: ArtifactRecord) -> ArtifactRecord:
        """Persist the latest state for one artifact."""
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE artifacts
                SET run_id = ?, step_id = ?, artifact_type = ?, uri = ?, display_name = ?, role = ?,
                    format = ?, mime_type = ?, size_bytes = ?, source = ?, is_final = ?,
                    preview_kind = ?, parent_artifact_id = ?, summary = ?, metadata_json = ?,
                    created_at = ?
                WHERE id = ?
                """,
                (
                    artifact.run_id,
                    artifact.step_id,
                    artifact.artifact_type,
                    artifact.uri,
                    artifact.display_name,
                    artifact.role,
                    artifact.format,
                    artifact.mime_type,
                    artifact.size_bytes,
                    artifact.source,
                    int(artifact.is_final),
                    artifact.preview_kind,
                    artifact.parent_artifact_id,
                    artifact.summary,
                    self._json_dump(artifact.metadata),
                    artifact.created_at,
                    artifact.id,
                ),
            ).rowcount
        if updated == 0:
            raise KeyError(f"Artifact not found: {artifact.id}")
        return artifact

    def get_artifact(
        self,
        artifact_id: str,
        *,
        account_id: str | None = None,
    ) -> ArtifactRecord | None:
        """Return one artifact by id if present."""
        params: list[Any] = [artifact_id]
        sql = """
            SELECT artifacts.*
            FROM artifacts
        """
        if account_id is not None:
            sql += """
                INNER JOIN runs
                    ON runs.id = artifacts.run_id
            """
        sql += " WHERE artifacts.id = ?"
        if account_id is not None:
            sql += " AND runs.account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
        if row is None:
            return None
        return self._artifact_from_row(row)

    def list_artifacts(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
    ) -> list[ArtifactRecord]:
        """Return artifacts for one run ordered newest-first."""
        params: list[Any] = [run_id]
        sql = """
            SELECT artifacts.*
            FROM artifacts
        """
        if account_id is not None:
            sql += """
                INNER JOIN runs
                    ON runs.id = artifacts.run_id
            """
        sql += " WHERE artifacts.run_id = ?"
        if account_id is not None:
            sql += " AND runs.account_id = ?"
            params.append(account_id)
        sql += " ORDER BY artifacts.created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(
                sql,
                tuple(params),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]
