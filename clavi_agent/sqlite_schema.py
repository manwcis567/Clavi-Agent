"""Shared SQLite schema helpers for Clavi Agent repositories."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


SESSION_DB_SCOPE = "session_db"
AGENT_DB_SCOPE = "agent_db"
ROOT_ACCOUNT_ID = "root"
CURRENT_SESSION_DB_VERSION = 11
CURRENT_AGENT_DB_VERSION = 10


def utc_now_iso() -> str:
    """Return a compact UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Configure one SQLite connection for repository use."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def ensure_schema_migrations_table(conn: sqlite3.Connection) -> None:
    """Create the schema_migrations bookkeeping table if needed."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            scope TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def set_schema_version(conn: sqlite3.Connection, scope: str, version: int) -> None:
    """Persist the active schema version for one logical DB scope."""
    conn.execute(
        """
        INSERT INTO schema_migrations (scope, version, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(scope) DO UPDATE SET
            version = excluded.version,
            updated_at = excluded.updated_at
        """,
        (scope, version, utc_now_iso()),
    )


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Return whether a table exists."""
    row = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name = ?
        """,
        (table_name,),
    ).fetchone()
    return row is not None


def column_names(conn: sqlite3.Connection, table_name: str) -> set[str]:
    """Return the column names for an existing table."""
    if not table_exists(conn, table_name):
        return set()
    return {
        str(info["name"])
        for info in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def ensure_column(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
) -> None:
    """Add one missing column to an existing table."""
    if column_name in column_names(conn, table_name):
        return
    conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")


def backfill_account_id(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    where_clause: str = "account_id IS NULL OR account_id = ''",
    account_id: str = ROOT_ACCOUNT_ID,
) -> None:
    """Populate missing account ownership for historical rows."""
    if "account_id" not in column_names(conn, table_name):
        return
    conn.execute(
        f"""
        UPDATE {table_name}
        SET account_id = ?
        WHERE {where_clause}
        """,
        (account_id,),
    )


def rebuild_session_history_fts(conn: sqlite3.Connection) -> None:
    """Rebuild the session history FTS index from the current persisted sources."""
    if not table_exists(conn, "session_history_fts"):
        return

    conn.execute("DELETE FROM session_history_fts")
    conn.execute(
        """
        INSERT INTO session_history_fts (
            source_key,
            account_id,
            session_id,
            run_id,
            message_seq,
            created_at,
            source_type,
            role,
            title,
            content
        )
        SELECT
            'message:' || session_messages.id,
            sessions.account_id,
            session_messages.session_id,
            '',
            CAST(session_messages.seq AS TEXT),
            session_messages.created_at,
            'session_message',
            session_messages.role,
            COALESCE(session_messages.name, ''),
            COALESCE(session_messages.search_text, '')
        FROM session_messages
        INNER JOIN sessions ON sessions.session_id = session_messages.session_id
        """
    )
    conn.execute(
        """
        INSERT INTO session_history_fts (
            source_key,
            account_id,
            session_id,
            run_id,
            message_seq,
            created_at,
            source_type,
            role,
            title,
            content
        )
        SELECT
            'run:' || runs.id,
            runs.account_id,
            runs.session_id,
            runs.id,
            '',
            COALESCE(runs.started_at, runs.created_at),
            'run_goal',
            runs.agent_template_id,
            'Run goal',
            COALESCE(runs.goal, '')
        FROM runs
        """
    )
    conn.execute(
        """
        INSERT INTO session_history_fts (
            source_key,
            account_id,
            session_id,
            run_id,
            message_seq,
            created_at,
            source_type,
            role,
            title,
            content
        )
        SELECT
            'run_step:' || run_steps.id,
            runs.account_id,
            runs.session_id,
            run_steps.run_id,
            '',
            COALESCE(run_steps.finished_at, run_steps.started_at, runs.created_at),
            CASE
                WHEN run_steps.step_type = 'completion' THEN 'run_completion'
                ELSE 'run_failure'
            END,
            run_steps.step_type,
            run_steps.title,
            CASE
                WHEN run_steps.step_type = 'completion' THEN COALESCE(run_steps.output_summary, '')
                ELSE COALESCE(run_steps.error_summary, '')
            END
        FROM run_steps
        INNER JOIN runs ON runs.id = run_steps.run_id
        WHERE run_steps.step_type IN ('completion', 'failure')
          AND TRIM(
              CASE
                  WHEN run_steps.step_type = 'completion' THEN COALESCE(run_steps.output_summary, '')
                  ELSE COALESCE(run_steps.error_summary, '')
              END
          ) <> ''
        """
    )
    if table_exists(conn, "shared_context_entries"):
        conn.execute(
            """
            INSERT INTO session_history_fts (
                source_key,
                account_id,
                session_id,
                run_id,
                message_seq,
                created_at,
                source_type,
                role,
                title,
                content
            )
            SELECT
                'shared_context:' || shared_context_entries.id,
                shared_context_entries.account_id,
                shared_context_entries.session_id,
                COALESCE(shared_context_entries.run_id, ''),
                '',
                shared_context_entries.created_at,
                'shared_context',
                COALESCE(shared_context_entries.category, ''),
                COALESCE(shared_context_entries.title, ''),
                COALESCE(shared_context_entries.content, '')
            FROM shared_context_entries
            WHERE TRIM(COALESCE(shared_context_entries.content, '')) <> ''
               OR TRIM(COALESCE(shared_context_entries.title, '')) <> ''
            """
        )


def ensure_session_db_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate the shared session/run SQLite schema."""
    ensure_schema_migrations_table(conn)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            workspace_dir TEXT NOT NULL,
            agent_id TEXT,
            ui_state_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0,
            last_message_preview TEXT NOT NULL DEFAULT ''
        )
        """
    )

    if table_exists(conn, "messages") and not table_exists(conn, "session_messages"):
        conn.execute("ALTER TABLE messages RENAME TO session_messages")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            seq INTEGER NOT NULL,
            role TEXT NOT NULL,
            content_json TEXT NOT NULL,
            thinking TEXT,
            tool_calls_json TEXT,
            tool_call_id TEXT,
            name TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
        """
    )
    ensure_column(
        conn,
        "session_messages",
        "search_text",
        "search_text TEXT NOT NULL DEFAULT ''",
    )

    ensure_column(conn, "sessions", "agent_id", "agent_id TEXT")
    ensure_column(conn, "sessions", "ui_state_json", "ui_state_json TEXT NOT NULL DEFAULT '{}'")
    ensure_column(
        conn,
        "sessions",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "sessions")

    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_session_messages_session_seq
        ON session_messages(session_id, seq)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sessions_updated_at
        ON sessions(updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sessions_account_id
        ON sessions(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_sessions_account_updated_at
        ON sessions(account_id, updated_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            agent_template_id TEXT NOT NULL,
            agent_template_snapshot_json TEXT NOT NULL,
            status TEXT NOT NULL,
            goal TEXT NOT NULL,
            trigger_message_ref_json TEXT,
            parent_run_id TEXT,
            run_metadata_json TEXT NOT NULL DEFAULT '{}',
            deliverable_manifest_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            current_step_index INTEGER NOT NULL DEFAULT 0,
            last_checkpoint_at TEXT,
            error_summary TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (parent_run_id) REFERENCES runs(id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(
        conn,
        "runs",
        "run_metadata_json",
        "run_metadata_json TEXT NOT NULL DEFAULT '{}'",
    )
    ensure_column(
        conn,
        "runs",
        "deliverable_manifest_json",
        "deliverable_manifest_json TEXT NOT NULL DEFAULT '{}'",
    )
    ensure_column(
        conn,
        "runs",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "runs")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_session_created_at
        ON runs(session_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_status_created_at
        ON runs(status, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_parent_run_id
        ON runs(parent_run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_account_id
        ON runs(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_runs_account_status_created_at
        ON runs(account_id, status, created_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_steps (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            step_type TEXT NOT NULL,
            status TEXT NOT NULL,
            title TEXT NOT NULL,
            input_summary TEXT NOT NULL DEFAULT '',
            output_summary TEXT NOT NULL DEFAULT '',
            started_at TEXT,
            finished_at TEXT,
            error_summary TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_run_steps_run_sequence
        ON run_steps(run_id, sequence)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_run_steps_run_status
        ON run_steps(run_id, status, sequence)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS shared_context_entries (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            run_id TEXT NOT NULL DEFAULT '',
            parent_run_id TEXT NOT NULL DEFAULT '',
            root_run_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL DEFAULT 'general',
            title TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shared_context_entries_session_created
        ON shared_context_entries(session_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_shared_context_entries_account_created
        ON shared_context_entries(account_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS learned_workflow_candidates (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL DEFAULT 'root',
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            agent_template_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending_review'
                CHECK(status IN ('pending_review', 'approved', 'rejected', 'installed')),
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            signal_types_json TEXT NOT NULL DEFAULT '[]',
            source_run_ids_json TEXT NOT NULL DEFAULT '[]',
            tool_names_json TEXT NOT NULL DEFAULT '[]',
            step_titles_json TEXT NOT NULL DEFAULT '[]',
            artifact_ids_json TEXT NOT NULL DEFAULT '[]',
            suggested_skill_name TEXT NOT NULL DEFAULT '',
            generated_skill_markdown TEXT NOT NULL DEFAULT '',
            review_notes TEXT NOT NULL DEFAULT '',
            installed_agent_id TEXT,
            installed_skill_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            approved_at TEXT,
            rejected_at TEXT,
            installed_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_learned_workflow_candidates_run
        ON learned_workflow_candidates(run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_learned_workflow_candidates_account_status_updated
        ON learned_workflow_candidates(account_id, status, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_learned_workflow_candidates_agent_updated
        ON learned_workflow_candidates(agent_template_id, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS skill_improvement_proposals (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL DEFAULT 'root',
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            agent_template_id TEXT NOT NULL,
            skill_name TEXT NOT NULL,
            target_skill_path TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending_review'
                CHECK(status IN ('pending_review', 'approved', 'rejected', 'applied')),
            title TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            signal_types_json TEXT NOT NULL DEFAULT '[]',
            source_run_ids_json TEXT NOT NULL DEFAULT '[]',
            base_version INTEGER NOT NULL DEFAULT 1,
            proposed_version INTEGER NOT NULL DEFAULT 2,
            current_skill_markdown TEXT NOT NULL DEFAULT '',
            proposed_skill_markdown TEXT NOT NULL DEFAULT '',
            changelog_entry TEXT NOT NULL DEFAULT '',
            review_notes TEXT NOT NULL DEFAULT '',
            applied_skill_path TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            approved_at TEXT,
            rejected_at TEXT,
            applied_at TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_skill_improvement_proposals_run_skill
        ON skill_improvement_proposals(run_id, skill_name)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_skill_improvement_proposals_account_status_updated
        ON skill_improvement_proposals(account_id, status, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_skill_improvement_proposals_agent_skill_updated
        ON skill_improvement_proposals(agent_template_id, skill_name, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS session_history_fts USING fts5(
            source_key UNINDEXED,
            account_id UNINDEXED,
            session_id UNINDEXED,
            run_id UNINDEXED,
            message_seq UNINDEXED,
            created_at UNINDEXED,
            source_type UNINDEXED,
            role,
            title,
            content,
            tokenize = 'unicode61'
        )
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_message_insert
        AFTER INSERT ON session_messages
        BEGIN
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            VALUES (
                'message:' || NEW.id,
                COALESCE((SELECT account_id FROM sessions WHERE session_id = NEW.session_id), 'root'),
                NEW.session_id,
                '',
                CAST(NEW.seq AS TEXT),
                NEW.created_at,
                'session_message',
                NEW.role,
                COALESCE(NEW.name, ''),
                COALESCE(NEW.search_text, '')
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_message_update
        AFTER UPDATE ON session_messages
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'message:' || OLD.id;
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            VALUES (
                'message:' || NEW.id,
                COALESCE((SELECT account_id FROM sessions WHERE session_id = NEW.session_id), 'root'),
                NEW.session_id,
                '',
                CAST(NEW.seq AS TEXT),
                NEW.created_at,
                'session_message',
                NEW.role,
                COALESCE(NEW.name, ''),
                COALESCE(NEW.search_text, '')
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_message_delete
        AFTER DELETE ON session_messages
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'message:' || OLD.id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_run_insert
        AFTER INSERT ON runs
        BEGIN
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            VALUES (
                'run:' || NEW.id,
                NEW.account_id,
                NEW.session_id,
                NEW.id,
                '',
                COALESCE(NEW.started_at, NEW.created_at),
                'run_goal',
                NEW.agent_template_id,
                'Run goal',
                COALESCE(NEW.goal, '')
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_run_update
        AFTER UPDATE ON runs
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'run:' || OLD.id;
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            VALUES (
                'run:' || NEW.id,
                NEW.account_id,
                NEW.session_id,
                NEW.id,
                '',
                COALESCE(NEW.started_at, NEW.created_at),
                'run_goal',
                NEW.agent_template_id,
                'Run goal',
                COALESCE(NEW.goal, '')
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_run_delete
        AFTER DELETE ON runs
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'run:' || OLD.id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_run_step_insert
        AFTER INSERT ON run_steps
        WHEN NEW.step_type IN ('completion', 'failure')
             AND TRIM(
                 CASE
                     WHEN NEW.step_type = 'completion' THEN COALESCE(NEW.output_summary, '')
                     ELSE COALESCE(NEW.error_summary, '')
                 END
             ) <> ''
        BEGIN
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            SELECT
                'run_step:' || NEW.id,
                runs.account_id,
                runs.session_id,
                NEW.run_id,
                '',
                COALESCE(NEW.finished_at, NEW.started_at, runs.created_at),
                CASE
                    WHEN NEW.step_type = 'completion' THEN 'run_completion'
                    ELSE 'run_failure'
                END,
                NEW.step_type,
                NEW.title,
                CASE
                    WHEN NEW.step_type = 'completion' THEN COALESCE(NEW.output_summary, '')
                    ELSE COALESCE(NEW.error_summary, '')
                END
            FROM runs
            WHERE runs.id = NEW.run_id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_run_step_update
        AFTER UPDATE ON run_steps
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'run_step:' || OLD.id;
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            SELECT
                'run_step:' || NEW.id,
                runs.account_id,
                runs.session_id,
                NEW.run_id,
                '',
                COALESCE(NEW.finished_at, NEW.started_at, runs.created_at),
                CASE
                    WHEN NEW.step_type = 'completion' THEN 'run_completion'
                    ELSE 'run_failure'
                END,
                NEW.step_type,
                NEW.title,
                CASE
                    WHEN NEW.step_type = 'completion' THEN COALESCE(NEW.output_summary, '')
                    ELSE COALESCE(NEW.error_summary, '')
                END
            FROM runs
            WHERE runs.id = NEW.run_id
              AND NEW.step_type IN ('completion', 'failure')
              AND TRIM(
                  CASE
                      WHEN NEW.step_type = 'completion' THEN COALESCE(NEW.output_summary, '')
                      ELSE COALESCE(NEW.error_summary, '')
                  END
              ) <> '';
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_run_step_delete
        AFTER DELETE ON run_steps
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'run_step:' || OLD.id;
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_shared_context_insert
        AFTER INSERT ON shared_context_entries
        WHEN TRIM(COALESCE(NEW.content, '')) <> '' OR TRIM(COALESCE(NEW.title, '')) <> ''
        BEGIN
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            VALUES (
                'shared_context:' || NEW.id,
                NEW.account_id,
                NEW.session_id,
                COALESCE(NEW.run_id, ''),
                '',
                NEW.created_at,
                'shared_context',
                COALESCE(NEW.category, ''),
                COALESCE(NEW.title, ''),
                COALESCE(NEW.content, '')
            );
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_shared_context_update
        AFTER UPDATE ON shared_context_entries
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'shared_context:' || OLD.id;
            INSERT INTO session_history_fts (
                source_key, account_id, session_id, run_id, message_seq,
                created_at, source_type, role, title, content
            )
            SELECT
                'shared_context:' || NEW.id,
                NEW.account_id,
                NEW.session_id,
                COALESCE(NEW.run_id, ''),
                '',
                NEW.created_at,
                'shared_context',
                COALESCE(NEW.category, ''),
                COALESCE(NEW.title, ''),
                COALESCE(NEW.content, '')
            WHERE TRIM(COALESCE(NEW.content, '')) <> '' OR TRIM(COALESCE(NEW.title, '')) <> '';
        END
        """
    )
    conn.execute(
        """
        CREATE TRIGGER IF NOT EXISTS trg_session_history_fts_shared_context_delete
        AFTER DELETE ON shared_context_entries
        BEGIN
            DELETE FROM session_history_fts WHERE source_key = 'shared_context:' || OLD.id;
        END
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_checkpoints (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            step_sequence INTEGER NOT NULL DEFAULT 0,
            trigger TEXT NOT NULL DEFAULT 'llm_response',
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
        )
        """
    )
    ensure_column(
        conn,
        "run_checkpoints",
        "trigger",
        "trigger TEXT NOT NULL DEFAULT 'llm_response'",
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_run_checkpoints_run_created_at
        ON run_checkpoints(run_id, created_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifacts (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            step_id TEXT,
            artifact_type TEXT NOT NULL,
            uri TEXT NOT NULL,
            display_name TEXT NOT NULL DEFAULT '',
            role TEXT NOT NULL DEFAULT 'intermediate_file',
            format TEXT NOT NULL DEFAULT '',
            mime_type TEXT NOT NULL DEFAULT '',
            size_bytes INTEGER,
            source TEXT NOT NULL DEFAULT 'agent_generated',
            is_final INTEGER NOT NULL DEFAULT 0,
            preview_kind TEXT NOT NULL DEFAULT 'none',
            parent_artifact_id TEXT,
            summary TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (step_id) REFERENCES run_steps(id) ON DELETE SET NULL,
            FOREIGN KEY (parent_artifact_id) REFERENCES artifacts(id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(
        conn,
        "artifacts",
        "display_name",
        "display_name TEXT NOT NULL DEFAULT ''",
    )
    ensure_column(
        conn,
        "artifacts",
        "role",
        "role TEXT NOT NULL DEFAULT 'intermediate_file'",
    )
    ensure_column(
        conn,
        "artifacts",
        "format",
        "format TEXT NOT NULL DEFAULT ''",
    )
    ensure_column(
        conn,
        "artifacts",
        "mime_type",
        "mime_type TEXT NOT NULL DEFAULT ''",
    )
    ensure_column(
        conn,
        "artifacts",
        "size_bytes",
        "size_bytes INTEGER",
    )
    ensure_column(
        conn,
        "artifacts",
        "source",
        "source TEXT NOT NULL DEFAULT 'agent_generated'",
    )
    ensure_column(
        conn,
        "artifacts",
        "is_final",
        "is_final INTEGER NOT NULL DEFAULT 0",
    )
    ensure_column(
        conn,
        "artifacts",
        "preview_kind",
        "preview_kind TEXT NOT NULL DEFAULT 'none'",
    )
    ensure_column(
        conn,
        "artifacts",
        "parent_artifact_id",
        "parent_artifact_id TEXT",
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifacts_run_created_at
        ON artifacts(run_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_artifacts_run_is_final_created_at
        ON artifacts(run_id, is_final, created_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS uploads (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            run_id TEXT,
            original_name TEXT NOT NULL,
            safe_name TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            absolute_path TEXT NOT NULL,
            mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
            size_bytes INTEGER NOT NULL,
            checksum TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            created_by TEXT NOT NULL DEFAULT 'user',
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(
        conn,
        "uploads",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "uploads")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_uploads_session_created_at
        ON uploads(session_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_uploads_run_created_at
        ON uploads(run_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_uploads_account_id
        ON uploads(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_uploads_account_created_at
        ON uploads(account_id, created_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS approval_requests (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            step_id TEXT,
            tool_name TEXT NOT NULL,
            risk_level TEXT NOT NULL,
            status TEXT NOT NULL,
            parameter_summary TEXT NOT NULL DEFAULT '',
            impact_summary TEXT NOT NULL DEFAULT '',
            requested_at TEXT NOT NULL,
            resolved_at TEXT,
            decision_notes TEXT NOT NULL DEFAULT '',
            decision_scope TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (step_id) REFERENCES run_steps(id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(
        conn,
        "approval_requests",
        "decision_scope",
        "decision_scope TEXT",
    )
    ensure_column(
        conn,
        "approval_requests",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "approval_requests")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_approval_requests_status_requested_at
        ON approval_requests(status, requested_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_approval_requests_run_status
        ON approval_requests(run_id, status, requested_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_approval_requests_account_id
        ON approval_requests(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_approval_requests_account_status_requested_at
        ON approval_requests(account_id, status, requested_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trace_events (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            parent_run_id TEXT,
            step_id TEXT,
            sequence INTEGER NOT NULL DEFAULT 0,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT '',
            payload_summary TEXT NOT NULL DEFAULT '',
            duration_ms INTEGER,
            created_at TEXT NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE,
            FOREIGN KEY (parent_run_id) REFERENCES runs(id) ON DELETE SET NULL,
            FOREIGN KEY (step_id) REFERENCES run_steps(id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(
        conn,
        "trace_events",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "trace_events")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trace_events_run_created_at
        ON trace_events(run_id, created_at ASC, sequence ASC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trace_events_parent_run
        ON trace_events(parent_run_id, created_at ASC, sequence ASC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trace_events_account_id
        ON trace_events(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_trace_events_account_status_created_at
        ON trace_events(account_id, status, created_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS integrations (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL DEFAULT 'root',
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'disabled',
            display_name TEXT NOT NULL DEFAULT '',
            tenant_id TEXT NOT NULL DEFAULT '',
            webhook_path TEXT NOT NULL,
            config_json TEXT NOT NULL DEFAULT '{}',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_verified_at TEXT,
            last_error TEXT NOT NULL DEFAULT ''
        )
        """
    )
    ensure_column(
        conn,
        "integrations",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "integrations")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integrations_kind_status
        ON integrations(kind, status, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integrations_updated_at
        ON integrations(updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integrations_account_id
        ON integrations(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integrations_account_status_updated_at
        ON integrations(account_id, status, updated_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS integration_credentials (
            id TEXT PRIMARY KEY,
            integration_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            credential_key TEXT NOT NULL,
            storage_kind TEXT NOT NULL,
            secret_ref TEXT NOT NULL DEFAULT '',
            secret_ciphertext TEXT NOT NULL DEFAULT '',
            masked_value TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (integration_id) REFERENCES integrations(id)
        )
        """
    )
    ensure_column(
        conn,
        "integration_credentials",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "integration_credentials")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_integration_credentials_integration_key
        ON integration_credentials(integration_id, credential_key)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integration_credentials_account_id
        ON integration_credentials(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_integration_credentials_account_updated_at
        ON integration_credentials(account_id, updated_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inbound_events (
            id TEXT PRIMARY KEY,
            integration_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            provider_event_id TEXT,
            provider_message_id TEXT,
            provider_chat_id TEXT NOT NULL DEFAULT '',
            provider_thread_id TEXT NOT NULL DEFAULT '',
            provider_user_id TEXT NOT NULL DEFAULT '',
            event_type TEXT NOT NULL DEFAULT 'message',
            received_at TEXT NOT NULL,
            signature_valid INTEGER NOT NULL DEFAULT 0,
            dedup_key TEXT NOT NULL DEFAULT '',
            raw_headers_json TEXT NOT NULL DEFAULT '{}',
            raw_headers_size_bytes INTEGER NOT NULL DEFAULT 0,
            raw_headers_truncated INTEGER NOT NULL DEFAULT 0,
            raw_headers_redacted_fields_json TEXT NOT NULL DEFAULT '[]',
            raw_payload_json TEXT NOT NULL DEFAULT '{}',
            raw_payload_size_bytes INTEGER NOT NULL DEFAULT 0,
            raw_payload_truncated INTEGER NOT NULL DEFAULT 0,
            raw_payload_redacted_fields_json TEXT NOT NULL DEFAULT '[]',
            normalized_status TEXT NOT NULL DEFAULT 'received',
            normalized_error TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            FOREIGN KEY (integration_id) REFERENCES integrations(id)
        )
        """
    )
    ensure_column(
        conn,
        "inbound_events",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "inbound_events")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_inbound_events_integration_received_at
        ON inbound_events(integration_id, received_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_inbound_events_integration_chat_thread
        ON inbound_events(integration_id, provider_chat_id, provider_thread_id, received_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_inbound_events_integration_message_id
        ON inbound_events(integration_id, provider_message_id)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_inbound_events_integration_provider_event_id
        ON inbound_events(integration_id, provider_event_id)
        WHERE provider_event_id IS NOT NULL AND provider_event_id != ''
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_inbound_events_integration_dedup_key
        ON inbound_events(integration_id, dedup_key)
        WHERE dedup_key != ''
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_inbound_events_account_id
        ON inbound_events(account_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_inbound_events_account_status_received_at
        ON inbound_events(account_id, normalized_status, received_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_bindings (
            id TEXT PRIMARY KEY,
            integration_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            tenant_id TEXT NOT NULL DEFAULT '',
            chat_id TEXT NOT NULL DEFAULT '',
            thread_id TEXT NOT NULL DEFAULT '',
            binding_scope TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_message_at TEXT,
            FOREIGN KEY (integration_id) REFERENCES integrations(id)
        )
        """
    )
    ensure_column(
        conn,
        "conversation_bindings",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "conversation_bindings")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_bindings_unique_scope_agent
        ON conversation_bindings(
            integration_id,
            tenant_id,
            chat_id,
            thread_id,
            binding_scope,
            agent_id
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_bindings_lookup
        ON conversation_bindings(
            integration_id,
            tenant_id,
            chat_id,
            thread_id,
            enabled,
            updated_at DESC
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_bindings_session
        ON conversation_bindings(session_id, enabled, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_conversation_bindings_account_updated_at
        ON conversation_bindings(account_id, updated_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS routing_rules (
            id TEXT PRIMARY KEY,
            integration_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            priority INTEGER NOT NULL DEFAULT 100,
            match_type TEXT NOT NULL,
            match_value TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            session_strategy TEXT NOT NULL DEFAULT 'reuse',
            enabled INTEGER NOT NULL DEFAULT 1,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (integration_id) REFERENCES integrations(id)
        )
        """
    )
    ensure_column(
        conn,
        "routing_rules",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "routing_rules")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_routing_rules_integration_priority
        ON routing_rules(integration_id, enabled, priority ASC, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_routing_rules_match
        ON routing_rules(integration_id, match_type, match_value, agent_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_routing_rules_account_updated_at
        ON routing_rules(account_id, updated_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS outbound_deliveries (
            id TEXT PRIMARY KEY,
            integration_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            inbound_event_id TEXT,
            provider_chat_id TEXT NOT NULL,
            provider_thread_id TEXT NOT NULL DEFAULT '',
            provider_message_id TEXT NOT NULL DEFAULT '',
            delivery_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'pending',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            last_attempt_at TEXT,
            error_summary TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (integration_id) REFERENCES integrations(id),
            FOREIGN KEY (inbound_event_id) REFERENCES inbound_events(id)
        )
        """
    )
    ensure_column(
        conn,
        "outbound_deliveries",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "outbound_deliveries")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_outbound_deliveries_integration_status_updated
        ON outbound_deliveries(integration_id, status, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_outbound_deliveries_run_created
        ON outbound_deliveries(run_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_outbound_deliveries_session_created
        ON outbound_deliveries(session_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_outbound_deliveries_account_status_updated
        ON outbound_deliveries(account_id, status, updated_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS delivery_attempts (
            id TEXT PRIMARY KEY,
            delivery_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            attempt_number INTEGER NOT NULL,
            status TEXT NOT NULL,
            request_payload_json TEXT NOT NULL DEFAULT '{}',
            response_payload_json TEXT NOT NULL DEFAULT '{}',
            error_summary TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            finished_at TEXT,
            FOREIGN KEY (delivery_id) REFERENCES outbound_deliveries(id) ON DELETE CASCADE
        )
        """
    )
    ensure_column(
        conn,
        "delivery_attempts",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "delivery_attempts")
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_delivery_attempts_delivery_attempt
        ON delivery_attempts(delivery_id, attempt_number)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_delivery_attempts_status_started
        ON delivery_attempts(status, started_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_delivery_attempts_account_status_started
        ON delivery_attempts(account_id, status, started_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL DEFAULT 'root',
            name TEXT NOT NULL,
            cron_expression TEXT NOT NULL,
            timezone TEXT NOT NULL DEFAULT 'server_local',
            agent_id TEXT NOT NULL,
            prompt TEXT NOT NULL,
            integration_id TEXT,
            target_chat_id TEXT NOT NULL DEFAULT '',
            target_thread_id TEXT NOT NULL DEFAULT '',
            reply_to_message_id TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            session_id TEXT,
            next_run_at TEXT,
            last_scheduled_for TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (integration_id) REFERENCES integrations(id) ON DELETE SET NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(
        conn,
        "scheduled_tasks",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "scheduled_tasks")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_enabled_next_run
        ON scheduled_tasks(enabled, next_run_at ASC, updated_at ASC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_agent_updated
        ON scheduled_tasks(agent_id, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_tasks_account_updated
        ON scheduled_tasks(account_id, updated_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scheduled_task_executions (
            id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL,
            account_id TEXT NOT NULL DEFAULT 'root',
            trigger_kind TEXT NOT NULL,
            scheduled_for TEXT,
            run_id TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            error_summary TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id) ON DELETE CASCADE,
            FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE SET NULL
        )
        """
    )
    ensure_column(
        conn,
        "scheduled_task_executions",
        "account_id",
        f"account_id TEXT NOT NULL DEFAULT '{ROOT_ACCOUNT_ID}'",
    )
    backfill_account_id(conn, "scheduled_task_executions")
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_task_executions_task_created
        ON scheduled_task_executions(task_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_task_executions_run
        ON scheduled_task_executions(run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_task_executions_status_updated
        ON scheduled_task_executions(status, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_scheduled_task_executions_account_status_updated
        ON scheduled_task_executions(account_id, status, updated_at DESC)
        """
    )

    set_schema_version(conn, SESSION_DB_SCOPE, CURRENT_SESSION_DB_VERSION)


def ensure_agent_db_schema(conn: sqlite3.Connection) -> None:
    """Create or migrate the agent-template SQLite schema."""
    ensure_schema_migrations_table(conn)

    if table_exists(conn, "agents") and not table_exists(conn, "agent_templates"):
        conn.execute("ALTER TABLE agents RENAME TO agent_templates")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS accounts (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            display_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active'
                CHECK(status IN ('active', 'disabled')),
            is_root INTEGER NOT NULL DEFAULT 0
                CHECK(is_root IN (0, 1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_accounts_username
        ON accounts(username)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_accounts_status_updated_at
        ON accounts(status, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_accounts_is_root
        ON accounts(is_root)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_password_credentials (
            account_id TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            password_algo TEXT NOT NULL,
            password_updated_at TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_web_sessions (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            session_token_hash TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            user_agent TEXT NOT NULL DEFAULT '',
            ip TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_account_web_sessions_token_hash
        ON account_web_sessions(session_token_hash)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_account_web_sessions_account_last_seen
        ON account_web_sessions(account_id, last_seen_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_account_web_sessions_expires_at
        ON account_web_sessions(expires_at ASC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_api_configs (
            id TEXT PRIMARY KEY,
            account_id TEXT NOT NULL,
            name TEXT NOT NULL,
            provider TEXT NOT NULL
                CHECK(provider IN ('anthropic', 'openai')),
            api_base TEXT NOT NULL DEFAULT 'https://api.minimax.io',
            model TEXT NOT NULL DEFAULT 'MiniMax-M2',
            api_key TEXT NOT NULL,
            reasoning_enabled INTEGER NOT NULL DEFAULT 0
                CHECK(reasoning_enabled IN (0, 1)),
            llm_routing_policy_json TEXT NOT NULL DEFAULT '{}',
            is_active INTEGER NOT NULL DEFAULT 0
                CHECK(is_active IN (0, 1)),
            last_used_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (account_id) REFERENCES accounts(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_account_api_configs_account_name
        ON account_api_configs(account_id, name)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_account_api_configs_account_updated_at
        ON account_api_configs(account_id, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_account_api_configs_account_active
        ON account_api_configs(account_id, is_active, updated_at DESC)
        """
    )
    ensure_column(
        conn,
        "account_api_configs",
        "llm_routing_policy_json",
        "llm_routing_policy_json TEXT NOT NULL DEFAULT '{}'",
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            profile_json TEXT NOT NULL DEFAULT '{}',
            summary TEXT NOT NULL DEFAULT '',
            writer_type TEXT NOT NULL DEFAULT 'system',
            writer_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES accounts(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_profiles_updated_at
        ON user_profiles(updated_at DESC)
        """
    )
    ensure_column(
        conn,
        "user_profiles",
        "writer_type",
        "writer_type TEXT NOT NULL DEFAULT 'system'",
    )
    ensure_column(
        conn,
        "user_profiles",
        "writer_id",
        "writer_id TEXT NOT NULL DEFAULT ''",
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_memory_entries (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            memory_type TEXT NOT NULL
                CHECK(memory_type IN (
                    'preference',
                    'communication_style',
                    'goal',
                    'constraint',
                    'project_fact',
                    'workflow_fact',
                    'correction'
                )),
            content TEXT NOT NULL,
            summary TEXT NOT NULL DEFAULT '',
            source_session_id TEXT,
            source_run_id TEXT,
            writer_type TEXT NOT NULL DEFAULT 'system',
            writer_id TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.5
                CHECK(confidence >= 0 AND confidence <= 1),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            superseded_by TEXT,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            deleted_at TEXT,
            deleted_reason TEXT NOT NULL DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES accounts(id) ON DELETE CASCADE,
            FOREIGN KEY (superseded_by) REFERENCES user_memory_entries(id) ON DELETE SET NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_memory_entries_user_updated_at
        ON user_memory_entries(user_id, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_memory_entries_user_type_updated_at
        ON user_memory_entries(user_id, memory_type, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_memory_entries_user_session
        ON user_memory_entries(user_id, source_session_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_memory_entries_user_run
        ON user_memory_entries(user_id, source_run_id)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_user_memory_entries_superseded_by
        ON user_memory_entries(superseded_by)
        """
    )
    ensure_column(
        conn,
        "user_memory_entries",
        "writer_type",
        "writer_type TEXT NOT NULL DEFAULT 'system'",
    )
    ensure_column(
        conn,
        "user_memory_entries",
        "writer_id",
        "writer_id TEXT NOT NULL DEFAULT ''",
    )
    ensure_column(
        conn,
        "user_memory_entries",
        "is_deleted",
        "is_deleted INTEGER NOT NULL DEFAULT 0",
    )
    ensure_column(
        conn,
        "user_memory_entries",
        "deleted_at",
        "deleted_at TEXT",
    )
    ensure_column(
        conn,
        "user_memory_entries",
        "deleted_reason",
        "deleted_reason TEXT NOT NULL DEFAULT ''",
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_audit_events (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            target_scope TEXT NOT NULL,
            target_id TEXT NOT NULL,
            action TEXT NOT NULL,
            writer_type TEXT NOT NULL DEFAULT 'system',
            writer_id TEXT NOT NULL DEFAULT '',
            session_id TEXT,
            run_id TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES accounts(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_audit_events_user_created_at
        ON memory_audit_events(user_id, created_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_memory_audit_events_target
        ON memory_audit_events(target_scope, target_id, created_at DESC)
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS agent_templates (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT NOT NULL DEFAULT '',
            system_prompt TEXT NOT NULL,
            skills_json TEXT NOT NULL DEFAULT '[]',
            tools_json TEXT NOT NULL DEFAULT '[]',
            mcp_configs_json TEXT NOT NULL DEFAULT '[]',
            workspace_type TEXT NOT NULL DEFAULT 'isolated',
            workspace_policy_json TEXT NOT NULL DEFAULT '{}',
            approval_policy_json TEXT NOT NULL DEFAULT '{}',
            run_policy_json TEXT NOT NULL DEFAULT '{}',
            delegation_policy_json TEXT NOT NULL DEFAULT '{}',
            llm_routing_policy_json TEXT NOT NULL DEFAULT '{}',
            template_version INTEGER NOT NULL DEFAULT 1,
            is_system BOOLEAN NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    ensure_column(
        conn,
        "agent_templates",
        "skills_json",
        "skills_json TEXT NOT NULL DEFAULT '[]'",
    )
    ensure_column(
        conn,
        "agent_templates",
        "workspace_policy_json",
        "workspace_policy_json TEXT NOT NULL DEFAULT '{}'",
    )
    ensure_column(
        conn,
        "agent_templates",
        "approval_policy_json",
        "approval_policy_json TEXT NOT NULL DEFAULT '{}'",
    )
    ensure_column(
        conn,
        "agent_templates",
        "run_policy_json",
        "run_policy_json TEXT NOT NULL DEFAULT '{}'",
    )
    ensure_column(
        conn,
        "agent_templates",
        "delegation_policy_json",
        "delegation_policy_json TEXT NOT NULL DEFAULT '{}'",
    )
    ensure_column(
        conn,
        "agent_templates",
        "llm_routing_policy_json",
        "llm_routing_policy_json TEXT NOT NULL DEFAULT '{}'",
    )
    ensure_column(
        conn,
        "agent_templates",
        "template_version",
        "template_version INTEGER NOT NULL DEFAULT 1",
    )
    ensure_column(
        conn,
        "agent_templates",
        "account_id",
        "account_id TEXT",
    )
    backfill_account_id(
        conn,
        "agent_templates",
        where_clause="is_system = 0 AND (account_id IS NULL OR account_id = '')",
    )
    conn.execute(
        """
        UPDATE agent_templates
        SET account_id = NULL
        WHERE is_system = 1
        """
    )

    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_templates_updated_at
        ON agent_templates(updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_templates_is_system
        ON agent_templates(is_system)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_agent_templates_account_updated_at
        ON agent_templates(account_id, updated_at DESC)
        """
    )

    set_schema_version(conn, AGENT_DB_SCOPE, CURRENT_AGENT_DB_VERSION)

