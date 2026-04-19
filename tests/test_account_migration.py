import sqlite3
from pathlib import Path

from clavi_agent.account_constants import ROOT_ACCOUNT_ID
from clavi_agent.account_migration import (
    format_root_migration_report,
    migrate_historical_data_to_root,
)
from clavi_agent.sqlite_schema import (
    AGENT_DB_SCOPE,
    CURRENT_AGENT_DB_VERSION,
    CURRENT_SESSION_DB_VERSION,
    SESSION_DB_SCOPE,
    configure_connection,
)


def _create_legacy_session_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                workspace_dir TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER NOT NULL DEFAULT 0,
                last_message_preview TEXT NOT NULL DEFAULT ''
            );

            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                agent_template_id TEXT NOT NULL,
                agent_template_snapshot_json TEXT NOT NULL,
                status TEXT NOT NULL,
                goal TEXT NOT NULL,
                trigger_message_ref_json TEXT,
                parent_run_id TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE uploads (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                run_id TEXT,
                original_name TEXT NOT NULL,
                safe_name TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                absolute_path TEXT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                size_bytes INTEGER NOT NULL,
                checksum TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                created_by TEXT NOT NULL DEFAULT 'user'
            );

            INSERT INTO sessions (
                session_id, title, workspace_dir, created_at, updated_at, message_count, last_message_preview
            ) VALUES (
                'session-1', 'Legacy Session', 'D:/legacy', '2026-04-15T00:00:00+00:00',
                '2026-04-15T00:00:00+00:00', 1, 'legacy preview'
            );

            INSERT INTO runs (
                id, session_id, agent_template_id, agent_template_snapshot_json, status, goal, created_at
            ) VALUES (
                'run-1', 'session-1', 'agent-1', '{}', 'completed', 'legacy goal', '2026-04-15T00:01:00+00:00'
            );

            INSERT INTO uploads (
                id, session_id, run_id, original_name, safe_name, relative_path, absolute_path,
                mime_type, size_bytes, checksum, created_at, created_by
            ) VALUES (
                'upload-1', 'session-1', 'run-1', 'legacy.txt', 'legacy.txt',
                '.clavi_agent/uploads/session-1/upload-1/legacy.txt', 'D:/legacy/legacy.txt',
                'text/plain', 10, 'sum', '2026-04-15T00:02:00+00:00', 'user'
            );
            """
        )


def _create_legacy_agent_db(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE agent_templates (
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
                template_version INTEGER NOT NULL DEFAULT 1,
                is_system BOOLEAN NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            INSERT INTO agent_templates (
                id, name, description, system_prompt, template_version, is_system, created_at, updated_at
            ) VALUES (
                'custom-agent', 'Custom Agent', '', 'custom prompt', 1, 0,
                '2026-04-15T00:00:00+00:00', '2026-04-15T00:00:00+00:00'
            );

            INSERT INTO agent_templates (
                id, name, description, system_prompt, template_version, is_system, created_at, updated_at
            ) VALUES (
                'system-agent', 'System Agent', '', 'system prompt', 1, 1,
                '2026-04-15T00:00:00+00:00', '2026-04-15T00:00:00+00:00'
            );
            """
        )


def test_migrate_historical_data_to_root_creates_backup_and_reports(tmp_path: Path):
    session_db_path = tmp_path / "sessions.db"
    agent_db_path = tmp_path / "agents.db"
    backup_dir = tmp_path / "backups"

    _create_legacy_session_db(session_db_path)
    _create_legacy_agent_db(agent_db_path)

    report = migrate_historical_data_to_root(
        session_db_path=session_db_path,
        agent_db_path=agent_db_path,
        root_password="RootPass123!",
        backup_dir=backup_dir,
    )

    assert report.session_backup_path is not None
    assert report.agent_backup_path is not None
    assert Path(report.session_backup_path).exists()
    assert Path(report.agent_backup_path).exists()
    assert report.root_account_created is True
    assert report.root_credential_created is True
    assert report.bootstrap_root_password is None

    report_by_table = {item.table_name: item for item in report.table_reports}
    assert report_by_table["sessions"].total_rows == 1
    assert report_by_table["sessions"].backfilled_rows == 1
    assert report_by_table["sessions"].empty_account_rows == 0
    assert report_by_table["runs"].backfilled_rows == 1
    assert report_by_table["uploads"].backfilled_rows == 1
    assert report_by_table["agent_templates"].total_rows == 2
    assert report_by_table["agent_templates"].backfilled_rows == 1
    assert report_by_table["agent_templates"].empty_account_rows == 0

    with configure_connection(sqlite3.connect(session_db_path)) as conn:
        session_row = conn.execute(
            "SELECT account_id FROM sessions WHERE session_id = 'session-1'"
        ).fetchone()
        run_row = conn.execute(
            "SELECT account_id FROM runs WHERE id = 'run-1'"
        ).fetchone()
        upload_row = conn.execute(
            "SELECT account_id FROM uploads WHERE id = 'upload-1'"
        ).fetchone()

    with configure_connection(sqlite3.connect(agent_db_path)) as conn:
        custom_agent_row = conn.execute(
            "SELECT account_id FROM agent_templates WHERE id = 'custom-agent'"
        ).fetchone()
        system_agent_row = conn.execute(
            "SELECT account_id FROM agent_templates WHERE id = 'system-agent'"
        ).fetchone()
        root_account_row = conn.execute(
            "SELECT id, is_root FROM accounts WHERE id = ?",
            (ROOT_ACCOUNT_ID,),
        ).fetchone()

    assert session_row["account_id"] == ROOT_ACCOUNT_ID
    assert run_row["account_id"] == ROOT_ACCOUNT_ID
    assert upload_row["account_id"] == ROOT_ACCOUNT_ID
    assert custom_agent_row["account_id"] == ROOT_ACCOUNT_ID
    assert system_agent_row["account_id"] is None
    assert root_account_row["id"] == ROOT_ACCOUNT_ID
    assert root_account_row["is_root"] == 1

    rendered = format_root_migration_report(report)
    assert "sessions" in rendered
    assert "agent_templates" in rendered


def test_migrate_historical_data_to_root_initializes_empty_databases(tmp_path: Path):
    session_db_path = tmp_path / "sessions.db"
    agent_db_path = tmp_path / "agents.db"

    report = migrate_historical_data_to_root(
        session_db_path=session_db_path,
        agent_db_path=agent_db_path,
        root_password="RootPass123!",
    )

    assert report.session_backup_path is None
    assert report.agent_backup_path is None
    assert report.root_account_created is True
    assert report.root_credential_created is True
    assert report.bootstrap_root_password is None
    assert session_db_path.exists()
    assert agent_db_path.exists()
    assert all(item.total_rows == 0 for item in report.table_reports)
    assert all(item.backfilled_rows == 0 for item in report.table_reports)
    assert all(item.empty_account_rows == 0 for item in report.table_reports)

    with configure_connection(sqlite3.connect(session_db_path)) as conn:
        session_version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (SESSION_DB_SCOPE,),
        ).fetchone()
        sessions_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
        }

    with configure_connection(sqlite3.connect(agent_db_path)) as conn:
        agent_version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (AGENT_DB_SCOPE,),
        ).fetchone()
        root_account_row = conn.execute(
            "SELECT id, username, is_root FROM accounts WHERE id = ?",
            (ROOT_ACCOUNT_ID,),
        ).fetchone()

    assert session_version_row is not None
    assert session_version_row["version"] == CURRENT_SESSION_DB_VERSION
    assert "account_id" in sessions_columns
    assert agent_version_row is not None
    assert agent_version_row["version"] == CURRENT_AGENT_DB_VERSION
    assert root_account_row is not None
    assert root_account_row["id"] == ROOT_ACCOUNT_ID
    assert root_account_row["username"] == "root"
    assert root_account_row["is_root"] == 1


def test_migrate_historical_data_to_root_is_idempotent(tmp_path: Path):
    session_db_path = tmp_path / "sessions.db"
    agent_db_path = tmp_path / "agents.db"

    _create_legacy_session_db(session_db_path)
    _create_legacy_agent_db(agent_db_path)

    first_report = migrate_historical_data_to_root(
        session_db_path=session_db_path,
        agent_db_path=agent_db_path,
        root_password="RootPass123!",
    )
    second_report = migrate_historical_data_to_root(
        session_db_path=session_db_path,
        agent_db_path=agent_db_path,
        root_password="OtherPass456!",
    )

    first_sessions = {item.table_name: item for item in first_report.table_reports}
    second_sessions = {item.table_name: item for item in second_report.table_reports}

    assert first_sessions["sessions"].backfilled_rows == 1
    assert second_sessions["sessions"].backfilled_rows == 0
    assert second_sessions["runs"].backfilled_rows == 0
    assert second_sessions["uploads"].backfilled_rows == 0
    assert second_sessions["agent_templates"].backfilled_rows == 0
    assert all(item.empty_account_rows == 0 for item in second_report.table_reports)
    assert second_report.root_account_created is False
    assert second_report.root_credential_created is False
    assert second_report.bootstrap_root_password is None

