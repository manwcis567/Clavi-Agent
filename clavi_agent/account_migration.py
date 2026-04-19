"""Utilities for migrating historical resources into the root account."""

from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .account_store import AccountStore, DEFAULT_ROOT_DISPLAY_NAME, DEFAULT_ROOT_USERNAME
from .config import Config, PRIMARY_ROOT_PASSWORD_ENV
from .sqlite_schema import (
    ROOT_ACCOUNT_ID,
    column_names,
    configure_connection,
    ensure_agent_db_schema,
    ensure_session_db_schema,
    table_exists,
)


@dataclass(frozen=True)
class MigrationTableSpec:
    """Describe one table that needs root ownership backfill."""

    scope: str
    table_name: str
    empty_where_clause: str = "account_id IS NULL OR account_id = ''"
    note: str = ""


@dataclass(frozen=True)
class MigrationTableReport:
    """Summarize migration status for one table."""

    scope: str
    table_name: str
    total_rows: int
    backfilled_rows: int
    empty_account_rows: int
    note: str = ""


@dataclass(frozen=True)
class RootOwnershipMigrationReport:
    """High-level report for one root-ownership migration run."""

    session_db_path: str
    agent_db_path: str
    session_backup_path: str | None
    agent_backup_path: str | None
    root_account_created: bool
    root_credential_created: bool
    bootstrap_root_password: str | None
    table_reports: list[MigrationTableReport]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report into a JSON-friendly dict."""
        return {
            "session_db_path": self.session_db_path,
            "agent_db_path": self.agent_db_path,
            "session_backup_path": self.session_backup_path,
            "agent_backup_path": self.agent_backup_path,
            "root_account_created": self.root_account_created,
            "root_credential_created": self.root_credential_created,
            "bootstrap_root_password": self.bootstrap_root_password,
            "table_reports": [asdict(item) for item in self.table_reports],
        }


SESSION_DB_TABLE_SPECS: tuple[MigrationTableSpec, ...] = (
    MigrationTableSpec(scope="session_db", table_name="sessions"),
    MigrationTableSpec(scope="session_db", table_name="runs"),
    MigrationTableSpec(scope="session_db", table_name="uploads"),
    MigrationTableSpec(scope="session_db", table_name="approval_requests"),
    MigrationTableSpec(scope="session_db", table_name="trace_events"),
    MigrationTableSpec(scope="session_db", table_name="integrations"),
    MigrationTableSpec(scope="session_db", table_name="integration_credentials"),
    MigrationTableSpec(scope="session_db", table_name="inbound_events"),
    MigrationTableSpec(scope="session_db", table_name="conversation_bindings"),
    MigrationTableSpec(scope="session_db", table_name="routing_rules"),
    MigrationTableSpec(scope="session_db", table_name="outbound_deliveries"),
    MigrationTableSpec(scope="session_db", table_name="delivery_attempts"),
    MigrationTableSpec(scope="session_db", table_name="scheduled_tasks"),
    MigrationTableSpec(scope="session_db", table_name="scheduled_task_executions"),
)

AGENT_DB_TABLE_SPECS: tuple[MigrationTableSpec, ...] = (
    MigrationTableSpec(
        scope="agent_db",
        table_name="agent_templates",
        empty_where_clause="is_system = 0 AND (account_id IS NULL OR account_id = '')",
        note="仅统计自定义模板，系统模板保留 account_id 为空。",
    ),
)

ALL_TABLE_SPECS = SESSION_DB_TABLE_SPECS + AGENT_DB_TABLE_SPECS


def _utc_timestamp_slug() -> str:
    """Return a filesystem-safe UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _resolve_db_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _backup_sqlite_database(
    db_path: Path,
    *,
    backup_dir: str | Path | None = None,
    backup_suffix: str = "pre_root_migration",
) -> Path | None:
    """Create a consistent SQLite backup before mutating the live DB."""
    if not db_path.exists():
        return None

    if backup_dir is None:
        resolved_backup_dir = db_path.parent / "backups"
    else:
        resolved_backup_dir = _resolve_db_path(backup_dir)
    resolved_backup_dir.mkdir(parents=True, exist_ok=True)

    backup_path = resolved_backup_dir / (
        f"{db_path.stem}.{backup_suffix}.{_utc_timestamp_slug()}{db_path.suffix}"
    )
    with sqlite3.connect(db_path) as source_conn:
        with sqlite3.connect(backup_path) as backup_conn:
            source_conn.backup(backup_conn)
    return backup_path


def _count_rows(
    conn: sqlite3.Connection,
    table_name: str,
    *,
    where_clause: str | None = None,
) -> int:
    """Count rows from one SQLite table with an optional filter."""
    if not table_exists(conn, table_name):
        return 0
    sql = f"SELECT COUNT(*) AS row_count FROM {table_name}"
    if where_clause:
        sql += f" WHERE {where_clause}"
    row = conn.execute(sql).fetchone()
    return int(row["row_count"]) if row else 0


def _count_pre_migration_backfill_rows(
    conn: sqlite3.Connection,
    spec: MigrationTableSpec,
) -> int:
    """Estimate how many rows need root backfill from the pre-migration DB."""
    if not table_exists(conn, spec.table_name):
        return 0

    columns = column_names(conn, spec.table_name)
    if "account_id" not in columns:
        if spec.table_name == "agent_templates":
            return _count_rows(conn, spec.table_name, where_clause="is_system = 0")
        return _count_rows(conn, spec.table_name)

    return _count_rows(conn, spec.table_name, where_clause=spec.empty_where_clause)


def _count_empty_account_rows(
    conn: sqlite3.Connection,
    spec: MigrationTableSpec,
) -> int:
    """Count rows that are still missing account ownership after migration."""
    if not table_exists(conn, spec.table_name):
        return 0

    columns = column_names(conn, spec.table_name)
    if "account_id" not in columns:
        if spec.table_name == "agent_templates":
            return _count_rows(conn, spec.table_name, where_clause="is_system = 0")
        return _count_rows(conn, spec.table_name)

    return _count_rows(conn, spec.table_name, where_clause=spec.empty_where_clause)


def _collect_reports(
    session_db_path: Path,
    agent_db_path: Path,
    *,
    session_pre_backfill_counts: dict[str, int],
    agent_pre_backfill_counts: dict[str, int],
) -> list[MigrationTableReport]:
    """Read post-migration table stats and combine them with pre-migration counts."""
    reports: list[MigrationTableReport] = []

    with configure_connection(sqlite3.connect(session_db_path)) as conn:
        for spec in SESSION_DB_TABLE_SPECS:
            reports.append(
                MigrationTableReport(
                    scope=spec.scope,
                    table_name=spec.table_name,
                    total_rows=_count_rows(conn, spec.table_name),
                    backfilled_rows=session_pre_backfill_counts.get(spec.table_name, 0),
                    empty_account_rows=_count_empty_account_rows(conn, spec),
                    note=spec.note,
                )
            )

    with configure_connection(sqlite3.connect(agent_db_path)) as conn:
        for spec in AGENT_DB_TABLE_SPECS:
            reports.append(
                MigrationTableReport(
                    scope=spec.scope,
                    table_name=spec.table_name,
                    total_rows=_count_rows(conn, spec.table_name),
                    backfilled_rows=agent_pre_backfill_counts.get(spec.table_name, 0),
                    empty_account_rows=_count_empty_account_rows(conn, spec),
                    note=spec.note,
                )
            )

    return reports


def _load_migration_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load only the migration-relevant config fields without requiring an API key."""
    if config_path is None:
        candidate = Config.get_default_config_path()
    else:
        candidate = _resolve_db_path(config_path)

    if not candidate.exists():
        return {}

    with open(candidate, encoding="utf-8") as file_obj:
        data = yaml.safe_load(file_obj) or {}
    if not isinstance(data, dict):
        return {}
    return data


def _resolve_root_password(auth_data: dict[str, Any]) -> str | None:
    """Resolve root password from config first, then from environment."""
    direct_value = str(auth_data.get("root_password") or "").strip()
    if direct_value:
        return direct_value

    env_name = str(auth_data.get("root_password_env") or PRIMARY_ROOT_PASSWORD_ENV).strip()
    if not env_name:
        return None

    env_value = os.environ.get(env_name, "").strip()
    return env_value or None


def migrate_historical_data_to_root(
    *,
    session_db_path: str | Path,
    agent_db_path: str | Path,
    root_username: str = DEFAULT_ROOT_USERNAME,
    root_display_name: str = DEFAULT_ROOT_DISPLAY_NAME,
    root_password: str | None = None,
    backup_dir: str | Path | None = None,
) -> RootOwnershipMigrationReport:
    """Backup, migrate, and verify historical resources under the root account."""
    resolved_session_db_path = _resolve_db_path(session_db_path)
    resolved_agent_db_path = _resolve_db_path(agent_db_path)
    resolved_session_db_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_agent_db_path.parent.mkdir(parents=True, exist_ok=True)

    session_backup_path = _backup_sqlite_database(
        resolved_session_db_path,
        backup_dir=backup_dir,
    )
    agent_backup_path = _backup_sqlite_database(
        resolved_agent_db_path,
        backup_dir=backup_dir,
    )

    session_pre_counts = {spec.table_name: 0 for spec in SESSION_DB_TABLE_SPECS}
    if session_backup_path is not None:
        with configure_connection(sqlite3.connect(session_backup_path)) as conn:
            session_pre_counts = {
                spec.table_name: _count_pre_migration_backfill_rows(conn, spec)
                for spec in SESSION_DB_TABLE_SPECS
            }

    agent_pre_counts = {spec.table_name: 0 for spec in AGENT_DB_TABLE_SPECS}
    if agent_backup_path is not None:
        with configure_connection(sqlite3.connect(agent_backup_path)) as conn:
            agent_pre_counts = {
                spec.table_name: _count_pre_migration_backfill_rows(conn, spec)
                for spec in AGENT_DB_TABLE_SPECS
            }

    with configure_connection(sqlite3.connect(resolved_session_db_path)) as conn:
        ensure_session_db_schema(conn)

    with configure_connection(sqlite3.connect(resolved_agent_db_path)) as conn:
        ensure_agent_db_schema(conn)

    account_store = AccountStore(
        resolved_agent_db_path,
        auto_seed_root=False,
    )
    seed_result = account_store.ensure_root_account(
        username=root_username,
        display_name=root_display_name,
        password=root_password,
    )

    reports = _collect_reports(
        resolved_session_db_path,
        resolved_agent_db_path,
        session_pre_backfill_counts=session_pre_counts,
        agent_pre_backfill_counts=agent_pre_counts,
    )

    return RootOwnershipMigrationReport(
        session_db_path=str(resolved_session_db_path),
        agent_db_path=str(resolved_agent_db_path),
        session_backup_path=str(session_backup_path) if session_backup_path else None,
        agent_backup_path=str(agent_backup_path) if agent_backup_path else None,
        root_account_created=seed_result.created,
        root_credential_created=seed_result.credential_created,
        bootstrap_root_password=seed_result.bootstrap_password,
        table_reports=reports,
    )


def format_root_migration_report(report: RootOwnershipMigrationReport) -> str:
    """Render a concise human-readable migration report."""
    lines = [
        "历史数据迁移到 root 已完成。",
        f"session_db: {report.session_db_path}",
        f"agent_db: {report.agent_db_path}",
        f"session_db 备份: {report.session_backup_path or '未生成（源文件不存在）'}",
        f"agent_db 备份: {report.agent_backup_path or '未生成（源文件不存在）'}",
        f"root 账号新增: {'是' if report.root_account_created else '否'}",
        f"root 密码凭证新增: {'是' if report.root_credential_created else '否'}",
        "校验报告：",
    ]
    for item in report.table_reports:
        line = (
            f"- {item.scope}.{item.table_name}: 总数={item.total_rows}，"
            f"已回填={item.backfilled_rows}，仍为空={item.empty_account_rows}"
        )
        if item.note:
            line += f"；说明：{item.note}"
        lines.append(line)

    if report.bootstrap_root_password:
        lines.append(
            f"首次补齐 root 密码，已生成临时密码：{report.bootstrap_root_password}"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for root-ownership migration."""
    parser = argparse.ArgumentParser(
        description="备份 SQLite 并将历史数据迁移到 root 账号。",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径；未提供时默认尝试读取 clavi_agent/config/config.yaml。",
    )
    parser.add_argument(
        "--session-db",
        type=str,
        default=None,
        help="session_db 路径；未提供时从配置读取。",
    )
    parser.add_argument(
        "--agent-db",
        type=str,
        default=None,
        help="agent_db 路径；未提供时从配置读取。",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="备份目录；未提供时默认写入数据库同级 backups 目录。",
    )
    parser.add_argument(
        "--root-username",
        type=str,
        default=None,
        help="root 用户名；未提供时从配置读取，默认 root。",
    )
    parser.add_argument(
        "--root-display-name",
        type=str,
        default=None,
        help="root 展示名；未提供时从配置读取，默认 Root。",
    )
    parser.add_argument(
        "--root-password",
        type=str,
        default=None,
        help="root 密码；未提供时从配置或环境变量读取。",
    )
    return parser


def main() -> None:
    """CLI entry point for the root-ownership migration."""
    args = build_arg_parser().parse_args()
    config_data = _load_migration_config(args.config)
    agent_data = config_data.get("agent", {}) if isinstance(config_data.get("agent"), dict) else {}
    auth_data = config_data.get("auth", {}) if isinstance(config_data.get("auth"), dict) else {}

    session_db_path = (
        args.session_db
        or config_data.get("session_store_path")
        or "./workspace/.clavi_agent/sessions.db"
    )
    agent_db_path = (
        args.agent_db
        or agent_data.get("agent_store_path")
        or "./workspace/.clavi_agent/agents.db"
    )
    root_username = (
        args.root_username
        or auth_data.get("root_username")
        or DEFAULT_ROOT_USERNAME
    )
    root_display_name = (
        args.root_display_name
        or auth_data.get("root_display_name")
        or DEFAULT_ROOT_DISPLAY_NAME
    )
    root_password = args.root_password or _resolve_root_password(auth_data)

    report = migrate_historical_data_to_root(
        session_db_path=session_db_path,
        agent_db_path=agent_db_path,
        root_username=root_username,
        root_display_name=root_display_name,
        root_password=root_password,
        backup_dir=args.backup_dir,
    )
    print(format_root_migration_report(report))


__all__ = [
    "ALL_TABLE_SPECS",
    "AGENT_DB_TABLE_SPECS",
    "MigrationTableReport",
    "MigrationTableSpec",
    "RootOwnershipMigrationReport",
    "SESSION_DB_TABLE_SPECS",
    "format_root_migration_report",
    "migrate_historical_data_to_root",
]


if __name__ == "__main__":
    main()


