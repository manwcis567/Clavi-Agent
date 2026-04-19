"""基于共享会话 SQLite 的上传文件持久化存储。"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .account_constants import ROOT_ACCOUNT_ID
from .sqlite_schema import configure_connection, ensure_session_db_schema
from .upload_models import UploadRecord


class UploadStore:
    """会话上传文件的 SQLite 仓储。"""

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

    @staticmethod
    def _upload_from_row(row: sqlite3.Row) -> UploadRecord:
        return UploadRecord.model_validate(dict(row))

    def create_upload(self, upload: UploadRecord) -> UploadRecord:
        with self._connect() as conn:
            resolved_account_id = self._resolve_session_account_id(
                conn,
                upload.session_id,
                upload.account_id,
            )
            conn.execute(
                """
                INSERT INTO uploads (
                    id, session_id, account_id, run_id, original_name, safe_name, relative_path,
                    absolute_path, mime_type, size_bytes, checksum, created_at, created_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    upload.id,
                    upload.session_id,
                    resolved_account_id,
                    upload.run_id,
                    upload.original_name,
                    upload.safe_name,
                    upload.relative_path,
                    upload.absolute_path,
                    upload.mime_type,
                    upload.size_bytes,
                    upload.checksum,
                    upload.created_at,
                    upload.created_by,
                ),
            )
        return upload.model_copy(update={"account_id": resolved_account_id})

    def get_upload(
        self,
        upload_id: str,
        *,
        account_id: str | None = None,
    ) -> UploadRecord | None:
        params: list[str] = [upload_id]
        sql = "SELECT * FROM uploads WHERE id = ?"
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
        return self._upload_from_row(row)

    def list_uploads(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
        run_id: str | None = None,
    ) -> list[UploadRecord]:
        params: list[str] = [session_id]
        where_clause = "WHERE session_id = ?"
        if account_id is not None:
            where_clause += " AND account_id = ?"
            params.append(account_id)
        if run_id is not None:
            where_clause += " AND run_id = ?"
            params.append(run_id)

        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM uploads
                {where_clause}
                ORDER BY created_at DESC, id DESC
                """,
                tuple(params),
            ).fetchall()
        return [self._upload_from_row(row) for row in rows]

    def delete_upload(self, upload_id: str, *, account_id: str | None = None) -> bool:
        params: list[str] = [upload_id]
        sql = "DELETE FROM uploads WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            deleted = conn.execute(
                sql,
                tuple(params),
            ).rowcount
        return deleted > 0
