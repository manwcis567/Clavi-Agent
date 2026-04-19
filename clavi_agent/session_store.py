"""Persistent session storage backed by SQLite."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from .account_constants import ROOT_ACCOUNT_ID
from .session_models import SessionHistorySearchResult, SessionRecord
from .schema import Message, ToolCall, message_content_summary, normalize_message_content
from .sqlite_schema import (
    configure_connection,
    ensure_session_db_schema,
    rebuild_session_history_fts,
    utc_now_iso,
)


DEFAULT_SESSION_TITLE = "新对话"


def normalize_title(content: str | list[dict[str, Any]], limit: int = 40) -> str:
    """Generate a stable first-message title."""
    normalized = " ".join(message_content_summary(content).split())
    if not normalized:
        return DEFAULT_SESSION_TITLE
    return normalized[:limit]


def message_to_preview(message: Message) -> str:
    """Build a short preview for session list display."""
    text = " ".join(message_content_summary(message.content).split())
    return text[:120]


class SessionStore:
    """SQLite repository for persisted chat sessions."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return configure_connection(sqlite3.connect(self.db_path))

    def _initialize(self):
        with self._connect() as conn:
            ensure_session_db_schema(conn)
            self._backfill_message_search_text(conn)
            rebuild_session_history_fts(conn)

    def _serialize_message(self, message: Message) -> dict[str, Any]:
        tool_calls_json = None
        if message.tool_calls:
            tool_calls_json = json.dumps(
                [tool_call.model_dump(mode="json") for tool_call in message.tool_calls],
                ensure_ascii=False,
            )

        return {
            "role": message.role,
            "content_json": json.dumps(
                normalize_message_content(message.content),
                ensure_ascii=False,
            ),
            "search_text": message_content_summary(message.content),
            "thinking": message.thinking,
            "tool_calls_json": tool_calls_json,
            "tool_call_id": message.tool_call_id,
            "name": message.name,
        }

    def _backfill_message_search_text(self, conn: sqlite3.Connection) -> None:
        rows = conn.execute(
            """
            SELECT id, content_json
            FROM session_messages
            WHERE search_text IS NULL OR search_text = ''
            """
        ).fetchall()
        for row in rows:
            try:
                search_text = message_content_summary(json.loads(row["content_json"]))
            except (TypeError, json.JSONDecodeError, ValueError):
                search_text = str(row["content_json"] or "")
            conn.execute(
                "UPDATE session_messages SET search_text = ? WHERE id = ?",
                (search_text, row["id"]),
            )

    @staticmethod
    def _build_history_match_query(query: str) -> str:
        normalized_query = " ".join(str(query or "").split())
        if not normalized_query:
            raise ValueError("query is required.")

        tokens = re.findall(r"[0-9A-Za-z_]+|[\u4e00-\u9fff]+", normalized_query)
        if not tokens:
            escaped = normalized_query.replace('"', '""')
            return f'"{escaped}"'

        clauses: list[str] = []
        for token in tokens:
            escaped = token.replace('"', '""')
            if token.isascii():
                clauses.append(f"{escaped}*")
            else:
                clauses.append(f'"{escaped}"')
        return " AND ".join(clauses)

    @staticmethod
    def _deserialize_history_result(row: sqlite3.Row) -> SessionHistorySearchResult:
        return SessionHistorySearchResult(
            source_key=row["source_key"],
            source_type=row["source_type"],
            account_id=row["account_id"] or ROOT_ACCOUNT_ID,
            session_id=row["session_id"],
            session_title=row["session_title"] or "",
            run_id=row["run_id"] or None,
            message_seq=row["message_seq"],
            role=row["role"] or "",
            title=row["title"] or "",
            content=row["content"] or "",
            snippet=row["snippet"] or row["content"] or "",
            created_at=row["created_at"],
            score=float(row["score"] or 0.0),
        )

    def _deserialize_message(self, row: sqlite3.Row) -> Message:
        tool_calls = None
        if row["tool_calls_json"]:
            tool_calls = [
                ToolCall.model_validate(item)
                for item in json.loads(row["tool_calls_json"])
            ]

        raw_content = json.loads(row["content_json"])
        return Message(
            role=row["role"],
            content=normalize_message_content(raw_content),
            thinking=row["thinking"],
            tool_calls=tool_calls,
            tool_call_id=row["tool_call_id"],
            name=row["name"],
        )

    def _deserialize_session(self, row: sqlite3.Row) -> SessionRecord:
        """Return typed persisted session metadata."""
        try:
            account_id = row["account_id"]
        except IndexError:
            account_id = ROOT_ACCOUNT_ID
        try:
            agent_id = row["agent_id"]
        except IndexError:
            agent_id = None
        try:
            ui_state = json.loads(row["ui_state_json"] or "{}")
        except (IndexError, TypeError, json.JSONDecodeError):
            ui_state = {}

        return SessionRecord(
            session_id=row["session_id"],
            account_id=account_id or ROOT_ACCOUNT_ID,
            title=row["title"],
            workspace_dir=row["workspace_dir"],
            agent_id=agent_id,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
            last_message_preview=row["last_message_preview"],
            ui_state=ui_state,
        )

    def create_session(
        self,
        session_id: str,
        workspace_dir: str,
        messages: list[Message],
        title: str = DEFAULT_SESSION_TITLE,
        agent_id: str | None = None,
        account_id: str = ROOT_ACCOUNT_ID,
    ):
        """Create a new persisted session."""
        now = utc_now_iso()
        preview = self._build_last_preview(messages)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (
                    session_id, account_id, title, workspace_dir, agent_id, ui_state_json,
                    created_at, updated_at, message_count, last_message_preview
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    account_id,
                    title,
                    workspace_dir,
                    agent_id,
                    "{}",
                    now,
                    now,
                    len(messages),
                    preview,
                ),
            )
            self._insert_messages(conn, session_id, messages, now)

    def _insert_messages(
        self,
        conn: sqlite3.Connection,
        session_id: str,
        messages: list[Message],
        created_at: str,
    ):
        for seq, message in enumerate(messages):
            payload = self._serialize_message(message)
            conn.execute(
                """
                INSERT INTO session_messages (
                    session_id, seq, role, content_json, search_text, thinking, tool_calls_json, tool_call_id, name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    seq,
                    payload["role"],
                    payload["content_json"],
                    payload["search_text"],
                    payload["thinking"],
                    payload["tool_calls_json"],
                    payload["tool_call_id"],
                    payload["name"],
                    created_at,
                ),
            )

    def replace_messages(
        self,
        session_id: str,
        messages: list[Message],
        *,
        account_id: str | None = None,
    ):
        """Replace all messages for a session with a fresh snapshot."""
        session = self.get_session(session_id, account_id=account_id)
        if not session:
            raise KeyError(f"Session not found: {session_id}")

        now = utc_now_iso()
        title = session["title"]
        if title == DEFAULT_SESSION_TITLE:
            first_user = next((msg for msg in messages if msg.role == "user"), None)
            if first_user:
                title = normalize_title(first_user.content)

        preview = self._build_last_preview(messages)

        with self._connect() as conn:
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
            self._insert_messages(conn, session_id, messages, now)
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, updated_at = ?, message_count = ?, last_message_preview = ?
                WHERE session_id = ?
                """,
                (title, now, len(messages), preview, session_id),
            )

    def append_message(
        self,
        session_id: str,
        message: Message,
        *,
        account_id: str | None = None,
    ):
        """Append a message immediately before a full snapshot is available."""
        session = self.get_session(session_id, account_id=account_id)
        if not session:
            raise KeyError(f"Session not found: {session_id}")

        now = utc_now_iso()
        next_seq = session["message_count"]
        title = session["title"]
        if (
            title == DEFAULT_SESSION_TITLE
            and session["message_count"] <= 1
            and message.role == "user"
        ):
            title = normalize_title(message.content)

        preview = session["last_message_preview"]
        if message.role in {"user", "assistant"}:
            preview = message_to_preview(message)

        payload = self._serialize_message(message)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO session_messages (
                    session_id, seq, role, content_json, search_text, thinking, tool_calls_json, tool_call_id, name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    next_seq,
                    payload["role"],
                    payload["content_json"],
                    payload["search_text"],
                    payload["thinking"],
                    payload["tool_calls_json"],
                    payload["tool_call_id"],
                    payload["name"],
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE sessions
                SET title = ?, updated_at = ?, message_count = ?, last_message_preview = ?
                WHERE session_id = ?
                """,
                (title, now, next_seq + 1, preview, session_id),
            )

    def update_session_agent_id(
        self,
        session_id: str,
        agent_id: str | None,
        *,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        """更新会话绑定的 Agent 模板标识。"""
        session = self.get_session(session_id, account_id=account_id)
        if not session:
            raise KeyError(f"Session not found: {session_id}")

        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE sessions
                SET agent_id = ?, updated_at = ?
                WHERE session_id = ?
                """,
                (agent_id, now, session_id),
            )
        refreshed = self.get_session(session_id, account_id=account_id)
        if refreshed is None:
            raise KeyError(f"Session not found: {session_id}")
        return refreshed

    def append_shared_context_entry(
        self,
        session_id: str,
        entry: dict[str, Any],
        *,
        account_id: str | None = None,
    ) -> None:
        """Append one shared-context entry into the durable session index."""
        session = self.get_session_record(session_id, account_id=account_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        entry_id = str(entry.get("id") or "").strip()
        if not entry_id:
            raise ValueError("shared context entry id is required.")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO shared_context_entries (
                    id, session_id, account_id, run_id, parent_run_id, root_run_id,
                    source, category, title, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry_id,
                    session_id,
                    session.account_id,
                    str(entry.get("run_id") or "").strip(),
                    str(entry.get("parent_run_id") or "").strip(),
                    str(entry.get("root_run_id") or "").strip(),
                    str(entry.get("source") or "").strip(),
                    str(entry.get("category") or "general").strip() or "general",
                    str(entry.get("title") or "").strip(),
                    str(entry.get("content") or "").strip(),
                    str(entry.get("timestamp") or utc_now_iso()).strip() or utc_now_iso(),
                ),
            )

    def get_session_record(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
    ) -> SessionRecord | None:
        """Return typed stored session metadata."""
        params: list[Any] = [session_id]
        sql = """
            SELECT session_id, account_id, title, workspace_dir, agent_id, ui_state_json,
                   created_at, updated_at, message_count, last_message_preview
            FROM sessions
            WHERE session_id = ?
        """
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()

        if not row:
            return None

        return self._deserialize_session(row)

    def get_session(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Return stored session metadata."""
        session = self.get_session_record(session_id, account_id=account_id)
        if session is None:
            return None
        return session.model_dump()

    def list_session_records(
        self,
        *,
        account_id: str | None = None,
    ) -> list[SessionRecord]:
        """Return all typed persisted sessions ordered by recent activity."""
        params: list[Any] = []
        sql = """
            SELECT session_id, account_id, title, workspace_dir, agent_id, ui_state_json,
                   created_at, updated_at, message_count, last_message_preview
            FROM sessions
        """
        if account_id is not None:
            sql += " WHERE account_id = ?"
            params.append(account_id)
        sql += " ORDER BY updated_at DESC, created_at DESC"
        with self._connect() as conn:
            rows = conn.execute(
                sql,
                tuple(params),
            ).fetchall()

        return [self._deserialize_session(row) for row in rows]

    def list_sessions(self, *, account_id: str | None = None) -> list[dict[str, Any]]:
        """Return all persisted sessions ordered by recent activity."""
        return [
            session.model_dump()
            for session in self.list_session_records(account_id=account_id)
        ]

    def search_history_records(
        self,
        query: str,
        *,
        account_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        exclude_run_id: str | None = None,
        source_types: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
    ) -> list[SessionHistorySearchResult]:
        """Search persisted session history across messages and run summaries."""
        match_query = self._build_history_match_query(query)
        params: list[Any] = [match_query]
        sql = """
            SELECT
                session_history_fts.source_key,
                session_history_fts.source_type,
                session_history_fts.account_id,
                session_history_fts.session_id,
                sessions.title AS session_title,
                NULLIF(session_history_fts.run_id, '') AS run_id,
                CASE
                    WHEN session_history_fts.message_seq = '' THEN NULL
                    ELSE CAST(session_history_fts.message_seq AS INTEGER)
                END AS message_seq,
                session_history_fts.role,
                session_history_fts.title,
                session_history_fts.content,
                snippet(session_history_fts, 9, '[', ']', ' … ', 16) AS snippet,
                session_history_fts.created_at,
                bm25(session_history_fts) AS score
            FROM session_history_fts
            LEFT JOIN sessions ON sessions.session_id = session_history_fts.session_id
            LEFT JOIN runs history_runs ON history_runs.id = session_history_fts.run_id
            WHERE session_history_fts MATCH ?
        """
        if account_id is not None:
            sql += " AND session_history_fts.account_id = ?"
            params.append(account_id)
        if session_id is not None:
            sql += " AND session_history_fts.session_id = ?"
            params.append(session_id)
        if agent_id is not None:
            sql += " AND COALESCE(history_runs.agent_template_id, sessions.agent_id, '') = ?"
            params.append(agent_id)
        if exclude_run_id is not None:
            sql += " AND session_history_fts.run_id != ?"
            params.append(exclude_run_id)
        if source_types:
            placeholders = ", ".join("?" for _ in source_types)
            sql += f" AND session_history_fts.source_type IN ({placeholders})"
            params.extend(source_types)
        if date_from is not None:
            sql += " AND session_history_fts.created_at >= ?"
            params.append(date_from)
        if date_to is not None:
            sql += " AND session_history_fts.created_at <= ?"
            params.append(date_to)
        sql += " ORDER BY score ASC, session_history_fts.created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._deserialize_history_result(row) for row in rows]

    def search_history(
        self,
        query: str,
        *,
        account_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        exclude_run_id: str | None = None,
        source_types: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search persisted session history and return JSON-ready payloads."""
        return [
            item.model_dump()
            for item in self.search_history_records(
                query,
                account_id=account_id,
                session_id=session_id,
                agent_id=agent_id,
                exclude_run_id=exclude_run_id,
                source_types=source_types,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
            )
        ]

    def get_messages(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
    ) -> list[Message]:
        """Return full message history for a session."""
        if account_id is not None and self.get_session_record(session_id, account_id=account_id) is None:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content_json, thinking, tool_calls_json, tool_call_id, name
                FROM session_messages
                WHERE session_id = ?
                ORDER BY seq ASC
                """,
                (session_id,),
            ).fetchall()

        return [self._deserialize_message(row) for row in rows]

    def delete_session(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
    ) -> bool:
        """Delete a session and all of its messages."""
        params: list[Any] = [session_id]
        sql = "DELETE FROM sessions WHERE session_id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            deleted = conn.execute(
                sql,
                tuple(params),
            ).rowcount
            conn.execute("DELETE FROM session_messages WHERE session_id = ?", (session_id,))
        return deleted > 0

    def _build_last_preview(self, messages: list[Message]) -> str:
        for message in reversed(messages):
            if message.role in {"user", "assistant"}:
                return message_to_preview(message)
        return ""
