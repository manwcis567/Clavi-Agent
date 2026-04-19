"""Session history retrieval tools backed by the durable SQLite session store."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..session_store import SessionStore
from .base import Tool, ToolResult

_HISTORY_SOURCE_TYPES = [
    "all",
    "session_message",
    "run_goal",
    "run_completion",
    "run_failure",
    "shared_context",
]


class SearchSessionHistoryTool(Tool):
    """Search persisted session history across sessions for the current account."""

    def __init__(
        self,
        *,
        db_path: str,
        account_id: str | None = None,
        session_id: str | None = None,
    ):
        self.db_path = Path(db_path)
        self.account_id = str(account_id or "").strip() or None
        self.session_id = str(session_id or "").strip() or None
        self._session_store: SessionStore | None = None

    @property
    def name(self) -> str:
        return "search_session_history"

    @property
    def description(self) -> str:
        return (
            "Search persisted session history for the current user across prior chats and runs. "
            "Use this when you need to recall what was discussed, decided, or completed before."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or short phrases to search for in persisted session history.",
                },
                "scope_session_id": {
                    "type": "string",
                    "description": "Optional session id to restrict search to one session.",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Optional agent template id to restrict results to one agent context.",
                },
                "source_type": {
                    "type": "string",
                    "enum": _HISTORY_SOURCE_TYPES,
                    "description": "Which kind of historical source to search.",
                },
                "date_from": {
                    "type": "string",
                    "description": "Optional inclusive ISO timestamp lower bound.",
                },
                "date_to": {
                    "type": "string",
                    "description": "Optional inclusive ISO timestamp upper bound.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                    "description": "Maximum number of history matches to return.",
                },
            },
            "required": ["query"],
        }

    def _get_session_store(self) -> SessionStore:
        if self._session_store is None:
            self._session_store = SessionStore(self.db_path)
        return self._session_store

    @staticmethod
    def _normalize_source_types(source_type: str | None) -> list[str] | None:
        normalized = str(source_type or "all").strip()
        if normalized not in _HISTORY_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type: {source_type}")
        if normalized == "all":
            return None
        return [normalized]

    @staticmethod
    def _format_result(entry: dict[str, Any], index: int) -> str:
        session_line = f"   session: {entry.get('session_title') or entry.get('session_id')}"
        if entry.get("run_id"):
            session_line += f" | run: {entry['run_id']}"
        if entry.get("message_seq") is not None:
            session_line += f" | seq: {entry['message_seq']}"

        title = str(entry.get("title") or entry.get("role") or entry.get("source_type") or "").strip()
        if title:
            title = f" ({title})"

        lines = [
            f"{index}. [{entry.get('source_type', 'history')}] {entry.get('snippet') or entry.get('content')}{title}",
            session_line,
            f"   created_at: {entry.get('created_at', '')}",
        ]
        return "\n".join(lines)

    async def execute(
        self,
        query: str,
        scope_session_id: str | None = None,
        agent_id: str | None = None,
        source_type: str = "all",
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 8,
    ) -> ToolResult:
        try:
            normalized_query = " ".join(str(query or "").split())
            if not normalized_query:
                raise ValueError("query is required.")

            search_session_id = str(scope_session_id or self.session_id or "").strip() or None
            results = self._get_session_store().search_history(
                normalized_query,
                account_id=self.account_id,
                session_id=search_session_id,
                agent_id=str(agent_id or "").strip() or None,
                source_types=self._normalize_source_types(source_type),
                date_from=str(date_from or "").strip() or None,
                date_to=str(date_to or "").strip() or None,
                limit=max(1, min(int(limit), 20)),
            )
            if not results:
                return ToolResult(success=True, content="No matching session history found.")

            lines = ["Session History Search Results:"]
            for index, entry in enumerate(results, start=1):
                lines.append(self._format_result(entry, index))
            return ToolResult(success=True, content="\n".join(lines))
        except Exception as exc:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to search session history: {str(exc)}",
            )
