"""Shared context tools for cross-agent coordination."""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..session_store import SessionStore
from .base import Tool, ToolResult

# Per-file asyncio locks; keyed by resolved absolute path string.
# Created lazily and reused within the same event loop.
_FILE_LOCKS: dict[str, asyncio.Lock] = {}
_CATEGORY_ALIASES = {
    "requirement": "requirements",
    "requirements": "requirements",
    "plan": "plan",
    "plans": "plan",
    "finding": "findings",
    "findings": "findings",
    "blocker": "blockers",
    "blockers": "blockers",
    "handoff": "handoff",
    "handoffs": "handoff",
    "decision": "decisions",
    "decisions": "decisions",
    "risk": "risks",
    "risks": "risks",
    "general": "general",
}


def _get_file_lock(path: str) -> asyncio.Lock:
    """Return (or create) the asyncio.Lock for *path*.

    asyncio.Lock is bound to the running event loop, so we create it the first
    time it is requested and cache it.  All callers run in the same event loop,
    so the cached lock is always valid for the lifetime of the process.
    """
    if path not in _FILE_LOCKS:
        _FILE_LOCKS[path] = asyncio.Lock()
    return _FILE_LOCKS[path]


def _normalize_shared_category(value: str | None) -> str:
    """归一化共享上下文类别，避免多 worker 使用不同别名导致分栏漂移。"""
    normalized = str(value or "").strip().lower()
    if not normalized:
        return "general"
    return _CATEGORY_ALIASES.get(normalized, normalized)


class _SharedContextStore:
    """Small JSON-backed store used as a session-level coordination blackboard."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)

    def _lock(self) -> asyncio.Lock:
        return _get_file_lock(str(self.file_path.resolve()))

    def _load_entries_unlocked(self) -> list[dict[str, Any]]:
        if not self.file_path.exists():
            return []

        try:
            data = json.loads(self.file_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        return data if isinstance(data, list) else []

    async def load_entries(self) -> list[dict[str, Any]]:
        async with self._lock():
            return self._load_entries_unlocked()

    async def append_entry(self, entry: dict[str, Any]) -> None:
        async with self._lock():
            entries = self._load_entries_unlocked()
            entries.append(entry)
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


class ShareContextTool(Tool):
    """Publish information to the session-level shared coordination board."""

    def __init__(
        self,
        shared_file: str,
        agent_name: str,
        *,
        db_path: str | None = None,
        session_id: str | None = None,
        account_id: str | None = None,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
    ):
        self.store = _SharedContextStore(shared_file)
        self.agent_name = agent_name
        self.db_path = str(db_path or "").strip() or None
        self.session_id = str(session_id or "").strip() or None
        self.account_id = str(account_id or "").strip() or None
        self.run_id = run_id
        self.parent_run_id = parent_run_id
        self.root_run_id = root_run_id
        self._session_store: SessionStore | None = None

    @property
    def name(self) -> str:
        return "share_context"

    @property
    def description(self) -> str:
        return (
            "Publish important context to the session shared board so the main agent and "
            "other sub-agents can read it later. Use the standard sections "
            "requirements, plan, findings, blockers, handoff, decisions, and risks "
            "to keep multi-worker coordination consistent."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to publish to the shared board.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category. Recommended sections: requirements, plan, findings, blockers, handoff, decisions, or risks. Singular/plural aliases are normalized automatically.",
                },
                "title": {
                    "type": "string",
                    "description": "Optional short headline that helps other agents scan the board quickly.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        category: str = "general",
        title: str | None = None,
    ) -> ToolResult:
        try:
            entry_id = str(uuid.uuid4())
            normalized_category = _normalize_shared_category(category)
            entry = {
                "id": entry_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": self.agent_name,
                "category": normalized_category,
                "title": title or "",
                "content": content,
            }
            if self.run_id:
                entry["run_id"] = self.run_id
            if self.parent_run_id:
                entry["parent_run_id"] = self.parent_run_id
            if self.root_run_id:
                entry["root_run_id"] = self.root_run_id
            await self.store.append_entry(entry)
            if self.db_path and self.session_id:
                self._get_session_store().append_shared_context_entry(
                    self.session_id,
                    entry,
                    account_id=self.account_id,
                )

            headline = f"{title} - " if title else ""
            return ToolResult(
                success=True,
                content=(
                    f"Shared context recorded: {headline}{content} "
                    f"(category: {normalized_category}, source: {self.agent_name}, id: {entry_id})"
                ),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to share context: {e}",
            )

    def _get_session_store(self) -> SessionStore:
        if self._session_store is None:
            self._session_store = SessionStore(self.db_path)
        return self._session_store


class ReadSharedContextTool(Tool):
    """Read information from the session-level shared coordination board."""

    def __init__(self, shared_file: str, *, root_run_id: str | None = None):
        self.store = _SharedContextStore(shared_file)
        self.root_run_id = root_run_id

    @property
    def name(self) -> str:
        return "read_shared_context"

    @property
    def description(self) -> str:
        return (
            "Read context published by the main agent or other sub-agents in this same "
            "session. Use this before starting delegated work and whenever you need to "
            "synchronize with shared requirements, plan updates, findings, blockers, handoff notes, or decisions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category filter. Singular/plural aliases are normalized automatically.",
                },
                "source": {
                    "type": "string",
                    "description": "Optional source/agent filter.",
                },
                "query": {
                    "type": "string",
                    "description": "Optional keyword search over titles and content.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of most recent entries to return.",
                },
            },
        }

    async def execute(
        self,
        category: str | None = None,
        source: str | None = None,
        query: str | None = None,
        limit: int = 10,
    ) -> ToolResult:
        try:
            entries = await self.store.load_entries()
            if not entries:
                return ToolResult(success=True, content="No shared context has been published yet.")

            if self.root_run_id:
                entries = [
                    entry
                    for entry in entries
                    if entry.get("root_run_id") in {None, "", self.root_run_id}
                ]

            if category:
                normalized_category = _normalize_shared_category(category)
                entries = [
                    entry
                    for entry in entries
                    if _normalize_shared_category(entry.get("category")) == normalized_category
                ]

            if source:
                entries = [entry for entry in entries if entry.get("source") == source]

            if query:
                normalized = query.lower()
                entries = [
                    entry
                    for entry in entries
                    if normalized in entry.get("title", "").lower()
                    or normalized in entry.get("content", "").lower()
                ]

            if not entries:
                return ToolResult(success=True, content="No shared context matched the requested filters.")

            limit = max(1, min(limit, 50))
            entries = list(reversed(entries))[:limit]

            formatted_entries = []
            for index, entry in enumerate(entries, 1):
                title = entry.get("title")
                title_prefix = f"{title} - " if title else ""
                formatted_entries.append(
                    (
                        f"{index}. [{entry.get('category', 'general')}] {title_prefix}{entry.get('content', '')}\n"
                        f"   source: {entry.get('source', 'unknown')}\n"
                        f"   time: {entry.get('timestamp', 'unknown')}\n"
                        f"   id: {entry.get('id', 'unknown')}"
                    )
                )

            return ToolResult(
                success=True,
                content="Shared Context:\n" + "\n".join(formatted_entries),
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to read shared context: {e}",
            )
