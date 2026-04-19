"""Structured memory tools with backward-compatible session note behavior."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ..memory_provider import LocalMemoryProvider, MemoryProvider
from .base import Tool, ToolResult

_UTF8 = "utf-8"
_MEMORY_TYPES = [
    "preference",
    "communication_style",
    "goal",
    "constraint",
    "project_fact",
    "workflow_fact",
    "correction",
]
_MEMORY_SCOPES = ["auto", "user_memory", "user_profile", "agent_memory", "all"]
_LEGACY_CATEGORY_TO_MEMORY_TYPE = {
    "user_preference": "preference",
    "preference": "preference",
    "communication_style": "communication_style",
    "style": "communication_style",
    "goal": "goal",
    "constraint": "constraint",
    "decision": "constraint",
    "project_info": "project_fact",
    "project_fact": "project_fact",
    "workflow": "workflow_fact",
    "workflow_fact": "workflow_fact",
    "correction": "correction",
}
_SIMILARITY_MERGE_TYPES = {
    "preference",
    "communication_style",
    "constraint",
    "workflow_fact",
    "correction",
}
_USER_MEMORY_LIMITS = {
    "preference": 24,
    "communication_style": 16,
    "goal": 20,
    "constraint": 20,
    "project_fact": 40,
    "workflow_fact": 30,
    "correction": 24,
}
_SECRET_PATTERNS = [
    re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret[_-]?key|password)\b\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{16,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
]
_TRANSIENT_PATH_HINTS = (
    "\\temp\\",
    "/tmp/",
    "\\appdata\\local\\temp\\",
    "/var/folders/",
    ".log",
    ".tmp",
    ".cache",
)
_STABILITY_HINTS = (
    "偏好",
    "喜欢",
    "不喜欢",
    "要求",
    "必须",
    "统一",
    "约定",
    "规范",
    "默认",
    "流程",
    "习惯",
    "长期",
    "总是",
    "避免",
    "prefer",
    "always",
    "must",
    "avoid",
    "workflow",
    "convention",
)


class _MemoryToolSupport:
    def __init__(
        self,
        *,
        memory_file: str = "./workspace/.agent_memory.json",
        user_id: str | None = None,
        db_path: str | None = None,
        memory_provider: MemoryProvider | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        self.memory_file = Path(memory_file)
        self.user_id = str(user_id or "").strip() or None
        self.db_path = str(db_path or "").strip() or None
        self.memory_provider = memory_provider
        self.session_id = str(session_id or "").strip() or None
        self.run_id = str(run_id or "").strip() or None
        self._local_provider: MemoryProvider | None = None

    def _get_memory_provider(self) -> MemoryProvider | None:
        if self.memory_provider is not None:
            return self.memory_provider
        if not self.user_id or not self.db_path:
            return None
        if self._local_provider is None:
            self._local_provider = LocalMemoryProvider(self.db_path)
        return self._local_provider

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        return str(value or "").strip()

    @classmethod
    def _normalize_memory_type(
        cls,
        memory_type: str | None,
        category: str | None,
    ) -> str:
        normalized_memory_type = cls._normalize_text(memory_type)
        if normalized_memory_type:
            if normalized_memory_type not in _MEMORY_TYPES:
                raise ValueError(f"Unsupported memory_type: {memory_type}")
            return normalized_memory_type

        normalized_category = cls._normalize_text(category).casefold()
        inferred = _LEGACY_CATEGORY_TO_MEMORY_TYPE.get(normalized_category)
        if inferred:
            return inferred
        return "project_fact"

    @classmethod
    def _normalize_scope(cls, scope: str | None, *, has_user_store: bool) -> str:
        normalized_scope = cls._normalize_text(scope) or "auto"
        if normalized_scope not in _MEMORY_SCOPES:
            raise ValueError(f"Unsupported scope: {scope}")
        if normalized_scope == "auto":
            return "user_memory" if has_user_store else "agent_memory"
        return normalized_scope

    @staticmethod
    def _normalize_confidence(confidence: float | int) -> float:
        normalized = float(confidence)
        if normalized < 0 or normalized > 1:
            raise ValueError("confidence must be between 0 and 1.")
        return normalized

    @staticmethod
    def _content_key(content: str) -> str:
        return " ".join(str(content or "").split()).casefold()

    @staticmethod
    def _merge_text(existing: str | None, incoming: str | None) -> str:
        existing_text = str(existing or "").strip()
        incoming_text = str(incoming or "").strip()
        if len(incoming_text) > len(existing_text):
            return incoming_text
        return existing_text or incoming_text

    @staticmethod
    def _tokenize_similarity_text(value: str) -> set[str]:
        normalized = str(value or "").casefold()
        tokens = {
            token
            for token in re.split(r"[^0-9a-z_]+", normalized)
            if len(token) >= 2
        }
        compact = "".join(
            char
            for char in normalized
            if char.isalnum() or char == "_" or ("\u4e00" <= char <= "\u9fff")
        )
        if len(compact) >= 2:
            tokens.update(compact[index : index + 2] for index in range(len(compact) - 1))
        return tokens

    @classmethod
    def _similarity_score(cls, left: str, right: str) -> float:
        left_tokens = cls._tokenize_similarity_text(left)
        right_tokens = cls._tokenize_similarity_text(right)
        if not left_tokens or not right_tokens:
            return 0.0
        overlap = len(left_tokens & right_tokens)
        union = len(left_tokens | right_tokens)
        containment = overlap / min(len(left_tokens), len(right_tokens))
        jaccard = overlap / union if union else 0.0
        return max(jaccard, containment)

    @staticmethod
    def _looks_secret_like(*values: Any) -> bool:
        for value in values:
            text = str(value or "").strip()
            if not text:
                continue
            for pattern in _SECRET_PATTERNS:
                if pattern.search(text):
                    return True
        return False

    @staticmethod
    def _looks_like_dump(text: str) -> bool:
        normalized = str(text or "")
        non_empty_lines = [line for line in normalized.splitlines() if line.strip()]
        if len(non_empty_lines) >= 18:
            return True
        if len(normalized) >= 1600:
            return True
        lowered = normalized.casefold()
        if "traceback (most recent call last):" in lowered:
            return True
        if lowered.count(" at ") >= 8 and "\n" in lowered:
            return True
        if normalized.count("{") >= 12 and normalized.count(":") >= 12:
            return True
        return False

    @classmethod
    def _looks_like_transient_noise(
        cls,
        content: str,
        *,
        memory_type: str,
    ) -> bool:
        lowered = str(content or "").casefold()
        if any(hint in lowered for hint in _TRANSIENT_PATH_HINTS):
            return True
        if re.fullmatch(r"(?:[a-z]:\\|/)[^\r\n]+", lowered):
            return True
        if memory_type in {"project_fact", "workflow_fact"}:
            if any(marker in lowered for marker in ("stderr", "stdout", "exit code", "npm err!", "stack trace")):
                return True
        return False

    @classmethod
    def _should_skip_memory_write(
        cls,
        *,
        content: str,
        summary: str,
        memory_type: str,
        profile_updates: dict[str, Any] | None = None,
    ) -> str | None:
        payloads = [content, summary]
        if profile_updates:
            payloads.append(json.dumps(profile_updates, ensure_ascii=False))
        if cls._looks_secret_like(*payloads):
            return "content appears to contain secret-like material"
        if cls._looks_like_dump(content):
            return "content looks like a raw log or dump"
        if cls._looks_like_transient_noise(content, memory_type=memory_type):
            return "content looks transient or easy to rediscover"
        if memory_type in {"project_fact", "workflow_fact"}:
            lowered = f"{content}\n{summary}".casefold()
            if (
                not summary.strip()
                and not any(hint in lowered for hint in _STABILITY_HINTS)
                and len(content.strip()) < 18
            ):
                return "content is too short to be a stable long-term memory"
        return None

    @staticmethod
    def _memory_rank_key(entry: dict[str, Any]) -> tuple[float, int, int, str, str]:
        confidence = float(entry.get("confidence", 0) or 0)
        summary_length = len(str(entry.get("summary", "") or "").strip())
        content_length = len(str(entry.get("content", "") or "").strip())
        updated_at = str(entry.get("updated_at", "") or "")
        entry_id = str(entry.get("id", "") or "")
        return (confidence, summary_length, content_length, updated_at, entry_id)

    @classmethod
    def _find_similar_entry(
        cls,
        entries: list[dict[str, Any]],
        *,
        content: str,
        memory_type: str,
    ) -> dict[str, Any] | None:
        if memory_type not in _SIMILARITY_MERGE_TYPES:
            return None
        best_entry: dict[str, Any] | None = None
        best_score = 0.0
        for entry in entries:
            existing_content = str(entry.get("content", "") or "")
            score = cls._similarity_score(existing_content, content)
            if score >= 0.75 and score > best_score:
                best_entry = entry
                best_score = score
        return best_entry

    def _load_agent_notes(self) -> list[dict[str, Any]]:
        if not self.memory_file.exists():
            return []

        try:
            payload = json.loads(self.memory_file.read_text(encoding=_UTF8))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []

        notes: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, dict):
                notes.append(item)
        return notes

    def _save_agent_notes(self, notes: list[dict[str, Any]]) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        self.memory_file.write_text(
            json.dumps(notes, indent=2, ensure_ascii=False),
            encoding=_UTF8,
        )

    @staticmethod
    def _matches_query(entry: dict[str, Any], query: str) -> bool:
        normalized_query = " ".join(str(query or "").casefold().split())
        if not normalized_query:
            return True
        haystack = " ".join(
            [
                str(entry.get("content", "")),
                str(entry.get("summary", "")),
                str(entry.get("category", "")),
                str(entry.get("memory_type", "")),
            ]
        ).casefold()
        return all(term in haystack for term in normalized_query.split())

    @staticmethod
    def _format_profile_section(profile: dict[str, Any]) -> str:
        lines = ["User Profile Summary:"]
        summary = str(profile.get("summary", "") or "").strip()
        if summary:
            lines.append(f"- summary: {summary}")
        payload = profile.get("profile") or {}
        if isinstance(payload, dict) and payload:
            for key in sorted(payload):
                lines.append(f"- {key}: {payload[key]}")
        if len(lines) == 1:
            lines.append("- empty")
        return "\n".join(lines)

    @staticmethod
    def _format_memory_entry(
        entry: dict[str, Any],
        *,
        index: int,
        detailed: bool,
    ) -> str:
        memory_type = entry.get("memory_type") or entry.get("category") or "general"
        primary_text = str(entry.get("summary") or entry.get("content") or "").strip()
        lines = [f"{index}. [{memory_type}] {primary_text}"]
        if detailed:
            content = str(entry.get("content") or "").strip()
            if content and content != primary_text:
                lines.append(f"   content: {content}")
            confidence = entry.get("confidence")
            if confidence is not None:
                lines.append(f"   confidence: {float(confidence):.2f}")
            source_session_id = entry.get("source_session_id")
            if source_session_id:
                lines.append(f"   session: {source_session_id}")
            source_run_id = entry.get("source_run_id")
            if source_run_id:
                lines.append(f"   run: {source_run_id}")
            writer_type = str(entry.get("writer_type") or "").strip()
            writer_id = str(entry.get("writer_id") or "").strip()
            if writer_type or writer_id:
                lines.append(f"   writer: {writer_type or 'unknown'}:{writer_id or 'unknown'}")
            timestamp = entry.get("updated_at") or entry.get("timestamp")
            if timestamp:
                lines.append(f"   updated_at: {timestamp}")
            entry_id = entry.get("id")
            if entry_id:
                lines.append(f"   id: {entry_id}")
        return "\n".join(lines)


class SessionNoteTool(_MemoryToolSupport, Tool):
    """Record structured user memory, user profile facts, or agent-local notes."""

    def __init__(
        self,
        memory_file: str = "./workspace/.agent_memory.json",
        *,
        user_id: str | None = None,
        db_path: str | None = None,
        memory_provider: MemoryProvider | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        super().__init__(
            memory_file=memory_file,
            user_id=user_id,
            db_path=db_path,
            memory_provider=memory_provider,
            session_id=session_id,
            run_id=run_id,
        )

    @property
    def name(self) -> str:
        return "record_note"

    @property
    def description(self) -> str:
        return (
            "Save durable memory in a structured way. "
            "Use scope=user_memory for long-term user memory, "
            "scope=user_profile for structured profile fields, "
            "or scope=agent_memory for workspace-local notes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to record as a note. Be concise but specific.",
                },
                "category": {
                    "type": "string",
                    "description": "Legacy category tag kept for compatibility.",
                },
                "scope": {
                    "type": "string",
                    "enum": _MEMORY_SCOPES[:-1],
                    "description": "Where to save the memory.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": _MEMORY_TYPES,
                    "description": "Structured type for user_memory entries.",
                },
                "summary": {
                    "type": "string",
                    "description": "Compact summary used during future recall.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score for the memory.",
                },
                "profile_updates": {
                    "type": "object",
                    "description": "Structured fields to merge into user_profile.",
                },
                "profile_summary": {
                    "type": "string",
                    "description": "Optional summary for the user profile.",
                },
                "supersede_entry_id": {
                    "type": "string",
                    "description": "Mark an older user_memory entry as superseded by this new one.",
                },
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        category: str = "general",
        scope: str = "auto",
        memory_type: str | None = None,
        summary: str = "",
        confidence: float = 0.7,
        profile_updates: dict[str, Any] | None = None,
        profile_summary: str = "",
        supersede_entry_id: str | None = None,
    ) -> ToolResult:
        try:
            normalized_content = self._normalize_text(content)
            if not normalized_content:
                raise ValueError("content is required.")

            memory_provider = self._get_memory_provider()
            resolved_scope = self._normalize_scope(
                scope,
                has_user_store=memory_provider is not None and memory_provider.is_available,
            )
            normalized_confidence = self._normalize_confidence(confidence)

            if resolved_scope == "user_profile":
                if memory_provider is None or not memory_provider.is_available or self.user_id is None:
                    raise ValueError("user_profile scope requires both user_id and db_path.")
                updates = dict(profile_updates or {})
                if not updates:
                    raise ValueError("profile_updates is required for user_profile scope.")
                skip_reason = self._should_skip_memory_write(
                    content=normalized_content,
                    summary=profile_summary or summary,
                    memory_type="preference",
                    profile_updates=updates,
                )
                if skip_reason:
                    return ToolResult(
                        success=True,
                        content=f"Skipped saving user profile: {skip_reason}.",
                    )
                memory_provider.upsert_user_profile(
                    self.user_id,
                    profile=updates,
                    summary=profile_summary or summary or normalized_content,
                    merge=True,
                    source_session_id=self.session_id,
                    source_run_id=self.run_id,
                    writer_type="tool",
                    writer_id=self.name,
                )
                keys = ", ".join(sorted(updates))
                return ToolResult(
                    success=True,
                    content=f"Updated user profile fields: {keys or 'none'}",
                )

            if resolved_scope == "user_memory":
                if memory_provider is None or not memory_provider.is_available or self.user_id is None:
                    raise ValueError("user_memory scope requires both user_id and db_path.")
                resolved_memory_type = self._normalize_memory_type(memory_type, category)
                skip_reason = self._should_skip_memory_write(
                    content=normalized_content,
                    summary=summary,
                    memory_type=resolved_memory_type,
                )
                if skip_reason:
                    return ToolResult(
                        success=True,
                        content=f"Skipped saving user memory: {skip_reason}.",
                    )
                active_entries = memory_provider.list_memory_entries(
                    self.user_id,
                    memory_types=[resolved_memory_type],
                    include_superseded=False,
                    limit=100,
                )
                duplicate = next(
                    (
                        entry
                        for entry in active_entries
                        if self._content_key(entry.get("content", ""))
                        == self._content_key(normalized_content)
                    ),
                    None,
                )

                if duplicate is not None and not supersede_entry_id:
                    updated = memory_provider.update_memory_entry(
                        duplicate["id"],
                        user_id=self.user_id,
                        summary=summary or duplicate.get("summary", ""),
                        source_session_id=self.session_id,
                        source_run_id=self.run_id,
                        writer_type="tool",
                        writer_id=self.name,
                        confidence=max(float(duplicate.get("confidence", 0)), normalized_confidence),
                    )
                    return ToolResult(
                        success=True,
                        content=(
                            "Updated existing user memory entry: "
                            f"{updated['id']} (type: {resolved_memory_type})"
                        ),
                    )

                similar_entry = None
                if not supersede_entry_id:
                    similar_entry = self._find_similar_entry(
                        active_entries,
                        content=normalized_content,
                        memory_type=resolved_memory_type,
                    )
                if similar_entry is not None:
                    updated = memory_provider.update_memory_entry(
                        similar_entry["id"],
                        user_id=self.user_id,
                        content=self._merge_text(
                            similar_entry.get("content", ""),
                            normalized_content,
                        ),
                        summary=summary or similar_entry.get("summary", ""),
                        source_session_id=self.session_id,
                        source_run_id=self.run_id,
                        writer_type="tool",
                        writer_id=self.name,
                        confidence=max(
                            float(similar_entry.get("confidence", 0)),
                            normalized_confidence,
                        ),
                    )
                    return ToolResult(
                        success=True,
                        content=(
                            "Merged into existing user memory entry: "
                            f"{updated['id']} (type: {resolved_memory_type})"
                        ),
                    )

                memory_limit = _USER_MEMORY_LIMITS.get(resolved_memory_type, 24)
                if len(active_entries) >= memory_limit and not supersede_entry_id:
                    weakest_entry = min(active_entries, key=self._memory_rank_key)
                    candidate_rank = self._memory_rank_key(
                        {
                            "confidence": normalized_confidence,
                            "summary": summary,
                            "content": normalized_content,
                            "updated_at": datetime.now().isoformat(),
                            "id": "",
                        }
                    )
                    if candidate_rank <= self._memory_rank_key(weakest_entry):
                        return ToolResult(
                            success=True,
                            content=(
                                "Skipped saving user memory: type capacity reached and "
                                "existing entries are stronger."
                            ),
                        )

                created = memory_provider.create_memory_entry(
                    user_id=self.user_id,
                    memory_type=resolved_memory_type,
                    content=normalized_content,
                    summary=summary,
                    source_session_id=self.session_id,
                    source_run_id=self.run_id,
                    writer_type="tool",
                    writer_id=self.name,
                    confidence=normalized_confidence,
                )
                if supersede_entry_id:
                    memory_provider.supersede_memory_entry(
                        supersede_entry_id,
                        superseded_by=created["id"],
                        user_id=self.user_id,
                        source_session_id=self.session_id,
                        source_run_id=self.run_id,
                        writer_type="tool",
                        writer_id=self.name,
                    )
                compacted_entry_ids = memory_provider.enforce_memory_capacity(
                    self.user_id,
                    memory_type=resolved_memory_type,
                    max_active=memory_limit,
                    preferred_entry_id=created["id"],
                    source_session_id=self.session_id,
                    source_run_id=self.run_id,
                    writer_type="tool",
                    writer_id=self.name,
                )
                compacted_suffix = ""
                if compacted_entry_ids:
                    compacted_suffix = f"; compacted {len(compacted_entry_ids)} older entries"
                return ToolResult(
                    success=True,
                    content=(
                        "Saved user memory entry: "
                        f"{created['id']} (type: {resolved_memory_type}){compacted_suffix}"
                    ),
                )

            skip_reason = self._should_skip_memory_write(
                content=normalized_content,
                summary=summary,
                memory_type=self._normalize_memory_type(memory_type, category),
            )
            if skip_reason:
                return ToolResult(
                    success=True,
                    content=f"Skipped saving agent memory: {skip_reason}.",
                )
            notes = self._load_agent_notes()
            note_key = self._content_key(normalized_content)
            duplicate_index = next(
                (
                    index
                    for index, item in enumerate(notes)
                    if self._content_key(item.get("content", "")) == note_key
                    and self._normalize_text(item.get("category")) == self._normalize_text(category)
                ),
                None,
            )
            note = {
                "timestamp": datetime.now().isoformat(),
                "scope": "agent_memory",
                "category": category,
                "memory_type": memory_type or "",
                "summary": summary,
                "content": normalized_content,
                "confidence": normalized_confidence,
                "source_session_id": self.session_id,
                "source_run_id": self.run_id,
                "writer_type": "tool",
                "writer_id": self.name,
            }
            if duplicate_index is None:
                notes.append(note)
            else:
                existing = notes[duplicate_index]
                existing.update(note)
                existing["confidence"] = max(
                    float(existing.get("confidence", 0)),
                    normalized_confidence,
                )
                notes[duplicate_index] = existing
            self._save_agent_notes(notes)
            return ToolResult(
                success=True,
                content=f"Recorded agent memory: {normalized_content} (category: {category})",
            )
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to record note: {str(e)}",
            )


class RecallNoteTool(_MemoryToolSupport, Tool):
    """Recall structured memory summaries or legacy agent-local notes."""

    def __init__(
        self,
        memory_file: str = "./workspace/.agent_memory.json",
        *,
        user_id: str | None = None,
        db_path: str | None = None,
        memory_provider: MemoryProvider | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        super().__init__(
            memory_file=memory_file,
            user_id=user_id,
            db_path=db_path,
            memory_provider=memory_provider,
            session_id=session_id,
            run_id=run_id,
        )

    @property
    def name(self) -> str:
        return "recall_notes"

    @property
    def description(self) -> str:
        return (
            "Recall compact structured memory summaries. "
            "Use scope=user_memory or scope=user_profile for long-term memory, "
            "scope=agent_memory for local notes, or scope=all to combine them."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "enum": _MEMORY_SCOPES,
                    "description": "Which memory layer to recall.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional legacy category filter.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": _MEMORY_TYPES,
                    "description": "Optional structured type filter for user_memory.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum entries per section.",
                },
                "include_superseded": {
                    "type": "boolean",
                    "description": "Whether to include superseded user memories.",
                },
                "detailed": {
                    "type": "boolean",
                    "description": "Whether to include provenance details.",
                },
            },
        }

    async def execute(
        self,
        scope: str = "all",
        category: str | None = None,
        memory_type: str | None = None,
        limit: int = 8,
        include_superseded: bool = False,
        detailed: bool = False,
    ) -> ToolResult:
        try:
            normalized_limit = max(1, min(int(limit), 50))
            memory_provider = self._get_memory_provider()
            resolved_scope = self._normalize_scope(
                scope,
                has_user_store=memory_provider is not None and memory_provider.is_available,
            )
            sections: list[str] = []

            if (
                resolved_scope in {"all", "user_profile"}
                and memory_provider is not None
                and memory_provider.is_available
                and self.user_id
            ):
                profile = memory_provider.get_user_profile(self.user_id)
                if profile is not None:
                    sections.append(self._format_profile_section(profile))

            if (
                resolved_scope in {"all", "user_memory"}
                and memory_provider is not None
                and memory_provider.is_available
                and self.user_id
            ):
                resolved_memory_type = (
                    self._normalize_memory_type(memory_type, category)
                    if memory_type or category
                    else None
                )
                entries = memory_provider.list_memory_entries(
                    self.user_id,
                    memory_types=[resolved_memory_type] if resolved_memory_type else None,
                    include_superseded=include_superseded,
                    limit=normalized_limit,
                )
                if entries:
                    lines = ["User Memory Summary:"]
                    for index, entry in enumerate(entries, start=1):
                        lines.append(
                            self._format_memory_entry(
                                entry,
                                index=index,
                                detailed=detailed,
                            )
                        )
                    sections.append("\n".join(lines))

            if resolved_scope in {"all", "agent_memory"}:
                notes = self._load_agent_notes()
                normalized_category = self._normalize_text(category)
                if normalized_category:
                    notes = [
                        note
                        for note in notes
                        if self._normalize_text(note.get("category")) == normalized_category
                    ]
                notes = notes[:normalized_limit]
                if notes:
                    lines = ["Agent Memory Notes:"]
                    for index, note in enumerate(notes, start=1):
                        lines.append(
                            self._format_memory_entry(
                                note,
                                index=index,
                                detailed=detailed,
                            )
                        )
                    sections.append("\n".join(lines))

            if not sections:
                return ToolResult(success=True, content="No notes recorded yet.")

            return ToolResult(success=True, content="\n\n".join(sections))
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to recall notes: {str(e)}",
            )


class SearchMemoryTool(_MemoryToolSupport, Tool):
    """Search detailed historical user memory or agent-local notes."""

    def __init__(
        self,
        memory_file: str = "./workspace/.agent_memory.json",
        *,
        user_id: str | None = None,
        db_path: str | None = None,
        memory_provider: MemoryProvider | None = None,
        session_id: str | None = None,
        run_id: str | None = None,
    ):
        super().__init__(
            memory_file=memory_file,
            user_id=user_id,
            db_path=db_path,
            memory_provider=memory_provider,
            session_id=session_id,
            run_id=run_id,
        )

    @property
    def name(self) -> str:
        return "search_memory"

    @property
    def description(self) -> str:
        return (
            "Search detailed historical memory. "
            "Prefer this for keyword lookups across long-term user memory or local agent notes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords to search for.",
                },
                "scope": {
                    "type": "string",
                    "enum": _MEMORY_SCOPES,
                    "description": "Which memory layer to search.",
                },
                "memory_type": {
                    "type": "string",
                    "enum": _MEMORY_TYPES,
                    "description": "Optional structured type filter for user_memory.",
                },
                "category": {
                    "type": "string",
                    "description": "Optional category filter for agent_memory.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "description": "Maximum results per section.",
                },
                "include_superseded": {
                    "type": "boolean",
                    "description": "Whether to include superseded user memories.",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        scope: str = "all",
        memory_type: str | None = None,
        category: str | None = None,
        limit: int = 10,
        include_superseded: bool = False,
    ) -> ToolResult:
        try:
            normalized_query = self._normalize_text(query)
            if not normalized_query:
                raise ValueError("query is required.")

            normalized_limit = max(1, min(int(limit), 50))
            memory_provider = self._get_memory_provider()
            resolved_scope = self._normalize_scope(
                scope,
                has_user_store=memory_provider is not None and memory_provider.is_available,
            )
            sections: list[str] = []

            if (
                resolved_scope in {"all", "user_memory"}
                and memory_provider is not None
                and memory_provider.is_available
                and self.user_id
            ):
                resolved_memory_type = (
                    self._normalize_memory_type(memory_type, category)
                    if memory_type or category
                    else None
                )
                entries = memory_provider.search_memory_entries(
                    self.user_id,
                    query=normalized_query,
                    memory_types=[resolved_memory_type] if resolved_memory_type else None,
                    include_superseded=include_superseded,
                    limit=normalized_limit,
                )
                if entries:
                    lines = ["User Memory Search Results:"]
                    for index, entry in enumerate(entries, start=1):
                        lines.append(
                            self._format_memory_entry(
                                entry,
                                index=index,
                                detailed=True,
                            )
                        )
                    sections.append("\n".join(lines))

            if resolved_scope in {"all", "agent_memory"}:
                notes = self._load_agent_notes()
                normalized_category = self._normalize_text(category)
                filtered_notes = []
                for note in notes:
                    if normalized_category and self._normalize_text(note.get("category")) != normalized_category:
                        continue
                    if not self._matches_query(note, normalized_query):
                        continue
                    filtered_notes.append(note)
                filtered_notes = filtered_notes[:normalized_limit]
                if filtered_notes:
                    lines = ["Agent Memory Search Results:"]
                    for index, note in enumerate(filtered_notes, start=1):
                        lines.append(
                            self._format_memory_entry(
                                note,
                                index=index,
                                detailed=True,
                            )
                        )
                    sections.append("\n".join(lines))

            if not sections:
                return ToolResult(success=True, content="No matching memory found.")

            return ToolResult(success=True, content="\n\n".join(sections))
        except Exception as e:
            return ToolResult(
                success=False,
                content="",
                error=f"Failed to search memory: {str(e)}",
            )
