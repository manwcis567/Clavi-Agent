"""Long-term memory provider abstractions and default adapters."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .user_memory_store import UserMemoryStore


MemoryProviderKind = Literal["local", "mcp", "disabled"]
MemoryProviderStatus = Literal["ready", "degraded", "disabled"]


class MemoryProviderHealth(BaseModel):
    """Structured health report for the active long-term memory provider."""

    configured_provider: MemoryProviderKind
    active_provider: MemoryProviderKind
    status: MemoryProviderStatus
    fallback_active: bool = False
    inject_memories: bool = True
    expose_tools: bool = True
    sync_conversation_turns: bool = True
    capabilities: dict[str, bool] = Field(default_factory=dict)
    message: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryProvider(ABC):
    """Abstract interface for long-term memory backends."""

    def __init__(
        self,
        *,
        configured_provider: MemoryProviderKind,
        active_provider: MemoryProviderKind,
        inject_memories: bool = True,
        expose_tools: bool = True,
        sync_conversation_turns: bool = True,
        health_status: MemoryProviderStatus | None = None,
        health_message: str = "",
        health_metadata: dict[str, Any] | None = None,
    ):
        self.configured_provider = configured_provider
        self.active_provider = active_provider
        self.inject_memories = bool(inject_memories)
        self.expose_tools = bool(expose_tools)
        self.sync_conversation_turns = bool(sync_conversation_turns)
        self._health_status = health_status
        self._health_message = str(health_message or "").strip()
        self._health_metadata = dict(health_metadata or {})

    @property
    def capabilities(self) -> dict[str, bool]:
        """Return provider capability flags."""
        return {
            "prefetch_relevant_memories": self.is_available,
            "write_memory_entry": self.is_available,
            "sync_completed_conversation_turns": self.is_available,
            "search_memories": self.is_available,
            "profile_modeling": self.is_available,
        }

    @property
    def is_available(self) -> bool:
        """Whether long-term memory operations are available."""
        return True

    def get_health(self) -> dict[str, Any]:
        """Return a JSON-ready health payload."""
        return MemoryProviderHealth(
            configured_provider=self.configured_provider,
            active_provider=self.active_provider,
            status=(
                self._health_status
                if self._health_status is not None
                else ("ready" if self.is_available else "disabled")
            ),
            fallback_active=self.configured_provider != self.active_provider,
            inject_memories=self.inject_memories,
            expose_tools=self.expose_tools,
            sync_conversation_turns=self.sync_conversation_turns,
            capabilities=self.capabilities,
            message=self._health_message,
            metadata=dict(self._health_metadata),
        ).model_dump(mode="python")

    @abstractmethod
    def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Get the effective structured user profile."""

    @abstractmethod
    def inspect_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Get the structured user profile with field metadata."""

    @abstractmethod
    def upsert_user_profile(self, user_id: str, **payload: Any) -> dict[str, Any]:
        """Create or merge a structured user profile."""

    @abstractmethod
    def update_user_profile(self, user_id: str, **payload: Any) -> dict[str, Any] | None:
        """Update one structured user profile."""

    @abstractmethod
    def create_memory_entry(self, **payload: Any) -> dict[str, Any]:
        """Create one long-term memory entry."""

    @abstractmethod
    def list_memory_entries(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        """List long-term memory entries for one user."""

    @abstractmethod
    def search_memory_entries(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        """Search long-term memory entries for one user."""

    @abstractmethod
    def get_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        """Get one long-term memory entry."""

    @abstractmethod
    def update_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        """Update one long-term memory entry."""

    @abstractmethod
    def supersede_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        """Mark one long-term memory entry as superseded."""

    @abstractmethod
    def delete_memory_entry(self, entry_id: str, **payload: Any) -> bool:
        """Soft-delete one long-term memory entry."""

    @abstractmethod
    def list_audit_events(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        """List long-term memory audit events."""

    @abstractmethod
    def enforce_memory_capacity(self, user_id: str, **payload: Any) -> list[str]:
        """Compact weaker entries when a memory bucket exceeds capacity."""

    @abstractmethod
    def sync_completed_conversation_turn(
        self,
        *,
        user_id: str | None,
        session_id: str,
        run_id: str | None,
        messages: list[Any],
    ) -> dict[str, Any]:
        """Sync a completed conversation turn into the provider backend."""


class LocalMemoryProvider(MemoryProvider):
    """Default SQLite-backed long-term memory provider."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        configured_provider: MemoryProviderKind = "local",
        active_provider: MemoryProviderKind = "local",
        inject_memories: bool = True,
        expose_tools: bool = True,
        sync_conversation_turns: bool = True,
        health_status: MemoryProviderStatus | None = None,
        health_message: str = "",
        health_metadata: dict[str, Any] | None = None,
    ):
        super().__init__(
            configured_provider=configured_provider,
            active_provider=active_provider,
            inject_memories=inject_memories,
            expose_tools=expose_tools,
            sync_conversation_turns=sync_conversation_turns,
            health_status=health_status,
            health_message=health_message,
            health_metadata=health_metadata,
        )
        self.db_path = Path(db_path).resolve()
        self._store = UserMemoryStore(self.db_path)

    def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        return self._store.get_user_profile(user_id)

    def inspect_user_profile(self, user_id: str) -> dict[str, Any] | None:
        return self._store.inspect_user_profile(user_id)

    def upsert_user_profile(self, user_id: str, **payload: Any) -> dict[str, Any]:
        return self._store.upsert_user_profile(user_id, **payload)

    def update_user_profile(self, user_id: str, **payload: Any) -> dict[str, Any] | None:
        return self._store.update_user_profile(user_id, **payload)

    def create_memory_entry(self, **payload: Any) -> dict[str, Any]:
        return self._store.create_memory_entry(**payload)

    def list_memory_entries(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        return self._store.list_memory_entries(user_id, **payload)

    def search_memory_entries(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        return self._store.search_memory_entries(user_id, **payload)

    def get_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        return self._store.get_memory_entry(entry_id, **payload)

    def update_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        return self._store.update_memory_entry(entry_id, **payload)

    def supersede_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        return self._store.supersede_memory_entry(entry_id, **payload)

    def delete_memory_entry(self, entry_id: str, **payload: Any) -> bool:
        return self._store.delete_memory_entry(entry_id, **payload)

    def list_audit_events(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        return self._store.list_audit_events(user_id, **payload)

    def enforce_memory_capacity(self, user_id: str, **payload: Any) -> list[str]:
        return self._store.enforce_memory_capacity(user_id, **payload)

    def sync_completed_conversation_turn(
        self,
        *,
        user_id: str | None,
        session_id: str,
        run_id: str | None,
        messages: list[Any],
    ) -> dict[str, Any]:
        if not self.sync_conversation_turns:
            return {
                "provider": self.active_provider,
                "synced": False,
                "reason": "sync_disabled",
            }
        return {
            "provider": self.active_provider,
            "synced": False,
            "reason": "local_session_store_is_already_authoritative",
            "user_id": str(user_id or "").strip() or None,
            "session_id": session_id,
            "run_id": run_id,
            "message_count": len(messages),
        }


class DisabledMemoryProvider(MemoryProvider):
    """Provider implementation used when long-term memory is disabled."""

    def __init__(
        self,
        *,
        configured_provider: MemoryProviderKind = "disabled",
        message: str = "Long-term memory provider is disabled.",
        metadata: dict[str, Any] | None = None,
        inject_memories: bool = False,
        expose_tools: bool = False,
        sync_conversation_turns: bool = False,
    ):
        super().__init__(
            configured_provider=configured_provider,
            active_provider="disabled",
            inject_memories=inject_memories,
            expose_tools=expose_tools,
            sync_conversation_turns=sync_conversation_turns,
        )
        self._message = message
        self._metadata = dict(metadata or {})

    @property
    def is_available(self) -> bool:
        return False

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            "prefetch_relevant_memories": False,
            "write_memory_entry": False,
            "sync_completed_conversation_turns": False,
            "search_memories": False,
            "profile_modeling": False,
        }

    def get_health(self) -> dict[str, Any]:
        return MemoryProviderHealth(
            configured_provider=self.configured_provider,
            active_provider="disabled",
            status="disabled",
            fallback_active=False,
            inject_memories=self.inject_memories,
            expose_tools=self.expose_tools,
            sync_conversation_turns=self.sync_conversation_turns,
            capabilities=self.capabilities,
            message=self._message,
            metadata=self._metadata,
        ).model_dump(mode="python")

    def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        return None

    def inspect_user_profile(self, user_id: str) -> dict[str, Any] | None:
        return None

    def upsert_user_profile(self, user_id: str, **payload: Any) -> dict[str, Any]:
        raise RuntimeError(self._message)

    def update_user_profile(self, user_id: str, **payload: Any) -> dict[str, Any] | None:
        return None

    def create_memory_entry(self, **payload: Any) -> dict[str, Any]:
        raise RuntimeError(self._message)

    def list_memory_entries(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        return []

    def search_memory_entries(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        return []

    def get_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        return None

    def update_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        return None

    def supersede_memory_entry(self, entry_id: str, **payload: Any) -> dict[str, Any] | None:
        return None

    def delete_memory_entry(self, entry_id: str, **payload: Any) -> bool:
        return False

    def list_audit_events(self, user_id: str, **payload: Any) -> list[dict[str, Any]]:
        return []

    def enforce_memory_capacity(self, user_id: str, **payload: Any) -> list[str]:
        return []

    def sync_completed_conversation_turn(
        self,
        *,
        user_id: str | None,
        session_id: str,
        run_id: str | None,
        messages: list[Any],
    ) -> dict[str, Any]:
        return {
            "provider": "disabled",
            "synced": False,
            "reason": "provider_disabled",
            "user_id": str(user_id or "").strip() or None,
            "session_id": session_id,
            "run_id": run_id,
            "message_count": len(messages),
        }


class MCPMemoryProviderAdapter(LocalMemoryProvider):
    """Adapter shell for the legacy disabled MCP memory configuration."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        mcp_config_path: str | Path | None,
        mcp_server_name: str,
        inject_memories: bool = True,
        expose_tools: bool = True,
        sync_conversation_turns: bool = True,
    ):
        super().__init__(
            db_path,
            configured_provider="mcp",
            active_provider="local",
            inject_memories=inject_memories,
            expose_tools=expose_tools,
            sync_conversation_turns=sync_conversation_turns,
        )
        self.mcp_config_path = (
            Path(mcp_config_path).resolve()
            if mcp_config_path is not None
            else None
        )
        self.mcp_server_name = str(mcp_server_name or "memory").strip() or "memory"
        self._server_metadata = self._read_server_metadata()

    def _read_server_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "server_name": self.mcp_server_name,
            "config_path": (
                str(self.mcp_config_path)
                if self.mcp_config_path is not None
                else None
            ),
            "server_configured": False,
            "server_disabled": False,
        }
        if self.mcp_config_path is None or not self.mcp_config_path.exists():
            metadata["message"] = "MCP config file not found; falling back to local memory provider."
            return metadata

        try:
            payload = json.loads(self.mcp_config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            metadata["message"] = (
                f"Failed to read MCP config with UTF-8: {exc}; falling back to local memory provider."
            )
            return metadata

        servers = payload.get("mcpServers", {})
        if not isinstance(servers, dict):
            metadata["message"] = (
                "MCP config has no valid mcpServers object; falling back to local memory provider."
            )
            return metadata

        server_config = servers.get(self.mcp_server_name)
        if not isinstance(server_config, dict):
            metadata["message"] = (
                f"MCP server '{self.mcp_server_name}' is not configured; falling back to local memory provider."
            )
            return metadata

        metadata["server_configured"] = True
        metadata["server_disabled"] = bool(server_config.get("disabled", False))
        metadata["description"] = str(server_config.get("description", "")).strip()
        metadata["command"] = str(server_config.get("command", "")).strip()
        metadata["args"] = list(server_config.get("args", []))
        if metadata["server_disabled"]:
            metadata["message"] = (
                f"MCP server '{self.mcp_server_name}' is disabled; falling back to local memory provider."
            )
        else:
            metadata["message"] = (
                f"MCP server '{self.mcp_server_name}' is configured, but the protocol-specific "
                "memory bridge is not enabled yet; falling back to local memory provider."
            )
        return metadata

    def get_health(self) -> dict[str, Any]:
        return MemoryProviderHealth(
            configured_provider="mcp",
            active_provider="local",
            status="degraded",
            fallback_active=True,
            inject_memories=self.inject_memories,
            expose_tools=self.expose_tools,
            sync_conversation_turns=self.sync_conversation_turns,
            capabilities=self.capabilities,
            message=str(self._server_metadata.get("message") or "").strip(),
            metadata=dict(self._server_metadata),
        ).model_dump(mode="python")


def build_memory_provider(
    *,
    configured_provider: MemoryProviderKind,
    db_path: str | Path,
    inject_memories: bool = True,
    expose_tools: bool = True,
    sync_conversation_turns: bool = True,
    enable_external_providers: bool = True,
    allow_fallback_to_local: bool = True,
    mcp_config_path: str | Path | None = None,
    mcp_server_name: str = "memory",
) -> MemoryProvider:
    """Create the configured memory provider with safe local fallback."""
    provider_kind = str(configured_provider or "local").strip().lower() or "local"
    if provider_kind == "disabled":
        return DisabledMemoryProvider(
            configured_provider="disabled",
            message="Long-term memory provider is disabled by configuration.",
            metadata={"configured_provider": "disabled"},
            inject_memories=False,
            expose_tools=False,
            sync_conversation_turns=False,
        )
    if provider_kind == "mcp":
        if not enable_external_providers:
            return LocalMemoryProvider(
                db_path,
                configured_provider="mcp",
                active_provider="local",
                inject_memories=inject_memories,
                expose_tools=expose_tools,
                sync_conversation_turns=sync_conversation_turns,
                health_status="degraded",
                health_message=(
                    "External memory providers are disabled by feature flag; "
                    "falling back to local memory provider."
                ),
                health_metadata={
                    "reason": "feature_flag_disabled",
                    "feature_flag": "enable_external_memory_providers",
                    "mcp_server_name": str(mcp_server_name or "memory").strip() or "memory",
                },
            )
        adapter = MCPMemoryProviderAdapter(
            db_path,
            mcp_config_path=mcp_config_path,
            mcp_server_name=mcp_server_name,
            inject_memories=inject_memories,
            expose_tools=expose_tools,
            sync_conversation_turns=sync_conversation_turns,
        )
        if allow_fallback_to_local:
            return adapter
        return DisabledMemoryProvider(
            configured_provider="mcp",
            message=str(adapter.get_health().get("message") or "").strip()
            or "External MCP memory provider is unavailable and local fallback is disabled.",
            metadata=adapter.get_health().get("metadata", {}),
            inject_memories=False,
            expose_tools=False,
            sync_conversation_turns=False,
        )
    return LocalMemoryProvider(
        db_path,
        configured_provider="local",
        active_provider="local",
        inject_memories=inject_memories,
        expose_tools=expose_tools,
        sync_conversation_turns=sync_conversation_turns,
    )
