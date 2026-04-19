"""消息渠道路由与绑定管理。"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from ..integration_models import ConversationBindingRecord, IntegrationConfigRecord, RoutingRuleRecord
from ..integration_store import ConversationBindingStore, IntegrationStore
from ..sqlite_schema import utc_now_iso
from .models import ParsedInboundEvent

if TYPE_CHECKING:
    from ..session import SessionManager

ROOT_THREAD_ID = "__root__"
RoutingSource = Literal["binding", "rule", "default", "unbound"]


class RouteResolution(BaseModel):
    """一次消息路由解析结果。"""

    source: RoutingSource
    binding: ConversationBindingRecord | None = None
    matched_rule: RoutingRuleRecord | None = None
    agent_id: str = ""
    session_id: str = ""
    binding_scope: str = ""
    session_key: str = ""
    message: str = ""
    created_binding: bool = False
    disabled_binding_id: str = ""
    disabled_binding_reason: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IntegrationRouter:
    """统一处理显式绑定、规则解析与默认 Agent 回退。"""

    def __init__(self, session_manager: "SessionManager"):
        self._session_manager = session_manager
        self._integration_store: IntegrationStore | None = None
        self._binding_store: ConversationBindingStore | None = None

    def _ensure_stores(self) -> None:
        if self._integration_store is not None and self._binding_store is not None:
            return

        config = self._session_manager._config
        if config is None:
            raise RuntimeError("SessionManager 尚未加载配置。")

        db_path = Path(config.agent.session_store_path)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        resolved_db_path = db_path.resolve()

        self._integration_store = IntegrationStore(resolved_db_path)
        self._binding_store = ConversationBindingStore(resolved_db_path)

    def get_integration(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> IntegrationConfigRecord | None:
        self._ensure_stores()
        integration = self._integration_store.get_integration(integration_id, account_id=account_id)
        if integration is not None:
            return integration
        if strict and account_id is not None and self._integration_store.get_integration(integration_id) is not None:
            raise PermissionError("Integration does not belong to the current account.")
        return None

    def get_binding(
        self,
        binding_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> ConversationBindingRecord | None:
        self._ensure_stores()
        binding = self._binding_store.get_binding(binding_id, account_id=account_id)
        if binding is not None:
            return binding
        if strict and account_id is not None and self._binding_store.get_binding(binding_id) is not None:
            raise PermissionError("Binding does not belong to the current account.")
        return None

    def list_bindings(
        self,
        *,
        account_id: str | None = None,
        integration_id: str,
        tenant_id: str | None = None,
        chat_id: str | None = None,
        thread_id: str | None = None,
        binding_scope: str | None = None,
        agent_id: str | None = None,
        enabled: bool | None = None,
    ) -> list[ConversationBindingRecord]:
        self._ensure_stores()
        normalized_thread_id = self.normalize_thread_id_for_scope(
            thread_id,
            binding_scope=binding_scope,
        )
        return self._binding_store.list_bindings(
            account_id=account_id,
            integration_id=integration_id,
            tenant_id=tenant_id,
            chat_id=chat_id,
            thread_id=normalized_thread_id,
            binding_scope=binding_scope,
            agent_id=agent_id,
            enabled=enabled,
        )

    def build_session_key(
        self,
        *,
        integration_id: str,
        tenant_id: str,
        chat_id: str,
        thread_id: str,
        binding_scope: str,
        agent_id: str,
    ) -> str:
        normalized_thread_id = self.normalize_thread_id_for_scope(
            thread_id,
            binding_scope=binding_scope,
        )
        return ":".join(
            [
                integration_id.strip(),
                tenant_id.strip(),
                chat_id.strip(),
                normalized_thread_id,
                binding_scope.strip(),
                agent_id.strip(),
            ]
        )

    def build_session_key_for_binding(self, binding: ConversationBindingRecord) -> str:
        return self.build_session_key(
            integration_id=binding.integration_id,
            tenant_id=binding.tenant_id,
            chat_id=binding.chat_id,
            thread_id=binding.thread_id,
            binding_scope=binding.binding_scope,
            agent_id=binding.agent_id,
        )

    @classmethod
    def normalize_thread_id_for_scope(
        cls,
        thread_id: str | None,
        *,
        binding_scope: str | None,
    ) -> str | None:
        if binding_scope is None and thread_id is None:
            return None
        normalized_scope = str(binding_scope or "").strip().lower()
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_scope:
            return normalized_thread_id or None
        if normalized_scope == "thread":
            return normalized_thread_id or ROOT_THREAD_ID
        return ROOT_THREAD_ID

    def preferred_binding_scope(
        self,
        parsed_event: ParsedInboundEvent,
        *,
        session_strategy: str = "reuse",
    ) -> str:
        normalized_strategy = str(session_strategy or "reuse").strip().lower()
        if normalized_strategy == "chat":
            return "chat"
        if normalized_strategy == "thread":
            return "thread"
        if not parsed_event.is_group:
            return "thread"
        if str(parsed_event.provider_thread_id or "").strip():
            return "thread"
        return "chat"

    def find_explicit_binding(
        self,
        parsed_event: ParsedInboundEvent,
    ) -> ConversationBindingRecord | None:
        self._ensure_stores()
        tenant_id = str(parsed_event.tenant_id or "").strip()
        chat_id = str(parsed_event.provider_chat_id or "").strip()
        thread_id = str(parsed_event.provider_thread_id or "").strip()

        binding_candidates: list[tuple[str, list[str]]] = []
        if thread_id or not parsed_event.is_group:
            binding_candidates.append(("thread", [thread_id or ROOT_THREAD_ID, ""]))
        binding_candidates.append(("chat", [ROOT_THREAD_ID, ""]))

        for binding_scope, thread_candidates in binding_candidates:
            for candidate_thread_id in thread_candidates:
                binding = self._binding_store.find_binding(
                    integration_id=parsed_event.integration_id,
                    tenant_id=tenant_id,
                    chat_id=chat_id,
                    thread_id=candidate_thread_id,
                    binding_scope=binding_scope,
                    enabled=True,
                )
                if binding is not None:
                    return binding
        return None

    async def resolve_route(
        self,
        integration: IntegrationConfigRecord,
        parsed_event: ParsedInboundEvent,
    ) -> RouteResolution:
        self._ensure_stores()

        disabled_binding_id = ""
        disabled_binding_reason = ""
        explicit_binding = self.find_explicit_binding(parsed_event)
        if explicit_binding is not None:
            explicit_resolution = await self._heal_or_disable_binding(
                explicit_binding,
                received_at=parsed_event.received_at,
            )
            if explicit_resolution.binding is not None:
                return explicit_resolution
            disabled_binding_id = explicit_resolution.disabled_binding_id
            disabled_binding_reason = explicit_resolution.disabled_binding_reason

        for rule in self._integration_store.list_routing_rules(integration.id, enabled=True):
            if not self._rule_matches(parsed_event, rule):
                continue
            if not self._agent_exists(rule.agent_id, account_id=integration.account_id):
                continue

            binding_scope = self.preferred_binding_scope(
                parsed_event,
                session_strategy=rule.session_strategy,
            )
            binding = await self.upsert_binding(
                integration_id=integration.id,
                tenant_id=parsed_event.tenant_id,
                chat_id=parsed_event.provider_chat_id,
                thread_id=parsed_event.provider_thread_id,
                binding_scope=binding_scope,
                agent_id=rule.agent_id,
                metadata={
                    "binding_source": "routing_rule",
                    "routing_rule_id": rule.id,
                    "session_key": self.build_session_key(
                        integration_id=integration.id,
                        tenant_id=parsed_event.tenant_id,
                        chat_id=parsed_event.provider_chat_id,
                        thread_id=parsed_event.provider_thread_id,
                        binding_scope=binding_scope,
                        agent_id=rule.agent_id,
                    ),
                },
                updated_at=parsed_event.received_at,
                touch_last_message=True,
            )
            return RouteResolution(
                source="rule",
                binding=binding,
                matched_rule=rule,
                agent_id=binding.agent_id,
                session_id=binding.session_id,
                binding_scope=binding.binding_scope,
                session_key=self.build_session_key_for_binding(binding),
                message=f"已命中路由规则：{rule.id}",
                created_binding=binding.metadata.get("binding_source") == "routing_rule",
                disabled_binding_id=disabled_binding_id,
                disabled_binding_reason=disabled_binding_reason,
                metadata={"routing_rule_id": rule.id},
            )

        default_agent_id = self._resolve_default_agent_id(integration)
        if default_agent_id and self._agent_exists(
            default_agent_id,
            account_id=integration.account_id,
        ):
            default_scope = self.preferred_binding_scope(
                parsed_event,
                session_strategy=str(integration.config.get("default_session_strategy") or "reuse"),
            )
            binding = await self.upsert_binding(
                integration_id=integration.id,
                tenant_id=parsed_event.tenant_id,
                chat_id=parsed_event.provider_chat_id,
                thread_id=parsed_event.provider_thread_id,
                binding_scope=default_scope,
                agent_id=default_agent_id,
                metadata={
                    "binding_source": "default_agent",
                    "session_key": self.build_session_key(
                        integration_id=integration.id,
                        tenant_id=parsed_event.tenant_id,
                        chat_id=parsed_event.provider_chat_id,
                        thread_id=parsed_event.provider_thread_id,
                        binding_scope=default_scope,
                        agent_id=default_agent_id,
                    ),
                },
                updated_at=parsed_event.received_at,
                touch_last_message=True,
            )
            return RouteResolution(
                source="default",
                binding=binding,
                agent_id=binding.agent_id,
                session_id=binding.session_id,
                binding_scope=binding.binding_scope,
                session_key=self.build_session_key_for_binding(binding),
                message=f"已回退到默认 Agent：{binding.agent_id}",
                created_binding=binding.metadata.get("binding_source") == "default_agent",
                disabled_binding_id=disabled_binding_id,
                disabled_binding_reason=disabled_binding_reason,
            )

        message = "当前会话尚未绑定 Agent，请先使用 /bind 或在 Integrations 页面配置绑定。"
        if disabled_binding_reason:
            message = f"{disabled_binding_reason}\n{message}"
        return RouteResolution(
            source="unbound",
            message=message,
            disabled_binding_id=disabled_binding_id,
            disabled_binding_reason=disabled_binding_reason,
        )

    async def upsert_binding(
        self,
        *,
        account_id: str | None = None,
        integration_id: str,
        tenant_id: str,
        chat_id: str,
        thread_id: str,
        binding_scope: str,
        agent_id: str,
        metadata: dict[str, Any] | None = None,
        updated_at: str | None = None,
        force_new_session: bool = False,
        enabled: bool = True,
        touch_last_message: bool = False,
        preferred_binding_id: str | None = None,
    ) -> ConversationBindingRecord:
        self._ensure_stores()
        integration = self.get_integration(
            integration_id,
            account_id=account_id,
            strict=account_id is not None,
        )
        if integration is None:
            raise KeyError(f"Integration not found: {integration_id}")
        if not self._agent_exists(agent_id, account_id=integration.account_id):
            raise ValueError(f"未找到 Agent：{agent_id}")

        normalized_scope = str(binding_scope or "").strip().lower()
        if normalized_scope not in {"chat", "thread"}:
            raise ValueError("binding_scope 仅支持 chat 或 thread。")

        normalized_thread_id = self.normalize_thread_id_for_scope(
            thread_id,
            binding_scope=normalized_scope,
        )
        now = updated_at or utc_now_iso()
        candidates = self._binding_store.list_bindings(
            account_id=integration.account_id,
            integration_id=integration_id,
            tenant_id=str(tenant_id or "").strip(),
            chat_id=str(chat_id or "").strip(),
            thread_id=normalized_thread_id,
            binding_scope=normalized_scope,
            enabled=None,
        )

        target_binding = self._select_target_binding(
            candidates=candidates,
            agent_id=agent_id,
            preferred_binding_id=preferred_binding_id,
        )

        session_id = ""
        if target_binding is not None and not force_new_session:
            session_id = self._resolve_reusable_session_id(target_binding, agent_id)
        if not session_id:
            session_id = await self._session_manager.create_session(
                agent_id=agent_id,
                account_id=integration.account_id,
            )

        merged_metadata = dict(target_binding.metadata if target_binding is not None else {})
        merged_metadata.update(metadata or {})
        merged_metadata["session_key"] = self.build_session_key(
            integration_id=integration_id,
            tenant_id=str(tenant_id or "").strip(),
            chat_id=str(chat_id or "").strip(),
            thread_id=normalized_thread_id,
            binding_scope=normalized_scope,
            agent_id=agent_id,
        )

        if target_binding is None:
            binding = ConversationBindingRecord(
                id=str(uuid.uuid4()),
                integration_id=integration_id,
                account_id=integration.account_id,
                tenant_id=str(tenant_id or "").strip(),
                chat_id=str(chat_id or "").strip(),
                thread_id=normalized_thread_id,
                binding_scope=normalized_scope,
                agent_id=agent_id,
                session_id=session_id,
                enabled=enabled,
                metadata=merged_metadata,
                created_at=now,
                updated_at=now,
                last_message_at=now if touch_last_message else None,
            )
            binding = self._binding_store.create_binding(binding)
        else:
            binding = target_binding.model_copy(
                update={
                    "tenant_id": str(tenant_id or "").strip(),
                    "chat_id": str(chat_id or "").strip(),
                    "thread_id": normalized_thread_id,
                    "binding_scope": normalized_scope,
                    "agent_id": agent_id,
                    "session_id": session_id,
                    "enabled": enabled,
                    "metadata": merged_metadata,
                    "updated_at": now,
                    "last_message_at": now if touch_last_message else target_binding.last_message_at,
                }
            )
            binding = self._binding_store.update_binding(binding)

        for candidate in candidates:
            if candidate.id == binding.id or not candidate.enabled:
                continue
            self._binding_store.update_binding(
                candidate.model_copy(
                    update={
                        "enabled": False,
                        "updated_at": now,
                    }
                )
            )
        return binding

    def disable_binding(
        self,
        binding_id: str,
        *,
        account_id: str | None = None,
        updated_at: str | None = None,
        reason: str = "",
    ) -> ConversationBindingRecord:
        self._ensure_stores()
        binding = self.get_binding(
            binding_id,
            account_id=account_id,
            strict=account_id is not None,
        )
        if binding is None:
            raise KeyError(f"Conversation binding not found: {binding_id}")

        metadata = dict(binding.metadata)
        if reason:
            metadata["disabled_reason"] = reason

        updated = binding.model_copy(
            update={
                "enabled": False,
                "metadata": metadata,
                "updated_at": updated_at or utc_now_iso(),
            }
        )
        return self._binding_store.update_binding(updated)

    async def reset_binding_session(
        self,
        binding_id: str,
        *,
        account_id: str | None = None,
        updated_at: str | None = None,
    ) -> ConversationBindingRecord:
        self._ensure_stores()
        binding = self.get_binding(
            binding_id,
            account_id=account_id,
            strict=account_id is not None,
        )
        if binding is None:
            raise KeyError(f"Conversation binding not found: {binding_id}")
        if not self._agent_exists(binding.agent_id, account_id=binding.account_id):
            raise ValueError(f"未找到 Agent：{binding.agent_id}")

        session_id = await self._session_manager.create_session(
            agent_id=binding.agent_id,
            account_id=binding.account_id,
        )
        updated = binding.model_copy(
            update={
                "session_id": session_id,
                "enabled": True,
                "updated_at": updated_at or utc_now_iso(),
            }
        )
        return self._binding_store.update_binding(updated)

    async def _heal_or_disable_binding(
        self,
        binding: ConversationBindingRecord,
        *,
        received_at: str,
    ) -> RouteResolution:
        if not self._agent_exists(binding.agent_id, account_id=binding.account_id):
            reason = f"原绑定 Agent 已失效：{binding.agent_id}"
            disabled = self.disable_binding(
                binding.id,
                account_id=binding.account_id,
                updated_at=received_at,
                reason=reason,
            )
            return RouteResolution(
                source="unbound",
                disabled_binding_id=disabled.id,
                disabled_binding_reason=reason,
                message=reason,
            )

        session_id = self._resolve_reusable_session_id(binding, binding.agent_id)
        if session_id and session_id == binding.session_id:
            return RouteResolution(
                source="binding",
                binding=binding,
                agent_id=binding.agent_id,
                session_id=binding.session_id,
                binding_scope=binding.binding_scope,
                session_key=self.build_session_key_for_binding(binding),
                message=f"命中显式绑定：{binding.id}",
            )

        healed_binding = binding.model_copy(
            update={
                "session_id": session_id
                or await self._session_manager.create_session(
                    agent_id=binding.agent_id,
                    account_id=binding.account_id,
                ),
                "enabled": True,
                "updated_at": received_at,
            }
        )
        healed_binding = self._binding_store.update_binding(healed_binding)
        return RouteResolution(
            source="binding",
            binding=healed_binding,
            agent_id=healed_binding.agent_id,
            session_id=healed_binding.session_id,
            binding_scope=healed_binding.binding_scope,
            session_key=self.build_session_key_for_binding(healed_binding),
            message=f"命中显式绑定：{healed_binding.id}",
        )

    def _resolve_default_agent_id(self, integration: IntegrationConfigRecord) -> str:
        config = integration.config or {}
        for key in ("default_agent_id", "default_agent_template_id"):
            value = str(config.get(key) or "").strip()
            if value:
                return value
        return ""

    def _agent_exists(self, agent_id: str, account_id: str | None = None) -> bool:
        if self._session_manager._agent_store is None:
            return False
        return (
            self._session_manager._agent_store.get_agent_template(
                agent_id,
                account_id=account_id,
            )
            is not None
        )

    def _resolve_reusable_session_id(
        self,
        binding: ConversationBindingRecord,
        expected_agent_id: str,
    ) -> str:
        session_info = self._session_manager.get_session_info(
            binding.session_id,
            account_id=binding.account_id,
        )
        if session_info is None:
            return ""
        if str(session_info.get("agent_id") or "").strip() != expected_agent_id:
            return ""
        return binding.session_id

    @staticmethod
    def _rule_matches(parsed_event: ParsedInboundEvent, rule: RoutingRuleRecord) -> bool:
        match_value = str(rule.match_value or "").strip()
        if not match_value:
            return False

        if rule.match_type == "integration_id":
            return match_value in {parsed_event.integration_id, "*"}
        if rule.match_type == "chat_id":
            return match_value == str(parsed_event.provider_chat_id or "").strip()
        if rule.match_type == "thread_id":
            actual_thread_id = str(parsed_event.provider_thread_id or "").strip() or ROOT_THREAD_ID
            return match_value == actual_thread_id
        return False

    @staticmethod
    def _select_target_binding(
        *,
        candidates: list[ConversationBindingRecord],
        agent_id: str,
        preferred_binding_id: str | None,
    ) -> ConversationBindingRecord | None:
        if preferred_binding_id:
            for candidate in candidates:
                if candidate.id == preferred_binding_id:
                    return candidate
        for candidate in candidates:
            if candidate.agent_id == agent_id:
                return candidate
        if preferred_binding_id:
            return None
        return candidates[0] if candidates else None
