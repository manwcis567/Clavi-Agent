"""渠道集成管理后台服务。"""

from __future__ import annotations

import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..integration_models import (
    IntegrationConfigRecord,
    IntegrationCredentialRecord,
    OutboundDeliveryRecord,
    RoutingRuleRecord,
    mask_secret,
)
from ..integration_store import (
    ConversationBindingStore,
    DeliveryStore,
    InboundEventStore,
    IntegrationStore,
)
from ..sqlite_schema import utc_now_iso
from .adapter_base import ChannelAdapterRegistry
from .feishu_adapter import FeishuAdapter
from .mock_adapter import MockChannelAdapter
from .wechat_adapter import WeChatAdapter

if TYPE_CHECKING:
    from ..session import SessionManager

_VALID_INTEGRATION_ID_PATTERN = re.compile(r"[^a-z0-9]+")
_SUPPORTED_CREDENTIAL_STORAGE_KINDS = {"env", "external_ref", "local_encrypted"}
_SUPPORTED_ROUTING_MATCH_TYPES = {"integration_id", "chat_id", "thread_id"}
_SUPPORTED_SESSION_STRATEGIES = {"reuse", "chat", "thread"}


@dataclass(slots=True)
class IntegrationVerificationResult:
    ok: bool
    message: str
    integration: IntegrationConfigRecord


def _default_adapter_registry() -> ChannelAdapterRegistry:
    return ChannelAdapterRegistry([MockChannelAdapter(), FeishuAdapter(), WeChatAdapter()])


def _validate_integration_record(
    service: "IntegrationAdminService",
    record: IntegrationConfigRecord,
) -> tuple[bool, str]:
    if not service._adapter_registry.has(record.kind):
        return False, f"未注册的渠道类型：{record.kind}"

    resolved_credentials = service._resolve_credentials(record.id)
    config = dict(record.config)

    if record.kind == "mock":
        if bool(config.get("require_signing_secret")) and not resolved_credentials.get("signing_secret"):
            return False, "Mock 渠道开启签名校验时，必须提供 signing_secret 凭证。"
        return True, "Mock 渠道配置已通过本地校验，可用于本地联调。"

    if record.kind == "feishu":
        connection_mode = FeishuAdapter.connection_mode(config)
        default_agent_id = str(
            config.get("default_agent_id") or config.get("default_agent_template_id") or ""
        ).strip()
        missing_fields: list[str] = []
        if not str(config.get("app_id") or "").strip():
            missing_fields.append("config.app_id")
        if connection_mode == "webhook" and not (
            str(config.get("verification_token") or "").strip()
            or str(config.get("verify_token") or "").strip()
            or resolved_credentials.get("verification_token")
            or resolved_credentials.get("verify_token")
        ):
            missing_fields.append("verification_token")
        if not (
            resolved_credentials.get("app_secret")
            or str(config.get("app_secret") or "").strip()
        ):
            missing_fields.append("app_secret")
        if missing_fields:
            return False, f"飞书配置不完整，缺少：{', '.join(missing_fields)}。"
        if default_agent_id and not service._agent_exists(
            default_agent_id,
            account_id=record.account_id,
        ):
            return False, f"飞书默认 Agent 不存在：{default_agent_id}。"
        if connection_mode == "webhook":
            return True, "飞书 webhook 配置已通过本地校验，可继续执行事件订阅联调。"
        return True, "飞书长连接配置已通过本地校验，启用后服务会主动建立连接。"

    if record.kind == "wechat":
        default_agent_id = str(config.get("default_agent_id") or "").strip()
        setup_state = record.metadata.get("wechat_setup")
        setup_payload = setup_state if isinstance(setup_state, dict) else {}
        normalized_setup_state = str(setup_payload.get("state") or "").strip().lower()
        bot_token = str(resolved_credentials.get("bot_token") or "").strip()
        ilink_bot_id = str(resolved_credentials.get("ilink_bot_id") or "").strip()
        if not default_agent_id:
            return False, "微信渠道缺少默认 Agent，请先选择一个需要绑定的 Agent。"
        if not service._agent_exists(default_agent_id, account_id=record.account_id):
            return False, f"微信默认 Agent 不存在：{default_agent_id}。"
        if normalized_setup_state == "succeeded" and (not bot_token or not ilink_bot_id):
            return False, "微信扫码状态已完成，但缺少 iLink 凭证，请重新扫码连接。"
        if normalized_setup_state == "succeeded":
            return True, "微信 iLink 连接已完成，当前默认 Agent 可用。"
        if normalized_setup_state in {"queued", "running", "waiting_scan", "scanned"}:
            return False, "微信扫码连接仍在进行中，请先完成扫码后再校验。"
        setup_error = str(setup_payload.get("error") or "").strip()
        if setup_error:
            return False, f"微信扫码未完成：{setup_error}"
        return False, "微信渠道尚未完成 iLink 扫码配对，请先在集成页面生成二维码并用微信扫码。"

    return True, f"{record.kind} 配置已通过基础校验。"


class IntegrationAdminService:
    """提供 Integrations 页面所需的配置、日志与运维管理能力。"""

    def __init__(
        self,
        session_manager: "SessionManager",
        *,
        adapter_registry: ChannelAdapterRegistry | None = None,
    ):
        self._session_manager = session_manager
        self._adapter_registry = adapter_registry or _default_adapter_registry()
        self._integration_store: IntegrationStore | None = None
        self._binding_store: ConversationBindingStore | None = None
        self._inbound_event_store: InboundEventStore | None = None
        self._delivery_store: DeliveryStore | None = None

    async def list_integrations(
        self,
        *,
        account_id: str | None = None,
        kind: str | None = None,
        status: str | None = None,
        include_deleted: bool = False,
    ) -> list[IntegrationConfigRecord]:
        await self._session_manager.initialize()
        self._ensure_stores()
        records = self._integration_store.list_integrations(
            account_id=account_id,
            kind=kind,
            status=status,
        )
        if include_deleted:
            return records
        return [record for record in records if not self.is_deleted(record)]

    async def get_integration(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> IntegrationConfigRecord | None:
        await self._session_manager.initialize()
        self._ensure_stores()
        record = self._integration_store.get_integration(integration_id, account_id=account_id)
        if record is not None:
            return record
        if strict and account_id is not None and self._integration_store.get_integration(integration_id) is not None:
            raise PermissionError("Integration does not belong to the current account.")
        return None

    async def create_integration(
        self,
        *,
        account_id: str | None = None,
        name: str,
        kind: str,
        display_name: str = "",
        tenant_id: str = "",
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        credentials: list[dict[str, Any]] | None = None,
        enabled: bool = False,
    ) -> IntegrationConfigRecord:
        await self._session_manager.initialize()
        self._ensure_stores()

        normalized_kind = str(kind or "").strip().lower()
        if not normalized_kind:
            raise ValueError("集成类型不能为空。")
        if not self._adapter_registry.has(normalized_kind):
            raise ValueError(f"暂不支持的集成类型：{normalized_kind}")

        normalized_name = str(name or "").strip()
        if not normalized_name:
            raise ValueError("集成名称不能为空。")

        now = utc_now_iso()
        integration_id = self._generate_integration_id(normalized_name, normalized_kind)
        record = IntegrationConfigRecord(
            id=integration_id,
            account_id=account_id or IntegrationConfigRecord.model_fields["account_id"].default,
            name=normalized_name,
            kind=normalized_kind,
            status="active" if enabled else "disabled",
            display_name=str(display_name or normalized_name).strip(),
            tenant_id=str(tenant_id or "").strip(),
            webhook_path=self.build_webhook_path(normalized_kind, integration_id),
            config=dict(config or {}),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._integration_store.create_integration(record)
        self._replace_credentials(
            integration_id=integration_id,
            account_id=record.account_id,
            credentials=credentials or [],
            updated_at=now,
        )
        return self._integration_store.get_integration(
            integration_id,
            account_id=record.account_id,
        ) or record

    async def update_integration(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
        name: str | None = None,
        kind: str | None = None,
        display_name: str | None = None,
        tenant_id: str | None = None,
        config: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        credentials: list[dict[str, Any]] | None = None,
    ) -> IntegrationConfigRecord:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = self._require_integration(integration_id, account_id=account_id)

        normalized_kind = existing.kind if kind is None else str(kind).strip().lower()
        if not self._adapter_registry.has(normalized_kind):
            raise ValueError(f"暂不支持的集成类型：{normalized_kind}")

        updated_metadata = dict(existing.metadata) if metadata is None else dict(metadata)
        record = existing.model_copy(
            update={
                "name": existing.name if name is None else str(name or "").strip() or existing.name,
                "kind": normalized_kind,
                "display_name": (
                    existing.display_name
                    if display_name is None
                    else str(display_name or "").strip()
                )
                or (str(name or "").strip() or existing.name),
                "tenant_id": existing.tenant_id if tenant_id is None else str(tenant_id or "").strip(),
                "webhook_path": self.build_webhook_path(normalized_kind, integration_id),
                "config": existing.config if config is None else dict(config),
                "metadata": updated_metadata,
                "updated_at": utc_now_iso(),
            }
        )
        self._integration_store.update_integration(record)
        if credentials is not None:
            self._replace_credentials(
                integration_id=integration_id,
                account_id=existing.account_id,
                credentials=credentials,
                updated_at=record.updated_at,
            )
        return self._integration_store.get_integration(
            integration_id,
            account_id=existing.account_id,
        ) or record

    async def verify_integration(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
    ) -> IntegrationVerificationResult:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = self._require_integration(integration_id, account_id=account_id)
        ok, message = _validate_integration_record(self, existing)
        now = utc_now_iso()

        next_status = existing.status
        if ok:
            if next_status == "error":
                next_status = "disabled"
            updated = existing.model_copy(
                update={
                    "status": next_status,
                    "last_verified_at": now,
                    "last_error": "",
                    "updated_at": now,
                }
            )
        else:
            updated = existing.model_copy(
                update={
                    "status": "error" if not self.is_deleted(existing) else existing.status,
                    "last_error": message,
                    "updated_at": now,
                }
            )
        self._integration_store.update_integration(updated)
        return IntegrationVerificationResult(
            ok=ok,
            message=message,
            integration=self._integration_store.get_integration(
                integration_id,
                account_id=existing.account_id,
            ) or updated,
        )

    async def set_integration_status(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
        status: str,
    ) -> IntegrationConfigRecord:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = self._require_integration(integration_id, account_id=account_id)
        if self.is_deleted(existing):
            raise ValueError("已删除的集成不能再启用或停用。")
        normalized_status = str(status or "").strip().lower()
        if normalized_status not in {"active", "disabled"}:
            raise ValueError(f"不支持的集成状态：{normalized_status}")
        now = utc_now_iso()
        if normalized_status == "active":
            ok, message = _validate_integration_record(self, existing)
            if not ok:
                errored = existing.model_copy(
                    update={
                        "status": "error",
                        "last_error": message,
                        "updated_at": now,
                    }
                )
                self._integration_store.update_integration(errored)
                raise ValueError(message)
        updated = existing.model_copy(
            update={
                "status": normalized_status,
                "updated_at": now,
                "last_error": "" if normalized_status == "active" else existing.last_error,
                "last_verified_at": now if normalized_status == "active" else existing.last_verified_at,
            }
        )
        self._integration_store.update_integration(updated)
        return self._integration_store.get_integration(
            integration_id,
            account_id=existing.account_id,
        ) or updated

    async def soft_delete_integration(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
    ) -> IntegrationConfigRecord:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = self._require_integration(integration_id, account_id=account_id)
        now = utc_now_iso()
        metadata = dict(existing.metadata)
        metadata["deleted_at"] = now
        metadata["deleted_by"] = "api"
        updated = existing.model_copy(
            update={
                "status": "disabled",
                "metadata": metadata,
                "updated_at": now,
            }
        )
        self._integration_store.update_integration(updated)
        self._disable_related_bindings(
            integration_id,
            account_id=existing.account_id,
            updated_at=now,
        )
        self._disable_related_routing_rules(
            integration_id,
            account_id=existing.account_id,
            updated_at=now,
        )
        return self._integration_store.get_integration(
            integration_id,
            account_id=existing.account_id,
        ) or updated

    async def list_credentials(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
    ) -> list[IntegrationCredentialRecord]:
        await self._session_manager.initialize()
        self._ensure_stores()
        integration = self._require_integration(integration_id, account_id=account_id)
        return self._integration_store.list_credentials(
            integration_id,
            account_id=integration.account_id,
        )

    async def list_events(
        self,
        *,
        account_id: str | None = None,
        integration_id: str,
        status: str | None = None,
        limit: int = 20,
    ):
        await self._session_manager.initialize()
        self._ensure_stores()
        integration = self._require_integration(integration_id, account_id=account_id)
        return self._inbound_event_store.list_events(
            account_id=integration.account_id,
            integration_id=integration_id,
            status=status,
            limit=limit,
        )

    async def list_deliveries(
        self,
        *,
        account_id: str | None = None,
        integration_id: str,
        status: str | None = None,
        limit: int = 20,
    ) -> list[OutboundDeliveryRecord]:
        await self._session_manager.initialize()
        self._ensure_stores()
        integration = self._require_integration(integration_id, account_id=account_id)
        return self._delivery_store.list_deliveries(
            account_id=integration.account_id,
            integration_id=integration_id,
            status=status,
            limit=limit,
        )

    async def list_delivery_attempts(
        self,
        delivery_id: str,
        *,
        account_id: str | None = None,
    ):
        await self._session_manager.initialize()
        self._ensure_stores()
        delivery = self._delivery_store.get_delivery(delivery_id, account_id=account_id)
        if delivery is not None:
            return self._delivery_store.list_attempts(delivery_id, account_id=delivery.account_id)
        if account_id is not None and self._delivery_store.get_delivery(delivery_id) is not None:
            raise PermissionError("Outbound delivery does not belong to the current account.")
        if self._delivery_store.get_delivery(delivery_id) is None:
            raise KeyError(f"Outbound delivery not found: {delivery_id}")
        return self._delivery_store.list_attempts(delivery_id)

    async def get_routing_rule(
        self,
        rule_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> RoutingRuleRecord | None:
        await self._session_manager.initialize()
        self._ensure_stores()
        rule = self._integration_store.get_routing_rule(rule_id, account_id=account_id)
        if rule is not None:
            return rule
        if strict and account_id is not None and self._integration_store.get_routing_rule(rule_id) is not None:
            raise PermissionError("Routing rule does not belong to the current account.")
        return None

    async def list_routing_rules(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
        enabled: bool | None = None,
    ) -> list[RoutingRuleRecord]:
        await self._session_manager.initialize()
        self._ensure_stores()
        integration = self._require_integration(integration_id, account_id=account_id)
        return self._integration_store.list_routing_rules(
            integration_id,
            account_id=integration.account_id,
            enabled=enabled,
        )

    async def create_routing_rule(
        self,
        *,
        account_id: str | None = None,
        integration_id: str,
        priority: int,
        match_type: str,
        match_value: str,
        agent_id: str,
        session_strategy: str,
        enabled: bool,
        metadata: dict[str, Any] | None = None,
    ) -> RoutingRuleRecord:
        await self._session_manager.initialize()
        self._ensure_stores()
        integration = self._require_integration(integration_id, account_id=account_id)
        self._require_agent(agent_id, account_id=integration.account_id)

        normalized_match_type = str(match_type or "").strip().lower()
        normalized_session_strategy = str(session_strategy or "reuse").strip().lower()
        normalized_match_value = str(match_value or "").strip()
        if normalized_match_type not in _SUPPORTED_ROUTING_MATCH_TYPES:
            raise ValueError(f"不支持的路由匹配类型：{normalized_match_type}")
        if normalized_session_strategy not in _SUPPORTED_SESSION_STRATEGIES:
            raise ValueError(f"不支持的会话策略：{normalized_session_strategy}")
        if not normalized_match_value:
            raise ValueError("路由匹配值不能为空。")

        now = utc_now_iso()
        rule = RoutingRuleRecord(
            id=str(uuid.uuid4()),
            integration_id=integration_id,
            account_id=integration.account_id,
            priority=int(priority),
            match_type=normalized_match_type,
            match_value=normalized_match_value,
            agent_id=agent_id,
            session_strategy=normalized_session_strategy,
            enabled=bool(enabled),
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        return self._integration_store.create_routing_rule(rule)

    async def update_routing_rule(
        self,
        rule_id: str,
        *,
        account_id: str | None = None,
        priority: int | None = None,
        match_type: str | None = None,
        match_value: str | None = None,
        agent_id: str | None = None,
        session_strategy: str | None = None,
        enabled: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RoutingRuleRecord:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = await self.get_routing_rule(rule_id, account_id=account_id, strict=True)
        if existing is None:
            raise KeyError(f"Routing rule not found: {rule_id}")

        next_agent_id = existing.agent_id if agent_id is None else str(agent_id or "").strip()
        self._require_agent(next_agent_id, account_id=existing.account_id)

        next_match_type = existing.match_type if match_type is None else str(match_type or "").strip().lower()
        next_match_value = existing.match_value if match_value is None else str(match_value or "").strip()
        next_session_strategy = (
            existing.session_strategy
            if session_strategy is None
            else str(session_strategy or "").strip().lower()
        )
        if next_match_type not in _SUPPORTED_ROUTING_MATCH_TYPES:
            raise ValueError(f"不支持的路由匹配类型：{next_match_type}")
        if next_session_strategy not in _SUPPORTED_SESSION_STRATEGIES:
            raise ValueError(f"不支持的会话策略：{next_session_strategy}")
        if not next_match_value:
            raise ValueError("路由匹配值不能为空。")

        updated = existing.model_copy(
            update={
                "priority": existing.priority if priority is None else int(priority),
                "match_type": next_match_type,
                "match_value": next_match_value,
                "agent_id": next_agent_id,
                "session_strategy": next_session_strategy,
                "enabled": existing.enabled if enabled is None else bool(enabled),
                "metadata": dict(existing.metadata) if metadata is None else dict(metadata),
                "updated_at": utc_now_iso(),
            }
        )
        self._integration_store.update_routing_rule(updated)
        return self._integration_store.get_routing_rule(
            rule_id,
            account_id=existing.account_id,
        ) or updated

    async def delete_routing_rule(
        self,
        rule_id: str,
        *,
        account_id: str | None = None,
    ) -> RoutingRuleRecord:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = await self.get_routing_rule(rule_id, account_id=account_id, strict=True)
        if existing is None:
            raise KeyError(f"Routing rule not found: {rule_id}")
        self._integration_store.delete_routing_rule(rule_id, account_id=existing.account_id)
        return existing

    async def retry_delivery(
        self,
        delivery_id: str,
        *,
        account_id: str | None = None,
    ) -> OutboundDeliveryRecord:
        await self._session_manager.initialize()
        dispatcher = self._session_manager._integration_reply_dispatcher
        if dispatcher is None:
            raise RuntimeError("IntegrationReplyDispatcher 尚未初始化。")
        delivery = self._delivery_store.get_delivery(delivery_id, account_id=account_id)
        if delivery is None:
            if account_id is not None and self._delivery_store.get_delivery(delivery_id) is not None:
                raise PermissionError("Outbound delivery does not belong to the current account.")
            raise KeyError(f"Outbound delivery not found: {delivery_id}")
        return await dispatcher.retry_delivery(delivery_id)

    @staticmethod
    def build_webhook_path(kind: str, integration_id: str) -> str:
        return f"/api/integrations/{kind}/{integration_id}/webhook"

    @staticmethod
    def is_deleted(record: IntegrationConfigRecord) -> bool:
        return bool(str(record.metadata.get("deleted_at") or "").strip())

    def _ensure_stores(self) -> None:
        if (
            self._integration_store is not None
            and self._binding_store is not None
            and self._inbound_event_store is not None
            and self._delivery_store is not None
        ):
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
        self._inbound_event_store = InboundEventStore(resolved_db_path)
        self._delivery_store = DeliveryStore(resolved_db_path)

    def _require_integration(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
    ) -> IntegrationConfigRecord:
        integration = self._integration_store.get_integration(integration_id, account_id=account_id)
        if integration is not None:
            return integration
        if account_id is not None and self._integration_store.get_integration(integration_id) is not None:
            raise PermissionError("Integration does not belong to the current account.")
        integration = self._integration_store.get_integration(integration_id)
        if integration is None:
            raise KeyError(f"Integration not found: {integration_id}")
        return integration

    def _require_agent(self, agent_id: str, *, account_id: str | None = None) -> None:
        normalized_agent_id = str(agent_id or "").strip()
        if not normalized_agent_id:
            raise ValueError("Agent ID 不能为空。")
        if not self._agent_exists(normalized_agent_id, account_id=account_id):
            raise ValueError(f"Agent 不存在：{normalized_agent_id}")

    def _agent_exists(self, agent_id: str, account_id: str | None = None) -> bool:
        normalized_agent_id = str(agent_id or "").strip()
        if not normalized_agent_id:
            return False
        agent_store = self._session_manager._agent_store
        return (
            agent_store is not None
            and agent_store.get_agent_template(
                normalized_agent_id,
                account_id=account_id,
            )
            is not None
        )

    def _generate_integration_id(self, name: str, kind: str) -> str:
        normalized_name = _VALID_INTEGRATION_ID_PATTERN.sub("-", name.strip().lower()).strip("-")
        normalized_name = normalized_name[:48]
        base = normalized_name or f"{kind}-integration"
        if not base.startswith(f"{kind}-"):
            base = f"{kind}-{base}"

        candidate = base
        counter = 2
        while self._integration_store.get_integration(candidate) is not None:
            candidate = f"{base}-{counter}"
            counter += 1
        return candidate

    def _replace_credentials(
        self,
        *,
        integration_id: str,
        account_id: str | None,
        credentials: list[dict[str, Any]],
        updated_at: str,
    ) -> None:
        existing_records = {
            record.credential_key: record
            for record in self._integration_store.list_credentials(
                integration_id,
                account_id=account_id,
            )
        }
        seen_keys: set[str] = set()

        for raw_item in credentials:
            item = dict(raw_item or {})
            credential_key = str(item.get("credential_key") or "").strip()
            if not credential_key:
                raise ValueError("凭证键不能为空。")
            if credential_key in seen_keys:
                raise ValueError(f"凭证键重复：{credential_key}")

            storage_kind = str(item.get("storage_kind") or "env").strip()
            if storage_kind not in _SUPPORTED_CREDENTIAL_STORAGE_KINDS:
                raise ValueError(f"不支持的凭证存储类型：{storage_kind}")

            existing = existing_records.get(credential_key)
            secret_ref = str(item.get("secret_ref") or "").strip()
            secret_value = str(item.get("secret_value") or "")
            secret_ciphertext = ""
            if storage_kind == "local_encrypted":
                secret_ciphertext = secret_value or (
                    existing.secret_ciphertext
                    if existing is not None and existing.storage_kind == storage_kind
                    else ""
                )
                if not secret_ciphertext:
                    raise ValueError(f"凭证 {credential_key} 缺少密文内容。")
            else:
                if not secret_ref and existing is not None and existing.storage_kind == storage_kind:
                    secret_ref = existing.secret_ref
                if not secret_ref:
                    raise ValueError(f"凭证 {credential_key} 缺少 secret_ref。")

            display_source = (
                secret_ref
                or secret_value
                or (existing.masked_value if existing is not None else "")
            )
            record = IntegrationCredentialRecord(
                id=existing.id if existing is not None else f"{integration_id}:{credential_key}",
                integration_id=integration_id,
                credential_key=credential_key,
                storage_kind=storage_kind,
                secret_ref=secret_ref,
                secret_ciphertext=secret_ciphertext,
                masked_value=mask_secret(display_source),
                metadata=dict(item.get("metadata") or {}),
                created_at=existing.created_at if existing is not None else updated_at,
                updated_at=updated_at,
            )
            if existing is None:
                self._integration_store.create_credential(record)
            else:
                self._integration_store.update_credential(record)
            seen_keys.add(credential_key)

        for credential_key, existing in existing_records.items():
            if credential_key in seen_keys:
                continue
            self._integration_store.delete_credential(existing.id, account_id=account_id)

    def _disable_related_bindings(
        self,
        integration_id: str,
        *,
        account_id: str | None,
        updated_at: str,
    ) -> None:
        for binding in self._binding_store.list_bindings(
            integration_id=integration_id,
            account_id=account_id,
        ):
            if not binding.enabled:
                continue
            metadata = dict(binding.metadata)
            metadata["disabled_reason"] = "integration_deleted"
            self._binding_store.update_binding(
                binding.model_copy(
                    update={
                        "enabled": False,
                        "metadata": metadata,
                        "updated_at": updated_at,
                    }
                )
            )

    def _disable_related_routing_rules(
        self,
        integration_id: str,
        *,
        account_id: str | None,
        updated_at: str,
    ) -> None:
        for rule in self._integration_store.list_routing_rules(
            integration_id,
            account_id=account_id,
            enabled=None,
        ):
            if not rule.enabled:
                continue
            metadata = dict(rule.metadata)
            metadata["disabled_reason"] = "integration_deleted"
            self._integration_store.update_routing_rule(
                rule.model_copy(
                    update={
                        "enabled": False,
                        "metadata": metadata,
                        "updated_at": updated_at,
                    }
                )
            )

    def _resolve_credentials(
        self,
        integration_id: str,
        *,
        account_id: str | None = None,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for record in self._integration_store.list_credentials(
            integration_id,
            account_id=account_id,
        ):
            value = self._resolve_credential_value(record)
            if value:
                resolved[record.credential_key] = value
        return resolved

    @staticmethod
    def _resolve_credential_value(record: IntegrationCredentialRecord) -> str:
        if record.storage_kind == "local_encrypted":
            return record.secret_ciphertext.strip()

        reference = record.secret_ref.strip()
        if not reference:
            return ""
        if record.storage_kind == "env":
            env_name = reference[4:] if reference.startswith("env:") else reference
            return os.environ.get(env_name, "").strip()
        return reference

    def _validate_integration(self, record: IntegrationConfigRecord) -> tuple[bool, str]:
        if not self._adapter_registry.has(record.kind):
            return False, f"未注册的渠道类型：{record.kind}"

        resolved_credentials = self._resolve_credentials(record.id)
        config = dict(record.config)

        if record.kind == "mock":
            if bool(config.get("require_signing_secret")) and not resolved_credentials.get("signing_secret"):
                return False, "Mock 渠道开启签名校验时，必须提供 signing_secret 凭证。"
            return True, "Mock 渠道配置已通过本地校验，可用于本地联调。"

        if record.kind == "feishu":
            connection_mode = FeishuAdapter.connection_mode(config)
            default_agent_id = str(
                config.get("default_agent_id") or config.get("default_agent_template_id") or ""
            ).strip()
            missing_fields: list[str] = []
            if not str(config.get("app_id") or "").strip():
                missing_fields.append("config.app_id")
            if connection_mode == "webhook" and not (
                str(config.get("verification_token") or "").strip()
                or str(config.get("verify_token") or "").strip()
                or resolved_credentials.get("verification_token")
                or resolved_credentials.get("verify_token")
            ):
                missing_fields.append("verification_token")
            if not (
                resolved_credentials.get("app_secret")
                or str(config.get("app_secret") or "").strip()
            ):
                missing_fields.append("app_secret")
            if missing_fields:
                return False, f"飞书配置不完整，缺少：{', '.join(missing_fields)}。"
            if default_agent_id and not self._agent_exists(
                default_agent_id,
                account_id=record.account_id,
            ):
                return False, f"飞书默认 Agent 不存在：{default_agent_id}。"
            if connection_mode == "webhook":
                return True, "飞书 webhook 配置已通过本地校验，可继续执行事件订阅联调。"
            return True, "飞书长连接配置已通过本地校验，启用后服务会主动建立连接。"

        return True, f"{record.kind} 配置已通过基础校验。"
