"""消息渠道 webhook ingress 网关。"""

from __future__ import annotations

import hashlib
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import httpx
from fastapi import Request
from pydantic import BaseModel, Field

from ..integration_models import (
    ConversationBindingRecord,
    InboundEventRecord,
    IntegrationConfigRecord,
    IntegrationCredentialRecord,
)
from ..integration_store import (
    ConversationBindingStore,
    InboundEventStore,
    IntegrationStore,
)
from ..sqlite_schema import utc_now_iso
from ..upload_models import UploadCreatePayload, UploadRecord, sanitize_upload_filename
from .adapter_base import ChannelAdapterRegistry, decode_json_body
from .feishu_adapter import FeishuAdapter
from .mock_adapter import MockChannelAdapter
from .wechat_adapter import WeChatAdapter
from .models import (
    ChannelContext,
    ChannelRequest,
    MsgAttachment,
    MsgContextEnvelope,
    ParsedInboundEvent,
    QuickAckIntent,
)
from .router import IntegrationRouter

if TYPE_CHECKING:
    from ..session import SessionManager

_CONTROL_COMMAND_PATTERN = re.compile(
    r"^\s*/(?P<command>help|reset|bind|status)(?:\s+(?P<args>.+?))?\s*$",
    re.IGNORECASE,
)


class IntegrationGatewayError(RuntimeError):
    """网关处理异常。"""

    def __init__(self, detail: str, *, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class GatewayCommandResult(BaseModel):
    """控制命令处理结果。"""

    command: str
    handled: bool = True
    response_text: str
    binding_id: str = ""
    session_id: str = ""
    agent_id: str = ""


class GatewayProcessResult(BaseModel):
    """一次 webhook 请求处理结果。"""

    integration_id: str
    channel_kind: str
    quick_ack: QuickAckIntent
    event: InboundEventRecord | None = None
    msg_context: MsgContextEnvelope | None = None
    binding_id: str = ""
    session_id: str = ""
    duplicate_of_event_id: str = ""
    command_result: GatewayCommandResult | None = None
    created_uploads: list[UploadRecord] = Field(default_factory=list)
    should_route: bool = False


def _default_adapter_registry() -> ChannelAdapterRegistry:
    return ChannelAdapterRegistry([MockChannelAdapter(), FeishuAdapter(), WeChatAdapter()])


class IntegrationGateway:
    """统一处理渠道 webhook 入站。"""

    def __init__(
        self,
        session_manager: "SessionManager",
        *,
        adapter_registry: ChannelAdapterRegistry | None = None,
        download_transport: httpx.AsyncBaseTransport | None = None,
        download_timeout_seconds: float = 15.0,
    ):
        self._session_manager = session_manager
        self._adapter_registry = adapter_registry or _default_adapter_registry()
        self._download_transport = download_transport
        self._download_timeout_seconds = download_timeout_seconds
        self._integration_store: IntegrationStore | None = None
        self._inbound_event_store: InboundEventStore | None = None
        self._binding_store: ConversationBindingStore | None = None
        self._router = IntegrationRouter(session_manager)

    async def handle_http_request(
        self,
        integration_id: str,
        request: Request,
        *,
        expected_kind: str | None = None,
    ) -> GatewayProcessResult:
        channel_request = await self.build_channel_request(request)
        await self._session_manager.initialize()
        self._ensure_stores()
        integration = self._require_integration(
            integration_id,
            expected_kind=expected_kind,
        )
        if self._should_ignore_feishu_http_webhook(integration):
            return self._build_feishu_long_connection_http_ack(
                integration_id=integration.id,
                request=channel_request,
            )
        return await self.handle_channel_request(
            integration_id,
            channel_request,
            expected_kind=expected_kind,
        )

    async def build_channel_request(self, request: Request) -> ChannelRequest:
        body = await request.body()
        return ChannelRequest(
            method=request.method,
            path=request.url.path,
            headers={key: value for key, value in request.headers.items()},
            query_params={key: value for key, value in request.query_params.items()},
            body=body,
            received_at=utc_now_iso(),
            remote_addr=request.client.host if request.client else "",
        )

    async def handle_channel_request(
        self,
        integration_id: str,
        request: ChannelRequest,
        *,
        expected_kind: str | None = None,
    ) -> GatewayProcessResult:
        await self._session_manager.initialize()
        self._ensure_stores()

        integration = self._require_integration(
            integration_id,
            expected_kind=expected_kind,
        )
        adapter = self._adapter_registry.get(integration.kind)
        context = ChannelContext(
            integration=integration,
            credentials=self._resolve_credentials(integration_id),
        )

        try:
            verification = adapter.verify_request(context, request)
        except Exception as exc:
            event = self._create_fallback_event(
                integration_id=integration.id,
                channel_kind=integration.kind,
                request=request,
                event_type="verification_failed",
                status="failed",
                error_message=str(exc),
            )
            self._inbound_event_store.create_event(event)
            raise IntegrationGatewayError(
                f"Webhook 验签失败：{exc}",
                status_code=400,
            ) from exc

        if not verification.accepted:
            event = self._create_fallback_event(
                integration_id=integration.id,
                channel_kind=integration.kind,
                request=request,
                event_type="request_rejected",
                status="rejected",
                error_message=verification.reason or "Webhook 请求未通过校验。",
                signature_valid=verification.signature_valid,
                raw_payload=verification.body_json,
                extra_metadata={"verification_metadata": verification.metadata},
            )
            self._inbound_event_store.create_event(event)
            raise IntegrationGatewayError(
                verification.reason or "Webhook 请求未通过校验。",
                status_code=401,
            )

        try:
            parsed_event = adapter.parse_inbound_event(context, request, verification)
        except Exception as exc:
            event = self._create_fallback_event(
                integration_id=integration.id,
                channel_kind=integration.kind,
                request=request,
                event_type="parse_failed",
                status="failed",
                error_message=str(exc),
                signature_valid=verification.signature_valid,
                raw_payload=verification.body_json,
                extra_metadata={"verification_metadata": verification.metadata},
            )
            self._inbound_event_store.create_event(event)
            raise IntegrationGatewayError(
                f"Webhook 解析失败：{exc}",
                status_code=400,
            ) from exc

        quick_ack = adapter.emit_quick_ack(context, parsed_event)
        integration = self._capture_default_chat_target_if_missing(
            integration,
            parsed_event,
        )
        context = ChannelContext(
            integration=integration,
            credentials=context.credentials,
        )
        duplicate = self._find_duplicate(integration.id, parsed_event, request)
        if duplicate is not None:
            duplicate = self._mark_duplicate_hit(duplicate, request.received_at)
            return GatewayProcessResult(
                integration_id=integration.id,
                channel_kind=integration.kind,
                quick_ack=quick_ack,
                event=duplicate,
                duplicate_of_event_id=duplicate.id,
                should_route=False,
            )

        event = self._create_event_record(
            integration_id=integration.id,
            channel_kind=integration.kind,
            request=request,
            parsed_event=parsed_event,
            verification_metadata=verification.metadata,
        )
        event = self._inbound_event_store.create_event(event)
        event = self._update_event_status(
            event,
            next_status="verified",
            metadata_updates={
                "channel_kind": integration.kind,
                "message_type": parsed_event.message_type,
            },
        )

        if self._is_self_message(context, parsed_event):
            event = self._update_event_status(
                event,
                next_status="command_handled",
                metadata_updates={"handled_by": "ignore_self_message"},
            )
            return GatewayProcessResult(
                integration_id=integration.id,
                channel_kind=integration.kind,
                quick_ack=quick_ack,
                event=event,
                should_route=False,
            )

        command_result = await self._handle_control_command(
            context=context,
            parsed_event=parsed_event,
        )
        if command_result is not None:
            event = self._update_event_status(
                event,
                next_status="command_handled",
                metadata_updates={
                    "handled_by": f"command:{command_result.command}",
                    "command_response": command_result.response_text,
                    "binding_id": command_result.binding_id,
                    "session_id": command_result.session_id,
                    "agent_id": command_result.agent_id,
                },
            )
            return GatewayProcessResult(
                integration_id=integration.id,
                channel_kind=integration.kind,
                quick_ack=quick_ack,
                event=event,
                binding_id=command_result.binding_id,
                session_id=command_result.session_id,
                command_result=command_result,
                should_route=False,
            )

        if not self._requires_routing(parsed_event):
            event = self._update_event_status(
                event,
                next_status="command_handled",
                metadata_updates={"handled_by": "quick_ack_only"},
            )
            return GatewayProcessResult(
                integration_id=integration.id,
                channel_kind=integration.kind,
                quick_ack=quick_ack,
                event=event,
                should_route=False,
            )

        msg_context = adapter.build_msg_context(context, parsed_event)
        msg_context = await adapter.prepare_msg_context(context, msg_context)
        route_resolution = await self._router.resolve_route(integration, parsed_event)
        binding = route_resolution.binding
        if binding is None:
            event = self._update_event_status(
                event,
                next_status="failed",
                error_message=route_resolution.message,
                metadata_updates={
                    "route_source": route_resolution.source,
                    "route_message": route_resolution.message,
                    "disabled_binding_id": route_resolution.disabled_binding_id,
                    "disabled_binding_reason": route_resolution.disabled_binding_reason,
                },
            )
            return GatewayProcessResult(
                integration_id=integration.id,
                channel_kind=integration.kind,
                quick_ack=quick_ack,
                event=event,
                msg_context=msg_context,
                should_route=False,
            )

        event = self._update_event_status(
            event,
            next_status="routed",
            metadata_updates={
                "binding_id": binding.id,
                "session_id": binding.session_id,
                "agent_id": binding.agent_id,
                "route_source": route_resolution.source,
                "route_message": route_resolution.message,
                "routing_rule_id": route_resolution.matched_rule.id if route_resolution.matched_rule else "",
                "session_key": route_resolution.session_key,
                "disabled_binding_id": route_resolution.disabled_binding_id,
                "disabled_binding_reason": route_resolution.disabled_binding_reason,
            },
        )

        created_uploads: list[UploadRecord] = []
        binding = self._touch_binding(binding, parsed_event.received_at)
        if msg_context.attachments:
            try:
                created_uploads = await self._bridge_attachments(
                    context=context,
                    adapter=adapter,
                    msg_context=msg_context,
                    binding=binding,
                )
            except Exception as exc:
                event = self._update_event_status(
                    event,
                    next_status="failed",
                    error_message=f"附件桥接失败：{exc}",
                )
                raise IntegrationGatewayError(
                    f"附件桥接失败：{exc}",
                    status_code=502,
                ) from exc

        if created_uploads:
            metadata = dict(msg_context.metadata)
            metadata["binding_id"] = binding.id
            metadata["session_id"] = binding.session_id
            metadata["attachment_upload_ids"] = [upload.id for upload in created_uploads]
            msg_context = msg_context.model_copy(update={"metadata": metadata})
            event = self._update_event_status(
                event,
                next_status="bridged",
                metadata_updates={
                    "attachment_upload_ids": [upload.id for upload in created_uploads],
                    "attachment_session_id": binding.session_id,
                },
            )
        return GatewayProcessResult(
            integration_id=integration.id,
            channel_kind=integration.kind,
            quick_ack=quick_ack,
            event=event,
            msg_context=msg_context,
            binding_id=binding.id if binding else "",
            session_id=binding.session_id if binding else "",
            created_uploads=created_uploads,
            should_route=True,
        )

    def _ensure_stores(self) -> None:
        if (
            self._integration_store is not None
            and self._inbound_event_store is not None
            and self._binding_store is not None
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
        self._inbound_event_store = InboundEventStore(resolved_db_path)
        self._binding_store = ConversationBindingStore(resolved_db_path)

    def _require_integration(
        self,
        integration_id: str,
        *,
        expected_kind: str | None,
    ):
        integration = self._integration_store.get_integration(integration_id)
        if integration is None:
            raise IntegrationGatewayError("未找到对应的集成配置。", status_code=404)
        if expected_kind and integration.kind != expected_kind:
            raise IntegrationGatewayError("集成类型与 webhook 路径不匹配。", status_code=404)
        if integration.status != "active":
            raise IntegrationGatewayError("当前集成未启用。", status_code=409)
        if not self._adapter_registry.has(integration.kind):
            raise IntegrationGatewayError(
                f"暂不支持的集成类型：{integration.kind}",
                status_code=400,
            )
        return integration

    @staticmethod
    def _should_ignore_feishu_http_webhook(integration) -> bool:
        return integration.kind == "feishu" and FeishuAdapter.uses_long_connection(integration.config)

    def _build_feishu_long_connection_http_ack(
        self,
        *,
        integration_id: str,
        request: ChannelRequest,
    ) -> GatewayProcessResult:
        try:
            payload = decode_json_body(request.body)
        except Exception:
            payload = {}

        quick_ack = QuickAckIntent(
            body_type="json",
            body_json={
                "ok": True,
                "ignored": True,
                "connection_mode": "long_connection",
            },
            headers={"X-Clavi-Agent-Integration-Mode": "long_connection"},
        )
        if str(payload.get("type") or "").strip() == "url_verification":
            quick_ack = QuickAckIntent(
                body_type="json",
                body_json={"challenge": str(payload.get("challenge") or "")},
                headers={"X-Clavi-Agent-Integration-Mode": "long_connection"},
            )

        return GatewayProcessResult(
            integration_id=integration_id,
            channel_kind="feishu",
            quick_ack=quick_ack,
            should_route=False,
        )

    def _resolve_credentials(self, integration_id: str) -> dict[str, str]:
        credentials: dict[str, str] = {}
        for credential in self._integration_store.list_credentials(integration_id):
            value = self._resolve_credential_value(credential)
            if value:
                credentials[credential.credential_key] = value
        return credentials

    @staticmethod
    def _resolve_credential_value(record: IntegrationCredentialRecord) -> str:
        if record.storage_kind == "local_encrypted":
            return record.secret_ciphertext.strip()

        reference = record.secret_ref.strip()
        if not reference:
            return ""
        if reference.startswith("env:"):
            return os.environ.get(reference[4:], "").strip()
        return os.environ.get(reference, "").strip()

    def _create_event_record(
        self,
        *,
        integration_id: str,
        channel_kind: str,
        request: ChannelRequest,
        parsed_event: ParsedInboundEvent,
        verification_metadata: dict[str, Any],
    ) -> InboundEventRecord:
        payload_hash = hashlib.sha256(request.body).hexdigest()
        effective_dedup_key = self._effective_dedup_key(parsed_event, payload_hash)
        return InboundEventRecord(
            id=str(uuid.uuid4()),
            integration_id=integration_id,
            provider_event_id=parsed_event.provider_event_id,
            provider_message_id=parsed_event.provider_message_id,
            provider_chat_id=parsed_event.provider_chat_id,
            provider_thread_id=parsed_event.provider_thread_id,
            provider_user_id=parsed_event.provider_user_id,
            event_type=parsed_event.event_type,
            received_at=parsed_event.received_at,
            signature_valid=parsed_event.signature_valid,
            dedup_key=effective_dedup_key,
            raw_headers=parsed_event.raw_headers or request.headers,
            raw_payload=parsed_event.raw_payload,
            normalized_status="received",
            metadata={
                "channel_kind": channel_kind,
                "message_type": parsed_event.message_type,
                "payload_hash": payload_hash,
                "verification_metadata": verification_metadata,
                "request_path": request.path,
                "query_params": request.query_params,
                "remote_addr": request.remote_addr,
            },
        )

    def _create_fallback_event(
        self,
        *,
        integration_id: str,
        channel_kind: str,
        request: ChannelRequest,
        event_type: str,
        status: str,
        error_message: str,
        signature_valid: bool = False,
        raw_payload: object | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> InboundEventRecord:
        payload_hash = hashlib.sha256(request.body).hexdigest()
        metadata: dict[str, Any] = {
            "channel_kind": channel_kind,
            "payload_hash": payload_hash,
            "request_path": request.path,
            "query_params": request.query_params,
            "remote_addr": request.remote_addr,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        return InboundEventRecord(
            id=str(uuid.uuid4()),
            integration_id=integration_id,
            event_type=event_type,
            received_at=request.received_at,
            signature_valid=signature_valid,
            dedup_key=f"payload:{payload_hash}",
            raw_headers=request.headers,
            raw_payload=raw_payload if raw_payload is not None else {},
            normalized_status=status,
            normalized_error=error_message,
            metadata=metadata,
        )

    @staticmethod
    def _effective_dedup_key(parsed_event: ParsedInboundEvent, payload_hash: str) -> str:
        explicit = str(parsed_event.dedup_key or "").strip()
        if explicit:
            return explicit
        provider_message_id = str(parsed_event.provider_message_id or "").strip()
        if provider_message_id:
            return provider_message_id
        provider_event_id = str(parsed_event.provider_event_id or "").strip()
        if provider_event_id:
            return provider_event_id
        return f"payload:{payload_hash}"

    def _find_duplicate(
        self,
        integration_id: str,
        parsed_event: ParsedInboundEvent,
        request: ChannelRequest,
    ) -> InboundEventRecord | None:
        provider_event_id = str(parsed_event.provider_event_id or "").strip()
        if provider_event_id:
            duplicate = self._inbound_event_store.get_event_by_provider_event_id(
                integration_id,
                provider_event_id,
            )
            if duplicate is not None:
                return duplicate

        provider_message_id = str(parsed_event.provider_message_id or "").strip()
        if provider_message_id:
            duplicate = self._inbound_event_store.get_event_by_provider_message_id(
                integration_id,
                provider_message_id,
            )
            if duplicate is not None:
                return duplicate

        dedup_key = self._effective_dedup_key(
            parsed_event,
            hashlib.sha256(request.body).hexdigest(),
        )
        if dedup_key:
            return self._inbound_event_store.get_event_by_dedup_key(
                integration_id,
                dedup_key,
            )
        return None

    def _mark_duplicate_hit(
        self,
        event: InboundEventRecord,
        received_at: str,
    ) -> InboundEventRecord:
        metadata = dict(event.metadata)
        metadata["duplicate_count"] = int(metadata.get("duplicate_count") or 0) + 1
        metadata["last_duplicate_at"] = received_at
        updated = event.model_copy(update={"metadata": metadata})
        return self._inbound_event_store.update_event(updated)

    def _capture_default_chat_target_if_missing(
        self,
        integration: IntegrationConfigRecord,
        parsed_event: ParsedInboundEvent,
    ) -> IntegrationConfigRecord:
        provider_chat_id = str(parsed_event.provider_chat_id or "").strip()
        if not provider_chat_id:
            return integration

        config = dict(integration.config or {})
        for key in ("default_chat_id", "default_target_id", "target_id", "receive_id"):
            if str(config.get(key) or "").strip():
                return integration

        metadata = dict(integration.metadata or {})
        metadata.update(
            {
                "default_chat_id_source": "auto_detected_from_inbound",
                "default_chat_id_detected_at": utc_now_iso(),
                "default_chat_id_detected_message_id": str(
                    parsed_event.provider_message_id or ""
                ).strip(),
            }
        )
        updated = integration.model_copy(
            update={
                "config": {
                    **config,
                    "default_chat_id": provider_chat_id,
                },
                "metadata": metadata,
                "updated_at": utc_now_iso(),
            }
        )
        self._integration_store.update_integration(updated)
        return updated

    def _update_event_status(
        self,
        event: InboundEventRecord,
        *,
        next_status: str | None = None,
        error_message: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InboundEventRecord:
        updated = event
        if next_status is not None and next_status != event.normalized_status:
            if event.can_transition_to(next_status):
                updated = event.transition_to(next_status, error_message=error_message)
            else:
                updates: dict[str, Any] = {"normalized_status": next_status}
                if error_message is not None:
                    updates["normalized_error"] = error_message
                updated = event.model_copy(update=updates)
        elif error_message is not None:
            updated = updated.model_copy(update={"normalized_error": error_message})

        if metadata_updates:
            merged_metadata = dict(updated.metadata)
            for key, value in metadata_updates.items():
                if value is None:
                    continue
                if isinstance(value, str) and not value:
                    continue
                merged_metadata[key] = value
            updated = updated.model_copy(update={"metadata": merged_metadata})

        return self._inbound_event_store.update_event(updated)

    def _is_self_message(
        self,
        context: ChannelContext,
        parsed_event: ParsedInboundEvent,
    ) -> bool:
        metadata = dict(parsed_event.metadata or {})
        if bool(metadata.get("is_from_self")):
            return True

        sender_type = str(metadata.get("sender_type") or "").strip().lower()
        if sender_type in {"app", "bot"}:
            return True

        sender_id = str(parsed_event.provider_user_id or "").strip()
        if not sender_id:
            return False

        configured_sender_ids: list[str] = []
        for key in ("bot_user_id", "bot_open_id", "bot_union_id"):
            value = context.get_secret(key)
            if value:
                configured_sender_ids.append(value)

        extra_sender_ids = context.integration.config.get("ignore_sender_ids") or []
        if isinstance(extra_sender_ids, list):
            configured_sender_ids.extend(
                str(item).strip()
                for item in extra_sender_ids
                if str(item).strip()
            )

        return sender_id in set(configured_sender_ids)

    async def _handle_control_command(
        self,
        *,
        context: ChannelContext,
        parsed_event: ParsedInboundEvent,
    ) -> GatewayCommandResult | None:
        match = _CONTROL_COMMAND_PATTERN.match(str(parsed_event.text or ""))
        if match is None:
            return None

        command = match.group("command").lower()
        args = (match.group("args") or "").strip()

        if command == "help":
            return GatewayCommandResult(
                command="help",
                response_text=(
                    "可用命令：\n"
                    "/help 查看帮助\n"
                    "/status 查看当前绑定状态\n"
                    "/bind <agent_id> [chat|thread] 绑定当前渠道会话\n"
                    "/reset 重新创建当前绑定会话"
                ),
            )

        if command == "status":
            binding = self._find_binding_for_event(parsed_event)
            if binding is None:
                return GatewayCommandResult(
                    command="status",
                    response_text=(
                        f"当前集成：{context.integration.display_name or context.integration.name}\n"
                        "当前渠道会话尚未绑定 Agent。"
                    ),
                )
            return GatewayCommandResult(
                command="status",
                response_text=(
                    f"当前集成：{context.integration.display_name or context.integration.name}\n"
                    f"绑定 Agent：{binding.agent_id}\n"
                    f"会话 ID：{binding.session_id}\n"
                    f"绑定范围：{binding.binding_scope}"
                ),
                binding_id=binding.id,
                session_id=binding.session_id,
                agent_id=binding.agent_id,
            )

        if command == "bind":
            return await self._handle_bind_command(
                context=context,
                parsed_event=parsed_event,
                args=args,
            )

        if command == "reset":
            return await self._handle_reset_command(parsed_event=parsed_event)

        return None

    async def _handle_bind_command(
        self,
        *,
        context: ChannelContext,
        parsed_event: ParsedInboundEvent,
        args: str,
    ) -> GatewayCommandResult:
        if self._session_manager._agent_store is None:
            raise IntegrationGatewayError("AgentStore 未初始化。", status_code=500)

        parts = [part for part in args.split() if part]
        if not parts:
            return GatewayCommandResult(
                command="bind",
                response_text="用法：/bind <agent_id> [chat|thread]",
            )

        agent_id = parts[0]
        requested_scope = parts[1].lower() if len(parts) > 1 else ""
        if requested_scope not in {"", "chat", "thread"}:
            return GatewayCommandResult(
                command="bind",
                response_text="绑定范围仅支持 chat 或 thread。",
                agent_id=agent_id,
            )

        if (
            self._session_manager._agent_store.get_agent_template(
                agent_id,
                account_id=context.integration.account_id,
            )
            is None
        ):
            return GatewayCommandResult(
                command="bind",
                response_text=f"未找到 Agent：{agent_id}",
                agent_id=agent_id,
            )

        scope = requested_scope or self._router.preferred_binding_scope(parsed_event)
        if scope == "thread" and parsed_event.is_group and not parsed_event.provider_thread_id:
            return GatewayCommandResult(
                command="bind",
                response_text="当前消息没有 thread_id，无法使用 thread 级绑定。",
                agent_id=agent_id,
            )

        now = parsed_event.received_at
        binding = await self._router.upsert_binding(
            account_id=context.integration.account_id,
            integration_id=context.integration_id,
            tenant_id=parsed_event.tenant_id,
            chat_id=parsed_event.provider_chat_id,
            thread_id=parsed_event.provider_thread_id,
            binding_scope=scope,
            agent_id=agent_id,
            metadata={"binding_source": "command_bind"},
            updated_at=now,
            force_new_session=True,
            touch_last_message=True,
        )

        return GatewayCommandResult(
            command="bind",
            response_text=(
                f"已绑定到 Agent：{agent_id}\n"
                f"绑定范围：{binding.binding_scope}\n"
                f"会话 ID：{binding.session_id}"
            ),
            binding_id=binding.id,
            session_id=binding.session_id,
            agent_id=agent_id,
        )

    async def _handle_reset_command(
        self,
        *,
        parsed_event: ParsedInboundEvent,
    ) -> GatewayCommandResult:
        binding = self._find_binding_for_event(parsed_event)
        if binding is None:
            return GatewayCommandResult(
                command="reset",
                response_text="当前渠道会话尚未绑定 Agent，无法重置。",
            )

        binding = await self._router.reset_binding_session(
            binding.id,
            updated_at=parsed_event.received_at,
        )
        return GatewayCommandResult(
            command="reset",
            response_text=(
                f"已重置会话上下文。\n"
                f"绑定 Agent：{binding.agent_id}\n"
                f"新会话 ID：{binding.session_id}"
            ),
            binding_id=binding.id,
            session_id=binding.session_id,
            agent_id=binding.agent_id,
        )

    def _find_binding_for_event(
        self,
        parsed_event: ParsedInboundEvent,
    ) -> ConversationBindingRecord | None:
        return self._router.find_explicit_binding(parsed_event)

    @staticmethod
    def _requires_routing(parsed_event: ParsedInboundEvent) -> bool:
        return parsed_event.event_type == "message"

    def _touch_binding(
        self,
        binding: ConversationBindingRecord,
        timestamp: str,
    ) -> ConversationBindingRecord:
        updated = binding.model_copy(
            update={
                "updated_at": timestamp,
                "last_message_at": timestamp,
            }
        )
        return self._binding_store.update_binding(updated)

    async def _bridge_attachments(
        self,
        *,
        context: ChannelContext,
        adapter,
        msg_context: MsgContextEnvelope,
        binding: ConversationBindingRecord,
    ) -> list[UploadRecord]:
        uploads: list[UploadCreatePayload] = []
        for index, attachment in enumerate(msg_context.attachments, start=1):
            payload = await self._download_attachment(
                context=context,
                adapter=adapter,
                attachment=attachment,
                fallback_name=f"attachment-{index}",
            )
            if payload is not None:
                uploads.append(payload)

        if not uploads:
            return []

        return self._session_manager.create_session_uploads(
            binding.session_id,
            uploads,
            account_id=binding.account_id,
            created_by=f"integration:{context.integration_id}",
        )

    async def _download_attachment(
        self,
        *,
        context: ChannelContext,
        adapter,
        attachment: MsgAttachment,
        fallback_name: str,
    ) -> UploadCreatePayload | None:
        hint = await adapter.prepare_upload_download(context, attachment)
        if hint is None or not hint.download_url:
            return None

        async with httpx.AsyncClient(
            timeout=self._download_timeout_seconds,
            transport=self._download_transport,
            follow_redirects=True,
        ) as client:
            response = await client.get(
                hint.download_url,
                headers=dict(hint.headers),
            )
        response.raise_for_status()

        content_type = str(response.headers.get("content-type") or "").split(";")[0].strip()
        filename = self._resolve_attachment_filename(
            attachment=attachment,
            download_url=hint.download_url,
            content_disposition=str(response.headers.get("content-disposition") or ""),
            suggested_filename=hint.suggested_filename or attachment.name,
            mime_type=attachment.mime_type or content_type,
            fallback_name=fallback_name,
        )
        return UploadCreatePayload(
            original_name=filename,
            content_bytes=response.content,
            mime_type=attachment.mime_type or content_type,
        )

    def _resolve_attachment_filename(
        self,
        *,
        attachment: MsgAttachment,
        download_url: str,
        content_disposition: str,
        suggested_filename: str,
        mime_type: str,
        fallback_name: str,
    ) -> str:
        filename = suggested_filename.strip()
        if not filename:
            filename = self._filename_from_content_disposition(content_disposition)
        if not filename:
            parsed = urlparse(download_url)
            filename = unquote(Path(parsed.path).name)
        if not filename:
            filename = attachment.provider_file_id.strip() or fallback_name
        if "." not in filename:
            guessed_extension = mimetypes.guess_extension(mime_type or "")
            if guessed_extension:
                filename = f"{filename}{guessed_extension}"
        return sanitize_upload_filename(filename)

    @staticmethod
    def _filename_from_content_disposition(content_disposition: str) -> str:
        if not content_disposition:
            return ""
        match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if match:
            return unquote(match.group(1).strip().strip('"'))
        match = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

