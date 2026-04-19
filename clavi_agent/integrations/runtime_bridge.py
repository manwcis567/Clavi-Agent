"""渠道消息到 Session / Run 的执行桥接。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from ..integration_models import InboundEventRecord
from ..integration_store import InboundEventStore
from ..schema import message_content_summary
from ..sqlite_schema import utc_now_iso
from .gateway import GatewayProcessResult

if TYPE_CHECKING:
    from ..run_models import RunRecord
    from ..session import SessionManager


class IntegrationRunBridgeError(RuntimeError):
    """渠道执行桥接异常。"""

    def __init__(
        self,
        detail: str,
        *,
        event: InboundEventRecord | None = None,
    ):
        super().__init__(detail)
        self.detail = detail
        self.event = event


class IntegrationRunBridgeResult(BaseModel):
    """一次渠道消息桥接到 durable run 的结果。"""

    integration_id: str
    inbound_event_id: str
    session_id: str
    run_id: str
    goal: str
    event: InboundEventRecord
    user_message_content: str | list[dict[str, Any]]
    run_metadata: dict[str, Any] = Field(default_factory=dict)


class IntegrationRunBridge:
    """把已经完成路由的渠道消息桥接进现有 Session / Run 链路。"""

    def __init__(self, session_manager: "SessionManager"):
        self._session_manager = session_manager
        self._inbound_event_store: InboundEventStore | None = None

    async def bridge_gateway_result(
        self,
        result: GatewayProcessResult,
    ) -> IntegrationRunBridgeResult | None:
        """把网关处理结果转换成 durable run。"""
        if not result.should_route:
            return None
        if result.event is None:
            raise IntegrationRunBridgeError("缺少入站事件记录，无法创建 run。")
        if result.msg_context is None:
            raise IntegrationRunBridgeError("缺少消息上下文，无法创建 run。", event=result.event)
        if not result.session_id.strip():
            raise IntegrationRunBridgeError("缺少绑定会话，无法创建 run。", event=result.event)

        await self._session_manager.initialize()
        self._ensure_store()
        event = self._refresh_event(result.event)

        try:
            attachment_ids = self._extract_attachment_upload_ids(result.msg_context.metadata)
            user_message_content = self._session_manager.build_chat_message_content(
                result.session_id,
                result.msg_context.text,
                account_id=event.account_id,
                attachment_ids=attachment_ids,
            )
            goal = (
                message_content_summary(user_message_content)
                or result.msg_context.text.strip()
                or "处理渠道消息"
            )
            run_metadata = self._build_run_metadata(
                result=result,
                event=event,
                user_message_content=user_message_content,
                attachment_ids=attachment_ids,
            )
            run = self._session_manager.start_run(
                result.session_id,
                goal,
                account_id=event.account_id,
                run_metadata=run_metadata,
            )
        except Exception as exc:
            failed_event = self._update_event(
                event,
                next_status="failed",
                error_message=f"渠道执行桥接失败：{exc}",
                metadata_updates={
                    "bridge_status": "failed",
                    "bridge_error_type": exc.__class__.__name__,
                },
            )
            raise IntegrationRunBridgeError(
                f"渠道执行桥接失败：{exc}",
                event=failed_event,
            ) from exc

        quick_response_metadata: dict[str, Any] = {}
        try:
            scheduled_task = self._session_manager.schedule_integration_quick_response(
                run,
                msg_context=result.msg_context,
            )
        except Exception as exc:
            quick_response_metadata = {
                "quick_response_status": "schedule_failed",
                "quick_response_schedule_error": str(exc),
            }
        else:
            if scheduled_task is not None:
                quick_response_metadata["quick_response_status"] = "scheduled"

        bridged_event = self._update_event(
            event,
            next_status="bridged",
            metadata_updates={
                "bridge_status": "queued",
                "bridge_enqueued_at": utc_now_iso(),
                "run_id": run.id,
                "root_run_id": str(run.run_metadata.get("root_run_id") or run.id),
                "run_goal": goal,
                **quick_response_metadata,
            },
        )
        return IntegrationRunBridgeResult(
            integration_id=result.integration_id,
            inbound_event_id=bridged_event.id,
            session_id=result.session_id,
            run_id=run.id,
            goal=goal,
            event=bridged_event,
            user_message_content=user_message_content,
            run_metadata=run.run_metadata,
        )

    def _ensure_store(self) -> None:
        if self._inbound_event_store is not None:
            return

        config = self._session_manager._config
        if config is None:
            raise RuntimeError("SessionManager 尚未加载配置。")

        db_path = Path(config.agent.session_store_path)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        self._inbound_event_store = InboundEventStore(db_path.resolve())

    def _refresh_event(self, event: InboundEventRecord) -> InboundEventRecord:
        if self._inbound_event_store is None:
            return event
        stored = self._inbound_event_store.get_event(
            event.id,
            account_id=event.account_id,
        )
        return stored or event

    @staticmethod
    def _extract_attachment_upload_ids(metadata: dict[str, Any]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        raw_ids = metadata.get("attachment_upload_ids")
        if not isinstance(raw_ids, list):
            return []

        for item in raw_ids:
            upload_id = str(item).strip()
            if not upload_id or upload_id in seen:
                continue
            normalized.append(upload_id)
            seen.add(upload_id)
        return normalized

    def _build_run_metadata(
        self,
        *,
        result: GatewayProcessResult,
        event: InboundEventRecord,
        user_message_content: str | list[dict[str, Any]],
        attachment_ids: list[str],
    ) -> dict[str, Any]:
        msg_context = result.msg_context
        if msg_context is None:
            return {}

        provider_metadata = dict(msg_context.metadata)
        channel_context = {
            "tenant_id": msg_context.tenant_id,
            "chat_id": msg_context.chat_id,
            "thread_id": msg_context.thread_id,
            "message_id": msg_context.message_id,
            "sender_id": msg_context.sender_id,
            "sender_name": msg_context.sender_name,
            "is_group": msg_context.is_group,
            "locale": msg_context.locale,
            "timezone": msg_context.timezone,
            "received_at": msg_context.received_at,
            "event_type": str(provider_metadata.get("event_type") or ""),
            "provider_event_type": str(provider_metadata.get("provider_event_type") or ""),
            "message_type": str(provider_metadata.get("message_type") or ""),
            "root_id": str(provider_metadata.get("root_id") or ""),
            "parent_id": str(provider_metadata.get("parent_id") or ""),
            "sender_id_type": str(provider_metadata.get("sender_id_type") or ""),
            "reply_context": str(provider_metadata.get("reply_context") or ""),
            "mentions": [
                mention.model_dump(mode="python")
                for mention in msg_context.mentions
            ],
            "attachments": [
                {
                    "kind": attachment.kind,
                    "provider_file_id": attachment.provider_file_id,
                    "provider_message_id": attachment.provider_message_id,
                    "name": attachment.name,
                    "mime_type": attachment.mime_type,
                    "size_bytes": attachment.size_bytes,
                }
                for attachment in msg_context.attachments
            ],
            "attachment_upload_ids": attachment_ids,
            "provider_metadata": provider_metadata,
        }

        return {
            "source_kind": "integration",
            "source_label": f"integration:{result.channel_kind}",
            "integration_id": result.integration_id,
            "channel_kind": result.channel_kind,
            "binding_id": result.binding_id or str(event.metadata.get("binding_id") or ""),
            "session_key": str(event.metadata.get("session_key") or ""),
            "route_source": str(event.metadata.get("route_source") or ""),
            "route_message": str(event.metadata.get("route_message") or ""),
            "inbound_event_id": event.id,
            "provider_event_id": event.provider_event_id,
            "provider_message_id": event.provider_message_id,
            "provider_chat_id": event.provider_chat_id,
            "provider_thread_id": event.provider_thread_id,
            "provider_root_message_id": str(provider_metadata.get("root_id") or ""),
            "provider_parent_message_id": str(provider_metadata.get("parent_id") or ""),
            "provider_user_id": event.provider_user_id,
            "user_message_content": user_message_content,
            "channel_context": channel_context,
        }

    def _update_event(
        self,
        event: InboundEventRecord,
        *,
        next_status: str,
        error_message: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InboundEventRecord:
        updated = event
        if next_status and next_status != event.normalized_status:
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

        if self._inbound_event_store is None:
            return updated
        return self._inbound_event_store.update_event(updated)
