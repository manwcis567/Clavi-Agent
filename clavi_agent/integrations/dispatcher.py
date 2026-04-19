"""渠道回复分发器。"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..integration_models import (
    DeliveryAttemptRecord,
    InboundEventRecord,
    IntegrationConfigRecord,
    IntegrationCredentialRecord,
    OutboundDeliveryRecord,
)
from ..integration_store import DeliveryStore, InboundEventStore, IntegrationStore
from ..run_models import ArtifactRecord, RunRecord
from ..schema import Message, message_content_summary
from ..sqlite_schema import utc_now_iso
from .adapter_base import ChannelAdapterRegistry
from .feishu_adapter import FeishuAdapter
from .mock_adapter import MockChannelAdapter
from .wechat_adapter import WeChatAdapter
from .models import (
    ChannelContext,
    MsgContextEnvelope,
    OutboundFile,
    OutboundMessage,
    OutboundReaction,
    OutboundSendResult,
    ReplyAttachmentRef,
    ReplyEnvelope,
)

if TYPE_CHECKING:
    from ..agent_runtime import AgentRuntimeContext
    from ..session import SessionManager


class IntegrationReplyDispatcher:
    """把 durable run 的最终结果回写到外部渠道。"""

    def __init__(
        self,
        session_manager: "SessionManager",
        *,
        adapter_registry: ChannelAdapterRegistry | None = None,
    ):
        self._session_manager = session_manager
        self._adapter_registry = adapter_registry or ChannelAdapterRegistry(
            [MockChannelAdapter(), FeishuAdapter(), WeChatAdapter()]
        )
        self._integration_store: IntegrationStore | None = None
        self._delivery_store: DeliveryStore | None = None
        self._inbound_event_store: InboundEventStore | None = None
        self._scheduled_tasks: dict[str, asyncio.Task[None]] = {}

    async def handle_terminal_run(self, run: RunRecord) -> None:
        """在 run 进入终态后调度一次渠道回写。"""
        if not self._is_integration_root_run(run):
            return

        envelope = self.build_reply_envelope(run)
        if envelope is None:
            self._finalize_inbound_event(
                inbound_event_id=str(run.run_metadata.get("inbound_event_id") or ""),
                next_status="failed",
                error_message="缺少可回写的渠道目标，无法完成回复分发。",
                metadata_updates={
                    "reply_dispatch_status": "failed",
                    "reply_dispatch_error": "missing_delivery_target",
                    "run_status": run.status,
                },
            )
            return

        self.schedule_envelope(envelope)

    def schedule_quick_response(
        self,
        run: RunRecord,
        *,
        msg_context: MsgContextEnvelope,
    ) -> asyncio.Task[None] | None:
        if not self._is_integration_root_run(run):
            return None

        self._ensure_stores()
        integration_id = str(run.run_metadata.get("integration_id") or "").strip()
        if not integration_id:
            return None

        integration = self._integration_store.get_integration(integration_id)
        if integration is None:
            return None

        adapter = self._adapter_registry.get(integration.kind)
        context = ChannelContext(
            integration=integration,
            credentials=self._resolve_credentials(integration_id),
        )
        reaction = adapter.build_quick_reaction(context, msg_context)
        if reaction is None:
            return None

        task_key = self._quick_response_task_key(run.id)
        existing_task = self._scheduled_tasks.get(task_key)
        if existing_task is not None and not existing_task.done():
            return existing_task

        delivery_id = self._quick_response_delivery_id(run.id)
        inbound_event_id = str(run.run_metadata.get("inbound_event_id") or "").strip()
        self._finalize_inbound_event(
            inbound_event_id=inbound_event_id,
            next_status="",
            metadata_updates={
                "quick_response_status": "scheduled",
                "quick_response_delivery_id": delivery_id,
                "quick_response_reaction_type": reaction.reaction_type,
                "quick_response_target_message_id": reaction.message_id,
            },
        )

        task = asyncio.create_task(
            self.dispatch_quick_response(
                run=run,
                context=context,
                adapter=adapter,
                msg_context=msg_context,
                reaction=reaction,
            ),
            name=f"integration-quick-response:{run.id}",
        )
        self._scheduled_tasks[task_key] = task

        def _cleanup(finished_task: asyncio.Task[None]) -> None:
            self._scheduled_tasks.pop(task_key, None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                finished_task.exception()

        task.add_done_callback(_cleanup)
        return task

    def schedule_envelope(self, envelope: ReplyEnvelope) -> asyncio.Task[None]:
        """异步调度一条渠道回复，不阻塞 run 终态收尾。"""
        existing_task = self._scheduled_tasks.get(envelope.run_id)
        if existing_task is not None and not existing_task.done():
            return existing_task

        task = asyncio.create_task(
            self.dispatch_envelope(envelope),
            name=f"integration-reply:{envelope.run_id}",
        )
        self._scheduled_tasks[envelope.run_id] = task

        def _cleanup(finished_task: asyncio.Task[None]) -> None:
            self._scheduled_tasks.pop(envelope.run_id, None)
            with contextlib.suppress(asyncio.CancelledError, Exception):
                finished_task.exception()

        task.add_done_callback(_cleanup)
        return task

    async def dispatch_quick_response(
        self,
        *,
        run: RunRecord,
        context: ChannelContext,
        adapter,
        msg_context: MsgContextEnvelope,
        reaction: OutboundReaction,
        force: bool = False,
    ) -> OutboundDeliveryRecord:
        delivery = await self._deliver_message(
            context=context,
            adapter=adapter,
            envelope=ReplyEnvelope(
                integration_id=context.integration_id,
                channel_kind=context.channel_kind,
                run_id=run.id,
                session_id=run.session_id,
                inbound_event_id="",
                binding_id=str(run.run_metadata.get("binding_id") or ""),
                provider_chat_id=msg_context.chat_id,
                provider_thread_id=msg_context.thread_id,
                reply_to_message_id=msg_context.message_id,
                run_status=run.status,
                metadata={
                    "quick_response": True,
                    "quick_response_event_id": str(run.run_metadata.get("inbound_event_id") or ""),
                    "quick_response_reaction_type": reaction.reaction_type,
                    "quick_response_target_message_id": reaction.message_id,
                },
            ),
            delivery_id=self._quick_response_delivery_id(run.id),
            delivery_type="reaction",
            payload=reaction.model_dump(mode="python"),
            send_callable=adapter.send_outbound_reaction,
            request_payload=reaction,
            force=force,
        )
        self._sync_quick_response_event_metadata(delivery)
        return delivery

    async def dispatch_run(self, run_id: str, *, force: bool = False) -> ReplyEnvelope | None:
        """按 run_id 读取运行结果并执行一次回写。"""
        await self._session_manager.initialize()
        run_store = self._session_manager._run_store
        if run_store is None:
            raise RuntimeError("RunStore 尚未初始化。")

        run = run_store.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")

        envelope = self.build_reply_envelope(run)
        if envelope is None:
            return None
        await self.dispatch_envelope(envelope, force=force)
        return envelope

    async def dispatch_tool_file(
        self,
        *,
        runtime_context: "AgentRuntimeContext",
        local_path: Path,
        file_name: str,
        text_fallback: str = "",
    ) -> OutboundDeliveryRecord:
        """Send one runtime-selected file through the bound channel immediately."""
        self._ensure_stores()
        integration_id = str(runtime_context.integration_id or "").strip()
        provider_chat_id = str(runtime_context.provider_chat_id or "").strip()
        if not integration_id or not provider_chat_id:
            raise ValueError("Runtime context is missing integration delivery target.")

        integration = self._integration_store.get_integration(integration_id)
        if integration is None:
            raise KeyError(f"Integration not found: {integration_id}")

        adapter = self._adapter_registry.get(integration.kind)
        context = ChannelContext(
            integration=integration,
            credentials=self._resolve_credentials(integration_id),
        )
        resolved_path = local_path.resolve()
        normalized_file_name = str(file_name or resolved_path.name).strip() or resolved_path.name
        payload = OutboundFile(
            target_id=provider_chat_id,
            reply_to_message_id=str(runtime_context.provider_message_id or "").strip(),
            thread_id=str(runtime_context.provider_thread_id or "").strip(),
            file_kind="file",
            url=resolved_path.as_uri(),
            file_name=normalized_file_name,
            text_fallback=(
                str(text_fallback or "").strip()
                or f"已发送文件：{normalized_file_name}"
            ),
            dedup_key=f"{runtime_context.run_id or runtime_context.session_id}:tool-file:{uuid.uuid4().hex}",
            metadata={
                "local_path": str(resolved_path),
                "sent_via_runtime_tool": True,
                "runtime_tool_name": "send_channel_file",
            },
        )
        envelope = ReplyEnvelope(
            integration_id=integration_id,
            channel_kind=str(runtime_context.channel_kind or integration.kind),
            run_id=str(runtime_context.run_id or ""),
            session_id=runtime_context.session_id,
            inbound_event_id=str(runtime_context.inbound_event_id or ""),
            binding_id=str(runtime_context.binding_id or ""),
            provider_chat_id=provider_chat_id,
            provider_thread_id=str(runtime_context.provider_thread_id or "").strip(),
            reply_to_message_id=str(runtime_context.provider_message_id or "").strip(),
            run_status="running",
            text="",
            attachments=[],
            metadata={
                "binding_id": str(runtime_context.binding_id or ""),
                "reply_to_message_id": str(runtime_context.provider_message_id or ""),
                "run_status": "running",
                "runtime_tool_name": "send_channel_file",
            },
        )
        return await self._deliver_message(
            context=context,
            adapter=adapter,
            envelope=envelope,
            delivery_id=f"{runtime_context.run_id or runtime_context.session_id}:tool_file:{uuid.uuid4().hex}",
            delivery_type="tool_file",
            payload=payload.model_dump(mode="python"),
            send_callable=adapter.send_outbound_file,
            request_payload=payload,
            force=False,
        )

    async def retry_delivery(self, delivery_id: str) -> OutboundDeliveryRecord:
        """对单条失败或已完成的出站投递执行手动重试。"""
        await self._session_manager.initialize()
        self._ensure_stores()

        delivery = self._delivery_store.get_delivery(delivery_id)
        if delivery is None:
            raise KeyError(f"Outbound delivery not found: {delivery_id}")

        integration = self._integration_store.get_integration(delivery.integration_id)
        if integration is None:
            raise KeyError(f"Integration not found: {delivery.integration_id}")

        adapter = self._adapter_registry.get(integration.kind)
        context = ChannelContext(
            integration=integration,
            credentials=self._resolve_credentials(delivery.integration_id),
        )
        metadata = dict(delivery.metadata)
        envelope = ReplyEnvelope(
            integration_id=delivery.integration_id,
            channel_kind=integration.kind,
            run_id=delivery.run_id,
            session_id=delivery.session_id,
            inbound_event_id=str(delivery.inbound_event_id or ""),
            binding_id=str(metadata.get("binding_id") or ""),
            provider_chat_id=delivery.provider_chat_id,
            provider_thread_id=delivery.provider_thread_id,
            reply_to_message_id=str(metadata.get("reply_to_message_id") or ""),
            run_status=str(metadata.get("run_status") or "completed"),
            text="",
            attachments=[],
            metadata=metadata,
        )

        payload = dict(delivery.payload)
        if delivery.delivery_type == "artifact_ref":
            request_payload = OutboundFile.model_validate(payload)
            send_callable = adapter.send_outbound_file
        elif delivery.delivery_type == "reaction":
            request_payload = OutboundReaction.model_validate(payload)
            send_callable = adapter.send_outbound_reaction
        else:
            request_payload = OutboundMessage.model_validate(payload)
            send_callable = adapter.send_outbound_message

        retried_delivery = await self._deliver_message(
            context=context,
            adapter=adapter,
            envelope=envelope,
            delivery_id=delivery.id,
            delivery_type=delivery.delivery_type,
            payload=payload,
            send_callable=send_callable,
            request_payload=request_payload,
            force=True,
        )
        if delivery.delivery_type == "reaction":
            self._sync_quick_response_event_metadata(retried_delivery)
        else:
            self._refresh_inbound_event_status_from_deliveries(
                inbound_event_id=str(delivery.inbound_event_id or ""),
                run_id=delivery.run_id,
            )
        return retried_delivery

    def build_reply_envelope(self, run: RunRecord) -> ReplyEnvelope | None:
        """把终态 run 转换成统一的渠道回复结构。"""
        if not self._is_integration_root_run(run):
            return None

        self._ensure_stores()
        integration_id = str(run.run_metadata.get("integration_id") or "").strip()
        provider_chat_id = str(run.run_metadata.get("provider_chat_id") or "").strip()
        provider_thread_id = str(run.run_metadata.get("provider_thread_id") or "").strip()
        reply_to_message_id = str(run.run_metadata.get("provider_message_id") or "").strip()
        if not integration_id or not provider_chat_id:
            return None

        integration = self._integration_store.get_integration(integration_id)
        if integration is None:
            return None

        channel_context = run.run_metadata.get("channel_context") or {}
        provider_root_message_id = str(
            run.run_metadata.get("provider_root_message_id")
            or channel_context.get("root_id")
            or ""
        ).strip()
        provider_parent_message_id = str(
            run.run_metadata.get("provider_parent_message_id")
            or channel_context.get("parent_id")
            or ""
        ).strip()
        reply_text = self._build_reply_text(run)
        attachments = self._build_reply_attachments(run, integration)
        if not reply_text and not attachments:
            reply_text = "任务已处理完成。"

        return ReplyEnvelope(
            integration_id=integration_id,
            channel_kind=str(run.run_metadata.get("channel_kind") or integration.kind),
            run_id=run.id,
            session_id=run.session_id,
            inbound_event_id=str(run.run_metadata.get("inbound_event_id") or ""),
            binding_id=str(run.run_metadata.get("binding_id") or ""),
            provider_chat_id=provider_chat_id,
            provider_thread_id=provider_thread_id,
            reply_to_message_id=reply_to_message_id,
            run_status=run.status,
            text=reply_text,
            attachments=attachments,
            metadata={
                "goal": run.goal,
                "route_source": str(run.run_metadata.get("route_source") or ""),
                "session_key": str(run.run_metadata.get("session_key") or ""),
                "primary_artifact_id": run.deliverable_manifest.primary_artifact_id,
                "provider_root_message_id": provider_root_message_id,
                "provider_parent_message_id": provider_parent_message_id,
                "reply_context": str(channel_context.get("reply_context") or ""),
            },
        )

    async def dispatch_envelope(
        self,
        envelope: ReplyEnvelope,
        *,
        force: bool = False,
    ) -> None:
        """执行实际的渠道回写与投递落库。"""
        self._ensure_stores()
        integration = self._integration_store.get_integration(envelope.integration_id)
        if integration is None:
            raise KeyError(f"Integration not found: {envelope.integration_id}")
        adapter = self._adapter_registry.get(integration.kind)
        context = ChannelContext(
            integration=integration,
            credentials=self._resolve_credentials(envelope.integration_id),
        )

        delivery_ids: list[str] = []
        provider_message_ids: list[str] = []
        failed_error = ""

        text_limit = self._text_limit(integration)
        text_chunks = self._split_text_chunks(envelope.text, text_limit)
        if text_chunks:
            delivery_type = "error" if envelope.run_status != "completed" else "text"
            for index, chunk in enumerate(text_chunks, start=1):
                message = OutboundMessage(
                    target_id=envelope.provider_chat_id,
                    reply_to_message_id=envelope.reply_to_message_id,
                    thread_id=envelope.provider_thread_id,
                    message_type="text",
                    text=chunk,
                    dedup_key=f"{envelope.run_id}:text:{index}",
                    metadata={
                        **envelope.metadata,
                        "chunk_index": index,
                        "chunk_count": len(text_chunks),
                        "run_status": envelope.run_status,
                    },
                )
                delivery = await self._deliver_message(
                    context=context,
                    adapter=adapter,
                    envelope=envelope,
                    delivery_id=f"{envelope.run_id}:text:{index}",
                    delivery_type=delivery_type,
                    payload=message.model_dump(mode="python"),
                    send_callable=adapter.send_outbound_message,
                    request_payload=message,
                    force=force,
                )
                delivery_ids.append(delivery.id)
                if delivery.status != "delivered":
                    failed_error = delivery.error_summary or "文本回写失败。"
                    break
                if delivery.provider_message_id:
                    provider_message_ids.append(delivery.provider_message_id)

        if not failed_error:
            for index, attachment in enumerate(envelope.attachments, start=1):
                outbound_file = OutboundFile(
                    target_id=envelope.provider_chat_id,
                    reply_to_message_id=envelope.reply_to_message_id,
                    thread_id=envelope.provider_thread_id,
                    file_kind=attachment.file_kind,
                    url=attachment.download_url,
                    file_name=attachment.display_name,
                    text_fallback=attachment.text_fallback,
                    dedup_key=f"{envelope.run_id}:artifact:{attachment.artifact_id or index}",
                    metadata={
                        **envelope.metadata,
                        **attachment.metadata,
                        "artifact_id": attachment.artifact_id,
                        "run_status": envelope.run_status,
                    },
                )
                delivery = await self._deliver_message(
                    context=context,
                    adapter=adapter,
                    envelope=envelope,
                    delivery_id=f"{envelope.run_id}:artifact:{attachment.artifact_id or index}",
                    delivery_type="artifact_ref",
                    payload=outbound_file.model_dump(mode="python"),
                    send_callable=adapter.send_outbound_file,
                    request_payload=outbound_file,
                    force=force,
                )
                delivery_ids.append(delivery.id)
                if delivery.status != "delivered":
                    failed_error = delivery.error_summary or "产物引用回写失败。"
                    break
                if delivery.provider_message_id:
                    provider_message_ids.append(delivery.provider_message_id)

        if failed_error:
            self._finalize_inbound_event(
                inbound_event_id=envelope.inbound_event_id,
                next_status="failed",
                error_message=failed_error,
                metadata_updates={
                    "reply_dispatch_status": "failed",
                    "delivery_ids": delivery_ids,
                    "provider_reply_message_ids": provider_message_ids,
                    "run_status": envelope.run_status,
                },
            )
            return

        self._finalize_inbound_event(
            inbound_event_id=envelope.inbound_event_id,
            next_status="completed",
            metadata_updates={
                "reply_dispatch_status": "delivered",
                "reply_dispatched_at": utc_now_iso(),
                "delivery_ids": delivery_ids,
                "provider_reply_message_ids": provider_message_ids,
                "run_status": envelope.run_status,
            },
        )

    async def _deliver_message(
        self,
        *,
        context: ChannelContext,
        adapter,
        envelope: ReplyEnvelope,
        delivery_id: str,
        delivery_type: str,
        payload: dict[str, Any],
        send_callable,
        request_payload,
        force: bool,
    ) -> OutboundDeliveryRecord:
        existing = self._delivery_store.get_delivery(delivery_id)
        if existing is None:
            existing = self._delivery_store.create_delivery(
                OutboundDeliveryRecord(
                    id=delivery_id,
                    integration_id=envelope.integration_id,
                    run_id=envelope.run_id,
                    session_id=envelope.session_id,
                    inbound_event_id=envelope.inbound_event_id or None,
                    provider_chat_id=envelope.provider_chat_id,
                    provider_thread_id=envelope.provider_thread_id,
                    provider_message_id="",
                    delivery_type=delivery_type,
                    payload=payload,
                    status="pending",
                    metadata={
                        **envelope.metadata,
                        "binding_id": envelope.binding_id,
                        "reply_to_message_id": envelope.reply_to_message_id,
                        "run_status": envelope.run_status,
                    },
                    created_at=utc_now_iso(),
                    updated_at=utc_now_iso(),
                )
            )
        elif existing.status in {"delivered", "failed"} and not force:
            return existing
        elif force and existing.status in {"delivered", "failed"}:
            existing = self._delivery_store.update_delivery(
                existing.model_copy(
                    update={
                        "status": "pending",
                        "provider_message_id": "",
                        "error_summary": "",
                        "updated_at": utc_now_iso(),
                    }
                )
            )

        max_attempts = self._max_attempts(context.integration)
        backoff_seconds = self._retry_backoff_seconds(context.integration)
        delivery = existing
        start_attempt = delivery.attempt_count + 1

        for attempt_number in range(start_attempt, max_attempts + 1):
            sending_status = "retrying" if delivery.status == "retrying" else "pending"
            if delivery.status == sending_status:
                delivery = delivery.transition_to("sending", changed_at=utc_now_iso())
            elif delivery.status == "failed" and force:
                delivery = delivery.model_copy(
                    update={
                        "status": "sending",
                        "updated_at": utc_now_iso(),
                    }
                )
            elif delivery.status != "sending":
                delivery = delivery.transition_to("sending", changed_at=utc_now_iso())
            delivery = self._delivery_store.update_delivery(delivery)

            started_at = utc_now_iso()
            try:
                result = await send_callable(context, request_payload)
            except Exception as exc:
                result = OutboundSendResult(
                    ok=False,
                    provider_chat_id=envelope.provider_chat_id,
                    provider_thread_id=envelope.provider_thread_id,
                    raw_response={"exception_type": exc.__class__.__name__},
                    error=adapter.normalize_error(exc),
                )

            finished_at = utc_now_iso()
            error_summary = ""
            response_payload: dict[str, Any] = {}
            if getattr(result, "error", None) is not None:
                error_summary = str(result.error.message or "").strip()
            if getattr(result, "raw_response", None) is not None:
                response_payload = {
                    "raw_response": result.raw_response,
                    "http_status": result.http_status,
                    "metadata": getattr(result, "metadata", {}) or {},
                }

            self._delivery_store.create_attempt(
                DeliveryAttemptRecord(
                    id=f"{delivery.id}:attempt:{attempt_number}:{finished_at}",
                    delivery_id=delivery.id,
                    attempt_number=attempt_number,
                    status="delivered" if result.ok else "failed",
                    request_payload=payload,
                    response_payload=response_payload,
                    error_summary=error_summary,
                    started_at=started_at,
                    finished_at=finished_at,
                )
            )

            if result.ok:
                delivery = delivery.transition_to(
                    "delivered",
                    changed_at=finished_at,
                    attempt_count=attempt_number,
                    error_summary="",
                    provider_message_id=str(result.provider_message_id or ""),
                )
                delivery = delivery.model_copy(
                    update={
                        "provider_chat_id": str(result.provider_chat_id or delivery.provider_chat_id),
                        "provider_thread_id": str(
                            result.provider_thread_id or delivery.provider_thread_id
                        ),
                        "metadata": {
                            **delivery.metadata,
                            "http_status": result.http_status,
                            "response_metadata": getattr(result, "metadata", {}) or {},
                        },
                    }
                )
                return self._delivery_store.update_delivery(delivery)

            retryable = bool(getattr(result, "error", None) and result.error.retryable)
            has_retry = retryable and attempt_number < max_attempts
            if has_retry:
                delivery = delivery.transition_to(
                    "retrying",
                    changed_at=finished_at,
                    attempt_count=attempt_number,
                    error_summary=error_summary or "渠道回写失败，等待重试。",
                )
                delivery = self._delivery_store.update_delivery(delivery)
                if backoff_seconds > 0:
                    await asyncio.sleep(backoff_seconds * max(1, 2 ** (attempt_number - 1)))
                continue

            delivery = delivery.transition_to(
                "failed",
                changed_at=finished_at,
                attempt_count=attempt_number,
                error_summary=error_summary or "渠道回写失败。",
            )
            return self._delivery_store.update_delivery(delivery)

        return delivery

    def _build_reply_text(self, run: RunRecord) -> str:
        if run.status == "completed":
            assistant_message = self._last_assistant_message(run.session_id)
            if assistant_message is not None:
                summary = message_content_summary(assistant_message.content).strip()
                if summary:
                    return summary
            if run.deliverable_manifest.items:
                return "任务已处理完成，已生成可回传产物。"
            return "任务已处理完成。"

        error_summary = run.error_summary.strip() or self._default_run_status_message(run.status)
        return f"处理失败：{error_summary}"

    def _build_reply_attachments(
        self,
        run: RunRecord,
        integration: IntegrationConfigRecord,
    ) -> list[ReplyAttachmentRef]:
        if run.status != "completed":
            return []

        run_store = self._session_manager._run_store
        if run_store is None:
            return []

        attachments: list[ReplyAttachmentRef] = []
        seen_keys: set[str] = set()
        max_artifacts = self._max_artifact_refs(integration)
        sent_runtime_tool_paths = self._sent_runtime_tool_file_paths(run.id)
        artifacts = (
            self._collect_ordered_reply_artifacts(run)
            if integration.kind == "feishu"
            else self._collect_manifest_artifacts(run)
        )
        for artifact in artifacts:
            if not self._artifact_is_reply_file_candidate(artifact):
                continue
            attachment = self._artifact_to_reply_ref(
                artifact,
                integration,
                session_id=run.session_id,
            )
            local_path = str(attachment.metadata.get("local_path") or "").strip()
            normalized_local_path = self._normalize_delivery_local_path(local_path)
            if normalized_local_path and normalized_local_path in sent_runtime_tool_paths:
                continue
            dedupe_key = self._attachment_dedupe_key(artifact, attachment)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            attachments.append(attachment)
            if len(attachments) >= max_artifacts:
                break
        return attachments

    def _collect_manifest_artifacts(self, run: RunRecord) -> list[ArtifactRecord]:
        run_store = self._session_manager._run_store
        if run_store is None:
            return []
        artifact_map = {
            artifact.id: artifact
            for artifact in run_store.list_artifacts(run.id)
        }
        ordered: list[ArtifactRecord] = []
        for item in run.deliverable_manifest.items:
            artifact = artifact_map.get(item.artifact_id)
            if artifact is not None:
                ordered.append(artifact)
        return ordered

    def _artifact_to_reply_ref(
        self,
        artifact: ArtifactRecord,
        integration: IntegrationConfigRecord,
        *,
        session_id: str,
    ) -> ReplyAttachmentRef:
        display_name = artifact.display_name or Path(artifact.uri).name or artifact.id
        relative_url = f"/api/artifacts/{artifact.id}"
        base_url = str(
            integration.config.get("public_base_url")
            or os.environ.get("MINI_AGENT_PUBLIC_BASE_URL")
            or ""
        ).strip()
        download_url = (
            f"{base_url.rstrip('/')}{relative_url}"
            if base_url
            else relative_url
        )
        local_path = self._resolve_artifact_local_path(session_id, artifact.uri)
        text_fallback = (
            f"已生成产物：{display_name}\n"
            f"下载地址：{download_url}"
        )
        return ReplyAttachmentRef(
            artifact_id=artifact.id,
            file_kind=self._infer_file_kind(artifact.mime_type),
            display_name=display_name,
            mime_type=artifact.mime_type,
            download_url=download_url,
            text_fallback=text_fallback,
            metadata={
                "uri": artifact.uri,
                "local_path": local_path,
                "artifact_type": artifact.artifact_type,
                "preview_kind": artifact.preview_kind,
                "summary": artifact.summary,
            },
        )

    def _collect_ordered_reply_artifacts(self, run: RunRecord) -> list[ArtifactRecord]:
        run_store = self._session_manager._run_store
        if run_store is None:
            return []

        reply_runs = self._collect_reply_runs(run)
        artifacts_by_run = {
            item.id: run_store.list_artifacts(item.id)
            for item in reply_runs
        }
        artifact_map = {
            artifact.id: artifact
            for artifacts in artifacts_by_run.values()
            for artifact in artifacts
        }

        ordered: list[ArtifactRecord] = []
        for reply_run in reply_runs:
            for deliverable in reply_run.deliverable_manifest.items:
                artifact = artifact_map.get(deliverable.artifact_id)
                if artifact is not None:
                    ordered.append(artifact)

        ordered_ids = {item.id for item in ordered}
        remaining = [
            artifact
            for artifacts in artifacts_by_run.values()
            for artifact in artifacts
            if artifact.id not in ordered_ids
        ]
        remaining.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        ordered.extend(remaining)
        return ordered

    def _collect_reply_runs(self, root_run: RunRecord) -> list[RunRecord]:
        run_store = self._session_manager._run_store
        if run_store is None:
            return [root_run]

        ordered_runs: list[RunRecord] = [root_run]
        queue: list[str] = [root_run.id]
        seen: set[str] = {root_run.id}
        while queue:
            parent_run_id = queue.pop(0)
            children = run_store.list_runs(parent_run_id=parent_run_id)
            children.sort(key=lambda item: (item.created_at, item.id))
            for child in children:
                if child.id in seen:
                    continue
                seen.add(child.id)
                ordered_runs.append(child)
                queue.append(child.id)
        return ordered_runs

    @staticmethod
    def _artifact_is_reply_file_candidate(artifact: ArtifactRecord) -> bool:
        if artifact.artifact_type not in {"workspace_file", "document"}:
            return False
        if artifact.source == "system_generated":
            return False
        normalized_uri = str(artifact.uri or "").strip()
        if not normalized_uri or "://" in normalized_uri:
            return False
        return True

    def _sent_runtime_tool_file_paths(self, run_id: str) -> set[str]:
        if self._delivery_store is None:
            return set()
        sent_paths: set[str] = set()
        for delivery in self._delivery_store.list_deliveries(run_id=run_id):
            if delivery.delivery_type != "tool_file" or delivery.status != "delivered":
                continue
            local_path = str((delivery.payload or {}).get("metadata", {}).get("local_path") or "").strip()
            normalized = self._normalize_delivery_local_path(local_path)
            if normalized:
                sent_paths.add(normalized)
        return sent_paths

    def _attachment_dedupe_key(
        self,
        artifact: ArtifactRecord,
        attachment: ReplyAttachmentRef,
    ) -> str:
        local_path = self._normalize_delivery_local_path(
            str(attachment.metadata.get("local_path") or "").strip()
        )
        if local_path:
            return f"path:{local_path}"
        normalized_uri = str(artifact.uri or "").strip()
        if normalized_uri:
            return f"uri:{normalized_uri}"
        return f"artifact:{artifact.id}"

    @staticmethod
    def _normalize_delivery_local_path(path_value: str) -> str:
        normalized = str(path_value or "").strip()
        if not normalized:
            return ""
        try:
            return str(Path(normalized).resolve())
        except Exception:
            return normalized

    def _resolve_artifact_local_path(self, session_id: str, artifact_uri: str) -> str:
        session = self._session_manager.get_session_info(session_id)
        if session is None:
            return ""

        normalized_uri = str(artifact_uri or "").strip()
        if not normalized_uri or "://" in normalized_uri:
            return ""

        workspace_dir = Path(str(session.get("workspace_dir") or "")).resolve()
        artifact_path = Path(normalized_uri)
        if artifact_path.is_absolute():
            return str(artifact_path)
        return str((workspace_dir / artifact_path).resolve())

    @staticmethod
    def _infer_file_kind(mime_type: str) -> str:
        normalized = str(mime_type or "").strip().lower()
        if normalized.startswith("image/"):
            return "image"
        if normalized.startswith("audio/"):
            return "audio"
        if normalized.startswith("video/"):
            return "media"
        return "file"

    def _last_assistant_message(self, session_id: str) -> Message | None:
        session_store = self._session_manager._session_store
        if session_store is None:
            return None
        messages = session_store.get_messages(session_id)
        for message in reversed(messages):
            if message.role == "assistant":
                return message
        return None

    @staticmethod
    def _split_text_chunks(text: str, limit: int) -> list[str]:
        normalized = str(text or "").strip()
        if not normalized:
            return []
        if limit <= 0 or len(normalized) <= limit:
            return [normalized]

        chunks: list[str] = []
        remaining = normalized
        min_split_index = max(1, limit // 2)
        while remaining:
            if len(remaining) <= limit:
                chunks.append(remaining)
                break

            split_index = remaining.rfind("\n\n", 0, limit + 1)
            if split_index < min_split_index:
                split_index = remaining.rfind("\n", 0, limit + 1)
            if split_index < min_split_index:
                split_index = remaining.rfind(" ", 0, limit + 1)
            if split_index < min_split_index:
                split_index = limit

            chunk = remaining[:split_index].strip()
            if not chunk:
                chunk = remaining[:limit].strip()
                split_index = len(chunk)
            chunks.append(chunk)
            remaining = remaining[split_index:].strip()

        return chunks

    @staticmethod
    def _default_run_status_message(run_status: str) -> str:
        normalized = str(run_status or "").strip().lower()
        if normalized == "timed_out":
            return "执行超时。"
        if normalized == "cancelled":
            return "任务已取消。"
        if normalized == "interrupted":
            return "任务已中断。"
        if normalized == "failed":
            return "任务执行失败。"
        return "任务未成功完成。"

    @staticmethod
    def _quick_response_task_key(run_id: str) -> str:
        return f"{run_id}:quick-response"

    @staticmethod
    def _quick_response_delivery_id(run_id: str) -> str:
        return f"{run_id}:reaction"

    def _sync_quick_response_event_metadata(self, delivery: OutboundDeliveryRecord) -> None:
        event_id = str(delivery.metadata.get("quick_response_event_id") or "").strip()
        if not event_id:
            return

        payload = delivery.payload if isinstance(delivery.payload, dict) else {}
        metadata_updates: dict[str, Any] = {
            "quick_response_status": delivery.status,
            "quick_response_delivery_id": delivery.id,
            "quick_response_reaction_type": str(
                delivery.metadata.get("quick_response_reaction_type")
                or payload.get("reaction_type")
                or ""
            ),
            "quick_response_target_message_id": str(
                delivery.metadata.get("quick_response_target_message_id")
                or payload.get("message_id")
                or ""
            ),
        }
        if delivery.status == "delivered":
            metadata_updates["quick_response_delivered_at"] = utc_now_iso()
        elif delivery.status == "failed":
            metadata_updates["quick_response_error"] = delivery.error_summary

        self._finalize_inbound_event(
            inbound_event_id=event_id,
            next_status="",
            metadata_updates=metadata_updates,
        )

    def _finalize_inbound_event(
        self,
        *,
        inbound_event_id: str,
        next_status: str,
        error_message: str | None = None,
        metadata_updates: dict[str, Any] | None = None,
    ) -> InboundEventRecord | None:
        if not inbound_event_id:
            return None

        self._ensure_stores()
        event = self._inbound_event_store.get_event(inbound_event_id)
        if event is None:
            return None

        updated = event
        if next_status and next_status != event.normalized_status:
            if event.can_transition_to(next_status):
                updated = event.transition_to(next_status, error_message=error_message)
            else:
                payload: dict[str, Any] = {"normalized_status": next_status}
                if error_message is not None:
                    payload["normalized_error"] = error_message
                updated = event.model_copy(update=payload)
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

    def _refresh_inbound_event_status_from_deliveries(
        self,
        *,
        inbound_event_id: str,
        run_id: str,
    ) -> None:
        if not inbound_event_id:
            return

        deliveries = [
            delivery
            for delivery in self._delivery_store.list_deliveries(run_id=run_id)
            if str(delivery.inbound_event_id or "") == inbound_event_id
            and delivery.delivery_type != "reaction"
        ]
        if not deliveries:
            return

        delivery_ids = [delivery.id for delivery in deliveries]
        provider_reply_message_ids = [
            delivery.provider_message_id
            for delivery in deliveries
            if str(delivery.provider_message_id or "").strip()
        ]
        failed_delivery = next((delivery for delivery in deliveries if delivery.status == "failed"), None)
        if failed_delivery is not None:
            self._finalize_inbound_event(
                inbound_event_id=inbound_event_id,
                next_status="failed",
                error_message=failed_delivery.error_summary or "仍有渠道回写失败。",
                metadata_updates={
                    "reply_dispatch_status": "failed",
                    "delivery_ids": delivery_ids,
                    "provider_reply_message_ids": provider_reply_message_ids,
                },
            )
            return

        if all(delivery.status == "delivered" for delivery in deliveries):
            self._finalize_inbound_event(
                inbound_event_id=inbound_event_id,
                next_status="completed",
                metadata_updates={
                    "reply_dispatch_status": "delivered",
                    "reply_dispatched_at": utc_now_iso(),
                    "delivery_ids": delivery_ids,
                    "provider_reply_message_ids": provider_reply_message_ids,
                },
            )

    def _ensure_stores(self) -> None:
        if (
            self._integration_store is not None
            and self._delivery_store is not None
            and self._inbound_event_store is not None
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
        self._delivery_store = DeliveryStore(resolved_db_path)
        self._inbound_event_store = InboundEventStore(resolved_db_path)

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

    @staticmethod
    def _is_integration_root_run(run: RunRecord) -> bool:
        if run.parent_run_id is not None:
            return False
        return str(run.run_metadata.get("source_kind") or "").strip() in {
            "integration",
            "scheduled_task",
        }

    @staticmethod
    def _text_limit(integration: IntegrationConfigRecord) -> int:
        raw_value = integration.config.get("outbound_text_limit", 1500)
        try:
            return max(200, int(raw_value))
        except (TypeError, ValueError):
            return 1500

    @staticmethod
    def _max_attempts(integration: IntegrationConfigRecord) -> int:
        raw_value = integration.config.get("outbound_max_attempts", 3)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 3

    @staticmethod
    def _retry_backoff_seconds(integration: IntegrationConfigRecord) -> float:
        raw_value = integration.config.get("outbound_retry_backoff_seconds", 1.0)
        try:
            return max(0.0, float(raw_value))
        except (TypeError, ValueError):
            return 1.0

    @staticmethod
    def _max_artifact_refs(integration: IntegrationConfigRecord) -> int:
        raw_value = integration.config.get("outbound_artifact_limit", 5)
        try:
            return max(1, int(raw_value))
        except (TypeError, ValueError):
            return 5

    async def shutdown(self) -> None:
        """停止未完成的后台回写任务。"""
        pending = [
            task
            for task in self._scheduled_tasks.values()
            if not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._scheduled_tasks.clear()
