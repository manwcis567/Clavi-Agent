"""WeChat iLink long-poll runtime service."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ..integration_models import IntegrationConfigRecord, IntegrationCredentialRecord
from ..integration_store import IntegrationStore
from ..sqlite_schema import utc_now_iso
from .gateway import IntegrationGateway, IntegrationGatewayError
from .models import ChannelRequest
from .runtime_bridge import IntegrationRunBridge, IntegrationRunBridgeError
from .wechat_ilink import WeChatILinkClient, WeChatILinkCredentials

logger = logging.getLogger(__name__)

SESSION_EXPIRED_ERRCODE = -14
WECHAT_RUNTIME_METADATA_KEY = "wechat_runtime"
WECHAT_SYNC_CURSOR_KEY = "sync_cursor"


@dataclass(slots=True)
class _WeChatMonitorHandle:
    integration_id: str
    runner_task: asyncio.Task[None]
    spec: tuple[str, ...]


class WeChatLongPollService:
    """Manage native WeChat iLink polling workers for active integrations."""

    def __init__(
        self,
        session_manager,
        *,
        integration_gateway: IntegrationGateway,
        integration_run_bridge: IntegrationRunBridge,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._session_manager = session_manager
        self._integration_gateway = integration_gateway
        self._integration_run_bridge = integration_run_bridge
        self._transport = transport
        self._integration_store: IntegrationStore | None = None
        self._handles: dict[str, _WeChatMonitorHandle] = {}
        self._sync_lock = asyncio.Lock()

    async def start(self) -> None:
        await self._session_manager.initialize()
        await self.sync_all()

    async def shutdown(self) -> None:
        async with self._sync_lock:
            for integration_id in list(self._handles):
                await self._stop_monitor_locked(integration_id)

    async def sync_all(self) -> None:
        async with self._sync_lock:
            self._ensure_store()
            records = self._integration_store.list_integrations(kind="wechat", status=None)
            active_ids = {
                record.id
                for record in records
                if self._should_manage(record)
            }
            for integration_id in list(self._handles):
                if integration_id not in active_ids:
                    await self._stop_monitor_locked(integration_id)

            for record in records:
                if not self._should_manage(record):
                    continue
                await self._sync_integration_locked(record)

    async def sync_integration(self, integration_id: str) -> None:
        async with self._sync_lock:
            self._ensure_store()
            record = self._integration_store.get_integration(integration_id)
            if record is None or not self._should_manage(record):
                await self._stop_monitor_locked(integration_id)
                return
            await self._sync_integration_locked(record)

    def _ensure_store(self) -> None:
        if self._integration_store is not None:
            return

        config = self._session_manager._config
        if config is None:
            raise RuntimeError("SessionManager configuration is not initialized.")

        db_path = Path(config.agent.session_store_path)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        self._integration_store = IntegrationStore(db_path.resolve())

    def _should_manage(self, record: IntegrationConfigRecord) -> bool:
        if record.kind != "wechat":
            return False
        if record.status != "active":
            return False
        if str(record.metadata.get("deleted_at") or "").strip():
            return False
        setup_payload = record.metadata.get("wechat_setup")
        if not isinstance(setup_payload, dict):
            return False
        if str(setup_payload.get("state") or "").strip().lower() != "succeeded":
            return False
        try:
            self._build_client_spec(record)
        except Exception:
            return False
        return True

    async def _sync_integration_locked(self, record: IntegrationConfigRecord) -> None:
        existing = self._handles.get(record.id)
        try:
            spec = self._build_client_spec(record)
            if existing is not None and existing.spec == spec and not existing.runner_task.done():
                return
            if existing is not None:
                await self._stop_monitor_locked(record.id)

            task = asyncio.create_task(
                self._run_monitor(record.id, spec),
                name=f"wechat-ilink-monitor:{record.id}",
            )
            handle = _WeChatMonitorHandle(
                integration_id=record.id,
                runner_task=task,
                spec=spec,
            )
            self._handles[record.id] = handle
            task.add_done_callback(
                lambda finished_task, integration_id=record.id: self._handle_runner_done(
                    integration_id,
                    finished_task,
                )
            )
        except Exception as exc:
            await self._stop_monitor_locked(record.id)
            self._mark_integration_error(record.id, str(exc))
            logger.exception("Failed to start WeChat iLink worker. integration_id=%s", record.id)
        else:
            self._clear_integration_error(record.id)

    def _build_client_spec(self, record: IntegrationConfigRecord) -> tuple[str, ...]:
        credentials = self._resolve_credentials(record.id)
        bot_token = str(credentials.get("bot_token") or "").strip()
        ilink_bot_id = str(credentials.get("ilink_bot_id") or "").strip()
        base_url = str(credentials.get("base_url") or "").strip() or "https://ilinkai.weixin.qq.com"
        ilink_user_id = str(credentials.get("ilink_user_id") or "").strip()
        if not bot_token:
            raise RuntimeError("WeChat iLink bot_token is missing.")
        if not ilink_bot_id:
            raise RuntimeError("WeChat iLink ilink_bot_id is missing.")
        return (bot_token, ilink_bot_id, base_url, ilink_user_id)

    async def _run_monitor(self, integration_id: str, spec: tuple[str, ...]) -> None:
        bot_token, ilink_bot_id, base_url, ilink_user_id = spec
        client = WeChatILinkClient(
            WeChatILinkCredentials(
                bot_token=bot_token,
                ilink_bot_id=ilink_bot_id,
                base_url=base_url,
                ilink_user_id=ilink_user_id,
            ),
            transport=self._transport,
        )
        record = self._integration_store.get_integration(integration_id) if self._integration_store else None
        cursor = self._load_sync_cursor(record)
        restart_delay_seconds = 3.0

        while True:
            try:
                response = await client.get_updates(cursor)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "WeChat iLink poll failed, retrying later. integration_id=%s detail=%s",
                    integration_id,
                    exc,
                )
                await asyncio.sleep(restart_delay_seconds)
                restart_delay_seconds = min(restart_delay_seconds * 2, 30.0)
                continue

            restart_delay_seconds = 3.0
            errcode = int(response.get("errcode") or 0)
            ret = int(response.get("ret") or 0)

            if errcode == SESSION_EXPIRED_ERRCODE:
                if cursor:
                    logger.warning(
                        "WeChat iLink cursor expired, resetting cursor. integration_id=%s",
                        integration_id,
                    )
                    cursor = ""
                    self._persist_sync_cursor(integration_id, "")
                    await asyncio.sleep(5.0)
                    continue
                self._mark_setup_failed(
                    integration_id,
                    "The WeChat session expired. Reconnect the channel by scanning a new QR code.",
                )
                self._mark_integration_error(
                    integration_id,
                    "The WeChat session expired. Reconnect the channel by scanning a new QR code.",
                )
                return

            if ret != 0 or errcode != 0:
                logger.warning(
                    "WeChat iLink returned an error. integration_id=%s ret=%s errcode=%s errmsg=%s",
                    integration_id,
                    ret,
                    errcode,
                    str(response.get("errmsg") or ""),
                )
                await asyncio.sleep(3.0)
                continue

            next_cursor = str(response.get("get_updates_buf") or "").strip()
            if next_cursor and next_cursor != cursor:
                cursor = next_cursor
                self._persist_sync_cursor(integration_id, cursor)

            for message in list(response.get("msgs") or []):
                if not self._should_handle_message(message):
                    continue
                await self._handle_message_payload(integration_id, message)

    async def _handle_message_payload(self, integration_id: str, message_payload: dict[str, Any]) -> None:
        request = ChannelRequest(
            method="ILINK_POLL",
            path=f"/api/integrations/wechat/{integration_id}/ilink",
            headers={
                "content-type": "application/json",
                "x-wechat-connection-mode": "ilink_poll",
            },
            body=json.dumps({"message": message_payload}, ensure_ascii=False).encode("utf-8"),
            received_at=utc_now_iso(),
            remote_addr="wechat-ilink",
        )
        try:
            result = await self._integration_gateway.handle_channel_request(
                integration_id,
                request,
                expected_kind="wechat",
            )
        except IntegrationGatewayError as exc:
            logger.warning(
                "WeChat gateway processing failed. integration_id=%s detail=%s",
                integration_id,
                exc.detail,
            )
            return

        try:
            await self._integration_run_bridge.bridge_gateway_result(result)
        except IntegrationRunBridgeError as exc:
            logger.warning(
                "WeChat bridge processing failed. integration_id=%s detail=%s",
                integration_id,
                exc.detail,
            )

    async def _stop_monitor_locked(self, integration_id: str) -> None:
        handle = self._handles.pop(integration_id, None)
        if handle is None:
            return
        handle.runner_task.cancel()
        with suppress(asyncio.CancelledError):
            await handle.runner_task

    def _handle_runner_done(self, integration_id: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        self._handles.pop(integration_id, None)
        with suppress(Exception):
            exc = task.exception()
            if exc is None:
                return
            logger.exception(
                "WeChat iLink worker exited unexpectedly. integration_id=%s",
                integration_id,
                exc_info=exc,
            )

    def _resolve_credentials(self, integration_id: str) -> dict[str, str]:
        assert self._integration_store is not None
        resolved: dict[str, str] = {}
        for record in self._integration_store.list_credentials(integration_id):
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
        if reference.startswith("env:"):
            import os

            return os.environ.get(reference[4:], "").strip()

        import os

        return os.environ.get(reference, "").strip()

    @staticmethod
    def _load_sync_cursor(record: IntegrationConfigRecord | None) -> str:
        if record is None:
            return ""
        runtime_payload = record.metadata.get(WECHAT_RUNTIME_METADATA_KEY)
        if not isinstance(runtime_payload, dict):
            return ""
        return str(runtime_payload.get(WECHAT_SYNC_CURSOR_KEY) or "").strip()

    def _persist_sync_cursor(self, integration_id: str, cursor: str) -> None:
        if self._integration_store is None:
            return
        record = self._integration_store.get_integration(integration_id)
        if record is None:
            return

        normalized_cursor = str(cursor or "").strip()
        metadata = dict(record.metadata)
        runtime_payload = metadata.get(WECHAT_RUNTIME_METADATA_KEY)
        if not isinstance(runtime_payload, dict):
            runtime_payload = {}

        current_cursor = str(runtime_payload.get(WECHAT_SYNC_CURSOR_KEY) or "").strip()
        if current_cursor == normalized_cursor:
            return

        now = utc_now_iso()
        metadata[WECHAT_RUNTIME_METADATA_KEY] = {
            **runtime_payload,
            WECHAT_SYNC_CURSOR_KEY: normalized_cursor,
            "updated_at": now,
        }
        updated = record.model_copy(
            update={
                "metadata": metadata,
                "updated_at": now,
            }
        )
        self._integration_store.update_integration(updated)

    def _mark_integration_error(self, integration_id: str, message: str) -> None:
        if self._integration_store is None:
            return
        record = self._integration_store.get_integration(integration_id)
        if record is None:
            return
        updated = record.model_copy(
            update={
                "status": "error",
                "last_error": message,
                "updated_at": utc_now_iso(),
            }
        )
        self._integration_store.update_integration(updated)

    def _clear_integration_error(self, integration_id: str) -> None:
        if self._integration_store is None:
            return
        record = self._integration_store.get_integration(integration_id)
        if record is None:
            return
        updates: dict[str, Any] = {
            "last_error": "",
            "updated_at": utc_now_iso(),
        }
        if record.status == "error":
            updates["status"] = "active"
        updated = record.model_copy(update=updates)
        self._integration_store.update_integration(updated)

    def _mark_setup_failed(self, integration_id: str, message: str) -> None:
        if self._integration_store is None:
            return
        record = self._integration_store.get_integration(integration_id)
        if record is None:
            return
        metadata = dict(record.metadata)
        setup_payload = metadata.get("wechat_setup")
        if not isinstance(setup_payload, dict):
            setup_payload = {}
        runtime_payload = metadata.get(WECHAT_RUNTIME_METADATA_KEY)
        if not isinstance(runtime_payload, dict):
            runtime_payload = {}
        now = utc_now_iso()
        metadata["wechat_setup"] = {
            **setup_payload,
            "integration_id": integration_id,
            "state": "failed",
            "message": message,
            "error": message,
            "updated_at": now,
        }
        metadata[WECHAT_RUNTIME_METADATA_KEY] = {
            **runtime_payload,
            WECHAT_SYNC_CURSOR_KEY: "",
            "updated_at": now,
        }
        updated = record.model_copy(
            update={
                "metadata": metadata,
                "updated_at": now,
            }
        )
        self._integration_store.update_integration(updated)

    @staticmethod
    def _should_handle_message(message: dict[str, Any]) -> bool:
        message_type = int(message.get("message_type") or 0)
        message_state = int(message.get("message_state") or 0)
        return message_type == 1 and message_state == 2
