"""飞书长连接接入服务。"""

from __future__ import annotations

import asyncio
import logging
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..integration_models import IntegrationConfigRecord, IntegrationCredentialRecord
from ..integration_store import IntegrationStore
from ..sqlite_schema import utc_now_iso
from .feishu_adapter import FeishuAdapter
from .gateway import IntegrationGateway, IntegrationGatewayError
from .models import ChannelRequest
from .runtime_bridge import IntegrationRunBridge, IntegrationRunBridgeError

logger = logging.getLogger(__name__)


class FeishuLongConnectionError(RuntimeError):
    """飞书长连接启动或同步异常。"""


@dataclass(slots=True)
class _FeishuClientHandle:
    integration_id: str
    client: Any
    runner_task: asyncio.Task[None]
    stop_event: asyncio.Event
    spec: tuple[str, ...]


class FeishuLongConnectionService:
    """基于飞书官方 SDK 长连接接收事件，并复用现有网关链路。"""

    def __init__(
        self,
        session_manager,
        *,
        integration_gateway: IntegrationGateway,
        integration_run_bridge: IntegrationRunBridge,
    ) -> None:
        self._session_manager = session_manager
        self._integration_gateway = integration_gateway
        self._integration_run_bridge = integration_run_bridge
        self._integration_store: IntegrationStore | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._worker_thread: threading.Thread | None = None
        self._worker_ready = threading.Event()
        self._handles: dict[str, _FeishuClientHandle] = {}
        self._sync_lock = asyncio.Lock()
        self._lark_sdk: Any | None = None
        self._lark_ws_client_module: Any | None = None

    async def start(self) -> None:
        await self._session_manager.initialize()
        self._main_loop = asyncio.get_running_loop()
        await self.sync_all()

    async def shutdown(self) -> None:
        async with self._sync_lock:
            for integration_id in list(self._handles):
                await self._stop_client_locked(integration_id)
            await self._stop_worker_thread_locked()

    async def sync_all(self) -> None:
        async with self._sync_lock:
            self._ensure_store()
            integrations = self._integration_store.list_integrations(kind="feishu", status=None)
            active_ids = {
                record.id
                for record in integrations
                if self._should_manage(record)
            }
            for integration_id in list(self._handles):
                if integration_id not in active_ids:
                    await self._stop_client_locked(integration_id)

            for record in integrations:
                if not self._should_manage(record):
                    continue
                await self._sync_integration_locked(record)

            if not self._handles:
                await self._stop_worker_thread_locked()

    async def sync_integration(self, integration_id: str) -> None:
        async with self._sync_lock:
            self._ensure_store()
            record = self._integration_store.get_integration(integration_id)
            if record is None or not self._should_manage(record):
                await self._stop_client_locked(integration_id)
                if not self._handles:
                    await self._stop_worker_thread_locked()
                return
            await self._sync_integration_locked(record)

    def _ensure_store(self) -> None:
        if self._integration_store is not None:
            return

        config = self._session_manager._config
        if config is None:
            raise RuntimeError("SessionManager 尚未加载配置。")

        db_path = Path(config.agent.session_store_path)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        self._integration_store = IntegrationStore(db_path.resolve())

    async def _sync_integration_locked(self, record: IntegrationConfigRecord) -> None:
        existing = self._handles.get(record.id)
        try:
            spec = self._build_client_spec(record)
            if existing is not None and existing.spec == spec and not existing.runner_task.done():
                return

            if existing is not None:
                await self._stop_client_locked(record.id)

            await self._ensure_worker_runtime_locked()
            await self._start_client_locked(record, spec)
        except Exception as exc:
            if self._handles.get(record.id) is not None:
                await self._stop_client_locked(record.id)
            self._mark_integration_error(record.id, str(exc))
            logger.exception("飞书长连接启动失败: integration_id=%s", record.id)
        else:
            self._clear_integration_error(record.id)

    def _should_manage(self, record: IntegrationConfigRecord) -> bool:
        if record.kind != "feishu":
            return False
        if record.status != "active":
            return False
        if str(record.metadata.get("deleted_at") or "").strip():
            return False
        return FeishuAdapter.uses_long_connection(record.config)

    def _build_client_spec(self, record: IntegrationConfigRecord) -> tuple[str, ...]:
        credentials = self._resolve_credentials(record.id)
        app_id = str(record.config.get("app_id") or credentials.get("app_id") or "").strip()
        app_secret = str(record.config.get("app_secret") or credentials.get("app_secret") or "").strip()
        if not app_id:
            raise FeishuLongConnectionError("飞书长连接缺少 app_id。")
        if not app_secret:
            raise FeishuLongConnectionError("飞书长连接缺少 app_secret。")
        verification_token = str(
            record.config.get("verification_token")
            or record.config.get("verify_token")
            or credentials.get("verification_token")
            or credentials.get("verify_token")
            or ""
        ).strip()
        encrypt_key = str(record.config.get("encrypt_key") or credentials.get("encrypt_key") or "").strip()
        api_base_url = str(record.config.get("api_base_url") or "").strip()
        return (
            app_id,
            app_secret,
            verification_token,
            encrypt_key,
            api_base_url,
        )

    async def _ensure_worker_runtime_locked(self) -> None:
        self._ensure_sdk_loaded()
        if self._worker_loop is not None and self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._worker_ready.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_main,
            name="clavi-agent-feishu-long-connection",
            daemon=True,
        )
        self._worker_thread.start()
        ready = await asyncio.to_thread(self._worker_ready.wait, 5.0)
        if not ready or self._worker_loop is None:
            raise FeishuLongConnectionError("飞书长连接工作线程初始化失败。")

    def _ensure_sdk_loaded(self) -> None:
        if self._lark_sdk is not None and self._lark_ws_client_module is not None:
            return
        try:
            import lark_oapi as lark_sdk
            import lark_oapi.ws.client as lark_ws_client_module
        except ImportError as exc:
            raise FeishuLongConnectionError(
                "当前环境缺少 `lark-oapi`，无法启用飞书长连接。"
            ) from exc

        self._lark_sdk = lark_sdk
        self._lark_ws_client_module = lark_ws_client_module

    def _worker_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._worker_loop = loop
        if self._lark_ws_client_module is not None:
            self._lark_ws_client_module.loop = loop
        self._worker_ready.set()
        try:
            loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                with suppress(Exception):
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            with suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self._worker_loop = None

    async def _start_client_locked(
        self,
        record: IntegrationConfigRecord,
        spec: tuple[str, ...],
    ) -> None:
        if self._worker_loop is None:
            raise FeishuLongConnectionError("飞书长连接工作线程尚未就绪。")
        future = asyncio.run_coroutine_threadsafe(
            self._start_client_on_worker(record, spec),
            self._worker_loop,
        )
        await asyncio.wrap_future(future)

    async def _start_client_on_worker(
        self,
        record: IntegrationConfigRecord,
        spec: tuple[str, ...],
    ) -> None:
        if self._lark_sdk is None:
            raise FeishuLongConnectionError("飞书 SDK 尚未初始化。")

        app_id, app_secret, verification_token, encrypt_key, api_base_url = spec
        builder = self._lark_sdk.EventDispatcherHandler.builder(encrypt_key, verification_token)
        event_handler = (
            builder
            .register_p2_im_message_receive_v1(self._build_message_handler(record.id))
            .register_p2_im_message_message_read_v1(
                self._build_ignored_event_handler(record.id, "im.message.message_read_v1")
            )
            .build()
        )
        client_kwargs: dict[str, Any] = {
            "event_handler": event_handler,
            "log_level": self._lark_sdk.LogLevel.INFO,
            "auto_reconnect": False,
        }
        if api_base_url:
            client_kwargs["domain"] = api_base_url

        client = self._lark_sdk.ws.Client(app_id, app_secret, **client_kwargs)
        stop_event = asyncio.Event()
        runner_task = asyncio.create_task(self._run_client(record.id, client, stop_event))
        runner_task.add_done_callback(
            lambda task, integration_id=record.id: self._handle_runner_done(integration_id, task)
        )
        handle = _FeishuClientHandle(
            integration_id=record.id,
            client=client,
            runner_task=runner_task,
            stop_event=stop_event,
            spec=spec,
        )
        self._handles[record.id] = handle
        try:
            await asyncio.wait_for(asyncio.shield(runner_task), timeout=0.1)
        except asyncio.TimeoutError:
            return
        except Exception:
            self._handles.pop(record.id, None)
            raise
        else:
            self._handles.pop(record.id, None)
            raise FeishuLongConnectionError("飞书长连接在启动后立即退出。")

    async def _run_client(
        self,
        integration_id: str,
        client: Any,
        stop_event: asyncio.Event,
    ) -> None:
        try:
            reconnect_delay_seconds = 5.0
            while not stop_event.is_set():
                ping_task: asyncio.Task[None] | None = None
                try:
                    await client._connect()
                    ping_task = asyncio.create_task(client._ping_loop())
                    while not stop_event.is_set():
                        if getattr(client, "_conn", None) is None:
                            break
                        try:
                            await asyncio.wait_for(stop_event.wait(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                except Exception as exc:
                    logger.warning(
                        "飞书长连接连接失败，将稍后重试: integration_id=%s detail=%s",
                        integration_id,
                        exc,
                    )
                finally:
                    if ping_task is not None:
                        ping_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await ping_task
                    client._auto_reconnect = False
                    with suppress(Exception):
                        await client._disconnect()

                if stop_event.is_set():
                    break

                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=reconnect_delay_seconds)
                except asyncio.TimeoutError:
                    continue
        finally:
            client._auto_reconnect = False
            with suppress(Exception):
                await client._disconnect()
            self._handles.pop(integration_id, None)

    async def _stop_client_locked(self, integration_id: str) -> None:
        handle = self._handles.get(integration_id)
        if handle is None:
            return
        if self._worker_loop is None:
            self._handles.pop(integration_id, None)
            return
        future = asyncio.run_coroutine_threadsafe(
            self._stop_client_on_worker(integration_id),
            self._worker_loop,
        )
        with suppress(Exception):
            await asyncio.wrap_future(future)

    async def _stop_client_on_worker(self, integration_id: str) -> None:
        handle = self._handles.get(integration_id)
        if handle is None:
            return
        handle.stop_event.set()
        with suppress(Exception):
            await handle.runner_task

    async def _stop_worker_thread_locked(self) -> None:
        if self._worker_loop is not None:
            self._worker_loop.call_soon_threadsafe(self._worker_loop.stop)
        if self._worker_thread is not None:
            await asyncio.to_thread(self._worker_thread.join, 5.0)
        self._worker_thread = None
        self._worker_loop = None
        self._worker_ready.clear()

    def _build_message_handler(self, integration_id: str):
        def _handle_message(data: Any) -> None:
            if self._main_loop is None or self._lark_sdk is None:
                logger.error("飞书长连接事件被忽略：主事件循环尚未初始化。")
                return
            payload_text = self._lark_sdk.JSON.marshal(data)
            future = asyncio.run_coroutine_threadsafe(
                self._handle_event_payload(integration_id, payload_text),
                self._main_loop,
            )
            future.add_done_callback(
                lambda completed: self._log_event_result(integration_id, completed)
            )

        return _handle_message

    @staticmethod
    def _build_ignored_event_handler(integration_id: str, event_type: str):
        def _handle_event(_: Any) -> None:
            logger.debug(
                "Ignoring Feishu long connection event. integration_id=%s event_type=%s",
                integration_id,
                event_type,
            )

        return _handle_event

    @staticmethod
    def _log_event_result(integration_id: str, future) -> None:
        try:
            future.result()
        except Exception:
            logger.exception("飞书长连接事件处理失败: integration_id=%s", integration_id)

    def _handle_runner_done(self, integration_id: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            task_exc = task.exception()
        except Exception as exc:  # pragma: no cover - defensive path
            task_exc = exc
        if task_exc is None:
            return
        if self._main_loop is None:
            logger.exception("飞书长连接任务异常退出: integration_id=%s", integration_id, exc_info=task_exc)
            return
        future = asyncio.run_coroutine_threadsafe(
            self._handle_runner_failure(integration_id, str(task_exc)),
            self._main_loop,
        )
        future.add_done_callback(
            lambda completed: self._log_event_result(integration_id, completed)
        )

    async def _handle_runner_failure(self, integration_id: str, message: str) -> None:
        async with self._sync_lock:
            self._mark_integration_error(integration_id, f"飞书长连接已断开：{message}")

    async def _handle_event_payload(
        self,
        integration_id: str,
        payload_text: str,
    ) -> None:
        request = ChannelRequest(
            method="LONG_CONNECTION",
            path=f"/api/integrations/feishu/{integration_id}/long-connection",
            headers={
                "content-type": "application/json",
                "x-feishu-connection-mode": "long_connection",
            },
            body=payload_text.encode("utf-8"),
            received_at=utc_now_iso(),
            remote_addr="feishu-long-connection",
        )
        try:
            result = await self._integration_gateway.handle_channel_request(
                integration_id,
                request,
                expected_kind="feishu",
            )
        except IntegrationGatewayError as exc:
            logger.warning(
                "飞书长连接网关处理失败: integration_id=%s detail=%s",
                integration_id,
                exc.detail,
            )
            return

        try:
            await self._integration_run_bridge.bridge_gateway_result(result)
        except IntegrationRunBridgeError as exc:
            logger.warning(
                "飞书长连接桥接失败: integration_id=%s detail=%s",
                integration_id,
                exc.detail,
            )

    def _resolve_credentials(self, integration_id: str) -> dict[str, str]:
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

