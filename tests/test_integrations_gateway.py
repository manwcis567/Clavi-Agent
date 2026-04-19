import asyncio
import json
from pathlib import Path
import time

import httpx
import pytest
from fastapi.testclient import TestClient

import clavi_agent.server as server_module
from clavi_agent.config import (
    AgentConfig,
    Config,
    FeatureFlagsConfig,
    LLMConfig,
    RetryConfig,
    ToolsConfig,
)
from clavi_agent.integration_models import (
    ConversationBindingRecord,
    DeliveryAttemptRecord,
    InboundEventRecord,
    IntegrationConfigRecord,
    IntegrationCredentialRecord,
    OutboundDeliveryRecord,
)
from clavi_agent.integration_store import (
    ConversationBindingStore,
    DeliveryStore,
    InboundEventStore,
    IntegrationStore,
)
from clavi_agent.integrations import ChannelRequest, IntegrationGateway, IntegrationRunBridge, IntegrationRunBridgeError
from clavi_agent.integrations.feishu_long_connection import FeishuLongConnectionService
from clavi_agent.integrations.wechat_long_poll import WeChatLongPollService
from clavi_agent.server import create_app
from clavi_agent.session import SessionManager


def build_config(tmp_path: Path) -> Config:
    return Config(
        llm=LLMConfig(
            api_key="test-key",
            api_base="https://example.com",
            model="test-model",
            provider="openai",
            retry=RetryConfig(enabled=False),
        ),
        agent=AgentConfig(
            max_steps=5,
            max_concurrent_runs=2,
            workspace_dir=str(tmp_path / "workspace"),
            system_prompt_path="system_prompt.md",
            log_dir=str(tmp_path / "logs"),
            session_store_path=str(tmp_path / "sessions.db"),
            agent_store_path=str(tmp_path / "agents.db"),
        ),
        tools=ToolsConfig(
            enable_file_tools=False,
            enable_bash=False,
            enable_note=False,
            enable_skills=False,
            enable_mcp=False,
        ),
        feature_flags=FeatureFlagsConfig(
            enable_durable_runs=True,
            enable_run_trace=True,
            enable_approval_flow=True,
        ),
    )


def session_db_path(config: Config) -> Path:
    return Path(config.agent.session_store_path).resolve()


def create_mock_integration(
    config: Config,
    *,
    integration_id: str = "mock-integration",
    account_id: str = "root",
    integration_config: dict | None = None,
) -> IntegrationConfigRecord:
    record = IntegrationConfigRecord(
        id=integration_id,
        account_id=account_id,
        name=integration_id,
        kind="mock",
        status="active",
        webhook_path=f"/api/integrations/mock/{integration_id}/webhook",
        config={"verify_token": "mock-token", **(integration_config or {})},
        created_at="2026-04-13T12:00:00+00:00",
        updated_at="2026-04-13T12:00:00+00:00",
    )
    IntegrationStore(session_db_path(config)).create_integration(record)
    return record


def create_feishu_integration(
    config: Config,
    *,
    integration_id: str = "feishu-integration",
    integration_config: dict | None = None,
) -> IntegrationConfigRecord:
    record = IntegrationConfigRecord(
        id=integration_id,
        name=integration_id,
        kind="feishu",
        status="active",
        webhook_path=f"/api/integrations/feishu/{integration_id}/webhook",
        config={
            "app_id": "cli_test",
            "verification_token": "verify-token",
            **(integration_config or {}),
        },
        created_at="2026-04-13T12:00:00+00:00",
        updated_at="2026-04-13T12:00:00+00:00",
    )
    IntegrationStore(session_db_path(config)).create_integration(record)
    return record


def create_wechat_integration(
    config: Config,
    *,
    integration_id: str = "wechat-integration",
    sync_cursor: str = "",
) -> IntegrationConfigRecord:
    now = "2026-04-13T12:00:00+00:00"
    metadata = {
        "wechat_setup": {
            "integration_id": integration_id,
            "state": "succeeded",
            "message": "Connected",
            "error": "",
            "updated_at": now,
        }
    }
    if sync_cursor:
        metadata["wechat_runtime"] = {
            "sync_cursor": sync_cursor,
            "updated_at": now,
        }

    record = IntegrationConfigRecord(
        id=integration_id,
        name=integration_id,
        kind="wechat",
        status="active",
        webhook_path=f"/api/integrations/wechat/{integration_id}/webhook",
        config={"default_agent_id": "system-default-agent"},
        metadata=metadata,
        created_at=now,
        updated_at=now,
    )
    store = IntegrationStore(session_db_path(config))
    store.create_integration(record)

    for credential_key, secret_value in {
        "bot_token": "wechat-bot-token",
        "ilink_bot_id": "wx-bot-001",
        "base_url": "https://ilinkai.weixin.qq.com",
        "ilink_user_id": "wx-user-001",
    }.items():
        store.create_credential(
            IntegrationCredentialRecord(
                id=f"{integration_id}:{credential_key}",
                integration_id=integration_id,
                credential_key=credential_key,
                storage_kind="local_encrypted",
                secret_ciphertext=secret_value,
                masked_value="***",
                created_at=now,
                updated_at=now,
            )
        )
    return record


def build_mock_request(
    *,
    text: str = "hello",
    integration_token: str = "mock-token",
    event_id: str = "mock-evt-1",
    message_id: str = "mock-msg-1",
    chat_id: str = "chat-1",
    thread_id: str = "",
    sender_id: str = "user-1",
    sender_name: str = "测试用户",
    dedup_key: str | None = None,
    attachments: list[dict] | None = None,
    payload_overrides: dict | None = None,
) -> ChannelRequest:
    payload = {
        "event_id": event_id,
        "event_type": "message",
        "tenant_id": "tenant-1",
        "token": integration_token,
        "message": {
            "message_id": message_id,
            "chat_id": chat_id,
            "thread_id": thread_id,
            "sender_id": sender_id,
            "sender_name": sender_name,
            "message_type": "text",
            "text": text,
            "attachments": attachments or [],
        },
    }
    if dedup_key is not None:
        payload["dedup_key"] = dedup_key
    if payload_overrides:
        payload.update(payload_overrides)
    return ChannelRequest(
        headers={"content-type": "application/json"},
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        received_at="2026-04-13T12:00:00+00:00",
    )


def build_feishu_request(
    *,
    text: str = "hello",
    message_id: str = "om_message_1",
    chat_id: str = "oc_chat_1",
    thread_id: str = "",
    parent_id: str = "",
    root_id: str = "",
) -> ChannelRequest:
    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-feishu-1",
            "event_type": "im.message.receive_v1",
            "token": "verify-token",
            "app_id": "cli_test",
            "tenant_key": "tenant-key-1",
        },
        "event": {
            "sender": {
                "sender_id": {"open_id": "ou_user_1"},
                "sender_type": "user",
                "tenant_key": "tenant-key-1",
            },
            "message": {
                "message_id": message_id,
                "chat_id": chat_id,
                "thread_id": thread_id,
                "parent_id": parent_id,
                "root_id": root_id,
                "chat_type": "group" if thread_id else "p2p",
                "message_type": "text",
                "content": json.dumps({"text": text}, ensure_ascii=False),
                "mentions": [],
            },
        },
    }
    return ChannelRequest(
        headers={"content-type": "application/json"},
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        received_at="2026-04-13T12:00:00+00:00",
    )


def test_gateway_bind_command_creates_binding(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(text="/bind system-default-agent", thread_id="thread-1"),
        )
    )

    binding_store = ConversationBindingStore(session_db_path(config))
    binding = binding_store.find_binding(
        integration_id="mock-integration",
        tenant_id="tenant-1",
        chat_id="chat-1",
        thread_id="thread-1",
        binding_scope="thread",
        enabled=True,
    )

    assert result.command_result is not None
    assert result.command_result.command == "bind"
    assert result.event is not None
    assert result.event.normalized_status == "command_handled"
    assert binding is not None
    assert binding.agent_id == "system-default-agent"
    assert manager.get_session_info(binding.session_id) is not None

    asyncio.run(manager.cleanup())


def test_gateway_bind_command_rejects_foreign_account_agent(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    account_a = manager._account_store.create_account(
        username="gateway-alice",
        password="secret-a",
    )
    account_b = manager._account_store.create_account(
        username="gateway-bob",
        password="secret-b",
    )
    manager._agent_store.create_agent(
        agent_id="foreign-agent",
        name="Foreign Agent",
        system_prompt="You are foreign.",
        account_id=account_b["id"],
    )
    IntegrationStore(session_db_path(config)).create_integration(
        IntegrationConfigRecord(
            id="mock-account-a",
            account_id=account_a["id"],
            name="mock-account-a",
            kind="mock",
            status="active",
            webhook_path="/api/integrations/mock/mock-account-a/webhook",
            config={"verify_token": "mock-token"},
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
        )
    )

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-account-a",
            build_mock_request(text="/bind foreign-agent", thread_id="thread-1"),
        )
    )

    binding_store = ConversationBindingStore(session_db_path(config))
    bindings = binding_store.list_bindings(integration_id="mock-account-a")

    assert result.command_result is not None
    assert result.command_result.command == "bind"
    assert result.command_result.response_text == "未找到 Agent：foreign-agent"
    assert bindings == []

    asyncio.run(manager.cleanup())


def test_gateway_routes_feishu_message_receive_event_and_preserves_thread_metadata(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-feishu-1",
            integration_id="feishu-integration",
            tenant_id="tenant-key-1",
            chat_id="oc_chat_1",
            thread_id="omt_thread_1",
            binding_scope="thread",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "feishu-integration",
            build_feishu_request(
                text="请处理线程消息",
                thread_id="omt_thread_1",
                parent_id="om_parent_1",
                root_id="om_root_1",
            ),
        )
    )
    assert result.should_route is True
    assert result.event is not None
    assert result.event.normalized_status == "routed"
    assert result.msg_context is not None
    assert result.msg_context.metadata["provider_event_type"] == "im.message.receive_v1"

    bridge_result = asyncio.run(IntegrationRunBridge(manager).bridge_gateway_result(result))
    assert bridge_result is not None
    run = manager._run_store.get_run(bridge_result.run_id)
    assert run is not None
    assert run.run_metadata["provider_root_message_id"] == "om_root_1"
    assert run.run_metadata["provider_parent_message_id"] == "om_parent_1"
    assert run.run_metadata["channel_context"]["root_id"] == "om_root_1"
    assert run.run_metadata["channel_context"]["parent_id"] == "om_parent_1"

    asyncio.run(manager.cleanup())


def test_wechat_long_poll_loads_and_persists_sync_cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = build_config(tmp_path)
    create_wechat_integration(config, sync_cursor="cursor-1")
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    service = WeChatLongPollService(
        manager,
        integration_gateway=IntegrationGateway(manager),
        integration_run_bridge=IntegrationRunBridge(manager),
    )
    service._ensure_store()

    calls: list[str] = []

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get_updates(self, cursor: str = "") -> dict:
            calls.append(cursor)
            if len(calls) == 1:
                return {
                    "ret": 0,
                    "errcode": 0,
                    "errmsg": "",
                    "msgs": [],
                    "get_updates_buf": "cursor-2",
                }
            raise asyncio.CancelledError()

    monkeypatch.setattr("clavi_agent.integrations.wechat_long_poll.WeChatILinkClient", StubClient)

    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                service._run_monitor(
                    "wechat-integration",
                    ("wechat-bot-token", "wx-bot-001", "https://ilinkai.weixin.qq.com", "wx-user-001"),
                )
            )

        stored = IntegrationStore(session_db_path(config)).get_integration("wechat-integration")
        assert stored is not None
        assert stored.metadata["wechat_runtime"]["sync_cursor"] == "cursor-2"
        assert calls == ["cursor-1", "cursor-2"]
    finally:
        asyncio.run(manager.cleanup())


def test_wechat_long_poll_recovers_once_by_resetting_persisted_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = build_config(tmp_path)
    create_wechat_integration(config, sync_cursor="cursor-1")
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    service = WeChatLongPollService(
        manager,
        integration_gateway=IntegrationGateway(manager),
        integration_run_bridge=IntegrationRunBridge(manager),
    )
    service._ensure_store()

    calls: list[str] = []

    class StubClient:
        def __init__(self, *args, **kwargs):
            pass

        async def get_updates(self, cursor: str = "") -> dict:
            calls.append(cursor)
            if len(calls) == 1:
                return {
                    "ret": 0,
                    "errcode": -14,
                    "errmsg": "session expired",
                    "msgs": [],
                    "get_updates_buf": "",
                }
            if len(calls) == 2:
                return {
                    "ret": 0,
                    "errcode": 0,
                    "errmsg": "",
                    "msgs": [],
                    "get_updates_buf": "cursor-2",
                }
            raise asyncio.CancelledError()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("clavi_agent.integrations.wechat_long_poll.WeChatILinkClient", StubClient)
    monkeypatch.setattr("clavi_agent.integrations.wechat_long_poll.asyncio.sleep", no_sleep)

    try:
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                service._run_monitor(
                    "wechat-integration",
                    ("wechat-bot-token", "wx-bot-001", "https://ilinkai.weixin.qq.com", "wx-user-001"),
                )
            )

        stored = IntegrationStore(session_db_path(config)).get_integration("wechat-integration")
        assert stored is not None
        assert stored.status == "active"
        assert stored.metadata["wechat_setup"]["state"] == "succeeded"
        assert stored.metadata["wechat_runtime"]["sync_cursor"] == "cursor-2"
        assert calls == ["cursor-1", "", "cursor-2"]
    finally:
        asyncio.run(manager.cleanup())


def test_feishu_long_connection_service_reuses_gateway_pipeline(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(
        config,
        integration_config={"connection_mode": "long_connection"},
    )
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    assert manager._run_manager is not None
    manager._run_manager._schedule_dispatch = lambda: None

    session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-feishu-long-connection",
            integration_id="feishu-integration",
            tenant_id="tenant-key-1",
            chat_id="oc_chat_1",
            thread_id="omt_thread_1",
            binding_scope="thread",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    service = FeishuLongConnectionService(
        manager,
        integration_gateway=IntegrationGateway(manager),
        integration_run_bridge=IntegrationRunBridge(manager),
    )
    payload_text = build_feishu_request(
        text="通过长连接接收飞书消息",
        thread_id="omt_thread_1",
    ).body.decode("utf-8")
    payload = json.loads(payload_text)
    payload["header"].pop("token", None)
    payload_text = json.dumps(payload, ensure_ascii=False)

    asyncio.run(service._handle_event_payload("feishu-integration", payload_text))

    runs = manager.list_runs(session_id=session_id)
    assert len(runs) == 1
    run = runs[0]
    assert run["status"] == "queued"
    assert run["run_metadata"]["source_kind"] == "integration"
    assert run["run_metadata"]["provider_message_id"] == "om_message_1"
    assert run["run_metadata"]["channel_context"]["thread_id"] == "omt_thread_1"

    events = InboundEventStore(session_db_path(config)).list_events(
        integration_id="feishu-integration",
        limit=10,
    )
    bridged_event = next(event for event in events if event.provider_message_id == "om_message_1")
    assert bridged_event.normalized_status == "bridged"
    assert bridged_event.metadata["run_id"] == run["id"]

    asyncio.run(manager.cleanup())


def test_feishu_long_connection_service_registers_message_read_handler(tmp_path: Path):
    config = build_config(tmp_path)
    record = create_feishu_integration(
        config,
        integration_config={"connection_mode": "long_connection"},
    )
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    service = FeishuLongConnectionService(
        manager,
        integration_gateway=IntegrationGateway(manager),
        integration_run_bridge=IntegrationRunBridge(manager),
    )

    builder_args: dict[str, str] = {}
    registrations: list[tuple[str, object]] = []
    client_kwargs: dict[str, object] = {}

    class StubBuilder:
        def register_p2_im_message_receive_v1(self, handler):
            registrations.append(("im.message.receive_v1", handler))
            return self

        def register_p2_im_message_message_read_v1(self, handler):
            registrations.append(("im.message.message_read_v1", handler))
            return self

        def build(self):
            return {"registrations": registrations}

    class StubEventDispatcherHandler:
        @staticmethod
        def builder(encrypt_key, verification_token):
            builder_args["encrypt_key"] = encrypt_key
            builder_args["verification_token"] = verification_token
            return StubBuilder()

    class StubClient:
        def __init__(self, app_id, app_secret, **kwargs):
            client_kwargs["app_id"] = app_id
            client_kwargs["app_secret"] = app_secret
            client_kwargs.update(kwargs)

    class StubWs:
        Client = StubClient

    class StubLogLevel:
        INFO = "INFO"

    class StubLarkSdk:
        EventDispatcherHandler = StubEventDispatcherHandler
        LogLevel = StubLogLevel
        ws = StubWs

    async def stub_run_client(integration_id: str, client, stop_event: asyncio.Event) -> None:
        try:
            await stop_event.wait()
        finally:
            service._handles.pop(integration_id, None)

    service._lark_sdk = StubLarkSdk()
    service._run_client = stub_run_client  # type: ignore[method-assign]

    spec = ("cli_test", "app-secret", "verify-token", "encrypt-key", "")
    asyncio.run(service._start_client_on_worker(record, spec))

    assert builder_args == {
        "encrypt_key": "encrypt-key",
        "verification_token": "verify-token",
    }
    assert [event_type for event_type, _ in registrations] == [
        "im.message.receive_v1",
        "im.message.message_read_v1",
    ]
    assert client_kwargs["app_id"] == "cli_test"
    assert client_kwargs["app_secret"] == "app-secret"

    message_read_handler = dict(registrations)["im.message.message_read_v1"]
    message_read_handler(object())

    asyncio.run(service._stop_client_on_worker(record.id))
    asyncio.run(manager.cleanup())


def test_app_startup_tolerates_invalid_feishu_long_connection_config(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(
        config,
        integration_config={
            "app_id": "",
            "connection_mode": "long_connection",
        },
    )
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.get("/api/features")
        assert response.status_code == 200

    record = IntegrationStore(session_db_path(config)).get_integration("feishu-integration")
    assert record is not None
    assert record.status == "error"
    assert "app_id" in record.last_error


def test_gateway_deduplicates_by_provider_message_id(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    event_store = InboundEventStore(session_db_path(config))
    event_store.create_event(
        InboundEventRecord(
            id="existing-event",
            integration_id="mock-integration",
            provider_event_id="",
            provider_message_id="dup-msg",
            event_type="message",
            received_at="2026-04-13T11:59:00+00:00",
            signature_valid=True,
            dedup_key="existing-dedup",
            raw_headers={},
            raw_payload={"message": {"message_id": "dup-msg"}},
            normalized_status="verified",
        )
    )

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(
                event_id="",
                message_id="dup-msg",
                dedup_key="new-dedup",
            ),
        )
    )

    updated = event_store.get_event("existing-event")
    assert result.duplicate_of_event_id == "existing-event"
    assert updated is not None
    assert updated.metadata["duplicate_count"] == 1
    assert len(event_store.list_events(integration_id="mock-integration")) == 1

    asyncio.run(manager.cleanup())


def test_gateway_ignores_self_messages(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(
        config,
        integration_config={"ignore_sender_ids": ["bot-1"]},
    )
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(sender_id="bot-1"),
        )
    )

    assert result.event is not None
    assert result.event.normalized_status == "command_handled"
    assert result.event.metadata["handled_by"] == "ignore_self_message"
    assert result.should_route is False

    asyncio.run(manager.cleanup())


def test_gateway_marks_unbound_message_as_failed(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(),
        )
    )

    assert result.event is not None
    assert result.event.normalized_status == "failed"
    assert "当前会话尚未绑定 Agent" in result.event.normalized_error
    assert result.should_route is False

    asyncio.run(manager.cleanup())


def test_gateway_bridges_mock_attachments_into_session_uploads(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-1",
            integration_id="mock-integration",
            tenant_id="tenant-1",
            chat_id="chat-1",
            thread_id="",
            binding_scope="chat",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://download.example.com/demo.txt"
        return httpx.Response(
            200,
            content=b"demo attachment",
            headers={
                "content-type": "text/plain; charset=utf-8",
                "content-disposition": 'attachment; filename="demo.txt"',
            },
        )

    gateway = IntegrationGateway(manager, download_transport=httpx.MockTransport(handler))
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(
                attachments=[
                    {
                        "kind": "file",
                        "provider_file_id": "file-1",
                        "name": "demo.txt",
                        "mime_type": "text/plain",
                        "download_url": "https://download.example.com/demo.txt",
                    }
                ],
            ),
        )
    )

    assert result.event is not None
    assert result.event.normalized_status == "bridged"
    assert len(result.created_uploads) == 1
    assert result.msg_context is not None
    assert result.msg_context.metadata["attachment_upload_ids"] == [result.created_uploads[0].id]
    assert manager.get_upload_info(result.created_uploads[0].id) is not None

    asyncio.run(manager.cleanup())


def test_run_bridge_creates_queued_run_with_channel_metadata(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
    assert manager._run_manager is not None
    manager._run_manager._schedule_dispatch = lambda: None

    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-run-1",
            integration_id="mock-integration",
            tenant_id="tenant-1",
            chat_id="chat-1",
            thread_id="",
            binding_scope="chat",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(text="请总结这条来自渠道的消息"),
        )
    )

    bridge = IntegrationRunBridge(manager)
    bridge_result = asyncio.run(bridge.bridge_gateway_result(result))

    assert bridge_result is not None
    assert bridge_result.session_id == session_id
    assert bridge_result.goal == "请总结这条来自渠道的消息"
    assert bridge_result.user_message_content == "请总结这条来自渠道的消息"

    run = manager._run_store.get_run(bridge_result.run_id)
    assert run is not None
    assert run.status == "queued"
    assert run.run_metadata["source_kind"] == "integration"
    assert run.run_metadata["integration_id"] == "mock-integration"
    assert run.run_metadata["inbound_event_id"] == result.event.id
    assert run.run_metadata["provider_message_id"] == "mock-msg-1"
    assert run.run_metadata["channel_context"]["sender_name"] == "测试用户"
    assert run.run_metadata["channel_context"]["chat_id"] == "chat-1"
    assert run.run_metadata["user_message_content"] == "请总结这条来自渠道的消息"

    stored_event = InboundEventStore(session_db_path(config)).get_event(result.event.id)
    assert stored_event is not None
    assert stored_event.normalized_status == "bridged"
    assert stored_event.metadata["run_id"] == bridge_result.run_id
    assert stored_event.metadata["bridge_status"] == "queued"

    asyncio.run(manager.cleanup())


def test_run_bridge_includes_attachment_uploads_in_user_message_content(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
    assert manager._run_manager is not None
    manager._run_manager._schedule_dispatch = lambda: None

    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-run-attachment",
            integration_id="mock-integration",
            tenant_id="tenant-1",
            chat_id="chat-1",
            thread_id="",
            binding_scope="chat",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://download.example.com/spec.txt"
        return httpx.Response(
            200,
            content=b"attachment body",
            headers={
                "content-type": "text/plain; charset=utf-8",
                "content-disposition": 'attachment; filename="spec.txt"',
            },
        )

    gateway = IntegrationGateway(manager, download_transport=httpx.MockTransport(handler))
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(
                text="请结合附件处理",
                attachments=[
                    {
                        "kind": "file",
                        "provider_file_id": "file-spec",
                        "name": "spec.txt",
                        "mime_type": "text/plain",
                        "download_url": "https://download.example.com/spec.txt",
                    }
                ],
            ),
        )
    )

    bridge = IntegrationRunBridge(manager)
    bridge_result = asyncio.run(bridge.bridge_gateway_result(result))

    assert isinstance(bridge_result.user_message_content, list)
    assert bridge_result.user_message_content[0]["type"] == "text"
    assert bridge_result.user_message_content[1]["type"] == "uploaded_file"
    assert bridge_result.user_message_content[1]["upload_id"] == result.created_uploads[0].id

    run = manager._run_store.get_run(bridge_result.run_id)
    assert run is not None
    assert run.run_metadata["channel_context"]["attachment_upload_ids"] == [result.created_uploads[0].id]

    asyncio.run(manager.cleanup())


def test_run_bridge_marks_event_failed_when_message_payload_is_empty(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-run-empty",
            integration_id="mock-integration",
            tenant_id="tenant-1",
            chat_id="chat-1",
            thread_id="",
            binding_scope="chat",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(text=""),
        )
    )

    bridge = IntegrationRunBridge(manager)
    with pytest.raises(IntegrationRunBridgeError) as exc_info:
        asyncio.run(bridge.bridge_gateway_result(result))

    assert "渠道执行桥接失败" in str(exc_info.value)
    stored_event = InboundEventStore(session_db_path(config)).get_event(result.event.id)
    assert stored_event is not None
    assert stored_event.normalized_status == "failed"
    assert "Chat message must include text or attachments" in stored_event.normalized_error
    assert stored_event.metadata["bridge_status"] == "failed"

    asyncio.run(manager.cleanup())


def test_integration_webhook_api_handles_ack_and_rejection(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        ok_response = client.post(
            "/api/integrations/mock/mock-integration/webhook",
            json={
                "type": "url_verification",
                "challenge": "challenge-1",
                "token": "mock-token",
            },
        )
        assert ok_response.status_code == 200
        assert ok_response.json() == {"challenge": "challenge-1"}

        rejected_response = client.post(
            "/api/integrations/mock/mock-integration/webhook",
            json={
                "type": "message",
                "token": "bad-token",
                "message": {"message_id": "msg-2", "chat_id": "chat-1"},
            },
        )
        assert rejected_response.status_code == 401

    event_store = InboundEventStore(session_db_path(config))
    events = event_store.list_events(integration_id="mock-integration", limit=10)
    statuses = {event.normalized_status for event in events}
    assert "command_handled" in statuses
    assert "rejected" in statuses


def test_integration_webhook_api_creates_run_for_routed_message(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())
    assert manager._run_manager is not None
    manager._run_manager._schedule_dispatch = lambda: None

    session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-webhook-run",
            integration_id="mock-integration",
            tenant_id="tenant-1",
            chat_id="chat-1",
            thread_id="",
            binding_scope="chat",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    app = create_app(manager)
    payload = json.loads(
        build_mock_request(text="通过 webhook 创建 durable run").body.decode("utf-8")
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/mock/mock-integration/webhook",
            json=payload,
        )
        assert response.status_code == 200

    runs = manager.list_runs(session_id=session_id)
    assert len(runs) == 1
    run = runs[0]
    assert run["status"] == "queued"
    assert run["run_metadata"]["source_kind"] == "integration"
    assert run["run_metadata"]["provider_message_id"] == "mock-msg-1"
    assert run["run_metadata"]["channel_context"]["sender_id"] == "user-1"

    events = InboundEventStore(session_db_path(config)).list_events(
        integration_id="mock-integration",
        limit=10,
    )
    bridged_event = next(event for event in events if event.provider_message_id == "mock-msg-1")
    assert bridged_event.normalized_status == "bridged"
    assert bridged_event.metadata["run_id"] == run["id"]

    asyncio.run(manager.cleanup())


def test_integration_webhook_api_infers_account_context_from_integration_id(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())
    assert manager._account_store is not None
    assert manager._run_store is not None
    assert manager._run_manager is not None
    manager._run_manager._schedule_dispatch = lambda: None

    owned_account = manager._account_store.create_account(
        username="webhook-owner",
        password="WebhookPass123!",
        display_name="Webhook Owner",
    )
    manager.save_account_api_config(
        owned_account["id"],
        name="webhook-owner-default",
        api_key="owned-account-key",
        api_base="https://example.com",
        model="test-model",
        provider="openai",
        reasoning_enabled=False,
        activate=True,
    )
    create_mock_integration(
        config,
        integration_id="mock-owned-integration",
        account_id=owned_account["id"],
    )

    session_id = asyncio.run(
        manager.create_session(
            agent_id="system-default-agent",
            account_id=owned_account["id"],
        )
    )
    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-webhook-owned-account",
            integration_id="mock-owned-integration",
            account_id=owned_account["id"],
            tenant_id="tenant-1",
            chat_id="chat-1",
            thread_id="",
            binding_scope="chat",
            agent_id="system-default-agent",
            session_id=session_id,
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    app = create_app(manager)
    payload = json.loads(
        build_mock_request(text="通过 integration_id 继承账号上下文").body.decode("utf-8")
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/integrations/mock/mock-owned-integration/webhook",
            json=payload,
        )
        assert response.status_code == 200

    runs = manager._run_store.list_runs(
        session_id=session_id,
        account_id=owned_account["id"],
    )
    assert len(runs) == 1
    assert runs[0].account_id == owned_account["id"]

    events = InboundEventStore(session_db_path(config)).list_events(
        integration_id="mock-owned-integration",
        account_id=owned_account["id"],
        limit=10,
    )
    assert len(events) == 1
    assert events[0].account_id == owned_account["id"]
    assert events[0].normalized_status == "bridged"
    assert events[0].metadata["run_id"] == runs[0].id

    asyncio.run(manager.cleanup())


def test_feishu_long_connection_webhook_requests_are_ignored_by_http_ingress(tmp_path: Path):
    class StubFeishuLongConnectionManager:
        async def start(self):
            return None

        async def shutdown(self):
            return None

        async def sync_integration(self, integration_id: str):
            return None

    config = build_config(tmp_path)
    create_feishu_integration(
        config,
        integration_config={"connection_mode": "long_connection"},
    )
    manager = SessionManager(config=config)
    app = create_app(manager)
    app.state.feishu_long_connection_manager = StubFeishuLongConnectionManager()

    with TestClient(app) as client:
        verify_response = client.post(
            "/api/integrations/feishu/feishu-integration/webhook",
            json={
                "type": "url_verification",
                "challenge": "challenge-1",
                "token": "bad-token",
            },
        )
        assert verify_response.status_code == 200
        assert verify_response.json() == {"challenge": "challenge-1"}
        assert verify_response.headers["x-clavi-agent-integration-mode"] == "long_connection"

        ignored_response = client.post(
            "/api/integrations/feishu/feishu-integration/webhook",
            json={
                "schema": "2.0",
                "header": {
                    "event_id": "evt-http-probe",
                    "event_type": "im.message.receive_v1",
                    "token": "bad-token",
                },
                "event": {
                    "message": {
                        "message_id": "om_http_probe",
                        "chat_id": "oc_chat_1",
                        "message_type": "text",
                        "content": json.dumps({"text": "hello"}, ensure_ascii=False),
                    }
                },
            },
        )
        assert ignored_response.status_code == 200
        assert ignored_response.json() == {
            "ok": True,
            "ignored": True,
            "connection_mode": "long_connection",
        }

    events = InboundEventStore(session_db_path(config)).list_events(
        integration_id="feishu-integration",
        limit=10,
    )
    assert events == []


def test_binding_management_api_crud_flow(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations/mock-integration/bindings",
            json={
                "tenant_id": "tenant-1",
                "chat_id": "chat-1",
                "binding_scope": "chat",
                "agent_id": "system-default-agent",
            },
        )
        assert create_response.status_code == 201
        created_binding = create_response.json()
        binding_id = created_binding["id"]
        assert created_binding["enabled"] is True
        assert created_binding["session_key"].endswith(":chat:system-default-agent")

        list_response = client.get("/api/integrations/mock-integration/bindings")
        assert list_response.status_code == 200
        bindings = list_response.json()
        assert len(bindings) == 1
        assert bindings[0]["id"] == binding_id

        patch_response = client.patch(
            f"/api/bindings/{binding_id}",
            json={"enabled": False},
        )
        assert patch_response.status_code == 200
        assert patch_response.json()["enabled"] is False

        delete_response = client.delete(f"/api/bindings/{binding_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["enabled"] is False


def test_integration_management_api_crud_and_verify(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "本地联调渠道",
                "display_name": "Mock 联调入口",
                "kind": "mock",
                "config": {
                    "verify_token": "mock-token",
                    "default_session_strategy": "reuse",
                },
            },
        )
        assert create_response.status_code == 201
        created = create_response.json()
        integration_id = created["id"]
        assert created["status"] == "disabled"
        assert created["webhook_path"] == f"/api/integrations/mock/{integration_id}/webhook"

        list_response = client.get("/api/integrations")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert any(item["id"] == integration_id for item in listed)

        patch_response = client.patch(
            f"/api/integrations/{integration_id}",
            json={
                "display_name": "Mock 联调入口（已更新）",
                "config": {
                    "verify_token": "mock-token",
                    "default_session_strategy": "reuse",
                    "require_signing_secret": False,
                },
                "credentials": [
                    {
                        "credential_key": "signing_secret",
                        "storage_kind": "env",
                        "secret_ref": "MOCK_SIGNING_SECRET",
                    }
                ],
            },
        )
        assert patch_response.status_code == 200
        patched = patch_response.json()
        assert patched["display_name"] == "Mock 联调入口（已更新）"
        assert patched["credentials"][0]["credential_key"] == "signing_secret"
        assert patched["credentials"][0]["secret_ref"] == "MOCK_SIGNING_SECRET"

        verify_response = client.post(f"/api/integrations/{integration_id}/verify")
        assert verify_response.status_code == 200
        verify_payload = verify_response.json()
        assert verify_payload["success"] is True
        assert verify_payload["integration"]["last_verified_at"]

        disable_response = client.post(f"/api/integrations/{integration_id}/disable")
        assert disable_response.status_code == 200
        assert disable_response.json()["status"] == "disabled"

        enable_response = client.post(f"/api/integrations/{integration_id}/enable")
        assert enable_response.status_code == 200
        assert enable_response.json()["status"] == "active"

        delete_response = client.delete(f"/api/integrations/{integration_id}")
        assert delete_response.status_code == 200
        assert delete_response.json()["deleted"] is True
        assert delete_response.json()["status"] == "disabled"

        active_list_response = client.get("/api/integrations")
        assert active_list_response.status_code == 200
        assert all(item["id"] != integration_id for item in active_list_response.json())

        deleted_list_response = client.get("/api/integrations?include_deleted=true")
        assert deleted_list_response.status_code == 200
        assert any(
            item["id"] == integration_id and item["deleted"] is True
            for item in deleted_list_response.json()
        )


def _deprecated_test_wechat_setup_api_runs_official_installer_and_persists_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    async def fake_run_wechat_installer(*, on_output=None, cwd=None):
        chunks = [
            "检测到 OpenClaw 版本: 2026.3.22\n",
            "正在安装插件 @tencent-weixin/openclaw-weixin@latest...\n",
            "二维码如下，请使用微信扫码：\n",
            "████████████\n██  ██  ████\n████  ██████\n██  ██  ████\n████████████\n████  ██  ██\n██  ████████\n████████████\n",
        ]
        for chunk in chunks:
            if on_output is not None:
                result = on_output(chunk)
                if asyncio.iscoroutine(result):
                    await result
        return 0, "".join(chunks)

    monkeypatch.setattr(server_module, "_run_wechat_installer", fake_run_wechat_installer)

    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "wechat-main",
                "display_name": "微信主账号",
                "kind": "wechat",
                "config": {
                    "default_agent_id": "system-default-agent",
                },
            },
        )
        assert create_response.status_code == 201
        integration_id = create_response.json()["id"]

        idle_response = client.get(f"/api/integrations/{integration_id}/wechat/setup")
        assert idle_response.status_code == 200
        assert idle_response.json()["state"] == "idle"

        start_response = client.post(f"/api/integrations/{integration_id}/wechat/setup")
        assert start_response.status_code == 200

        status_payload = None
        for _ in range(20):
            status_payload = client.get(f"/api/integrations/{integration_id}/wechat/setup").json()
            if status_payload["state"] == "succeeded":
                break
            time.sleep(0.05)

        assert status_payload is not None
        assert status_payload["state"] == "succeeded"
        assert status_payload["openclaw_version"] == "2026.3.22"
        assert status_payload["plugin_spec"] == "@tencent-weixin/openclaw-weixin@latest"
        assert "████" in status_payload["qr_text"]

        verify_response = client.post(f"/api/integrations/{integration_id}/verify")
        assert verify_response.status_code == 200
        verify_payload = verify_response.json()
        assert verify_payload["success"] is True

        list_response = client.get("/api/integrations")
        assert list_response.status_code == 200
        wechat_record = next(item for item in list_response.json() if item["id"] == integration_id)
        assert wechat_record["setup_status"]["state"] == "succeeded"


def test_wechat_setup_api_runs_official_installer_and_persists_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    class StubWeChatLongPollManager:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.synced: list[str] = []

        async def start(self):
            self.started = True

        async def shutdown(self):
            self.stopped = True

        async def sync_integration(self, integration_id: str):
            self.synced.append(integration_id)

    async def fake_fetch_login_qr_code():
        return type(
            "QRCodePayload",
            (),
            {
                "qrcode": "wechat-qr-ticket",
                "qr_content": "weixin://native-ilink-qr",
            },
        )()

    async def fake_poll_login_status(qrcode, *, transport=None, on_status=None):
        assert qrcode == "wechat-qr-ticket"
        if on_status is not None:
            await on_status("wait")
            await on_status("scaned")
            await on_status("confirmed")
        return server_module.WeChatILinkCredentials(
            bot_token="wechat-bot-token",
            ilink_bot_id="wx-bot-001",
            base_url="https://ilinkai.weixin.qq.com",
            ilink_user_id="wx-user-001",
        )

    monkeypatch.setattr(server_module, "fetch_login_qr_code", fake_fetch_login_qr_code)
    monkeypatch.setattr(server_module, "poll_login_status", fake_poll_login_status)

    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)
    stub_manager = StubWeChatLongPollManager()
    app.state.wechat_long_poll_manager = stub_manager

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "wechat-main",
                "display_name": "WeChat Main",
                "kind": "wechat",
                "config": {
                    "default_agent_id": "system-default-agent",
                },
            },
        )
        assert create_response.status_code == 201
        integration_id = create_response.json()["id"]

        idle_response = client.get(f"/api/integrations/{integration_id}/wechat/setup")
        assert idle_response.status_code == 200
        assert idle_response.json()["state"] == "idle"

        start_response = client.post(f"/api/integrations/{integration_id}/wechat/setup")
        assert start_response.status_code == 200

        status_payload = None
        for _ in range(20):
            status_payload = client.get(f"/api/integrations/{integration_id}/wechat/setup").json()
            if status_payload["state"] == "succeeded":
                break
            time.sleep(0.05)

        assert status_payload is not None
        assert status_payload["state"] == "succeeded"
        assert status_payload["qr_content"] == "weixin://native-ilink-qr"
        assert status_payload["qr_text"] == "weixin://native-ilink-qr"
        assert status_payload["ilink_bot_id"] == "wx-bot-001"
        assert status_payload["ilink_user_id"] == "wx-user-001"
        assert status_payload["base_url"] == "https://ilinkai.weixin.qq.com"
        assert stub_manager.started is True
        assert stub_manager.synced[-1] == integration_id

        verify_response = client.post(f"/api/integrations/{integration_id}/verify")
        assert verify_response.status_code == 200
        verify_payload = verify_response.json()
        assert verify_payload["success"] is True

        integration = IntegrationStore(session_db_path(config)).get_integration(integration_id)
        assert integration is not None
        assert integration.status == "active"

        stored_credentials = {
            item.credential_key: item
            for item in IntegrationStore(session_db_path(config)).list_credentials(integration_id)
        }
        assert stored_credentials["bot_token"].secret_ciphertext == "wechat-bot-token"
        assert stored_credentials["ilink_bot_id"].secret_ciphertext == "wx-bot-001"
        assert stored_credentials["base_url"].secret_ciphertext == "https://ilinkai.weixin.qq.com"
        assert stored_credentials["ilink_user_id"].secret_ciphertext == "wx-user-001"

        list_response = client.get("/api/integrations")
        assert list_response.status_code == 200
        wechat_record = next(item for item in list_response.json() if item["id"] == integration_id)
        assert wechat_record["setup_status"]["state"] == "succeeded"

    assert stub_manager.stopped is True


def test_integration_api_syncs_wechat_long_poll_manager(tmp_path: Path):
    class StubWeChatLongPollManager:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.synced: list[str] = []

        async def start(self):
            self.started = True

        async def shutdown(self):
            self.stopped = True

        async def sync_integration(self, integration_id: str):
            self.synced.append(integration_id)

    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)
    stub_manager = StubWeChatLongPollManager()
    app.state.wechat_long_poll_manager = stub_manager

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "wechat-native",
                "display_name": "WeChat Native",
                "kind": "wechat",
                "config": {
                    "default_agent_id": "system-default-agent",
                },
                "enabled": True,
            },
        )
        assert create_response.status_code == 201
        integration_id = create_response.json()["id"]
        assert stub_manager.started is True
        assert stub_manager.synced == [integration_id]

        disable_response = client.post(f"/api/integrations/{integration_id}/disable")
        assert disable_response.status_code == 200
        assert stub_manager.synced[-1] == integration_id

    assert stub_manager.stopped is True


def test_integration_api_syncs_feishu_long_connection_manager(tmp_path: Path):
    class StubFeishuLongConnectionManager:
        def __init__(self):
            self.started = False
            self.stopped = False
            self.synced: list[str] = []

        async def start(self):
            self.started = True

        async def shutdown(self):
            self.stopped = True

        async def sync_integration(self, integration_id: str):
            self.synced.append(integration_id)

    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)
    stub_manager = StubFeishuLongConnectionManager()
    app.state.feishu_long_connection_manager = stub_manager

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "飞书长连接",
                "display_name": "飞书长连接入口",
                "kind": "feishu",
                "config": {
                    "app_id": "cli_test",
                    "connection_mode": "long_connection",
                },
                "credentials": [
                    {
                        "credential_key": "app_secret",
                        "storage_kind": "env",
                        "secret_ref": "FEISHU_APP_SECRET",
                    }
                ],
                "enabled": True,
            },
        )
        assert create_response.status_code == 201
        integration_id = create_response.json()["id"]
        assert stub_manager.started is True
        assert stub_manager.synced == [integration_id]

        disable_response = client.post(f"/api/integrations/{integration_id}/disable")
        assert disable_response.status_code == 200
        assert stub_manager.synced[-1] == integration_id

    assert stub_manager.stopped is True


def test_feishu_minimal_configuration_verifies_with_default_agent(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "飞书最小配置",
                "display_name": "飞书最小配置",
                "kind": "feishu",
                "config": {
                    "app_id": "cli_test",
                    "connection_mode": "long_connection",
                    "default_agent_id": "system-default-agent",
                },
                "credentials": [
                    {
                        "credential_key": "app_secret",
                        "storage_kind": "local_encrypted",
                        "secret_value": "app-secret-value",
                    }
                ],
            },
        )
        assert create_response.status_code == 201
        integration_id = create_response.json()["id"]

        verify_response = client.post(f"/api/integrations/{integration_id}/verify")
        assert verify_response.status_code == 200
        payload = verify_response.json()
        assert payload["success"] is True
        assert payload["integration"]["last_error"] == ""


def test_feishu_integration_api_persists_default_chat_id(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "椋炰功榛樿鐩爣",
                "display_name": "椋炰功榛樿鐩爣",
                "kind": "feishu",
                "config": {
                    "app_id": "cli_test",
                    "connection_mode": "long_connection",
                    "default_agent_id": "system-default-agent",
                    "default_chat_id": "oc_chat_saved",
                    "default_thread_id": "omt_saved_thread",
                },
                "credentials": [
                    {
                        "credential_key": "app_secret",
                        "storage_kind": "local_encrypted",
                        "secret_value": "app-secret-value",
                    }
                ],
            },
        )
        assert create_response.status_code == 201
        created = create_response.json()
        integration_id = created["id"]
        assert created["config"]["default_chat_id"] == "oc_chat_saved"
        assert created["config"]["default_thread_id"] == "omt_saved_thread"

        list_response = client.get("/api/integrations")
        assert list_response.status_code == 200
        listed = next(item for item in list_response.json() if item["id"] == integration_id)
        assert listed["config"]["default_chat_id"] == "oc_chat_saved"
        assert listed["config"]["default_thread_id"] == "omt_saved_thread"


def test_feishu_verify_rejects_unknown_default_agent(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/integrations",
            json={
                "name": "飞书非法 Agent",
                "display_name": "飞书非法 Agent",
                "kind": "feishu",
                "config": {
                    "app_id": "cli_test",
                    "connection_mode": "long_connection",
                    "default_agent_id": "missing-agent",
                },
                "credentials": [
                    {
                        "credential_key": "app_secret",
                        "storage_kind": "local_encrypted",
                        "secret_value": "app-secret-value",
                    }
                ],
            },
        )
        assert create_response.status_code == 201
        integration_id = create_response.json()["id"]

        verify_response = client.post(f"/api/integrations/{integration_id}/verify")
        assert verify_response.status_code == 200
        payload = verify_response.json()
        assert payload["success"] is False
        assert "默认 Agent 不存在" in payload["message"]


def test_gateway_auto_captures_default_chat_id_from_first_feishu_message(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(
        config,
        integration_config={
            "connection_mode": "long_connection",
            "default_agent_id": "system-default-agent",
        },
    )
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    gateway = IntegrationGateway(manager)
    result = asyncio.run(
        gateway.handle_channel_request(
            "feishu-integration",
            build_feishu_request(
                chat_id="oc_auto_detected",
                message_id="om_auto_detected",
                text="hello auto target",
            ),
        )
    )

    integration = IntegrationStore(session_db_path(config)).get_integration("feishu-integration")
    assert result.event is not None
    assert integration is not None
    assert integration.config["default_chat_id"] == "oc_auto_detected"
    assert integration.metadata["default_chat_id_source"] == "auto_detected_from_inbound"
    assert integration.metadata["default_chat_id_detected_message_id"] == "om_auto_detected"

    asyncio.run(manager.cleanup())


def test_integration_admin_api_routing_logs_and_retry_delivery(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))

        create_rule_response = client.post(
            "/api/integrations/mock-integration/routing-rules",
            json={
                "priority": 10,
                "match_type": "chat_id",
                "match_value": "chat-1",
                "agent_id": "system-default-agent",
                "session_strategy": "reuse",
                "enabled": True,
            },
        )
        assert create_rule_response.status_code == 201
        rule_id = create_rule_response.json()["id"]

        list_rules_response = client.get("/api/integrations/mock-integration/routing-rules")
        assert list_rules_response.status_code == 200
        assert len(list_rules_response.json()) == 1

        patch_rule_response = client.patch(
            f"/api/routing-rules/{rule_id}",
            json={"enabled": False},
        )
        assert patch_rule_response.status_code == 200
        assert patch_rule_response.json()["enabled"] is False

        event_store = InboundEventStore(session_db_path(config))
        delivery_store = DeliveryStore(session_db_path(config))
        event_store.create_event(
            InboundEventRecord(
                id="event-failed",
                integration_id="mock-integration",
                provider_event_id="evt-failed",
                provider_message_id="msg-failed",
                provider_chat_id="chat-1",
                provider_thread_id="",
                provider_user_id="user-1",
                event_type="message",
                received_at="2026-04-13T12:10:00+00:00",
                signature_valid=True,
                dedup_key="dedup-failed",
                raw_headers={"content-type": "application/json"},
                raw_payload={"message": {"text": "hello"}},
                normalized_status="failed",
                normalized_error="文本回写失败",
                metadata={},
            )
        )
        delivery_store.create_delivery(
            OutboundDeliveryRecord(
                id="delivery-failed",
                integration_id="mock-integration",
                run_id="run-failed",
                session_id=session_id,
                inbound_event_id="event-failed",
                provider_chat_id="chat-1",
                provider_thread_id="",
                provider_message_id="",
                delivery_type="text",
                payload={
                    "target_id": "chat-1",
                    "reply_to_message_id": "msg-failed",
                    "thread_id": "",
                    "message_type": "text",
                    "text": "处理完成",
                    "content": {"text": "处理完成"},
                    "dedup_key": "run-failed:text:1",
                    "metadata": {},
                },
                status="failed",
                attempt_count=1,
                last_attempt_at="2026-04-13T12:10:05+00:00",
                error_summary="文本回写失败",
                metadata={
                    "binding_id": "binding-1",
                    "reply_to_message_id": "msg-failed",
                    "run_status": "completed",
                },
                created_at="2026-04-13T12:10:00+00:00",
                updated_at="2026-04-13T12:10:05+00:00",
            )
        )
        delivery_store.create_attempt(
            DeliveryAttemptRecord(
                id="attempt-failed",
                delivery_id="delivery-failed",
                attempt_number=1,
                status="failed",
                request_payload={"text": "处理完成"},
                response_payload={"code": 500},
                error_summary="文本回写失败",
                started_at="2026-04-13T12:10:04+00:00",
                finished_at="2026-04-13T12:10:05+00:00",
            )
        )

        events_response = client.get("/api/integrations/mock-integration/events?status=failed")
        assert events_response.status_code == 200
        assert events_response.json()[0]["id"] == "event-failed"

        deliveries_response = client.get("/api/integrations/mock-integration/deliveries?status=failed")
        assert deliveries_response.status_code == 200
        deliveries_payload = deliveries_response.json()
        assert deliveries_payload[0]["id"] == "delivery-failed"
        assert deliveries_payload[0]["attempts"][0]["status"] == "failed"

        retry_response = client.post("/api/outbound-deliveries/delivery-failed/retry")
        assert retry_response.status_code == 200
        assert retry_response.json()["status"] == "delivered"

        retried_event = event_store.get_event("event-failed")
        assert retried_event is not None
        assert retried_event.normalized_status == "completed"

        delete_rule_response = client.delete(f"/api/routing-rules/{rule_id}")
        assert delete_rule_response.status_code == 200
        assert delete_rule_response.json()["id"] == rule_id


def test_api_create_feishu_integration_defaults_to_active(tmp_path: Path, monkeypatch):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)

    async def noop_sync(_self, _integration_id: str) -> None:
        return None

    monkeypatch.setattr(FeishuLongConnectionService, "sync_integration", noop_sync)
    monkeypatch.setattr(WeChatLongPollService, "sync_integration", noop_sync)

    app = create_app(manager)
    with TestClient(app) as client:
        response = client.post(
            "/api/integrations",
            json={
                "name": "feishu-default-active",
                "kind": "feishu",
                "display_name": "飞书默认启用",
                "config": {
                    "app_id": "cli_test",
                    "default_agent_id": "system-default-agent",
                },
                "credentials": [
                    {
                        "credential_key": "app_secret",
                        "storage_kind": "local_encrypted",
                        "secret_value": "app-secret",
                    }
                ],
            },
        )

    assert response.status_code == 201
    payload = response.json()
    assert payload["kind"] == "feishu"
    assert payload["status"] == "active"


def test_api_enable_invalid_integration_marks_error(tmp_path: Path, monkeypatch):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)

    async def noop_sync(_self, _integration_id: str) -> None:
        return None

    monkeypatch.setattr(FeishuLongConnectionService, "sync_integration", noop_sync)
    monkeypatch.setattr(WeChatLongPollService, "sync_integration", noop_sync)

    integration_id = "feishu-invalid"
    IntegrationStore(session_db_path(config)).create_integration(
        IntegrationConfigRecord(
            id=integration_id,
            name=integration_id,
            kind="feishu",
            status="disabled",
            webhook_path=f"/api/integrations/feishu/{integration_id}/webhook",
            config={
                "app_id": "cli_invalid",
                "default_agent_id": "system-default-agent",
            },
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
        )
    )

    app = create_app(manager)
    with TestClient(app) as client:
        response = client.post(f"/api/integrations/{integration_id}/enable")

    assert response.status_code == 400
    assert "app_secret" in response.json()["detail"]

    saved = IntegrationStore(session_db_path(config)).get_integration(integration_id)
    assert saved is not None
    assert saved.status == "error"
    assert "app_secret" in saved.last_error


