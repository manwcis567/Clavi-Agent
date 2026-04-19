import asyncio
import json
import mimetypes
from pathlib import Path
from types import SimpleNamespace

import httpx

from clavi_agent.agent import Agent
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
    InboundEventRecord,
    IntegrationConfigRecord,
)
from clavi_agent.integration_store import (
    ConversationBindingStore,
    DeliveryStore,
    InboundEventStore,
    IntegrationStore,
)
from clavi_agent.integrations import (
    ChannelAdapterRegistry,
    ChannelRequest,
    FeishuAdapter,
    IntegrationGateway,
    IntegrationRunBridge,
    MockChannelAdapter,
    NormalizedAdapterError,
    OutboundSendResult,
)
from clavi_agent.run_models import ArtifactRecord, RunDeliverableManifest, RunDeliverableRef
from clavi_agent.schema import Message
from clavi_agent.session import SessionManager
from clavi_agent.sqlite_schema import utc_now_iso


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
    integration_config: dict | None = None,
) -> IntegrationConfigRecord:
    record = IntegrationConfigRecord(
        id=integration_id,
        name=integration_id,
        kind="mock",
        status="active",
        webhook_path=f"/api/integrations/mock/{integration_id}/webhook",
        config={
            "verify_token": "mock-token",
            "outbound_retry_backoff_seconds": 0,
            **(integration_config or {}),
        },
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
            "app_secret": "app-secret",
            "verification_token": "verify-token",
            **(integration_config or {}),
        },
        created_at="2026-04-13T12:00:00+00:00",
        updated_at="2026-04-13T12:00:00+00:00",
    )
    IntegrationStore(session_db_path(config)).create_integration(record)
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
            "attachments": [],
        },
    }
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


def create_binding(
    config: Config,
    session_id: str,
    *,
    integration_id: str = "mock-integration",
) -> ConversationBindingRecord:
    binding = ConversationBindingRecord(
        id="binding-1",
        integration_id=integration_id,
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
    ConversationBindingStore(session_db_path(config)).create_binding(binding)
    return binding


class StubAgent(Agent):
    def __init__(self, workspace_dir: str, response_text: str):
        super().__init__(
            llm_client=SimpleNamespace(),
            system_prompt="test",
            tools=[],
            workspace_dir=workspace_dir,
        )
        self._response_text = response_text

    async def run_stream(self):
        yield {"type": "step", "data": {"current": 1}}
        self.messages.append(Message(role="assistant", content=self._response_text))
        yield {"type": "content", "data": {"content": self._response_text}}
        yield {"type": "done", "data": {"content": self._response_text}}


class RetryingMockAdapter(MockChannelAdapter):
    def __init__(self, failures_before_success: int):
        super().__init__()
        self.failures_before_success = failures_before_success
        self.message_attempts = 0

    async def send_outbound_message(self, context, message) -> OutboundSendResult:
        self.message_attempts += 1
        if self.message_attempts <= self.failures_before_success:
            return OutboundSendResult(
                ok=False,
                error=NormalizedAdapterError(
                    code="rate_limited",
                    message="rate limited",
                    retryable=True,
                ),
                raw_response={"code": 429},
            )
        return await super().send_outbound_message(context, message)


async def drain_run_events(manager: SessionManager, run_id: str) -> None:
    async for _event in manager.stream_run(run_id):
        pass


async def wait_for_deliveries(
    delivery_store: DeliveryStore,
    run_id: str,
    *,
    expected_count: int,
    timeout_seconds: float = 2.0,
) -> list:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        deliveries = delivery_store.list_deliveries(run_id=run_id)
        if len(deliveries) >= expected_count and all(
            delivery.status in {"delivered", "failed"} for delivery in deliveries
        ):
            return deliveries
        await asyncio.sleep(0.01)
    return delivery_store.list_deliveries(run_id=run_id)


def create_manual_integration_run(
    manager: SessionManager,
    config: Config,
    session_id: str,
    *,
    integration_id: str = "mock-integration",
    channel_kind: str = "mock",
    inbound_event_id: str,
    provider_message_id: str = "mock-msg-1",
    provider_chat_id: str = "chat-1",
    run_status: str = "completed",
    assistant_text: str = "已处理完成",
) -> tuple[str, str]:
    assert manager._run_manager is not None
    assert manager._run_store is not None
    assert manager._session_store is not None

    event = InboundEventRecord(
        id=inbound_event_id,
        integration_id=integration_id,
        provider_event_id=f"evt-{inbound_event_id}",
        provider_message_id=provider_message_id,
        provider_chat_id=provider_chat_id,
        provider_thread_id="",
        provider_user_id="user-1",
        event_type="message",
        received_at="2026-04-13T12:00:00+00:00",
        signature_valid=True,
        dedup_key=provider_message_id,
        raw_headers={},
        raw_payload={"message": {"message_id": provider_message_id}},
        normalized_status="bridged",
        metadata={"binding_id": "binding-1"},
    )
    InboundEventStore(session_db_path(config)).create_event(event)

    run = manager._run_manager.create_run(
        session_id,
        "manual integration run",
        run_metadata={
            "source_kind": "integration",
            "source_label": f"integration:{channel_kind}",
            "integration_id": integration_id,
            "channel_kind": channel_kind,
            "binding_id": "binding-1",
            "inbound_event_id": inbound_event_id,
            "provider_message_id": provider_message_id,
            "provider_chat_id": provider_chat_id,
            "provider_thread_id": "",
            "provider_user_id": "user-1",
        },
    )
    run = run.model_copy(
        update={
            "status": run_status,
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "error_summary": "" if run_status == "completed" else "LLM unavailable",
        }
    )
    manager._run_store.update_run(run)

    if assistant_text:
        manager._session_store.append_message(
            session_id,
            Message(role="assistant", content=assistant_text),
        )

    return run.id, inbound_event_id


def attach_deliverable_artifact(
    manager: SessionManager,
    session_id: str,
    run_id: str,
    *,
    artifact_id: str = "artifact-1",
    display_name: str = "report.md",
) -> None:
    assert manager._run_store is not None
    session = manager.get_session_info(session_id)
    assert session is not None
    workspace_dir = Path(session["workspace_dir"])
    output_dir = workspace_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / display_name
    artifact_path.write_text("# report\n", encoding="utf-8")

    artifact = ArtifactRecord(
        id=artifact_id,
        run_id=run_id,
        artifact_type="workspace_file",
        uri=str(artifact_path.relative_to(workspace_dir).as_posix()),
        display_name=display_name,
        role="final_deliverable",
        format="md",
        mime_type="text/markdown",
        is_final=True,
        preview_kind="text",
        summary="生成报告",
        created_at=utc_now_iso(),
    )
    manager._run_store.create_artifact(artifact)
    run = manager._run_store.get_run(run_id)
    assert run is not None
    manager._run_store.update_run(
        run.model_copy(
            update={
                "deliverable_manifest": RunDeliverableManifest(
                    primary_artifact_id=artifact.id,
                    items=[
                        RunDeliverableRef(
                            artifact_id=artifact.id,
                            uri=artifact.uri,
                            display_name=artifact.display_name,
                            format=artifact.format,
                            mime_type=artifact.mime_type,
                            role=artifact.role,
                            is_primary=True,
                        )
                    ],
                )
            }
        )
    )


def attach_generated_artifact(
    manager: SessionManager,
    session_id: str,
    run_id: str,
    *,
    artifact_id: str,
    display_name: str,
    content: str,
    role: str = "intermediate_file",
    is_final: bool = False,
) -> ArtifactRecord:
    assert manager._run_store is not None
    session = manager.get_session_info(session_id)
    assert session is not None
    workspace_dir = Path(session["workspace_dir"])
    output_dir = workspace_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / display_name
    artifact_path.write_text(content, encoding="utf-8")

    artifact = ArtifactRecord(
        id=artifact_id,
        run_id=run_id,
        artifact_type="workspace_file",
        uri=str(artifact_path.relative_to(workspace_dir).as_posix()),
        display_name=display_name,
        role=role,
        format=artifact_path.suffix.lstrip(".").lower(),
        mime_type=mimetypes.guess_type(display_name)[0] or "application/octet-stream",
        is_final=is_final,
        preview_kind="text",
        summary=f"生成文件 {display_name}",
        created_at=utc_now_iso(),
    )
    manager._run_store.create_artifact(artifact)
    return artifact


def test_terminal_run_callback_auto_dispatches_text_reply(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    mock_adapter = MockChannelAdapter()
    async def scenario() -> None:
        await manager.initialize()
        manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry([mock_adapter])

        session_id = await manager.create_session(agent_id="system-default-agent")
        create_binding(config, session_id)
        session = manager.get_session_info(session_id)
        assert session is not None
        manager.bind_session_agent(
            session_id,
            StubAgent(session["workspace_dir"], "渠道回复正文"),
        )

        gateway = IntegrationGateway(manager)
        result = await gateway.handle_channel_request(
            "mock-integration",
            build_mock_request(text="请回复这条消息"),
        )
        bridge_result = await IntegrationRunBridge(manager).bridge_gateway_result(result)
        assert bridge_result is not None

        await drain_run_events(manager, bridge_result.run_id)
        deliveries = await wait_for_deliveries(
            DeliveryStore(session_db_path(config)),
            bridge_result.run_id,
            expected_count=1,
        )

        assert len(deliveries) == 1
        assert deliveries[0].status == "delivered"
        assert mock_adapter.sent_payloads[0]["payload"]["text"] == "渠道回复正文"

        stored_event = InboundEventStore(session_db_path(config)).get_event(result.event.id)
        assert stored_event is not None
        assert stored_event.normalized_status == "completed"
        assert stored_event.metadata["reply_dispatch_status"] == "delivered"

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(manager.cleanup())


def test_run_bridge_schedules_feishu_quick_reaction(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(config)
    manager = SessionManager(config=config)
    recorded_requests: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages/om_message_1/reactions":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {
                "reaction_type": {
                    "emoji_type": "DONE",
                }
            }
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {},
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def scenario() -> None:
        await manager.initialize()
        assert manager._run_manager is not None
        manager._run_manager._schedule_dispatch = lambda: None
        manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry(
            [FeishuAdapter(transport=httpx.MockTransport(handler))]
        )

        session_id = await manager.create_session(agent_id="system-default-agent")
        ConversationBindingStore(session_db_path(config)).create_binding(
            ConversationBindingRecord(
                id="binding-feishu-quick-reaction",
                integration_id="feishu-integration",
                tenant_id="tenant-key-1",
                chat_id="oc_chat_1",
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
        result = await gateway.handle_channel_request(
            "feishu-integration",
            build_feishu_request(text="请先给我一个快速反馈"),
        )
        bridge_result = await IntegrationRunBridge(manager).bridge_gateway_result(result)
        assert bridge_result is not None

        deliveries = await wait_for_deliveries(
            DeliveryStore(session_db_path(config)),
            bridge_result.run_id,
            expected_count=1,
        )

        assert len(deliveries) == 1
        delivery = deliveries[0]
        assert delivery.delivery_type == "reaction"
        assert delivery.status == "delivered"
        assert delivery.payload["message_id"] == "om_message_1"
        assert delivery.payload["reaction_type"] == "DONE"

        stored_event = InboundEventStore(session_db_path(config)).get_event(result.event.id)
        assert stored_event is not None
        assert stored_event.metadata["quick_response_status"] == "delivered"
        assert stored_event.metadata["quick_response_reaction_type"] == "DONE"
        assert stored_event.metadata["quick_response_delivery_id"] == f"{bridge_result.run_id}:reaction"
        assert stored_event.metadata["quick_response_target_message_id"] == "om_message_1"

    try:
        asyncio.run(scenario())
    finally:
        asyncio.run(manager.cleanup())

    assert recorded_requests == [
        ("POST", "/open-apis/auth/v3/tenant_access_token/internal", b'{"app_id":"cli_test","app_secret":"app-secret"}'),
        ("POST", "/open-apis/im/v1/messages/om_message_1/reactions", b'{"reaction_type":{"emoji_type":"DONE"}}'),
    ]


def test_reply_dispatcher_sends_error_notice_for_failed_run(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    mock_adapter = MockChannelAdapter()
    manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry([mock_adapter])

    try:
        session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
        create_binding(config, session_id)
        run_id, inbound_event_id = create_manual_integration_run(
            manager,
            config,
            session_id,
            inbound_event_id="inbound-failed",
            run_status="failed",
            assistant_text="",
        )

        envelope = asyncio.run(manager._integration_reply_dispatcher.dispatch_run(run_id))
        deliveries = DeliveryStore(session_db_path(config)).list_deliveries(run_id=run_id)
        assert envelope is not None
        assert len(deliveries) == 1
        assert deliveries[0].delivery_type == "error"
        assert deliveries[0].status == "delivered"
        assert mock_adapter.sent_payloads[0]["payload"]["text"] == "处理失败：LLM unavailable"

        stored_event = InboundEventStore(session_db_path(config)).get_event(inbound_event_id)
        assert stored_event is not None
        assert stored_event.normalized_status == "completed"
        assert stored_event.metadata["run_status"] == "failed"
    finally:
        asyncio.run(manager.cleanup())


def test_reply_dispatcher_delivers_artifact_reference(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(
        config,
        integration_config={"public_base_url": "https://clavi-agent.example.com"},
    )
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    mock_adapter = MockChannelAdapter()
    manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry([mock_adapter])

    try:
        session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
        create_binding(config, session_id)
        run_id, inbound_event_id = create_manual_integration_run(
            manager,
            config,
            session_id,
            inbound_event_id="inbound-artifact",
            assistant_text="已生成报告",
        )
        attach_deliverable_artifact(manager, session_id, run_id)

        envelope = asyncio.run(manager._integration_reply_dispatcher.dispatch_run(run_id))
        deliveries = DeliveryStore(session_db_path(config)).list_deliveries(run_id=run_id)

        assert envelope is not None
        assert len(deliveries) == 2
        deliveries_by_type = {delivery.delivery_type: delivery for delivery in deliveries}
        assert deliveries_by_type["text"].status == "delivered"
        assert deliveries_by_type["artifact_ref"].status == "delivered"
        assert mock_adapter.sent_payloads[0]["kind"] == "message"
        assert mock_adapter.sent_payloads[1]["kind"] == "file"
        assert (
            mock_adapter.sent_payloads[1]["payload"]["url"]
            == "https://clavi-agent.example.com/api/artifacts/artifact-1"
        )

        stored_event = InboundEventStore(session_db_path(config)).get_event(inbound_event_id)
        assert stored_event is not None
        assert stored_event.normalized_status == "completed"
        assert stored_event.metadata["reply_dispatch_status"] == "delivered"
    finally:
        asyncio.run(manager.cleanup())


def test_reply_dispatcher_uploads_feishu_artifact_before_sending(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(config)
    manager = SessionManager(config=config)
    recorded_requests: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            payload = json.loads(request.content.decode("utf-8"))
            if payload["msg_type"] == "text":
                return httpx.Response(
                    200,
                    json={
                        "code": 0,
                        "msg": "ok",
                        "data": {"message_id": "om_text_1", "chat_id": "oc_chat_1"},
                    },
                )
            assert payload["msg_type"] == "file"
            assert json.loads(payload["content"]) == {"file_key": "file_v2_uploaded"}
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"message_id": "om_file_1", "chat_id": "oc_chat_1"},
                },
            )
        if request.url.path == "/open-apis/im/v1/files":
            assert b"report.md" in request.content
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"file_key": "file_v2_uploaded"},
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    asyncio.run(manager.initialize())
    manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry(
        [FeishuAdapter(transport=httpx.MockTransport(handler))]
    )

    try:
        session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
        create_binding(config, session_id, integration_id="feishu-integration")
        run_id, inbound_event_id = create_manual_integration_run(
            manager,
            config,
            session_id,
            integration_id="feishu-integration",
            channel_kind="feishu",
            inbound_event_id="inbound-feishu-artifact",
            provider_message_id="om_source",
            provider_chat_id="oc_chat_1",
            assistant_text="已生成报告",
        )
        attach_deliverable_artifact(manager, session_id, run_id)

        envelope = asyncio.run(manager._integration_reply_dispatcher.dispatch_run(run_id))
        deliveries = DeliveryStore(session_db_path(config)).list_deliveries(run_id=run_id)

        assert envelope is not None
        assert len(deliveries) == 2
        deliveries_by_type = {delivery.delivery_type: delivery for delivery in deliveries}
        assert deliveries_by_type["text"].status == "delivered"
        assert deliveries_by_type["artifact_ref"].status == "delivered"

        stored_event = InboundEventStore(session_db_path(config)).get_event(inbound_event_id)
        assert stored_event is not None
        assert stored_event.normalized_status == "completed"
        assert stored_event.metadata["reply_dispatch_status"] == "delivered"
    finally:
        asyncio.run(manager.cleanup())

    assert [path for _, path, _ in recorded_requests] == [
        "/open-apis/auth/v3/tenant_access_token/internal",
        "/open-apis/im/v1/messages",
        "/open-apis/im/v1/files",
        "/open-apis/im/v1/messages",
    ]


def test_feishu_reply_dispatcher_sends_root_and_child_generated_files(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(config)
    manager = SessionManager(config=config)
    recorded_requests: list[tuple[str, str, bytes]] = []
    counters = {"upload": 0, "message": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/files":
            counters["upload"] += 1
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"file_key": f"file_v2_{counters['upload']}"},
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            counters["message"] += 1
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "message_id": f"om_{counters['message']}",
                        "chat_id": "oc_chat_1",
                    },
                },
            )
        return httpx.Response(404, json={"code": 404, "msg": "not found"})

    asyncio.run(manager.initialize())
    manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry(
        [FeishuAdapter(transport=httpx.MockTransport(handler))]
    )

    try:
        session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
        create_binding(config, session_id, integration_id="feishu-integration")
        run_id, _ = create_manual_integration_run(
            manager,
            config,
            session_id,
            integration_id="feishu-integration",
            channel_kind="feishu",
            inbound_event_id="inbound-feishu-generated-files",
            provider_message_id="om_source",
            provider_chat_id="oc_chat_1",
            assistant_text="已生成多个文件",
        )
        attach_generated_artifact(
            manager,
            session_id,
            run_id,
            artifact_id="artifact-root",
            display_name="draft.txt",
            content="root file",
        )

        assert manager._run_manager is not None
        assert manager._run_store is not None
        child_run = manager._run_manager.create_run(
            session_id,
            "child artifact run",
            parent_run_id=run_id,
            run_metadata={
                "kind": "delegate_child",
                "agent_name": "worker-1",
                "root_run_id": run_id,
            },
        )
        child_run = child_run.model_copy(
            update={
                "status": "completed",
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
            }
        )
        manager._run_store.update_run(child_run)
        attach_generated_artifact(
            manager,
            session_id,
            child_run.id,
            artifact_id="artifact-child",
            display_name="child.csv",
            content="id\n1\n",
        )

        envelope = asyncio.run(manager._integration_reply_dispatcher.dispatch_run(run_id))
        deliveries = DeliveryStore(session_db_path(config)).list_deliveries(run_id=run_id)

        assert envelope is not None
        artifact_deliveries = [
            delivery for delivery in deliveries if delivery.delivery_type == "artifact_ref"
        ]
        assert len(artifact_deliveries) == 2
        assert {delivery.payload["file_name"] for delivery in artifact_deliveries} == {
            "draft.txt",
            "child.csv",
        }
    finally:
        asyncio.run(manager.cleanup())

    assert [path for _, path, _ in recorded_requests] == [
        "/open-apis/auth/v3/tenant_access_token/internal",
        "/open-apis/im/v1/messages",
        "/open-apis/im/v1/files",
        "/open-apis/im/v1/messages",
        "/open-apis/im/v1/files",
        "/open-apis/im/v1/messages",
    ]


def test_feishu_bound_run_agent_can_send_channel_file_and_dedupe_final_dispatch(tmp_path: Path):
    config = build_config(tmp_path)
    create_feishu_integration(config)
    manager = SessionManager(config=config)
    recorded_requests: list[tuple[str, str, bytes]] = []
    counters = {"upload": 0, "message": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/files":
            counters["upload"] += 1
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"file_key": f"tool_file_v2_{counters['upload']}"},
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            counters["message"] += 1
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "message_id": f"om_{counters['message']}",
                        "chat_id": "oc_chat_1",
                    },
                },
            )
        return httpx.Response(404, json={"code": 404, "msg": "not found"})

    asyncio.run(manager.initialize())
    manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry(
        [FeishuAdapter(transport=httpx.MockTransport(handler))]
    )

    try:
        session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
        create_binding(config, session_id, integration_id="feishu-integration")
        run_id, _ = create_manual_integration_run(
            manager,
            config,
            session_id,
            integration_id="feishu-integration",
            channel_kind="feishu",
            inbound_event_id="inbound-feishu-tool-file",
            provider_message_id="om_source",
            provider_chat_id="oc_chat_1",
            assistant_text="最终回复正文",
        )

        session = manager.get_session_info(session_id)
        assert session is not None
        manual_path = Path(session["workspace_dir"]) / "outputs" / "manual.txt"
        manual_path.parent.mkdir(parents=True, exist_ok=True)
        manual_path.write_text("manual file", encoding="utf-8")

        assert manager._run_store is not None
        run = manager._run_store.get_run(run_id)
        assert run is not None
        agent = manager._load_run_agent(run)
        assert agent is not None
        assert "send_channel_file" in agent.tools

        send_result = asyncio.run(
            agent.tools["send_channel_file"].execute(path="outputs/manual.txt")
        )
        assert send_result.success is True
        assert send_result.metadata["delivery_type"] == "tool_file"

        attach_generated_artifact(
            manager,
            session_id,
            run_id,
            artifact_id="artifact-manual",
            display_name="manual.txt",
            content="manual file",
        )

        envelope = asyncio.run(manager._integration_reply_dispatcher.dispatch_run(run_id))
        deliveries = DeliveryStore(session_db_path(config)).list_deliveries(run_id=run_id)

        assert envelope is not None
        assert any(delivery.delivery_type == "tool_file" for delivery in deliveries)
        assert any(delivery.delivery_type == "text" for delivery in deliveries)
        assert not any(delivery.delivery_type == "artifact_ref" for delivery in deliveries)
    finally:
        asyncio.run(manager.cleanup())

    assert [path for _, path, _ in recorded_requests] == [
        "/open-apis/auth/v3/tenant_access_token/internal",
        "/open-apis/im/v1/files",
        "/open-apis/im/v1/messages",
        "/open-apis/im/v1/messages",
    ]


def test_reply_dispatcher_retries_retryable_failures_and_is_idempotent(tmp_path: Path):
    config = build_config(tmp_path)
    create_mock_integration(
        config,
        integration_config={
            "outbound_max_attempts": 3,
            "outbound_retry_backoff_seconds": 0,
        },
    )
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    retrying_adapter = RetryingMockAdapter(failures_before_success=2)
    manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry([retrying_adapter])

    try:
        session_id = asyncio.run(manager.create_session(agent_id="system-default-agent"))
        create_binding(config, session_id)
        run_id, inbound_event_id = create_manual_integration_run(
            manager,
            config,
            session_id,
            inbound_event_id="inbound-retry",
            assistant_text="需要重试后才能发送",
        )

        asyncio.run(manager._integration_reply_dispatcher.dispatch_run(run_id))
        asyncio.run(manager._integration_reply_dispatcher.dispatch_run(run_id))
        delivery = DeliveryStore(session_db_path(config)).get_delivery(f"{run_id}:text:1")
        assert delivery is not None
        assert delivery.status == "delivered"
        assert delivery.attempt_count == 3
        assert retrying_adapter.message_attempts == 3
        assert len(retrying_adapter.sent_payloads) == 1

        stored_event = InboundEventStore(session_db_path(config)).get_event(inbound_event_id)
        assert stored_event is not None
        assert stored_event.normalized_status == "completed"
    finally:
        asyncio.run(manager.cleanup())


