import asyncio
from pathlib import Path

from clavi_agent.config import (
    AgentConfig,
    Config,
    FeatureFlagsConfig,
    LLMConfig,
    RetryConfig,
    ToolsConfig,
)
from clavi_agent.integration_models import ConversationBindingRecord, IntegrationConfigRecord, RoutingRuleRecord
from clavi_agent.integration_store import ConversationBindingStore, IntegrationStore
from clavi_agent.integrations import IntegrationRouter, ParsedInboundEvent, ROOT_THREAD_ID
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


def create_integration(
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
        config=integration_config or {},
        created_at="2026-04-13T12:00:00+00:00",
        updated_at="2026-04-13T12:00:00+00:00",
    )
    IntegrationStore(session_db_path(config)).create_integration(record)
    return record


def build_event(
    *,
    integration_id: str = "mock-integration",
    chat_id: str = "chat-1",
    thread_id: str = "",
    tenant_id: str = "tenant-1",
    sender_id: str = "user-1",
    is_group: bool = True,
    received_at: str = "2026-04-13T12:00:00+00:00",
) -> ParsedInboundEvent:
    return ParsedInboundEvent(
        integration_id=integration_id,
        channel_kind="mock",
        event_type="message",
        provider_event_id="evt-1",
        provider_message_id="msg-1",
        provider_chat_id=chat_id,
        provider_thread_id=thread_id,
        provider_user_id=sender_id,
        tenant_id=tenant_id,
        dedup_key="msg-1",
        received_at=received_at,
        signature_valid=True,
        message_type="text",
        text="hello",
        sender_name="测试用户",
        is_group=is_group,
    )


def test_router_creates_rule_binding_and_reuses_session(tmp_path: Path):
    config = build_config(tmp_path)
    integration = create_integration(config)
    IntegrationStore(session_db_path(config)).create_routing_rule(
        RoutingRuleRecord(
            id="rule-1",
            integration_id=integration.id,
            priority=10,
            match_type="chat_id",
            match_value="chat-1",
            agent_id="system-default-agent",
            session_strategy="chat",
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
        )
    )

    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())
    router = IntegrationRouter(manager)

    first = asyncio.run(router.resolve_route(integration, build_event()))
    second = asyncio.run(
        router.resolve_route(
            integration,
            build_event(received_at="2026-04-13T12:01:00+00:00"),
        )
    )

    assert first.source == "rule"
    assert first.binding is not None
    assert first.binding.binding_scope == "chat"
    assert first.binding.thread_id == ROOT_THREAD_ID
    assert first.session_key.endswith(f"{ROOT_THREAD_ID}:chat:system-default-agent")
    assert second.source == "binding"
    assert second.binding is not None
    assert second.binding.session_id == first.binding.session_id

    asyncio.run(manager.cleanup())


def test_router_private_chat_uses_root_thread_for_default_agent(tmp_path: Path):
    config = build_config(tmp_path)
    integration = create_integration(
        config,
        integration_config={"default_agent_id": "system-default-agent"},
    )
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())
    router = IntegrationRouter(manager)

    result = asyncio.run(
        router.resolve_route(
            integration,
            build_event(is_group=False),
        )
    )

    assert result.source == "default"
    assert result.binding is not None
    assert result.binding.binding_scope == "thread"
    assert result.binding.thread_id == ROOT_THREAD_ID
    assert result.session_key.endswith(f"{ROOT_THREAD_ID}:thread:system-default-agent")

    asyncio.run(manager.cleanup())


def test_router_ignores_foreign_account_default_agent(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())

    account_a = manager._account_store.create_account(
        username="router-alice",
        password="secret-a",
    )
    account_b = manager._account_store.create_account(
        username="router-bob",
        password="secret-b",
    )
    manager._agent_store.create_agent(
        agent_id="foreign-agent",
        name="Foreign Agent",
        system_prompt="You are foreign.",
        account_id=account_b["id"],
    )

    integration = IntegrationConfigRecord(
        id="mock-foreign-default",
        account_id=account_a["id"],
        name="mock-foreign-default",
        kind="mock",
        status="active",
        webhook_path="/api/integrations/mock/mock-foreign-default/webhook",
        config={"default_agent_id": "foreign-agent"},
        created_at="2026-04-13T12:00:00+00:00",
        updated_at="2026-04-13T12:00:00+00:00",
    )
    IntegrationStore(session_db_path(config)).create_integration(integration)

    router = IntegrationRouter(manager)
    result = asyncio.run(
        router.resolve_route(
            integration,
            build_event(integration_id=integration.id),
        )
    )

    assert result.source == "unbound"
    assert "尚未绑定 Agent" in result.message

    asyncio.run(manager.cleanup())


def test_router_disables_invalid_binding_when_agent_missing(tmp_path: Path):
    config = build_config(tmp_path)
    integration = create_integration(config)
    binding_store = ConversationBindingStore(session_db_path(config))
    binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-1",
            integration_id=integration.id,
            tenant_id="tenant-1",
            chat_id="chat-1",
            thread_id=ROOT_THREAD_ID,
            binding_scope="chat",
            agent_id="deleted-agent",
            session_id="missing-session",
            enabled=True,
            created_at="2026-04-13T12:00:00+00:00",
            updated_at="2026-04-13T12:00:00+00:00",
            last_message_at="2026-04-13T12:00:00+00:00",
        )
    )

    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())
    router = IntegrationRouter(manager)

    result = asyncio.run(router.resolve_route(integration, build_event(thread_id="", is_group=True)))
    disabled_binding = binding_store.get_binding("binding-1")

    assert result.source == "unbound"
    assert "当前会话尚未绑定 Agent" in result.message
    assert result.disabled_binding_id == "binding-1"
    assert disabled_binding is not None
    assert disabled_binding.enabled is False

    asyncio.run(manager.cleanup())

