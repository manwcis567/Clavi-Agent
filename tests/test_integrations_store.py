import sqlite3
from pathlib import Path

import pytest

from clavi_agent.integration_models import (
    ConversationBindingRecord,
    DeliveryAttemptRecord,
    InboundEventRecord,
    IntegrationConfigRecord,
    IntegrationCredentialRecord,
    OutboundDeliveryRecord,
    RoutingRuleRecord,
    mask_secret,
)
from clavi_agent.integration_store import (
    ConversationBindingStore,
    DeliveryStore,
    InboundEventStore,
    IntegrationStore,
)
from clavi_agent.session_store import SessionStore
from clavi_agent.sqlite_schema import CURRENT_SESSION_DB_VERSION, SESSION_DB_SCOPE


def test_session_db_schema_adds_integrations_tables_and_indexes(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session("session-1", str(tmp_path), messages=[])
    IntegrationStore(db_path)

    with session_store._connect() as conn:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        indexes = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
        inbound_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(inbound_events)").fetchall()
        }
        delivery_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(outbound_deliveries)").fetchall()
        }
        version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (SESSION_DB_SCOPE,),
        ).fetchone()

    assert {
        "integrations",
        "integration_credentials",
        "inbound_events",
        "conversation_bindings",
        "routing_rules",
        "outbound_deliveries",
        "delivery_attempts",
    }.issubset(tables)
    assert {
        "idx_inbound_events_integration_provider_event_id",
        "idx_inbound_events_integration_dedup_key",
        "idx_conversation_bindings_lookup",
        "idx_outbound_deliveries_integration_status_updated",
        "idx_delivery_attempts_delivery_attempt",
    }.issubset(indexes)
    assert {
        "raw_payload_json",
        "raw_payload_size_bytes",
        "raw_payload_truncated",
        "raw_payload_redacted_fields_json",
    }.issubset(inbound_columns)
    assert {
        "inbound_event_id",
        "attempt_count",
        "last_attempt_at",
        "metadata_json",
    }.issubset(delivery_columns)
    assert version_row is not None
    assert version_row["version"] == CURRENT_SESSION_DB_VERSION


def test_integration_repositories_crud_flow(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session("session-1", str(tmp_path), messages=[])

    integration_store = IntegrationStore(db_path)
    binding_store = ConversationBindingStore(db_path)
    delivery_store = DeliveryStore(db_path)

    integration = IntegrationConfigRecord(
        id="integration-1",
        name="feishu-prod",
        kind="feishu",
        status="active",
        display_name="飞书生产机器人",
        tenant_id="tenant-1",
        webhook_path="/api/integrations/feishu/integration-1/webhook",
        config={"app_id": "cli_xxx", "encrypt_enabled": True},
        metadata={"owner": "qa"},
        created_at="2026-04-13T12:00:00+00:00",
        updated_at="2026-04-13T12:00:00+00:00",
    )
    integration_store.create_integration(integration)

    fetched_integration = integration_store.get_integration("integration-1")
    assert fetched_integration is not None
    assert fetched_integration.status == "active"
    assert fetched_integration.config["app_id"] == "cli_xxx"

    credential = IntegrationCredentialRecord(
        id="credential-1",
        integration_id="integration-1",
        credential_key="app_secret",
        storage_kind="env",
        secret_ref="FEISHU_APP_SECRET",
        created_at="2026-04-13T12:01:00+00:00",
        updated_at="2026-04-13T12:01:00+00:00",
    )
    integration_store.create_credential(credential)

    fetched_credential = integration_store.get_credential_by_key(
        "integration-1",
        "app_secret",
    )
    assert fetched_credential is not None
    assert fetched_credential.masked_value == mask_secret("FEISHU_APP_SECRET")

    rule_primary = RoutingRuleRecord(
        id="rule-1",
        integration_id="integration-1",
        priority=10,
        match_type="chat_id",
        match_value="oc_primary",
        agent_id="agent-1",
        session_strategy="reuse",
        enabled=True,
        created_at="2026-04-13T12:02:00+00:00",
        updated_at="2026-04-13T12:02:00+00:00",
    )
    rule_fallback = RoutingRuleRecord(
        id="rule-2",
        integration_id="integration-1",
        priority=50,
        match_type="integration_id",
        match_value="integration-1",
        agent_id="agent-2",
        session_strategy="reuse",
        enabled=True,
        created_at="2026-04-13T12:03:00+00:00",
        updated_at="2026-04-13T12:03:00+00:00",
    )
    integration_store.create_routing_rule(rule_fallback)
    integration_store.create_routing_rule(rule_primary)

    rules = integration_store.list_routing_rules("integration-1", enabled=True)
    assert [rule.id for rule in rules] == ["rule-1", "rule-2"]

    binding = ConversationBindingRecord(
        id="binding-1",
        integration_id="integration-1",
        tenant_id="tenant-1",
        chat_id="oc_primary",
        thread_id="",
        binding_scope="chat",
        agent_id="agent-1",
        session_id="session-1",
        metadata={"created_by": "system"},
        created_at="2026-04-13T12:04:00+00:00",
        updated_at="2026-04-13T12:04:00+00:00",
    )
    binding_store.create_binding(binding)

    found_binding = binding_store.find_binding(
        integration_id="integration-1",
        tenant_id="tenant-1",
        chat_id="oc_primary",
        thread_id="",
        binding_scope="chat",
    )
    assert found_binding is not None
    assert found_binding.session_id == "session-1"

    delivery = OutboundDeliveryRecord(
        id="delivery-1",
        integration_id="integration-1",
        run_id="run-1",
        session_id="session-1",
        inbound_event_id=None,
        provider_chat_id="oc_primary",
        provider_thread_id="",
        delivery_type="text",
        payload={"text": "处理完成"},
        created_at="2026-04-13T12:05:00+00:00",
        updated_at="2026-04-13T12:05:00+00:00",
    )
    delivery_store.create_delivery(delivery)

    sending_delivery = delivery.transition_to(
        "sending",
        changed_at="2026-04-13T12:05:10+00:00",
    )
    delivery_store.update_delivery(sending_delivery)

    attempt = DeliveryAttemptRecord(
        id="attempt-1",
        delivery_id="delivery-1",
        attempt_number=1,
        status="failed",
        request_payload={"text": "处理完成"},
        response_payload={"code": 429},
        error_summary="rate limited",
        started_at="2026-04-13T12:05:11+00:00",
        finished_at="2026-04-13T12:05:12+00:00",
    )
    delivery_store.create_attempt(attempt)

    retrying_delivery = sending_delivery.transition_to(
        "retrying",
        changed_at="2026-04-13T12:05:12+00:00",
        attempt_count=1,
        error_summary="rate limited",
    )
    delivery_store.update_delivery(retrying_delivery)

    fetched_delivery = delivery_store.get_delivery("delivery-1")
    assert fetched_delivery is not None
    assert fetched_delivery.status == "retrying"
    assert fetched_delivery.attempt_count == 1
    assert fetched_delivery.last_attempt_at == "2026-04-13T12:05:12+00:00"
    assert delivery_store.list_attempts("delivery-1")[0].error_summary == "rate limited"


def test_inbound_event_store_redacts_payload_and_enforces_dedup(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    IntegrationStore(db_path).create_integration(
        IntegrationConfigRecord(
            id="integration-1",
            name="mock-local",
            kind="mock",
            status="active",
            webhook_path="/api/integrations/mock/integration-1/webhook",
            created_at="2026-04-13T12:10:00+00:00",
            updated_at="2026-04-13T12:10:00+00:00",
        )
    )

    event_store = InboundEventStore(
        db_path,
        max_headers_bytes=128,
        max_payload_bytes=256,
    )
    event = InboundEventRecord(
        id="event-1",
        integration_id="integration-1",
        provider_event_id="evt-1",
        provider_message_id="msg-1",
        provider_chat_id="chat-1",
        provider_thread_id="thread-1",
        provider_user_id="user-1",
        event_type="message",
        received_at="2026-04-13T12:11:00+00:00",
        signature_valid=True,
        dedup_key="dedup-1",
        raw_headers={
            "Authorization": "Bearer secret-token",
            "X-Trace-Id": "trace-1",
        },
        raw_payload={
            "event": {
                "token": "token-123",
                "nested": {"app_secret": "secret-456"},
                "text": "hello",
            },
            "items": [{"password": "p@ssw0rd"}],
            "blob": "x" * 4096,
        },
        metadata={"source": "mock"},
    )
    stored_event = event_store.create_event(event)

    assert stored_event.raw_headers["Authorization"] == mask_secret("Bearer secret-token")
    assert stored_event.raw_headers_redacted_fields == ["Authorization"]
    assert stored_event.raw_payload_truncated is True
    assert {
        "event.token",
        "event.nested.app_secret",
        "items[0].password",
    }.issubset(set(stored_event.raw_payload_redacted_fields))
    assert stored_event.raw_payload["_truncated"] is True
    assert stored_event.raw_payload["_original_size_bytes"] > 256

    verified_event = stored_event.transition_to("verified")
    event_store.update_event(verified_event)

    fetched_event = event_store.get_event_by_dedup_key("integration-1", "dedup-1")
    assert fetched_event is not None
    assert fetched_event.normalized_status == "verified"
    assert event_store.get_event_by_provider_event_id("integration-1", "evt-1") is not None

    with pytest.raises(sqlite3.IntegrityError):
        event_store.create_event(
            event.model_copy(
                update={
                    "id": "event-2",
                    "provider_event_id": "evt-2",
                }
            )
        )


def test_integration_store_filters_by_account_and_children_inherit(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    SessionStore(db_path).create_session(
        "session-1",
        str(tmp_path),
        messages=[],
        account_id="account-a",
    )

    integration_store = IntegrationStore(db_path)
    binding_store = ConversationBindingStore(db_path)
    event_store = InboundEventStore(db_path)
    delivery_store = DeliveryStore(db_path)

    integration_store.create_integration(
        IntegrationConfigRecord(
            id="integration-a",
            account_id="account-a",
            name="owned",
            kind="mock",
            status="active",
            webhook_path="/api/integrations/mock/integration-a/webhook",
            created_at="2026-04-15T01:00:00+00:00",
            updated_at="2026-04-15T01:00:00+00:00",
        )
    )
    integration_store.create_integration(
        IntegrationConfigRecord(
            id="integration-b",
            account_id="account-b",
            name="other",
            kind="mock",
            status="active",
            webhook_path="/api/integrations/mock/integration-b/webhook",
            created_at="2026-04-15T01:01:00+00:00",
            updated_at="2026-04-15T01:01:00+00:00",
        )
    )

    credential = integration_store.create_credential(
        IntegrationCredentialRecord(
            id="cred-a",
            integration_id="integration-a",
            credential_key="token",
            storage_kind="env",
            secret_ref="TOKEN_ENV",
            created_at="2026-04-15T01:02:00+00:00",
            updated_at="2026-04-15T01:02:00+00:00",
        )
    )
    binding = binding_store.create_binding(
        ConversationBindingRecord(
            id="binding-a",
            integration_id="integration-a",
            chat_id="chat-a",
            thread_id="",
            binding_scope="chat",
            agent_id="agent-a",
            session_id="session-1",
            created_at="2026-04-15T01:03:00+00:00",
            updated_at="2026-04-15T01:03:00+00:00",
        )
    )
    event = event_store.create_event(
        InboundEventRecord(
            id="event-a",
            integration_id="integration-a",
            provider_event_id="evt-a",
            dedup_key="dedup-a",
            received_at="2026-04-15T01:04:00+00:00",
        )
    )
    delivery = delivery_store.create_delivery(
        OutboundDeliveryRecord(
            id="delivery-a",
            integration_id="integration-a",
            run_id="run-a",
            session_id="session-1",
            provider_chat_id="chat-a",
            delivery_type="text",
            created_at="2026-04-15T01:05:00+00:00",
            updated_at="2026-04-15T01:05:00+00:00",
        )
    )
    attempt = delivery_store.create_attempt(
        DeliveryAttemptRecord(
            id="attempt-a",
            delivery_id="delivery-a",
            attempt_number=1,
            status="sent",
            started_at="2026-04-15T01:06:00+00:00",
        )
    )

    assert credential.account_id == "account-a"
    assert binding.account_id == "account-a"
    assert event.account_id == "account-a"
    assert delivery.account_id == "account-a"
    assert attempt.account_id == "account-a"
    assert [item.id for item in integration_store.list_integrations(account_id="account-a")] == [
        "integration-a"
    ]
    assert integration_store.get_integration("integration-b", account_id="account-a") is None
    assert integration_store.get_credential("cred-a", account_id="account-a") is not None
    assert integration_store.get_credential("cred-a", account_id="account-b") is None
    assert binding_store.get_binding("binding-a", account_id="account-a") is not None
    assert binding_store.get_binding("binding-a", account_id="account-b") is None
    assert delivery_store.get_delivery("delivery-a", account_id="account-a") is not None
    assert delivery_store.get_delivery("delivery-a", account_id="account-b") is None
    assert len(delivery_store.list_attempts("delivery-a", account_id="account-a")) == 1

