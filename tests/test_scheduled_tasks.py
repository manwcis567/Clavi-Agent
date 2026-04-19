import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from clavi_agent.config import (
    AgentConfig,
    Config,
    FeatureFlagsConfig,
    LLMConfig,
    RetryConfig,
    ToolsConfig,
)
from clavi_agent.integration_models import IntegrationConfigRecord
from clavi_agent.integration_store import DeliveryStore, IntegrationStore
from clavi_agent.integrations import ChannelAdapterRegistry, MockChannelAdapter
from clavi_agent.scheduled_task_service import ScheduledTaskService
from clavi_agent.scheduled_task_store import ScheduledTaskStore
from clavi_agent.schema import LLMResponse
from clavi_agent.server import create_app
from clavi_agent.session import SessionManager
from clavi_agent.session_store import SessionStore
from clavi_agent.sqlite_schema import CURRENT_SESSION_DB_VERSION, SESSION_DB_SCOPE
from clavi_agent.scheduled_task_models import ScheduledTaskExecutionRecord, ScheduledTaskRecord


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
    integration_id: str = "mock-scheduled",
    *,
    integration_config: dict | None = None,
) -> None:
    IntegrationStore(session_db_path(config)).create_integration(
        IntegrationConfigRecord(
            id=integration_id,
            name=integration_id,
            kind="mock",
            status="active",
            webhook_path=f"/api/integrations/mock/{integration_id}/webhook",
            config=integration_config
            or {
                "default_agent_id": "system-default-agent",
                "default_chat_id": "ops-room",
                "outbound_retry_backoff_seconds": 0,
            },
            created_at="2026-04-14T00:00:00+00:00",
            updated_at="2026-04-14T00:00:00+00:00",
        )
    )


def wait_for(predicate, timeout_seconds: float = 2.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for condition.")


def test_session_db_schema_adds_scheduled_task_tables_and_indexes(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    SessionStore(db_path).create_session("session-1", str(tmp_path), messages=[])
    ScheduledTaskStore(db_path)

    with SessionStore(db_path)._connect() as conn:
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
        version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (SESSION_DB_SCOPE,),
        ).fetchone()

    assert {"scheduled_tasks", "scheduled_task_executions"}.issubset(tables)
    assert {
        "idx_scheduled_tasks_enabled_next_run",
        "idx_scheduled_task_executions_task_created",
        "idx_scheduled_task_executions_run",
    }.issubset(indexes)
    assert version_row is not None
    assert version_row["version"] == CURRENT_SESSION_DB_VERSION


@patch("clavi_agent.session.LLMClient")
def test_scheduled_task_api_crud_run_and_detail(mock_llm_class, tmp_path: Path):
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="scheduled reply", finish_reason="stop")
    )

    config = build_config(tmp_path)
    create_mock_integration(config)
    manager = SessionManager(config=config)
    app = create_app(manager)
    mock_adapter = MockChannelAdapter()

    with TestClient(app) as client:
        manager._integration_reply_dispatcher._adapter_registry = ChannelAdapterRegistry(
            [mock_adapter]
        )

        create_response = client.post(
            "/api/scheduled-tasks",
            json={
                "name": "Daily report",
                "cron_expression": "0 8 * * *",
                "prompt": "Prepare the daily report.",
                "integration_id": "mock-scheduled",
            },
        )
        assert create_response.status_code == 201
        created = create_response.json()
        task_id = created["id"]
        assert created["next_run_at"]
        assert created["agent_id"] == "system-default-agent"
        assert created["resolved_target_chat_id"] == "ops-room"

        update_response = client.patch(
            f"/api/scheduled-tasks/{task_id}",
            json={"name": "Daily report v2"},
        )
        assert update_response.status_code == 200
        assert update_response.json()["name"] == "Daily report v2"

        disable_response = client.post(f"/api/scheduled-tasks/{task_id}/disable")
        assert disable_response.status_code == 200
        assert disable_response.json()["enabled"] is False

        enable_response = client.post(f"/api/scheduled-tasks/{task_id}/enable")
        assert enable_response.status_code == 200
        assert enable_response.json()["enabled"] is True

        run_response = client.post(f"/api/scheduled-tasks/{task_id}/run")
        assert run_response.status_code == 200
        execution = run_response.json()
        execution_id = execution["id"]
        run_id = execution["run_id"]
        assert run_id

        delivery_store = DeliveryStore(session_db_path(config))

        wait_for(
            lambda: manager._run_store.get_run(run_id) is not None
            and manager._run_store.get_run(run_id).status == "completed"
        )
        wait_for(lambda: len(delivery_store.list_deliveries(run_id=run_id)) == 1)

        executions_response = client.get(f"/api/scheduled-tasks/{task_id}/executions")
        assert executions_response.status_code == 200
        executions = executions_response.json()
        assert executions[0]["id"] == execution_id
        assert executions[0]["status"] == "completed"
        assert executions[0]["delivery_status"] == "delivered"

        detail_response = client.get(f"/api/scheduled-task-executions/{execution_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["task"]["id"] == task_id
        assert detail["execution"]["run_id"] == run_id
        assert detail["run"]["run_metadata"]["source_kind"] == "scheduled_task"
        assert detail["timeline"]
        assert detail["deliveries"][0]["status"] == "delivered"
        assert mock_adapter.sent_payloads[0]["payload"]["target_id"] == "ops-room"
        assert mock_adapter.sent_payloads[0]["payload"]["text"] == "scheduled reply"

        delete_response = client.delete(f"/api/scheduled-tasks/{task_id}")
        assert delete_response.status_code == 200
        assert client.get(f"/api/scheduled-tasks/{task_id}").status_code == 404


@patch("clavi_agent.session.LLMClient")
def test_scheduled_task_api_requires_integration_bound_agent_and_delivery_target(
    mock_llm_class,
    tmp_path: Path,
):
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="scheduled reply", finish_reason="stop")
    )

    config = build_config(tmp_path)
    create_mock_integration(
        config,
        integration_id="missing-agent",
        integration_config={
            "default_chat_id": "ops-room",
            "outbound_retry_backoff_seconds": 0,
        },
    )
    create_mock_integration(
        config,
        integration_id="missing-target",
        integration_config={
            "default_agent_id": "system-default-agent",
            "outbound_retry_backoff_seconds": 0,
        },
    )
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        missing_agent_response = client.post(
            "/api/scheduled-tasks",
            json={
                "name": "Agent missing",
                "cron_expression": "0 8 * * *",
                "prompt": "Prepare the daily report.",
                "integration_id": "missing-agent",
            },
        )
        assert missing_agent_response.status_code == 400
        assert "default_agent_id" in missing_agent_response.json()["detail"]

        missing_target_response = client.post(
            "/api/scheduled-tasks",
            json={
                "name": "Target missing",
                "cron_expression": "0 8 * * *",
                "prompt": "Prepare the daily report.",
                "integration_id": "missing-target",
            },
        )
        assert missing_target_response.status_code == 400
        assert "default_chat_id" in missing_target_response.json()["detail"]


@patch("clavi_agent.session.LLMClient")
def test_scheduled_task_service_dispatches_due_tasks(mock_llm_class, tmp_path: Path):
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="nightly check complete", finish_reason="stop")
    )

    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    service = ScheduledTaskService(manager, poll_interval_seconds=60)

    async def scenario() -> None:
        await manager.initialize()
        created = await service.create_task(
            name="Nightly check",
            cron_expression="*/5 * * * *",
            agent_id="system-default-agent",
            prompt="Run the nightly health check.",
        )
        task_id = created["id"]
        task_record = service._task_store.get_task(task_id)
        assert task_record is not None
        service._task_store.update_task(
            task_record.model_copy(
                update={
                    "next_run_at": "2026-04-14T00:00:00+00:00",
                    "updated_at": "2026-04-14T00:00:00+00:00",
                }
            )
        )

        executions = await service.dispatch_due_tasks(
            now=datetime(2026, 4, 14, 0, 0, 1, tzinfo=timezone.utc)
        )
        assert len(executions) == 1
        execution = executions[0]
        assert execution.trigger_kind == "schedule"
        assert execution.run_id

        async def wait_for_run_completion() -> None:
            deadline = asyncio.get_running_loop().time() + 2
            while asyncio.get_running_loop().time() < deadline:
                run = manager._run_store.get_run(execution.run_id)
                if run is not None and run.status == "completed":
                    return
                await asyncio.sleep(0.05)
            raise AssertionError("Scheduled execution did not complete in time.")

        await wait_for_run_completion()

        detail = await service.get_execution_detail(execution.id)
        refreshed_task = await service.get_task(task_id)
        assert detail["execution"]["status"] == "completed"
        assert detail["run"]["run_metadata"]["scheduled_task_id"] == task_id
        assert refreshed_task["last_execution"]["status"] == "completed"
        assert refreshed_task["next_run_at"] != "2026-04-14T00:00:00+00:00"

    asyncio.run(scenario())


def test_scheduled_task_store_filters_by_account(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    store = ScheduledTaskStore(db_path)

    task_a = store.create_task(
        ScheduledTaskRecord(
            id="task-a",
            account_id="account-a",
            name="Task A",
            cron_expression="0 * * * *",
            agent_id="agent-a",
            prompt="run a",
            created_at="2026-04-15T02:00:00+00:00",
            updated_at="2026-04-15T02:00:00+00:00",
        )
    )
    store.create_task(
        ScheduledTaskRecord(
            id="task-b",
            account_id="account-b",
            name="Task B",
            cron_expression="0 * * * *",
            agent_id="agent-b",
            prompt="run b",
            created_at="2026-04-15T02:01:00+00:00",
            updated_at="2026-04-15T02:01:00+00:00",
        )
    )
    execution = store.create_execution(
        ScheduledTaskExecutionRecord(
            id="exec-a",
            task_id="task-a",
            trigger_kind="manual",
            created_at="2026-04-15T02:02:00+00:00",
            updated_at="2026-04-15T02:02:00+00:00",
        )
    )

    assert task_a.account_id == "account-a"
    assert execution.account_id == "account-a"
    assert [item.id for item in store.list_tasks(account_id="account-a")] == ["task-a"]
    assert store.get_task("task-b", account_id="account-a") is None
    assert store.get_execution("exec-a", account_id="account-a") is not None
    assert store.get_execution("exec-a", account_id="account-b") is None

