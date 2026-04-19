import sqlite3
from pathlib import Path

import pytest

from clavi_agent.account_store import AccountStore, ROOT_ACCOUNT_ID
from clavi_agent.config import (
    AgentConfig,
    AuthConfig,
    Config,
    LLMConfig,
    RetryConfig,
    ToolsConfig,
)
from clavi_agent.session import SessionManager
from clavi_agent.sqlite_schema import AGENT_DB_SCOPE, CURRENT_AGENT_DB_VERSION


def build_config(tmp_path: Path, *, root_password: str | None = None) -> Config:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        llm=LLMConfig(
            api_key="test-key",
            api_base="https://example.com",
            model="test-model",
            provider="openai",
            reasoning_enabled=False,
            retry=RetryConfig(enabled=False),
        ),
        agent=AgentConfig(
            workspace_dir=str(workspace_dir),
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
        auth=AuthConfig(
            auto_seed_root=True,
            root_username="root",
            root_display_name="Root",
            root_password=root_password,
        ),
    )


def test_account_store_crud_and_password_verification(tmp_path: Path):
    store = AccountStore(tmp_path / "accounts.db", auto_seed_root=False)

    account = store.create_account(
        username="Alice",
        password="Secret123!",
        display_name="Alice",
    )

    assert account["username"] == "alice"
    assert account["display_name"] == "Alice"
    assert account["status"] == "active"

    credential = store.get_password_credential(account["id"])
    assert credential is not None
    assert credential["password_algo"] == "argon2id"
    assert credential["password_hash"] != "Secret123!"
    assert credential["password_hash"].startswith("$argon2id$")

    authenticated = store.authenticate("Alice", "Secret123!")
    assert authenticated is not None
    assert authenticated.id == account["id"]
    assert store.authenticate("Alice", "WrongPassword!") is None

    updated = store.update_account(account["id"], status="disabled")
    assert updated is not None
    assert updated["status"] == "disabled"
    assert store.authenticate("Alice", "Secret123!") is None


def test_account_store_web_session_lifecycle(tmp_path: Path):
    store = AccountStore(tmp_path / "accounts.db", auto_seed_root=False)
    account = store.create_account(
        username="bob",
        password="Secret123!",
        display_name="Bob",
    )

    session, raw_token = store.create_web_session(
        account["id"],
        session_token="raw-session-token",
        expires_at="2099-01-01T00:00:00+00:00",
        user_agent="pytest",
        ip="127.0.0.1",
    )

    assert raw_token == "raw-session-token"
    assert session["session_token_hash"] != raw_token
    assert session["user_agent"] == "pytest"
    assert session["ip"] == "127.0.0.1"

    authenticated = store.get_authenticated_session(
        raw_token,
        now="2026-04-15T00:00:00+00:00",
    )
    assert authenticated is not None
    assert authenticated.account.id == account["id"]
    assert authenticated.web_session.id == session["id"]

    refreshed = store.touch_web_session(
        session["id"],
        last_seen_at="2026-04-15T08:00:00+00:00",
    )
    assert refreshed is not None
    assert refreshed["last_seen_at"] == "2026-04-15T08:00:00+00:00"

    deleted = store.delete_expired_web_sessions(
        now="2099-01-02T00:00:00+00:00",
    )
    assert deleted == 1
    assert store.get_authenticated_session(
        raw_token,
        now="2099-01-02T00:00:00+00:00",
    ) is None


def test_account_store_api_config_lifecycle(tmp_path: Path):
    store = AccountStore(tmp_path / "accounts.db", auto_seed_root=False)
    account = store.create_account(
        username="alice",
        password="Secret123!",
        display_name="Alice",
    )

    first = store.upsert_api_config(
        account["id"],
        name="work",
        api_key="key-work-123",
        provider="openai",
        api_base="https://example.com",
        model="gpt-test",
        reasoning_enabled=True,
        llm_routing_policy={
            "worker_profile": {
                "model": "gpt-worker",
                "reasoning_enabled": False,
            }
        },
        activate=True,
    )
    second = store.upsert_api_config(
        account["id"],
        name="backup",
        api_key="key-backup-456",
        provider="anthropic",
        api_base="https://api.minimax.io",
        model="MiniMax-M2",
        reasoning_enabled=False,
        llm_routing_policy={
            "planner_profile": {
                "model": "MiniMax-Planner",
            }
        },
        activate=False,
    )
    updated_first = store.upsert_api_config(
        account["id"],
        name="work",
        api_key="key-work-123",
        provider="openai",
        api_base="https://example.com",
        model="gpt-test",
        reasoning_enabled=True,
        llm_routing_policy={
            "worker_api_config_id": second.id,
            "worker_profile": {
                "model": "gpt-worker",
                "reasoning_enabled": False,
            },
        },
        activate=True,
    )

    listed = store.list_api_config_records(account["id"])
    assert [item.name for item in listed] == ["work", "backup"]
    assert listed[0].llm_routing_policy.worker_profile is not None
    assert listed[0].llm_routing_policy.worker_profile.model == "gpt-worker"
    assert listed[0].llm_routing_policy.worker_api_config_id == second.id
    assert listed[1].llm_routing_policy.planner_profile is not None
    assert listed[1].llm_routing_policy.planner_profile.model == "MiniMax-Planner"
    assert updated_first.id == first.id
    assert store.get_active_api_config_record(account["id"]).id == first.id

    activated = store.activate_api_config(account["id"], second.id)
    assert activated.id == second.id
    assert activated.is_active is True
    assert store.get_active_api_config_record(account["id"]).id == second.id

    touched = store.touch_api_config_last_used(account["id"], second.id)
    assert touched is not None
    assert touched.last_used_at is not None

    assert store.delete_api_config(account["id"], first.id) is True
    remaining = store.list_api_config_records(account["id"])
    assert [item.id for item in remaining] == [second.id]


def test_account_store_auto_seeds_root_account(tmp_path: Path):
    db_path = tmp_path / "accounts.db"
    store = AccountStore(
        db_path,
        root_password="RootPass123!",
    )

    root_account = store.get_root_account_record()
    assert root_account is not None
    assert root_account.id == ROOT_ACCOUNT_ID
    assert root_account.username == "root"
    assert root_account.is_root is True
    assert store.authenticate("root", "RootPass123!") is not None

    reopened = AccountStore(
        db_path,
        root_password="OtherPass123!",
    )
    assert reopened.authenticate("root", "RootPass123!") is not None
    assert reopened.authenticate("root", "OtherPass123!") is None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (AGENT_DB_SCOPE,),
        ).fetchone()

    assert {
        "accounts",
        "account_api_configs",
        "account_password_credentials",
        "account_web_sessions",
        "user_profiles",
        "user_memory_entries",
    }.issubset(
        tables
    )
    assert version_row is not None
    assert version_row["version"] == CURRENT_AGENT_DB_VERSION


@pytest.mark.asyncio
async def test_session_manager_initialization_auto_seeds_root(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path, root_password="ManagerRoot123!"))

    await manager.initialize()

    assert manager._account_store is not None
    root_account = manager._account_store.get_root_account_record()
    assert root_account is not None
    assert root_account.username == "root"
    assert manager._account_store.authenticate("root", "ManagerRoot123!") is not None

    await manager.cleanup()

