import asyncio
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from clavi_agent.account_store import ROOT_ACCOUNT_ID
from clavi_agent.config import AgentConfig, AuthConfig, Config, LLMConfig, RetryConfig, ToolsConfig
from clavi_agent.run_models import ArtifactRecord, RunRecord
from clavi_agent.server import create_app
from clavi_agent.session import SessionManager
from clavi_agent.upload_models import UploadCreatePayload


ROOT_PASSWORD = "RootPass123!"


def build_config(
    tmp_path: Path,
    *,
    root_password: str = ROOT_PASSWORD,
    api_key: str = "test-key",
) -> Config:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    return Config(
        llm=LLMConfig(
            api_key=api_key,
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


def login(client: TestClient, *, username: str = "root", password: str = ROOT_PASSWORD):
    return client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )


def bind_api_config(manager: SessionManager, account_id: str, *, name: str = "default") -> None:
    manager.save_account_api_config(
        account_id,
        name=name,
        api_key=f"{account_id}-key",
        api_base="https://example.com",
        model="test-model",
        provider="openai",
        reasoning_enabled=False,
        activate=True,
    )


def test_auth_requires_login_for_protected_routes(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        me_response = client.get("/api/auth/me")
        sessions_response = client.get("/api/sessions")

    assert me_response.status_code == 401
    assert me_response.json()["detail"] == "Authentication required."
    assert sessions_response.status_code == 401
    assert sessions_response.json()["detail"] == "Authentication required."


def test_auth_login_sets_cookie_and_returns_current_account(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        response = login(client)
        me_response = client.get("/api/auth/me")
        sessions_response = client.get("/api/sessions")

    assert response.status_code == 200
    assert response.json()["account"]["username"] == "root"
    assert config.auth.web_session_cookie_name in client.cookies
    assert me_response.status_code == 200
    assert me_response.json()["account"]["id"] == ROOT_ACCOUNT_ID
    assert sessions_response.status_code == 200
    assert sessions_response.json() == []


def test_setup_mode_exposes_status_and_keeps_login_available(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path, api_key=""))
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        login_response = login(client)
        status_response = client.get("/api/setup/status")

    assert login_response.status_code == 200
    assert status_response.status_code == 200
    assert status_response.json()["setup_required"] is True
    assert status_response.json()["runtime_ready"] is False
    assert status_response.json()["config_count"] == 0


def test_setup_config_endpoint_persists_and_enables_runtime(tmp_path: Path, monkeypatch):
    config_dir = tmp_path / "clavi_agent" / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'api_key: ""',
                'api_base: "https://example.com"',
                'model: "test-model"',
                'provider: "openai"',
                "reasoning_enabled: false",
                "max_steps: 10",
                'workspace_dir: "./workspace"',
                'log_dir: "./logs"',
                'session_store_path: "./sessions.db"',
                "auth:",
                "  auto_seed_root: true",
                '  root_username: "root"',
                '  root_display_name: "Root"',
                f'  root_password: "{ROOT_PASSWORD}"',
                "tools:",
                "  enable_file_tools: false",
                "  enable_bash: false",
                "  enable_note: false",
                "  enable_skills: false",
                "  enable_mcp: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    app = create_app()
    with TestClient(app) as client:
        login_response = login(client)
        before_response = client.get("/api/setup/status")
        save_response = client.post(
            "/api/setup/config",
            json={
                "name": "root-default",
                "api_key": "new-test-key",
                "api_base": "https://runtime.example.com",
                "model": "MiniMax-M2",
                "provider": "anthropic",
                "reasoning_enabled": True,
            },
        )
        after_response = client.get("/api/setup/status")
        config_list_response = client.get("/api/setup/configs")

    assert login_response.status_code == 200
    assert before_response.status_code == 200
    assert before_response.json()["setup_required"] is True
    assert save_response.status_code == 200
    assert save_response.json()["setup_required"] is False
    assert save_response.json()["runtime_ready"] is True
    assert save_response.json()["active_config_name"] == "root-default"
    assert after_response.status_code == 200
    assert after_response.json()["runtime_ready"] is True
    assert config_list_response.status_code == 200
    assert len(config_list_response.json()) == 1
    saved_config = config_list_response.json()[0]
    assert saved_config["name"] == "root-default"
    assert saved_config["provider"] == "anthropic"
    assert saved_config["is_active"] is True
    persisted = Config.from_yaml(config_path, require_api_key=False)
    assert persisted.llm.api_key == ""


def test_auth_login_rejects_wrong_password(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        response = login(client, password="WrongPass123!")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid username or password."


def test_auth_logout_clears_session_cookie(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        login_response = login(client)
        logout_response = client.post("/api/auth/logout")
        me_response = client.get("/api/auth/me")

    assert login_response.status_code == 200
    assert logout_response.status_code == 200
    assert logout_response.json()["status"] == "logged_out"
    assert config.auth.web_session_cookie_name not in client.cookies
    assert me_response.status_code == 401


def test_setup_config_requires_login(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path, api_key=""))
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        response = client.post(
            "/api/setup/config",
            json={
                "name": "default",
                "api_key": "user-key-123",
                "api_base": "https://example.com",
                "model": "test-model",
                "provider": "openai",
                "reasoning_enabled": False,
            },
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Authentication required."


def test_regular_user_can_store_multiple_api_configs_and_switch(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path, api_key=""))
    asyncio.run(manager.initialize())
    store = manager._account_store
    assert store is not None
    store.create_account(
        username="alice",
        password="AlicePass123!",
        display_name="Alice",
    )
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        login_response = login(client, username="alice", password="AlicePass123!")
        first_response = client.post(
            "/api/setup/config",
            json={
                "name": "work",
                "api_key": "alice-work-key",
                "api_base": "https://api.work.example.com",
                "model": "alice-model-1",
                "provider": "openai",
                "reasoning_enabled": True,
            },
        )
        second_response = client.post(
            "/api/setup/config",
            json={
                "name": "backup",
                "api_key": "alice-backup-key",
                "api_base": "https://api.backup.example.com",
                "model": "alice-model-2",
                "provider": "anthropic",
                "reasoning_enabled": False,
                "activate": False,
            },
        )
        listed_response = client.get("/api/setup/configs")
        listed_configs = listed_response.json()
        backup_config = next(item for item in listed_configs if item["name"] == "backup")
        work_config = next(item for item in listed_configs if item["name"] == "work")
        routed_response = client.patch(
            f"/api/setup/configs/{work_config['id']}/routing",
            json={
                "worker_api_config_id": backup_config["id"],
            },
        )
        routed_listed_response = client.get("/api/setup/configs")
        routed_configs = routed_listed_response.json()
        activate_response = client.post(f"/api/setup/configs/{backup_config['id']}/activate")
        after_activate = client.get("/api/setup/status")

    assert login_response.status_code == 200
    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert routed_response.status_code == 200
    assert listed_response.status_code == 200
    assert routed_listed_response.status_code == 200
    assert len(listed_configs) == 2
    updated_work_config = next(item for item in routed_configs if item["name"] == "work")
    assert updated_work_config["is_active"] is True
    assert updated_work_config["llm_routing_policy"]["worker_api_config_id"] == backup_config["id"]
    assert backup_config["is_active"] is False
    assert activate_response.status_code == 200
    assert after_activate.status_code == 200
    assert after_activate.json()["active_config_name"] == "backup"
    assert after_activate.json()["setup_required"] is False


def test_auth_expired_session_is_rejected_and_cookie_is_cleared(tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        login_response = login(client)
        with sqlite3.connect(config.agent.agent_store_path) as conn:
            conn.execute(
                "UPDATE account_web_sessions SET expires_at = ?",
                ("2000-01-01T00:00:00+00:00",),
            )
            conn.commit()
        me_response = client.get("/api/auth/me")

    assert login_response.status_code == 200
    assert me_response.status_code == 401
    assert me_response.json()["detail"] == "Authentication required."
    assert config.auth.web_session_cookie_name not in client.cookies


def test_auth_change_password_updates_credentials(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        login_response = login(client)
        change_response = client.post(
            "/api/auth/change-password",
            json={
                "current_password": ROOT_PASSWORD,
                "new_password": "NewRootPass123!",
            },
        )
        logout_response = client.post("/api/auth/logout")
        old_login_response = login(client, password=ROOT_PASSWORD)
        new_login_response = login(client, password="NewRootPass123!")

    assert login_response.status_code == 200
    assert change_response.status_code == 200
    assert change_response.json()["account"]["username"] == "root"
    assert logout_response.status_code == 200
    assert old_login_response.status_code == 401
    assert new_login_response.status_code == 200


def test_auth_forbids_cross_account_session_access(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path))
    asyncio.run(manager.initialize())
    store = manager._account_store
    assert store is not None

    user_account = store.create_account(
        username="alice",
        password="AlicePass123!",
        display_name="Alice",
    )
    bind_api_config(manager, user_account["id"], name="alice-default")
    root_session_id = asyncio.run(manager.create_session(account_id=ROOT_ACCOUNT_ID))
    user_session_id = asyncio.run(manager.create_session(account_id=user_account["id"]))
    app = create_app(manager, enable_auth=True)

    with TestClient(app) as client:
        login_response = login(client)
        list_response = client.get("/api/sessions")
        root_session_response = client.get(f"/api/sessions/{root_session_id}")
        forbidden_response = client.get(f"/api/sessions/{user_session_id}")

    assert login_response.status_code == 200
    assert list_response.status_code == 200
    assert [item["session_id"] for item in list_response.json()] == [root_session_id]
    assert root_session_response.status_code == 200
    assert root_session_response.json()["session_id"] == root_session_id
    assert forbidden_response.status_code == 403
    assert forbidden_response.json()["detail"] == "Session does not belong to the current account."


def test_auth_forbids_cross_account_agent_and_file_access(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path))
    asyncio.run(manager.initialize())
    store = manager._account_store
    assert store is not None
    assert manager._run_store is not None
    assert manager._agent_store is not None

    user_account = store.create_account(
        username="alice",
        password="AlicePass123!",
        display_name="Alice",
    )
    bind_api_config(manager, user_account["id"], name="alice-default")
    foreign_agent = manager._agent_store.create_agent(
        agent_id="alice-agent",
        name="Alice Agent",
        system_prompt="You are Alice's agent.",
        account_id=user_account["id"],
    )
    user_session_id = asyncio.run(manager.create_session(account_id=user_account["id"]))
    user_session = manager.get_session_info(
        user_session_id,
        account_id=user_account["id"],
        strict=True,
    )
    assert user_session is not None
    workspace_dir = Path(user_session["workspace_dir"]).resolve()

    uploads = manager.create_session_uploads(
        user_session_id,
        [
            UploadCreatePayload(
                original_name="alice.md",
                content_bytes=b"# Alice\n",
                mime_type="text/markdown",
            )
        ],
        account_id=user_account["id"],
    )
    artifact_path = workspace_dir / "exports" / "alice-report.md"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("# Alice Report\n", encoding="utf-8")

    run = manager._run_store.create_run(
        RunRecord(
            id="alice-run-1",
            session_id=user_session_id,
            account_id=user_account["id"],
            agent_template_id="system-default-agent",
            agent_template_snapshot=manager._agent_store.snapshot_agent_template(
                "system-default-agent"
            ),
            status="completed",
            goal="alice artifact",
            created_at="2026-04-15T08:00:00+00:00",
            started_at="2026-04-15T08:00:00+00:00",
            finished_at="2026-04-15T08:00:01+00:00",
            current_step_index=1,
        )
    )
    manager._run_store.create_artifact(
        ArtifactRecord(
            id="alice-artifact-1",
            run_id=run.id,
            artifact_type="workspace_file",
            uri="exports/alice-report.md",
            display_name="alice-report.md",
            role="final_deliverable",
            format="md",
            mime_type="text/markdown",
            size_bytes=artifact_path.stat().st_size,
            source="agent_generated",
            is_final=True,
            preview_kind="markdown",
            summary="Alice report",
            created_at="2026-04-15T08:00:01+00:00",
        )
    )

    app = create_app(manager, enable_auth=True)
    with TestClient(app) as client:
        login_response = login(client)
        delete_agent_response = client.delete(f"/api/agents/{foreign_agent['id']}")
        upload_download_response = client.get(f"/api/uploads/{uploads[0].id}")
        artifact_download_response = client.get("/api/artifacts/alice-artifact-1")

    assert login_response.status_code == 200
    assert delete_agent_response.status_code == 403
    assert delete_agent_response.json()["detail"] == (
        "Agent template does not belong to the current account."
    )
    assert upload_download_response.status_code == 403
    assert upload_download_response.json()["detail"] == (
        "Upload does not belong to the current account."
    )
    assert artifact_download_response.status_code == 403
    assert artifact_download_response.json()["detail"] == (
        "Artifact does not belong to the current account."
    )


def test_non_root_session_workspace_is_account_scoped_and_cannot_be_overridden(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path))
    asyncio.run(manager.initialize())
    store = manager._account_store
    assert store is not None

    user_account = store.create_account(
        username="alice",
        password="AlicePass123!",
        display_name="Alice",
    )
    bind_api_config(manager, user_account["id"], name="alice-default")
    app = create_app(manager, enable_auth=True)

    custom_workspace = tmp_path / "escape-workspace"
    with TestClient(app) as client:
        login_response = login(client, username="alice", password="AlicePass123!")
        create_response = client.post("/api/sessions", json={})
        override_response = client.post(
            "/api/sessions",
            json={"workspace_dir": str(custom_workspace)},
        )

    assert login_response.status_code == 200
    assert create_response.status_code == 200
    workspace_dir = Path(create_response.json()["workspace_dir"]).resolve()
    expected_sessions_root = (
        tmp_path / "workspace" / "accounts" / user_account["id"] / "sessions"
    ).resolve()
    assert workspace_dir.parent == expected_sessions_root
    assert override_response.status_code == 403
    assert override_response.json()["detail"] == "普通账号不允许自定义工作区路径。"


def test_root_can_still_create_session_with_custom_workspace(tmp_path: Path):
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager, enable_auth=True)
    custom_workspace = (tmp_path / "root-custom-workspace").resolve()

    with TestClient(app) as client:
        login_response = login(client)
        create_response = client.post(
            "/api/sessions",
            json={"workspace_dir": str(custom_workspace)},
        )

    assert login_response.status_code == 200
    assert create_response.status_code == 200
    assert Path(create_response.json()["workspace_dir"]).resolve() == custom_workspace

