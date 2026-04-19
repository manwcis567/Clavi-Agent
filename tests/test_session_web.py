"""Persistent session manager and API tests."""

import asyncio
import json
import subprocess
import time
import threading
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from clavi_agent.account_constants import ROOT_ACCOUNT_ID
from clavi_agent.account_store import AccountStore
from clavi_agent.agent_template_models import AgentTemplateSnapshot
from clavi_agent.config import (
    AgentConfig,
    Config,
    FeatureFlagsConfig,
    LLMConfig,
    MemoryProviderConfig,
    RetryConfig,
    ToolsConfig,
)
from clavi_agent.agent import Agent
from clavi_agent.run_models import (
    ApprovalRequestRecord,
    ArtifactRecord,
    RunDeliverableManifest,
    RunDeliverableRef,
    RunRecord,
    RunStepRecord,
    TraceEventRecord,
)
from clavi_agent.run_store import RunStore
from clavi_agent.runtime_tools import resolve_clawhub_command_prefix as resolve_runtime_clawhub_command_prefix
from clavi_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from clavi_agent.session_store import SessionStore
from clavi_agent.server import _parse_clawhub_search_output, _resolve_clawhub_command_prefix, create_app
from clavi_agent.session import SessionManager
from clavi_agent.tools.base import Tool, ToolResult
from clavi_agent.user_memory_store import UserMemoryStore


def build_config(
    tmp_path: Path,
    *,
    enable_skills: bool = False,
    skills_dir: str | None = None,
    run_timeout_seconds: int | None = None,
    max_concurrent_runs: int = 4,
    enable_durable_runs: bool = True,
    enable_run_trace: bool = True,
    enable_approval_flow: bool = True,
    enable_supervisor_mode: bool = True,
    enable_worker_model_routing: bool = True,
    enable_compact_prompt_memory: bool = True,
    enable_session_retrieval: bool = True,
    enable_learned_workflow_generation: bool = True,
    enable_external_memory_providers: bool = True,
    memory_provider: str = "local",
    allow_memory_fallback: bool = True,
    inject_memories: bool = True,
) -> Config:
    """Build a lightweight test config with persistence enabled."""
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
            max_concurrent_runs=max_concurrent_runs,
            run_timeout_seconds=run_timeout_seconds,
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
            enable_skills=enable_skills,
            skills_dir=skills_dir or "./skills",
            enable_mcp=False,
        ),
        memory_provider=MemoryProviderConfig(
            provider=memory_provider,
            allow_fallback_to_local=allow_memory_fallback,
            inject_memories=inject_memories,
        ),
        feature_flags=FeatureFlagsConfig(
            enable_durable_runs=enable_durable_runs,
            enable_run_trace=enable_run_trace,
            enable_approval_flow=enable_approval_flow,
            enable_supervisor_mode=enable_supervisor_mode,
            enable_worker_model_routing=enable_worker_model_routing,
            enable_compact_prompt_memory=enable_compact_prompt_memory,
            enable_session_retrieval=enable_session_retrieval,
            enable_learned_workflow_generation=enable_learned_workflow_generation,
            enable_external_memory_providers=enable_external_memory_providers,
        ),
    )


async def collect_events(generator):
    """Collect async generator results into a list."""
    events = []
    async for event in generator:
        events.append(event)
    return events


@patch("clavi_agent.session.LLMClient")
async def test_session_manager_restores_from_store(mock_llm_class, tmp_path: Path):
    """A fresh SessionManager instance should restore and continue a stored session."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(content="first reply", finish_reason="stop"),
            LLMResponse(content="second reply", finish_reason="stop"),
        ]
    )

    config = build_config(tmp_path)

    manager = SessionManager(config=config)
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    first_events = await collect_events(manager.chat(session_id, "first question"))

    assert any(event["type"] == "content" for event in first_events)
    assert manager.get_session_info(session_id)["title"] == "first question"

    restored_manager = SessionManager(config=config)
    await restored_manager.initialize()
    restored_agent = restored_manager.restore_session(session_id)

    assert restored_agent is not None
    assert [message.role for message in restored_agent.messages] == ["system", "user", "assistant"]
    assert restored_agent.messages[-1].content == "first reply"

    second_events = await collect_events(restored_manager.chat(session_id, "follow up"))

    assert any(event["type"] == "content" for event in second_events)
    session_messages = restored_manager.get_session_messages(session_id)
    assert [message.role for message in session_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]


@patch("clavi_agent.session.LLMClient")
def test_user_profile_endpoint_returns_structured_profile(mock_llm_class, tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)
    AccountStore(tmp_path / "agents.db", auto_seed_root=True)
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        ROOT_ACCOUNT_ID,
        profile={
            "preferred_language": "zh-CN",
            "response_length": "concise",
        },
        summary="偏好中文，回复简洁。",
        profile_source="explicit",
        profile_confidence=1.0,
        writer_type="tool",
        writer_id="record_note",
    )

    with TestClient(app) as client:
        response = client.get("/api/user-profile")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["preferred_language"] == "zh-CN"
    assert payload["normalized_profile"]["response_length"] == "concise"
    assert payload["field_meta"]["preferred_language"]["source"] == "explicit"


@patch("clavi_agent.session.LLMClient")
def test_memory_provider_health_endpoint_reports_mcp_fallback(mock_llm_class, tmp_path: Path):
    config = build_config(tmp_path, memory_provider="mcp")
    manager = SessionManager(config=config)
    app = create_app(manager)

    mcp_config_path = tmp_path / "mcp-memory.json"
    mcp_config_path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "memory": {
                        "description": "Legacy memory server",
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-memory"],
                        "disabled": True,
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    original_find = Config.find_config_file

    def fake_find_config_file(filename: str):
        if filename == config.tools.mcp_config_path:
            return mcp_config_path
        return original_find(filename)

    with patch("clavi_agent.session.Config.find_config_file", side_effect=fake_find_config_file):
        with TestClient(app) as client:
            response = client.get("/api/memory-provider/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured_provider"] == "mcp"
    assert payload["active_provider"] == "local"
    assert payload["status"] == "degraded"
    assert payload["fallback_active"] is True
    assert payload["metadata"]["server_name"] == "memory"
    assert payload["metadata"]["server_disabled"] is True


@patch("clavi_agent.session.LLMClient")
def test_feature_flag_can_disable_external_memory_provider_rollout(
    mock_llm_class,
    tmp_path: Path,
):
    config = build_config(
        tmp_path,
        memory_provider="mcp",
        enable_external_memory_providers=False,
    )
    manager = SessionManager(config=config)
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.get("/api/memory-provider/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured_provider"] == "mcp"
    assert payload["active_provider"] == "local"
    assert payload["status"] == "degraded"
    assert payload["fallback_active"] is True
    assert payload["metadata"]["reason"] == "feature_flag_disabled"
    assert payload["metadata"]["feature_flag"] == "enable_external_memory_providers"


@patch("clavi_agent.session.LLMClient")
def test_disabled_memory_provider_turns_off_user_memory_surfaces(mock_llm_class, tmp_path: Path):
    config = build_config(
        tmp_path,
        memory_provider="disabled",
        inject_memories=False,
    )
    manager = SessionManager(config=config)
    app = create_app(manager)
    AccountStore(tmp_path / "agents.db", auto_seed_root=True)
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        ROOT_ACCOUNT_ID,
        profile={"preferred_language": "zh-CN"},
        summary="偏好中文。",
        writer_type="tool",
        writer_id="record_note",
    )
    store.create_memory_entry(
        user_id=ROOT_ACCOUNT_ID,
        memory_type="constraint",
        content="所有文本文件统一使用 UTF-8。",
        summary="统一使用 UTF-8。",
        confidence=0.99,
        writer_type="tool",
        writer_id="record_note",
    )

    with TestClient(app) as client:
        health_response = client.get("/api/memory-provider/health")
        profile_response = client.get("/api/user-profile")
        memory_response = client.get("/api/user-memory")

    assert health_response.status_code == 200
    health_payload = health_response.json()
    assert health_payload["configured_provider"] == "disabled"
    assert health_payload["active_provider"] == "disabled"
    assert health_payload["status"] == "disabled"
    assert profile_response.status_code == 404
    assert memory_response.status_code == 200
    assert memory_response.json() == []


@patch("clavi_agent.session.LLMClient")
def test_user_memory_inspection_endpoints_return_entries_and_audit(mock_llm_class, tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)
    AccountStore(tmp_path / "agents.db", auto_seed_root=True)
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        ROOT_ACCOUNT_ID,
        profile={"preferred_language": "zh-CN"},
        summary="偏好中文。",
        profile_source="explicit",
        profile_confidence=1.0,
        writer_type="tool",
        writer_id="record_note",
        source_session_id="session-profile",
        source_run_id="run-profile",
    )
    preference_entry = store.create_memory_entry(
        user_id=ROOT_ACCOUNT_ID,
        memory_type="preference",
        content="用户希望界面中的新增功能说明使用中文。",
        summary="新增功能说明使用中文。",
        source_session_id="session-a",
        source_run_id="run-a",
        confidence=0.98,
        writer_type="tool",
        writer_id="record_note",
    )
    workflow_entry = store.create_memory_entry(
        user_id=ROOT_ACCOUNT_ID,
        memory_type="workflow_fact",
        content="生成或修改文本文件前，需要先确认使用 UTF-8 编码。",
        summary="文本文件统一 UTF-8。",
        source_session_id="session-b",
        source_run_id="run-b",
        confidence=0.9,
        writer_type="tool",
        writer_id="record_note",
    )
    store.update_memory_entry(
        preference_entry["id"],
        user_id=ROOT_ACCOUNT_ID,
        summary="功能说明默认中文。",
        writer_type="user",
        writer_id="manual-review",
        source_session_id="session-a",
        source_run_id="run-c",
        confidence=1.0,
    )

    with TestClient(app) as client:
        list_response = client.get(
            "/api/user-memory",
            params={"memory_type": "workflow_fact", "limit": 5},
        )
        search_response = client.get(
            "/api/user-memory",
            params={"query": "UTF-8", "limit": 5},
        )
        detail_response = client.get(f"/api/user-memory/{preference_entry['id']}")
        memory_audit_response = client.get(
            "/api/user-memory/audit",
            params={
                "target_scope": "user_memory",
                "target_id": preference_entry["id"],
                "limit": 10,
            },
        )
        profile_audit_response = client.get(
            "/api/user-memory/audit",
            params={
                "target_scope": "user_profile",
                "target_id": ROOT_ACCOUNT_ID,
                "limit": 10,
            },
        )
        profile_patch_response = client.patch(
            "/api/user-profile",
            json={
                "profile_updates": {"preferred_language": "en-US"},
                "remove_fields": [],
                "profile_source": "explicit",
                "profile_confidence": 1.0,
            },
        )
        memory_patch_response = client.patch(
            f"/api/user-memory/{workflow_entry['id']}",
            json={
                "summary": "文本和代码文件统一 UTF-8。",
                "confidence": 0.96,
            },
        )
        memory_delete_response = client.delete(f"/api/user-memory/{preference_entry['id']}")
        deleted_detail_response = client.get(f"/api/user-memory/{preference_entry['id']}")
        refreshed_memory_list = client.get("/api/user-memory", params={"limit": 10})
        refreshed_profile_response = client.get("/api/user-profile")
        deleted_memory_audit = client.get(
            "/api/user-memory/audit",
            params={
                "target_scope": "user_memory",
                "target_id": preference_entry["id"],
                "limit": 10,
            },
        )

    assert list_response.status_code == 200
    list_payload = list_response.json()
    assert len(list_payload) == 1
    assert list_payload[0]["id"] == workflow_entry["id"]
    assert list_payload[0]["memory_type"] == "workflow_fact"

    assert search_response.status_code == 200
    search_payload = search_response.json()
    assert len(search_payload) == 1
    assert search_payload[0]["id"] == workflow_entry["id"]

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["id"] == preference_entry["id"]
    assert detail_payload["summary"] == "功能说明默认中文。"
    assert detail_payload["writer_id"] == "manual-review"

    assert memory_audit_response.status_code == 200
    memory_audit_payload = memory_audit_response.json()
    assert [event["action"] for event in memory_audit_payload[:2]] == [
        "memory_update",
        "memory_create",
    ]

    assert profile_audit_response.status_code == 200
    profile_audit_payload = profile_audit_response.json()
    assert profile_audit_payload[0]["target_scope"] == "user_profile"
    assert profile_audit_payload[0]["action"] in {"profile_create", "profile_upsert"}

    assert profile_patch_response.status_code == 200
    assert profile_patch_response.json()["profile"]["preferred_language"] == "en-US"

    assert memory_patch_response.status_code == 200
    assert memory_patch_response.json()["summary"] == "文本和代码文件统一 UTF-8。"
    assert memory_patch_response.json()["confidence"] == 0.96
    assert memory_patch_response.json()["writer_id"] == "web_ui"

    assert memory_delete_response.status_code == 200
    assert memory_delete_response.json() == {
        "status": "deleted",
        "target_id": preference_entry["id"],
    }
    assert deleted_detail_response.status_code == 404

    assert refreshed_memory_list.status_code == 200
    refreshed_memory_payload = refreshed_memory_list.json()
    assert {item["id"] for item in refreshed_memory_payload} == {workflow_entry["id"]}

    assert refreshed_profile_response.status_code == 200
    assert refreshed_profile_response.json()["field_meta"]["preferred_language"]["writer_id"] == "web_ui"

    assert deleted_memory_audit.status_code == 200
    assert deleted_memory_audit.json()[0]["action"] == "memory_delete"


@patch("clavi_agent.session.LLMClient")
def test_session_history_endpoint_supports_filters(mock_llm_class, tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    app = create_app(manager)
    AccountStore(tmp_path / "agents.db", auto_seed_root=True)

    session_store = SessionStore(tmp_path / "sessions.db")
    run_store = RunStore(tmp_path / "sessions.db")
    session_store.create_session(
        session_id="session-a",
        workspace_dir=str(tmp_path / "workspace-a"),
        agent_id="agent-alpha",
        messages=[
            Message(role="system", content="system prompt"),
            Message(role="assistant", content="已记录 UTF-8 编码决策。"),
        ],
        account_id=ROOT_ACCOUNT_ID,
    )
    session_store.create_session(
        session_id="session-b",
        workspace_dir=str(tmp_path / "workspace-b"),
        agent_id="agent-beta",
        messages=[
            Message(role="system", content="system prompt"),
            Message(role="assistant", content="另一个会话也提到了 UTF-8。"),
        ],
        account_id=ROOT_ACCOUNT_ID,
    )

    snapshot_alpha = AgentTemplateSnapshot(
        template_id="agent-alpha",
        template_version=1,
        captured_at="2026-04-16T11:30:00+00:00",
        name="Alpha",
        system_prompt="You are alpha.",
    )
    run_store.create_run(
        RunRecord(
            id="run-alpha",
            session_id="session-a",
            account_id=ROOT_ACCOUNT_ID,
            agent_template_id="agent-alpha",
            agent_template_snapshot=snapshot_alpha,
            status="completed",
            goal="复盘 UTF-8 决策",
            created_at="2026-04-16T11:30:00+00:00",
            started_at="2026-04-16T11:30:01+00:00",
            finished_at="2026-04-16T11:30:02+00:00",
        )
    )
    run_store.create_step(
        RunStepRecord(
            id="step-alpha",
            run_id="run-alpha",
            sequence=1,
            step_type="completion",
            status="completed",
            title="Run completed",
            output_summary="上周已经确认文本文件统一使用 UTF-8。",
            started_at="2026-04-16T11:30:01+00:00",
            finished_at="2026-04-16T11:30:02+00:00",
        )
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/session-history",
            params={
                "query": "UTF-8",
                "agent_id": "agent-alpha",
                "source_type": "run_completion",
                "date_from": "2026-04-16T00:00:00+00:00",
                "limit": 5,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["session_id"] == "session-a"
    assert payload[0]["run_id"] == "run-alpha"
    assert payload[0]["source_type"] == "run_completion"


@patch("clavi_agent.session.LLMClient")
def test_learned_workflow_apis_review_and_install_candidate(mock_llm_class, tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())
    app = create_app(manager)

    session_store = SessionStore(tmp_path / "sessions.db")
    run_store = RunStore(tmp_path / "sessions.db")
    session_store.create_session(
        session_id="session-a",
        workspace_dir=str(tmp_path / "workspace-a"),
        agent_id="agent-alpha",
        messages=[Message(role="system", content="system prompt")],
        account_id=ROOT_ACCOUNT_ID,
    )
    snapshot = AgentTemplateSnapshot(
        template_id="agent-alpha",
        template_version=1,
        captured_at="2026-04-16T11:30:00+00:00",
        name="Alpha",
        system_prompt="You are alpha.",
    )
    run_store.create_run(
        RunRecord(
            id="run-alpha",
            session_id="session-a",
            account_id=ROOT_ACCOUNT_ID,
            agent_template_id="agent-alpha",
            agent_template_snapshot=snapshot,
            status="completed",
            goal="复用报告产出流程",
            created_at="2026-04-16T11:30:00+00:00",
            started_at="2026-04-16T11:30:01+00:00",
            finished_at="2026-04-16T11:30:02+00:00",
        )
    )
    manager._learned_workflow_store.upsert_candidate_for_run(
        account_id=ROOT_ACCOUNT_ID,
        run_id="run-alpha",
        session_id="session-a",
        agent_template_id="agent-alpha",
        title="复用报告产出流程",
        summary="统一写报告并导出产物。",
        description="适用于固定格式报告的生成任务。",
        signal_types=["successful_complex_run"],
        source_run_ids=["run-alpha"],
        tool_names=["write_file"],
        step_titles=["write_file", "Run completed"],
        artifact_ids=["artifact-1"],
        suggested_skill_name="report-workflow",
        generated_skill_markdown=(
            "---\n"
            "name: report-workflow\n"
            "description: 统一写报告并导出产物。\n"
            "---\n\n"
            "# 报告工作流\n"
        ),
    )
    agent = manager._agent_store.create_agent(
        name="Reusable Agent",
        description="Owns learned workflows",
        system_prompt="Use reusable workflows.",
        tools=[],
    )

    with TestClient(app) as client:
        list_response = client.get("/api/learned-workflows")
        assert list_response.status_code == 200
        candidate_id = list_response.json()[0]["id"]

        install_before_review = client.post(
            f"/api/learned-workflows/{candidate_id}/install",
            json={"agent_id": agent["id"], "skill_name": "report-workflow"},
        )
        assert install_before_review.status_code == 400

        approve_response = client.post(
            f"/api/learned-workflows/{candidate_id}/approve",
            json={"review_notes": "可复用"},
        )
        assert approve_response.status_code == 200
        assert approve_response.json()["status"] == "approved"

        install_response = client.post(
            f"/api/learned-workflows/{candidate_id}/install",
            json={"agent_id": agent["id"], "skill_name": "report-workflow"},
        )
        assert install_response.status_code == 200
        payload = install_response.json()
        assert payload["status"] == "installed"
        assert payload["installed_agent_id"] == agent["id"]

    skill_file = (
        manager._agent_store.get_agent_skills_dir(agent["id"]) / "report-workflow" / "SKILL.md"
    )
    assert skill_file.exists()
    assert "report-workflow" in skill_file.read_text(encoding="utf-8")
    refreshed_agent = manager._agent_store.get_agent_template(agent["id"])
    assert refreshed_agent is not None
    assert refreshed_agent["skills"][0]["name"] == "report-workflow"


@patch("clavi_agent.session.LLMClient")
def test_skill_improvement_apis_review_and_apply_proposal(mock_llm_class, tmp_path: Path):
    config = build_config(tmp_path)
    manager = SessionManager(config=config)
    asyncio.run(manager.initialize())
    app = create_app(manager)

    agent = manager._agent_store.create_agent(
        name="Skill Maintainer",
        description="Owns installed skills",
        system_prompt="Maintain installed skills safely.",
        tools=[],
    )
    skill_dir = manager._agent_store.get_agent_skills_dir(agent["id"]) / "report-skill"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\n"
        "name: report-skill\n"
        "description: Generate reports.\n"
        "version: 1\n"
        "---\n\n"
        "# Report Skill\n",
        encoding="utf-8",
    )
    manager._agent_store.refresh_agent_skills_from_directory(agent["id"])

    session_store = SessionStore(tmp_path / "sessions.db")
    run_store = RunStore(tmp_path / "sessions.db")
    session_store.create_session(
        session_id="session-a",
        workspace_dir=str(tmp_path / "workspace-a"),
        agent_id=agent["id"],
        messages=[Message(role="system", content="system prompt")],
        account_id=ROOT_ACCOUNT_ID,
    )
    snapshot = AgentTemplateSnapshot(
        template_id=agent["id"],
        template_version=1,
        captured_at="2026-04-16T11:30:00+00:00",
        name="Skill Maintainer",
        system_prompt="Maintain installed skills safely.",
    )
    run_store.create_run(
        RunRecord(
            id="run-alpha",
            session_id="session-a",
            account_id=ROOT_ACCOUNT_ID,
            agent_template_id=agent["id"],
            agent_template_snapshot=snapshot,
            status="completed",
            goal="改进报告技能",
            created_at="2026-04-16T11:30:00+00:00",
            started_at="2026-04-16T11:30:01+00:00",
            finished_at="2026-04-16T11:30:02+00:00",
        )
    )
    manager._skill_improvement_store.upsert_proposal_for_skill(
        account_id=ROOT_ACCOUNT_ID,
        run_id="run-alpha",
        session_id="session-a",
        agent_template_id=agent["id"],
        skill_name="report-skill",
        target_skill_path=str(skill_file.resolve()),
        title="Improve skill: report-skill",
        summary="Add a pre-export checklist.",
        signal_types=["manual_successful_refinement"],
        source_run_ids=["run-alpha"],
        base_version=1,
        proposed_version=2,
        current_skill_markdown=skill_file.read_text(encoding="utf-8"),
        proposed_skill_markdown=(
            "---\n"
            "name: report-skill\n"
            "description: Generate reports.\n"
            "version: 2\n"
            "---\n\n"
            "# Report Skill\n\n"
            "## Maintainer Update v2\n\n"
            "Add a pre-export checklist.\n"
        ),
        changelog_entry="v2 (2026-04-16): Add a pre-export checklist.",
    )

    with TestClient(app) as client:
        list_response = client.get("/api/skill-improvements")
        assert list_response.status_code == 200
        proposal_id = list_response.json()[0]["id"]

        apply_before_review = client.post(f"/api/skill-improvements/{proposal_id}/apply")
        assert apply_before_review.status_code == 400

        approve_response = client.post(
            f"/api/skill-improvements/{proposal_id}/approve",
            json={"review_notes": "可以应用"},
        )
        assert approve_response.status_code == 200
        assert approve_response.json()["status"] == "approved"

        apply_response = client.post(f"/api/skill-improvements/{proposal_id}/apply")
        assert apply_response.status_code == 200
        payload = apply_response.json()
        assert payload["status"] == "applied"
        assert payload["applied_skill_path"] is not None

    applied_markdown = skill_file.read_text(encoding="utf-8")
    assert "version: 2" in applied_markdown
    assert "## Maintainer Update v2" in applied_markdown


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_session_manager_can_interrupt_running_chat(mock_llm_class, tmp_path: Path):
    """Interrupting a running session should stop the current stream cleanly."""
    mock_llm = mock_llm_class.return_value
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def fake_generate_stream(messages, tools):  # noqa: ANN001
        started.set()
        yield {"type": "content_delta", "data": {"delta": "Hel"}}
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    mock_llm.generate_stream = fake_generate_stream
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="fallback should not be used", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))

    async def collect_chat_events():
        events = []
        async for event in manager.chat(session_id, "please start working"):
            events.append(event)
        return events

    chat_task = asyncio.create_task(collect_chat_events())
    await asyncio.wait_for(started.wait(), timeout=1)

    interrupted = await manager.interrupt_session(session_id)
    events = await asyncio.wait_for(chat_task, timeout=1)

    assert interrupted is True
    assert cancelled.is_set()
    assert any(event["type"] == "content_delta" for event in events)
    assert any(event["type"] == "interrupted" for event in events)
    assert manager.is_session_running(session_id) is False
    assert [message.role for message in manager.get_session_messages(session_id)] == ["system", "user"]


class BlockingTool(Tool):
    """Tool stub that blocks until the run gets cancelled."""

    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    @property
    def name(self) -> str:
        return "blocking_tool"

    @property
    def description(self) -> str:
        return "Blocks until interrupted."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, *args, **kwargs) -> ToolResult:  # noqa: ANN002, ANN003
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return ToolResult(success=True, content="unexpected completion")


class GateTool(Tool):
    """Tool stub that waits until the test explicitly releases it."""

    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    @property
    def name(self) -> str:
        return "gate_tool"

    @property
    def description(self) -> str:
        return "Waits for an external release signal."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, *args, **kwargs) -> ToolResult:  # noqa: ANN002, ANN003
        self.started.set()
        await asyncio.to_thread(self.release.wait)
        return ToolResult(success=True, content="released")


class SimpleWriteTool(Tool):
    """Write-like tool used to exercise approval-gated API flows."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Writes a file."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        target = self.workspace_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(success=True, content=f"wrote {target.name}")


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_disabling_approval_flow_executes_approval_required_tool_without_waiting(
    mock_llm_class,
    tmp_path: Path,
):
    """When approval rollout is disabled, approval-marked tools should not block the run."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Write the file first.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/approval-disabled.md",
                                "content": "approval disabled",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="write finished", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path, enable_approval_flow=False))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Approval Disabled Agent",
        description="Requires approval when the feature is enabled.",
        system_prompt="Try to write files when asked.",
        tools=["WriteTool"],
        approval_policy={
            "mode": "default",
            "require_approval_tools": ["write_file"],
        },
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-approval-disabled"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=6,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    events = await collect_events(manager.chat(session_id, "write despite disabled approvals"))
    run = manager._run_store.list_runs(session_id=session_id)[0]

    assert run.status == "completed"
    assert manager._approval_store.list_requests(run_id=run.id) == []
    assert not any(event["type"] == "approval_requested" for event in events)
    assert (workspace_dir / "docs" / "approval-disabled.md").read_text(encoding="utf-8") == (
        "approval disabled"
    )


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_session_manager_repairs_tool_history_after_interrupt(mock_llm_class, tmp_path: Path):
    """Interrupted runs should persist synthetic tool results so the next turn is valid."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Using a tool first.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="blocking_tool", arguments={}),
                    )
                ],
            ),
            LLMResponse(content="resumed after intervention", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    blocking_tool = BlockingTool()
    manager.bind_session_agent(
        session_id,
        Agent(
        llm_client=manager._llm_client,
        system_prompt="You are a test assistant.",
        tools=[blocking_tool],
        max_steps=5,
        workspace_dir=str(tmp_path / "workspace-a"),
        config=manager._config,
        ),
    )

    async def collect_chat_events(message: str):
        events = []
        async for event in manager.chat(session_id, message):
            events.append(event)
        return events

    chat_task = asyncio.create_task(collect_chat_events("please start working"))
    await asyncio.wait_for(blocking_tool.started.wait(), timeout=1)

    interrupted = await manager.interrupt_session(session_id)
    first_events = await asyncio.wait_for(chat_task, timeout=1)

    assert interrupted is True
    assert blocking_tool.cancelled.is_set()
    assert any(event["type"] == "interrupted" for event in first_events)

    persisted_messages = manager.get_session_messages(session_id)
    assert [message.role for message in persisted_messages] == ["system", "user", "assistant", "tool"]
    assert persisted_messages[-1].tool_call_id == "call_1"
    assert str(persisted_messages[-1].content).startswith("Error: Tool execution was interrupted")

    second_events = await collect_events(manager.chat(session_id, "continue with my updated guidance"))
    assert any(
        event["type"] == "content" and event["data"]["content"] == "resumed after intervention"
        for event in second_events
    )


@patch("clavi_agent.session.LLMClient")
def test_session_api_persists_and_lists(mock_llm_class, tmp_path: Path):
    """API should create, chat, list, restore, and delete persisted sessions."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(content="reply one", finish_reason="stop"),
            LLMResponse(content="reply two", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        first = client.post("/api/sessions", json={}).json()
        second = client.post("/api/sessions", json={}).json()

        with client.stream(
            "POST",
            f"/api/sessions/{first['session_id']}/chat",
            json={"message": "alpha session"},
        ) as response:
            assert response.status_code == 200
            body = "\n".join(response.iter_text())
            assert "reply one" in body
        first_runs = manager._run_store.list_runs(session_id=first["session_id"])
        assert len(first_runs) == 1
        assert first_runs[0].status == "completed"
        assert first_runs[0].goal == "alpha session"

        with client.stream(
            "POST",
            f"/api/sessions/{second['session_id']}/chat",
            json={"message": "beta session"},
        ) as response:
            assert response.status_code == 200
            body = "\n".join(response.iter_text())
            assert "reply two" in body
        second_runs = manager._run_store.list_runs(session_id=second["session_id"])
        assert len(second_runs) == 1
        assert second_runs[0].status == "completed"
        assert second_runs[0].goal == "beta session"

        sessions = client.get("/api/sessions").json()
        assert len(sessions) == 2
        assert sessions[0]["title"] == "beta session"
        assert sessions[1]["title"] == "alpha session"

        detail = client.get(f"/api/sessions/{first['session_id']}/messages").json()
        assert [message["role"] for message in detail["messages"]] == ["system", "user", "assistant"]
        assert detail["messages"][-1]["content"] == "reply one"

        shared_context_file = (
            Path(first["workspace_dir"])
            / ".clavi_agent"
            / "shared_context"
            / f"{first['session_id']}.json"
        )
        shared_context_file.parent.mkdir(parents=True, exist_ok=True)
        shared_context_file.write_text(
            json.dumps(
                [
                    {
                        "id": "entry-1",
                        "timestamp": "2026-03-28T10:00:00+00:00",
                        "source": "worker-1",
                        "category": "finding",
                        "title": "Socket retry",
                        "content": "WebSocket reconnect should back off exponentially.",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        shared_context = client.get(
            f"/api/sessions/{first['session_id']}/shared-context?limit=20"
        ).json()
        assert shared_context["session_id"] == first["session_id"]
        assert len(shared_context["entries"]) == 1
        assert shared_context["entries"][0]["source"] == "worker-1"
        assert shared_context["entries"][0]["title"] == "Socket retry"
        assert shared_context["entries"][0]["content"] == "WebSocket reconnect should back off exponentially."

        delete_response = client.delete(f"/api/sessions/{first['session_id']}")
        assert delete_response.status_code == 200
        assert client.get(f"/api/sessions/{first['session_id']}").status_code == 404


@patch("clavi_agent.session.LLMClient")
def test_session_api_can_switch_agent_template_and_refresh_prompt(
    mock_llm_class,
    tmp_path: Path,
):
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        store = UserMemoryStore(tmp_path / "agents.db")
        store.upsert_user_profile(
            ROOT_ACCOUNT_ID,
            profile={"preferred_language": "zh-CN"},
            summary="偏好中文。",
        )
        store.create_memory_entry(
            user_id=ROOT_ACCOUNT_ID,
            memory_type="constraint",
            content="所有读写统一使用 UTF-8。",
            summary="统一使用 UTF-8。",
            confidence=0.96,
        )

        planner = manager._agent_store.create_agent(
            name="Planner",
            description="Plans work",
            system_prompt="You are a planning assistant.",
            tools=[],
        )
        coder = manager._agent_store.create_agent(
            name="Coder",
            description="Writes code",
            system_prompt="You are a coding assistant.",
            tools=[],
        )

        created = client.post("/api/sessions", json={"agent_id": planner["id"]})
        assert created.status_code == 200
        session_id = created.json()["session_id"]

        switched = client.put(
            f"/api/sessions/{session_id}/agent",
            json={"agent_id": coder["id"]},
        )
        assert switched.status_code == 200
        assert switched.json()["agent_id"] == coder["id"]

        detail = client.get(f"/api/sessions/{session_id}/messages")
        assert detail.status_code == 200
        system_prompt = str(detail.json()["messages"][0]["content"])

        assert "You are a coding assistant." in system_prompt
        assert "User Profile Summary" in system_prompt
        assert "preferred_language: zh-CN" in system_prompt
        assert "Stable Working Preferences" in system_prompt
        assert "统一使用 UTF-8。" in system_prompt


@patch("clavi_agent.session.LLMClient")
def test_session_api_streaming_headers_and_delta_events(mock_llm_class, tmp_path: Path):
    """SSE endpoint should expose anti-buffer headers, ids, and classified event names."""
    mock_llm = mock_llm_class.return_value

    async def fake_generate_stream(messages, tools):  # noqa: ANN001
        yield {"type": "content_delta", "data": {"delta": "Hel"}}
        yield {"type": "content_delta", "data": {"delta": "lo"}}
        yield {
            "type": "final_response",
            "data": {
                "response": LLMResponse(
                    content="Hello",
                    finish_reason="stop",
                )
            },
        }

    mock_llm.generate_stream = fake_generate_stream
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="fallback should not be used", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "stream test"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers.get("cache-control") == "no-cache, no-transform"
            assert response.headers.get("x-accel-buffering") == "no"
            assert "keep-alive" in response.headers.get("connection", "").lower()

            body = "\n".join(response.iter_text())
            assert "id: 0" in body
            assert "event: ui" in body
            assert "event: state" in body
            delta_index = body.find('"type": "content_delta"')
            done_index = body.find("[DONE]")
            assert delta_index >= 0
            assert done_index >= 0
            assert delta_index < done_index


@patch("clavi_agent.session.LLMClient")
def test_session_upload_api_persists_lists_and_downloads_files(mock_llm_class, tmp_path: Path):
    """Upload APIs should persist metadata, list session files, and expose preview/download links."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        upload_response = client.post(
            f"/api/sessions/{session['session_id']}/uploads",
            files=[
                ("files", ("draft.md", b"# Draft\n\nHello", "text/markdown")),
                ("files", ("notes.txt", "todo".encode("utf-8"), "text/plain")),
            ],
        )

        assert upload_response.status_code == 201
        payload = upload_response.json()
        assert payload["session_id"] == session["session_id"]
        assert [item["original_name"] for item in payload["uploads"]] == ["draft.md", "notes.txt"]
        assert payload["uploads"][0]["mime_type"] == "text/markdown"
        assert payload["uploads"][1]["mime_type"] == "text/plain"

        workspace_dir = Path(session["workspace_dir"]).resolve()
        stored_path = workspace_dir / Path(payload["uploads"][0]["relative_path"])
        assert stored_path.read_text(encoding="utf-8") == "# Draft\n\nHello"

        list_response = client.get(f"/api/sessions/{session['session_id']}/uploads")
        assert list_response.status_code == 200
        listed_payload = list_response.json()
        assert {item["id"] for item in listed_payload["uploads"]} == {
            item["id"] for item in payload["uploads"]
        }

        download_response = client.get(f"/api/uploads/{payload['uploads'][0]['id']}")
        assert download_response.status_code == 200
        assert download_response.content == b"# Draft\n\nHello"
        assert "filename=" in download_response.headers["content-disposition"]

        inline_response = client.get(
            f"/api/uploads/{payload['uploads'][0]['id']}",
            params={"disposition": "inline"},
        )
        assert inline_response.status_code == 200
        assert "inline" in inline_response.headers["content-disposition"]

        preview_response = client.get(f"/api/uploads/{payload['uploads'][0]['id']}/preview")
        assert preview_response.status_code == 200
        preview_payload = preview_response.json()
        assert preview_payload["preview_kind"] == "markdown"
        assert preview_payload["preview_supported"] is True
        assert preview_payload["text_content"] == "# Draft\n\nHello"
        assert preview_payload["open_url"].endswith("?disposition=inline")
        assert preview_payload["download_url"].endswith(payload["uploads"][0]["id"])


@patch("clavi_agent.session.LLMClient")
def test_artifact_file_api_serves_downloads_and_previews(mock_llm_class, tmp_path: Path):
    """Artifact file APIs should safely serve run outputs for preview and download."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        workspace_dir = Path(session["workspace_dir"]).resolve()
        artifact_path = workspace_dir / "docs" / "result.md"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"# Result\n\nReady")

        run = manager._run_store.create_run(
            RunRecord(
                id="run-artifact-preview",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=manager._agent_store.snapshot_agent_template(
                    "system-default-agent"
                ),
                status="completed",
                goal="artifact preview",
                created_at="2026-04-13T08:00:00+00:00",
                started_at="2026-04-13T08:00:00+00:00",
                finished_at="2026-04-13T08:00:02+00:00",
                current_step_index=1,
            )
        )
        manager._run_store.create_artifact(
            ArtifactRecord(
                id="artifact-preview-1",
                run_id=run.id,
                artifact_type="workspace_file",
                uri="docs/result.md",
                display_name="result.md",
                role="final_deliverable",
                format="md",
                mime_type="text/markdown",
                size_bytes=artifact_path.stat().st_size,
                source="agent_generated",
                is_final=True,
                preview_kind="markdown",
                summary="Generated deliverable",
                created_at="2026-04-13T08:00:01+00:00",
            )
        )

        download_response = client.get("/api/artifacts/artifact-preview-1")
        assert download_response.status_code == 200
        assert download_response.content == b"# Result\n\nReady"
        assert "filename=" in download_response.headers["content-disposition"]

        inline_response = client.get(
            "/api/artifacts/artifact-preview-1",
            params={"disposition": "inline"},
        )
        assert inline_response.status_code == 200
        assert "inline" in inline_response.headers["content-disposition"]

        preview_response = client.get("/api/artifacts/artifact-preview-1/preview")
        assert preview_response.status_code == 200
        preview_payload = preview_response.json()
        assert preview_payload["preview_kind"] == "markdown"
        assert preview_payload["preview_supported"] is True
        assert preview_payload["text_content"] == "# Result\n\nReady"
        assert preview_payload["open_url"].endswith("?disposition=inline")
        assert preview_payload["download_url"].endswith("artifact-preview-1")


@patch("clavi_agent.session.LLMClient")
def test_root_run_artifacts_api_includes_child_run_outputs(mock_llm_class, tmp_path: Path):
    """Root run artifact listings should surface delegated child outputs for direct preview."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()

        root_run = manager._run_store.create_run(
            RunRecord(
                id="run-root-artifacts",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=manager._agent_store.snapshot_agent_template(
                    "system-default-agent"
                ),
                status="completed",
                goal="root run outputs",
                created_at="2026-04-13T09:00:00+00:00",
                started_at="2026-04-13T09:00:00+00:00",
                finished_at="2026-04-13T09:00:03+00:00",
                current_step_index=1,
            )
        )
        child_run = manager._run_store.create_run(
            RunRecord(
                id="run-child-artifacts",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=manager._agent_store.snapshot_agent_template(
                    "system-default-agent"
                ),
                status="completed",
                goal="child run outputs",
                parent_run_id=root_run.id,
                created_at="2026-04-13T09:00:01+00:00",
                started_at="2026-04-13T09:00:01+00:00",
                finished_at="2026-04-13T09:00:02+00:00",
                current_step_index=1,
            )
        )
        manager._run_store.create_artifact(
            ArtifactRecord(
                id="artifact-child-output",
                run_id=child_run.id,
                artifact_type="workspace_file",
                uri="docs/child-output.md",
                display_name="child-output.md",
                role="final_deliverable",
                format="md",
                mime_type="text/markdown",
                source="agent_generated",
                is_final=True,
                preview_kind="markdown",
                summary="Delegated worker wrote the file.",
                created_at="2026-04-13T09:00:02+00:00",
            )
        )

        root_artifacts_response = client.get(f"/api/runs/{root_run.id}/artifacts")
        assert root_artifacts_response.status_code == 200
        root_artifacts = root_artifacts_response.json()
        assert [artifact["id"] for artifact in root_artifacts] == ["artifact-child-output"]
        assert root_artifacts[0]["run_id"] == child_run.id
        assert root_artifacts[0]["preview_kind"] == "markdown"

        child_artifacts_response = client.get(f"/api/runs/{child_run.id}/artifacts")
        assert child_artifacts_response.status_code == 200
        child_artifacts = child_artifacts_response.json()
        assert [artifact["id"] for artifact in child_artifacts] == ["artifact-child-output"]


@patch("clavi_agent.session.LLMClient")
def test_artifact_file_api_marks_pdf_as_inline_previewable(mock_llm_class, tmp_path: Path):
    """PDF artifacts should expose inline preview metadata for the Web UI."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        workspace_dir = Path(session["workspace_dir"]).resolve()
        artifact_path = workspace_dir / "exports" / "report.pdf"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"%PDF-1.4\n% inline preview test\n")

        run = manager._run_store.create_run(
            RunRecord(
                id="run-artifact-pdf-preview",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=manager._agent_store.snapshot_agent_template(
                    "system-default-agent"
                ),
                status="completed",
                goal="artifact pdf preview",
                created_at="2026-04-13T08:00:00+00:00",
                started_at="2026-04-13T08:00:00+00:00",
                finished_at="2026-04-13T08:00:02+00:00",
                current_step_index=1,
            )
        )
        manager._run_store.create_artifact(
            ArtifactRecord(
                id="artifact-preview-pdf-1",
                run_id=run.id,
                artifact_type="workspace_file",
                uri="exports/report.pdf",
                display_name="report.pdf",
                role="final_deliverable",
                format="pdf",
                mime_type="application/pdf",
                size_bytes=artifact_path.stat().st_size,
                source="agent_generated",
                is_final=True,
                preview_kind="pdf",
                summary="Generated PDF deliverable",
                created_at="2026-04-13T08:00:01+00:00",
            )
        )

        preview_response = client.get("/api/artifacts/artifact-preview-pdf-1/preview")
        assert preview_response.status_code == 200
        preview_payload = preview_response.json()
        assert preview_payload["preview_kind"] == "pdf"
        assert preview_payload["preview_supported"] is True
        assert preview_payload["inline_url"].endswith("?disposition=inline")
        assert preview_payload["open_url"].endswith("?disposition=inline")
        assert preview_payload["download_url"].endswith("artifact-preview-pdf-1")


@patch("clavi_agent.session.LLMClient")
def test_artifact_file_api_blocks_paths_outside_workspace(mock_llm_class, tmp_path: Path):
    """Artifact file APIs should reject paths that escape the owning session workspace."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        run = manager._run_store.create_run(
            RunRecord(
                id="run-artifact-escape",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=manager._agent_store.snapshot_agent_template(
                    "system-default-agent"
                ),
                status="completed",
                goal="artifact escape",
                created_at="2026-04-13T08:10:00+00:00",
                started_at="2026-04-13T08:10:00+00:00",
                finished_at="2026-04-13T08:10:02+00:00",
                current_step_index=1,
            )
        )
        manager._run_store.create_artifact(
            ArtifactRecord(
                id="artifact-escape-1",
                run_id=run.id,
                artifact_type="workspace_file",
                uri="../outside.md",
                display_name="outside.md",
                role="final_deliverable",
                format="md",
                mime_type="text/markdown",
                preview_kind="markdown",
                created_at="2026-04-13T08:10:01+00:00",
            )
        )

        preview_response = client.get("/api/artifacts/artifact-escape-1/preview")
        assert preview_response.status_code == 400
        assert "escapes the session workspace" in preview_response.json()["detail"]


@patch("clavi_agent.session.LLMClient")
def test_chat_api_accepts_attachment_ids_and_persists_structured_user_message(
    mock_llm_class,
    tmp_path: Path,
):
    """Chat API should accept attachment ids and persist structured user content blocks."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="已收到并开始处理", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        upload_response = client.post(
            f"/api/sessions/{session['session_id']}/uploads",
            files=[("files", ("draft.md", b"# Draft\n\nHello", "text/markdown"))],
        )

        assert upload_response.status_code == 201
        upload_id = upload_response.json()["uploads"][0]["id"]

        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={
                "message": "请修订这份草稿",
                "attachment_ids": [upload_id],
            },
        ) as response:
            assert response.status_code == 200
            body = "\n".join(response.iter_text())
            assert "已收到并开始处理" in body

        detail = client.get(f"/api/sessions/{session['session_id']}/messages").json()
        user_message = detail["messages"][1]
        assert user_message["role"] == "user"
        assert isinstance(user_message["content"], list)
        assert user_message["content"][0] == {"type": "text", "text": "请修订这份草稿"}
        assert user_message["content"][1]["type"] == "uploaded_file"
        assert user_message["content"][1]["upload_id"] == upload_id
        assert user_message["content"][1]["relative_path"].endswith("/draft.md")

        runs = manager._run_store.list_runs(session_id=session["session_id"])
        assert len(runs) == 1
        assert "请修订这份草稿" in runs[0].goal
        assert "draft.md" in runs[0].goal


@patch("clavi_agent.session.LLMClient")
def test_upload_revision_flow_exposes_revised_deliverable_in_run_detail(
    mock_llm_class,
    tmp_path: Path,
):
    mock_llm = mock_llm_class.return_value

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        template = manager._agent_store.create_agent(
            name="Reviser",
            description="Revises uploaded files",
            system_prompt="Revise uploaded files by creating a copy.",
            tools=["WriteTool"],
        )
        session = client.post(
            "/api/sessions",
            json={"agent_id": template["id"]},
        ).json()
        workspace_dir = Path(session["workspace_dir"]).resolve()
        manager.bind_session_agent(
            session["session_id"],
            Agent(
                llm_client=manager._llm_client,
                system_prompt="You are a test assistant.",
                tools=[SimpleWriteTool(workspace_dir)],
                max_steps=6,
                workspace_dir=str(workspace_dir),
                config=manager._config,
            ),
        )

        upload_response = client.post(
            f"/api/sessions/{session['session_id']}/uploads",
            files=[("files", ("draft.md", b"# Draft\n\nOriginal body", "text/markdown"))],
        )
        assert upload_response.status_code == 201
        upload = upload_response.json()["uploads"][0]
        revised_path = Path(upload["relative_path"]).with_name("draft.revised.md").as_posix()

        mock_llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="先生成修订版文件。",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_write",
                            type="function",
                            function=FunctionCall(
                                name="write_file",
                                arguments={
                                    "path": revised_path,
                                    "content": "# Revised\n\nUpdated body",
                                },
                            ),
                        )
                    ],
                ),
                LLMResponse(content="修订完成", finish_reason="stop"),
            ]
        )

        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={
                "message": "请修订这份草稿",
                "attachment_ids": [upload["id"]],
            },
        ) as response:
            assert response.status_code == 200
            run_id = response.headers["x-run-id"]
            body = "\n".join(response.iter_text())
            assert "修订完成" in body

        run_detail_response = client.get(f"/api/runs/{run_id}")
        assert run_detail_response.status_code == 200
        run_payload = run_detail_response.json()
        assert run_payload["deliverable_manifest"]["primary_artifact_id"]

        artifacts_response = client.get(f"/api/runs/{run_id}/artifacts")
        assert artifacts_response.status_code == 200
        artifacts = artifacts_response.json()
        assert len(artifacts) == 1

        artifact = artifacts[0]
        assert artifact["id"] == run_payload["deliverable_manifest"]["primary_artifact_id"]
        assert artifact["role"] == "revised_file"
        assert artifact["source"] == "agent_revised"
        assert artifact["is_final"] is True
        assert artifact["metadata"]["parent_upload_id"] == upload["id"]
        assert artifact["metadata"]["parent_upload_name"] == "draft.md"
        assert artifact["metadata"]["revision_mode"] == "copy_on_write"
        assert artifact["uri"].endswith("draft.revised.md")

        manifest_item = run_payload["deliverable_manifest"]["items"][0]
        assert manifest_item["artifact_id"] == artifact["id"]
        assert manifest_item["role"] == "revised_file"
        assert manifest_item["uri"].endswith("draft.revised.md")

        revised_file = workspace_dir / Path(artifact["uri"])
        assert revised_file.read_text(encoding="utf-8") == "# Revised\n\nUpdated body"


@patch("clavi_agent.session.LLMClient")
def test_session_upload_api_rejects_invalid_extension(mock_llm_class, tmp_path: Path):
    """Upload API should block risky executable file types."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        upload_response = client.post(
            f"/api/sessions/{session['session_id']}/uploads",
            files=[("files", ("payload.exe", b"MZ", "application/octet-stream"))],
        )

        assert upload_response.status_code == 400
        assert "Unsupported upload file type" in upload_response.json()["detail"]


@patch("clavi_agent.session.LLMClient")
def test_session_upload_api_rejects_oversize_file(mock_llm_class, tmp_path: Path):
    """Upload API should enforce the configured upload size limit."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client, patch("clavi_agent.session.MAX_UPLOAD_SIZE_BYTES", 4):
        session = client.post("/api/sessions", json={}).json()
        upload_response = client.post(
            f"/api/sessions/{session['session_id']}/uploads",
            files=[("files", ("draft.md", b"12345", "text/markdown"))],
        )

        assert upload_response.status_code == 400
        assert "Upload exceeds size limit" in upload_response.json()["detail"]


@patch("clavi_agent.session.LLMClient")
def test_session_upload_api_returns_not_found_for_missing_session(mock_llm_class, tmp_path: Path):
    """Upload API should reject missing session ids before parsing files."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        upload_response = client.post(
            "/api/sessions/missing-session/uploads",
            files=[("files", ("draft.md", b"# Draft", "text/markdown"))],
        )

        assert upload_response.status_code == 404
        assert upload_response.json()["detail"] == "Session not found"


@patch("clavi_agent.session.LLMClient")
def test_feature_flags_api_returns_effective_rollout_state(mock_llm_class, tmp_path: Path):
    """Feature API should expose effective rollout flags after dependency resolution."""
    manager = SessionManager(
        config=build_config(
            tmp_path,
            enable_durable_runs=False,
            enable_run_trace=True,
            enable_approval_flow=True,
        )
    )
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.get("/api/features")

    assert response.status_code == 200
    assert response.json() == {
        "enable_durable_runs": False,
        "enable_run_trace": False,
        "enable_approval_flow": False,
        "enable_supervisor_mode": True,
        "enable_worker_model_routing": True,
        "enable_compact_prompt_memory": True,
        "enable_session_retrieval": True,
        "enable_learned_workflow_generation": True,
        "enable_external_memory_providers": True,
    }


@patch("clavi_agent.session.LLMClient")
def test_disabling_session_retrieval_hides_history_api_but_keeps_session_chat_available(
    mock_llm_class,
    tmp_path: Path,
):
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="retrieval disabled reply", finish_reason="stop")
    )

    manager = SessionManager(
        config=build_config(
            tmp_path,
            enable_session_retrieval=False,
        )
    )
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        history_response = client.get("/api/session-history", params={"query": "UTF-8"})
        sessions_response = client.get("/api/sessions")

        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "继续处理当前任务"},
        ) as response:
            assert response.status_code == 200
            body = "\n".join(response.iter_text())

    assert history_response.status_code == 404
    assert sessions_response.status_code == 200
    assert len(sessions_response.json()) == 1
    assert "retrieval disabled reply" in body


@patch("clavi_agent.session.LLMClient")
def test_disabling_durable_run_apis_keeps_legacy_chat_endpoint_available(
    mock_llm_class,
    tmp_path: Path,
):
    """Run/approval APIs may be gated off while the legacy chat endpoint remains usable."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="legacy reply", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path, enable_durable_runs=False))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()

        assert client.get("/api/runs").status_code == 404
        assert client.get("/api/approvals").status_code == 404

        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "fallback chat"},
        ) as response:
            assert response.status_code == 200
            body = "\n".join(response.iter_text())
            assert "legacy reply" in body

        runs = manager._run_store.list_runs(session_id=session["session_id"])
        assert len(runs) == 1
        assert runs[0].goal == "fallback chat"
        assert runs[0].status == "completed"


@patch("clavi_agent.session.LLMClient")
def test_disabling_run_trace_hides_trace_endpoints_but_keeps_run_queries(
    mock_llm_class,
    tmp_path: Path,
):
    """Trace rollout can be gated separately from the base durable-run APIs."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="trace disabled reply", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path, enable_run_trace=False))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "create run"},
        ) as response:
            assert response.status_code == 200
            list(response.iter_text())

        run = manager._run_store.list_runs(session_id=session["session_id"])[0]
        assert client.get(f"/api/runs/{run.id}").status_code == 200
        assert client.get(f"/api/runs/{run.id}/steps").status_code == 200
        assert client.get(f"/api/runs/{run.id}/trace/timeline").status_code == 404
        assert client.get("/api/runs/metrics").status_code == 404


@patch("clavi_agent.session.LLMClient")
def test_workspace_static_shell_exposes_run_observability_panels(mock_llm_class, tmp_path: Path):
    """Workspace HTML should expose the run-focused UI regions used by the durable-run frontend."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.get("/static/workspace.html")
        assert response.status_code == 200
        body = response.text
        assert 'id="active-run-banner"' in body
        assert 'data-sidebar-tab="runs"' in body
        assert 'id="run-session-summary"' in body
        assert 'id="approval-inbox-list"' in body
        assert 'id="approval-inbox-count"' in body
        assert 'id="run-history-list"' in body
        assert 'id="run-detail-panel"' in body


@patch("clavi_agent.session.LLMClient")
def test_agent_marketplace_static_shell_exposes_template_policy_sections(
    mock_llm_class,
    tmp_path: Path,
):
    """Marketplace HTML should expose template policy and delegation controls while routing stays in API settings."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.get("/static/agent_marketplace.html")
        assert response.status_code == 200
        body = response.text
        assert 'id="marketplace-boundary-note"' in body
        assert 'id="agentTemplatePolicySection"' in body
        assert 'id="agentDefaultToolsContainer"' in body
        assert 'id="agentWorkspaceModeInput"' in body
        assert 'id="agentApprovalModeInput"' in body
        assert 'id="agentRunTimeoutInput"' in body
        assert 'id="agentRunConcurrencyInput"' in body
        assert 'id="agentReadonlyBanner"' in body
        assert 'id="agentDelegationModeInput"' in body
        assert 'id="agentRequireDelegateWriteInput"' in body
        assert 'id="agentRequireDelegateShellInput"' in body
        assert 'id="agentRequireDelegateMcpInput"' in body
        assert 'id="agentAllowMainReadToolsInput"' in body
        assert 'id="agentVerifyWorkerOutputInput"' in body
        assert 'id="agentPreferBatchDelegateInput"' in body
        assert 'id="agentRoutingConfigNote"' in body


@patch("clavi_agent.session.LLMClient")
def test_agent_routing_static_shell_exposes_account_level_agent_routing_controls(
    mock_llm_class,
    tmp_path: Path,
):
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.get("/static/agent_routing.html")
        assert response.status_code == 200
        body = response.text
        assert 'id="routing-target-config-id"' in body
        assert 'id="routing-planner-api-config-id"' in body
        assert 'id="routing-worker-api-config-id"' in body
        assert 'id="routing-summary"' in body


@patch("clavi_agent.session.LLMClient")
def test_workspace_static_shell_exposes_separate_agent_routing_entry(
    mock_llm_class,
    tmp_path: Path,
):
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        response = client.get("/static/workspace.html")
        assert response.status_code == 200
        body = response.text
        assert 'id="btn-api-config"' in body
        assert 'id="btn-agent-routing"' in body


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_stream_disconnect_does_not_cancel_background_run(mock_llm_class, tmp_path: Path):
    """Dropping one run stream subscriber should not cancel the underlying durable run."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need the gate tool.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="gate_tool", arguments={}),
                    )
                ],
            ),
            LLMResponse(content="completed after disconnect", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    gate_tool = GateTool()
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[gate_tool],
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace-a"),
            config=manager._config,
        ),
    )

    run = manager.start_chat_run(session_id, "keep running after disconnect")

    async def consume_until_disconnect() -> list[dict]:
        events = []
        async for event in manager.stream_run(run.id):
            events.append(event)
            if event["type"] == "tool_call":
                break
        return events

    subscriber = asyncio.create_task(consume_until_disconnect())
    await asyncio.to_thread(gate_tool.started.wait, 1)
    first_events = await asyncio.wait_for(subscriber, timeout=1)
    assert any(event["type"] == "tool_call" for event in first_events)

    gate_tool.release.set()

    deadline = time.monotonic() + 2
    while True:
        persisted_run = manager._run_store.get_run(run.id)
        assert persisted_run is not None
        if persisted_run.status == "completed":
            break
        if time.monotonic() >= deadline:
            raise AssertionError("Background run did not complete after stream disconnect.")
        await asyncio.sleep(0.05)

    assert manager._run_store.get_run(run.id).status == "completed"


@patch("clavi_agent.session.LLMClient")
def test_run_api_exposes_run_metadata_and_reconnect_stream(mock_llm_class, tmp_path: Path):
    """Run APIs should expose the run id, metadata, and replayable event stream."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="reply one", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "alpha session"},
        ) as response:
            assert response.status_code == 200
            run_id = response.headers["x-run-id"]
            assert run_id
            body = "\n".join(response.iter_text())
            assert "reply one" in body

        run_detail = client.get(f"/api/runs/{run_id}")
        assert run_detail.status_code == 200
        assert run_detail.json()["id"] == run_id
        assert run_detail.json()["status"] == "completed"

        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            assert response.status_code == 200
            assert response.headers["x-run-id"] == run_id
            body = "\n".join(response.iter_text())
            assert "event: ui" in body
            assert "event: state" in body
            assert '"type": "content"' in body
            assert '"type": "done"' in body


@patch("clavi_agent.session.LLMClient")
def test_run_event_stream_resumes_from_last_event_id(mock_llm_class, tmp_path: Path):
    """Reconnecting with Last-Event-ID should skip already delivered run events."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="reply one", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        create_response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "resume me",
            },
        )
        assert create_response.status_code == 201
        run_id = create_response.json()["id"]

        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            assert response.status_code == 200
            first_body = "\n".join(response.iter_text())
            assert "id: 0" in first_body
            assert "id: 1" in first_body
            assert "id: 2" in first_body
            assert '"type": "step"' in first_body
            assert '"type": "content"' in first_body
            assert '"type": "done"' in first_body

        with client.stream(
            "GET",
            f"/api/runs/{run_id}/events",
            headers={"Last-Event-ID": "0"},
        ) as response:
            assert response.status_code == 200
            resumed_body = "\n".join(response.iter_text())
            assert "id: 0" not in resumed_body
            assert "id: 1" in resumed_body
            assert "id: 2" in resumed_body
            assert '"type": "step"' not in resumed_body
            assert '"type": "content"' in resumed_body
            assert '"type": "done"' in resumed_body


@patch("clavi_agent.session.LLMClient")
def test_run_event_stream_replays_trace_from_offset_without_memory_state(
    mock_llm_class,
    tmp_path: Path,
):
    """Trace replay should work from a requested offset even after live state is gone."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="reply one", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()

        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "trace replay"},
        ) as response:
            assert response.status_code == 200
            run_id = response.headers["x-run-id"]
            list(response.iter_text())

        manager._run_manager._execution_states.pop(run_id, None)

        trace_events = manager._trace_store.list_events(run_id)
        assert len(trace_events) >= 3

        with client.stream(
            "GET",
            f"/api/runs/{run_id}/events",
            params={"trace_offset": 1},
        ) as response:
            assert response.status_code == 200
            body = "\n".join(response.iter_text())
            assert "event: trace" in body
            assert '"type": "trace"' in body
            assert f'"sequence": {trace_events[0].sequence}' not in body
            assert f'"sequence": {trace_events[1].sequence}' in body
            assert "[DONE]" in body


@patch("clavi_agent.session.LLMClient")
def test_run_create_api_starts_run_without_chat_stream(mock_llm_class, tmp_path: Path):
    """POST /api/runs should create a durable run that clients can subscribe to later."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="created through run api", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        create_response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "alpha run",
            },
        )
        assert create_response.status_code == 201
        created_run = create_response.json()
        assert created_run["session_id"] == session["session_id"]
        assert created_run["goal"] == "alpha run"

        run_id = created_run["id"]
        with client.stream("GET", f"/api/runs/{run_id}/events") as response:
            assert response.status_code == 200
            assert response.headers["x-run-id"] == run_id
            body = "\n".join(response.iter_text())
            assert "created through run api" in body
            assert '"type": "done"' in body

        run_detail = client.get(f"/api/runs/{run_id}")
        assert run_detail.status_code == 200
        assert run_detail.json()["status"] == "completed"


@patch("clavi_agent.session.LLMClient")
def test_run_create_api_resolves_policy_hierarchy_with_session_override(
    mock_llm_class,
    tmp_path: Path,
):
    """Run creation should resolve system defaults, template policy, and session override into one snapshot."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="policy reply", finish_reason="stop")
    )

    manager = SessionManager(
        config=build_config(
            tmp_path,
            run_timeout_seconds=120,
            max_concurrent_runs=5,
        )
    )
    app = create_app(manager)

    with TestClient(app) as client:
        template = manager._agent_store.create_agent(
            name="Policy Template",
            description="Template policy precedence coverage",
            system_prompt="Stay within policy.",
            tools=[],
            workspace_policy={
                "mode": "shared",
                "allow_session_override": True,
                "readable_roots": ["docs", "reports"],
                "writable_roots": ["docs"],
                "read_only_tools": ["bash"],
                "disabled_tools": ["delegate_task"],
                "allowed_shell_command_prefixes": ["git status"],
                "allowed_network_domains": ["example.com"],
            },
            approval_policy={
                "mode": "default",
                "require_approval_tools": ["write_file"],
                "auto_approve_tools": [],
                "require_approval_risk_levels": ["critical"],
                "require_approval_risk_categories": ["external_network"],
                "notes": "Template approval notes.",
            },
            run_policy={
                "max_concurrent_runs": 2,
            },
            delegation_policy={
                "mode": "prefer_delegate",
                "require_delegate_for_write_actions": True,
            },
        )
        session = client.post(
            "/api/sessions",
            json={"agent_id": template["id"]},
        ).json()

        response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "resolve policy hierarchy",
                "workspace_policy": {
                    "mode": "isolated",
                    "writable_roots": ["reports"],
                },
                "approval_policy": {
                    "mode": "strict",
                    "auto_approve_tools": ["read_file"],
                    "notes": "Session override approval notes.",
                },
                "run_policy": {
                    "timeout_seconds": 45,
                },
                "delegation_policy": {
                    "mode": "supervisor_only",
                    "allow_main_agent_read_tools": False,
                },
            },
        )

        assert response.status_code == 201
        run = response.json()
        snapshot = run["agent_template_snapshot"]

        assert snapshot["workspace_policy"] == {
            "mode": "isolated",
            "allow_session_override": True,
            "readable_roots": ["docs", "reports"],
            "writable_roots": ["reports"],
            "read_only_tools": ["bash"],
            "disabled_tools": ["delegate_task"],
            "allowed_shell_command_prefixes": ["git status"],
            "allowed_network_domains": ["example.com"],
        }
        assert snapshot["approval_policy"] == {
            "mode": "strict",
            "require_approval_tools": ["write_file"],
            "auto_approve_tools": ["read_file"],
            "require_approval_risk_levels": ["critical"],
            "require_approval_risk_categories": ["external_network"],
            "notes": "Session override approval notes.",
        }
        assert snapshot["run_policy"] == {
            "timeout_seconds": 45,
            "max_concurrent_runs": 2,
        }
        assert snapshot["delegation_policy"] == {
            "mode": "supervisor_only",
            "require_delegate_for_write_actions": True,
            "require_delegate_for_shell": False,
            "require_delegate_for_stateful_mcp": False,
            "allow_main_agent_read_tools": False,
            "verify_worker_output": True,
            "prefer_batch_delegate": True,
        }
        assert run["run_metadata"]["policy_hierarchy"] == {
            "system_template_id": "system-default-agent",
            "template_id": template["id"],
            "session_override": True,
            "runtime_approval_field": "approval_auto_grant_tools",
        }
        assert run["run_metadata"]["session_policy_override"] == {
            "workspace_policy": {
                "mode": "isolated",
                "writable_roots": ["reports"],
            },
            "approval_policy": {
                "mode": "strict",
                "auto_approve_tools": ["read_file"],
                "notes": "Session override approval notes.",
            },
            "run_policy": {
                "timeout_seconds": 45,
            },
            "delegation_policy": {
                "mode": "supervisor_only",
                "allow_main_agent_read_tools": False,
            },
        }


@patch("clavi_agent.session.LLMClient")
def test_run_create_api_inherits_system_default_run_policy_when_template_omits_timeout(
    mock_llm_class,
    tmp_path: Path,
):
    """Run snapshots should inherit system-default policy fields before template/session overrides apply."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="policy reply", finish_reason="stop")
    )

    manager = SessionManager(
        config=build_config(
            tmp_path,
            run_timeout_seconds=120,
            max_concurrent_runs=5,
        )
    )
    app = create_app(manager)

    with TestClient(app) as client:
        template = manager._agent_store.create_agent(
            name="Inherited Policy Template",
            description="System default inheritance coverage",
            system_prompt="Inherit defaults first.",
            tools=[],
            run_policy={
                "max_concurrent_runs": 2,
            },
        )
        session = client.post(
            "/api/sessions",
            json={"agent_id": template["id"]},
        ).json()

        response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "inherit system defaults",
            },
        )

        assert response.status_code == 201
        snapshot = response.json()["agent_template_snapshot"]
        assert snapshot["run_policy"] == {
            "timeout_seconds": 120,
            "max_concurrent_runs": 2,
        }


@patch("clavi_agent.session.LLMClient")
def test_run_create_api_rejects_session_policy_override_when_template_disables_it(
    mock_llm_class,
    tmp_path: Path,
):
    """Run creation should reject session overrides when the template disallows them."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="policy reply", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path, run_timeout_seconds=120))
    app = create_app(manager)

    with TestClient(app) as client:
        template = manager._agent_store.create_agent(
            name="Locked Policy Template",
            description="Override lock coverage",
            system_prompt="Do not allow session overrides.",
            tools=[],
            workspace_policy={
                "mode": "shared",
                "allow_session_override": False,
                "writable_roots": ["docs"],
            },
        )
        session = client.post(
            "/api/sessions",
            json={"agent_id": template["id"]},
        ).json()

        response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "attempt locked override",
                "run_policy": {
                    "timeout_seconds": 45,
                },
            },
        )

        assert response.status_code == 400
        assert "does not allow session policy overrides" in response.json()["detail"]


@patch("clavi_agent.session.LLMClient")
def test_run_query_apis_list_runs_steps_trace_artifacts_and_approvals(mock_llm_class, tmp_path: Path):
    """Run query APIs should expose persisted run diagnostics and approval queue data."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(content="reply one", finish_reason="stop"),
            LLMResponse(content="reply two", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()

        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "successful run"},
        ) as response:
            assert response.status_code == 200
            successful_run_id = response.headers["x-run-id"]
            list(response.iter_text())

        with client.stream(
            "POST",
            f"/api/sessions/{session['session_id']}/chat",
            json={"message": "failed run"},
        ) as response:
            assert response.status_code == 200
            failed_run_id = response.headers["x-run-id"]
            list(response.iter_text())

        failed_run = manager._run_store.get_run(failed_run_id)
        assert failed_run is not None
        failed_run = failed_run.model_copy(
            update={
                "status": "failed",
                "finished_at": "2026-04-10T00:00:00+00:00",
                "error_summary": "synthetic failure",
            }
        )
        manager._run_store.update_run(failed_run)

        manager._run_store.create_artifact(
            ArtifactRecord(
                id="artifact-1",
                run_id=successful_run_id,
                artifact_type="document",
                uri="docs/output.md",
                display_name="output.md",
                role="final_deliverable",
                format="md",
                mime_type="text/markdown",
                source="agent_generated",
                is_final=True,
                preview_kind="markdown",
                summary="Generated report",
                metadata={"kind": "report"},
                created_at="2026-04-10T00:00:01+00:00",
            )
        )
        successful_run = manager._run_store.get_run(successful_run_id)
        assert successful_run is not None
        manager._run_store.update_run(
            successful_run.model_copy(
                update={
                    "deliverable_manifest": RunDeliverableManifest(
                        primary_artifact_id="artifact-1",
                        items=[
                            RunDeliverableRef(
                                artifact_id="artifact-1",
                                uri="docs/output.md",
                                display_name="output.md",
                                format="md",
                                mime_type="text/markdown",
                                role="final_deliverable",
                                is_primary=True,
                            )
                        ],
                    )
                }
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-extra-1",
                run_id=successful_run_id,
                sequence=999,
                event_type="diagnostic",
                status="completed",
                payload_summary="extra trace",
                created_at="2026-04-10T00:00:02+00:00",
            )
        )
        manager._approval_store.create_request(
            ApprovalRequestRecord(
                id="approval-1",
                run_id=successful_run_id,
                tool_name="BashTool",
                risk_level="high",
                status="pending",
                parameter_summary="git push",
                impact_summary="remote mutation",
                requested_at="2026-04-10T00:00:03+00:00",
            )
        )
        other_session = client.post("/api/sessions", json={}).json()
        other_run = manager._run_store.create_run(
            RunRecord(
                id="run-other-approval",
                session_id=other_session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=manager._agent_store.snapshot_agent_template(
                    "system-default-agent"
                ),
                status="waiting_approval",
                goal="other session approval",
                created_at="2026-04-10T00:00:04+00:00",
                started_at="2026-04-10T00:00:04+00:00",
                current_step_index=1,
            )
        )
        manager._approval_store.create_request(
            ApprovalRequestRecord(
                id="approval-other-session",
                run_id=other_run.id,
                tool_name="write_file",
                risk_level="high",
                status="pending",
                parameter_summary="docs/other.md",
                impact_summary="writes outside current run",
                requested_at="2026-04-10T00:00:05+00:00",
            )
        )

        runs_response = client.get(
            "/api/runs",
            params={"session_id": session["session_id"], "limit": 10},
        )
        assert runs_response.status_code == 200
        runs = runs_response.json()
        assert [run["id"] for run in runs] == [failed_run_id, successful_run_id]

        failed_runs_response = client.get("/api/runs/failed", params={"limit": 10})
        assert failed_runs_response.status_code == 200
        failed_runs = failed_runs_response.json()
        assert [run["id"] for run in failed_runs] == [failed_run_id]

        run_detail_response = client.get(f"/api/runs/{successful_run_id}")
        assert run_detail_response.status_code == 200
        assert run_detail_response.json()["deliverable_manifest"]["primary_artifact_id"] == "artifact-1"

        steps_response = client.get(f"/api/runs/{successful_run_id}/steps")
        assert steps_response.status_code == 200
        assert [step["step_type"] for step in steps_response.json()] == ["llm_call", "completion"]

        trace_response = client.get(f"/api/runs/{successful_run_id}/trace")
        assert trace_response.status_code == 200
        assert any(event["event_type"] == "diagnostic" for event in trace_response.json())

        artifacts_response = client.get(f"/api/runs/{successful_run_id}/artifacts")
        assert artifacts_response.status_code == 200
        assert artifacts_response.json()[0]["uri"] == "docs/output.md"
        assert artifacts_response.json()[0]["is_final"] is True
        assert artifacts_response.json()[0]["preview_kind"] == "markdown"

        approvals_response = client.get("/api/approvals", params={"status": "pending"})
        assert approvals_response.status_code == 200
        assert [item["id"] for item in approvals_response.json()] == [
            "approval-other-session",
            "approval-1",
        ]

        session_approvals_response = client.get(
            "/api/approvals",
            params={"status": "pending", "session_id": session["session_id"]},
        )
        assert session_approvals_response.status_code == 200
        assert [item["id"] for item in session_approvals_response.json()] == ["approval-1"]


@patch("clavi_agent.session.LLMClient")
def test_run_trace_diagnostics_apis_expose_timeline_tree_export_and_locations(
    mock_llm_class,
    tmp_path: Path,
):
    """Trace diagnostics APIs should expose normalized timelines, replay views, exports, and disk locations."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="unused", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    def runtime_payload(
        *,
        run_id: str,
        agent_name: str,
        depth: int,
        is_main_agent: bool,
        data: dict,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
    ) -> str:
        return json.dumps(
            {
                "context": {
                    "session_id": session["session_id"],
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "is_main_agent": is_main_agent,
                    "depth": depth,
                    "parent_run_id": parent_run_id,
                    "root_run_id": root_run_id or run_id,
                },
                "data": data,
            },
            ensure_ascii=False,
        )

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        snapshot = manager._agent_store.snapshot_agent_template("system-default-agent")
        assert snapshot is not None

        root_run = manager._run_store.create_run(
            RunRecord(
                id="run-root-1",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=snapshot,
                status="completed",
                goal="root trace diagnostics",
                created_at="2026-04-10T10:00:00+00:00",
                started_at="2026-04-10T10:00:01+00:00",
                finished_at="2026-04-10T10:00:08+00:00",
                current_step_index=2,
                run_metadata={
                    "kind": "root",
                    "agent_name": "main",
                    "root_run_id": "run-root-1",
                    "depth": 0,
                },
            )
        )
        child_run = manager._run_store.create_run(
            RunRecord(
                id="run-child-1",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=snapshot,
                status="completed",
                goal="child trace diagnostics",
                created_at="2026-04-10T10:00:02+00:00",
                started_at="2026-04-10T10:00:03+00:00",
                finished_at="2026-04-10T10:00:07+00:00",
                current_step_index=1,
                parent_run_id=root_run.id,
                run_metadata={
                    "kind": "delegate_child",
                    "agent_name": "worker-1",
                    "root_run_id": root_run.id,
                    "depth": 1,
                },
            )
        )

        manager._run_store.create_step(
            RunStepRecord(
                id="step-root-delegate",
                run_id=root_run.id,
                sequence=0,
                step_type="delegate",
                status="completed",
                title="delegate_task",
                input_summary='{"task": "delegate"}',
                output_summary="delegated",
                started_at="2026-04-10T10:00:02+00:00",
                finished_at="2026-04-10T10:00:07+00:00",
                error_summary="",
            )
        )
        manager._run_store.create_step(
            RunStepRecord(
                id="step-child-tool",
                run_id=child_run.id,
                sequence=0,
                step_type="tool_call",
                status="completed",
                title="write_file",
                input_summary='{"path": "docs/result.md"}',
                output_summary="wrote file",
                started_at="2026-04-10T10:00:03+00:00",
                finished_at="2026-04-10T10:00:05+00:00",
                error_summary="",
            )
        )

        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-root-1",
                run_id=root_run.id,
                sequence=0,
                event_type="run_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=root_run.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"step": 1},
                ),
                created_at="2026-04-10T10:00:01+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-root-2",
                run_id=root_run.id,
                sequence=1,
                event_type="delegate_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=root_run.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={
                        "step": 1,
                        "tool_call_id": "delegate-call-1",
                        "name": "delegate_task",
                        "tool_class": "DelegateTool",
                        "parameter_summary": '{"task": "delegate"}',
                        "risk_category": "delegate",
                        "risk_level": "medium",
                        "requires_approval": False,
                        "impact_summary": "Launch a child run for delegated work.",
                        "started_at": "2026-04-10T10:00:02+00:00",
                    },
                ),
                created_at="2026-04-10T10:00:02+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-child-1",
                run_id=child_run.id,
                parent_run_id=root_run.id,
                sequence=0,
                event_type="run_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=child_run.id,
                    agent_name="worker-1",
                    depth=1,
                    is_main_agent=False,
                    data={"step": 1},
                    parent_run_id=root_run.id,
                    root_run_id=root_run.id,
                ),
                created_at="2026-04-10T10:00:03+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-child-2",
                run_id=child_run.id,
                parent_run_id=root_run.id,
                sequence=1,
                event_type="tool_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=child_run.id,
                    agent_name="worker-1",
                    depth=1,
                    is_main_agent=False,
                    parent_run_id=root_run.id,
                    root_run_id=root_run.id,
                    data={
                        "step": 1,
                        "tool_call_id": "write-call-1",
                        "name": "write_file",
                        "tool_class": "WriteTool",
                        "parameter_summary": '{"path": "docs/result.md"}',
                        "risk_category": "filesystem_write",
                        "risk_level": "high",
                        "requires_approval": True,
                        "impact_summary": "Writes a file into the workspace.",
                        "started_at": "2026-04-10T10:00:03+00:00",
                    },
                ),
                created_at="2026-04-10T10:00:03+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-child-3",
                run_id=child_run.id,
                parent_run_id=root_run.id,
                sequence=2,
                event_type="tool_finished",
                status="running",
                duration_ms=1800,
                payload_summary=runtime_payload(
                    run_id=child_run.id,
                    agent_name="worker-1",
                    depth=1,
                    is_main_agent=False,
                    parent_run_id=root_run.id,
                    root_run_id=root_run.id,
                    data={
                        "step": 1,
                        "tool_call_id": "write-call-1",
                        "name": "write_file",
                        "tool_class": "WriteTool",
                        "parameter_summary": '{"path": "docs/result.md"}',
                        "risk_category": "filesystem_write",
                        "risk_level": "high",
                        "requires_approval": True,
                        "impact_summary": "Writes a file into the workspace.",
                        "success": True,
                        "content": "wrote result.md",
                        "artifacts": [
                            {
                                "artifact_type": "workspace_file",
                                "uri": "docs/result.md",
                                "summary": "Generated result document",
                                "metadata": {"source_tool": "write_file"},
                            }
                        ],
                        "finished_at": "2026-04-10T10:00:05+00:00",
                        "duration_ms": 1800,
                    },
                ),
                created_at="2026-04-10T10:00:05+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-child-4",
                run_id=child_run.id,
                parent_run_id=root_run.id,
                sequence=3,
                event_type="run_completed",
                status="completed",
                payload_summary=runtime_payload(
                    run_id=child_run.id,
                    agent_name="worker-1",
                    depth=1,
                    is_main_agent=False,
                    parent_run_id=root_run.id,
                    root_run_id=root_run.id,
                    data={"step": 1, "message": "child done"},
                ),
                created_at="2026-04-10T10:00:07+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-root-3",
                run_id=root_run.id,
                sequence=2,
                event_type="delegate_finished",
                status="running",
                duration_ms=5000,
                payload_summary=runtime_payload(
                    run_id=root_run.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={
                        "step": 1,
                        "tool_call_id": "delegate-call-1",
                        "name": "delegate_task",
                        "tool_class": "DelegateTool",
                        "parameter_summary": '{"task": "delegate"}',
                        "risk_category": "delegate",
                        "risk_level": "medium",
                        "requires_approval": False,
                        "impact_summary": "Launch a child run for delegated work.",
                        "success": True,
                        "content": "delegation completed",
                        "duration_ms": 5000,
                        "finished_at": "2026-04-10T10:00:07+00:00",
                    },
                ),
                created_at="2026-04-10T10:00:07+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="trace-root-4",
                run_id=root_run.id,
                sequence=3,
                event_type="run_completed",
                status="completed",
                payload_summary=runtime_payload(
                    run_id=root_run.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"step": 1, "message": "root done"},
                ),
                created_at="2026-04-10T10:00:08+00:00",
            )
        )

        manager._run_store.create_artifact(
            ArtifactRecord(
                id="artifact-child-1",
                run_id=child_run.id,
                step_id="step-child-tool",
                artifact_type="workspace_file",
                uri="docs/result.md",
                summary="Generated result document",
                metadata={"source_tool": "write_file"},
                created_at="2026-04-10T10:00:05+00:00",
            )
        )

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        (logs_dir / f"agent_run_20260410_180000_{root_run.id}_main.log").write_text(
            '{"run_id": "run-root-1"}',
            encoding="utf-8",
        )
        (logs_dir / f"agent_run_20260410_180001_{child_run.id}_worker-1.log").write_text(
            '{"run_id": "run-child-1"}',
            encoding="utf-8",
        )

        timeline_response = client.get(f"/api/runs/{root_run.id}/trace/timeline")
        assert timeline_response.status_code == 200
        timeline = timeline_response.json()
        assert [item["event_type"] for item in timeline] == [
            "run_started",
            "delegate_started",
            "run_started",
            "tool_started",
            "tool_finished",
            "run_completed",
            "delegate_finished",
            "run_completed",
        ]
        assert {item["run_id"] for item in timeline} == {root_run.id, child_run.id}
        assert any(item["agent_name"] == "worker-1" for item in timeline)

        tree_response = client.get(f"/api/runs/{root_run.id}/trace/tree")
        assert tree_response.status_code == 200
        tree = tree_response.json()
        assert tree["id"] == root_run.id
        assert tree["children"][0]["id"] == child_run.id
        assert tree["children"][0]["agent_name"] == "worker-1"
        assert tree["children"][0]["artifact_count"] == 1

        tools_response = client.get(f"/api/runs/{root_run.id}/trace/tools")
        assert tools_response.status_code == 200
        tool_calls = tools_response.json()
        assert [item["tool_name"] for item in tool_calls] == ["delegate_task", "write_file"]
        assert tool_calls[1]["requires_approval"] is True
        assert tool_calls[1]["duration_ms"] == 1800
        assert tool_calls[1]["artifacts"][0]["uri"] == "docs/result.md"

        export_response = client.get(f"/api/runs/{root_run.id}/trace/export")
        assert export_response.status_code == 200
        export_payload = export_response.json()
        assert export_payload["root_run"]["id"] == root_run.id
        assert export_payload["summary"] == {
            "run_count": 2,
            "trace_event_count": 8,
            "tool_call_count": 2,
            "artifact_count": 1,
        }
        assert export_payload["tree"]["children"][0]["id"] == child_run.id
        assert export_payload["tool_calls"][1]["tool_name"] == "write_file"

        replay_response = client.get(f"/api/runs/{root_run.id}/trace/replay")
        assert replay_response.status_code == 200
        replay = replay_response.json()
        assert replay["requested_run_id"] == root_run.id
        assert replay["root_run_id"] == root_run.id
        assert replay["playback"] == {
            "started_at": "2026-04-10T10:00:01+00:00",
            "finished_at": "2026-04-10T10:00:08+00:00",
            "duration_ms": 7000,
        }
        assert replay["summary"] == {
            "run_count": 2,
            "frame_count": 8,
            "tool_call_count": 2,
            "artifact_count": 1,
            "event_type_counts": {
                "delegate_finished": 1,
                "delegate_started": 1,
                "run_completed": 2,
                "run_started": 2,
                "tool_finished": 1,
                "tool_started": 1,
            },
        }
        assert [frame["event_type"] for frame in replay["frames"]] == [
            "run_started",
            "delegate_started",
            "run_started",
            "tool_started",
            "tool_finished",
            "run_completed",
            "delegate_finished",
            "run_completed",
        ]
        assert replay["frames"][1]["title"] == "Delegate started: delegate_task"
        assert replay["frames"][3]["tool_call"]["tool_name"] == "write_file"
        assert replay["frames"][4]["summary"] == "wrote result.md"
        assert replay["frames"][4]["relative_ms"] == 4000
        assert replay["runs"][1]["agent_name"] == "worker-1"

        diagnostics_response = client.get(f"/api/runs/{root_run.id}/diagnostics")
        assert diagnostics_response.status_code == 200
        diagnostics = diagnostics_response.json()
        assert diagnostics["requested_run_id"] == root_run.id
        assert diagnostics["database_path"].endswith("sessions.db")
        assert diagnostics["trace_event_count"] == 8
        assert diagnostics["tool_call_count"] == 2
        assert diagnostics["artifact_count"] == 1
        assert len(diagnostics["log_files"]) == 2
        assert any(root_run.id in path for path in diagnostics["log_files"])
        assert any(child_run.id in path for path in diagnostics["log_files"])


@patch("clavi_agent.session.LLMClient")
def test_run_started_trace_timeline_preserves_retrieved_context_body(mock_llm_class, tmp_path: Path):
    """run_started trace events should preserve prompt memory section bodies for the workspace UI."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="unused", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    asyncio.run(manager.initialize())
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        snapshot = manager._agent_store.snapshot_agent_template("system-default-agent")
        assert snapshot is not None

        run = manager._run_store.create_run(
            RunRecord(
                id="run-memory-trace-1",
                session_id=session["session_id"],
                account_id=ROOT_ACCOUNT_ID,
                agent_template_id="system-default-agent",
                agent_template_snapshot=snapshot,
                status="running",
                goal="inspect retrieved context body",
                created_at="2026-04-17T09:00:00+00:00",
                started_at="2026-04-17T09:00:01+00:00",
                run_metadata={
                    "kind": "root",
                    "agent_name": "main",
                    "root_run_id": "run-memory-trace-1",
                    "depth": 0,
                },
            )
        )

        retrieved_body = "\n".join(
            [
                f"- [history/session_message] 2026-04-17: 第 {index:02d} 条检索上下文内容，确认需要完整保留。"
                for index in range(1, 25)
            ]
        )
        payload_summary = json.dumps(
            {
                "context": {
                    "session_id": session["session_id"],
                    "run_id": run.id,
                    "agent_name": "main",
                    "is_main_agent": True,
                    "depth": 0,
                    "parent_run_id": None,
                    "root_run_id": run.id,
                },
                "data": {
                    "prompt": {
                        "memory_sections": [
                            {
                                "key": "retrieved_context",
                                "title": "Recent Retrieved Context",
                                "source": "retrieval",
                                "chars": len(retrieved_body),
                                "items": 24,
                                "sources": ["history:session-a", "memory:workflow_fact:entry-a"],
                                "body": retrieved_body,
                            }
                        ],
                        "memory_section_count": 1,
                        "memory_prompt_char_count": len(retrieved_body),
                    }
                },
            },
            ensure_ascii=False,
        )

        state = manager._run_manager._build_execution_state(run, append_user_message=False)
        manager._run_manager._record_trace(
            state=state,
            run=run,
            event_type="run_started",
            status="running",
            payload_summary=payload_summary,
        )

        stored_event = manager._trace_store.list_events(run.id, account_id=ROOT_ACCOUNT_ID)[0]
        assert len(stored_event.payload_summary) > 1000

        timeline_response = client.get(f"/api/runs/{run.id}/trace/timeline")
        assert timeline_response.status_code == 200
        timeline = timeline_response.json()
        assert len(timeline) == 1
        section = timeline[0]["data"]["prompt"]["memory_sections"][0]
        assert section["key"] == "retrieved_context"
        assert section["body"] == retrieved_body


@patch("clavi_agent.session.LLMClient")
def test_run_metrics_apis_expose_summary_and_prometheus_export(mock_llm_class, tmp_path: Path):
    """Run metrics APIs should aggregate root-run health and expose monitoring-friendly export text."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="unused", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    def runtime_payload(
        *,
        run_id: str,
        agent_name: str,
        depth: int,
        is_main_agent: bool,
        data: dict,
        parent_run_id: str | None = None,
        root_run_id: str | None = None,
    ) -> str:
        return json.dumps(
            {
                "context": {
                    "session_id": session["session_id"],
                    "run_id": run_id,
                    "agent_name": agent_name,
                    "is_main_agent": is_main_agent,
                    "depth": depth,
                    "parent_run_id": parent_run_id,
                    "root_run_id": root_run_id or run_id,
                },
                "data": data,
            },
            ensure_ascii=False,
        )

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        snapshot = manager._agent_store.snapshot_agent_template("system-default-agent")
        assert snapshot is not None

        completed_root = manager._run_store.create_run(
            RunRecord(
                id="metrics-root-completed",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=snapshot,
                status="completed",
                goal="successful metrics run",
                created_at="2026-04-10T10:00:00+00:00",
                started_at="2026-04-10T10:00:01+00:00",
                finished_at="2026-04-10T10:00:09+00:00",
                current_step_index=2,
                run_metadata={
                    "kind": "root",
                    "agent_name": "main",
                    "root_run_id": "metrics-root-completed",
                    "depth": 0,
                },
            )
        )
        child_run = manager._run_store.create_run(
            RunRecord(
                id="metrics-child-1",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=snapshot,
                status="completed",
                goal="delegated child metrics run",
                created_at="2026-04-10T10:00:02+00:00",
                started_at="2026-04-10T10:00:03+00:00",
                finished_at="2026-04-10T10:00:07+00:00",
                current_step_index=1,
                parent_run_id=completed_root.id,
                run_metadata={
                    "kind": "delegate_child",
                    "agent_name": "worker-1",
                    "root_run_id": completed_root.id,
                    "depth": 1,
                },
            )
        )
        failed_root = manager._run_store.create_run(
            RunRecord(
                id="metrics-root-failed",
                session_id=session["session_id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=snapshot,
                status="failed",
                goal="failed metrics run",
                created_at="2026-04-10T11:00:00+00:00",
                started_at="2026-04-10T11:00:01+00:00",
                finished_at="2026-04-10T11:00:05+00:00",
                current_step_index=1,
                error_summary="tool failure",
                run_metadata={
                    "kind": "root",
                    "agent_name": "main",
                    "root_run_id": "metrics-root-failed",
                    "depth": 0,
                },
            )
        )

        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-0",
                run_id=completed_root.id,
                sequence=0,
                event_type="run_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=completed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={
                        "llm": {
                            "profile_role": "planner",
                            "provider": "openai",
                            "model": "planner-model",
                            "reasoning_enabled": True,
                        }
                    },
                ),
                created_at="2026-04-10T10:00:01+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-1",
                run_id=completed_root.id,
                sequence=1,
                event_type="delegate_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=completed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"tool_call_id": "delegate-call-1", "name": "delegate_task"},
                ),
                created_at="2026-04-10T10:00:02+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-1a",
                run_id=completed_root.id,
                sequence=2,
                event_type="llm_request",
                status="running",
                payload_summary=runtime_payload(
                    run_id=completed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"step": 1},
                ),
                created_at="2026-04-10T10:00:02+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-2",
                run_id=child_run.id,
                parent_run_id=completed_root.id,
                sequence=0,
                event_type="run_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=child_run.id,
                    agent_name="worker-1",
                    depth=1,
                    is_main_agent=False,
                    parent_run_id=completed_root.id,
                    root_run_id=completed_root.id,
                    data={
                        "llm": {
                            "profile_role": "worker",
                            "provider": "openai",
                            "model": "worker-model",
                            "reasoning_enabled": False,
                        }
                    },
                ),
                created_at="2026-04-10T10:00:03+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-2a",
                run_id=child_run.id,
                parent_run_id=completed_root.id,
                sequence=1,
                event_type="llm_request",
                status="running",
                payload_summary=runtime_payload(
                    run_id=child_run.id,
                    agent_name="worker-1",
                    depth=1,
                    is_main_agent=False,
                    parent_run_id=completed_root.id,
                    root_run_id=completed_root.id,
                    data={"step": 1},
                ),
                created_at="2026-04-10T10:00:03+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-2b",
                run_id=child_run.id,
                parent_run_id=completed_root.id,
                sequence=2,
                event_type="tool_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=child_run.id,
                    agent_name="worker-1",
                    depth=1,
                    is_main_agent=False,
                    parent_run_id=completed_root.id,
                    root_run_id=completed_root.id,
                    data={"tool_call_id": "write-call-1", "name": "write_file"},
                ),
                created_at="2026-04-10T10:00:03+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-2c",
                run_id=completed_root.id,
                sequence=3,
                event_type="delegate_reviewed",
                status="accepted",
                payload_summary=runtime_payload(
                    run_id=completed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"action": "accepted"},
                ),
                created_at="2026-04-10T10:00:08+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-3",
                run_id=failed_root.id,
                sequence=0,
                event_type="run_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=failed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={
                        "llm": {
                            "profile_role": "planner",
                            "provider": "openai",
                            "model": "planner-model",
                            "reasoning_enabled": True,
                        }
                    },
                ),
                created_at="2026-04-10T11:00:01+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-3a",
                run_id=failed_root.id,
                sequence=1,
                event_type="delegate_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=failed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"tool_call_id": "delegate-call-2", "name": "delegate_tasks"},
                ),
                created_at="2026-04-10T11:00:01+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-3b",
                run_id=failed_root.id,
                sequence=2,
                event_type="llm_request",
                status="running",
                payload_summary=runtime_payload(
                    run_id=failed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"step": 1},
                ),
                created_at="2026-04-10T11:00:02+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-3c",
                run_id=failed_root.id,
                sequence=3,
                event_type="tool_started",
                status="running",
                payload_summary=runtime_payload(
                    run_id=failed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"tool_call_id": "shell-call-1", "name": "run_shell"},
                ),
                created_at="2026-04-10T11:00:02+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-3d",
                run_id=failed_root.id,
                sequence=4,
                event_type="tool_finished",
                status="failed",
                payload_summary=runtime_payload(
                    run_id=failed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={
                        "tool_call_id": "shell-call-1",
                        "name": "run_shell",
                        "policy_allowed": False,
                        "policy_denied_reason": "supervisor_only requires delegation",
                        "error": "Tool execution blocked by policy.",
                    },
                ),
                created_at="2026-04-10T11:00:02+00:00",
            )
        )
        manager._trace_store.create_event(
            TraceEventRecord(
                id="metrics-trace-3e",
                run_id=failed_root.id,
                sequence=5,
                event_type="delegate_reviewed",
                status="retry_delegated",
                payload_summary=runtime_payload(
                    run_id=failed_root.id,
                    agent_name="main",
                    depth=0,
                    is_main_agent=True,
                    data={"action": "retry_delegated"},
                ),
                created_at="2026-04-10T11:00:04+00:00",
            )
        )

        manager._approval_store.create_request(
            ApprovalRequestRecord(
                id="metrics-approval-1",
                run_id=child_run.id,
                tool_name="write_file",
                risk_level="high",
                status="granted",
                parameter_summary="docs/result.md",
                impact_summary="writes a workspace file",
                requested_at="2026-04-10T10:00:03+00:00",
                resolved_at="2026-04-10T10:00:06+00:00",
                decision_notes="approved",
            )
        )
        manager._approval_store.create_request(
            ApprovalRequestRecord(
                id="metrics-approval-2",
                run_id=failed_root.id,
                tool_name="run_shell",
                risk_level="high",
                status="pending",
                parameter_summary="rm -rf",
                impact_summary="destructive shell command",
                requested_at="2026-04-10T11:00:02+00:00",
            )
        )

        metrics_response = client.get(
            "/api/runs/metrics",
            params={"session_id": session["session_id"]},
        )
        assert metrics_response.status_code == 200
        metrics = metrics_response.json()
        assert metrics["filters"] == {
            "session_id": session["session_id"],
            "scope": "root_runs",
        }
        assert metrics["summary"] == {
            "root_run_count": 2,
            "total_run_count": 3,
            "child_run_count": 1,
            "terminal_run_count": 2,
            "completed_run_count": 1,
            "approval_request_count": 2,
            "pending_approval_count": 1,
            "success_rate": 0.5,
            "average_duration_ms": 6000,
            "average_tool_call_count": 1.0,
            "average_delegate_count": 1.0,
            "average_batch_delegate_count": 0.5,
            "average_approval_wait_ms": 3000,
            "batch_delegate_call_count": 1,
            "delegate_batch_usage_rate": 0.5,
            "forbidden_main_tool_attempt_count": 1,
            "reviewed_root_run_count": 2,
            "delegate_review_count": 2,
            "worker_first_pass_acceptance_rate": 0.5,
            "worker_rework_rate": 0.5,
            "total_llm_call_count": 3,
            "planner_llm_call_count": 2,
            "worker_llm_call_count": 1,
            "planner_llm_call_share": 0.6667,
            "worker_llm_call_share": 0.3333,
            "average_child_run_count": 0.5,
            "average_parallel_child_runs": 0.5,
            "duration_sample_count": 2,
            "approval_wait_sample_count": 1,
        }
        assert metrics["status_counts"] == {"completed": 1, "failed": 1}
        assert metrics["failure_type_distribution"] == {"failed": 1}
        assert metrics["delegate_review_action_counts"] == {
            "accepted": 1,
            "retry_delegated": 1,
        }
        assert metrics["llm_call_role_counts"] == {
            "planner": 2,
            "worker": 1,
        }

        export_response = client.get(
            "/api/runs/metrics/export",
            params={"session_id": session["session_id"], "format": "prometheus"},
        )
        assert export_response.status_code == 200
        assert export_response.headers["content-type"].startswith("text/plain")
        export_body = export_response.text
        assert (
            f'clavi_agent_root_runs_total{{session_id="{session["session_id"]}"}} 2'
            in export_body
        )
        assert (
            f'clavi_agent_run_status_total{{session_id="{session["session_id"]}",status="completed"}} 1'
            in export_body
        )
        assert (
            f'clavi_agent_run_failure_type_total{{failure_type="failed",session_id="{session["session_id"]}"}} 1'
            in export_body
        )
        assert (
            f'clavi_agent_approval_wait_ms_avg{{session_id="{session["session_id"]}"}} 3000'
            in export_body
        )
        assert (
            f'clavi_agent_delegate_tasks_usage_rate{{session_id="{session["session_id"]}"}} 0.5'
            in export_body
        )
        assert (
            f'clavi_agent_main_forbidden_tool_attempts_total{{session_id="{session["session_id"]}"}} 1'
            in export_body
        )
        assert (
            f'clavi_agent_planner_llm_calls_total{{session_id="{session["session_id"]}"}} 2'
            in export_body
        )
        assert (
            f'clavi_agent_delegate_review_action_total{{action="accepted",session_id="{session["session_id"]}"}} 1'
            in export_body
        )

        invalid_export = client.get(
            "/api/runs/metrics/export",
            params={"format": "csv"},
        )
        assert invalid_export.status_code == 400
        assert "Unsupported metrics export format" in invalid_export.json()["detail"]


@patch("clavi_agent.session.LLMClient")
def test_approval_decision_apis_resolve_pending_requests(mock_llm_class, tmp_path: Path):
    """Approval decision APIs should resolve pending requests and reject double resolution."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="reply one", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        run_response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "approval run",
            },
        )
        assert run_response.status_code == 201
        run_id = run_response.json()["id"]

        manager._approval_store.create_request(
            ApprovalRequestRecord(
                id="approval-grant-1",
                run_id=run_id,
                tool_name="BashTool",
                risk_level="high",
                status="pending",
                parameter_summary="git push",
                impact_summary="remote mutation",
                requested_at="2026-04-10T01:00:00+00:00",
            )
        )
        manager._approval_store.create_request(
            ApprovalRequestRecord(
                id="approval-deny-1",
                run_id=run_id,
                tool_name="WriteTool",
                risk_level="high",
                status="pending",
                parameter_summary="overwrite config",
                impact_summary="workspace mutation",
                requested_at="2026-04-10T01:00:01+00:00",
            )
        )

        grant_response = client.post(
            "/api/approvals/approval-grant-1/grant",
            json={"decision_notes": "approved for this run"},
        )
        assert grant_response.status_code == 200
        granted = grant_response.json()
        assert granted["status"] == "granted"
        assert granted["decision_notes"] == "approved for this run"
        assert granted["decision_scope"] == "once"
        assert granted["resolved_at"]

        deny_response = client.post(
            "/api/approvals/approval-deny-1/deny",
            json={"decision_notes": "unsafe mutation"},
        )
        assert deny_response.status_code == 200
        denied = deny_response.json()
        assert denied["status"] == "denied"
        assert denied["decision_notes"] == "unsafe mutation"
        assert denied["resolved_at"]

        granted_list = client.get("/api/approvals", params={"status": "granted"})
        assert granted_list.status_code == 200
        assert [item["id"] for item in granted_list.json()] == ["approval-grant-1"]

        denied_list = client.get("/api/approvals", params={"status": "denied"})
        assert denied_list.status_code == 200
        assert [item["id"] for item in denied_list.json()] == ["approval-deny-1"]

        second_grant = client.post("/api/approvals/approval-grant-1/grant")
        assert second_grant.status_code == 400
        assert "already resolved" in second_grant.json()["detail"]


@patch("clavi_agent.session.LLMClient")
def test_approval_grant_api_resumes_waiting_run(mock_llm_class, tmp_path: Path):
    """Granting a pending approval through the API should unblock the waiting run."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need approval before writing.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/from-api.md",
                                "content": "api approved body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="api approval completed", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        template = manager._agent_store.create_agent(
            name="Approver",
            description="Needs approval before writes",
            system_prompt="Request approval before writing files.",
            tools=["WriteTool"],
            approval_policy={
                "mode": "default",
                "require_approval_tools": ["write_file"],
            },
        )
        session = client.post(
            "/api/sessions",
            json={"agent_id": template["id"]},
        ).json()

        workspace_dir = tmp_path / "workspace-a"
        manager.bind_session_agent(
            session["session_id"],
            Agent(
                llm_client=manager._llm_client,
                system_prompt="You are a test assistant.",
                tools=[SimpleWriteTool(workspace_dir)],
                max_steps=5,
                workspace_dir=str(workspace_dir),
                config=manager._config,
            ),
        )

        run_response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "write after api approval",
            },
        )
        assert run_response.status_code == 201
        run_id = run_response.json()["id"]

        pending_request = None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pending = manager._approval_store.list_requests(status="pending", run_id=run_id)
            if pending:
                pending_request = pending[0]
                break
            time.sleep(0.05)

        assert pending_request is not None
        waiting_run = manager._run_store.get_run(run_id)
        assert waiting_run is not None
        assert waiting_run.status == "waiting_approval"

        grant_response = client.post(
            f"/api/approvals/{pending_request.id}/grant",
            json={"decision_notes": "approved via api"},
        )
        assert grant_response.status_code == 200
        assert grant_response.json()["status"] == "granted"
        assert grant_response.json()["decision_scope"] == "once"

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            persisted_run = manager._run_store.get_run(run_id)
            assert persisted_run is not None
            if persisted_run.status == "completed":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Approved run did not resume to completion in time.")

        run_detail = client.get(f"/api/runs/{run_id}")
        assert run_detail.status_code == 200
        assert run_detail.json()["status"] == "completed"

        approvals = client.get("/api/approvals", params={"status": "granted", "run_id": run_id})
        assert approvals.status_code == 200
        assert [item["id"] for item in approvals.json()] == [pending_request.id]

        written_file = workspace_dir / "docs" / "from-api.md"
        assert written_file.read_text(encoding="utf-8") == "api approved body"


@patch("clavi_agent.session.LLMClient")
def test_approval_template_scope_updates_agent_policy_and_skips_future_requests(
    mock_llm_class,
    tmp_path: Path,
):
    """Template-scoped approval grants should persist an auto-approve rule for future runs."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need approval before first write.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write_template_1",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/template-first.md",
                                "content": "template body one",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="first template run completed", finish_reason="stop"),
            LLMResponse(
                content="Write again after policy update.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write_template_2",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/template-second.md",
                                "content": "template body two",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="second template run completed", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        template = manager._agent_store.create_agent(
            name="Template Scoped Approver",
            description="Persists approval decisions for writes",
            system_prompt="Request approval before writing files.",
            tools=["WriteTool"],
            approval_policy={
                "mode": "default",
                "require_approval_tools": ["write_file"],
            },
        )
        session = client.post(
            "/api/sessions",
            json={"agent_id": template["id"]},
        ).json()

        workspace_dir = tmp_path / "workspace-template-scope"
        manager.bind_session_agent(
            session["session_id"],
            Agent(
                llm_client=manager._llm_client,
                system_prompt="You are a test assistant.",
                tools=[SimpleWriteTool(workspace_dir)],
                max_steps=6,
                workspace_dir=str(workspace_dir),
                config=manager._config,
            ),
        )

        first_run_response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "persist approval policy for writes",
            },
        )
        assert first_run_response.status_code == 201
        first_run_id = first_run_response.json()["id"]

        pending_request = None
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            pending = manager._approval_store.list_requests(status="pending", run_id=first_run_id)
            if pending:
                pending_request = pending[0]
                break
            time.sleep(0.05)

        assert pending_request is not None

        grant_response = client.post(
            f"/api/approvals/{pending_request.id}/grant",
            json={
                "decision_notes": "persist approval for this tool",
                "decision_scope": "template",
            },
        )
        assert grant_response.status_code == 200
        assert grant_response.json()["status"] == "granted"
        assert grant_response.json()["decision_scope"] == "template"

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            persisted_run = manager._run_store.get_run(first_run_id)
            assert persisted_run is not None
            if persisted_run.status == "completed":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Template-scoped approval run did not finish in time.")

        updated_template = manager._agent_store.get_agent_template(template["id"])
        assert updated_template is not None
        assert "write_file" in updated_template["approval_policy"]["auto_approve_tools"]

        second_run_response = client.post(
            "/api/runs",
            json={
                "session_id": session["session_id"],
                "goal": "write again after template approval update",
            },
        )
        assert second_run_response.status_code == 201
        second_run_id = second_run_response.json()["id"]

        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            persisted_run = manager._run_store.get_run(second_run_id)
            assert persisted_run is not None
            if persisted_run.status == "completed":
                break
            time.sleep(0.05)
        else:
            raise AssertionError("Second run did not finish after template policy update.")

        second_run_approvals = manager._approval_store.list_requests(run_id=second_run_id)
        assert second_run_approvals == []

        first_file = workspace_dir / "docs" / "template-first.md"
        second_file = workspace_dir / "docs" / "template-second.md"
        assert first_file.read_text(encoding="utf-8") == "template body one"
        assert second_file.read_text(encoding="utf-8") == "template body two"


@patch("clavi_agent.session.LLMClient")
def test_session_api_interrupt_endpoint_reports_idle_when_nothing_is_running(mock_llm_class, tmp_path: Path):
    """Interrupt endpoint should exist and report idle when there is no active run."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        session = client.post("/api/sessions", json={}).json()
        response = client.post(f"/api/sessions/{session['session_id']}/interrupt")

        assert response.status_code == 200
        assert response.json() == {
            "status": "idle",
            "session_id": session["session_id"],
        }


def test_parse_clawhub_search_output():
    """Plain text clawhub search results should become structured candidates."""
    parsed = _parse_clawhub_search_output(
        "a6-github-intel v1.0.0  GitHub Intelligence  (66.021)\n"
        "explorer v1.0.6  GitHub Projects Explorer  (43.979)\n"
    )

    assert parsed == [
        {
            "package_name": "a6-github-intel",
            "version": "v1.0.0",
            "description": "GitHub Intelligence",
            "score": "66.021",
            "label": "a6-github-intel v1.0.0",
        },
        {
            "package_name": "explorer",
            "version": "v1.0.6",
            "description": "GitHub Projects Explorer",
            "score": "43.979",
            "label": "explorer v1.0.6",
        },
    ]


def test_resolve_clawhub_command_prefix_uses_env_npm_prefix_on_linux(tmp_path: Path):
    """Linux deployments should honor NPM_CONFIG_PREFIX/bin without Windows-only paths."""
    prefix = tmp_path / "npm-prefix"
    clawhub_path = prefix / "bin" / "clawhub"
    clawhub_path.parent.mkdir(parents=True)
    clawhub_path.write_text("#!/bin/sh\n", encoding="utf-8")

    with patch("clavi_agent.runtime_tools.platform.system", return_value="Linux"), patch(
        "clavi_agent.runtime_tools.shutil.which",
        return_value=None,
    ), patch.dict(
        "clavi_agent.runtime_tools.os.environ",
        {"NPM_CONFIG_PREFIX": str(prefix)},
        clear=True,
    ), patch("clavi_agent.runtime_tools.subprocess.run") as mock_run:
        assert resolve_runtime_clawhub_command_prefix() == [str(clawhub_path)]
        mock_run.assert_not_called()


def test_resolve_clawhub_command_prefix_reads_npm_global_prefix_on_linux(tmp_path: Path):
    """Linux deployments should fall back to npm's configured global prefix."""
    prefix = tmp_path / "global-prefix"
    clawhub_path = prefix / "bin" / "clawhub"
    clawhub_path.parent.mkdir(parents=True)
    clawhub_path.write_text("#!/bin/sh\n", encoding="utf-8")
    npm_path = tmp_path / "bin" / "npm"
    npm_path.parent.mkdir(parents=True, exist_ok=True)
    npm_path.write_text("#!/bin/sh\n", encoding="utf-8")

    def fake_which(command: str) -> str | None:
        if command == "npm":
            return str(npm_path)
        return None

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        assert command == [str(npm_path), "config", "get", "prefix"]
        return subprocess.CompletedProcess(command, 0, stdout=str(prefix), stderr="")

    with patch("clavi_agent.runtime_tools.platform.system", return_value="Linux"), patch(
        "clavi_agent.runtime_tools.shutil.which",
        side_effect=fake_which,
    ), patch.dict("clavi_agent.runtime_tools.os.environ", {}, clear=True), patch(
        "clavi_agent.runtime_tools.subprocess.run",
        side_effect=fake_run,
    ):
        assert resolve_runtime_clawhub_command_prefix() == [str(clawhub_path)]


def test_server_resolve_clawhub_command_prefix_prefers_config_override(tmp_path: Path):
    """Server wrapper should use persisted config overrides before runtime search."""
    clawhub_path = tmp_path / "tools" / "clawhub"
    clawhub_path.parent.mkdir(parents=True)
    clawhub_path.write_text("#!/bin/sh\n", encoding="utf-8")

    with patch(
        "clavi_agent.server.Config.get_tool_path_overrides",
        return_value={"clawhub_bin": str(clawhub_path), "npm_bin": "/usr/bin/npm"},
    ), patch("clavi_agent.runtime_tools.shutil.which", return_value=None), patch.dict(
        "clavi_agent.server.os.environ",
        {},
        clear=True,
    ), patch.dict(
        "clavi_agent.runtime_tools.os.environ",
        {},
        clear=True,
    ):
        assert _resolve_clawhub_command_prefix() == [str(clawhub_path)]


def test_normalize_capability_dimensions():
    from clavi_agent.server import _normalize_capability_dimensions

    normalized = _normalize_capability_dimensions(
        [
            {"name": "内容搜索", "keyword": "抖音搜索", "reason": "先搜集素材"},
            {"capability": "视频制作", "search_keyword": "视频生成", "description": "再生成视频"},
            {"name": "内容搜索", "keyword": "抖音搜索", "reason": "重复项应被去重"},
        ]
    )

    assert normalized == [
        {"name": "内容搜索", "keyword": "抖音搜索", "reason": "先搜集素材"},
        {"name": "视频制作", "keyword": "视频生成", "reason": "再生成视频"},
    ]


@patch("clavi_agent.session.LLMClient")
def test_agent_skill_search_and_delete_api(mock_llm_class, tmp_path: Path):
    """Skill search endpoint should proxy registry results and delete should remove installed skills."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with patch("clavi_agent.server._search_clawhub_skills", new=AsyncMock(return_value=[
        {
            "package_name": "frontend-design",
            "version": "v1.0.0",
            "description": "Frontend workflow",
            "score": "91.0",
            "label": "frontend-design v1.0.0",
        }
    ])):
        with TestClient(app) as client:
            search_response = client.get("/api/skills/search", params={"keyword": "frontend"})
            assert search_response.status_code == 200
            assert search_response.json()[0]["package_name"] == "frontend-design"

            agent = manager._agent_store.create_agent(
                name="Designer",
                description="UI specialist",
                system_prompt="You are a UI specialist.",
                tools=[],
            )
            skill_dir = manager._agent_store.get_agent_skills_dir(agent["id"]) / "frontend-design"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                """---
name: frontend-design
description: Frontend workflow
---

Detailed skill content.
""",
                encoding="utf-8",
            )
            manager._agent_store.refresh_agent_skills_from_directory(agent["id"])

            delete_response = client.delete(f"/api/agents/{agent['id']}/skills/frontend-design")
            assert delete_response.status_code == 200
            body = delete_response.json()
            assert body["skills"] == []
            assert not skill_dir.exists()


@patch("clavi_agent.session.LLMClient")
def test_agent_brainstorm_api_uses_account_llm_runtime_and_parses_wrapped_json(
    mock_llm_class,
    tmp_path: Path,
):
    """Brainstorm endpoint should resolve the active account client and accept fenced JSON."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="""Here is the result:
```json
{
  "name": "短视频编导助手",
  "description": "帮助策划、撰写和优化短视频内容。",
  "system_prompt": "你是一名短视频编导助手，负责完成选题、脚本和发布建议。",
  "capability_dimensions": [
    {
      "name": "选题研究",
      "keyword": "短视频选题",
      "reason": "需要先明确热点和受众切入点。"
    },
    {
      "name": "脚本创作",
      "keyword": "脚本写作",
      "reason": "需要生成结构化脚本和镜头建议。"
    }
  ]
}
```""",
            finish_reason="stop",
        )
    )

    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with patch(
        "clavi_agent.server._search_clawhub_skills",
        new=AsyncMock(
            side_effect=[
                [
                    {
                        "package_name": "topic-skill",
                        "version": "1.0.0",
                        "description": "topic",
                        "score": "90",
                        "label": "topic-skill 1.0.0",
                    }
                ],
                [
                    {
                        "package_name": "script-skill",
                        "version": "1.0.0",
                        "description": "script",
                        "score": "88",
                        "label": "script-skill 1.0.0",
                    }
                ],
            ]
        ),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/api/agents/brainstorm",
                json={"name": "短视频助手", "description": "", "system_prompt": ""},
            )

    assert response.status_code == 200
    payload = response.json()
    assert payload["name"] == "短视频编导助手"
    assert payload["recommended_skill_keyword"] == "短视频选题"
    assert [group["capability_name"] for group in payload["skill_capability_groups"]] == [
        "选题研究",
        "脚本创作",
    ]


@patch("clavi_agent.session.LLMClient")
def test_agent_api_persists_template_policy_fields(mock_llm_class, tmp_path: Path):
    """Agent template APIs should persist marketplace policy fields for workspace, approval, and run defaults."""
    manager = SessionManager(config=build_config(tmp_path))
    app = create_app(manager)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/agents",
            json={
                "name": "Policy Driven Agent",
                "description": "Template policy coverage",
                "system_prompt": "Keep operations controlled.",
                "selected_skill_packages": [],
                "mcp_configs": [],
                "workspace_type": "shared",
                "workspace_policy": {
                    "mode": "shared",
                    "allow_session_override": False,
                    "readable_roots": ["docs", "reports", "logs"],
                    "writable_roots": ["docs", "reports"],
                    "read_only_tools": ["bash"],
                    "disabled_tools": ["delegate_task"],
                    "allowed_shell_command_prefixes": ["git status", "python -m pytest"],
                    "allowed_network_domains": ["example.com", "github.com"],
                },
                "approval_policy": {
                    "mode": "strict",
                    "require_approval_tools": ["write_file"],
                    "auto_approve_tools": ["read_file"],
                    "require_approval_risk_levels": ["high", "critical"],
                    "require_approval_risk_categories": ["external_network", "credentials"],
                    "notes": "Review writes carefully.",
                },
                "run_policy": {
                    "timeout_seconds": 90,
                    "max_concurrent_runs": 2,
                },
                "delegation_policy": {
                    "mode": "prefer_delegate",
                    "require_delegate_for_write_actions": True,
                    "prefer_batch_delegate": True,
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["workspace_policy"] == {
            "mode": "shared",
            "allow_session_override": False,
            "readable_roots": ["docs", "reports", "logs"],
            "writable_roots": ["docs", "reports"],
            "read_only_tools": ["bash"],
            "disabled_tools": ["delegate_task"],
            "allowed_shell_command_prefixes": ["git status", "python -m pytest"],
            "allowed_network_domains": ["example.com", "github.com"],
        }
        assert created["approval_policy"] == {
            "mode": "strict",
            "require_approval_tools": ["write_file"],
            "auto_approve_tools": ["read_file"],
            "require_approval_risk_levels": ["high", "critical"],
            "require_approval_risk_categories": ["external_network", "credentials"],
            "notes": "Review writes carefully.",
        }
        assert created["run_policy"] == {
            "timeout_seconds": 90,
            "max_concurrent_runs": 2,
        }
        assert created["delegation_policy"] == {
            "mode": "prefer_delegate",
            "require_delegate_for_write_actions": True,
            "require_delegate_for_shell": False,
            "require_delegate_for_stateful_mcp": False,
            "allow_main_agent_read_tools": True,
            "verify_worker_output": True,
            "prefer_batch_delegate": True,
        }

        stored = manager._agent_store.get_agent_template(created["id"])
        assert stored is not None
        assert stored["run_policy"]["timeout_seconds"] == 90
        assert stored["run_policy"]["max_concurrent_runs"] == 2
        assert stored["delegation_policy"]["mode"] == "prefer_delegate"

        update_response = client.put(
            f"/api/agents/{created['id']}",
            json={
                "workspace_type": "isolated",
                "workspace_policy": {
                    "mode": "isolated",
                    "allow_session_override": True,
                    "readable_roots": ["workspace"],
                    "writable_roots": [],
                    "read_only_tools": ["ReadTool"],
                    "disabled_tools": ["custom_review"],
                    "allowed_shell_command_prefixes": ["git status"],
                    "allowed_network_domains": ["internal.example.com"],
                },
                "approval_policy": {
                    "mode": "default",
                    "require_approval_tools": [],
                    "auto_approve_tools": ["read_file", "list_runs"],
                    "require_approval_risk_levels": ["critical"],
                    "require_approval_risk_categories": ["credentials"],
                    "notes": "Relaxed for diagnostics.",
                },
                "run_policy": {
                    "timeout_seconds": 45,
                    "max_concurrent_runs": 3,
                },
                "delegation_policy": {
                    "mode": "supervisor_only",
                    "allow_main_agent_read_tools": False,
                    "require_delegate_for_shell": True,
                },
            },
        )
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["workspace_policy"] == {
            "mode": "isolated",
            "allow_session_override": True,
            "readable_roots": ["workspace"],
            "writable_roots": [],
            "read_only_tools": ["ReadTool"],
            "disabled_tools": ["custom_review"],
            "allowed_shell_command_prefixes": ["git status"],
            "allowed_network_domains": ["internal.example.com"],
        }
        assert updated["approval_policy"] == {
            "mode": "default",
            "require_approval_tools": [],
            "auto_approve_tools": ["read_file", "list_runs"],
            "require_approval_risk_levels": ["critical"],
            "require_approval_risk_categories": ["credentials"],
            "notes": "Relaxed for diagnostics.",
        }
        assert updated["run_policy"] == {
            "timeout_seconds": 45,
            "max_concurrent_runs": 3,
        }
        assert updated["delegation_policy"] == {
            "mode": "supervisor_only",
            "require_delegate_for_write_actions": False,
            "require_delegate_for_shell": True,
            "require_delegate_for_stateful_mcp": False,
            "allow_main_agent_read_tools": False,
            "verify_worker_output": True,
            "prefer_batch_delegate": True,
        }
        assert updated["version"] == 2


@patch("clavi_agent.session.LLMClient")
async def test_session_manager_builds_prompt_from_agent_skills(mock_llm_class, tmp_path: Path):
    """Agent system prompt should append persisted skill summaries and install get_skill from the agent folder."""
    config = build_config(tmp_path, enable_skills=True, skills_dir=str(tmp_path / "unused-skills-dir"))
    manager = SessionManager(config=config)
    await manager.initialize()

    agent_config = manager._agent_store.create_agent(
        name="Designer",
        description="UI specialist",
        system_prompt="You are a UI specialist.",
        tools=[],
    )
    skill_dir = manager._agent_store.get_agent_skills_dir(agent_config["id"]) / "frontend-design"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: frontend-design
description: Frontend workflow
---

Detailed skill content.
""",
        encoding="utf-8",
    )
    manager._agent_store.refresh_agent_skills_from_directory(agent_config["id"])

    session_id = await manager.create_session(agent_id=agent_config["id"])
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "You are a UI specialist." in agent.messages[0].content
    assert "Available Skills" in agent.messages[0].content
    assert "`frontend-design`: Frontend workflow" in agent.messages[0].content
    expected_workspace = (
        tmp_path / "workspace" / "accounts" / "root" / "sessions" / session_id
    ).resolve()

    assert "get_skill" in agent.tools
    assert Path(agent.workspace_dir).resolve() == expected_workspace


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_isolated_agent_sessions_default_to_session_specific_workspace(
    mock_llm_class,
    tmp_path: Path,
):
    """Isolated template sessions should use a session-scoped workspace by default."""
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    agent_config = manager._agent_store.create_agent(
        name="Researcher",
        description="Research specialist",
        system_prompt="You are a research specialist.",
        tools=[],
    )

    session_id = await manager.create_session(agent_id=agent_config["id"])
    session_info = manager.get_session_info(session_id)

    expected_workspace = (
        tmp_path / "workspace" / "accounts" / "root" / "sessions" / session_id
    ).resolve()

    assert session_info is not None
    assert Path(session_info["workspace_dir"]).resolve() == expected_workspace
    assert expected_workspace.exists()


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_shared_agent_sessions_use_agent_specific_workspace(mock_llm_class, tmp_path: Path):
    """Shared template sessions should reuse the template's workspace directory."""
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    agent_config = manager._agent_store.create_agent(
        name="Shared Researcher",
        description="Research specialist",
        system_prompt="You are a research specialist.",
        tools=[],
        workspace_type="shared",
        workspace_policy={
            "mode": "shared",
            "allow_session_override": True,
        },
    )

    session_id = await manager.create_session(agent_id=agent_config["id"])
    session_info = manager.get_session_info(session_id)

    expected_workspace = manager._agent_store.get_agent_workspace_dir(agent_config["id"]).resolve()

    assert session_info is not None
    assert Path(session_info["workspace_dir"]).resolve() == expected_workspace
    assert expected_workspace.exists()

