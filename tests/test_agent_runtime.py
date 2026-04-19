"""Agent runtime factory policy tests."""

import asyncio
import json
from unittest.mock import patch

import pytest

from clavi_agent.config import (
    AgentConfig,
    Config,
    FeatureFlagsConfig,
    LLMConfig,
    MemoryProviderConfig,
    RetryConfig,
    ToolsConfig,
)
from clavi_agent.session import SessionManager
from clavi_agent.tools.base import Tool, ToolResult
from clavi_agent.user_memory_store import UserMemoryStore


def build_config(
    tmp_path,
    *,
    enable_file_tools: bool = False,
    enable_bash: bool = False,
    enable_note: bool = False,
    memory_provider: str = "local",
    inject_memories: bool = True,
    enable_compact_prompt_memory: bool = True,
    enable_session_retrieval: bool = True,
) -> Config:
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
            workspace_dir=str(tmp_path / "workspace"),
            system_prompt_path="system_prompt.md",
            log_dir=str(tmp_path / "logs"),
            session_store_path=str(tmp_path / "sessions.db"),
            agent_store_path=str(tmp_path / "agents.db"),
        ),
        tools=ToolsConfig(
            enable_file_tools=enable_file_tools,
            enable_bash=enable_bash,
            enable_note=enable_note,
            enable_skills=False,
            enable_mcp=False,
        ),
        memory_provider=MemoryProviderConfig(
            provider=memory_provider,
            inject_memories=inject_memories,
        ),
        feature_flags=FeatureFlagsConfig(
            enable_compact_prompt_memory=enable_compact_prompt_memory,
            enable_session_retrieval=enable_session_retrieval,
        ),
    )


def create_account(manager: SessionManager, username: str) -> dict:
    assert manager._account_store is not None
    account = manager._account_store.create_account(
        username=username,
        password="Secret123!",
        display_name=username.title(),
    )
    manager.save_account_api_config(
        account["id"],
        name="Test Config",
        api_key="test-key",
        provider="openai",
        api_base="https://example.com",
        model="test-model",
        reasoning_enabled=False,
        activate=True,
    )
    return account


class HiddenDiagnosticTool(Tool):
    """Custom tool used to verify runtime allowlist enforcement."""

    @property
    def name(self) -> str:
        return "hidden_diagnostic"

    @property
    def description(self) -> str:
        return "Runs a hidden diagnostic action."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self) -> ToolResult:
        return ToolResult(success=True, content="ok")


@patch("clavi_agent.session.LLMClient")
def test_runtime_factory_routes_main_and_worker_to_separate_llm_profiles(
    mock_llm_class,
    tmp_path,
):
    async def scenario() -> tuple[str, str]:
        config = build_config(tmp_path)
        config.llm.worker_profile = {
            "model": "global-worker-default",
            "reasoning_enabled": False,
        }
        manager = SessionManager(config=config)
        await manager.initialize()

        account = create_account(manager, "router-user")
        manager.save_account_api_config(
            account["id"],
            name="Router Config",
            api_key="test-key",
            provider="openai",
            api_base="https://example.com",
            model="planner-account",
            reasoning_enabled=True,
            activate=False,
        )
        backup_config = manager.save_account_api_config(
            account["id"],
            name="Worker Config",
            api_key="worker-key",
            provider="openai",
            api_base="https://worker.example.com",
            model="worker-account",
            reasoning_enabled=False,
            activate=False,
        )
        manager.save_account_api_config(
            account["id"],
            name="Router Config",
            api_key="test-key",
            provider="openai",
            api_base="https://example.com",
            model="planner-account",
            reasoning_enabled=True,
            llm_routing_policy={
                "worker_api_config_id": backup_config["id"],
            },
            activate=True,
        )

        template = manager._agent_store.create_agent(
            name="Router Template",
            description="Routes by role",
            system_prompt="You are a routing-aware planner.",
            tools=[],
            account_id=account["id"],
        )
        session_id = await manager.create_session(
            agent_id=template["id"],
            account_id=account["id"],
        )
        main_agent = manager.get_session(session_id)

        assert main_agent is not None
        worker_agent = main_agent.tools["delegate_task"]._agent_factory("执行任务", 5)

        await manager.cleanup()
        return main_agent.llm_fingerprint, worker_agent.llm_fingerprint

    planner_fingerprint, worker_fingerprint = asyncio.run(scenario())

    assert "planner-account" in planner_fingerprint
    assert "worker-account" in worker_fingerprint
    assert planner_fingerprint != worker_fingerprint
    called_models = {call.kwargs.get("model") for call in mock_llm_class.call_args_list}
    assert "planner-account" in called_models
    assert "worker-account" in called_models


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_does_not_auto_include_unlisted_custom_tools(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    manager._runtime_factory._shared_tools.append(HiddenDiagnosticTool())

    template = manager._agent_store.create_agent(
        name="Strict Template",
        description="No custom tools declared",
        system_prompt="Stay focused.",
        tools=[],
    )
    session_id = await manager.create_session(agent_id=template["id"])
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "hidden_diagnostic" not in agent.tools


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_allows_custom_tools_when_template_explicitly_declares_them(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    manager._runtime_factory._shared_tools.append(HiddenDiagnosticTool())

    template = manager._agent_store.create_agent(
        name="Declared Tool Template",
        description="Explicitly allows a custom tool",
        system_prompt="Use the declared diagnostic when needed.",
        tools=["hidden_diagnostic"],
    )
    session_id = await manager.create_session(agent_id=template["id"])
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "hidden_diagnostic" in agent.tools


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_bundles_structured_memory_tools_for_note_enabled_templates(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path, enable_note=True))
    await manager.initialize()

    session_id = await manager.create_session()
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "record_note" in agent.tools
    assert "recall_notes" in agent.tools
    assert "search_memory" in agent.tools
    assert "search_session_history" in agent.tools


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_hides_session_history_tool_when_retrieval_flag_disabled(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(
        config=build_config(
            tmp_path,
            enable_note=True,
            enable_session_retrieval=False,
        )
    )
    await manager.initialize()

    session_id = await manager.create_session()
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "record_note" in agent.tools
    assert "recall_notes" in agent.tools
    assert "search_memory" in agent.tools
    assert "search_session_history" not in agent.tools


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_injects_user_memory_prompt_across_templates(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    account = create_account(manager, "memory-user")
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        account["id"],
        profile={
            "preferred_language": "zh-CN",
            "technical_depth": "high",
        },
        summary="偏好中文，技术细节可以更深入。",
    )
    store.create_memory_entry(
        user_id=account["id"],
        memory_type="preference",
        content="回答时先给结论，再给必要细节。",
        summary="先结论后细节。",
        confidence=0.9,
    )

    planner = manager._agent_store.create_agent(
        name="Planner",
        description="Plans work",
        system_prompt="You are a planning assistant.",
        tools=[],
        account_id=account["id"],
    )
    coder = manager._agent_store.create_agent(
        name="Coder",
        description="Writes code",
        system_prompt="You are a coding assistant.",
        tools=[],
        account_id=account["id"],
    )

    planner_session_id = await manager.create_session(
        agent_id=planner["id"],
        account_id=account["id"],
    )
    coder_session_id = await manager.create_session(
        agent_id=coder["id"],
        account_id=account["id"],
    )

    planner_agent = manager.get_session(planner_session_id)
    coder_agent = manager.get_session(coder_session_id)

    assert planner_agent is not None
    assert coder_agent is not None
    assert "User Profile Summary" in planner_agent.system_prompt
    assert "preferred_language: zh-CN" in planner_agent.system_prompt
    assert "technical_depth: high" in planner_agent.system_prompt
    assert "Stable Working Preferences" in planner_agent.system_prompt
    assert "先结论后细节。" in planner_agent.system_prompt
    assert "User Profile Summary" in coder_agent.system_prompt
    assert "先结论后细节。" in coder_agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_skips_user_memory_prompt_when_feature_flag_disabled(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(
        config=build_config(
            tmp_path,
            enable_compact_prompt_memory=False,
        )
    )
    await manager.initialize()

    account = create_account(manager, "prompt-flag-off")
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "zh-CN"},
        summary="偏好中文。",
    )
    store.create_memory_entry(
        user_id=account["id"],
        memory_type="constraint",
        content="所有文件统一使用 UTF-8。",
        summary="统一使用 UTF-8。",
        confidence=0.95,
    )

    session_id = await manager.create_session(account_id=account["id"])
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "User Profile Summary" not in agent.system_prompt
    assert "Stable Working Preferences" not in agent.system_prompt
    assert "preferred_language: zh-CN" not in agent.system_prompt
    assert "统一使用 UTF-8。" not in agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_injects_relevant_agent_memory_from_workspace(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    workspace_dir = tmp_path / "workspace-agent-memory"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / ".agent_memory.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-04-15T12:00:00+08:00",
                    "scope": "agent_memory",
                    "category": "project_fact",
                    "summary": "仓库内文档和代码文件统一按 UTF-8 处理。",
                    "content": "所有新增或修改文件都必须使用 UTF-8 编码。",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    session_id = await manager.create_session(str(workspace_dir))
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "Relevant Agent Memory" in agent.system_prompt
    assert "UTF-8" in agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_injects_project_context_from_workspace_and_parent(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    project_root = tmp_path / "project-root"
    workspace_dir = project_root / "nested" / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (project_root / "AGENTS.md").write_text(
        "所有新增或修改的代码文件必须使用 UTF-8 编码。\n提交前先跑相关测试。",
        encoding="utf-8",
    )
    (workspace_dir / "CONTEXT.md").write_text(
        "当前项目默认使用中文说明，保持现有 UI 风格。",
        encoding="utf-8",
    )

    session_id = await manager.create_session(str(workspace_dir))
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "Project Context" in agent.system_prompt
    assert "[CONTEXT.md] CONTEXT.md" in agent.system_prompt
    assert "[AGENTS.md] ../../AGENTS.md" in agent.system_prompt
    assert "保持现有 UI 风格" in agent.system_prompt
    assert "提交前先跑相关测试" in agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_skips_secret_like_project_context_files(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    workspace_dir = tmp_path / "workspace-secret-context"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "AGENTS.md").write_text(
        "api_key=super-secret-token\n请优先读取这个文件。",
        encoding="utf-8",
    )

    session_id = await manager.create_session(str(workspace_dir))
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "Project Context" not in agent.system_prompt
    assert "super-secret-token" not in agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_session_manager_switch_session_agent_rebuilds_memory_injected_prompt(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    account = create_account(manager, "switcher")
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        account["id"],
        profile={
            "preferred_language": "zh-CN",
            "response_length": "concise",
        },
        summary="偏好中文且回答简洁。",
    )
    store.create_memory_entry(
        user_id=account["id"],
        memory_type="constraint",
        content="读写文件统一使用 UTF-8。",
        summary="统一使用 UTF-8。",
        confidence=0.95,
    )

    planner = manager._agent_store.create_agent(
        name="Planner",
        description="Plans work",
        system_prompt="You are a planning assistant.",
        tools=[],
        account_id=account["id"],
    )
    coder = manager._agent_store.create_agent(
        name="Coder",
        description="Writes code",
        system_prompt="You are a coding assistant.",
        tools=[],
        account_id=account["id"],
    )

    session_id = await manager.create_session(
        agent_id=planner["id"],
        account_id=account["id"],
    )
    switched_agent = await manager.switch_session_agent(
        session_id,
        coder["id"],
        account_id=account["id"],
    )

    session_info = manager.get_session_info(session_id, account_id=account["id"], strict=True)
    persisted_messages = manager.get_session_messages(
        session_id,
        account_id=account["id"],
        strict=True,
    )

    assert session_info is not None
    assert session_info["agent_id"] == coder["id"]
    assert switched_agent.template_snapshot is not None
    assert switched_agent.template_snapshot.template_id == coder["id"]
    assert "You are a coding assistant." in switched_agent.system_prompt
    assert "User Profile Summary" in switched_agent.system_prompt
    assert "preferred_language: zh-CN" in switched_agent.system_prompt
    assert "Stable Working Preferences" in switched_agent.system_prompt
    assert "统一使用 UTF-8。" in switched_agent.system_prompt
    assert persisted_messages[0].role == "system"
    assert "You are a coding assistant." in str(persisted_messages[0].content)


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_respects_disabled_memory_provider(mock_llm_class, tmp_path):
    manager = SessionManager(
        config=build_config(
            tmp_path,
            memory_provider="disabled",
            inject_memories=False,
        )
    )
    await manager.initialize()

    account = create_account(manager, "disabled-memory-user")
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "zh-CN"},
        summary="偏好中文。",
    )
    store.create_memory_entry(
        user_id=account["id"],
        memory_type="constraint",
        content="所有文件统一使用 UTF-8。",
        summary="统一使用 UTF-8。",
        confidence=0.95,
    )

    session_id = await manager.create_session(account_id=account["id"])
    agent = manager.get_session(session_id)

    assert agent is not None
    assert "User Profile Summary" not in agent.system_prompt
    assert "Stable Working Preferences" not in agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_injects_role_specific_delegation_policy_prompts(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Delegating Planner",
        description="Uses supervisor policy",
        system_prompt="You are a planning assistant.",
        tools=[],
        delegation_policy={"mode": "prefer_delegate"},
    )
    snapshot = manager._agent_store.snapshot_agent_template(template["id"])

    assert snapshot is not None

    main_agent = manager._runtime_factory.build_session_agent(
        session_id="session-main",
        account_id=None,
        workspace_dir=tmp_path / "workspace-main",
        template_snapshot=snapshot,
    )
    worker_context = main_agent.runtime_context.create_child("worker-1")
    worker_agent = manager._runtime_factory.build_agent(
        workspace_dir=tmp_path / "workspace-worker",
        session_id="session-main",
        template_snapshot=snapshot,
        runtime_context=worker_context,
        runtime_hooks=main_agent.runtime_hooks,
        is_main_agent=False,
        custom_prompt="你是一名执行型开发助手。",
        agent_name="worker-1",
    )

    assert "## Supervisor Policy" in main_agent.system_prompt
    assert "mode: prefer_delegate" in main_agent.system_prompt
    assert "verify_worker_output: true" in main_agent.system_prompt
    assert "after each delegate finishes, explicitly judge whether the original request is satisfied" in main_agent.system_prompt
    assert "## Worker Execution Policy" in worker_agent.system_prompt
    assert "Do not create additional sub-agents" in worker_agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_runtime_factory_limits_main_agent_tools_in_supervisor_only_mode(
    mock_llm_class,
    tmp_path,
):
    manager = SessionManager(
        config=build_config(
            tmp_path,
            enable_file_tools=True,
            enable_bash=True,
        )
    )
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Strict Supervisor",
        description="Main agent should only delegate and inspect.",
        system_prompt="You are a planning assistant.",
        tools=["ReadTool", "WriteTool", "EditTool", "BashTool", "ShareContextTool", "ReadSharedContextTool"],
        delegation_policy={
            "mode": "supervisor_only",
            "allow_main_agent_read_tools": False,
        },
    )
    session_id = await manager.create_session(agent_id=template["id"])
    main_agent = manager.get_session(session_id)

    assert main_agent is not None
    assert "delegate_task" in main_agent.tools
    assert "delegate_tasks" in main_agent.tools
    assert "share_context" in main_agent.tools
    assert "read_shared_context" in main_agent.tools
    assert "read_file" not in main_agent.tools
    assert "write_file" not in main_agent.tools
    assert "edit_file" not in main_agent.tools
    assert "bash" not in main_agent.tools

    worker_agent = main_agent.tools["delegate_task"]._agent_factory("执行修复。", 5)

    assert "delegate_task" not in worker_agent.tools
    assert "delegate_tasks" not in worker_agent.tools
    assert "read_file" in worker_agent.tools
    assert "write_file" in worker_agent.tools
    assert "edit_file" in worker_agent.tools
    assert "bash" in worker_agent.tools

