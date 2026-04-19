from pathlib import Path
from unittest.mock import patch

import pytest

from clavi_agent.config import AgentConfig, Config, LLMConfig, RetryConfig, ToolsConfig
from clavi_agent.session import SessionManager


def build_config(tmp_path: Path) -> Config:
    """Build a lightweight config for session manager tests."""
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
        ),
        tools=ToolsConfig(
            enable_file_tools=False,
            enable_bash=False,
            enable_note=False,
            enable_skills=False,
            enable_mcp=False,
        ),
    )


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_main_and_sub_agents_share_the_same_session_board(mock_llm_class, tmp_path: Path):
    """Delegated workers should see and update the same shared context as the main agent."""
    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace"))
    main_agent = manager.get_session(session_id)

    assert main_agent is not None
    delegate_tool = main_agent.tools["delegate_task"]
    assert "delegate_tasks" in main_agent.tools
    main_share = main_agent.tools["share_context"]
    main_read = main_agent.tools["read_shared_context"]

    sub_agent = delegate_tool._agent_factory("You are a focused worker.", 4)
    assert "delegate_tasks" not in sub_agent.tools
    sub_share = sub_agent.tools["share_context"]
    sub_read = sub_agent.tools["read_shared_context"]

    published = await main_share.execute(
        content="Follow the repository coding style and keep edits minimal.",
        category="requirement",
        title="Global guardrails",
    )
    seen_by_worker = await sub_read.execute(category="requirement")

    assert published.success is True
    assert seen_by_worker.success is True
    assert "Global guardrails" in seen_by_worker.content
    assert "keep edits minimal" in seen_by_worker.content

    worker_note = await sub_share.execute(
        content="Found the best hook point in SessionManager workspace tool injection.",
        category="finding",
        title="Integration point",
    )
    seen_by_main = await main_read.execute(category="finding")

    assert worker_note.success is True
    assert seen_by_main.success is True
    assert "worker-1" in seen_by_main.content
    assert "SessionManager workspace tool injection" in seen_by_main.content


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_shared_context_is_isolated_per_session(mock_llm_class, tmp_path: Path):
    """Two sessions in the same workspace should not read each other's shared board."""
    manager = SessionManager(config=build_config(tmp_path))
    workspace = str(tmp_path / "workspace")

    first_session = await manager.create_session(workspace)
    second_session = await manager.create_session(workspace)

    first_agent = manager.get_session(first_session)
    second_agent = manager.get_session(second_session)

    await first_agent.tools["share_context"].execute(
        content="Session one requirement",
        category="requirement",
    )
    second_read = await second_agent.tools["read_shared_context"].execute()

    assert second_read.success is True
    assert "No shared context has been published yet." == second_read.content

