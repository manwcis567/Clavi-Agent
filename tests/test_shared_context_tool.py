from pathlib import Path

import pytest

from clavi_agent.tools.shared_context_tool import ReadSharedContextTool, ShareContextTool


@pytest.mark.asyncio
async def test_shared_context_tools_support_cross_agent_handoffs(tmp_path: Path):
    """Multiple tool instances should publish to and read from the same board."""
    shared_file = tmp_path / "shared-context.json"
    planner = ShareContextTool(shared_file=str(shared_file), agent_name="planner")
    worker = ShareContextTool(shared_file=str(shared_file), agent_name="worker-1")
    reader = ReadSharedContextTool(shared_file=str(shared_file))

    first = await planner.execute(
        title="API contract",
        content="Use a session-scoped JSON board for all delegated workers.",
        category="decision",
    )
    second = await worker.execute(
        title="Risk",
        content="Concurrent writes need locking to avoid corrupted state.",
        category="risk",
    )
    result = await reader.execute()

    assert first.success is True
    assert second.success is True
    assert result.success is True
    assert "planner" in result.content
    assert "worker-1" in result.content
    assert "session-scoped JSON board" in result.content
    assert "Concurrent writes need locking" in result.content


@pytest.mark.asyncio
async def test_read_shared_context_filters_by_category_and_query(tmp_path: Path):
    """Readers should be able to narrow context to the most relevant entries."""
    shared_file = tmp_path / "shared-context.json"
    writer = ShareContextTool(shared_file=str(shared_file), agent_name="main")
    reader = ReadSharedContextTool(shared_file=str(shared_file))

    await writer.execute(content="Need debounce on websocket reconnect.", category="risk", title="Realtime")
    await writer.execute(content="Landing page copy approved.", category="decision", title="Content")

    risk_result = await reader.execute(category="risk")
    search_result = await reader.execute(query="landing")

    assert risk_result.success is True
    assert "debounce" in risk_result.content
    assert "Landing page copy approved" not in risk_result.content

    assert search_result.success is True
    assert "Landing page copy approved" in search_result.content
    assert "debounce" not in search_result.content


@pytest.mark.asyncio
async def test_read_shared_context_isolated_by_root_run_id(tmp_path: Path):
    """Readers bound to one run tree should ignore entries from another run tree."""
    shared_file = tmp_path / "shared-context.json"
    first_writer = ShareContextTool(
        shared_file=str(shared_file),
        agent_name="worker-1",
        run_id="child-run-1",
        parent_run_id="root-run-1",
        root_run_id="root-run-1",
    )
    second_writer = ShareContextTool(
        shared_file=str(shared_file),
        agent_name="worker-2",
        run_id="child-run-2",
        parent_run_id="root-run-2",
        root_run_id="root-run-2",
    )
    first_reader = ReadSharedContextTool(
        shared_file=str(shared_file),
        root_run_id="root-run-1",
    )

    await first_writer.execute(content="Only visible to run tree one.", category="decision")
    await second_writer.execute(content="Only visible to run tree two.", category="decision")

    result = await first_reader.execute(category="decision")

    assert result.success is True
    assert "run tree one" in result.content
    assert "run tree two" not in result.content


@pytest.mark.asyncio
async def test_shared_context_normalizes_standard_categories(tmp_path: Path):
    """单复数类别别名应落到同一个共享上下文分栏。"""
    shared_file = tmp_path / "shared-context.json"
    writer = ShareContextTool(shared_file=str(shared_file), agent_name="planner")
    reader = ReadSharedContextTool(shared_file=str(shared_file))

    write_result = await writer.execute(
        content="Document the delegated task contract before implementation.",
        category="requirement",
        title="Contract",
    )
    plural_read = await reader.execute(category="requirements")
    singular_read = await reader.execute(category="requirement")

    assert write_result.success is True
    assert "(category: requirements," in write_result.content
    assert plural_read.success is True
    assert singular_read.success is True
    assert "[requirements]" in plural_read.content
    assert "delegated task contract" in plural_read.content
    assert "delegated task contract" in singular_read.content

