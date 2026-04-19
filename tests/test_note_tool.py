"""Test cases for Session Note Tool."""

import tempfile
from pathlib import Path

import pytest

from clavi_agent.agent_template_models import AgentTemplateSnapshot
from clavi_agent.account_store import AccountStore
from clavi_agent.run_models import RunRecord, RunStepRecord
from clavi_agent.run_store import RunStore
from clavi_agent.schema import Message
from clavi_agent.session_store import SessionStore
from clavi_agent.tools.history_tool import SearchSessionHistoryTool
from clavi_agent.tools.note_tool import RecallNoteTool, SearchMemoryTool, SessionNoteTool
from clavi_agent.user_memory_store import UserMemoryStore


def create_account(db_path: Path, username: str) -> dict:
    account_store = AccountStore(db_path, auto_seed_root=False)
    return account_store.create_account(
        username=username,
        password="Secret123!",
        display_name=username.title(),
    )


@pytest.mark.asyncio
async def test_record_and_recall_notes():
    """Test recording and recalling notes."""
    print("\n=== Testing Note Record and Recall ===")

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".json") as f:
        note_file = f.name

    try:
        # Create tools
        record_tool = SessionNoteTool(memory_file=note_file)
        recall_tool = RecallNoteTool(memory_file=note_file)

        # Record a note
        result = await record_tool.execute(
            content="User prefers concise responses",
            category="user_preference",
        )
        assert result.success
        print(f"Record result: {result.content}")

        # Record another note
        result = await record_tool.execute(
            content="Project uses Python 3.12",
            category="project_info",
        )
        assert result.success
        print(f"Record result: {result.content}")

        # Recall all notes
        result = await recall_tool.execute()
        assert result.success
        assert "User prefers concise responses" in result.content
        assert "Python 3.12" in result.content
        print(f"\nAll notes:\n{result.content}")

        # Recall filtered by category
        result = await recall_tool.execute(category="user_preference")
        assert result.success
        assert "User prefers concise responses" in result.content
        assert "Python 3.12" not in result.content
        print(f"\nFiltered notes:\n{result.content}")

        print("✅ Note record and recall test passed")

    finally:
        Path(note_file).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_empty_notes():
    """Test recalling empty notes."""
    print("\n=== Testing Empty Notes ===")

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".json") as f:
        note_file = f.name

    # Delete the file to test empty state
    Path(note_file).unlink()

    try:
        recall_tool = RecallNoteTool(memory_file=note_file)

        # Recall empty notes
        result = await recall_tool.execute()
        assert result.success
        assert "No notes recorded yet" in result.content
        print(f"Empty notes result: {result.content}")

        print("✅ Empty notes test passed")

    finally:
        Path(note_file).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_note_persistence():
    """Test that notes persist across tool instances."""
    print("\n=== Testing Note Persistence ===")

    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False, suffix=".json") as f:
        note_file = f.name

    try:
        # First instance - record note
        record_tool1 = SessionNoteTool(memory_file=note_file)
        result = await record_tool1.execute(
            content="Important fact to remember",
            category="test",
        )
        assert result.success

        # Second instance - recall note (simulates new session)
        recall_tool2 = RecallNoteTool(memory_file=note_file)
        result = await recall_tool2.execute()
        assert result.success
        assert "Important fact to remember" in result.content
        print(f"Persisted note: {result.content}")

        print("✅ Note persistence test passed")

    finally:
        Path(note_file).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_structured_user_memory_and_profile_flow(tmp_path: Path):
    """Test structured user memory save/recall/search/profile behavior."""
    db_path = tmp_path / "agents.db"
    memory_file = tmp_path / ".agent_memory.json"
    account = create_account(db_path, "dora")

    record_tool = SessionNoteTool(
        memory_file=str(memory_file),
        user_id=account["id"],
        db_path=str(db_path),
        session_id="session-1",
        run_id="run-1",
    )
    recall_tool = RecallNoteTool(
        memory_file=str(memory_file),
        user_id=account["id"],
        db_path=str(db_path),
    )
    search_tool = SearchMemoryTool(
        memory_file=str(memory_file),
        user_id=account["id"],
        db_path=str(db_path),
    )

    profile_result = await record_tool.execute(
        content="用户偏好中文回答。",
        scope="user_profile",
        profile_updates={"preferred_language": "zh-CN", "response_length": "concise"},
        profile_summary="偏好中文，回复简洁。",
    )
    memory_result = await record_tool.execute(
        content="用户要求代码和文档读写统一使用 UTF-8。",
        category="correction",
        summary="UTF-8 是硬性要求。",
        confidence=0.95,
    )
    recall_result = await recall_tool.execute(scope="all", detailed=True)
    search_result = await search_tool.execute(query="UTF-8", scope="user_memory")
    audit_events = UserMemoryStore(db_path).list_audit_events(
        account["id"],
        target_scope="user_memory",
    )

    assert profile_result.success
    assert memory_result.success
    assert recall_result.success
    assert "preferred_language: zh-CN" in recall_result.content
    assert "UTF-8 是硬性要求。" in recall_result.content
    assert "writer: tool:record_note" in recall_result.content
    assert search_result.success
    assert "UTF-8 是硬性要求。" in search_result.content
    assert audit_events[0]["writer_id"] == "record_note"


@pytest.mark.asyncio
async def test_structured_user_memory_dedupes_and_supersedes(tmp_path: Path):
    """Test user memory dedupe and correction replacement semantics."""
    db_path = tmp_path / "agents.db"
    memory_file = tmp_path / ".agent_memory.json"
    account = create_account(db_path, "erin")
    record_tool = SessionNoteTool(
        memory_file=str(memory_file),
        user_id=account["id"],
        db_path=str(db_path),
        session_id="session-1",
        run_id="run-1",
    )
    store = UserMemoryStore(db_path)

    first = await record_tool.execute(
        content="提交前先跑针对性测试。",
        category="workflow",
        summary="提交前先验证。",
        confidence=0.6,
    )
    duplicate = await record_tool.execute(
        content="提交前先跑针对性测试。",
        category="workflow",
        summary="测试后再提交。",
        confidence=0.9,
    )
    existing_entries = store.list_memory_entries(
        account["id"],
        memory_types=["workflow_fact"],
    )

    original = store.create_memory_entry(
        user_id=account["id"],
        memory_type="constraint",
        content="旧要求：可以使用任意文件编码。",
        summary="旧编码要求。",
        confidence=0.2,
    )
    replacement = await record_tool.execute(
        content="新要求：所有读写统一使用 UTF-8。",
        memory_type="correction",
        summary="UTF-8 统一编码。",
        supersede_entry_id=original["id"],
        confidence=0.98,
    )
    all_entries = store.list_memory_entries(account["id"], include_superseded=True)

    assert first.success
    assert duplicate.success
    assert len(existing_entries) == 1
    assert existing_entries[0]["summary"] == "测试后再提交。"
    assert existing_entries[0]["confidence"] == 0.9
    assert replacement.success

    superseded_original = next(item for item in all_entries if item["id"] == original["id"])
    replacement_entry = next(
        item
        for item in all_entries
        if item["content"] == "新要求：所有读写统一使用 UTF-8。"
    )
    assert superseded_original["superseded_by"] == replacement_entry["id"]


@pytest.mark.asyncio
async def test_structured_user_memory_skips_secret_like_and_transient_noise(
    tmp_path: Path,
):
    """Test sensitive or transient content is not persisted as long-term memory."""
    db_path = tmp_path / "agents.db"
    memory_file = tmp_path / ".agent_memory.json"
    account = create_account(db_path, "frank")
    record_tool = SessionNoteTool(
        memory_file=str(memory_file),
        user_id=account["id"],
        db_path=str(db_path),
        session_id="session-guardrail",
        run_id="run-guardrail",
    )
    store = UserMemoryStore(db_path)

    secret_result = await record_tool.execute(
        content="OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz123456",
        memory_type="constraint",
        summary="真实密钥",
    )
    transient_result = await record_tool.execute(
        content=r"C:\Users\frank\AppData\Local\Temp\agent_trace.log",
        memory_type="project_fact",
        summary="临时日志路径",
    )
    raw_dump_result = await record_tool.execute(
        content="\n".join(f"line {index}: traceback detail" for index in range(25)),
        memory_type="workflow_fact",
        summary="调试堆栈",
    )

    assert secret_result.success
    assert "Skipped saving user memory" in secret_result.content
    assert transient_result.success
    assert "Skipped saving user memory" in transient_result.content
    assert raw_dump_result.success
    assert "Skipped saving user memory" in raw_dump_result.content
    assert store.list_memory_entries(account["id"]) == []


@pytest.mark.asyncio
async def test_structured_user_memory_merges_similar_entries_and_caps_capacity(
    tmp_path: Path,
):
    """Test similar memories merge and per-type capacity stays bounded."""
    db_path = tmp_path / "agents.db"
    memory_file = tmp_path / ".agent_memory.json"
    account = create_account(db_path, "gina")
    record_tool = SessionNoteTool(
        memory_file=str(memory_file),
        user_id=account["id"],
        db_path=str(db_path),
        session_id="session-capacity",
        run_id="run-capacity",
    )
    store = UserMemoryStore(db_path)

    first = await record_tool.execute(
        content="提交前先跑针对性测试。",
        memory_type="workflow_fact",
        summary="提交前先验证。",
        confidence=0.6,
    )
    merged = await record_tool.execute(
        content="提交代码前需要先跑针对性测试。",
        memory_type="workflow_fact",
        summary="提交流程要先测试。",
        confidence=0.9,
    )
    merged_entries = store.list_memory_entries(
        account["id"],
        memory_types=["workflow_fact"],
        include_superseded=False,
    )

    assert first.success
    assert merged.success
    assert "Merged into existing user memory entry" in merged.content
    assert len(merged_entries) == 1
    assert merged_entries[0]["content"] == "提交代码前需要先跑针对性测试。"
    assert merged_entries[0]["confidence"] == 0.9

    last_result = None
    for index in range(21):
        last_result = await record_tool.execute(
            content=f"长期目标 {index}: 将 goal_{index} 模块覆盖率提升到 {70 + index}%。",
            memory_type="goal",
            summary=f"目标 {index}",
            confidence=0.4 + (index / 100),
        )

    active_goals = store.list_memory_entries(
        account["id"],
        memory_types=["goal"],
        include_superseded=False,
    )
    all_goals = store.list_memory_entries(
        account["id"],
        memory_types=["goal"],
        include_superseded=True,
    )
    audit_events = store.list_audit_events(
        account["id"],
        target_scope="user_memory",
        limit=50,
    )

    assert last_result is not None
    assert last_result.success
    assert "compacted 1 older entries" in last_result.content
    assert len(active_goals) == 20
    assert len(all_goals) == 21
    assert any(item["superseded_by"] for item in all_goals)
    assert any(event["action"] == "memory_capacity_compact" for event in audit_events)


@pytest.mark.asyncio
async def test_search_session_history_tool_reads_cross_session_history(tmp_path: Path):
    """Session history tool should search persisted messages and run summaries."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    run_store = RunStore(db_path)
    session_store.create_session(
        session_id="session-1",
        workspace_dir=str(tmp_path),
        messages=[Message(role="system", content="system prompt")],
        account_id="account-1",
    )
    session_store.append_message(
        "session-1",
        message=Message(role="assistant", content="之前已经确认文档统一使用 UTF-8。"),
        account_id="account-1",
    )

    snapshot = AgentTemplateSnapshot(
        template_id="template-1",
        template_version=1,
        captured_at="2026-04-15T12:00:00+00:00",
        name="Historian",
        system_prompt="You are a historian.",
    )
    run_store.create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            account_id="account-1",
            agent_template_id="template-1",
            agent_template_snapshot=snapshot,
            status="completed",
            goal="回顾 UTF-8 编码约定",
            created_at="2026-04-15T12:00:00+00:00",
            started_at="2026-04-15T12:00:01+00:00",
            finished_at="2026-04-15T12:00:03+00:00",
        )
    )
    run_store.create_step(
        RunStepRecord(
            id="step-1",
            run_id="run-1",
            sequence=1,
            step_type="completion",
            status="completed",
            title="Run completed",
            output_summary="最终决定：仓库内文本文件一律采用 UTF-8 编码。",
            started_at="2026-04-15T12:00:02+00:00",
            finished_at="2026-04-15T12:00:03+00:00",
        )
    )

    tool = SearchSessionHistoryTool(
        db_path=str(db_path),
        account_id="account-1",
    )
    result = await tool.execute(query="UTF-8", agent_id="template-1", limit=5)

    assert result.success
    assert "Session History Search Results" in result.content
    assert "run_completion" in result.content
    assert "最终决定" in result.content


async def main():
    """Run all session note tool tests."""
    print("=" * 80)
    print("Running Session Note Tool Tests")
    print("=" * 80)

    await test_record_and_recall_notes()
    await test_empty_notes()
    await test_note_persistence()
    with tempfile.TemporaryDirectory() as tmpdir:
        await test_structured_user_memory_and_profile_flow(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        await test_structured_user_memory_dedupes_and_supersedes(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        await test_structured_user_memory_skips_secret_like_and_transient_noise(Path(tmpdir))
    with tempfile.TemporaryDirectory() as tmpdir:
        await test_structured_user_memory_merges_similar_entries_and_caps_capacity(Path(tmpdir))

    print("\n" + "=" * 80)
    print("All Session Note Tool tests passed! ✅")
    print("=" * 80)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

