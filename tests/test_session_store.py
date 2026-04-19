"""Tests for persistent session storage."""

from pathlib import Path

from clavi_agent.agent_template_models import AgentTemplateSnapshot
from clavi_agent.run_models import RunRecord, RunStepRecord
from clavi_agent.run_store import RunStore
from clavi_agent.account_constants import ROOT_ACCOUNT_ID
from clavi_agent.schema import FunctionCall, Message, ToolCall
from clavi_agent.session_store import DEFAULT_SESSION_TITLE, SessionStore


def test_session_store_round_trip(tmp_path: Path):
    """SessionStore should persist and restore messages without shape loss."""
    store = SessionStore(tmp_path / "sessions.db")
    messages = [
        Message(role="system", content="system prompt"),
        Message(role="user", content="hello world"),
        Message(
            role="assistant",
            content="working on it",
            thinking="reasoning",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    type="function",
                    function=FunctionCall(name="read_file", arguments={"path": "a.txt"}),
                )
            ],
        ),
        Message(
            role="tool",
            content="file content",
            tool_call_id="call-1",
            name="read_file",
        ),
    ]

    store.create_session(
        session_id="session-1",
        workspace_dir=str(tmp_path),
        messages=messages[:1],
    )
    store.replace_messages("session-1", messages)

    restored = store.get_messages("session-1")
    summary = store.get_session("session-1")

    assert restored == messages
    assert summary is not None
    assert summary["title"] == "hello world"
    assert summary["message_count"] == len(messages)
    assert summary["last_message_preview"] == "working on it"


def test_session_store_returns_typed_session_record_without_runtime_fields(tmp_path: Path):
    """Persisted session metadata should stay separate from in-memory runtime state."""
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(
        session_id="session-record",
        workspace_dir=str(tmp_path),
        messages=[Message(role="system", content="system prompt")],
        agent_id="agent-template-1",
    )

    record = store.get_session_record("session-record")

    assert record is not None
    assert record.session_id == "session-record"
    assert record.agent_id == "agent-template-1"
    assert "active_task" not in record.model_dump()
    assert "sub_agent_counter" not in record.model_dump()


def test_session_store_append_and_ordering(tmp_path: Path):
    """Sessions should sort by most recent update time."""
    store = SessionStore(tmp_path / "sessions.db")
    base_message = [Message(role="system", content="system prompt")]

    store.create_session("older", str(tmp_path), base_message, title=DEFAULT_SESSION_TITLE)
    store.create_session("newer", str(tmp_path), base_message, title=DEFAULT_SESSION_TITLE)

    store.append_message("older", Message(role="user", content="first chat"))
    store.append_message("newer", Message(role="user", content="second chat"))
    store.replace_messages(
        "newer",
        [Message(role="system", content="system prompt"), Message(role="user", content="second chat")],
    )

    sessions = store.list_sessions()

    assert [session["session_id"] for session in sessions][:2] == ["newer", "older"]


def test_session_store_delete_session(tmp_path: Path):
    """Deleting a session should remove both metadata and messages."""
    store = SessionStore(tmp_path / "sessions.db")
    store.create_session(
        session_id="session-delete",
        workspace_dir=str(tmp_path),
        messages=[Message(role="system", content="system prompt")],
    )

    deleted = store.delete_session("session-delete")

    assert deleted is True
    assert store.get_session("session-delete") is None
    assert store.get_messages("session-delete") == []


def test_session_store_round_trips_structured_user_messages(tmp_path: Path):
    """Structured user content should persist without degrading into raw JSON previews."""
    store = SessionStore(tmp_path / "sessions.db")
    structured_content = [
        {"type": "text", "text": "请修订这份草稿"},
        {
            "type": "uploaded_file",
            "upload_id": "upload-1",
            "original_name": "draft.md",
            "safe_name": "draft.md",
            "relative_path": ".clavi_agent/uploads/session-1/upload-1/draft.md",
            "mime_type": "text/markdown",
            "size_bytes": 128,
            "checksum": "abc123",
        },
    ]

    store.create_session(
        session_id="session-structured",
        workspace_dir=str(tmp_path),
        messages=[Message(role="system", content="system prompt")],
    )
    store.append_message(
        "session-structured",
        Message(role="user", content=structured_content),
    )

    restored = store.get_messages("session-structured")
    summary = store.get_session("session-structured")

    assert restored[1].content == structured_content
    assert summary is not None
    assert "请修订这份草稿" in summary["title"]
    assert "draft.md" in summary["last_message_preview"]
    assert "uploaded_file" not in summary["last_message_preview"]


def test_session_store_filters_by_account_id(tmp_path: Path):
    """Session queries should support explicit account scoping."""
    store = SessionStore(tmp_path / "sessions.db")
    base_message = [Message(role="system", content="system prompt")]

    store.create_session(
        "root-session",
        str(tmp_path),
        base_message,
        account_id=ROOT_ACCOUNT_ID,
    )
    store.create_session(
        "user-session",
        str(tmp_path),
        base_message,
        account_id="account-user",
    )

    user_sessions = store.list_sessions(account_id="account-user")

    assert [session["session_id"] for session in user_sessions] == ["user-session"]
    assert user_sessions[0]["account_id"] == "account-user"
    assert store.get_session("root-session", account_id="account-user") is None
    assert store.delete_session("user-session", account_id=ROOT_ACCOUNT_ID) is False
    assert store.delete_session("user-session", account_id="account-user") is True


def test_session_store_searches_cross_session_history_with_fts(tmp_path: Path):
    """History search should cover session messages, run goals, and run completion summaries."""
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
        Message(role="user", content="请记录：所有代码和文档统一使用 UTF-8 编码。"),
        account_id="account-1",
    )

    snapshot = AgentTemplateSnapshot(
        template_id="template-1",
        template_version=1,
        captured_at="2026-04-15T12:00:00+00:00",
        name="Coder",
        system_prompt="You are a coder.",
    )
    run_store.create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            account_id="account-1",
            agent_template_id="template-1",
            agent_template_snapshot=snapshot,
            status="completed",
            goal="确认仓库里的文本文件都改为 UTF-8 编码",
            created_at="2026-04-15T12:00:00+00:00",
            started_at="2026-04-15T12:00:01+00:00",
            finished_at="2026-04-15T12:00:05+00:00",
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
            output_summary="已确认后续文件读写统一使用 UTF-8，并补齐相关校验。",
            started_at="2026-04-15T12:00:04+00:00",
            finished_at="2026-04-15T12:00:05+00:00",
        )
    )

    results = session_store.search_history_records(
        "UTF-8",
        account_id="account-1",
        limit=10,
    )

    assert {"session_message", "run_goal", "run_completion"}.issubset(
        {item.source_type for item in results}
    )
    assert any(item.run_id == "run-1" and "UTF-8" in item.content for item in results)
    assert all(item.account_id == "account-1" for item in results)


def test_session_store_indexes_shared_context_entries_in_history_search(tmp_path: Path):
    """History search should include shared board decisions and findings."""
    session_store = SessionStore(tmp_path / "sessions.db")
    session_store.create_session(
        session_id="session-1",
        workspace_dir=str(tmp_path),
        messages=[Message(role="system", content="system prompt")],
        account_id="account-1",
    )
    session_store.append_shared_context_entry(
        "session-1",
        {
            "id": "ctx-1",
            "timestamp": "2026-04-16T10:00:00+00:00",
            "source": "worker-1",
            "category": "decision",
            "title": "编码约定",
            "content": "共享上下文确认所有文本资产继续统一使用 UTF-8 编码。",
            "run_id": "run-ctx",
        },
        account_id="account-1",
    )

    results = session_store.search_history_records(
        "UTF-8",
        account_id="account-1",
        source_types=["shared_context"],
        limit=10,
    )

    assert len(results) == 1
    assert results[0].source_type == "shared_context"
    assert results[0].run_id == "run-ctx"
    assert results[0].title == "编码约定"
    assert "共享上下文确认" in results[0].content


def test_session_store_history_search_can_exclude_current_run(tmp_path: Path):
    """History search should omit the current run when building retrieval context."""
    session_store = SessionStore(tmp_path / "sessions.db")
    run_store = RunStore(tmp_path / "sessions.db")

    session_store.create_session(
        session_id="session-1",
        workspace_dir=str(tmp_path),
        messages=[Message(role="system", content="system prompt")],
        account_id="account-1",
    )

    snapshot = AgentTemplateSnapshot(
        template_id="template-1",
        template_version=1,
        captured_at="2026-04-16T09:00:00+00:00",
        name="Coder",
        system_prompt="You are a coder.",
    )
    run_store.create_run(
        RunRecord(
            id="run-current",
            session_id="session-1",
            account_id="account-1",
            agent_template_id="template-1",
            agent_template_snapshot=snapshot,
            status="running",
            goal="UTF-8 约定",
            created_at="2026-04-16T09:00:00+00:00",
            started_at="2026-04-16T09:00:01+00:00",
        )
    )
    run_store.create_run(
        RunRecord(
            id="run-previous",
            session_id="session-1",
            account_id="account-1",
            agent_template_id="template-1",
            agent_template_snapshot=snapshot,
            status="completed",
            goal="UTF-8 约定",
            created_at="2026-04-15T09:00:00+00:00",
            started_at="2026-04-15T09:00:01+00:00",
            finished_at="2026-04-15T09:00:05+00:00",
        )
    )
    run_store.create_step(
        RunStepRecord(
            id="step-previous",
            run_id="run-previous",
            sequence=1,
            step_type="completion",
            status="completed",
            title="Run completed",
            output_summary="已确认 UTF-8 约定需要覆盖所有文档与代码文件。",
            started_at="2026-04-15T09:00:04+00:00",
            finished_at="2026-04-15T09:00:05+00:00",
        )
    )

    results = session_store.search_history_records(
        "UTF-8 约定",
        account_id="account-1",
        exclude_run_id="run-current",
        limit=10,
    )

    assert results
    assert all(item.run_id != "run-current" for item in results if item.run_id is not None)
    assert any(item.run_id == "run-previous" for item in results)


def test_session_store_history_search_can_filter_by_agent_id(tmp_path: Path):
    """History search should support agent-template scoped filtering."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    run_store = RunStore(db_path)

    session_store.create_session(
        session_id="session-a",
        workspace_dir=str(tmp_path / "workspace-a"),
        agent_id="agent-alpha",
        messages=[
            Message(role="system", content="system prompt"),
            Message(role="assistant", content="继续保持 UTF-8 编码。"),
        ],
        account_id="account-1",
    )
    session_store.create_session(
        session_id="session-b",
        workspace_dir=str(tmp_path / "workspace-b"),
        agent_id="agent-beta",
        messages=[
            Message(role="system", content="system prompt"),
            Message(role="assistant", content="另一个代理也提到 UTF-8 编码。"),
        ],
        account_id="account-1",
    )

    snapshot_alpha = AgentTemplateSnapshot(
        template_id="agent-alpha",
        template_version=1,
        captured_at="2026-04-16T11:00:00+00:00",
        name="Alpha",
        system_prompt="You are alpha.",
    )
    snapshot_beta = AgentTemplateSnapshot(
        template_id="agent-beta",
        template_version=1,
        captured_at="2026-04-16T11:00:00+00:00",
        name="Beta",
        system_prompt="You are beta.",
    )
    run_store.create_run(
        RunRecord(
            id="run-alpha",
            session_id="session-a",
            account_id="account-1",
            agent_template_id="agent-alpha",
            agent_template_snapshot=snapshot_alpha,
            status="completed",
            goal="整理 UTF-8 规范",
            created_at="2026-04-16T11:00:00+00:00",
            started_at="2026-04-16T11:00:01+00:00",
            finished_at="2026-04-16T11:00:02+00:00",
        )
    )
    run_store.create_run(
        RunRecord(
            id="run-beta",
            session_id="session-b",
            account_id="account-1",
            agent_template_id="agent-beta",
            agent_template_snapshot=snapshot_beta,
            status="completed",
            goal="整理 UTF-8 规范",
            created_at="2026-04-16T11:10:00+00:00",
            started_at="2026-04-16T11:10:01+00:00",
            finished_at="2026-04-16T11:10:02+00:00",
        )
    )

    results = session_store.search_history_records(
        "UTF-8",
        account_id="account-1",
        agent_id="agent-alpha",
        limit=10,
    )

    assert results
    assert {item.session_id for item in results} == {"session-a"}
    assert all(item.run_id in {None, "run-alpha"} for item in results)

