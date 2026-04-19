"""Tests for durable run persistence repositories."""

import sqlite3
from pathlib import Path

from clavi_agent.account_constants import ROOT_ACCOUNT_ID
from clavi_agent.agent_template_models import AgentTemplateSnapshot
from clavi_agent.approval_store import ApprovalStore
from clavi_agent.run_models import (
    ApprovalRequestRecord,
    ArtifactRecord,
    CheckpointMessageSnapshot,
    PendingToolCallSnapshot,
    RunCheckpointPayload,
    RunCheckpointRecord,
    RunDeliverableManifest,
    RunDeliverableRef,
    RunRecord,
    RunStepRecord,
    SharedContextRef,
    SubAgentCheckpointSummary,
    TraceEventRecord,
    TriggerMessageRef,
)
from clavi_agent.run_store import RunStore
from clavi_agent.session_store import SessionStore
from clavi_agent.sqlite_schema import CURRENT_SESSION_DB_VERSION, SESSION_DB_SCOPE
from clavi_agent.trace_store import TraceStore
from clavi_agent.upload_models import UploadRecord
from clavi_agent.upload_store import UploadStore


def build_snapshot() -> AgentTemplateSnapshot:
    """Build a minimal template snapshot for store tests."""
    return AgentTemplateSnapshot(
        template_id="template-1",
        template_version=2,
        captured_at="2026-04-09T12:00:00+00:00",
        name="Researcher",
        system_prompt="You are a researcher.",
    )


def test_session_store_migrates_legacy_messages_table_and_creates_durable_schema(tmp_path: Path):
    """Legacy session DBs should migrate in place to the durable schema."""
    db_path = tmp_path / "sessions.db"
    store = SessionStore(db_path)
    store.create_session(
        session_id="session-1",
        workspace_dir=str(tmp_path),
        messages=[],
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("ALTER TABLE session_messages RENAME TO messages")

    migrated_store = SessionStore(db_path)

    with migrated_store._connect() as conn:
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
        checkpoint_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(run_checkpoints)").fetchall()
        }
        run_columns = {row["name"] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        artifact_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(artifacts)").fetchall()
        }
        session_message_columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(session_messages)").fetchall()
        }
        version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (SESSION_DB_SCOPE,),
        ).fetchone()

    assert "messages" not in tables
    assert {
        "sessions",
        "session_messages",
        "runs",
        "run_steps",
        "run_checkpoints",
        "artifacts",
        "uploads",
        "approval_requests",
        "trace_events",
        "session_history_fts",
    }.issubset(tables)
    assert {
        "idx_runs_status_created_at",
        "idx_trace_events_run_created_at",
        "idx_approval_requests_status_requested_at",
    }.issubset(indexes)
    assert "trigger" in checkpoint_columns
    assert "search_text" in session_message_columns
    assert "deliverable_manifest_json" in run_columns
    assert {
        "display_name",
        "role",
        "format",
        "mime_type",
        "size_bytes",
        "source",
        "is_final",
        "preview_kind",
        "parent_artifact_id",
    }.issubset(artifact_columns)
    assert version_row is not None
    assert version_row["version"] == CURRENT_SESSION_DB_VERSION


def test_run_store_crud_and_related_records(tmp_path: Path):
    """RunStore should persist runs, steps, checkpoints, and artifacts."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session("session-1", str(tmp_path), messages=[])
    store = RunStore(db_path)

    run = RunRecord(
        id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        agent_template_snapshot=build_snapshot(),
        status="queued",
        goal="Design durable runs",
        trigger_message_ref=TriggerMessageRef(message_id="17"),
        created_at="2026-04-09T12:01:00+00:00",
        run_metadata={"kind": "root", "root_run_id": "run-1"},
    )
    child_run = RunRecord(
        id="run-1-child",
        session_id="session-1",
        agent_template_id="template-1",
        agent_template_snapshot=build_snapshot(),
        status="queued",
        goal="Inspect delegate flow",
        created_at="2026-04-09T12:01:30+00:00",
        parent_run_id="run-1",
        run_metadata={
            "kind": "delegate_child",
            "agent_name": "worker-1",
            "root_run_id": "run-1",
        },
    )
    step = RunStepRecord(
        id="step-1",
        run_id="run-1",
        sequence=0,
        step_type="tool_call",
        status="completed",
        title="Read task.md",
        input_summary="Load current task split",
        output_summary="Phase 2 requirements collected",
        started_at="2026-04-09T12:02:00+00:00",
        finished_at="2026-04-09T12:03:00+00:00",
    )
    checkpoint = RunCheckpointRecord(
        id="checkpoint-1",
        run_id="run-1",
        step_sequence=0,
        trigger="tool_completed",
        payload=RunCheckpointPayload(
            message_snapshot=[
                CheckpointMessageSnapshot(role="user", content="Design durable runs"),
                CheckpointMessageSnapshot(
                    role="assistant",
                    content="Reading task.md",
                    tool_call_names=["ReadTool"],
                ),
            ],
            current_step_index=1,
            active_step_id="step-1",
            incomplete_tool_calls=[
                PendingToolCallSnapshot(
                    tool_call_id="call-1",
                    tool_name="ReadTool",
                    arguments={"path": "task.md"},
                    issued_in_step_sequence=0,
                )
            ],
            sub_agent_states=[
                SubAgentCheckpointSummary(
                    run_id="child-run-1",
                    agent_name="researcher-1",
                    status="running",
                    current_step_index=0,
                    summary="Inspecting architecture draft",
                )
            ],
            shared_context_refs=[
                SharedContextRef(
                    uri=".clavi_agent/shared_context/session-1.json",
                    title="Session context",
                )
            ],
            metadata={"cursor": 1},
        ),
        created_at="2026-04-09T12:04:00+00:00",
    )
    artifact = ArtifactRecord(
        id="artifact-1",
        run_id="run-1",
        step_id="step-1",
        artifact_type="document",
        uri="ARCHITECTURE.md",
        display_name="ARCHITECTURE.md",
        role="final_deliverable",
        format="md",
        mime_type="text/markdown",
        size_bytes=2048,
        source="agent_generated",
        is_final=True,
        preview_kind="markdown",
        summary="Updated durable-run design",
        metadata={"format": "markdown"},
        created_at="2026-04-09T12:05:00+00:00",
    )

    store.create_run(run)
    store.create_run(child_run)
    store.create_step(step)
    running = run.transition_to("running", changed_at="2026-04-09T12:02:00+00:00")
    store.update_run(running)
    store.save_checkpoint(checkpoint)
    store.create_artifact(artifact)
    latest_run = store.get_run("run-1")
    assert latest_run is not None
    store.update_run(
        latest_run.model_copy(
            update={
                "deliverable_manifest": RunDeliverableManifest(
                    primary_artifact_id="artifact-1",
                    items=[
                        RunDeliverableRef(
                            artifact_id="artifact-1",
                            uri="ARCHITECTURE.md",
                            display_name="ARCHITECTURE.md",
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

    fetched = store.get_run("run-1")
    assert fetched is not None
    assert fetched.status == "running"
    assert fetched.trigger_message_ref is not None
    assert fetched.trigger_message_ref.message_id == "17"
    assert fetched.run_metadata == {"kind": "root", "root_run_id": "run-1"}
    assert fetched.deliverable_manifest.primary_artifact_id == "artifact-1"
    assert fetched.deliverable_manifest.items[0].display_name == "ARCHITECTURE.md"
    assert fetched.last_checkpoint_at == "2026-04-09T12:04:00+00:00"
    assert [item.id for item in store.list_steps("run-1")] == ["step-1"]
    assert store.list_checkpoints("run-1")[0].trigger == "tool_completed"
    assert store.list_checkpoints("run-1")[0].payload.metadata == {"cursor": 1}
    assert store.get_latest_checkpoint("run-1") is not None
    assert store.list_artifacts("run-1")[0].metadata == {"format": "markdown"}
    assert store.list_artifacts("run-1")[0].is_final is True
    assert store.list_artifacts("run-1")[0].preview_kind == "markdown"
    assert [item.id for item in store.list_runs(parent_run_id="run-1")] == ["run-1-child"]


def test_upload_store_round_trip_and_session_cascade(tmp_path: Path):
    """UploadStore should persist upload metadata and cascade on session deletion."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session("session-1", str(tmp_path), messages=[])
    run_store = RunStore(db_path)
    run_store.create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            goal="Track uploads",
            created_at="2026-04-13T07:01:00+00:00",
        )
    )
    upload_store = UploadStore(db_path)

    upload = UploadRecord(
        id="upload-1",
        session_id="session-1",
        run_id="run-1",
        original_name="Draft Report.md",
        safe_name="Draft Report.md",
        relative_path=".clavi_agent/uploads/session-1/upload-1/Draft Report.md",
        absolute_path=str(tmp_path / ".clavi_agent" / "uploads" / "session-1" / "upload-1" / "Draft Report.md"),
        mime_type="text/markdown",
        size_bytes=128,
        checksum="abc123",
        created_at="2026-04-13T07:02:00+00:00",
        created_by="user",
    )

    upload_store.create_upload(upload)

    fetched = upload_store.get_upload("upload-1")
    assert fetched is not None
    assert fetched.original_name == "Draft Report.md"
    assert fetched.run_id == "run-1"
    assert upload_store.list_uploads("session-1")[0].checksum == "abc123"

    assert session_store.delete_session("session-1") is True
    assert upload_store.get_upload("upload-1") is None


def test_run_store_updates_artifact_metadata(tmp_path: Path):
    """Artifact updates should persist revised lineage and final-deliverable metadata."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session("session-1", str(tmp_path), messages=[])
    store = RunStore(db_path)
    store.create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            goal="Promote report artifact",
            created_at="2026-04-13T08:00:00+00:00",
        )
    )

    artifact = ArtifactRecord(
        id="artifact-1",
        run_id="run-1",
        artifact_type="workspace_file",
        uri="docs/report.revised.md",
        display_name="report.revised.md",
        role="intermediate_file",
        format="md",
        mime_type="text/markdown",
        source="agent_generated",
        is_final=False,
        preview_kind="markdown",
        metadata={},
        created_at="2026-04-13T08:01:00+00:00",
    )
    store.create_artifact(artifact)

    updated = artifact.model_copy(
        update={
            "role": "revised_file",
            "source": "agent_revised",
            "is_final": True,
            "metadata": {
                "parent_upload_id": "upload-1",
                "revision_mode": "copy_on_write",
            },
        }
    )
    store.update_artifact(updated)

    fetched = store.list_artifacts("run-1")[0]
    assert fetched.role == "revised_file"
    assert fetched.source == "agent_revised"
    assert fetched.is_final is True
    assert fetched.metadata["parent_upload_id"] == "upload-1"


def test_run_store_migrates_checkpoint_trigger_and_filters_by_trigger(tmp_path: Path):
    """Checkpoint queries should expose persisted trigger semantics."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session("session-1", str(tmp_path), messages=[])
    store = RunStore(db_path)
    store.create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            goal="Resume one run",
            created_at="2026-04-09T12:01:00+00:00",
        )
    )

    store.save_checkpoint(
        RunCheckpointRecord(
            id="checkpoint-1",
            run_id="run-1",
            step_sequence=0,
            trigger="llm_response",
            payload=RunCheckpointPayload(metadata={"source": "llm"}),
            created_at="2026-04-09T12:02:00+00:00",
        )
    )
    store.save_checkpoint(
        RunCheckpointRecord(
            id="checkpoint-2",
            run_id="run-1",
            step_sequence=1,
            trigger="approval_wait",
            payload=RunCheckpointPayload(metadata={"source": "approval"}),
            created_at="2026-04-09T12:03:00+00:00",
        )
    )

    approval_checkpoints = store.list_checkpoints("run-1", trigger="approval_wait")
    latest = store.get_latest_checkpoint("run-1")

    assert [item.id for item in approval_checkpoints] == ["checkpoint-2"]
    assert latest is not None
    assert latest.id == "checkpoint-2"
    assert latest.trigger == "approval_wait"


def test_trace_and_approval_stores_round_trip(tmp_path: Path):
    """TraceStore and ApprovalStore should reuse the shared session DB schema."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session("session-1", str(tmp_path), messages=[])
    run_store = RunStore(db_path)
    run_store.create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            status="queued",
            goal="Trace one run",
            created_at="2026-04-09T12:01:00+00:00",
        )
    )

    trace_store = TraceStore(db_path)
    approval_store = ApprovalStore(db_path)

    trace_store.create_event(
        TraceEventRecord(
            id="trace-1",
            run_id="run-1",
            sequence=1,
            event_type="run_started",
            status="running",
            payload_summary="Run bootstrapped",
            created_at="2026-04-09T12:02:00+00:00",
        )
    )
    approval_store.create_request(
        ApprovalRequestRecord(
            id="approval-1",
            run_id="run-1",
            tool_name="BashTool",
            risk_level="high",
            status="pending",
            parameter_summary="git commit",
            impact_summary="Repository history mutation",
            requested_at="2026-04-09T12:03:00+00:00",
        )
    )

    events = trace_store.list_events("run-1")
    approvals = approval_store.list_requests(status="pending")

    assert len(events) == 1
    assert events[0].event_type == "run_started"
    assert len(approvals) == 1
    assert approvals[0].tool_name == "BashTool"


def test_run_related_stores_inherit_and_filter_account_id(tmp_path: Path):
    """Child run resources should inherit session ownership and support account filters."""
    db_path = tmp_path / "sessions.db"
    session_store = SessionStore(db_path)
    session_store.create_session(
        "session-1",
        str(tmp_path),
        messages=[],
        account_id="account-a",
    )
    session_store.create_session(
        "session-2",
        str(tmp_path),
        messages=[],
        account_id=ROOT_ACCOUNT_ID,
    )

    run_store = RunStore(db_path)
    upload_store = UploadStore(db_path)
    trace_store = TraceStore(db_path)
    approval_store = ApprovalStore(db_path)

    run = run_store.create_run(
        RunRecord(
            id="run-a",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            goal="Scoped run",
            created_at="2026-04-15T00:00:00+00:00",
        )
    )
    upload = upload_store.create_upload(
        UploadRecord(
            id="upload-a",
            session_id="session-1",
            run_id="run-a",
            original_name="report.md",
            safe_name="report.md",
            relative_path=".clavi_agent/uploads/session-1/upload-a/report.md",
            absolute_path=str(tmp_path / "report.md"),
            mime_type="text/markdown",
            size_bytes=10,
            checksum="sum",
            created_at="2026-04-15T00:01:00+00:00",
        )
    )
    event = trace_store.create_event(
        TraceEventRecord(
            id="trace-a",
            run_id="run-a",
            sequence=0,
            event_type="run_started",
            created_at="2026-04-15T00:02:00+00:00",
        )
    )
    request = approval_store.create_request(
        ApprovalRequestRecord(
            id="approval-a",
            run_id="run-a",
            tool_name="BashTool",
            risk_level="high",
            requested_at="2026-04-15T00:03:00+00:00",
        )
    )

    assert run.account_id == "account-a"
    assert upload.account_id == "account-a"
    assert event.account_id == "account-a"
    assert request.account_id == "account-a"
    assert run_store.get_run("run-a", account_id="account-a") is not None
    assert run_store.get_run("run-a", account_id=ROOT_ACCOUNT_ID) is None
    assert upload_store.get_upload("upload-a", account_id="account-a") is not None
    assert upload_store.get_upload("upload-a", account_id=ROOT_ACCOUNT_ID) is None
    assert len(trace_store.list_events("run-a", account_id="account-a")) == 1
    assert trace_store.list_events("run-a", account_id=ROOT_ACCOUNT_ID) == []
    assert len(approval_store.list_requests(account_id="account-a")) == 1
    assert approval_store.list_requests(account_id=ROOT_ACCOUNT_ID) == []

