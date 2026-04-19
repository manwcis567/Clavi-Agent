"""Tests for durable run domain models."""

import pytest

from clavi_agent.agent_template_models import AgentTemplateSnapshot
from clavi_agent.run_models import (
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
    TriggerMessageRef,
)


def build_snapshot() -> AgentTemplateSnapshot:
    """Build a minimal template snapshot for run model tests."""
    return AgentTemplateSnapshot(
        template_id="template-1",
        template_version=3,
        captured_at="2026-04-09T12:00:00+00:00",
        name="Researcher",
        system_prompt="You are a researcher.",
    )


def test_run_record_tracks_template_snapshot_and_trigger_message():
    """Run records should capture immutable template state and user trigger context."""
    run = RunRecord(
        id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        agent_template_snapshot=build_snapshot(),
        goal="Summarize the current architecture work",
        trigger_message_ref=TriggerMessageRef(message_id="message-17"),
        created_at="2026-04-09T12:01:00+00:00",
    )

    assert run.status == "queued"
    assert run.is_terminal is False
    assert run.trigger_message_ref is not None
    assert run.trigger_message_ref.message_id == "message-17"
    assert run.agent_template_snapshot.template_version == 3


def test_run_record_transition_enforces_status_machine_and_timestamps():
    """Run records should enforce the documented run lifecycle transitions."""
    run = RunRecord(
        id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        agent_template_snapshot=build_snapshot(),
        goal="Handle one user task",
        created_at="2026-04-09T12:01:00+00:00",
    )

    started = run.transition_to(
        "running",
        changed_at="2026-04-09T12:02:00+00:00",
        current_step_index=1,
    )
    paused = started.transition_to("waiting_approval")
    resumed = paused.transition_to(
        "running",
        last_checkpoint_at="2026-04-09T12:03:00+00:00",
    )
    completed = resumed.transition_to("completed", changed_at="2026-04-09T12:04:00+00:00")

    assert started.started_at == "2026-04-09T12:02:00+00:00"
    assert paused.status == "waiting_approval"
    assert resumed.last_checkpoint_at == "2026-04-09T12:03:00+00:00"
    assert completed.finished_at == "2026-04-09T12:04:00+00:00"
    assert completed.is_terminal is True

    with pytest.raises(ValueError, match="Invalid run status transition"):
        run.transition_to("completed")


def test_run_record_supports_timed_out_terminal_status():
    """Timed out runs should be terminal and capture finished_at."""
    run = RunRecord(
        id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        agent_template_snapshot=build_snapshot(),
        goal="Handle one long-running user task",
        created_at="2026-04-09T12:01:00+00:00",
    )

    running = run.transition_to("running", changed_at="2026-04-09T12:02:00+00:00")
    timed_out = running.transition_to(
        "timed_out",
        changed_at="2026-04-09T12:03:00+00:00",
        error_summary="Agent run timed out after 30 seconds.",
    )

    assert timed_out.status == "timed_out"
    assert timed_out.finished_at == "2026-04-09T12:03:00+00:00"
    assert timed_out.is_terminal is True


def test_run_step_record_tracks_step_types_and_transitions():
    """Run steps should model the per-step lifecycle independently from runs."""
    step = RunStepRecord(
        id="step-1",
        run_id="run-1",
        sequence=0,
        step_type="tool_call",
        title="Read architecture document",
        input_summary="Open ARCHITECTURE.md",
    )

    running = step.transition_to("running", changed_at="2026-04-09T12:02:00+00:00")
    completed = running.transition_to(
        "completed",
        changed_at="2026-04-09T12:03:00+00:00",
        output_summary="Collected durable run requirements",
    )

    assert running.started_at == "2026-04-09T12:02:00+00:00"
    assert completed.finished_at == "2026-04-09T12:03:00+00:00"
    assert completed.output_summary == "Collected durable run requirements"
    assert completed.is_terminal is True

    with pytest.raises(ValueError, match="Invalid run step status transition"):
        step.transition_to("completed")


def test_checkpoint_record_captures_resume_payload_and_trigger():
    """Checkpoint records should preserve structured recovery state."""
    checkpoint = RunCheckpointRecord(
        id="checkpoint-1",
        run_id="run-1",
        step_sequence=3,
        trigger="tool_completed",
        payload=RunCheckpointPayload(
            message_snapshot=[
                CheckpointMessageSnapshot(role="user", content="Summarize task.md"),
                CheckpointMessageSnapshot(
                    role="assistant",
                    content="I will inspect the run state.",
                    tool_call_names=["ReadTool"],
                ),
            ],
            current_step_index=3,
            active_step_id="step-3",
            incomplete_tool_calls=[
                PendingToolCallSnapshot(
                    tool_call_id="call-1",
                    tool_name="ReadTool",
                    arguments={"path": "task.md"},
                    issued_in_step_sequence=3,
                )
            ],
            sub_agent_states=[
                SubAgentCheckpointSummary(
                    run_id="child-run-1",
                    agent_name="researcher-1",
                    status="running",
                    current_step_index=1,
                    summary="Collecting architecture notes",
                )
            ],
            shared_context_refs=[
                SharedContextRef(
                    uri=".clavi_agent/shared_context/session-1.json",
                    title="Session shared context",
                    metadata={"entries": 2},
                )
            ],
            metadata={"checkpoint_version": 1},
        ),
        created_at="2026-04-09T12:05:00+00:00",
    )

    assert checkpoint.trigger == "tool_completed"
    assert checkpoint.payload.current_step_index == 3
    assert checkpoint.payload.incomplete_tool_calls[0].tool_name == "ReadTool"
    assert checkpoint.payload.sub_agent_states[0].run_id == "child-run-1"
    assert checkpoint.payload.shared_context_refs[0].uri.endswith("session-1.json")


def test_checkpoint_payload_wraps_legacy_dict_into_metadata():
    """Legacy checkpoint payloads should still deserialize without data loss."""
    checkpoint = RunCheckpointRecord.model_validate(
        {
            "id": "checkpoint-legacy",
            "run_id": "run-1",
            "step_sequence": 1,
            "payload": {"cursor": 1, "resume_from": "tool_call"},
            "created_at": "2026-04-09T12:05:00+00:00",
        }
    )

    assert checkpoint.trigger == "llm_response"
    assert checkpoint.payload.metadata == {"cursor": 1, "resume_from": "tool_call"}


def test_deliverable_metadata_round_trips_through_run_and_artifact_models():
    artifact = ArtifactRecord(
        id="artifact-1",
        run_id="run-1",
        step_id="step-2",
        artifact_type="workspace_file",
        uri=".clavi_agent/uploads/session-1/upload-1/draft.revised.md",
        display_name="draft.revised.md",
        role="revised_file",
        format="md",
        mime_type="text/markdown",
        size_bytes=256,
        source="agent_revised",
        is_final=True,
        preview_kind="markdown",
        parent_artifact_id="artifact-upload-1",
        summary="Revised upload deliverable",
        metadata={
            "parent_upload_id": "upload-1",
            "parent_upload_name": "draft.md",
            "revision_mode": "copy_on_write",
        },
        created_at="2026-04-13T09:00:00+00:00",
    )
    restored_artifact = ArtifactRecord.model_validate(artifact.model_dump(mode="json"))

    run = RunRecord(
        id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        agent_template_snapshot=build_snapshot(),
        goal="Revise uploaded draft",
        created_at="2026-04-13T09:01:00+00:00",
        deliverable_manifest=RunDeliverableManifest(
            primary_artifact_id=artifact.id,
            items=[
                RunDeliverableRef(
                    artifact_id=artifact.id,
                    uri=artifact.uri,
                    display_name=artifact.display_name,
                    format=artifact.format,
                    mime_type=artifact.mime_type,
                    role=artifact.role,
                    is_primary=True,
                )
            ],
        ),
    )
    restored_run = RunRecord.model_validate(run.model_dump(mode="json"))

    assert restored_artifact.parent_artifact_id == "artifact-upload-1"
    assert restored_artifact.source == "agent_revised"
    assert restored_artifact.metadata["parent_upload_id"] == "upload-1"
    assert restored_artifact.metadata["revision_mode"] == "copy_on_write"
    assert restored_run.deliverable_manifest.primary_artifact_id == artifact.id
    assert restored_run.deliverable_manifest.items[0].role == "revised_file"
    assert restored_run.deliverable_manifest.items[0].mime_type == "text/markdown"

