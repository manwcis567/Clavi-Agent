"""Tests for learned workflow candidate persistence."""

from pathlib import Path

from clavi_agent.agent_template_models import AgentTemplateSnapshot
from clavi_agent.learned_workflow_store import LearnedWorkflowStore
from clavi_agent.run_models import RunRecord
from clavi_agent.run_store import RunStore
from clavi_agent.session_store import SessionStore


def build_snapshot() -> AgentTemplateSnapshot:
    return AgentTemplateSnapshot(
        template_id="template-1",
        template_version=1,
        captured_at="2026-04-16T12:00:00+00:00",
        name="Writer",
        system_prompt="You are a writer.",
    )


def test_learned_workflow_store_upserts_and_filters_candidates(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    SessionStore(db_path).create_session("session-1", str(tmp_path), messages=[])
    RunStore(db_path).create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            status="completed",
            goal="Write the report",
            created_at="2026-04-16T12:00:00+00:00",
        )
    )
    store = LearnedWorkflowStore(db_path)

    candidate = store.upsert_candidate_for_run(
        account_id="root",
        run_id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        title="Write the report",
        summary="Create a report deliverable.",
        description="Use when the task is to produce a report file.",
        signal_types=["successful_complex_run"],
        source_run_ids=["run-1"],
        tool_names=["write_file"],
        step_titles=["write_file", "Run completed"],
        artifact_ids=["artifact-1"],
        suggested_skill_name="Write the report",
        generated_skill_markdown="---\nname: write-the-report\ndescription: test\n---\n",
    )

    assert candidate.suggested_skill_name == "write-the-report"
    assert store.get_candidate_by_run("run-1") is not None
    assert len(store.list_candidate_records(status="pending_review")) == 1

    updated = store.upsert_candidate_for_run(
        account_id="root",
        run_id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        title="Write the report",
        summary="Updated summary.",
        signal_types=["successful_complex_run", "repeated_task_pattern"],
        generated_skill_markdown="---\nname: write-the-report\ndescription: updated\n---\n",
    )

    assert updated.id == candidate.id
    assert updated.summary == "Updated summary."
    assert updated.signal_types == [
        "successful_complex_run",
        "repeated_task_pattern",
    ]


def test_learned_workflow_store_tracks_review_and_installation(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    SessionStore(db_path).create_session("session-1", str(tmp_path), messages=[])
    RunStore(db_path).create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            status="completed",
            goal="Export PDF",
            created_at="2026-04-16T12:10:00+00:00",
        )
    )
    store = LearnedWorkflowStore(db_path)
    candidate = store.upsert_candidate_for_run(
        account_id="root",
        run_id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        title="Export PDF",
        summary="Export a final PDF.",
        signal_types=["successful_complex_run"],
        generated_skill_markdown="---\nname: export-pdf\ndescription: test\n---\n",
    )

    approved = store.update_candidate_status(
        candidate.id,
        status="approved",
        review_notes="looks reusable",
    )
    installed = store.update_candidate_status(
        candidate.id,
        status="installed",
        installed_agent_id="agent-1",
        installed_skill_path=str(tmp_path / "skills" / "export-pdf" / "SKILL.md"),
    )

    assert approved.status == "approved"
    assert approved.review_notes == "looks reusable"
    assert approved.approved_at is not None
    assert installed.status == "installed"
    assert installed.installed_agent_id == "agent-1"
    assert installed.installed_skill_path.endswith("SKILL.md")
    assert installed.installed_at is not None

