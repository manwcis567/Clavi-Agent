"""Tests for skill improvement proposal persistence."""

from pathlib import Path

from clavi_agent.agent_template_models import AgentTemplateSnapshot
from clavi_agent.run_models import RunRecord
from clavi_agent.run_store import RunStore
from clavi_agent.session_store import SessionStore
from clavi_agent.skill_improvement_store import SkillImprovementStore


def build_snapshot() -> AgentTemplateSnapshot:
    return AgentTemplateSnapshot(
        template_id="template-1",
        template_version=1,
        captured_at="2026-04-16T12:00:00+00:00",
        name="Writer",
        system_prompt="You are a writer.",
    )


def test_skill_improvement_store_upserts_and_tracks_apply_status(tmp_path: Path):
    db_path = tmp_path / "sessions.db"
    SessionStore(db_path).create_session("session-1", str(tmp_path), messages=[])
    RunStore(db_path).create_run(
        RunRecord(
            id="run-1",
            session_id="session-1",
            agent_template_id="template-1",
            agent_template_snapshot=build_snapshot(),
            status="completed",
            goal="Improve export skill",
            created_at="2026-04-16T12:00:00+00:00",
        )
    )
    store = SkillImprovementStore(db_path)

    proposal = store.upsert_proposal_for_skill(
        account_id="root",
        run_id="run-1",
        session_id="session-1",
        agent_template_id="template-1",
        skill_name="export-skill",
        target_skill_path=str(tmp_path / "skills" / "export-skill" / "SKILL.md"),
        title="Improve skill: export-skill",
        summary="Add retry guidance.",
        signal_types=["manual_successful_refinement"],
        source_run_ids=["run-1"],
        base_version=1,
        proposed_version=2,
        current_skill_markdown="---\nname: export-skill\ndescription: Export.\nversion: 1\n---\n",
        proposed_skill_markdown="---\nname: export-skill\ndescription: Export.\nversion: 2\n---\n",
        changelog_entry="v2 (2026-04-16): Add retry guidance.",
    )

    assert proposal.status == "pending_review"
    assert proposal.proposed_version == 2

    approved = store.update_proposal_status(
        proposal.id,
        status="approved",
        review_notes="可以应用",
    )
    applied = store.update_proposal_status(
        proposal.id,
        status="applied",
        applied_skill_path=str(tmp_path / "skills" / "export-skill" / "SKILL.md"),
    )

    assert approved.approved_at is not None
    assert approved.review_notes == "可以应用"
    assert applied.status == "applied"
    assert applied.applied_skill_path is not None
    assert applied.applied_at is not None

