import shutil
from pathlib import Path

import pytest

from clavi_agent.account_constants import ROOT_ACCOUNT_ID
from clavi_agent.agent_store import AgentStore
from clavi_agent.sqlite_schema import AGENT_DB_SCOPE, CURRENT_AGENT_DB_VERSION


def create_test_skill(skill_dir: Path, name: str, description: str, content: str = "content"):
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: {description}
---

{content}
""",
        encoding="utf-8",
    )
    (skill_dir / "notes.txt").write_text("extra asset", encoding="utf-8")


@pytest.fixture
def temp_store(tmp_path):
    db_path = tmp_path / "agents.db"
    return AgentStore(db_path)


def test_agent_store_crud(temp_store: AgentStore):
    reserved_agent_id = temp_store.generate_agent_id()
    agent = temp_store.create_agent(
        name="Test Agent",
        system_prompt="You are a test agent.",
        description="Just for testing",
        agent_id=reserved_agent_id,
        tools=["BashTool"],
        workspace_type="shared",
    )
    assert agent["name"] == "Test Agent"
    assert agent["id"] == reserved_agent_id
    assert agent["tools"] == ["BashTool"]
    assert agent["skills"] == []
    assert agent["is_system"] is False
    assert agent["version"] == 1
    assert agent["workspace_policy"]["mode"] == "shared"
    assert agent["approval_policy"]["mode"] == "default"
    assert agent["delegation_policy"]["mode"] == "prefer_delegate"
    assert temp_store.get_agent_dir(reserved_agent_id) == (
        temp_store.agent_data_dir / "accounts" / ROOT_ACCOUNT_ID / reserved_agent_id
    )
    assert temp_store.get_agent_workspace_dir(reserved_agent_id).exists()
    assert temp_store.get_agent_skills_dir(reserved_agent_id).parent.parent == temp_store.get_agent_workspace_dir(reserved_agent_id)

    agent_id = agent["id"]
    create_test_skill(
        temp_store.get_agent_skills_dir(agent_id) / "frontend-design",
        "frontend-design",
        "Frontend workflow",
    )
    refreshed = temp_store.refresh_agent_skills_from_directory(agent_id)
    assert refreshed["skills"] == [
        {"name": "frontend-design", "description": "Frontend workflow"}
    ]

    fetched = temp_store.get_agent(agent_id)
    assert fetched is not None
    assert fetched["name"] == "Test Agent"

    updated = temp_store.update_agent(
        agent_id,
        name="Updated Agent",
    )
    assert updated["name"] == "Updated Agent"
    assert updated["system_prompt"] == "You are a test agent."
    assert updated["version"] == 3
    create_test_skill(
        temp_store.get_agent_skills_dir(agent_id) / "openai-docs",
        "openai-docs",
        "Official OpenAI docs helper",
    )
    refreshed = temp_store.refresh_agent_skills_from_directory(agent_id)
    assert refreshed["skills"] == [
        {"name": "frontend-design", "description": "Frontend workflow"},
        {"name": "openai-docs", "description": "Official OpenAI docs helper"},
    ]
    assert (temp_store.get_agent_skills_dir(agent_id) / "openai-docs" / "notes.txt").exists()

    ok = temp_store.delete_agent(agent_id)
    assert ok is True
    assert temp_store.get_agent(agent_id) is None
    assert not temp_store.get_agent_dir(agent_id).exists()


def test_sync_system_agents(temp_store: AgentStore):
    system_config = {
        "id": "sys-1",
        "name": "System Agent",
        "system_prompt": "I am system",
        "tools": ["ReadTool"],
    }

    temp_store.sync_system_agents([system_config])

    fetched = temp_store.get_agent("sys-1")
    assert fetched is not None
    assert fetched["is_system"] is True
    assert fetched["skills"] == []
    assert fetched["version"] == 1
    assert temp_store.get_agent_dir("sys-1") == (
        temp_store.agent_data_dir / "system" / "sys-1"
    )
    assert temp_store.get_agent_workspace_dir("sys-1").exists()
    create_test_skill(
        temp_store.get_agent_skills_dir("sys-1") / "frontend-design",
        "frontend-design",
        "Frontend workflow",
    )
    refreshed = temp_store.refresh_agent_skills_from_directory("sys-1")
    assert refreshed["skills"] == [
        {"name": "frontend-design", "description": "Frontend workflow"}
    ]

    with pytest.raises(ValueError):
        temp_store.update_agent("sys-1", name="Hacked System")

    with pytest.raises(ValueError):
        temp_store.delete_agent("sys-1")


def test_delete_agent_skill_updates_metadata_and_files(temp_store: AgentStore):
    agent = temp_store.create_agent(
        name="Skill Manager",
        system_prompt="Manage skills.",
        description="Testing skill deletion",
        tools=["ReadTool"],
    )

    agent_id = agent["id"]
    create_test_skill(
        temp_store.get_agent_skills_dir(agent_id) / "frontend-design",
        "frontend-design",
        "Frontend workflow",
    )
    create_test_skill(
        temp_store.get_agent_skills_dir(agent_id) / "openai-docs",
        "openai-docs",
        "Official docs helper",
    )
    temp_store.refresh_agent_skills_from_directory(agent_id)

    updated = temp_store.delete_agent_skill(agent_id, "frontend-design")

    assert updated is not None
    assert updated["skills"] == [
        {"name": "openai-docs", "description": "Official docs helper"}
    ]
    assert not (temp_store.get_agent_skills_dir(agent_id) / "frontend-design").exists()
    assert (temp_store.get_agent_skills_dir(agent_id) / "openai-docs").exists()


def test_legacy_skill_directory_is_migrated_into_workspace(temp_store: AgentStore):
    agent = temp_store.create_agent(
        name="Migrated Agent",
        system_prompt="Handle migration.",
        tools=[],
    )

    agent_id = agent["id"]
    legacy_skills_dir = temp_store.get_agent_dir(agent_id) / "skills"
    create_test_skill(
        legacy_skills_dir / "legacy-skill",
        "legacy-skill",
        "Legacy location",
    )

    migrated_skills_dir = temp_store.get_agent_skills_dir(agent_id)

    assert migrated_skills_dir.parent.parent == temp_store.get_agent_workspace_dir(agent_id)
    assert not legacy_skills_dir.exists()
    assert (migrated_skills_dir / "legacy-skill" / "SKILL.md").exists()


def test_agent_store_migrates_legacy_flat_agent_directory_into_account_scope(
    temp_store: AgentStore,
):
    agent = temp_store.create_agent(
        name="Account Scoped Agent",
        system_prompt="Handle account scope.",
        account_id="account-a",
    )

    scoped_dir = temp_store.get_agent_dir(agent["id"], account_id="account-a")
    legacy_dir = temp_store.agent_data_dir / agent["id"]
    shutil.move(str(scoped_dir), str(legacy_dir))
    marker = legacy_dir / "workspace" / "marker.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("legacy", encoding="utf-8")

    migrated_dir = temp_store.get_agent_dir(agent["id"])

    assert migrated_dir == scoped_dir
    assert migrated_dir.exists()
    assert not legacy_dir.exists()
    assert (migrated_dir / "workspace" / "marker.txt").read_text(encoding="utf-8") == "legacy"


def test_agent_template_fields_and_snapshot(temp_store: AgentStore):
    template = temp_store.create_agent_template(
        name="Template Agent",
        system_prompt="You are a templated agent.",
        description="Structured template config",
        tools=["ReadTool"],
        workspace_policy={
            "mode": "isolated",
            "allow_session_override": False,
            "readable_roots": ["docs", "logs"],
            "writable_roots": ["docs"],
            "read_only_tools": ["bash"],
            "disabled_tools": ["delegate_task"],
            "allowed_shell_command_prefixes": ["git status"],
            "allowed_network_domains": ["example.com"],
        },
        approval_policy={
            "mode": "strict",
            "require_approval_tools": ["BashTool"],
            "require_approval_risk_levels": ["high"],
            "require_approval_risk_categories": ["external_network"],
            "notes": "Review shell access",
        },
        delegation_policy={
            "mode": "prefer_delegate",
            "require_delegate_for_write_actions": True,
            "prefer_batch_delegate": True,
        },
    )

    assert template["workspace_policy"] == {
        "mode": "isolated",
        "allow_session_override": False,
        "readable_roots": ["docs", "logs"],
        "writable_roots": ["docs"],
        "read_only_tools": ["bash"],
        "disabled_tools": ["delegate_task"],
        "allowed_shell_command_prefixes": ["git status"],
        "allowed_network_domains": ["example.com"],
    }
    assert template["approval_policy"] == {
        "mode": "strict",
        "require_approval_tools": ["BashTool"],
        "auto_approve_tools": [],
        "require_approval_risk_levels": ["high"],
        "require_approval_risk_categories": ["external_network"],
        "notes": "Review shell access",
    }
    assert template["run_policy"] == {
        "timeout_seconds": None,
        "max_concurrent_runs": 1,
    }
    assert template["delegation_policy"] == {
        "mode": "prefer_delegate",
        "require_delegate_for_write_actions": True,
        "require_delegate_for_shell": False,
        "require_delegate_for_stateful_mcp": False,
        "allow_main_agent_read_tools": True,
        "verify_worker_output": True,
        "prefer_batch_delegate": True,
    }
    assert template["workspace_type"] == "isolated"
    assert template["version"] == 1

    updated = temp_store.update_agent_template(
        template["id"],
        description="Updated template config",
        workspace_type="shared",
        run_policy={
            "timeout_seconds": 90,
            "max_concurrent_runs": 2,
        },
        delegation_policy={
            "mode": "supervisor_only",
            "allow_main_agent_read_tools": False,
            "require_delegate_for_shell": True,
        },
    )
    assert updated is not None
    assert updated["version"] == 2
    assert updated["workspace_policy"]["mode"] == "shared"
    assert updated["run_policy"] == {
        "timeout_seconds": 90,
        "max_concurrent_runs": 2,
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

    snapshot = temp_store.snapshot_agent_template(template["id"])
    assert snapshot is not None
    assert snapshot.template_id == template["id"]
    assert snapshot.template_version == 2
    assert snapshot.description == "Updated template config"
    assert snapshot.workspace_type == "shared"
    assert snapshot.workspace_policy.readable_roots == ["docs", "logs"]
    assert snapshot.workspace_policy.allowed_network_domains == ["example.com"]
    assert snapshot.run_policy.timeout_seconds == 90
    assert snapshot.run_policy.max_concurrent_runs == 2
    assert snapshot.approval_policy.require_approval_risk_levels == ["high"]
    assert snapshot.delegation_policy.mode == "supervisor_only"
    assert snapshot.delegation_policy.allow_main_agent_read_tools is False


def test_agent_store_migrates_legacy_agents_table(tmp_path: Path):
    """Legacy agent DBs should migrate onto the agent_templates schema."""
    db_path = tmp_path / "agents.db"
    store = AgentStore(db_path)
    store.create_agent(
        name="Legacy Agent",
        system_prompt="You are legacy.",
        description="Legacy schema test",
    )

    with store._connect() as conn:
        conn.execute("ALTER TABLE agent_templates RENAME TO agents")

    migrated = AgentStore(db_path)
    records = migrated.list_agent_template_records()

    assert len(records) == 1
    assert records[0].name == "Legacy Agent"

    with migrated._connect() as conn:
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
        version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (AGENT_DB_SCOPE,),
        ).fetchone()

    assert "agents" not in tables
    assert "agent_templates" in tables
    assert {
        "idx_agent_templates_updated_at",
        "idx_agent_templates_is_system",
    }.issubset(indexes)
    assert version_row is not None
    assert version_row["version"] == CURRENT_AGENT_DB_VERSION


def test_agent_store_filters_custom_templates_by_account_and_keeps_system_visible(
    temp_store: AgentStore,
):
    own = temp_store.create_agent(name="Own Agent", system_prompt="Own", account_id="account-a")
    other = temp_store.create_agent(
        name="Other Agent",
        system_prompt="Other",
        account_id="account-b",
    )
    temp_store.sync_system_agents(
        [
            {
                "id": "sys-visible",
                "name": "Visible System",
                "system_prompt": "System",
            }
        ]
    )

    visible = temp_store.list_agent_templates(account_id="account-a")
    visible_ids = {item["id"] for item in visible}

    assert own["id"] in visible_ids
    assert "sys-visible" in visible_ids
    assert other["id"] not in visible_ids
    assert temp_store.get_agent(other["id"], account_id="account-a") is None
    system_agent = temp_store.get_agent("sys-visible", account_id="account-a")
    assert system_agent is not None
    assert system_agent["account_id"] is None

