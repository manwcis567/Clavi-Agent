import sqlite3
from pathlib import Path

from clavi_agent.account_store import AccountStore
from clavi_agent.agent_store import AgentStore
from clavi_agent.sqlite_schema import AGENT_DB_SCOPE, CURRENT_AGENT_DB_VERSION
from clavi_agent.user_memory_store import UserMemoryStore


def create_account(db_path: Path, username: str) -> dict:
    account_store = AccountStore(db_path, auto_seed_root=False)
    return account_store.create_account(
        username=username,
        password="Secret123!",
        display_name=username.title(),
    )


def test_user_memory_store_creates_profile_tables_and_merges_profile(tmp_path: Path):
    db_path = tmp_path / "agents.db"
    account = create_account(db_path, "alice")
    store = UserMemoryStore(db_path)

    created = store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "zh-CN", "technical_depth": "high"},
        summary="偏好中文，技术细节可以更深入。",
        writer_type="tool",
        writer_id="record_note",
    )
    updated = store.upsert_user_profile(
        account["id"],
        profile={"timezone": "Asia/Shanghai"},
        merge=True,
        source_session_id="session-profile",
        source_run_id="run-profile",
        writer_type="tool",
        writer_id="record_note",
    )
    audit_events = store.list_audit_events(account["id"], target_scope="user_profile")

    assert created["profile"] == {
        "preferred_language": "zh-CN",
        "technical_depth": "high",
    }
    assert updated["profile"] == {
        "preferred_language": "zh-CN",
        "technical_depth": "high",
        "timezone": "Asia/Shanghai",
    }
    assert updated["summary"] == "偏好中文，技术细节可以更深入。"
    assert updated["writer_type"] == "tool"
    assert updated["writer_id"] == "record_note"
    assert audit_events[0]["action"] == "profile_upsert"
    assert audit_events[0]["session_id"] == "session-profile"
    assert audit_events[0]["run_id"] == "run-profile"
    assert audit_events[1]["action"] == "profile_create"

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        version_row = conn.execute(
            "SELECT version FROM schema_migrations WHERE scope = ?",
            (AGENT_DB_SCOPE,),
        ).fetchone()

    assert {"user_profiles", "user_memory_entries"}.issubset(tables)
    assert version_row is not None
    assert version_row["version"] == CURRENT_AGENT_DB_VERSION


def test_user_memory_store_supports_crud_search_and_supersede(tmp_path: Path):
    db_path = tmp_path / "agents.db"
    account = create_account(db_path, "bob")
    store = UserMemoryStore(db_path)

    preference = store.create_memory_entry(
        user_id=account["id"],
        memory_type="preference",
        content="用户希望回答简洁，优先给结论。",
        summary="回答风格偏简洁。",
        source_session_id="session-1",
        source_run_id="run-1",
        writer_type="tool",
        writer_id="record_note",
        confidence=0.8,
    )
    correction = store.create_memory_entry(
        user_id=account["id"],
        memory_type="correction",
        content="用户明确要求文档和代码读写都使用 UTF-8。",
        summary="UTF-8 是硬性约束。",
        source_session_id="session-2",
        writer_type="tool",
        writer_id="record_note",
        confidence=0.95,
    )

    updated = store.update_memory_entry(
        preference["id"],
        summary="回答风格偏简洁，先结论后细节。",
        writer_type="tool",
        writer_id="record_note",
        confidence=0.9,
    )
    replacement = store.create_memory_entry(
        user_id=account["id"],
        memory_type="correction",
        content="用户补充要求：所有新增注释必须使用中文。",
        summary="新增注释统一中文。",
        source_session_id="session-3",
        writer_type="tool",
        writer_id="record_note",
        confidence=0.92,
    )
    superseded = store.supersede_memory_entry(
        correction["id"],
        superseded_by=replacement["id"],
        source_session_id="session-3",
        source_run_id="run-3",
        writer_type="tool",
        writer_id="record_note",
    )
    search_results = store.search_memory_entries(
        account["id"],
        query="简洁 结论",
        memory_types=["preference"],
    )
    active_entries = store.list_memory_entries(account["id"])
    all_entries = store.list_memory_entries(account["id"], include_superseded=True)
    audit_events = store.list_audit_events(account["id"], target_scope="user_memory", limit=10)

    assert updated is not None
    assert updated["summary"] == "回答风格偏简洁，先结论后细节。"
    assert updated["confidence"] == 0.9
    assert updated["writer_type"] == "tool"
    assert updated["writer_id"] == "record_note"
    assert superseded is not None
    assert superseded["superseded_by"] == replacement["id"]
    assert superseded["writer_type"] == "tool"
    assert superseded["writer_id"] == "record_note"
    assert len(search_results) == 1
    assert search_results[0]["id"] == preference["id"]
    assert {item["id"] for item in active_entries} == {preference["id"], replacement["id"]}
    assert {item["id"] for item in all_entries} == {
        preference["id"],
        correction["id"],
        replacement["id"],
    }
    assert [event["action"] for event in audit_events[:5]] == [
        "memory_supersede",
        "memory_create",
        "memory_update",
        "memory_create",
        "memory_create",
    ]
    assert audit_events[0]["payload"]["superseded_by"] == replacement["id"]


def test_user_memory_store_compacts_duplicates_and_is_shared_across_templates(
    tmp_path: Path,
):
    db_path = tmp_path / "agents.db"
    account = create_account(db_path, "carol")
    agent_store = AgentStore(db_path)
    template_a = agent_store.create_agent(
        name="Planner",
        system_prompt="Plan carefully.",
        account_id=account["id"],
    )
    template_b = agent_store.create_agent(
        name="Coder",
        system_prompt="Write code.",
        account_id=account["id"],
    )
    store = UserMemoryStore(db_path)

    first = store.create_memory_entry(
        user_id=account["id"],
        memory_type="workflow_fact",
        content="提交代码前需要先跑针对性测试。",
        summary="测试后再提交。",
        confidence=0.6,
    )
    duplicate = store.create_memory_entry(
        user_id=account["id"],
        memory_type="workflow_fact",
        content="提交代码前需要先跑针对性测试。",
        summary="提交前必须验证。",
        source_run_id="run-2",
        confidence=0.85,
    )

    result = store.compact_memory_entries(
        account["id"],
        memory_type="workflow_fact",
        writer_type="system",
        writer_id="memory_compactor",
    )
    reopened = UserMemoryStore(db_path)
    active_entries = reopened.list_memory_entries(
        account["id"],
        memory_types=["workflow_fact"],
    )
    all_entries = reopened.list_memory_entries(
        account["id"],
        memory_types=["workflow_fact"],
        include_superseded=True,
    )
    audit_events = reopened.list_audit_events(
        account["id"],
        target_scope="user_memory",
        target_id=duplicate["id"],
    )

    assert template_a["account_id"] == account["id"]
    assert template_b["account_id"] == account["id"]
    assert result.merged_group_count == 1
    assert len(result.canonical_entry_ids) == 1
    assert result.superseded_entry_ids == [first["id"]]
    assert len(active_entries) == 1
    assert active_entries[0]["id"] == duplicate["id"]
    assert active_entries[0]["summary"] == "提交前必须验证。"
    assert active_entries[0]["confidence"] == 0.85
    assert active_entries[0]["writer_type"] == "system"
    assert active_entries[0]["writer_id"] == "memory_compactor"
    assert {item["id"] for item in all_entries} == {first["id"], duplicate["id"]}
    superseded_first = next(item for item in all_entries if item["id"] == first["id"])
    assert superseded_first["superseded_by"] == duplicate["id"]
    assert audit_events[0]["action"] == "memory_compact_merge"
    assert audit_events[0]["writer_id"] == "memory_compactor"


def test_user_memory_store_enforces_capacity_and_tracks_audit(tmp_path: Path):
    db_path = tmp_path / "agents.db"
    account = create_account(db_path, "eve")
    store = UserMemoryStore(db_path)

    created_ids = []
    for index in range(4):
        created = store.create_memory_entry(
            user_id=account["id"],
            memory_type="constraint",
            content=f"约束 {index}: 统一遵循规则 {index}。",
            summary=f"规则 {index}",
            confidence=0.5 + (index / 10),
            writer_type="tool",
            writer_id="record_note",
        )
        created_ids.append(created["id"])

    compacted_ids = store.enforce_memory_capacity(
        account["id"],
        memory_type="constraint",
        max_active=3,
        preferred_entry_id=created_ids[-1],
        source_session_id="session-capacity",
        source_run_id="run-capacity",
        writer_type="system",
        writer_id="memory_guardrail",
    )
    active_entries = store.list_memory_entries(
        account["id"],
        memory_types=["constraint"],
    )
    all_entries = store.list_memory_entries(
        account["id"],
        memory_types=["constraint"],
        include_superseded=True,
    )
    audit_events = store.list_audit_events(
        account["id"],
        target_scope="user_memory",
        limit=20,
    )

    assert len(compacted_ids) == 1
    assert len(active_entries) == 3
    assert len(all_entries) == 4
    assert any(item["id"] == created_ids[-1] for item in active_entries)
    superseded_entry = next(item for item in all_entries if item["id"] == compacted_ids[0])
    assert superseded_entry["superseded_by"] == created_ids[-1]
    capacity_event = next(
        event for event in audit_events if event["action"] == "memory_capacity_compact"
    )
    assert capacity_event["writer_id"] == "memory_guardrail"
    assert capacity_event["payload"]["max_active"] == 3


def test_user_profile_field_meta_prefers_explicit_signals_and_exposes_inspection_view(
    tmp_path: Path,
):
    db_path = tmp_path / "agents.db"
    account = create_account(db_path, "dave")
    store = UserMemoryStore(db_path)

    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "en-US", "locale": "en-US"},
        summary="初始画像。",
        profile_source="inferred",
        profile_confidence=0.55,
        writer_type="system",
        writer_id="profile_inference",
    )
    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "zh-CN"},
        merge=True,
        profile_source="explicit",
        profile_confidence=1.0,
        source_session_id="session-explicit",
        source_run_id="run-explicit",
        writer_type="tool",
        writer_id="record_note",
    )
    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "fr-FR"},
        merge=True,
        profile_source="inferred",
        profile_confidence=0.4,
        writer_type="system",
        writer_id="profile_inference",
    )

    inspection = store.inspect_user_profile(account["id"])

    assert inspection is not None
    assert inspection["profile"]["preferred_language"] == "zh-CN"
    assert inspection["profile"]["locale"] == "en-US"
    assert inspection["field_meta"]["preferred_language"]["source"] == "explicit"
    assert inspection["field_meta"]["preferred_language"]["confidence"] == 1.0
    assert inspection["field_meta"]["preferred_language"]["source_session_id"] == "session-explicit"
    assert inspection["normalized_profile"]["preferred_language"] == "zh-CN"


def test_user_memory_store_updates_profile_fields_and_soft_deletes_memory(tmp_path: Path):
    db_path = tmp_path / "agents.db"
    account = create_account(db_path, "frank")
    store = UserMemoryStore(db_path)

    store.upsert_user_profile(
        account["id"],
        profile={
            "preferred_language": "zh-CN",
            "timezone": "Asia/Shanghai",
            "recurring_projects": ["Clavi Agent"],
        },
        summary="初始画像。",
        writer_type="tool",
        writer_id="record_note",
    )
    updated_profile = store.update_user_profile(
        account["id"],
        profile_updates={"preferred_language": "en-US"},
        remove_fields=["timezone"],
        profile_source="explicit",
        profile_confidence=1.0,
        writer_type="user",
        writer_id="web_ui",
    )
    memory = store.create_memory_entry(
        user_id=account["id"],
        memory_type="workflow_fact",
        content="修改代码后要先跑相关测试。",
        summary="先测试再提交。",
        writer_type="tool",
        writer_id="record_note",
        confidence=0.8,
    )
    deleted = store.delete_memory_entry(
        memory["id"],
        user_id=account["id"],
        reason="用户确认该条记忆不再适用。",
        writer_type="user",
        writer_id="web_ui",
    )

    inspection = store.inspect_user_profile(account["id"])
    active_entries = store.list_memory_entries(account["id"])
    search_entries = store.search_memory_entries(account["id"], query="测试", limit=10)
    deleted_entry = store.get_memory_entry(
        memory["id"],
        user_id=account["id"],
        include_deleted=True,
    )
    audit_events = store.list_audit_events(
        account["id"],
        target_scope="user_memory",
        target_id=memory["id"],
        limit=10,
    )
    profile_audit = store.list_audit_events(
        account["id"],
        target_scope="user_profile",
        target_id=account["id"],
        limit=10,
    )

    assert updated_profile is not None
    assert deleted is True
    assert inspection is not None
    assert inspection["profile"]["preferred_language"] == "en-US"
    assert "timezone" not in inspection["profile"]
    assert inspection["field_meta"]["preferred_language"]["writer_id"] == "web_ui"
    assert active_entries == []
    assert search_entries == []
    assert deleted_entry is not None
    assert deleted_entry["is_deleted"] is True
    assert deleted_entry["deleted_reason"] == "用户确认该条记忆不再适用。"
    assert audit_events[0]["action"] == "memory_delete"
    assert audit_events[0]["payload"]["reason"] == "用户确认该条记忆不再适用。"
    assert profile_audit[0]["action"] == "profile_update"
    assert profile_audit[0]["payload"]["removed_fields"] == ["timezone"]


