"""Persistent storage for agent templates backed by SQLite."""

from __future__ import annotations

import json
import shutil
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .account_constants import ROOT_ACCOUNT_ID
from .agent_template_models import (
    AgentTemplateRecord,
    AgentTemplateSnapshot,
    ApprovalPolicy,
    DelegationPolicy,
    RunPolicy,
    WorkspacePolicy,
)
from .llm_routing_models import LLMRoutingPolicy
from .sqlite_schema import configure_connection, ensure_agent_db_schema, utc_now_iso
from .tools.skill_loader import SkillLoader


class AgentStore:
    """SQLite repository for persisted agent templates."""

    def __init__(
        self,
        db_path: str | Path,
        skills_library_dir: str | Path | None = None,
        agent_data_dir: str | Path | None = None,
    ):
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.agent_data_dir = (
            Path(agent_data_dir).resolve()
            if agent_data_dir is not None
            else self.db_path.parent / "agents"
        )
        self.agent_data_dir.mkdir(parents=True, exist_ok=True)
        # Reserved for future compatibility; skills are now installed per-agent.
        self.skills_library_dir = (
            Path(skills_library_dir).resolve() if skills_library_dir is not None else None
        )
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return configure_connection(sqlite3.connect(self.db_path))

    @staticmethod
    def _json_loads(raw: str | None, fallback: Any) -> Any:
        """Decode JSON payloads and fall back on malformed data."""
        if not raw:
            return fallback
        try:
            return json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return fallback

    def _initialize(self):
        with self._connect() as conn:
            ensure_agent_db_schema(conn)

    def _normalize_workspace_policy(
        self,
        workspace_type: str | None = None,
        workspace_policy: WorkspacePolicy | dict[str, Any] | None = None,
    ) -> WorkspacePolicy:
        """Normalize persisted workspace policy data."""
        if isinstance(workspace_policy, WorkspacePolicy):
            policy = workspace_policy
        else:
            payload = dict(workspace_policy or {})
            if not payload.get("mode"):
                payload["mode"] = workspace_type or "isolated"
            policy = WorkspacePolicy.model_validate(payload)

        if workspace_type and policy.mode != workspace_type:
            policy = policy.model_copy(update={"mode": workspace_type})
        return policy

    def _normalize_approval_policy(
        self,
        approval_policy: ApprovalPolicy | dict[str, Any] | None = None,
    ) -> ApprovalPolicy:
        """Normalize persisted approval policy data."""
        if isinstance(approval_policy, ApprovalPolicy):
            return approval_policy
        return ApprovalPolicy.model_validate(approval_policy or {})

    def _normalize_run_policy(
        self,
        run_policy: RunPolicy | dict[str, Any] | None = None,
    ) -> RunPolicy:
        """Normalize persisted run policy data."""
        if isinstance(run_policy, RunPolicy):
            return run_policy
        return RunPolicy.model_validate(run_policy or {})

    def _normalize_delegation_policy(
        self,
        delegation_policy: DelegationPolicy | dict[str, Any] | None = None,
    ) -> DelegationPolicy:
        """Normalize persisted delegation policy data."""
        if isinstance(delegation_policy, DelegationPolicy):
            return delegation_policy
        return DelegationPolicy.model_validate(delegation_policy or {})

    def _normalize_llm_routing_policy(
        self,
        llm_routing_policy: LLMRoutingPolicy | dict[str, Any] | None = None,
    ) -> LLMRoutingPolicy:
        """Normalize persisted LLM routing policy data."""
        if isinstance(llm_routing_policy, LLMRoutingPolicy):
            return llm_routing_policy
        return LLMRoutingPolicy.model_validate(llm_routing_policy or {})

    def _template_from_row(self, row: sqlite3.Row) -> AgentTemplateRecord:
        """Hydrate a typed agent template record from SQLite."""
        workspace_policy = self._normalize_workspace_policy(
            workspace_type=row["workspace_type"],
            workspace_policy=self._json_loads(row["workspace_policy_json"], {}),
        )
        approval_policy = self._normalize_approval_policy(
            self._json_loads(row["approval_policy_json"], {})
        )
        run_policy = self._normalize_run_policy(
            self._json_loads(row["run_policy_json"], {})
        )
        delegation_policy = self._normalize_delegation_policy(
            self._json_loads(row["delegation_policy_json"], {})
        )
        llm_routing_policy = self._normalize_llm_routing_policy(
            self._json_loads(row["llm_routing_policy_json"], {})
        )
        return AgentTemplateRecord(
            id=row["id"],
            account_id=row["account_id"],
            name=row["name"],
            description=row["description"],
            system_prompt=row["system_prompt"],
            skills=self._normalize_skills(self._json_loads(row["skills_json"], [])),
            tools=list(self._json_loads(row["tools_json"], [])),
            mcp_configs=list(self._json_loads(row["mcp_configs_json"], [])),
            workspace_policy=workspace_policy,
            approval_policy=approval_policy,
            run_policy=run_policy,
            delegation_policy=delegation_policy,
            llm_routing_policy=llm_routing_policy,
            version=int(row["template_version"] or 1),
            is_system=bool(row["is_system"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _serialize_agent(self, row: sqlite3.Row) -> dict[str, Any]:
        """Serialize a row to the legacy dict payload used by current callers."""
        return self._template_from_row(row).to_legacy_dict()

    @staticmethod
    def generate_agent_id() -> str:
        """Generate a new agent identifier."""
        return str(uuid.uuid4())

    def _get_legacy_flat_agent_dir(self, agent_id: str) -> Path:
        return self.agent_data_dir / agent_id

    def _get_system_agent_root(self) -> Path:
        return self.agent_data_dir / "system"

    def _get_account_agent_root(self, account_id: str) -> Path:
        return self.agent_data_dir / "accounts" / account_id

    def _resolve_agent_storage_context(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        is_system: bool | None = None,
    ) -> tuple[str | None, bool | None]:
        resolved_account_id = account_id
        resolved_is_system = is_system

        if resolved_account_id is None or resolved_is_system is None:
            record = self.get_agent_template_record(
                agent_id,
                account_id=account_id,
                include_system=True,
            )
            if record is not None:
                if resolved_account_id is None:
                    resolved_account_id = record.account_id
                if resolved_is_system is None:
                    resolved_is_system = record.is_system

        return resolved_account_id, resolved_is_system

    def _merge_agent_directory(self, source_dir: Path, target_dir: Path) -> None:
        for child in list(source_dir.iterdir()):
            target_child = target_dir / child.name
            if not target_child.exists():
                child.replace(target_child)
                continue
            if child.is_dir() and target_child.is_dir():
                self._merge_agent_directory(child, target_child)
                if child.exists() and not any(child.iterdir()):
                    child.rmdir()

        if source_dir.exists() and not any(source_dir.iterdir()):
            source_dir.rmdir()

    def _ensure_agent_storage_dir(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        is_system: bool | None = None,
    ) -> Path:
        resolved_account_id, resolved_is_system = self._resolve_agent_storage_context(
            agent_id,
            account_id=account_id,
            is_system=is_system,
        )

        if resolved_is_system:
            target_dir = self._get_system_agent_root() / agent_id
        elif resolved_account_id:
            target_dir = self._get_account_agent_root(resolved_account_id) / agent_id
        else:
            target_dir = self._get_legacy_flat_agent_dir(agent_id)

        legacy_dir = self._get_legacy_flat_agent_dir(agent_id)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        if legacy_dir != target_dir and legacy_dir.exists():
            if target_dir.exists():
                self._merge_agent_directory(legacy_dir, target_dir)
            else:
                legacy_dir.replace(target_dir)
        return target_dir

    def get_agent_dir(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        is_system: bool | None = None,
    ) -> Path:
        """Return the root directory for one agent's persisted assets."""
        return self._ensure_agent_storage_dir(
            agent_id,
            account_id=account_id,
            is_system=is_system,
        )

    def _get_legacy_agent_skills_dir(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        is_system: bool | None = None,
    ) -> Path:
        """Return the pre-workspace skills directory for one agent."""
        return self.get_agent_dir(
            agent_id,
            account_id=account_id,
            is_system=is_system,
        ) / "skills"

    def get_agent_skills_dir(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        is_system: bool | None = None,
    ) -> Path:
        """Return the installed skills directory for one agent.

        Skills now live under the agent workspace so agents can access them
        without leaving their workspace boundary. If a legacy top-level skills
        folder exists, migrate it lazily on first access.
        """
        workspace_skills_dir = (
            self.get_agent_workspace_dir(
                agent_id,
                account_id=account_id,
                is_system=is_system,
            )
            / ".clavi_agent"
            / "skills"
        )
        legacy_skills_dir = self._get_legacy_agent_skills_dir(
            agent_id,
            account_id=account_id,
            is_system=is_system,
        )

        if legacy_skills_dir.exists() and not workspace_skills_dir.exists():
            workspace_skills_dir.parent.mkdir(parents=True, exist_ok=True)
            legacy_skills_dir.replace(workspace_skills_dir)
        elif legacy_skills_dir.exists() and workspace_skills_dir.exists():
            # Older agents may still have a legacy top-level skills folder.
            # If the new location is empty, replace it wholesale; otherwise
            # merge non-conflicting entries and remove the legacy folder once empty.
            if not any(workspace_skills_dir.iterdir()):
                shutil.rmtree(workspace_skills_dir, ignore_errors=True)
                workspace_skills_dir.parent.mkdir(parents=True, exist_ok=True)
                legacy_skills_dir.replace(workspace_skills_dir)
            else:
                for child in list(legacy_skills_dir.iterdir()):
                    target = workspace_skills_dir / child.name
                    if target.exists():
                        continue
                    child.replace(target)
                if not any(legacy_skills_dir.iterdir()):
                    legacy_skills_dir.rmdir()

        return workspace_skills_dir

    def get_agent_workspace_dir(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        is_system: bool | None = None,
    ) -> Path:
        """Return the dedicated runtime workspace directory for one agent."""
        return (
            self.get_agent_dir(
                agent_id,
                account_id=account_id,
                is_system=is_system,
            )
            / "workspace"
        )

    def _normalize_skills(self, skills: list[Any] | None) -> list[dict[str, str]]:
        """Normalize persisted skill metadata."""
        if not skills:
            return []

        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in skills:
            if isinstance(item, str):
                name = item.strip()
                description = ""
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                description = str(item.get("description", "")).strip()
            else:
                raise ValueError("Skills must be strings or objects with a 'name' field.")

            if not name or name in seen:
                continue

            normalized.append({"name": name, "description": description})
            seen.add(name)

        return normalized

    def collect_skills_metadata_from_directory(self, skills_dir: str | Path) -> list[dict[str, str]]:
        """Scan a skills directory and return normalized skill metadata."""
        path = Path(skills_dir)
        if not path.exists():
            return []

        loader = SkillLoader(str(path))
        loader.discover_skills()
        return self._normalize_skills(loader.get_skills_metadata())

    def _set_agent_skills(
        self,
        agent_id: str,
        skills: list[Any],
        *,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Persist normalized skill metadata for one agent."""
        agent = self.get_agent_template(agent_id, account_id=account_id)
        if not agent:
            return None

        normalized = self._normalize_skills(skills)
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_templates
                SET skills_json = ?, template_version = template_version + 1, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(normalized, ensure_ascii=False), now, agent_id),
            )
        return self.get_agent_template(agent_id, account_id=account_id)

    def refresh_agent_skills_from_directory(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Scan the agent-specific skills directory and persist discovered metadata."""
        skills_dir = self.get_agent_skills_dir(agent_id)
        if not skills_dir.exists():
            skills_dir.mkdir(parents=True, exist_ok=True)
            return self._set_agent_skills(agent_id, [], account_id=account_id)
        return self._set_agent_skills(
            agent_id,
            self.collect_skills_metadata_from_directory(skills_dir),
            account_id=account_id,
        )

    def delete_agent_skill(
        self,
        agent_id: str,
        skill_name: str,
        *,
        account_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Delete one installed skill from the agent-specific skills directory."""
        agent = self.get_agent(agent_id, account_id=account_id)
        if not agent:
            return None

        if agent["is_system"]:
            raise ValueError(f"Cannot manually modify system agent: {agent_id}")

        normalized = skill_name.strip()
        if not normalized:
            raise ValueError("Skill name is required.")

        skills_dir = self.get_agent_skills_dir(agent_id)
        skills_dir.mkdir(parents=True, exist_ok=True)
        resolved_skills_dir = skills_dir.resolve()
        target_path = (skills_dir / normalized).resolve()

        if target_path == resolved_skills_dir or resolved_skills_dir not in target_path.parents:
            raise ValueError("Invalid skill name.")

        if not target_path.exists():
            raise FileNotFoundError(f"Installed skill not found: {skill_name}")

        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

        return self.refresh_agent_skills_from_directory(agent_id, account_id=account_id)

    def create_agent_template(
        self,
        name: str,
        system_prompt: str,
        description: str = "",
        account_id: str | None = ROOT_ACCOUNT_ID,
        agent_id: str | None = None,
        skills: list[Any] | None = None,
        tools: list[str] | None = None,
        mcp_configs: list[dict] | None = None,
        workspace_type: str = "isolated",
        workspace_policy: WorkspacePolicy | dict[str, Any] | None = None,
        approval_policy: ApprovalPolicy | dict[str, Any] | None = None,
        run_policy: RunPolicy | dict[str, Any] | None = None,
        delegation_policy: DelegationPolicy | dict[str, Any] | None = None,
        llm_routing_policy: LLMRoutingPolicy | dict[str, Any] | None = None,
        is_system: bool = False,
    ) -> dict[str, Any]:
        """Create a new agent template."""
        agent_id = agent_id or self.generate_agent_id()
        resolved_account_id = None if is_system else str(account_id or ROOT_ACCOUNT_ID)
        now = utc_now_iso()
        skills_json = json.dumps(self._normalize_skills(skills), ensure_ascii=False)
        tools_json = json.dumps(tools or [], ensure_ascii=False)
        mcp_configs_json = json.dumps(mcp_configs or [], ensure_ascii=False)
        workspace_policy_model = self._normalize_workspace_policy(
            workspace_type=workspace_type,
            workspace_policy=workspace_policy,
        )
        approval_policy_model = self._normalize_approval_policy(approval_policy)
        run_policy_model = self._normalize_run_policy(run_policy)
        delegation_policy_model = self._normalize_delegation_policy(delegation_policy)
        llm_routing_policy_model = self._normalize_llm_routing_policy(llm_routing_policy)

        self.get_agent_skills_dir(
            agent_id,
            account_id=resolved_account_id,
            is_system=is_system,
        ).mkdir(parents=True, exist_ok=True)
        self.get_agent_workspace_dir(
            agent_id,
            account_id=resolved_account_id,
            is_system=is_system,
        ).mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_templates (
                    id, account_id, name, description, system_prompt, skills_json, tools_json, mcp_configs_json,
                    workspace_type, workspace_policy_json, approval_policy_json, run_policy_json, delegation_policy_json,
                    llm_routing_policy_json, template_version,
                    is_system, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    agent_id,
                    resolved_account_id,
                    name,
                    description,
                    system_prompt,
                    skills_json,
                    tools_json,
                    mcp_configs_json,
                    workspace_policy_model.mode,
                    json.dumps(workspace_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(approval_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(run_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(delegation_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(llm_routing_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    1,
                    int(is_system),
                    now,
                    now,
                ),
            )
        return self.get_agent_template(agent_id, account_id=resolved_account_id)

    def create_agent(
        self,
        name: str,
        system_prompt: str,
        description: str = "",
        account_id: str | None = ROOT_ACCOUNT_ID,
        agent_id: str | None = None,
        skills: list[Any] | None = None,
        tools: list[str] | None = None,
        mcp_configs: list[dict] | None = None,
        workspace_type: str = "isolated",
        workspace_policy: WorkspacePolicy | dict[str, Any] | None = None,
        approval_policy: ApprovalPolicy | dict[str, Any] | None = None,
        run_policy: RunPolicy | dict[str, Any] | None = None,
        delegation_policy: DelegationPolicy | dict[str, Any] | None = None,
        llm_routing_policy: LLMRoutingPolicy | dict[str, Any] | None = None,
        is_system: bool = False,
    ) -> dict[str, Any]:
        """Compatibility wrapper for create_agent_template."""
        return self.create_agent_template(
            name=name,
            system_prompt=system_prompt,
            description=description,
            account_id=account_id,
            agent_id=agent_id,
            skills=skills,
            tools=tools,
            mcp_configs=mcp_configs,
            workspace_type=workspace_type,
            workspace_policy=workspace_policy,
            approval_policy=approval_policy,
            run_policy=run_policy,
            delegation_policy=delegation_policy,
            llm_routing_policy=llm_routing_policy,
            is_system=is_system,
        )

    def update_agent_template(
        self,
        agent_id: str,
        account_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        skills: list[Any] | None = None,
        tools: list[str] | None = None,
        mcp_configs: list[dict] | None = None,
        workspace_type: str | None = None,
        workspace_policy: WorkspacePolicy | dict[str, Any] | None = None,
        approval_policy: ApprovalPolicy | dict[str, Any] | None = None,
        run_policy: RunPolicy | dict[str, Any] | None = None,
        delegation_policy: DelegationPolicy | dict[str, Any] | None = None,
        llm_routing_policy: LLMRoutingPolicy | dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Update an existing agent template."""
        template_record = self.get_agent_template_record(agent_id, account_id=account_id)
        if not template_record:
            return None

        if template_record.is_system:
            raise ValueError(f"Cannot manually modify system agent: {agent_id}")

        now = utc_now_iso()
        n_name = name if name is not None else template_record.name
        n_desc = description if description is not None else template_record.description
        n_prompt = system_prompt if system_prompt is not None else template_record.system_prompt
        n_skills = (
            self._normalize_skills(skills)
            if skills is not None
            else [skill.model_dump(mode="python") for skill in template_record.skills]
        )
        n_skills_json = json.dumps(n_skills, ensure_ascii=False)
        n_tools = json.dumps(
            tools if tools is not None else template_record.tools,
            ensure_ascii=False,
        )
        n_mcp = json.dumps(
            mcp_configs if mcp_configs is not None else template_record.mcp_configs,
            ensure_ascii=False,
        )
        workspace_policy_model = (
            self._normalize_workspace_policy(
                workspace_type=workspace_type or template_record.workspace_policy.mode,
                workspace_policy=(
                    workspace_policy
                    if workspace_policy is not None
                    else template_record.workspace_policy.model_dump(mode="python")
                ),
            )
            if workspace_policy is not None or workspace_type is not None
            else template_record.workspace_policy
        )
        approval_policy_model = (
            self._normalize_approval_policy(approval_policy)
            if approval_policy is not None
            else template_record.approval_policy
        )
        run_policy_model = (
            self._normalize_run_policy(run_policy)
            if run_policy is not None
            else template_record.run_policy
        )
        delegation_policy_model = (
            self._normalize_delegation_policy(delegation_policy)
            if delegation_policy is not None
            else template_record.delegation_policy
        )
        llm_routing_policy_model = (
            self._normalize_llm_routing_policy(llm_routing_policy)
            if llm_routing_policy is not None
            else template_record.llm_routing_policy
        )

        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_templates
                SET account_id = ?, name = ?, description = ?, system_prompt = ?, skills_json = ?, tools_json = ?,
                    mcp_configs_json = ?, workspace_type = ?, workspace_policy_json = ?,
                    approval_policy_json = ?, run_policy_json = ?, delegation_policy_json = ?, llm_routing_policy_json = ?,
                    template_version = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    template_record.account_id,
                    n_name,
                    n_desc,
                    n_prompt,
                    n_skills_json,
                    n_tools,
                    n_mcp,
                    workspace_policy_model.mode,
                    json.dumps(workspace_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(approval_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(run_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(delegation_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    json.dumps(llm_routing_policy_model.model_dump(mode="python"), ensure_ascii=False),
                    template_record.version + 1,
                    now,
                    agent_id,
                ),
            )
        return self.get_agent_template(agent_id, account_id=template_record.account_id)

    def update_agent(
        self,
        agent_id: str,
        account_id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        system_prompt: str | None = None,
        skills: list[Any] | None = None,
        tools: list[str] | None = None,
        mcp_configs: list[dict] | None = None,
        workspace_type: str | None = None,
        workspace_policy: WorkspacePolicy | dict[str, Any] | None = None,
        approval_policy: ApprovalPolicy | dict[str, Any] | None = None,
        run_policy: RunPolicy | dict[str, Any] | None = None,
        delegation_policy: DelegationPolicy | dict[str, Any] | None = None,
        llm_routing_policy: LLMRoutingPolicy | dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Compatibility wrapper for update_agent_template."""
        return self.update_agent_template(
            agent_id=agent_id,
            account_id=account_id,
            name=name,
            description=description,
            system_prompt=system_prompt,
            skills=skills,
            tools=tools,
            mcp_configs=mcp_configs,
            workspace_type=workspace_type,
            workspace_policy=workspace_policy,
            approval_policy=approval_policy,
            run_policy=run_policy,
            delegation_policy=delegation_policy,
            llm_routing_policy=llm_routing_policy,
        )

    def get_agent_template_record(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        include_system: bool = True,
    ) -> AgentTemplateRecord | None:
        """Get a typed agent template record by id."""
        params: list[Any] = [agent_id]
        sql = "SELECT * FROM agent_templates WHERE id = ?"
        if account_id is not None:
            if include_system:
                sql += " AND (is_system = 1 OR account_id = ?)"
            else:
                sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            row = conn.execute(
                sql,
                tuple(params),
            ).fetchone()

        if not row:
            return None
        return self._template_from_row(row)

    def get_agent_template(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        include_system: bool = True,
    ) -> dict[str, Any] | None:
        """Get agent template by id as a compatibility dict payload."""
        record = self.get_agent_template_record(
            agent_id,
            account_id=account_id,
            include_system=include_system,
        )
        if record is None:
            return None
        return record.to_legacy_dict()

    def get_agent(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        include_system: bool = True,
    ) -> dict[str, Any] | None:
        """Compatibility wrapper for get_agent_template."""
        return self.get_agent_template(
            agent_id,
            account_id=account_id,
            include_system=include_system,
        )

    def list_agent_template_records(
        self,
        *,
        account_id: str | None = None,
        include_system: bool = True,
    ) -> list[AgentTemplateRecord]:
        """List all agent templates as typed records."""
        params: list[Any] = []
        sql = "SELECT * FROM agent_templates"
        if account_id is not None:
            if include_system:
                sql += " WHERE is_system = 1 OR account_id = ?"
            else:
                sql += " WHERE account_id = ?"
            params.append(account_id)
        sql += " ORDER BY is_system DESC, updated_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._template_from_row(row) for row in rows]

    def list_agent_templates(
        self,
        *,
        account_id: str | None = None,
        include_system: bool = True,
    ) -> list[dict[str, Any]]:
        """List all agent templates (system and custom)."""
        return [
            record.to_legacy_dict()
            for record in self.list_agent_template_records(
                account_id=account_id,
                include_system=include_system,
            )
        ]

    def list_agents(
        self,
        *,
        account_id: str | None = None,
        include_system: bool = True,
    ) -> list[dict[str, Any]]:
        """Compatibility wrapper for list_agent_templates."""
        return self.list_agent_templates(
            account_id=account_id,
            include_system=include_system,
        )

    def snapshot_agent_template(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
        include_system: bool = True,
    ) -> AgentTemplateSnapshot | None:
        """Capture an immutable snapshot of one template for a future run."""
        record = self.get_agent_template_record(
            agent_id,
            account_id=account_id,
            include_system=include_system,
        )
        if record is None:
            return None
        return record.snapshot(captured_at=utc_now_iso())

    def delete_agent_template(
        self,
        agent_id: str,
        *,
        account_id: str | None = None,
    ) -> bool:
        """Delete an agent template."""
        agent = self.get_agent_template(agent_id, account_id=account_id)
        if not agent:
            return False

        if agent["is_system"]:
            raise ValueError(f"Cannot delete system agent: {agent_id}")

        shutil.rmtree(
            self.get_agent_dir(
                agent_id,
                account_id=agent["account_id"],
                is_system=agent["is_system"],
            ),
            ignore_errors=True,
        )
        shutil.rmtree(self._get_legacy_flat_agent_dir(agent_id), ignore_errors=True)
        params: list[Any] = [agent_id]
        sql = "DELETE FROM agent_templates WHERE id = ?"
        if account_id is not None:
            sql += " AND account_id = ?"
            params.append(account_id)
        with self._connect() as conn:
            deleted = conn.execute(
                sql,
                tuple(params),
            ).rowcount
        return deleted > 0

    def delete_agent(self, agent_id: str, *, account_id: str | None = None) -> bool:
        """Compatibility wrapper for delete_agent_template."""
        return self.delete_agent_template(agent_id, account_id=account_id)

    def sync_system_agents(self, system_configs: list[dict]):
        """Upsert system templates. Updates existing system templates by id or name."""
        now = utc_now_iso()
        with self._connect() as conn:
            for config in system_configs:
                agent_id = config.get("id")
                match_row = None

                if agent_id:
                    match_row = conn.execute(
                        "SELECT id, template_version FROM agent_templates WHERE id = ?",
                        (agent_id,),
                    ).fetchone()

                if not match_row:
                    match_row = conn.execute(
                        "SELECT id, template_version FROM agent_templates WHERE name = ? AND is_system = 1",
                        (config["name"],),
                    ).fetchone()

                n_id = (match_row["id"] if match_row else None) or agent_id or str(uuid.uuid4())
                n_name = config["name"]
                n_desc = config.get("description", "")
                n_prompt = config["system_prompt"]
                n_skills = self._normalize_skills(config.get("skills", []))
                n_skills_json = json.dumps(n_skills, ensure_ascii=False)
                n_tools = json.dumps(config.get("tools", []), ensure_ascii=False)
                n_mcp = json.dumps(config.get("mcp_configs", []), ensure_ascii=False)
                workspace_policy_model = self._normalize_workspace_policy(
                    workspace_type=config.get("workspace_type", "isolated"),
                    workspace_policy=config.get("workspace_policy"),
                )
                approval_policy_model = self._normalize_approval_policy(
                    config.get("approval_policy")
                )
                run_policy_model = self._normalize_run_policy(config.get("run_policy"))
                delegation_policy_model = self._normalize_delegation_policy(
                    config.get("delegation_policy")
                )
                llm_routing_policy_model = self._normalize_llm_routing_policy(
                    config.get("llm_routing_policy")
                )
                next_version = int(match_row["template_version"] or 1) + 1 if match_row else 1

                self.get_agent_skills_dir(
                    n_id,
                    is_system=True,
                ).mkdir(parents=True, exist_ok=True)
                self.get_agent_workspace_dir(
                    n_id,
                    is_system=True,
                ).mkdir(parents=True, exist_ok=True)

                if match_row:
                    conn.execute(
                        """
                        UPDATE agent_templates
                        SET account_id = NULL, name = ?, description = ?, system_prompt = ?, skills_json = ?, tools_json = ?,
                            mcp_configs_json = ?, workspace_type = ?, workspace_policy_json = ?,
                            approval_policy_json = ?, run_policy_json = ?, delegation_policy_json = ?, llm_routing_policy_json = ?,
                            template_version = ?, is_system = 1, updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            n_name,
                            n_desc,
                            n_prompt,
                            n_skills_json,
                            n_tools,
                            n_mcp,
                            workspace_policy_model.mode,
                            json.dumps(workspace_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(approval_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(run_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(delegation_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(llm_routing_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            next_version,
                            now,
                            match_row["id"],
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO agent_templates (
                            id, account_id, name, description, system_prompt, skills_json, tools_json, mcp_configs_json,
                            workspace_type, workspace_policy_json, approval_policy_json, run_policy_json, delegation_policy_json,
                            llm_routing_policy_json, template_version,
                            is_system, created_at, updated_at
                        ) VALUES (?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1, ?, ?)
                        """,
                        (
                            n_id,
                            n_name,
                            n_desc,
                            n_prompt,
                            n_skills_json,
                            n_tools,
                            n_mcp,
                            workspace_policy_model.mode,
                            json.dumps(workspace_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(approval_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(run_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(delegation_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            json.dumps(llm_routing_policy_model.model_dump(mode="python"), ensure_ascii=False),
                            now,
                            now,
                        ),
                    )


AgentTemplateStore = AgentStore

