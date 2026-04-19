"""Typed models for persisted agent templates, policies, and runtime snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any, Literal

from pydantic import BaseModel, Field

from .account_constants import ROOT_ACCOUNT_ID
from .llm_routing_models import LLMRoutingPolicy

if TYPE_CHECKING:
    from collections.abc import Mapping


class SkillMetadata(BaseModel):
    """Normalized persisted metadata for one installed skill."""

    name: str
    description: str = ""


class WorkspacePolicy(BaseModel):
    """Workspace-related template policy."""

    mode: Literal["isolated", "shared"] = "isolated"
    allow_session_override: bool = True
    readable_roots: list[str] = Field(default_factory=list)
    writable_roots: list[str] = Field(default_factory=list)
    read_only_tools: list[str] = Field(default_factory=list)
    disabled_tools: list[str] = Field(default_factory=list)
    allowed_shell_command_prefixes: list[str] = Field(default_factory=list)
    allowed_network_domains: list[str] = Field(default_factory=list)


class ApprovalPolicy(BaseModel):
    """Approval and policy defaults carried by one template."""

    mode: Literal["default", "on-request", "strict"] = "default"
    require_approval_tools: list[str] = Field(default_factory=list)
    auto_approve_tools: list[str] = Field(default_factory=list)
    require_approval_risk_levels: list[Literal["low", "medium", "high", "critical"]] = (
        Field(default_factory=list)
    )
    require_approval_risk_categories: list[str] = Field(default_factory=list)
    notes: str = ""


class RunPolicy(BaseModel):
    """Default execution policy captured on one template."""

    timeout_seconds: int | None = Field(default=None, ge=1)
    max_concurrent_runs: int = Field(default=1, ge=1)


class DelegationPolicy(BaseModel):
    """主/子 agent 委派边界策略。"""

    mode: Literal["hybrid", "prefer_delegate", "supervisor_only"] = "prefer_delegate"
    require_delegate_for_write_actions: bool = False
    require_delegate_for_shell: bool = False
    require_delegate_for_stateful_mcp: bool = False
    allow_main_agent_read_tools: bool = True
    verify_worker_output: bool = True
    prefer_batch_delegate: bool = True


class WorkspacePolicyOverride(BaseModel):
    """Session-scoped overrides applied on top of one resolved workspace policy."""

    mode: Literal["isolated", "shared"] | None = None
    readable_roots: list[str] | None = None
    writable_roots: list[str] | None = None
    read_only_tools: list[str] | None = None
    disabled_tools: list[str] | None = None
    allowed_shell_command_prefixes: list[str] | None = None
    allowed_network_domains: list[str] | None = None


class ApprovalPolicyOverride(BaseModel):
    """Session-scoped overrides applied on top of one resolved approval policy."""

    mode: Literal["default", "on-request", "strict"] | None = None
    require_approval_tools: list[str] | None = None
    auto_approve_tools: list[str] | None = None
    require_approval_risk_levels: list[Literal["low", "medium", "high", "critical"]] | None = None
    require_approval_risk_categories: list[str] | None = None
    notes: str | None = None


class RunPolicyOverride(BaseModel):
    """Session-scoped overrides applied on top of one resolved run policy."""

    timeout_seconds: int | None = Field(default=None, ge=1)
    max_concurrent_runs: int | None = Field(default=None, ge=1)


class DelegationPolicyOverride(BaseModel):
    """Session-scoped overrides applied on top of one resolved delegation policy."""

    mode: Literal["hybrid", "prefer_delegate", "supervisor_only"] | None = None
    require_delegate_for_write_actions: bool | None = None
    require_delegate_for_shell: bool | None = None
    require_delegate_for_stateful_mcp: bool | None = None
    allow_main_agent_read_tools: bool | None = None
    verify_worker_output: bool | None = None
    prefer_batch_delegate: bool | None = None


class SessionPolicyOverride(BaseModel):
    """One run-scoped policy override layer applied between template and runtime."""

    workspace_policy: WorkspacePolicyOverride | None = None
    approval_policy: ApprovalPolicyOverride | None = None
    run_policy: RunPolicyOverride | None = None
    delegation_policy: DelegationPolicyOverride | None = None

    def has_overrides(self) -> bool:
        """Whether the payload contains any effective session override."""
        return any(
            policy is not None and policy.model_dump(exclude_unset=True) != {}
            for policy in (
                self.workspace_policy,
                self.approval_policy,
                self.run_policy,
                self.delegation_policy,
            )
        )

    def to_metadata_payload(self) -> dict[str, Any]:
        """Serialize one override payload for run metadata/audit fields."""
        return self.model_dump(
            mode="python",
            exclude_none=True,
            exclude_unset=True,
        )


class AgentTemplateSnapshot(BaseModel):
    """Immutable template snapshot captured when a run starts."""

    template_id: str
    account_id: str | None = None
    template_version: int = Field(default=1, ge=1)
    captured_at: str
    name: str
    description: str = ""
    system_prompt: str
    skills: list[SkillMetadata] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    mcp_configs: list[dict[str, Any]] = Field(default_factory=list)
    workspace_policy: WorkspacePolicy = Field(default_factory=WorkspacePolicy)
    approval_policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    run_policy: RunPolicy = Field(default_factory=RunPolicy)
    delegation_policy: DelegationPolicy = Field(default_factory=DelegationPolicy)
    llm_routing_policy: LLMRoutingPolicy = Field(default_factory=LLMRoutingPolicy)

    @property
    def workspace_type(self) -> str:
        """Compatibility alias for callers that still expect workspace_type."""
        return self.workspace_policy.mode


class AgentTemplateRecord(BaseModel):
    """Persisted agent template with normalized policy fields."""

    id: str
    account_id: str | None = ROOT_ACCOUNT_ID
    name: str
    description: str = ""
    system_prompt: str
    skills: list[SkillMetadata] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    mcp_configs: list[dict[str, Any]] = Field(default_factory=list)
    workspace_policy: WorkspacePolicy = Field(default_factory=WorkspacePolicy)
    approval_policy: ApprovalPolicy = Field(default_factory=ApprovalPolicy)
    run_policy: RunPolicy = Field(default_factory=RunPolicy)
    delegation_policy: DelegationPolicy = Field(default_factory=DelegationPolicy)
    llm_routing_policy: LLMRoutingPolicy = Field(default_factory=LLMRoutingPolicy)
    version: int = Field(default=1, ge=1)
    is_system: bool = False
    created_at: str
    updated_at: str

    @property
    def workspace_type(self) -> str:
        """Compatibility alias for callers that still expect workspace_type."""
        return self.workspace_policy.mode

    @property
    def identity(self) -> dict[str, str]:
        """Return identity and descriptive fields grouped together."""
        return {
            "name": self.name,
            "description": self.description,
        }

    @property
    def prompt_config(self) -> dict[str, str]:
        """Return prompt-related fields."""
        return {"system_prompt": self.system_prompt}

    @property
    def capability_config(self) -> dict[str, Any]:
        """Return tools, skills, and MCP configuration fields."""
        return {
            "tools": list(self.tools),
            "skills": [skill.model_dump(mode="python") for skill in self.skills],
            "mcp_configs": list(self.mcp_configs),
        }

    @property
    def policy_config(self) -> dict[str, Any]:
        """Return workspace and approval policy fields."""
        return {
            "workspace_policy": self.workspace_policy.model_dump(mode="python"),
            "approval_policy": self.approval_policy.model_dump(mode="python"),
            "run_policy": self.run_policy.model_dump(mode="python"),
            "delegation_policy": self.delegation_policy.model_dump(mode="python"),
            "llm_routing_policy": self.llm_routing_policy.model_dump(mode="python"),
        }

    def to_legacy_dict(self) -> dict[str, Any]:
        """Serialize to the legacy dict payload shape used across the app."""
        payload = self.model_dump(mode="python")
        payload["workspace_type"] = self.workspace_type
        return payload

    def snapshot(self, captured_at: str) -> AgentTemplateSnapshot:
        """Freeze the current template state for a future run."""
        return AgentTemplateSnapshot(
            template_id=self.id,
            account_id=self.account_id,
            template_version=self.version,
            captured_at=captured_at,
            name=self.name,
            description=self.description,
            system_prompt=self.system_prompt,
            skills=self.skills,
            tools=self.tools,
            mcp_configs=self.mcp_configs,
            workspace_policy=self.workspace_policy,
            approval_policy=self.approval_policy,
            run_policy=self.run_policy,
            delegation_policy=self.delegation_policy,
            llm_routing_policy=self.llm_routing_policy,
        )


def resolve_template_snapshot_with_policies(
    *,
    template_snapshot: AgentTemplateSnapshot,
    system_default_snapshot: AgentTemplateSnapshot | None = None,
    session_override: SessionPolicyOverride | Mapping[str, Any] | None = None,
) -> AgentTemplateSnapshot:
    """Resolve `system default -> template -> session override` for one run snapshot."""
    resolved = template_snapshot
    if (
        system_default_snapshot is not None
        and system_default_snapshot.template_id != template_snapshot.template_id
    ):
        resolved = resolved.model_copy(
            update={
                "workspace_policy": _merge_policy_model(
                    system_default_snapshot.workspace_policy,
                    resolved.workspace_policy.model_dump(
                        mode="python",
                        exclude_none=True,
                    ),
                ),
                "approval_policy": _merge_policy_model(
                    system_default_snapshot.approval_policy,
                    resolved.approval_policy.model_dump(
                        mode="python",
                        exclude_none=True,
                    ),
                ),
                "run_policy": _merge_policy_model(
                    system_default_snapshot.run_policy,
                    resolved.run_policy.model_dump(
                        mode="python",
                        exclude_none=True,
                    ),
                ),
                "delegation_policy": _merge_policy_model(
                    system_default_snapshot.delegation_policy,
                    resolved.delegation_policy.model_dump(
                        mode="python",
                        exclude_none=True,
                    ),
                ),
                "llm_routing_policy": _merge_policy_model(
                    system_default_snapshot.llm_routing_policy,
                    resolved.llm_routing_policy.model_dump(
                        mode="python",
                        exclude_none=True,
                    ),
                ),
            }
        )

    override_model = _normalize_session_policy_override(session_override)
    if override_model is None or not override_model.has_overrides():
        return resolved

    if not resolved.workspace_policy.allow_session_override:
        raise ValueError(
            f"Agent template '{resolved.template_id}' does not allow session policy overrides."
        )

    updates: dict[str, Any] = {}
    if override_model.workspace_policy is not None:
        updates["workspace_policy"] = _merge_policy_model(
            resolved.workspace_policy,
            override_model.workspace_policy.model_dump(exclude_unset=True),
        )
    if override_model.approval_policy is not None:
        updates["approval_policy"] = _merge_policy_model(
            resolved.approval_policy,
            override_model.approval_policy.model_dump(exclude_unset=True),
        )
    if override_model.run_policy is not None:
        updates["run_policy"] = _merge_policy_model(
            resolved.run_policy,
            override_model.run_policy.model_dump(exclude_unset=True),
        )
    if override_model.delegation_policy is not None:
        updates["delegation_policy"] = _merge_policy_model(
            resolved.delegation_policy,
            override_model.delegation_policy.model_dump(exclude_unset=True),
        )
    return resolved.model_copy(update=updates)


def _normalize_session_policy_override(
    session_override: SessionPolicyOverride | Mapping[str, Any] | None,
) -> SessionPolicyOverride | None:
    if session_override is None:
        return None
    if isinstance(session_override, SessionPolicyOverride):
        return session_override
    return SessionPolicyOverride.model_validate(dict(session_override))


def _merge_policy_model(base_policy: BaseModel, override_payload: Mapping[str, Any]) -> BaseModel:
    merged_payload = base_policy.model_dump(mode="python")
    merged_payload.update(dict(override_payload))
    return type(base_policy).model_validate(merged_payload)
