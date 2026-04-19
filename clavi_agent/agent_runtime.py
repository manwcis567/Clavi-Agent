"""Run-scoped agent runtime context and factory helpers."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Literal

from .agent import Agent
from .agent_store import AgentStore
from .agent_template_models import AgentTemplateSnapshot
from .config import Config
from .llm import LLMClient
from .memory_provider import MemoryProvider
from .schema import Message
from .session_store import SessionStore
from .tool_execution import UploadedFileTarget
from .tools.base import Tool
from .tools.bash_tool import BashKillTool, BashOutputTool, BashTool
from .tools.delegate_tool import DelegateBatchTool, DelegateTool
from .tools.file_tools import EditTool, ReadTool, WriteTool
from .tools.history_tool import SearchSessionHistoryTool
from .tools.note_tool import RecallNoteTool, SearchMemoryTool, SessionNoteTool
from .tools.send_channel_file_tool import SendChannelFileTool
from .tools.shared_context_tool import ReadSharedContextTool, ShareContextTool
from .tools.skill_loader import build_skills_description_prompt
from .tools.skill_tool import create_skill_tools

if TYPE_CHECKING:
    from .run_models import RunRecord


RuntimeEventType = Literal[
    "run_started",
    "llm_request",
    "llm_response",
    "tool_started",
    "tool_finished",
    "delegate_started",
    "delegate_finished",
    "checkpoint_saved",
    "approval_requested",
    "run_completed",
    "run_failed",
]

RuntimeTraceSink = Callable[["AgentRuntimeContext", dict[str, Any]], Awaitable[None] | None]
RuntimeApprovalHook = Callable[
    ["AgentRuntimeContext", Agent, dict[str, Any]],
    Awaitable[dict[str, Any] | None] | dict[str, Any] | None,
]
RuntimeCheckpointHook = Callable[["AgentRuntimeContext", Agent], Awaitable[None] | None]
RuntimePromptRefreshHook = Callable[
    ["AgentRuntimeContext", Agent, dict[str, Any]],
    Awaitable[dict[str, Any] | None] | dict[str, Any] | None,
]


@dataclass(slots=True)
class ResolvedLLMRuntime:
    """Resolved LLM client selection for one runtime role."""

    client: LLMClient
    fingerprint: str
    profile_role: Literal["planner", "worker"]
    provider: str
    api_base: str
    model: str
    reasoning_enabled: bool
    source: dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        """Keep compatibility with legacy tuple unpacking call sites."""
        yield self.client
        yield self.fingerprint

    def to_trace_payload(self) -> dict[str, Any]:
        """Build a compact trace payload for startup observability."""
        payload = {
            "profile_role": self.profile_role,
            "provider": self.provider,
            "api_base": self.api_base,
            "model": self.model,
            "reasoning_enabled": self.reasoning_enabled,
            "fingerprint": self.fingerprint,
        }
        if self.source:
            payload["source"] = dict(self.source)
        return payload


@dataclass(slots=True)
class AgentRuntimeHooks:
    """Hook endpoints injected into one run-scoped agent runtime."""

    trace_sink: RuntimeTraceSink | None = None
    approval_hook: RuntimeApprovalHook | None = None
    checkpoint_hook: RuntimeCheckpointHook | None = None
    prompt_refresh_hook: RuntimePromptRefreshHook | None = None


@dataclass(slots=True)
class AgentRuntimeContext:
    """Execution context carried by one main agent or delegated sub-agent."""

    session_id: str
    account_id: str | None = None
    run_id: str | None = None
    agent_name: str = "main"
    template_snapshot: AgentTemplateSnapshot | None = None
    parent_run_id: str | None = None
    root_run_id: str | None = None
    is_main_agent: bool = True
    depth: int = 0
    prompt_trace_data: dict[str, Any] = field(default_factory=dict)
    approval_auto_grant_tools: list[str] = field(default_factory=list)
    uploaded_file_targets: list[UploadedFileTarget] = field(default_factory=list)
    base_system_prompt: str | None = None
    prompt_retrieval_query: str | None = None
    prompt_retrieval_exclude_run_id: str | None = None
    project_context_hints: list[str] = field(default_factory=list)
    integration_id: str | None = None
    channel_kind: str | None = None
    binding_id: str | None = None
    inbound_event_id: str | None = None
    provider_chat_id: str | None = None
    provider_thread_id: str | None = None
    provider_message_id: str | None = None

    def create_child(self, agent_name: str) -> "AgentRuntimeContext":
        """Derive a legacy sub-agent context within the same run tree."""
        root_run_id = self.root_run_id or self.run_id
        return AgentRuntimeContext(
            session_id=self.session_id,
            account_id=self.account_id,
            run_id=self.run_id,
            agent_name=agent_name,
            template_snapshot=self.template_snapshot,
            parent_run_id=self.run_id or self.parent_run_id,
            root_run_id=root_run_id,
            is_main_agent=False,
            depth=self.depth + 1,
            prompt_trace_data=dict(self.prompt_trace_data),
            approval_auto_grant_tools=list(self.approval_auto_grant_tools),
            uploaded_file_targets=list(self.uploaded_file_targets),
            base_system_prompt=self.base_system_prompt,
            prompt_retrieval_query=self.prompt_retrieval_query,
            prompt_retrieval_exclude_run_id=self.prompt_retrieval_exclude_run_id,
            project_context_hints=list(self.project_context_hints),
            integration_id=self.integration_id,
            channel_kind=self.channel_kind,
            binding_id=self.binding_id,
            inbound_event_id=self.inbound_event_id,
            provider_chat_id=self.provider_chat_id,
            provider_thread_id=self.provider_thread_id,
            provider_message_id=self.provider_message_id,
        )


@dataclass(slots=True)
class PromptMemorySection:
    """One compact prompt-memory section and its trace metadata."""

    key: str
    title: str
    body: str
    source_kind: str
    char_limit: int
    item_count: int
    sources: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trace_payload(self) -> dict[str, Any]:
        """Return a compact trace-friendly view of the injected section."""
        payload = {
            "key": self.key,
            "title": self.title,
            "source": self.source_kind,
            "chars": len(self.body),
            "items": self.item_count,
            "sources": self.sources,
            "body": self.body,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class ProjectContextEntry:
    """One discovered project-context file prepared for prompt injection."""

    path: Path
    display_path: str
    content: str


class AgentRuntimeFactory:
    """Build run-scoped agents from persisted template snapshots."""
    _UTF8 = "utf-8"
    _STABLE_PREFERENCE_TYPES = (
        "preference",
        "communication_style",
        "constraint",
        "correction",
    )
    _PROJECT_CONTEXT_FILENAMES = (
        "AGENTS.md",
        ".cursorrules",
        "CLAUDE.md",
        "CONTEXT.md",
    )
    _PROJECT_CONTEXT_SECRET_PATTERNS = (
        re.compile(
            r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|secret[_-]?key|password)\b\s*[:=]\s*\S+"
        ),
        re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._-]{16,}\b"),
        re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    )

    def __init__(
        self,
        *,
        config: Config,
        llm_client_resolver: Callable[
            [AgentRuntimeContext, AgentTemplateSnapshot | None],
            ResolvedLLMRuntime,
        ],
        agent_store: AgentStore,
        shared_tools: list[Tool],
        default_system_prompt: str,
        next_sub_agent_name: Callable[[str], str],
        shared_context_path_resolver: Callable[[Path, str], Path],
        agent_db_path: Path,
        memory_provider: MemoryProvider,
        delegate_executor: Callable[..., AsyncGenerator[dict[str, Any], None]] | None = None,
        channel_file_sender: Callable[[AgentRuntimeContext, Path, str, str], Awaitable[dict[str, Any]]] | None = None,
    ):
        self._config = config
        self._llm_client_resolver = llm_client_resolver
        self._agent_store = agent_store
        self._shared_tools = list(shared_tools)
        self._default_system_prompt = default_system_prompt
        self._supervisor_policy_prompt = self._load_role_policy_prompt(
            "supervisor_policy.md",
            fallback=(
                "## Supervisor Policy\n"
                "You are the main agent. Prioritize planning, delegation, coordination, "
                "and acceptance. Prefer completing executable work through workers."
            ),
        )
        self._worker_policy_prompt = self._load_role_policy_prompt(
            "worker_policy.md",
            fallback=(
                "## Worker Execution Policy\n"
                "You are a worker agent. Execute the assigned task directly, report blockers "
                "clearly, and do not create additional sub-agents."
            ),
        )
        self._next_sub_agent_name = next_sub_agent_name
        self._shared_context_path_resolver = shared_context_path_resolver
        self._agent_db_path = agent_db_path
        self._delegate_executor = delegate_executor
        self._channel_file_sender = channel_file_sender
        self._memory_provider = memory_provider
        self._session_store: SessionStore | None = None
        self._prompt_memory_config = config.agent.prompt_memory
        self._feature_flags = config.get_feature_flags()
        session_db_path = Path(config.agent.session_store_path).expanduser()
        if not session_db_path.is_absolute():
            session_db_path = Path.cwd() / session_db_path
        self._session_db_path = session_db_path.resolve()

    @staticmethod
    def _load_role_policy_prompt(filename: str, *, fallback: str) -> str:
        """Load one role policy prompt file with UTF-8 fallback handling."""
        policy_path = Path(__file__).resolve().parent / "config" / filename
        try:
            content = policy_path.read_text(encoding="utf-8").strip()
        except OSError:
            return fallback.strip()
        return content or fallback.strip()

    def _prompt_memory_enabled(self) -> bool:
        """Whether compact prompt-memory injection is enabled for this runtime."""
        return bool(self._feature_flags.get("enable_compact_prompt_memory", True))

    def _session_retrieval_enabled(self) -> bool:
        """Whether cross-session retrieval capabilities are enabled."""
        return bool(self._feature_flags.get("enable_session_retrieval", True))

    def resolve_workspace(
        self,
        agent_id: str,
        requested_workspace_dir: str | None = None,
    ) -> Path:
        """Resolve the effective workspace directory for one session/template."""
        if requested_workspace_dir:
            return Path(requested_workspace_dir).resolve()
        return self._agent_store.get_agent_workspace_dir(agent_id).resolve()

    def build_session_agent(
        self,
        *,
        session_id: str,
        account_id: str | None,
        workspace_dir: Path,
        template_snapshot: AgentTemplateSnapshot,
        messages: list[Message] | None = None,
    ) -> Agent:
        """Build the default runtime agent bound to one session container."""
        context = AgentRuntimeContext(
            session_id=session_id,
            account_id=account_id,
            agent_name="main",
            template_snapshot=template_snapshot,
            is_main_agent=True,
        )
        return self.build_agent(
            workspace_dir=workspace_dir,
            session_id=session_id,
            template_snapshot=template_snapshot,
            runtime_context=context,
            runtime_hooks=AgentRuntimeHooks(),
            seed_messages=messages,
        )

    def prepare_run_agent(
        self,
        *,
        run: RunRecord,
        workspace_dir: Path,
        seed_messages: list[Message],
        existing_agent: Agent | None = None,
        runtime_hooks: AgentRuntimeHooks | None = None,
        uploaded_file_targets: list[UploadedFileTarget] | None = None,
    ) -> Agent:
        """Return the main run agent for one durable run."""
        base_system_prompt = (
            run.agent_template_snapshot.system_prompt or self._default_system_prompt
        )
        context = AgentRuntimeContext(
            session_id=run.session_id,
            account_id=run.account_id,
            run_id=run.id,
            agent_name="main",
            template_snapshot=run.agent_template_snapshot,
            parent_run_id=run.parent_run_id,
            root_run_id=str(
                run.run_metadata.get("root_run_id")
                or run.parent_run_id
                or run.id
            ),
            is_main_agent=True,
            approval_auto_grant_tools=list(
                run.run_metadata.get("approval_auto_grant_tools", [])
            ),
            uploaded_file_targets=list(uploaded_file_targets or []),
            base_system_prompt=base_system_prompt,
            prompt_retrieval_query=run.goal,
            prompt_retrieval_exclude_run_id=run.id,
            integration_id=str(run.run_metadata.get("integration_id") or "").strip() or None,
            channel_kind=str(run.run_metadata.get("channel_kind") or "").strip() or None,
            binding_id=str(run.run_metadata.get("binding_id") or "").strip() or None,
            inbound_event_id=str(run.run_metadata.get("inbound_event_id") or "").strip() or None,
            provider_chat_id=str(run.run_metadata.get("provider_chat_id") or "").strip() or None,
            provider_thread_id=str(run.run_metadata.get("provider_thread_id") or "").strip() or None,
            provider_message_id=str(run.run_metadata.get("provider_message_id") or "").strip() or None,
        )
        hooks = runtime_hooks or AgentRuntimeHooks()
        resolved_llm = self._llm_client_resolver(context, run.agent_template_snapshot)
        llm_fingerprint = resolved_llm.fingerprint
        sys_prompt, startup_trace_data = self._build_agent_system_prompt(
            base_system_prompt,
            [skill.model_dump(mode="python") for skill in run.agent_template_snapshot.skills],
            is_main_agent=context.is_main_agent,
            delegation_policy=run.agent_template_snapshot.delegation_policy,
            account_id=context.account_id,
            workspace_dir=workspace_dir,
            session_id=run.session_id,
            retrieval_query=run.goal,
            exclude_run_id=run.id,
            project_context_hints=context.project_context_hints,
        )

        startup_trace_data["llm"] = resolved_llm.to_trace_payload()
        context.prompt_trace_data = dict(startup_trace_data)

        if existing_agent is not None and self._can_reuse_existing_agent(
            existing_agent,
            run.agent_template_snapshot,
        ) and self._can_reuse_existing_run_agent(existing_agent) and (
            not self._requires_fresh_run_agent(context)
        ) and (
            getattr(existing_agent, "llm_fingerprint", None) == llm_fingerprint
        ):
            existing_agent.messages = list(seed_messages)
            existing_agent.runtime_prompt_seed = base_system_prompt
            existing_agent.set_system_prompt(sys_prompt)
            existing_agent.set_startup_trace_data(startup_trace_data)
            existing_agent.bind_runtime(
                runtime_context=context,
                runtime_hooks=hooks,
                template_snapshot=run.agent_template_snapshot,
            )
            return existing_agent

        return self.build_agent(
            workspace_dir=workspace_dir,
            session_id=run.session_id,
            template_snapshot=run.agent_template_snapshot,
            runtime_context=context,
            runtime_hooks=hooks,
            seed_messages=seed_messages,
            prompt_retrieval_query=run.goal,
            prompt_retrieval_exclude_run_id=run.id,
        )

    @staticmethod
    def _can_reuse_existing_run_agent(agent: Agent) -> bool:
        """Whether an existing session agent can be rebound for a durable run."""
        if agent.manual_runtime_override:
            return True

        delegate_tool = agent.tools.get("delegate_task")
        if delegate_tool is not None and getattr(delegate_tool, "_delegate_executor", None) is None:
            return False
        delegate_batch_tool = agent.tools.get("delegate_tasks")
        if delegate_batch_tool is not None and getattr(
            delegate_batch_tool,
            "_delegate_executor",
            None,
        ) is None:
            return False
        return True

    @staticmethod
    def _requires_fresh_run_agent(runtime_context: AgentRuntimeContext) -> bool:
        """Run-bound channel tools require a fresh tool set per invocation."""
        return bool(
            str(runtime_context.channel_kind or "").strip().lower() == "feishu"
            and runtime_context.integration_id
            and runtime_context.provider_chat_id
        )

    def build_agent(
        self,
        *,
        workspace_dir: Path,
        session_id: str,
        template_snapshot: AgentTemplateSnapshot,
        runtime_context: AgentRuntimeContext,
        runtime_hooks: AgentRuntimeHooks,
        is_main_agent: bool = True,
        custom_prompt: str | None = None,
        max_steps: int | None = None,
        agent_name: str = "main",
        seed_messages: list[Message] | None = None,
        prompt_retrieval_query: str | None = None,
        prompt_retrieval_exclude_run_id: str | None = None,
    ) -> Agent:
        """Build one main agent or sub-agent from a template snapshot."""
        base_system_prompt = (
            custom_prompt or template_snapshot.system_prompt or self._default_system_prompt
        )
        runtime_context.base_system_prompt = base_system_prompt
        runtime_context.prompt_retrieval_query = prompt_retrieval_query
        runtime_context.prompt_retrieval_exclude_run_id = prompt_retrieval_exclude_run_id
        all_possible_tools = list(self._shared_tools) + self._build_workspace_tools(
            workspace_dir,
            session_id=session_id,
            agent_name=agent_name,
            runtime_context=runtime_context,
        )

        if self._config.tools.enable_skills and template_snapshot.skills:
            agent_skills_dir = self._agent_store.get_agent_skills_dir(template_snapshot.template_id)
            skill_tools, _ = create_skill_tools(str(agent_skills_dir))
            all_possible_tools.extend(skill_tools)

        allowed_tool_names = {
            str(item).strip() for item in template_snapshot.tools if str(item).strip()
        }
        selected_tools: list[Tool] = []
        for tool in all_possible_tools:
            if self._tool_is_enabled_for_template(
                tool=tool,
                template_snapshot=template_snapshot,
                allowed_tool_names=allowed_tool_names,
                runtime_context=runtime_context,
            ):
                selected_tools.append(tool)
        selected_tools = self._apply_delegation_tool_exposure(
            tools=selected_tools,
            runtime_context=runtime_context,
            delegation_policy=template_snapshot.delegation_policy,
        )

        sys_prompt, startup_trace_data = self._build_agent_system_prompt(
            base_system_prompt,
            [skill.model_dump(mode="python") for skill in template_snapshot.skills],
            is_main_agent=runtime_context.is_main_agent,
            delegation_policy=template_snapshot.delegation_policy,
            account_id=runtime_context.account_id,
            workspace_dir=workspace_dir,
            session_id=session_id,
            retrieval_query=prompt_retrieval_query,
            exclude_run_id=prompt_retrieval_exclude_run_id,
            project_context_hints=runtime_context.project_context_hints,
        )

        resolved_llm = self._llm_client_resolver(runtime_context, template_snapshot)
        llm_client = resolved_llm.client
        llm_fingerprint = resolved_llm.fingerprint
        startup_trace_data["llm"] = resolved_llm.to_trace_payload()
        runtime_context.prompt_trace_data = dict(startup_trace_data)

        agent_holder: dict[str, Agent | None] = {"agent": None}
        if is_main_agent:

            def sub_agent_factory(persona: str, sub_max_steps: int) -> Agent:
                parent_agent = agent_holder["agent"]
                parent_context = (
                    parent_agent.runtime_context
                    if parent_agent is not None and parent_agent.runtime_context is not None
                    else runtime_context
                )
                parent_hooks = (
                    parent_agent.runtime_hooks
                    if parent_agent is not None and parent_agent.runtime_hooks is not None
                    else runtime_hooks
                )
                child_name = self._next_sub_agent_name(session_id)
                return self.build_agent(
                    workspace_dir=workspace_dir,
                    session_id=session_id,
                    template_snapshot=template_snapshot,
                    runtime_context=parent_context.create_child(child_name),
                    runtime_hooks=parent_hooks,
                    is_main_agent=False,
                    custom_prompt=persona,
                    max_steps=sub_max_steps,
                    agent_name=child_name,
                )

            def delegate_executor(persona: str, task: str, sub_max_steps: int):
                parent_agent = agent_holder["agent"]
                parent_context = (
                    parent_agent.runtime_context
                    if parent_agent is not None and parent_agent.runtime_context is not None
                    else runtime_context
                )
                parent_hooks = (
                    parent_agent.runtime_hooks
                    if parent_agent is not None and parent_agent.runtime_hooks is not None
                    else runtime_hooks
                )
                child_name = self._next_sub_agent_name(session_id)
                return self._delegate_executor(
                    parent_context=parent_context,
                    parent_hooks=parent_hooks,
                    template_snapshot=template_snapshot,
                    workspace_dir=workspace_dir,
                    agent_name=child_name,
                    persona=persona,
                    task=task,
                    max_steps=sub_max_steps,
                )

            parallel_limit = max(1, int(self._config.agent.parallel_delegate_limit))
            selected_tools.append(
                DelegateTool(
                    agent_factory=sub_agent_factory,
                    delegate_executor=(
                        delegate_executor
                        if self._delegate_executor is not None and runtime_context.run_id is not None
                        else None
                    ),
                )
            )
            selected_tools.append(
                DelegateBatchTool(
                    agent_factory=sub_agent_factory,
                    max_parallel=parallel_limit,
                    delegate_executor=(
                        delegate_executor
                        if self._delegate_executor is not None and runtime_context.run_id is not None
                        else None
                    ),
                )
            )

        agent = Agent(
            llm_client=llm_client,
            system_prompt=sys_prompt,
            tools=selected_tools,
            max_steps=max_steps or self._config.agent.max_steps,
            workspace_dir=str(workspace_dir),
            config=self._config,
        )
        agent.runtime_prompt_seed = base_system_prompt
        agent.llm_fingerprint = llm_fingerprint
        agent.bind_runtime(
            runtime_context=runtime_context,
            runtime_hooks=runtime_hooks,
            template_snapshot=template_snapshot,
        )
        agent.set_startup_trace_data(startup_trace_data)
        if seed_messages is not None:
            agent.messages = list(seed_messages)
            agent.set_system_prompt(sys_prompt)
        agent_holder["agent"] = agent
        return agent

    def _build_workspace_tools(
        self,
        workspace_dir: Path,
        session_id: str,
        agent_name: str,
        runtime_context: AgentRuntimeContext | None = None,
    ) -> list[Tool]:
        """Build tools that depend on the concrete workspace location."""
        tools: list[Tool] = []
        shared_context_file = self._shared_context_path_resolver(workspace_dir, session_id)

        workspace_dir.mkdir(parents=True, exist_ok=True)

        if self._config.tools.enable_bash:
            tools.append(BashTool(workspace_dir=str(workspace_dir)))

        if self._config.tools.enable_file_tools:
            tools.extend(
                [
                    ReadTool(workspace_dir=str(workspace_dir)),
                    WriteTool(workspace_dir=str(workspace_dir)),
                    EditTool(workspace_dir=str(workspace_dir)),
                ]
            )

        tools.extend(
            [
                ShareContextTool(
                    shared_file=str(shared_context_file),
                    agent_name=agent_name,
                    db_path=str(self._session_db_path),
                    session_id=session_id,
                    account_id=(
                        runtime_context.account_id
                        if runtime_context is not None
                        else None
                    ),
                    run_id=runtime_context.run_id if runtime_context is not None else None,
                    parent_run_id=(
                        runtime_context.parent_run_id
                        if runtime_context is not None
                        else None
                    ),
                    root_run_id=(
                        runtime_context.root_run_id
                        if runtime_context is not None
                        else None
                    ),
                ),
                ReadSharedContextTool(
                    shared_file=str(shared_context_file),
                    root_run_id=(
                        runtime_context.root_run_id
                        if runtime_context is not None
                        else None
                    ),
                ),
            ]
        )

        if self._config.tools.enable_note:
            note_kwargs = {
                "memory_file": str(workspace_dir / ".agent_memory.json"),
                "user_id": runtime_context.account_id if runtime_context is not None else None,
                "memory_provider": self._memory_provider,
                "session_id": session_id,
                "run_id": (
                    runtime_context.run_id
                    if runtime_context is not None
                    else None
                ),
            }
            tools.extend(
                [
                    SessionNoteTool(**note_kwargs),
                    RecallNoteTool(**note_kwargs),
                    SearchMemoryTool(**note_kwargs),
                ]
            )

        if self._channel_file_tool_enabled(runtime_context):
            tools.append(
                SendChannelFileTool(
                    workspace_dir=str(workspace_dir),
                    runtime_context=runtime_context,
                    sender=self._channel_file_sender,
                )
            )

        if self._session_retrieval_enabled():
            tools.append(
                SearchSessionHistoryTool(
                    db_path=str(self._session_db_path),
                    account_id=(
                        runtime_context.account_id
                        if runtime_context is not None
                        else None
                    ),
                    session_id=session_id,
                )
            )

        return tools

    @staticmethod
    def _tool_is_enabled_for_template(
        *,
        tool: Tool,
        template_snapshot: AgentTemplateSnapshot,
        allowed_tool_names: set[str],
        runtime_context: AgentRuntimeContext | None = None,
    ) -> bool:
        """Whether the template explicitly or implicitly authorizes one runtime tool."""
        identifiers = {tool.__class__.__name__, tool.name}
        if identifiers & allowed_tool_names:
            return True

        if (
            tool.__class__.__name__
            in {"RecallNoteTool", "SearchMemoryTool", "SearchSessionHistoryTool"}
            and "SessionNoteTool" in allowed_tool_names
        ):
            return True

        if tool.__class__.__name__ == "GetSkillTool" and template_snapshot.skills:
            return True

        if tool.__class__.__name__ == "MCPTool" and template_snapshot.mcp_configs:
            return True

        if (
            tool.__class__.__name__ == "SendChannelFileTool"
            and runtime_context is not None
            and runtime_context.channel_kind == "feishu"
            and runtime_context.integration_id
            and runtime_context.provider_chat_id
        ):
            return True

        return False

    def _channel_file_tool_enabled(
        self,
        runtime_context: AgentRuntimeContext | None,
    ) -> bool:
        return bool(
            self._channel_file_sender is not None
            and runtime_context is not None
            and str(runtime_context.channel_kind or "").strip().lower() == "feishu"
            and str(runtime_context.integration_id or "").strip()
            and str(runtime_context.provider_chat_id or "").strip()
        )

    @staticmethod
    def _apply_delegation_tool_exposure(
        *,
        tools: list[Tool],
        runtime_context: AgentRuntimeContext,
        delegation_policy,
    ) -> list[Tool]:
        """根据主/子 agent 委派策略收口可见工具集合。"""
        if not runtime_context.is_main_agent:
            return list(tools)

        policy = delegation_policy
        mode = getattr(policy, "mode", "hybrid")
        allow_read_tools = bool(getattr(policy, "allow_main_agent_read_tools", True))
        if mode != "supervisor_only" and allow_read_tools:
            return list(tools)

        filtered: list[Tool] = []
        always_allowed_names = {
            "delegate_task",
            "delegate_tasks",
            "share_context",
            "read_shared_context",
        }
        read_only_tool_classes = {
            "ReadTool",
            "RecallNoteTool",
            "SearchMemoryTool",
            "SearchSessionHistoryTool",
            "BashOutputTool",
        }

        for tool in tools:
            if mode != "supervisor_only":
                if not allow_read_tools and (
                    tool.name == "read_file" or tool.__class__.__name__ in read_only_tool_classes
                ):
                    continue
                filtered.append(tool)
                continue
            if tool.name in always_allowed_names:
                filtered.append(tool)
                continue
            if allow_read_tools and (
                tool.name == "read_file" or tool.__class__.__name__ in read_only_tool_classes
            ):
                filtered.append(tool)

        return filtered

    def _build_agent_system_prompt(
        self,
        base_prompt: str,
        skills: list[dict[str, str]] | None,
        *,
        is_main_agent: bool,
        delegation_policy,
        account_id: str | None,
        workspace_dir: Path,
        session_id: str | None = None,
        retrieval_query: str | None = None,
        exclude_run_id: str | None = None,
        project_context_hints: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Compose the final system prompt from prompt text, memory, and skills."""
        sections = [str(base_prompt or "").rstrip()]
        role_policy_prompt, role_policy_trace = self._build_role_policy_prompt(
            is_main_agent=is_main_agent,
            delegation_policy=delegation_policy,
        )
        if role_policy_prompt:
            sections.append(role_policy_prompt)
        memory_prompt, memory_trace, memory_observability = self._build_compact_memory_prompt(
            account_id,
            workspace_dir=workspace_dir,
            session_id=session_id,
            retrieval_query=retrieval_query,
            exclude_run_id=exclude_run_id,
            project_context_hints=project_context_hints,
        )
        if memory_prompt:
            sections.append(memory_prompt)
        skills_prompt = build_skills_description_prompt(skills)
        if skills_prompt:
            sections.append(skills_prompt)
        return "\n\n".join(section for section in sections if section), {
            "prompt": {
                "agent_role": "main" if is_main_agent else "worker",
                "role_policy": role_policy_trace,
                "memory_sections": memory_trace,
                "memory_section_count": len(memory_trace),
                "memory_prompt_char_count": len(memory_prompt),
                **memory_observability,
            }
        }

    def _build_role_policy_prompt(
        self,
        *,
        is_main_agent: bool,
        delegation_policy,
    ) -> tuple[str, dict[str, Any]]:
        """Build the role-specific policy block appended after the base prompt."""
        mode = str(getattr(delegation_policy, "mode", "hybrid") or "hybrid")
        if is_main_agent:
            policy_lines = [
                "## Active Delegation Policy",
                f"- mode: {mode}",
                (
                    f"- allow_main_agent_read_tools: "
                    f"{'true' if getattr(delegation_policy, 'allow_main_agent_read_tools', True) else 'false'}"
                ),
                (
                    f"- require_delegate_for_write_actions: "
                    f"{'true' if getattr(delegation_policy, 'require_delegate_for_write_actions', False) else 'false'}"
                ),
                (
                    f"- require_delegate_for_shell: "
                    f"{'true' if getattr(delegation_policy, 'require_delegate_for_shell', False) else 'false'}"
                ),
                (
                    f"- require_delegate_for_stateful_mcp: "
                    f"{'true' if getattr(delegation_policy, 'require_delegate_for_stateful_mcp', False) else 'false'}"
                ),
                (
                    f"- verify_worker_output: "
                    f"{'true' if getattr(delegation_policy, 'verify_worker_output', True) else 'false'}"
                ),
                (
                    f"- prefer_batch_delegate: "
                    f"{'true' if getattr(delegation_policy, 'prefer_batch_delegate', True) else 'false'}"
                ),
            ]
            if mode == "hybrid":
                policy_lines.append(
                    "- Working rule: direct execution is allowed, but delegation is still preferred for multi-step or isolated execution work."
                )
            elif mode == "prefer_delegate":
                policy_lines.append(
                    "- Working rule: default to delegate executable work to workers; only keep planning, light inspection, and final acceptance in the main agent when practical."
                )
            else:
                policy_lines.append(
                    "- Working rule: stay in supervisor_only mode. Do not directly write files, run shell commands, or execute MCP actions with side effects; delegate them instead."
                )
            if getattr(delegation_policy, "verify_worker_output", True):
                policy_lines.extend(
                    [
                        "- Acceptance rule: after each delegate finishes, explicitly judge whether the original request is satisfied, whether more validation is still needed, and whether another worker must be delegated.",
                        "- Do not end the run immediately after a worker result when blockers, remaining risks, or missing verification still exist.",
                    ]
                )
            prompt = "\n\n".join(
                section for section in (self._supervisor_policy_prompt, "\n".join(policy_lines)) if section
            )
            return prompt, {
                "role": "main",
                "mode": mode,
                "allow_main_agent_read_tools": bool(
                    getattr(delegation_policy, "allow_main_agent_read_tools", True)
                ),
                "verify_worker_output": bool(
                    getattr(delegation_policy, "verify_worker_output", True)
                ),
            }

        prompt = self._worker_policy_prompt
        return prompt, {
            "role": "worker",
            "mode": mode,
        }

    @classmethod
    def _truncate_text(cls, value: str, limit: int) -> str:
        normalized = " ".join(str(value or "").split())
        if limit <= 0:
            return ""
        if len(normalized) <= limit:
            return normalized
        return normalized[: max(0, limit - 1)].rstrip() + "…"

    def _get_session_store(self) -> SessionStore:
        if self._session_store is None:
            self._session_store = SessionStore(self._session_db_path)
        return self._session_store

    @staticmethod
    def _stringify_profile_value(value: Any) -> str:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return str(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @classmethod
    def _cap_section(cls, lines: list[str], limit: int) -> str:
        if not lines:
            return ""
        body = "\n".join(line for line in lines if line)
        if len(body) <= limit:
            return body

        capped: list[str] = []
        for line in lines:
            candidate = "\n".join([*capped, line]) if capped else line
            if len(candidate) > limit:
                break
            capped.append(line)

        ellipsis = "- ..."
        if not capped:
            return cls._truncate_text(body, limit)
        candidate_with_ellipsis = "\n".join([*capped, ellipsis])
        if len(candidate_with_ellipsis) <= limit:
            capped.append(ellipsis)
            return "\n".join(capped)
        return "\n".join(capped)

    @classmethod
    def _looks_secret_like_project_context(cls, content: str) -> bool:
        normalized = str(content or "").strip()
        if not normalized:
            return False
        for pattern in cls._PROJECT_CONTEXT_SECRET_PATTERNS:
            if pattern.search(normalized):
                return True
        return False

    @staticmethod
    def _render_project_context_path(path: Path, workspace_dir: Path) -> str:
        try:
            return Path(os.path.relpath(path, workspace_dir)).as_posix()
        except Exception:
            return path.name

    def _iter_project_context_search_dirs(
        self,
        workspace_dir: Path,
        project_context_hints: list[str] | None = None,
    ) -> list[Path]:
        search_dirs: list[Path] = []
        seen_dirs: set[Path] = set()
        workspace_root = workspace_dir.resolve()
        max_depth = self._prompt_memory_config.project_context_parent_depth

        allowed_ancestors = {workspace_root}
        ancestor = workspace_root
        for _ in range(max_depth):
            if ancestor.parent == ancestor:
                break
            ancestor = ancestor.parent
            allowed_ancestors.add(ancestor)

        def append_dir(candidate_dir: Path) -> None:
            if candidate_dir in seen_dirs:
                return
            seen_dirs.add(candidate_dir)
            search_dirs.append(candidate_dir)

        for raw_hint in project_context_hints or []:
            normalized_hint = str(raw_hint or "").strip()
            if not normalized_hint:
                continue
            try:
                resolved_hint = Path(normalized_hint).resolve()
            except Exception:
                continue
            current_dir = resolved_hint if resolved_hint.is_dir() else resolved_hint.parent
            while True:
                if current_dir == workspace_root or current_dir in allowed_ancestors:
                    append_dir(current_dir)
                elif self._is_relative_to(current_dir, workspace_root):
                    append_dir(current_dir)
                else:
                    break
                if current_dir.parent == current_dir:
                    break
                current_dir = current_dir.parent

        current_dir = workspace_root
        for _ in range(max_depth + 1):
            append_dir(current_dir)
            if current_dir.parent == current_dir:
                break
            current_dir = current_dir.parent

        return search_dirs

    @staticmethod
    def _is_relative_to(path: Path, parent: Path) -> bool:
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False

    def _discover_project_context_paths(
        self,
        workspace_dir: Path,
        project_context_hints: list[str] | None = None,
    ) -> list[Path]:
        discovered: list[Path] = []
        seen: set[Path] = set()
        max_files = self._prompt_memory_config.project_context_max_files

        for current_dir in self._iter_project_context_search_dirs(
            workspace_dir,
            project_context_hints=project_context_hints,
        ):
            for filename in self._PROJECT_CONTEXT_FILENAMES:
                candidate = current_dir / filename
                if not candidate.exists() or not candidate.is_file():
                    continue
                try:
                    resolved_candidate = candidate.resolve()
                except Exception:
                    continue
                if resolved_candidate in seen:
                    continue
                seen.add(resolved_candidate)
                discovered.append(resolved_candidate)
                if len(discovered) >= max_files:
                    return discovered

        return discovered

    def _load_project_context_entries(
        self,
        workspace_dir: Path,
        project_context_hints: list[str] | None = None,
    ) -> list[ProjectContextEntry]:
        entries: list[ProjectContextEntry] = []
        byte_limit = self._prompt_memory_config.project_context_file_byte_limit
        char_limit = self._prompt_memory_config.project_context_file_char_limit

        for path in self._discover_project_context_paths(
            workspace_dir,
            project_context_hints=project_context_hints,
        ):
            try:
                if path.stat().st_size > byte_limit:
                    continue
            except OSError:
                continue

            try:
                content = path.read_text(encoding=self._UTF8)
            except Exception:
                continue

            normalized = content.replace("\ufeff", "").strip()
            if not normalized or "\x00" in normalized:
                continue
            if self._looks_secret_like_project_context(normalized):
                continue

            entries.append(
                ProjectContextEntry(
                    path=path,
                    display_path=self._render_project_context_path(path, workspace_dir),
                    content=self._truncate_text(normalized, char_limit),
                )
            )

        return entries

    @staticmethod
    def _sort_memory_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Order entries by recent update first so injected memory prefers fresh facts."""
        return sorted(
            entries,
            key=lambda item: (
                str(item.get("updated_at") or item.get("timestamp") or ""),
                str(item.get("id") or item.get("content") or ""),
            ),
            reverse=True,
        )

    @staticmethod
    def _sort_history_entries(entries: list[Any]) -> list[Any]:
        """Prefer richer historical summaries over bare goals when injecting retrieval context."""
        priority = {
            "run_completion": 0,
            "run_failure": 1,
            "shared_context": 2,
            "session_message": 3,
            "run_goal": 4,
        }
        return sorted(
            entries,
            key=lambda item: priority.get(str(getattr(item, "source_type", "") or ""), 99),
        )

    def _build_profile_section(self, account_id: str) -> PromptMemorySection | None:
        """Build the compact user profile prompt block."""
        if not self._prompt_memory_enabled() or not self._memory_provider.inject_memories:
            return None
        profile = self._memory_provider.get_user_profile(account_id)
        if not profile:
            return None

        lines = ["## User Profile Summary"]
        summary = self._truncate_text(
            str(profile.get("summary") or "").strip(),
            self._prompt_memory_config.profile_summary_char_limit,
        )
        if summary:
            lines.append(f"- summary: {summary}")
        payload = profile.get("profile") or {}
        item_count = 0
        if isinstance(payload, dict):
            for key in sorted(payload):
                rendered_value = self._stringify_profile_value(payload[key])
                lines.append(
                    f"- {key}: "
                    f"{self._truncate_text(rendered_value, self._prompt_memory_config.profile_field_value_char_limit)}"
                )
                item_count += 1

        body = self._cap_section(lines, self._prompt_memory_config.profile_char_limit)
        if not body:
            return None
        return PromptMemorySection(
            key="user_profile",
            title="User Profile Summary",
            body=body,
            source_kind="user_profile",
            char_limit=self._prompt_memory_config.profile_char_limit,
            item_count=item_count + (1 if summary else 0),
            sources=[f"user_profile:{account_id[:8]}"],
            metadata={
                "fields": sorted(payload) if isinstance(payload, dict) else [],
                "summary_included": bool(summary),
            },
        )

    def _build_preference_section(self, account_id: str) -> PromptMemorySection | None:
        """Build the stable working preferences prompt block."""
        if not self._prompt_memory_enabled() or not self._memory_provider.inject_memories:
            return None
        entries = self._memory_provider.list_memory_entries(
            account_id,
            memory_types=list(self._STABLE_PREFERENCE_TYPES),
            include_superseded=False,
            limit=self._prompt_memory_config.preference_entry_limit,
        )
        if not entries:
            return None

        lines = ["## Stable Working Preferences"]
        sources: list[str] = []
        item_count = 0
        for entry in self._sort_memory_entries(entries):
            memory_type = str(entry.get("memory_type") or "memory").strip()
            primary_text = str(entry.get("summary") or entry.get("content") or "").strip()
            if not primary_text:
                continue
            lines.append(
                f"- [{memory_type}] "
                f"{self._truncate_text(primary_text, self._prompt_memory_config.preference_entry_char_limit)}"
            )
            entry_id = str(entry.get("id") or "").strip()
            sources.append(f"{memory_type}:{entry_id[:8] or 'memory'}")
            item_count += 1

        body = self._cap_section(lines, self._prompt_memory_config.preference_char_limit)
        if not body or item_count == 0:
            return None
        return PromptMemorySection(
            key="stable_preferences",
            title="Stable Working Preferences",
            body=body,
            source_kind="user_memory",
            char_limit=self._prompt_memory_config.preference_char_limit,
            item_count=item_count,
            sources=sources,
        )

    def _build_retrieved_context_section(
        self,
        *,
        account_id: str,
        session_id: str | None,
        retrieval_query: str | None,
        exclude_run_id: str | None,
    ) -> PromptMemorySection | None:
        """Build a compact retrieval summary from prior history and task facts."""
        _ = session_id
        normalized_query = " ".join(str(retrieval_query or "").split())
        if (
            not normalized_query
            or not self._prompt_memory_enabled()
            or not self._session_retrieval_enabled()
            or not self._memory_provider.inject_memories
        ):
            return None

        entry_limit = self._prompt_memory_config.retrieved_context_entry_limit
        entry_char_limit = self._prompt_memory_config.retrieved_context_entry_char_limit
        lines = ["## Recent Retrieved Context"]
        sources: list[str] = []
        history_sources: list[str] = []
        memory_sources: list[str] = []
        item_count = 0

        history_entries = self._sort_history_entries(
            self._get_session_store().search_history_records(
                normalized_query,
                account_id=account_id,
                exclude_run_id=exclude_run_id,
                limit=max(entry_limit * 3, entry_limit),
            )
        )
        for entry in history_entries:
            primary_text = str(entry.snippet or entry.content or "").strip()
            if not primary_text:
                continue
            lines.append(
                f"- [history/{entry.source_type}] {str(entry.created_at or '')[:10]}: "
                f"{self._truncate_text(primary_text, entry_char_limit)}"
            )
            source_key = f"history:{entry.source_key}"
            sources.append(source_key)
            history_sources.append(source_key)
            item_count += 1
            if item_count >= entry_limit:
                break

        if item_count < entry_limit:
            memory_entries = self._memory_provider.search_memory_entries(
                account_id,
                query=normalized_query,
                memory_types=["goal", "project_fact", "workflow_fact"],
                include_superseded=False,
                limit=entry_limit,
            )
            for entry in self._sort_memory_entries(memory_entries):
                if item_count >= entry_limit:
                    break
                primary_text = str(entry.get("summary") or entry.get("content") or "").strip()
                if not primary_text:
                    continue
                memory_type = str(entry.get("memory_type") or "memory").strip()
                updated_at = str(entry.get("updated_at") or entry.get("created_at") or "")[:10]
                lines.append(
                    f"- [memory/{memory_type}] {updated_at}: "
                    f"{self._truncate_text(primary_text, entry_char_limit)}"
                )
                entry_id = str(entry.get("id") or "").strip()
                source_key = f"memory:{memory_type}:{entry_id[:8] or 'entry'}"
                sources.append(source_key)
                memory_sources.append(source_key)
                item_count += 1

        body = self._cap_section(lines, self._prompt_memory_config.retrieved_context_char_limit)
        if not body or item_count == 0:
            return None
        return PromptMemorySection(
            key="retrieved_context",
            title="Recent Retrieved Context",
            body=body,
            source_kind="retrieval",
            char_limit=self._prompt_memory_config.retrieved_context_char_limit,
            item_count=item_count,
            sources=sources,
            metadata={
                "query": normalized_query,
                "history_hit_count": len(history_sources),
                "memory_hit_count": len(memory_sources),
                "history_sources": history_sources,
                "memory_sources": memory_sources,
                "used_sources": sources,
            },
        )

    def _build_project_context_section(
        self,
        workspace_dir: Path,
        project_context_hints: list[str] | None = None,
    ) -> PromptMemorySection | None:
        """Build the compact project-context prompt block from discovered workspace files."""
        entries = self._load_project_context_entries(
            workspace_dir,
            project_context_hints=project_context_hints,
        )
        if not entries:
            return None

        lines = ["## Project Context"]
        sources: list[str] = []
        for entry in entries:
            lines.append(f"- [{entry.path.name}] {entry.display_path}")
            for snippet_line in entry.content.splitlines():
                normalized_line = snippet_line.rstrip()
                if not normalized_line:
                    continue
                lines.append(f"  {normalized_line}")
            sources.append(entry.path.as_posix())

        body = self._cap_section(lines, self._prompt_memory_config.project_context_char_limit)
        if not body:
            return None
        return PromptMemorySection(
            key="project_context",
            title="Project Context",
            body=body,
            source_kind="project_context",
            char_limit=self._prompt_memory_config.project_context_char_limit,
            item_count=len(entries),
            sources=sources,
        )

    def _load_agent_memory_entries(self, workspace_dir: Path) -> list[dict[str, Any]]:
        """Read workspace-local agent memory using UTF-8 for prompt injection."""
        memory_file = workspace_dir / ".agent_memory.json"
        if not memory_file.exists():
            return []
        try:
            payload = json.loads(memory_file.read_text(encoding=self._UTF8))
        except Exception:
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _build_agent_memory_section(self, workspace_dir: Path) -> PromptMemorySection | None:
        """Build the compact workspace-local agent memory prompt block."""
        if not self._prompt_memory_enabled():
            return None
        entries = self._load_agent_memory_entries(workspace_dir)
        if not entries:
            return None

        lines = ["## Relevant Agent Memory"]
        sources: list[str] = []
        item_count = 0
        limited_entries = self._sort_memory_entries(entries)[
            : self._prompt_memory_config.agent_memory_entry_limit
        ]
        for entry in limited_entries:
            memory_label = str(
                entry.get("memory_type") or entry.get("category") or "agent_memory"
            ).strip()
            primary_text = str(entry.get("summary") or entry.get("content") or "").strip()
            if not primary_text:
                continue
            lines.append(
                f"- [{memory_label}] "
                f"{self._truncate_text(primary_text, self._prompt_memory_config.agent_memory_entry_char_limit)}"
            )
            category = str(entry.get("category") or entry.get("memory_type") or "agent_memory").strip()
            updated_at = str(entry.get("timestamp") or entry.get("updated_at") or "").strip()
            sources.append(f"{category}:{updated_at[:10] or 'local'}")
            item_count += 1

        body = self._cap_section(lines, self._prompt_memory_config.agent_memory_char_limit)
        if not body or item_count == 0:
            return None
        return PromptMemorySection(
            key="agent_memory",
            title="Relevant Agent Memory",
            body=body,
            source_kind="agent_memory",
            char_limit=self._prompt_memory_config.agent_memory_char_limit,
            item_count=item_count,
            sources=sources,
        )

    def _build_compact_memory_prompt(
        self,
        account_id: str | None,
        *,
        workspace_dir: Path,
        session_id: str | None = None,
        retrieval_query: str | None = None,
        exclude_run_id: str | None = None,
        project_context_hints: list[str] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        """Build compact memory sections for runtime prompt injection."""
        normalized_account_id = str(account_id or "").strip()
        normalized_retrieval_query = " ".join(str(retrieval_query or "").split())
        prompt_sections: list[PromptMemorySection] = []
        if normalized_account_id:
            profile_section = self._build_profile_section(normalized_account_id)
            if profile_section is not None:
                prompt_sections.append(profile_section)
            preference_section = self._build_preference_section(normalized_account_id)
            if preference_section is not None:
                prompt_sections.append(preference_section)
            retrieved_context_section = self._build_retrieved_context_section(
                account_id=normalized_account_id,
                session_id=session_id,
                retrieval_query=normalized_retrieval_query,
                exclude_run_id=exclude_run_id,
            )
            if retrieved_context_section is not None:
                prompt_sections.append(retrieved_context_section)

        project_context_section = self._build_project_context_section(
            workspace_dir,
            project_context_hints=project_context_hints,
        )
        if project_context_section is not None:
            prompt_sections.append(project_context_section)

        agent_memory_section = self._build_agent_memory_section(workspace_dir)
        if agent_memory_section is not None:
            prompt_sections.append(agent_memory_section)

        observability: dict[str, Any] = {
            "memory_section_keys": [section.key for section in prompt_sections],
        }
        profile_section = next(
            (section for section in prompt_sections if section.key == "user_profile"),
            None,
        )
        if profile_section is not None:
            observability["profile_fields_used"] = list(
                profile_section.metadata.get("fields", [])
            )
        else:
            observability["profile_fields_used"] = []

        if normalized_retrieval_query:
            retrieval_section = next(
                (section for section in prompt_sections if section.key == "retrieved_context"),
                None,
            )
            if retrieval_section is not None:
                observability["retrieval"] = dict(retrieval_section.metadata)
            else:
                observability["retrieval"] = {
                    "query": normalized_retrieval_query,
                    "history_hit_count": 0,
                    "memory_hit_count": 0,
                    "history_sources": [],
                    "memory_sources": [],
                    "used_sources": [],
                }

        if not prompt_sections:
            return "", [], observability
        return (
            "\n\n".join(section.body for section in prompt_sections),
            [section.to_trace_payload() for section in prompt_sections],
            observability,
        )

    def refresh_agent_prompt_from_tool_payload(
        self,
        *,
        agent: Agent,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Refresh the agent prompt when a tool touches deeper project paths."""
        context = agent.runtime_context
        template_snapshot = context.template_snapshot if context is not None else None
        if context is None or template_snapshot is None:
            return None

        raw_touched_paths = payload.get("touched_paths")
        if not isinstance(raw_touched_paths, list):
            return None

        known_hints = set(context.project_context_hints)
        updated = False
        for raw_path in raw_touched_paths:
            normalized_path = str(raw_path or "").strip()
            if not normalized_path:
                continue
            try:
                resolved_path = Path(normalized_path).resolve().as_posix()
            except Exception:
                continue
            if resolved_path in known_hints:
                continue
            context.project_context_hints.append(resolved_path)
            known_hints.add(resolved_path)
            updated = True

        if not updated:
            return None

        workspace_dir = Path(agent.workspace_dir).resolve()
        base_system_prompt = (
            context.base_system_prompt
            or getattr(agent, "runtime_prompt_seed", None)
            or template_snapshot.system_prompt
            or self._default_system_prompt
        )
        sys_prompt, startup_trace_data = self._build_agent_system_prompt(
            base_system_prompt,
            [skill.model_dump(mode="python") for skill in template_snapshot.skills],
            is_main_agent=context.is_main_agent,
            delegation_policy=template_snapshot.delegation_policy,
            account_id=context.account_id,
            workspace_dir=workspace_dir,
            session_id=context.session_id,
            retrieval_query=context.prompt_retrieval_query,
            exclude_run_id=context.prompt_retrieval_exclude_run_id,
            project_context_hints=context.project_context_hints,
        )
        context.prompt_trace_data = dict(startup_trace_data)
        agent.runtime_prompt_seed = base_system_prompt
        agent.set_system_prompt(sys_prompt)
        agent.set_startup_trace_data(startup_trace_data)
        return startup_trace_data

    @staticmethod
    def _can_reuse_existing_agent(
        agent: Agent,
        template_snapshot: AgentTemplateSnapshot,
    ) -> bool:
        """Whether an existing agent can safely execute one run snapshot."""
        if agent.manual_runtime_override:
            return True

        current_snapshot = agent.template_snapshot
        if current_snapshot is None:
            return False

        return (
            current_snapshot.template_id == template_snapshot.template_id
            and current_snapshot.template_version == template_snapshot.template_version
        )
