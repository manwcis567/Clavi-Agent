"""Session Manager - manages multiple Agent sessions for Web/API access."""

import asyncio
import json
import logging
import shutil
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Any, AsyncGenerator, Literal

from .agent import Agent
from .agent_runtime import (
    AgentRuntimeContext,
    AgentRuntimeFactory,
    AgentRuntimeHooks,
    ResolvedLLMRuntime,
)
from .account_constants import ROOT_ACCOUNT_ID
from .account_store import AccountStore
from .agent_store import AgentStore
from .agent_template_models import AgentTemplateSnapshot, SessionPolicyOverride
from .approval_store import ApprovalStore
from .config import Config
from .integrations.dispatcher import IntegrationReplyDispatcher
from .integrations.models import MsgContextEnvelope
from .learned_workflow_store import LearnedWorkflowStore
from .llm_routing_models import LLMProfile, LLMProfileOverride, merge_llm_profile
from .llm import LLMClient
from .logger import AgentLogger
from .memory_provider import MemoryProvider, build_memory_provider
from .retry import RetryConfig as RetryConfigBase
from .run_manager import RunManager
from .run_models import ArtifactRecord, RunRecord
from .run_store import RunStore
from .schema import (
    FunctionCall,
    LLMProvider,
    Message,
    TextContentBlock,
    ToolCall,
    UploadedFileContentBlock,
    message_content_summary,
)
from .session_models import SessionRuntimeRegistry
from .session_store import DEFAULT_SESSION_TITLE, SessionStore
from .sqlite_schema import utc_now_iso
from .trace_analysis import (
    build_run_metrics,
    build_run_metrics_export,
    build_run_location_payload,
    build_trace_replay,
    build_run_tree,
    build_tool_call_drilldown,
    build_trace_export,
    build_trace_timeline,
    collect_run_family,
)
from .trace_store import TraceStore
from .tools.base import Tool
from .tools.bash_tool import BashKillTool, BashOutputTool
from .tool_execution import UploadedFileTarget
from .upload_models import (
    ALLOWED_UPLOAD_EXTENSIONS,
    BLOCKED_UPLOAD_EXTENSIONS,
    MAX_UPLOAD_SIZE_BYTES,
    UploadCreatePayload,
    UploadRecord,
    compute_upload_checksum,
    resolve_upload_mime_type,
    sanitize_upload_filename,
    upload_extension,
)
from .upload_store import UploadStore
from .skill_improvement_store import SkillImprovementStore
from .skill_improvement_utils import extract_skill_version
logger = logging.getLogger(__name__)
_UNSET = object()


class SessionManager:
    """Manages multiple concurrent Agent sessions.

    Each session has its own Agent instance with independent message history
    and tool state. Both CLI and Web API can use this as the unified backend.
    """

    def __init__(self, config: Config | None = None):
        self._config: Config | None = config
        self._config_path: Path | None = None
        self._shared_tools: list[Tool] | None = None
        self._system_prompt: str | None = None
        self._llm_client: LLMClient | None = None
        self._llm_client_cache: dict[str, LLMClient] = {}
        self._account_store: AccountStore | None = None
        self._session_store: SessionStore | None = None
        self._agent_store: AgentStore | None = None
        self._run_store: RunStore | None = None
        self._trace_store: TraceStore | None = None
        self._approval_store: ApprovalStore | None = None
        self._upload_store: UploadStore | None = None
        self._memory_provider: MemoryProvider | None = None
        self._learned_workflow_store: LearnedWorkflowStore | None = None
        self._skill_improvement_store: SkillImprovementStore | None = None
        self._run_manager: RunManager | None = None
        self._runtime_factory: AgentRuntimeFactory | None = None
        self._integration_reply_dispatcher: IntegrationReplyDispatcher | None = None
        self._initialized = False
        self._runtime_ready = False
        self._runtime_registry = SessionRuntimeRegistry()

    async def initialize(self):
        """Load config, shared tools, system prompt, and LLM client once."""
        if self._initialized:
            return

        # 1. Load config
        if self._config is None:
            config_path = self._config_path or Config.get_default_config_path()
            if not config_path.exists():
                config_path = Config.ensure_bootstrap_config()
            self._config_path = config_path
            self._config = Config.from_yaml(config_path, require_api_key=False)

        config = self._config

        # 2. Load system prompt
        system_prompt_path = Config.find_config_file(config.agent.system_prompt_path)
        if system_prompt_path and system_prompt_path.exists():
            self._system_prompt = system_prompt_path.read_text(encoding="utf-8")
        else:
            self._system_prompt = (
                "You are Clavi Agent, an intelligent assistant powered by MiniMax M2 "
                "that can help users complete various tasks."
            )
        self._system_prompt = self._system_prompt.replace("{SKILLS_METADATA}", "")

        session_store_path = Path(config.agent.session_store_path)
        if not session_store_path.is_absolute():
            session_store_path = Path.cwd() / session_store_path
        self._session_store = SessionStore(session_store_path)
        self._run_store = RunStore(session_store_path)
        self._trace_store = TraceStore(session_store_path)
        self._approval_store = ApprovalStore(session_store_path)
        self._upload_store = UploadStore(session_store_path)
        self._learned_workflow_store = LearnedWorkflowStore(session_store_path)
        self._skill_improvement_store = SkillImprovementStore(session_store_path)

        agent_store_path = Path(config.agent.agent_store_path)
        if not agent_store_path.is_absolute():
            agent_store_path = Path.cwd() / agent_store_path
        self._account_store = AccountStore(
            agent_store_path,
            auto_seed_root=config.auth.auto_seed_root,
            root_username=config.auth.root_username,
            root_display_name=config.auth.root_display_name,
            root_password=config.auth.resolve_root_password(),
        )
        if self._account_store.bootstrap_root_password:
            logger.warning(
                "首次初始化 root 账号，已生成临时密码。username=%s password=%s",
                config.auth.root_username,
                self._account_store.bootstrap_root_password,
            )
        self._agent_store = AgentStore(agent_store_path)
        mcp_config_path = Config.find_config_file(config.tools.mcp_config_path)
        feature_flags = config.get_feature_flags()
        self._memory_provider = build_memory_provider(
            configured_provider=str(config.memory_provider.provider or "local").strip().lower() or "local",
            db_path=agent_store_path,
            inject_memories=config.memory_provider.inject_memories,
            expose_tools=config.memory_provider.expose_tools,
            sync_conversation_turns=config.memory_provider.sync_conversation_turns,
            enable_external_providers=feature_flags.get("enable_external_memory_providers", True),
            allow_fallback_to_local=config.memory_provider.allow_fallback_to_local,
            mcp_config_path=mcp_config_path,
            mcp_server_name=config.memory_provider.mcp_server_name,
        )
        self._agent_store.sync_system_agents(config.get_system_agents(available_skills=[]))
        self._seed_legacy_root_api_config_if_needed()

        # 3. Load shared tools (non-workspace-dependent)
        self._shared_tools = []

        if config.tools.enable_bash:
            self._shared_tools.extend([BashOutputTool(), BashKillTool()])

        if config.tools.enable_mcp:
            try:
                from .tools.mcp_loader import load_mcp_tools_async

                mcp_config_path = Config.find_config_file(config.tools.mcp_config_path)
                if mcp_config_path:
                    mcp_tools = await load_mcp_tools_async(str(mcp_config_path))
                    if mcp_tools:
                        self._shared_tools.extend(mcp_tools)
            except Exception:
                pass

        self._runtime_factory = AgentRuntimeFactory(
            config=config,
            llm_client_resolver=self._get_account_llm_runtime,
            agent_store=self._agent_store,
            shared_tools=self._shared_tools,
            default_system_prompt=self._system_prompt,
            next_sub_agent_name=self._next_sub_agent_name,
            shared_context_path_resolver=self._get_shared_context_path,
            agent_db_path=agent_store_path,
            memory_provider=self._memory_provider,
            delegate_executor=self._execute_delegate_child_run,
            channel_file_sender=self._send_runtime_channel_file,
        )
        self._integration_reply_dispatcher = IntegrationReplyDispatcher(self)
        self._run_manager = RunManager(
            run_store=self._run_store,
            trace_store=self._trace_store,
            approval_store=self._approval_store,
            upload_store=self._upload_store,
            learned_workflow_store=self._learned_workflow_store,
            skill_improvement_store=self._skill_improvement_store,
            session_store=self._session_store,
            agent_store=self._agent_store,
            runtime_registry=self._runtime_registry,
            load_agent=self._load_run_agent,
            sync_session_snapshot=self._sync_session_snapshot,
            sync_memory_provider_turn=self._sync_memory_provider_turn,
            terminal_run_handler=self._handle_terminal_run,
            max_concurrent_runs=config.agent.max_concurrent_runs,
            run_timeout_seconds=config.agent.run_timeout_seconds,
            enable_learned_workflow_generation=feature_flags.get(
                "enable_learned_workflow_generation",
                True,
            ),
        )
        self._run_manager.recover_pending_runs()
        try:
            self._llm_client = self._get_account_llm_runtime(
                AgentRuntimeContext(
                    session_id="bootstrap",
                    account_id=ROOT_ACCOUNT_ID,
                    is_main_agent=True,
                ),
                None,
            ).client
        except RuntimeError:
            self._llm_client = None

        self._runtime_ready = True
        self._initialized = True

    def _seed_legacy_root_api_config_if_needed(self) -> None:
        """Import the legacy file-based root API key into SQLite once when available."""
        if self._config is None or self._account_store is None:
            return
        if not self._config.has_valid_api_key():
            return
        if self._account_store.list_api_config_records(ROOT_ACCOUNT_ID):
            return
        self._account_store.upsert_api_config(
            ROOT_ACCOUNT_ID,
            name="Migrated Local Default",
            api_key=self._config.llm.api_key,
            provider=self._config.llm.provider,
            api_base=self._config.llm.api_base,
            model=self._config.llm.model,
            reasoning_enabled=self._config.llm.reasoning_enabled,
            llm_routing_policy={
                "planner_profile": self._profile_override_payload(self._config.llm.planner_profile),
                "worker_profile": self._profile_override_payload(self._config.llm.worker_profile),
            },
            activate=True,
        )

    def _build_llm_client(
        self,
        *,
        api_key: str,
        provider: str,
        api_base: str,
        model: str,
        reasoning_enabled: bool,
    ) -> LLMClient:
        """Create one LLM client from a resolved account credential set."""
        normalized_provider = str(provider or "anthropic").strip().lower() or "anthropic"
        resolved_provider = (
            LLMProvider.ANTHROPIC
            if normalized_provider == "anthropic"
            else LLMProvider.OPENAI
        )
        if self._config is None:
            raise RuntimeError("Config not initialized.")
        retry_config = RetryConfigBase(
            enabled=self._config.llm.retry.enabled,
            max_retries=self._config.llm.retry.max_retries,
            initial_delay=self._config.llm.retry.initial_delay,
            max_delay=self._config.llm.retry.max_delay,
            exponential_base=self._config.llm.retry.exponential_base,
            retryable_exceptions=(Exception,),
        )
        return LLMClient(
            api_key=api_key,
            provider=resolved_provider,
            api_base=api_base,
            model=model,
            reasoning_enabled=reasoning_enabled,
            retry_config=retry_config if self._config.llm.retry.enabled else None,
        )

    @staticmethod
    def _profile_override_from_values(
        *,
        provider: str,
        api_base: str,
        model: str,
        reasoning_enabled: bool,
    ) -> LLMProfileOverride:
        return LLMProfileOverride(
            provider=str(provider or "anthropic").strip().lower() or "anthropic",
            api_base=str(api_base or "").strip() or "https://api.minimax.io",
            model=str(model or "").strip() or "MiniMax-M2",
            reasoning_enabled=bool(reasoning_enabled),
        )

    @staticmethod
    def _profile_override_payload(
        override: LLMProfileOverride | dict | None,
    ) -> dict | None:
        if override is None:
            return None
        if isinstance(override, LLMProfileOverride):
            return override.model_dump(exclude_none=True)
        return LLMProfileOverride.model_validate(override).model_dump(exclude_none=True)

    def _resolve_global_llm_profiles(self) -> tuple[LLMProfile, LLMProfile]:
        """Resolve default planner/worker profiles from file config."""
        if self._config is None:
            raise RuntimeError("Config not initialized.")
        base_profile = LLMProfile(
            provider=str(self._config.llm.provider or "anthropic").strip().lower() or "anthropic",
            api_base=str(self._config.llm.api_base or "").strip() or "https://api.minimax.io",
            model=str(self._config.llm.model or "").strip() or "MiniMax-M2",
            reasoning_enabled=bool(self._config.llm.reasoning_enabled),
        )
        planner_profile = merge_llm_profile(base_profile, self._config.llm.planner_profile)
        worker_profile = merge_llm_profile(planner_profile, self._config.llm.worker_profile)
        return planner_profile, worker_profile

    def _resolve_routed_api_config(
        self,
        *,
        account_id: str,
        active_config_id: str,
        routed_config_id: str | None,
        allow_active_config: bool = False,
    ):
        if self._account_store is None:
            raise RuntimeError("Account store not initialized.")
        normalized_id = str(routed_config_id or "").strip()
        if not normalized_id or normalized_id == active_config_id:
            return (
                self._account_store.get_api_config_record(
                    active_config_id,
                    account_id=account_id,
                )
                if normalized_id == active_config_id and allow_active_config
                else None
            )
        return self._account_store.get_api_config_record(
            normalized_id,
            account_id=account_id,
        )

    def _build_config_profile_override(self, api_config: Any | None) -> LLMProfileOverride | None:
        if api_config is None:
            return None
        return self._profile_override_from_values(
            provider=api_config.provider,
            api_base=api_config.api_base,
            model=api_config.model,
            reasoning_enabled=api_config.reasoning_enabled,
        )

    @staticmethod
    def _build_llm_fingerprint(
        *,
        account_id: str,
        source_key: str,
        profile_role: Literal["planner", "worker"],
        profile: LLMProfile,
        source_version: str = "",
    ) -> str:
        return (
            f"{account_id}:{source_key}:{source_version}:{profile_role}:"
            f"{profile.provider}:{profile.api_base}:{profile.model}:"
            f"{int(bool(profile.reasoning_enabled))}"
        )

    def _get_account_llm_runtime(
        self,
        runtime_context: AgentRuntimeContext | str | None,
        template_snapshot: AgentTemplateSnapshot | None = None,
    ) -> ResolvedLLMRuntime:
        """Resolve one role-specific LLM runtime for the current agent."""
        if isinstance(runtime_context, AgentRuntimeContext):
            resolved_context = runtime_context
        else:
            resolved_context = AgentRuntimeContext(
                session_id="compat",
                account_id=runtime_context,
                template_snapshot=template_snapshot,
                is_main_agent=True,
            )

        resolved_account_id = resolved_context.account_id or ROOT_ACCOUNT_ID
        profile_role: Literal["planner", "worker"] = (
            "planner" if resolved_context.is_main_agent else "worker"
        )
        if self._account_store is None:
            raise RuntimeError("Account store not initialized.")

        global_planner_profile, global_worker_profile = self._resolve_global_llm_profiles()
        api_config = self._account_store.get_active_api_config_record(resolved_account_id)

        if api_config is not None:
            routing_policy = api_config.llm_routing_policy
            planner_source_config = self._resolve_routed_api_config(
                account_id=resolved_account_id,
                active_config_id=api_config.id,
                routed_config_id=routing_policy.planner_api_config_id,
            )
            worker_source_config = self._resolve_routed_api_config(
                account_id=resolved_account_id,
                active_config_id=api_config.id,
                routed_config_id=routing_policy.worker_api_config_id,
                allow_active_config=True,
            )
            active_config_override = self._build_config_profile_override(api_config)
            planner_source_override = (
                self._build_config_profile_override(planner_source_config)
                or active_config_override
            )
            planner_profile = merge_llm_profile(
                global_planner_profile,
                planner_source_override,
                routing_policy.planner_profile,
            )
            if worker_source_config is None:
                worker_profile = merge_llm_profile(
                    planner_profile,
                    routing_policy.worker_profile,
                )
            else:
                worker_profile = merge_llm_profile(
                    global_worker_profile,
                    planner_source_override,
                    routing_policy.planner_profile,
                    self._build_config_profile_override(worker_source_config),
                    routing_policy.worker_profile,
                )
            resolved_profile = (
                planner_profile if profile_role == "planner" else worker_profile
            )
            source_config = (
                planner_source_config
                if profile_role == "planner" and planner_source_config is not None
                else worker_source_config
                if profile_role == "worker" and worker_source_config is not None
                else api_config
            )
            fingerprint = self._build_llm_fingerprint(
                account_id=resolved_account_id,
                source_key=source_config.id,
                source_version=source_config.updated_at,
                profile_role=profile_role,
                profile=resolved_profile,
            )
            client = self._llm_client_cache.get(fingerprint)
            if client is None:
                client = self._build_llm_client(
                    api_key=source_config.api_key,
                    provider=resolved_profile.provider,
                    api_base=resolved_profile.api_base,
                    model=resolved_profile.model,
                    reasoning_enabled=resolved_profile.reasoning_enabled,
                )
                self._llm_client_cache[fingerprint] = client
            if source_config.last_used_at is None:
                self._account_store.touch_api_config_last_used(
                    resolved_account_id,
                    source_config.id,
                )
            if source_config.id != api_config.id and api_config.last_used_at is None:
                self._account_store.touch_api_config_last_used(
                    resolved_account_id,
                    api_config.id,
                )
            return ResolvedLLMRuntime(
                client=client,
                fingerprint=fingerprint,
                profile_role=profile_role,
                provider=resolved_profile.provider,
                api_base=resolved_profile.api_base,
                model=resolved_profile.model,
                reasoning_enabled=resolved_profile.reasoning_enabled,
                source={
                    "account_config_id": api_config.id,
                    "account_config_name": api_config.name,
                    "config_scope": "account_active",
                    "template_id": (
                        template_snapshot.template_id if template_snapshot is not None else None
                    ),
                    "planner_api_config_id": routing_policy.planner_api_config_id,
                    "worker_api_config_id": routing_policy.worker_api_config_id,
                    "resolved_config_id": source_config.id,
                    "resolved_config_name": source_config.name,
                },
            )

        if (
            resolved_account_id == ROOT_ACCOUNT_ID
            and self._config is not None
            and self._config.has_valid_api_key()
        ):
            planner_profile = global_planner_profile
            worker_profile = global_worker_profile
            resolved_profile = (
                planner_profile if profile_role == "planner" else worker_profile
            )
            fingerprint = self._build_llm_fingerprint(
                account_id=resolved_account_id,
                source_key="legacy-root",
                source_version="file-config",
                profile_role=profile_role,
                profile=resolved_profile,
            )
            client = self._llm_client_cache.get(fingerprint)
            if client is None:
                client = self._build_llm_client(
                    api_key=self._config.llm.api_key,
                    provider=resolved_profile.provider,
                    api_base=resolved_profile.api_base,
                    model=resolved_profile.model,
                    reasoning_enabled=resolved_profile.reasoning_enabled,
                )
                self._llm_client_cache[fingerprint] = client
            return ResolvedLLMRuntime(
                client=client,
                fingerprint=fingerprint,
                profile_role=profile_role,
                provider=resolved_profile.provider,
                api_base=resolved_profile.api_base,
                model=resolved_profile.model,
                reasoning_enabled=resolved_profile.reasoning_enabled,
                source={
                    "config_scope": "legacy_root_file",
                    "template_id": (
                        template_snapshot.template_id if template_snapshot is not None else None
                    ),
                },
            )

        raise RuntimeError(
            "No active API key is configured for the current account. Bind one in API settings first."
        )

    def has_active_api_config(self, account_id: str) -> bool:
        """Return whether the account currently has an active API configuration."""
        if self._account_store is None:
            return False
        if self._account_store.get_active_api_config_record(account_id) is not None:
            return True
        return account_id == ROOT_ACCOUNT_ID and bool(
            self._config is not None and self._config.has_valid_api_key()
        )

    def list_account_api_configs(self, account_id: str) -> list[dict]:
        """List all API credential sets owned by one account."""
        if self._account_store is None:
            return []
        return [
            record.model_dump(mode="python")
            for record in self._account_store.list_api_config_records(account_id)
        ]

    def save_account_api_config(
        self,
        account_id: str,
        *,
        name: str,
        api_key: str,
        provider: str,
        api_base: str,
        model: str,
        reasoning_enabled: bool,
        llm_routing_policy: dict | None = None,
        activate: bool = True,
    ) -> dict:
        """Create or update one account-owned API credential set."""
        if self._account_store is None:
            raise RuntimeError("Account store not initialized.")
        routing_policy = llm_routing_policy or {}
        for field_name in ("planner_api_config_id", "worker_api_config_id"):
            config_id = str(routing_policy.get(field_name) or "").strip()
            if not config_id:
                continue
            if self._account_store.get_api_config_record(config_id, account_id=account_id) is None:
                raise ValueError(f"Unknown API configuration for {field_name}: {config_id}")
        record = self._account_store.upsert_api_config(
            account_id,
            name=name,
            api_key=api_key,
            provider=provider,
            api_base=api_base,
            model=model,
            reasoning_enabled=reasoning_enabled,
            llm_routing_policy=routing_policy,
            activate=activate,
        )
        self.invalidate_account_runtime(account_id)
        return record.model_dump(mode="python")

    def update_account_api_config_routing(
        self,
        account_id: str,
        config_id: str,
        *,
        llm_routing_policy: dict | None = None,
    ) -> dict:
        """Update only the routing policy for one existing account API config."""
        if self._account_store is None:
            raise RuntimeError("Account store not initialized.")
        existing = self._account_store.get_api_config_record(
            config_id,
            account_id=account_id,
        )
        if existing is None:
            raise KeyError(f"API configuration not found: {config_id}")
        routing_policy = llm_routing_policy or {}
        for field_name in ("planner_api_config_id", "worker_api_config_id"):
            routed_config_id = str(routing_policy.get(field_name) or "").strip()
            if not routed_config_id:
                continue
            if self._account_store.get_api_config_record(
                routed_config_id,
                account_id=account_id,
            ) is None:
                raise ValueError(f"Unknown API configuration for {field_name}: {routed_config_id}")
        record = self._account_store.upsert_api_config(
            account_id,
            name=existing.name,
            api_key=existing.api_key,
            provider=existing.provider,
            api_base=existing.api_base,
            model=existing.model,
            reasoning_enabled=existing.reasoning_enabled,
            llm_routing_policy=routing_policy,
            activate=existing.is_active,
        )
        self.invalidate_account_runtime(account_id)
        return record.model_dump(mode="python")

    def activate_account_api_config(self, account_id: str, config_id: str) -> dict:
        """Switch the active API credential set for one account."""
        if self._account_store is None:
            raise RuntimeError("Account store not initialized.")
        record = self._account_store.activate_api_config(account_id, config_id)
        self.invalidate_account_runtime(account_id)
        return record.model_dump(mode="python")

    def delete_account_api_config(self, account_id: str, config_id: str) -> bool:
        """Delete one account-owned API credential set."""
        if self._account_store is None:
            raise RuntimeError("Account store not initialized.")
        deleted = self._account_store.delete_api_config(account_id, config_id)
        if deleted:
            self.invalidate_account_runtime(account_id)
        return deleted

    def invalidate_account_runtime(self, account_id: str) -> None:
        """Drop cached clients and idle agents for one account after a config switch."""
        if self._account_store is None or self._session_store is None:
            return
        stale_fingerprints = [
            fingerprint
            for fingerprint in self._llm_client_cache
            if fingerprint.startswith(f"{account_id}:")
        ]
        for fingerprint in stale_fingerprints:
            self._llm_client_cache.pop(fingerprint, None)

        for session in self._session_store.list_session_records(account_id=account_id):
            if self.is_session_running(session.session_id):
                continue
            self._runtime_registry.remove(session.session_id)

    def _is_agent_llm_current(self, agent: Agent) -> bool:
        """Check whether an in-memory agent still matches the account's active API config."""
        context = agent.runtime_context or AgentRuntimeContext(
            session_id="",
            account_id=ROOT_ACCOUNT_ID,
            template_snapshot=agent.template_snapshot,
            is_main_agent=True,
        )
        template_snapshot = agent.template_snapshot or context.template_snapshot
        try:
            resolved_runtime = self._get_account_llm_runtime(context, template_snapshot)
        except RuntimeError:
            return False
        return getattr(agent, "llm_fingerprint", None) == resolved_runtime.fingerprint

    def get_feature_flags(self) -> dict[str, bool]:
        """Return effective runtime feature flags."""
        if self._config is None:
            return {
                "enable_durable_runs": True,
                "enable_run_trace": True,
                "enable_approval_flow": True,
                "enable_supervisor_mode": True,
                "enable_worker_model_routing": True,
                "enable_compact_prompt_memory": True,
                "enable_session_retrieval": True,
                "enable_learned_workflow_generation": True,
                "enable_external_memory_providers": True,
            }
        return self._config.get_feature_flags()

    def get_memory_provider_status(self) -> dict:
        """返回当前长时记忆 provider 的健康状态。"""
        if self._memory_provider is None:
            return {
                "configured_provider": "disabled",
                "active_provider": "disabled",
                "status": "disabled",
                "fallback_active": False,
                "inject_memories": False,
                "expose_tools": False,
                "sync_conversation_turns": False,
                "capabilities": {},
                "message": "Memory provider is not initialized.",
                "metadata": {},
            }
        health = self._memory_provider.get_health()
        prompt_memory_enabled = self.is_feature_enabled("enable_compact_prompt_memory")
        metadata = dict(health.get("metadata") or {})
        metadata["prompt_memory_feature_enabled"] = prompt_memory_enabled
        metadata["effective_inject_memories"] = bool(health.get("inject_memories")) and prompt_memory_enabled
        health["metadata"] = metadata
        if not prompt_memory_enabled:
            health["inject_memories"] = False
        return health

    def is_runtime_ready(self) -> bool:
        """Return whether runtime-dependent services are initialized."""
        return bool(self._initialized and self._runtime_ready and self._runtime_factory and self._run_manager)

    def _require_runtime_ready(self) -> None:
        """Guard runtime-only workflows when setup is incomplete."""
        if self.is_runtime_ready():
            return
        raise RuntimeError(
            "Clavi Agent runtime is not configured yet. Finish API setup in the web configuration page first."
        )

    def is_feature_enabled(self, flag_name: str) -> bool:
        """Check whether one named feature flag is enabled."""
        return bool(self.get_feature_flags().get(flag_name, False))

    def _get_shared_context_path(self, workspace_dir: Path, session_id: str) -> Path:
        """Return the shared context file path for a session."""
        return workspace_dir / ".clavi_agent" / "shared_context" / f"{session_id}.json"

    def _get_upload_root(self, workspace_dir: Path, session_id: str) -> Path:
        """返回当前会话的上传文件根目录。"""
        return workspace_dir / ".clavi_agent" / "uploads" / session_id

    def _get_account_record(self, account_id: str):
        """返回账号记录，不存在时直接报错。"""
        if self._account_store is None:
            raise RuntimeError("Account store not initialized.")
        account = self._account_store.get_account_record(account_id)
        if account is None:
            raise KeyError(f"Account not found: {account_id}")
        return account

    def _get_workspace_root(self) -> Path:
        """返回配置中的工作区根目录。"""
        if self._config is None:
            raise RuntimeError("Config not initialized.")
        workspace_root = Path(self._config.agent.workspace_dir).expanduser()
        if not workspace_root.is_absolute():
            workspace_root = Path.cwd() / workspace_root
        return workspace_root.resolve()

    def _get_account_workspace_root(self, account_id: str) -> Path:
        """返回账号级工作区根目录。"""
        return self._get_workspace_root() / "accounts" / account_id

    def _get_account_session_workspace(self, account_id: str, session_id: str) -> Path:
        """返回账号级会话工作区目录。"""
        return self._get_account_workspace_root(account_id) / "sessions" / session_id

    @staticmethod
    def _path_within_root(path: Path, root: Path) -> bool:
        """判断路径是否位于指定根目录下。"""
        try:
            path.relative_to(root)
        except ValueError:
            return False
        return True

    def _list_account_allowed_workspace_roots(self, account_id: str) -> list[Path]:
        """列出当前账号允许访问的工作区根目录。"""
        allowed_roots = [self._get_account_workspace_root(account_id).resolve()]
        if self._agent_store is not None:
            for template in self._agent_store.list_agent_template_records(
                account_id=account_id,
                include_system=False,
            ):
                allowed_roots.append(
                    self._agent_store.get_agent_workspace_dir(template.id).resolve()
                )

        deduped_roots: list[Path] = []
        seen: set[str] = set()
        for root in allowed_roots:
            key = str(root)
            if key in seen:
                continue
            deduped_roots.append(root)
            seen.add(key)
        return deduped_roots

    def _ensure_account_workspace_allowed(
        self,
        workspace_dir: str | Path,
        *,
        account_id: str,
        label: str,
    ) -> Path:
        """校验工作区路径是否仍在账号允许的磁盘边界内。"""
        resolved_workspace = Path(workspace_dir).resolve()
        account = self._get_account_record(account_id)
        if account.is_root:
            return resolved_workspace

        for allowed_root in self._list_account_allowed_workspace_roots(account_id):
            if self._path_within_root(resolved_workspace, allowed_root):
                return resolved_workspace

        raise PermissionError(f"{label} does not belong to the current account boundary.")

    def _resolve_session_workspace_dir(self, session) -> Path:
        """解析并校验会话记录中的工作区路径。"""
        return self._ensure_account_workspace_allowed(
            session.workspace_dir,
            account_id=session.account_id,
            label="Session workspace",
        )

    def _resolve_workspace_for_new_session(
        self,
        *,
        session_id: str,
        agent_id: str,
        template_snapshot,
        requested_workspace_dir: str | None,
        account_id: str,
    ) -> Path:
        """根据账号与模板规则解析新会话工作区。"""
        account = self._get_account_record(account_id)
        normalized_requested_workspace = str(requested_workspace_dir or "").strip()
        if normalized_requested_workspace:
            if not account.is_root:
                raise PermissionError("普通账号不允许自定义工作区路径。")
            return Path(normalized_requested_workspace).resolve()

        workspace_mode = str(template_snapshot.workspace_policy.mode or "isolated").strip().lower()
        if workspace_mode == "shared" and template_snapshot.account_id == account_id:
            return self._ensure_account_workspace_allowed(
                self._agent_store.get_agent_workspace_dir(agent_id),
                account_id=account_id,
                label="Session workspace",
            )

        return self._get_account_session_workspace(account_id, session_id).resolve()

    def _next_sub_agent_name(self, session_id: str) -> str:
        """Generate a unique incremental sub-agent label within one session.

        Uses ``itertools.count`` whose ``__next__`` is atomic under the GIL,
        ensuring concurrent sub_agent_factory calls always get distinct names.
        """
        return self._runtime_registry.next_sub_agent_name(session_id)

    def _load_run_agent(self, run: RunRecord) -> Agent | None:
        """Build or reuse the main runtime agent for one durable run."""
        if self._runtime_factory is None:
            return None

        session = self._session_store.get_session_record(run.session_id)
        if session is None:
            return None

        runtime_hooks = AgentRuntimeHooks(
            trace_sink=self._record_agent_runtime_trace_event,
            approval_hook=(
                self._request_run_approval
                if self.is_feature_enabled("enable_approval_flow")
                else None
            ),
            prompt_refresh_hook=self._refresh_run_prompt_after_tool,
        )
        workspace_dir = self._resolve_session_workspace_dir(session)
        uploaded_file_targets = self._build_uploaded_file_targets(run.session_id)

        if run.parent_run_id is None:
            existing_agent = self._runtime_registry.get_agent(run.session_id)
            seed_messages = (
                existing_agent.get_history()
                if existing_agent is not None
                else self._session_store.get_messages(run.session_id)
            )
            agent = self._runtime_factory.prepare_run_agent(
                run=run,
                workspace_dir=workspace_dir,
                seed_messages=seed_messages,
                existing_agent=existing_agent,
                runtime_hooks=runtime_hooks,
                uploaded_file_targets=uploaded_file_targets,
            )
            return self._runtime_registry.bind_agent(run.session_id, agent)

        seed_messages = self._load_child_run_messages(run)
        run_metadata = dict(run.run_metadata)
        raw_max_steps = run_metadata.get("max_steps", self._config.agent.max_steps)
        try:
            max_steps = max(1, int(raw_max_steps))
        except (TypeError, ValueError):
            max_steps = self._config.agent.max_steps
        raw_depth = run_metadata.get("depth", 1)
        try:
            depth = max(0, int(raw_depth))
        except (TypeError, ValueError):
            depth = 1

        return self._runtime_factory.build_agent(
            workspace_dir=workspace_dir,
            session_id=run.session_id,
            template_snapshot=run.agent_template_snapshot,
            runtime_context=AgentRuntimeContext(
                session_id=run.session_id,
                account_id=run.account_id,
                run_id=run.id,
                agent_name=str(run_metadata.get("agent_name") or "worker"),
                template_snapshot=run.agent_template_snapshot,
                parent_run_id=run.parent_run_id,
                root_run_id=str(
                    run_metadata.get("root_run_id")
                    or run.parent_run_id
                    or run.id
                ),
                is_main_agent=False,
                depth=depth,
                approval_auto_grant_tools=list(
                    run_metadata.get("approval_auto_grant_tools", [])
                ),
                uploaded_file_targets=uploaded_file_targets,
                integration_id=str(run_metadata.get("integration_id") or "").strip() or None,
                channel_kind=str(run_metadata.get("channel_kind") or "").strip() or None,
                binding_id=str(run_metadata.get("binding_id") or "").strip() or None,
                inbound_event_id=str(run_metadata.get("inbound_event_id") or "").strip() or None,
                provider_chat_id=str(run_metadata.get("provider_chat_id") or "").strip() or None,
                provider_thread_id=str(run_metadata.get("provider_thread_id") or "").strip() or None,
                provider_message_id=str(run_metadata.get("provider_message_id") or "").strip() or None,
            ),
            runtime_hooks=runtime_hooks,
            is_main_agent=False,
            custom_prompt=str(
                run_metadata.get("persona")
                or run.agent_template_snapshot.system_prompt
            ),
            max_steps=max_steps,
            agent_name=str(run_metadata.get("agent_name") or "worker"),
            seed_messages=seed_messages,
        )

    def _load_child_run_messages(self, run: RunRecord) -> list[Message]:
        """Restore child-run history from its latest durable checkpoint."""
        checkpoint = self._run_store.get_latest_checkpoint(run.id)
        if checkpoint is None:
            return []

        messages: list[Message] = []
        for snapshot in checkpoint.payload.message_snapshot:
            tool_calls = None
            if snapshot.tool_call_names:
                tool_calls = [
                    ToolCall(
                        id=f"restored-{index}-{name}",
                        type="function",
                        function=FunctionCall(name=name, arguments={}),
                    )
                    for index, name in enumerate(snapshot.tool_call_names, start=1)
                ]
            messages.append(
                Message(
                    role=snapshot.role,
                    content=snapshot.content,
                    thinking=snapshot.thinking or None,
                    tool_calls=tool_calls,
                    tool_call_id=snapshot.tool_call_id,
                    name=snapshot.name,
                )
            )
        return messages

    async def _execute_delegate_child_run(
        self,
        *,
        parent_context: AgentRuntimeContext,
        parent_hooks: AgentRuntimeHooks,  # noqa: ARG002
        template_snapshot,
        workspace_dir: Path,  # noqa: ARG002
        agent_name: str,
        persona: str,
        task: str,
        max_steps: int,
    ) -> AsyncGenerator[dict, None]:
        """Execute one delegated worker as a durable child run."""
        if self._run_manager is None or parent_context.run_id is None:
            return

        child_run = self._run_manager.start_run(
            parent_context.session_id,
            task,
            parent_run_id=parent_context.run_id,
            run_metadata={
                "kind": "delegate_child",
                "agent_name": agent_name,
                "persona": persona,
                "max_steps": max_steps,
                "depth": parent_context.depth + 1,
                "root_run_id": parent_context.root_run_id or parent_context.run_id,
                "template_id": template_snapshot.template_id,
                "template_version": template_snapshot.template_version,
                "integration_id": parent_context.integration_id or "",
                "channel_kind": parent_context.channel_kind or "",
                "binding_id": parent_context.binding_id or "",
                "inbound_event_id": parent_context.inbound_event_id or "",
                "provider_chat_id": parent_context.provider_chat_id or "",
                "provider_thread_id": parent_context.provider_thread_id or "",
                "provider_message_id": parent_context.provider_message_id or "",
            },
        )

        try:
            async for event in self._run_manager.stream_run(child_run.id):
                yield event
        except asyncio.CancelledError:
            with suppress(Exception):
                self._run_manager.cancel_run(child_run.id)
            raise

    def _record_agent_runtime_trace_event(
        self,
        context: AgentRuntimeContext,
        event: dict[str, object],
    ) -> None:
        """Persist one structured runtime event emitted by a run-scoped agent."""
        if self._run_manager is None:
            return
        self._run_manager.record_runtime_trace_event(context, event)

    def _refresh_run_prompt_after_tool(
        self,
        context: AgentRuntimeContext,
        agent: Agent,
        payload: dict[str, object],
    ) -> dict[str, object] | None:
        """根据工具触达路径刷新运行中 agent 的项目上下文提示词。"""
        del context
        if self._runtime_factory is None:
            return None
        return self._runtime_factory.refresh_agent_prompt_from_tool_payload(
            agent=agent,
            payload=dict(payload),
        )

    async def _send_runtime_channel_file(
        self,
        runtime_context: AgentRuntimeContext,
        local_path: Path,
        file_name: str,
        text_fallback: str,
    ) -> dict[str, Any]:
        if self._integration_reply_dispatcher is None:
            raise RuntimeError("Integration reply dispatcher is not initialized.")

        delivery = await self._integration_reply_dispatcher.dispatch_tool_file(
            runtime_context=runtime_context,
            local_path=local_path,
            file_name=file_name,
            text_fallback=text_fallback,
        )
        if delivery.status != "delivered":
            raise RuntimeError(delivery.error_summary or "Channel file delivery failed.")

        return {
            "delivery_id": delivery.id,
            "delivery_type": delivery.delivery_type,
            "provider_message_id": delivery.provider_message_id,
            "provider_chat_id": delivery.provider_chat_id,
            "provider_thread_id": delivery.provider_thread_id,
            "channel_kind": runtime_context.channel_kind or "",
            "file_name": file_name,
            "local_path": str(local_path),
        }

    async def _request_run_approval(
        self,
        context: AgentRuntimeContext,
        agent: Agent,
        payload: dict[str, object],
    ) -> dict[str, object] | None:
        """Persist one approval request and suspend execution until it is resolved."""
        if self._run_manager is None:
            return None
        return await self._run_manager.request_approval(context, agent, payload)

    @staticmethod
    def _resolve_account_scoped_record(
        scoped_record,
        global_record,
        *,
        strict: bool,
        forbidden_message: str,
    ):
        if scoped_record is not None:
            return scoped_record
        if strict and global_record is not None:
            raise PermissionError(forbidden_message)
        return None

    async def create_session(
        self,
        workspace_dir: str | None = None,
        agent_id: str | None = None,
        *,
        account_id: str = ROOT_ACCOUNT_ID,
    ) -> str:
        """Create a new session with its own Agent instance.

        Args:
            workspace_dir: Optional workspace directory override.
            agent_id: Optional agent template ID to define tools and persona.

        Returns:
            session_id (uuid string)
        """
        await self.initialize()
        self._require_runtime_ready()

        if not agent_id:
            agent_id = "system-default-agent"

        template_snapshot = self._agent_store.snapshot_agent_template(
            agent_id,
            account_id=account_id,
        )
        if not template_snapshot:
            if (
                agent_id != "system-default-agent"
                and self._agent_store.get_agent_template(agent_id) is not None
            ):
                raise PermissionError("Agent template does not belong to the current account.")
            agent_id = "system-default-agent"
            template_snapshot = self._agent_store.snapshot_agent_template(
                agent_id,
                account_id=account_id,
            )
        if template_snapshot is None:
            raise KeyError(f"Agent template not found: {agent_id}")

        session_id = str(uuid.uuid4())
        ws = self._resolve_workspace_for_new_session(
            session_id=session_id,
            agent_id=agent_id,
            template_snapshot=template_snapshot,
            requested_workspace_dir=workspace_dir,
            account_id=account_id,
        )
        agent = self._runtime_factory.build_session_agent(
            session_id=session_id,
            account_id=account_id,
            workspace_dir=ws,
            template_snapshot=template_snapshot,
        )

        self._runtime_registry.bind_agent(session_id, agent)
        self._session_store.create_session(
            session_id=session_id,
            workspace_dir=str(ws),
            messages=agent.get_history(),
            title=DEFAULT_SESSION_TITLE,
            agent_id=agent_id,
            account_id=account_id,
        )
        return session_id

    def get_session(self, session_id: str) -> Agent | None:
        """Get Agent instance for a session."""
        agent = self._runtime_registry.get_agent(session_id)
        if agent and self._is_agent_llm_current(agent):
            return agent
        if agent is not None:
            self._runtime_registry.remove(session_id)
        return self.restore_session(session_id)

    def bind_session_agent(self, session_id: str, agent: Agent) -> Agent:
        """Replace the in-memory runtime agent for an existing session."""
        session = self._session_store.get_session_record(session_id)
        if not session:
            raise KeyError(f"Session not found: {session_id}")
        agent.manual_runtime_override = True
        if getattr(agent, "llm_fingerprint", None) is None:
            try:
                resolved_runtime = self._get_account_llm_runtime(
                    agent.runtime_context
                    or AgentRuntimeContext(
                        session_id=session_id,
                        account_id=session.account_id,
                        template_snapshot=agent.template_snapshot,
                        is_main_agent=True,
                    ),
                    agent.template_snapshot,
                )
            except RuntimeError:
                resolved_runtime = None
            agent.llm_fingerprint = (
                resolved_runtime.fingerprint if resolved_runtime is not None else None
            )
        return self._runtime_registry.bind_agent(session_id, agent)

    async def switch_session_agent(
        self,
        session_id: str,
        agent_id: str,
        *,
        account_id: str | None = None,
    ) -> Agent:
        """切换现有会话绑定的 Agent 模板，并重建注入记忆后的运行时 Agent。"""
        await self.initialize()
        self._require_runtime_ready()

        normalized_agent_id = str(agent_id or "").strip() or "system-default-agent"
        session = self._resolve_account_scoped_record(
            self._session_store.get_session_record(session_id, account_id=account_id),
            self._session_store.get_session_record(session_id) if account_id is not None else None,
            strict=account_id is not None,
            forbidden_message="Session does not belong to the current account.",
        )
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if self.is_session_running(session_id):
            raise RuntimeError("Cannot switch agent template while the session is running.")

        template_snapshot = self._agent_store.snapshot_agent_template(
            normalized_agent_id,
            account_id=session.account_id,
        )
        if template_snapshot is None:
            if (
                normalized_agent_id != "system-default-agent"
                and self._agent_store.get_agent_template(normalized_agent_id) is not None
            ):
                raise PermissionError("Agent template does not belong to the current account.")
            raise KeyError(f"Agent template not found: {normalized_agent_id}")

        workspace_dir = self._resolve_session_workspace_dir(session)
        existing_agent = self.get_session(session_id)
        preserved_messages = (
            existing_agent.get_history()
            if existing_agent is not None
            else self._session_store.get_messages(session_id, account_id=session.account_id)
        )
        rebuilt_agent = self._runtime_factory.build_session_agent(
            session_id=session_id,
            account_id=session.account_id,
            workspace_dir=workspace_dir,
            template_snapshot=template_snapshot,
            messages=preserved_messages,
        )

        self._runtime_registry.bind_agent(session_id, rebuilt_agent)
        self._session_store.replace_messages(
            session_id,
            rebuilt_agent.get_history(),
            account_id=session.account_id,
        )
        self._session_store.update_session_agent_id(
            session_id,
            normalized_agent_id,
            account_id=session.account_id,
        )
        return rebuilt_agent

    def restore_session(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
    ) -> Agent | None:
        """Restore a session from persistent storage into memory."""
        if not self.is_runtime_ready():
            return None
        agent = self._runtime_registry.get_agent(session_id)
        if agent is not None:
            if not self._is_agent_llm_current(agent):
                self._runtime_registry.remove(session_id)
                agent = None
            else:
                if account_id is None:
                    return agent
                session = self._session_store.get_session_record(session_id, account_id=account_id)
                if session is None:
                    return None
                return agent

        session = self._session_store.get_session_record(session_id, account_id=account_id)
        if not session:
            return None

        workspace_dir = self._resolve_session_workspace_dir(session)
        agent_id = session.agent_id or "system-default-agent"
        template_snapshot = self._agent_store.snapshot_agent_template(
            agent_id,
            account_id=session.account_id,
        )
        if template_snapshot is None:
            template_snapshot = self._agent_store.snapshot_agent_template(
                "system-default-agent",
                account_id=session.account_id,
            )
        if template_snapshot is None:
            return None

        agent = self._runtime_factory.build_session_agent(
            session_id=session_id,
            account_id=session.account_id,
            workspace_dir=workspace_dir,
            template_snapshot=template_snapshot,
        )
        messages = self._session_store.get_messages(session_id, account_id=session.account_id)
        if messages:
            agent.messages = messages
            agent.set_system_prompt(agent.system_prompt)
        return self._runtime_registry.bind_agent(session_id, agent)

    def get_session_info(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> dict | None:
        """Get persisted metadata for a session."""
        session = self._resolve_account_scoped_record(
            self._session_store.get_session_record(session_id, account_id=account_id),
            self._session_store.get_session_record(session_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Session does not belong to the current account.",
        )
        if session is None:
            return None
        return session.model_dump()

    def get_user_profile_info(self, *, account_id: str) -> dict | None:
        """返回指定账号的结构化用户画像。"""
        if self._memory_provider is None:
            return None
        return self._memory_provider.inspect_user_profile(account_id)

    def update_user_profile_info(
        self,
        *,
        account_id: str,
        profile_updates: dict | None = None,
        remove_fields: list[str] | None = None,
        summary: str | object = _UNSET,
        profile_source: str = "explicit",
        profile_confidence: float = 1.0,
    ) -> dict | None:
        """更新指定账号的结构化用户画像。"""
        if self._memory_provider is None:
            return None
        payload = {
            "profile_updates": profile_updates,
            "remove_fields": remove_fields,
            "profile_source": profile_source,
            "profile_confidence": profile_confidence,
            "writer_type": "user",
            "writer_id": "web_ui",
        }
        if summary is not _UNSET:
            payload["summary"] = summary
        return self._memory_provider.update_user_profile(account_id, **payload)

    def list_user_memory_entries_info(
        self,
        *,
        account_id: str,
        query: str | None = None,
        memory_types: list[str] | None = None,
        include_superseded: bool = False,
        limit: int | None = None,
    ) -> list[dict]:
        """返回指定账号的用户长期记忆条目。"""
        if self._memory_provider is None:
            return []

        normalized_query = str(query or "").strip()
        if normalized_query:
            return self._memory_provider.search_memory_entries(
                account_id,
                query=normalized_query,
                memory_types=memory_types,
                include_superseded=include_superseded,
                limit=limit or 20,
            )

        return self._memory_provider.list_memory_entries(
            account_id,
            memory_types=memory_types,
            include_superseded=include_superseded,
            limit=limit,
        )

    def get_user_memory_entry_info(
        self,
        entry_id: str,
        *,
        account_id: str,
    ) -> dict | None:
        """返回指定账号的一条用户长期记忆。"""
        if self._memory_provider is None:
            return None
        return self._memory_provider.get_memory_entry(entry_id, user_id=account_id)

    def update_user_memory_entry_info(
        self,
        entry_id: str,
        *,
        account_id: str,
        memory_type: str | object = _UNSET,
        content: str | object = _UNSET,
        summary: str | object = _UNSET,
        confidence: float | object = _UNSET,
    ) -> dict | None:
        """更新指定账号的一条用户长期记忆。"""
        if self._memory_provider is None:
            return None
        payload = {
            "user_id": account_id,
            "writer_type": "user",
            "writer_id": "web_ui",
        }
        if memory_type is not _UNSET:
            payload["memory_type"] = memory_type
        if content is not _UNSET:
            payload["content"] = content
        if summary is not _UNSET:
            payload["summary"] = summary
        if confidence is not _UNSET:
            payload["confidence"] = confidence
        return self._memory_provider.update_memory_entry(entry_id, **payload)

    def delete_user_memory_entry_info(
        self,
        entry_id: str,
        *,
        account_id: str,
        reason: str | None = None,
    ) -> bool:
        """软删除指定账号的一条用户长期记忆。"""
        if self._memory_provider is None:
            return False
        return self._memory_provider.delete_memory_entry(
            entry_id,
            user_id=account_id,
            reason=reason,
            writer_type="user",
            writer_id="web_ui",
        )

    def list_user_memory_audit_events(
        self,
        *,
        account_id: str,
        target_scope: str | None = None,
        target_id: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """返回指定账号最近的记忆审计事件。"""
        if self._memory_provider is None:
            return []
        return self._memory_provider.list_audit_events(
            account_id,
            target_scope=target_scope,
            target_id=target_id,
            limit=limit,
        )

    def get_session_messages(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> list:
        """Get persisted messages for a session."""
        session = self._resolve_account_scoped_record(
            self._session_store.get_session_record(session_id, account_id=account_id),
            self._session_store.get_session_record(session_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Session does not belong to the current account.",
        )
        if session is None:
            return []
        return self._session_store.get_messages(session_id, account_id=session.account_id)

    def create_session_uploads(
        self,
        session_id: str,
        uploads: list[UploadCreatePayload],
        *,
        account_id: str | None = None,
        run_id: str | None = None,
        created_by: str = "user",
    ) -> list[UploadRecord]:
        """创建并持久化当前会话的上传文件。"""
        if self._session_store is None or self._upload_store is None:
            raise RuntimeError("Upload store not initialized.")

        session = self._resolve_account_scoped_record(
            self._session_store.get_session_record(session_id, account_id=account_id),
            self._session_store.get_session_record(session_id) if account_id is not None else None,
            strict=account_id is not None,
            forbidden_message="Session does not belong to the current account.",
        )
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        if run_id is not None:
            if self._run_store is None:
                raise RuntimeError("Run store not initialized.")
            run = self._resolve_account_scoped_record(
                self._run_store.get_run(run_id, account_id=session.account_id),
                self._run_store.get_run(run_id) if account_id is not None else None,
                strict=account_id is not None,
                forbidden_message="Run does not belong to the current account.",
            )
            if run is None:
                raise KeyError(f"Run not found: {run_id}")
            if run.session_id != session_id:
                raise ValueError("run_id does not belong to the target session.")

        normalized_creator = created_by.strip() or "user"
        workspace_dir = self._resolve_session_workspace_dir(session)
        upload_root = self._get_upload_root(workspace_dir, session_id)

        prepared_uploads: list[dict[str, object]] = []
        for upload in uploads:
            safe_name = sanitize_upload_filename(upload.original_name)
            extension = upload_extension(safe_name)
            if extension in BLOCKED_UPLOAD_EXTENSIONS:
                raise ValueError(f"Unsupported upload file type: {extension}")
            if extension not in ALLOWED_UPLOAD_EXTENSIONS:
                raise ValueError(f"Upload file type is not allowed: {extension or '[no extension]'}")

            size_bytes = len(upload.content_bytes)
            if size_bytes > MAX_UPLOAD_SIZE_BYTES:
                raise ValueError(
                    f"Upload exceeds size limit of {MAX_UPLOAD_SIZE_BYTES} bytes: {safe_name}"
                )

            prepared_uploads.append(
                {
                    "original_name": upload.original_name,
                    "safe_name": safe_name,
                    "mime_type": resolve_upload_mime_type(safe_name, upload.mime_type),
                    "size_bytes": size_bytes,
                    "checksum": compute_upload_checksum(upload.content_bytes),
                    "content_bytes": upload.content_bytes,
                }
            )

        created_uploads: list[UploadRecord] = []
        created_paths: list[Path] = []

        try:
            for prepared in prepared_uploads:
                upload_id = str(uuid.uuid4())
                target_dir = upload_root / upload_id
                target_dir.mkdir(parents=True, exist_ok=False)
                target_path = target_dir / str(prepared["safe_name"])
                target_path.write_bytes(bytes(prepared["content_bytes"]))
                created_paths.append(target_dir)

                relative_path = target_path.relative_to(workspace_dir).as_posix()
                record = UploadRecord(
                    id=upload_id,
                    session_id=session_id,
                    run_id=run_id,
                    original_name=str(prepared["original_name"]),
                    safe_name=str(prepared["safe_name"]),
                    relative_path=relative_path,
                    absolute_path=str(target_path.resolve()),
                    mime_type=str(prepared["mime_type"]),
                    size_bytes=int(prepared["size_bytes"]),
                    checksum=str(prepared["checksum"]),
                    created_at=utc_now_iso(),
                    created_by=normalized_creator,
                )
                self._upload_store.create_upload(record)
                created_uploads.append(record)
        except Exception:
            for record in created_uploads:
                self._upload_store.delete_upload(record.id)
            for path in reversed(created_paths):
                shutil.rmtree(path, ignore_errors=True)
            raise

        return created_uploads

    def _build_uploaded_file_targets(self, session_id: str) -> list[UploadedFileTarget]:
        if self._session_store is None or self._upload_store is None:
            return []

        session = self._session_store.get_session_record(session_id)
        if session is None:
            return []

        workspace_dir = self._resolve_session_workspace_dir(session)
        targets: list[UploadedFileTarget] = []
        for upload in self._upload_store.list_uploads(session_id):
            try:
                absolute_path = (workspace_dir / Path(upload.relative_path)).resolve()
            except Exception:
                absolute_path = Path(upload.absolute_path).resolve()
            targets.append(
                UploadedFileTarget(
                    upload_id=upload.id,
                    upload_name=upload.original_name or upload.safe_name or upload.id,
                    absolute_path=str(absolute_path),
                    relative_path=upload.relative_path,
                )
            )
        return targets

    def list_session_uploads(
        self,
        session_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> list[UploadRecord]:
        """列出一个会话的所有上传文件。"""
        if self._session_store is None or self._upload_store is None:
            return []
        session = self._resolve_account_scoped_record(
            self._session_store.get_session_record(session_id, account_id=account_id),
            self._session_store.get_session_record(session_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Session does not belong to the current account.",
        )
        if session is None:
            raise KeyError(f"Session not found: {session_id}")
        return self._upload_store.list_uploads(session_id, account_id=session.account_id)

    def get_upload_info(
        self,
        upload_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> UploadRecord | None:
        """返回一个上传文件的元数据。"""
        if self._upload_store is None:
            return None
        return self._resolve_account_scoped_record(
            self._upload_store.get_upload(upload_id, account_id=account_id),
            self._upload_store.get_upload(upload_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Upload does not belong to the current account.",
        )

    @staticmethod
    def _resolve_session_workspace_path(
        workspace_dir: Path,
        path_value: str,
        *,
        label: str,
    ) -> Path:
        candidate_path = Path(path_value)
        resolved_path = (
            candidate_path.resolve()
            if candidate_path.is_absolute()
            else (workspace_dir / candidate_path).resolve()
        )
        try:
            resolved_path.relative_to(workspace_dir)
        except ValueError as exc:
            raise ValueError(f"{label} path escapes the session workspace.") from exc
        return resolved_path

    def resolve_upload_path(
        self,
        upload_id: str,
        *,
        account_id: str | None = None,
    ) -> tuple[UploadRecord, Path]:
        """解析并校验上传文件在工作区内的真实路径。"""
        if self._session_store is None or self._upload_store is None:
            raise RuntimeError("Upload store not initialized.")

        upload = self._resolve_account_scoped_record(
            self._upload_store.get_upload(upload_id, account_id=account_id),
            self._upload_store.get_upload(upload_id) if account_id is not None else None,
            strict=account_id is not None,
            forbidden_message="Upload does not belong to the current account.",
        )
        if upload is None:
            raise KeyError(f"Upload not found: {upload_id}")

        session = self._session_store.get_session_record(
            upload.session_id,
            account_id=upload.account_id,
        )
        if session is None:
            raise KeyError(f"Session not found: {upload.session_id}")

        workspace_dir = self._resolve_session_workspace_dir(session)
        resolved_from_relative = self._resolve_session_workspace_path(
            workspace_dir,
            upload.relative_path,
            label="Upload",
        )
        resolved_from_absolute = self._resolve_session_workspace_path(
            workspace_dir,
            upload.absolute_path,
            label="Upload",
        )

        if resolved_from_relative != resolved_from_absolute:
            raise RuntimeError("Upload path metadata is inconsistent.")
        if not resolved_from_absolute.exists():
            raise FileNotFoundError(f"Upload file missing on disk: {upload_id}")

        return upload, resolved_from_absolute

    def get_artifact_info(
        self,
        artifact_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> ArtifactRecord | None:
        """返回一个运行产出的元数据。"""
        if self._run_store is None:
            return None
        return self._resolve_account_scoped_record(
            self._run_store.get_artifact(artifact_id, account_id=account_id),
            self._run_store.get_artifact(artifact_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Artifact does not belong to the current account.",
        )

    def resolve_artifact_path(
        self,
        artifact_id: str,
        *,
        account_id: str | None = None,
    ) -> tuple[ArtifactRecord, Path]:
        """解析并校验产出文件在工作区内的真实路径。"""
        if self._session_store is None or self._run_store is None:
            raise RuntimeError("Run store not initialized.")

        artifact = self._resolve_account_scoped_record(
            self._run_store.get_artifact(artifact_id, account_id=account_id),
            self._run_store.get_artifact(artifact_id) if account_id is not None else None,
            strict=account_id is not None,
            forbidden_message="Artifact does not belong to the current account.",
        )
        if artifact is None:
            raise KeyError(f"Artifact not found: {artifact_id}")

        if artifact.artifact_type not in {"workspace_file", "document"}:
            raise ValueError("Artifact is not backed by a workspace file.")

        run = self._resolve_account_scoped_record(
            self._run_store.get_run(artifact.run_id, account_id=account_id),
            self._run_store.get_run(artifact.run_id) if account_id is not None else None,
            strict=account_id is not None,
            forbidden_message="Run does not belong to the current account.",
        )
        if run is None:
            raise KeyError(f"Run not found: {artifact.run_id}")

        session = self._session_store.get_session_record(run.session_id, account_id=run.account_id)
        if session is None:
            raise KeyError(f"Session not found: {run.session_id}")

        artifact_uri = str(artifact.uri).strip()
        if not artifact_uri or artifact_uri.startswith("bash://"):
            raise ValueError("Artifact is not backed by a workspace file.")

        workspace_dir = self._resolve_session_workspace_dir(session)
        resolved_path = self._resolve_session_workspace_path(
            workspace_dir,
            artifact_uri,
            label="Artifact",
        )
        if not resolved_path.exists():
            display_name = str(artifact.display_name or artifact_id).strip() or artifact_id
            raise FileNotFoundError(
                f"Artifact file missing on disk: {display_name} ({artifact_id})"
            )
        if not resolved_path.is_file():
            raise FileNotFoundError(f"Artifact path is not a file: {artifact_id}")

        return artifact, resolved_path

    def get_shared_context_entries(
        self,
        session_id: str,
        limit: int = 200,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> list[dict]:
        """Get persisted shared-context entries for one session.

        Returns newest-first entries with normalized fields.
        """
        if self._session_store is None:
            return []

        session = self._resolve_account_scoped_record(
            self._session_store.get_session_record(session_id, account_id=account_id),
            self._session_store.get_session_record(session_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Session does not belong to the current account.",
        )
        if not session:
            return []

        shared_context_file = self._get_shared_context_path(
            self._resolve_session_workspace_dir(session),
            session_id,
        )
        if not shared_context_file.exists():
            return []

        try:
            raw = json.loads(shared_context_file.read_text(encoding="utf-8"))
        except Exception:
            return []

        if not isinstance(raw, list):
            return []

        try:
            parsed_limit = int(limit)
        except (TypeError, ValueError):
            parsed_limit = 200
        parsed_limit = max(1, min(parsed_limit, 500))

        normalized: list[dict] = []
        for entry in reversed(raw):
            if not isinstance(entry, dict):
                continue
            normalized.append(
                {
                    "id": str(entry.get("id", "")),
                    "timestamp": str(entry.get("timestamp", "")),
                    "source": str(entry.get("source", "unknown")),
                    "category": str(entry.get("category", "general")),
                    "title": str(entry.get("title", "")),
                    "content": str(entry.get("content", "")),
                }
            )
            if len(normalized) >= parsed_limit:
                break

        return normalized

    def _sync_session_snapshot(self, session_id: str):
        """Persist the latest in-memory session state."""
        agent = self._runtime_registry.get_agent(session_id)
        if not agent:
            return
        self._session_store.replace_messages(session_id, agent.get_history())

    def _sync_memory_provider_turn(self, run: RunRecord, messages: list[Message]) -> None:
        """将已完成 run 的会话轮次同步给长时记忆 provider。"""
        if self._memory_provider is None:
            return
        self._memory_provider.sync_completed_conversation_turn(
            user_id=run.account_id,
            session_id=run.session_id,
            run_id=run.id,
            messages=messages,
        )

    async def _handle_terminal_run(self, run: RunRecord) -> None:
        """在 root run 结束后异步调度渠道回复分发。"""
        if self._integration_reply_dispatcher is None:
            return
        await self._integration_reply_dispatcher.handle_terminal_run(run)

    def schedule_integration_quick_response(
        self,
        run: RunRecord,
        *,
        msg_context: MsgContextEnvelope,
    ) -> asyncio.Task[None] | None:
        if self._integration_reply_dispatcher is None:
            return None
        return self._integration_reply_dispatcher.schedule_quick_response(
            run,
            msg_context=msg_context,
        )

    def is_session_running(self, session_id: str) -> bool:
        """Whether a session currently has an active chat run."""
        return self._runtime_registry.is_running(session_id)

    async def interrupt_session(self, session_id: str) -> bool:
        """Interrupt the current run for a session if one exists."""
        agent = self.get_session(session_id)
        if not agent:
            return False

        if self._run_manager is not None:
            return self._run_manager.interrupt_session(session_id)
        return self._runtime_registry.interrupt(session_id)

    def get_run_info(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> dict | None:
        """Get persisted metadata for one durable run."""
        if self._run_manager is None:
            return None
        run = self._resolve_account_scoped_record(
            self._run_manager.get_run(run_id) if account_id is None else self._run_store.get_run(run_id, account_id=account_id),
            self._run_manager.get_run(run_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Run does not belong to the current account.",
        )
        if run is None:
            return None
        return run.model_dump(mode="json")

    def list_runs(
        self,
        *,
        account_id: str | None = None,
        session_id: str | None = None,
        status: str | None = None,
        parent_run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """List persisted runs with optional filters."""
        if self._run_store is None:
            return []
        runs = self._run_store.list_runs(
            account_id=account_id,
            session_id=session_id,
            status=status,
            parent_run_id=parent_run_id,
            limit=limit,
        )
        return [run.model_dump(mode="json") for run in runs]

    def list_run_steps(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> list[dict]:
        """List durable steps for one run."""
        if self._run_store is None:
            return []
        run = self._resolve_account_scoped_record(
            self._run_store.get_run(run_id, account_id=account_id),
            self._run_store.get_run(run_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Run does not belong to the current account.",
        )
        if run is None:
            return []
        return [
            step.model_dump(mode="json")
            for step in self._run_store.list_steps(run_id, account_id=run.account_id)
        ]

    def list_run_trace(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
        offset: int | None = None,
    ) -> list[dict]:
        """List durable trace events for one run."""
        if self._trace_store is None:
            return []
        run = self._resolve_account_scoped_record(
            self._run_store.get_run(run_id, account_id=account_id) if self._run_store is not None else None,
            self._run_store.get_run(run_id) if account_id is not None and self._run_store is not None else None,
            strict=strict,
            forbidden_message="Run does not belong to the current account.",
        )
        if run is None:
            return []
        return [
            event.model_dump(mode="json")
            for event in self._trace_store.list_events(
                run_id,
                account_id=run.account_id,
                offset=offset,
            )
        ]

    def get_run_metrics(
        self,
        *,
        account_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> dict:
        """Aggregate durable run metrics across recent root runs."""
        if self._run_store is None or self._trace_store is None:
            return {}

        runs = self._run_store.list_runs(account_id=account_id, session_id=session_id)
        root_runs = [run for run in runs if run.parent_run_id is None]
        if limit is not None:
            root_runs = root_runs[: max(1, int(limit))]

        family_runs_by_root_id: dict[str, list[RunRecord]] = {}
        trace_by_run_id: dict[str, list] = {}
        relevant_run_ids: set[str] = set()

        for root_run in root_runs:
            _, family_runs = collect_run_family(self._run_store, root_run.id)
            if not family_runs:
                family_runs = [root_run]
            family_runs_by_root_id[root_run.id] = family_runs
            for run in family_runs:
                relevant_run_ids.add(run.id)
                if run.id not in trace_by_run_id:
                    trace_by_run_id[run.id] = self._trace_store.list_events(
                        run.id,
                        account_id=run.account_id,
                    )

        approvals_by_run_id: dict[str, list] = {}
        if self._approval_store is not None and relevant_run_ids:
            for approval in self._approval_store.list_requests(account_id=account_id):
                if approval.run_id not in relevant_run_ids:
                    continue
                approvals_by_run_id.setdefault(approval.run_id, []).append(approval)

        return build_run_metrics(
            root_runs=root_runs,
            family_runs_by_root_id=family_runs_by_root_id,
            trace_by_run_id=trace_by_run_id,
            approvals_by_run_id=approvals_by_run_id,
            session_id=session_id,
        )

    def export_run_metrics(
        self,
        *,
        format: str = "json",
        account_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> dict | str:
        """Export aggregated durable run metrics in JSON or Prometheus text."""
        metrics = self.get_run_metrics(
            account_id=account_id,
            session_id=session_id,
            limit=limit,
        )
        if format == "json":
            return metrics
        if format == "prometheus":
            return build_run_metrics_export(metrics)
        raise ValueError(f"Unsupported metrics export format: {format}")

    def _resolve_log_dir(self) -> Path:
        """Return the normalized agent log directory."""
        if self._config is None:
            raise RuntimeError("Config not initialized.")
        return AgentLogger(self._config).log_dir.resolve()

    def _collect_trace_views(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> tuple[
        RunRecord,
        list[RunRecord],
        dict[str, list],
        dict[str, list],
        dict[str, list],
        list[dict],
        list[dict],
    ]:
        """Collect normalized trace artifacts for one run tree."""
        if self._run_store is None or self._trace_store is None:
            raise RuntimeError("Run or trace store not initialized.")

        scoped_run = self._resolve_account_scoped_record(
            self._run_store.get_run(run_id, account_id=account_id),
            self._run_store.get_run(run_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Run does not belong to the current account.",
        )
        if scoped_run is None:
            raise KeyError(f"Run not found: {run_id}")

        root_run, runs = collect_run_family(self._run_store, scoped_run.id)
        if root_run is None:
            raise KeyError(f"Run not found: {run_id}")
        if account_id is not None:
            unauthorized = next((item for item in runs if item.account_id != account_id), None)
            if unauthorized is not None:
                raise PermissionError("Run does not belong to the current account.")

        trace_by_run_id = {
            run.id: self._trace_store.list_events(run.id, account_id=run.account_id)
            for run in runs
        }
        steps_by_run_id = {
            run.id: self._run_store.list_steps(run.id, account_id=run.account_id)
            for run in runs
        }
        artifacts_by_run_id = {
            run.id: self._run_store.list_artifacts(run.id, account_id=run.account_id)
            for run in runs
        }
        timeline = build_trace_timeline(runs, trace_by_run_id)
        tool_calls = build_tool_call_drilldown(runs, trace_by_run_id, steps_by_run_id)
        return (
            root_run,
            runs,
            trace_by_run_id,
            steps_by_run_id,
            artifacts_by_run_id,
            timeline,
            tool_calls,
        )

    def get_run_trace_timeline(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> list[dict]:
        """Return a normalized timeline across one run tree."""
        _, _, _, _, _, timeline, _ = self._collect_trace_views(
            run_id,
            account_id=account_id,
            strict=strict,
        )
        return timeline

    def get_run_trace_tree(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> dict:
        """Return the nested run tree for one run root."""
        root_run, runs, trace_by_run_id, _, artifacts_by_run_id, _, _ = self._collect_trace_views(
            run_id,
            account_id=account_id,
            strict=strict,
        )
        return build_run_tree(root_run, runs, trace_by_run_id, artifacts_by_run_id)

    def get_run_tool_calls(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> list[dict]:
        """Return merged tool/delegate drill-down rows for one run tree."""
        _, _, _, _, _, _, tool_calls = self._collect_trace_views(
            run_id,
            account_id=account_id,
            strict=strict,
        )
        return tool_calls

    def export_run_trace(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> dict:
        """Return one JSON-friendly trace export payload."""
        (
            root_run,
            runs,
            trace_by_run_id,
            steps_by_run_id,
            artifacts_by_run_id,
            timeline,
            tool_calls,
        ) = self._collect_trace_views(
            run_id,
            account_id=account_id,
            strict=strict,
        )
        run_tree = build_run_tree(
            root_run,
            runs,
            trace_by_run_id,
            artifacts_by_run_id,
        )
        return build_trace_export(
            root_run=root_run,
            runs=runs,
            timeline=timeline,
            run_tree=run_tree,
            tool_calls=tool_calls,
            steps_by_run_id=steps_by_run_id,
            artifacts_by_run_id=artifacts_by_run_id,
        )

    def replay_run_trace(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> dict:
        """Return a developer-friendly replay payload for one run tree."""
        (
            root_run,
            runs,
            _trace_by_run_id,
            _steps_by_run_id,
            artifacts_by_run_id,
            timeline,
            tool_calls,
        ) = self._collect_trace_views(
            run_id,
            account_id=account_id,
            strict=strict,
        )
        return build_trace_replay(
            requested_run_id=run_id,
            root_run=root_run,
            runs=runs,
            timeline=timeline,
            tool_calls=tool_calls,
            artifacts_by_run_id=artifacts_by_run_id,
        )

    def locate_run_diagnostics(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> dict:
        """Return disk/database locations related to one run tree."""
        (
            root_run,
            runs,
            _trace_by_run_id,
            _steps_by_run_id,
            artifacts_by_run_id,
            timeline,
            tool_calls,
        ) = self._collect_trace_views(
            run_id,
            account_id=account_id,
            strict=strict,
        )
        if self._session_store is None:
            raise RuntimeError("Session store not initialized.")
        payload = build_run_location_payload(
            root_run=root_run,
            runs=runs,
            session_db_path=self._session_store.db_path,
            log_dir=self._resolve_log_dir(),
            timeline=timeline,
            tool_calls=tool_calls,
            artifacts_by_run_id=artifacts_by_run_id,
        )
        payload["requested_run_id"] = run_id
        return payload

    def list_run_artifacts(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        strict: bool = False,
    ) -> list[dict]:
        """List durable artifacts for one run."""
        if self._run_store is None:
            return []
        run = self._resolve_account_scoped_record(
            self._run_store.get_run(run_id, account_id=account_id),
            self._run_store.get_run(run_id) if account_id is not None else None,
            strict=strict,
            forbidden_message="Run does not belong to the current account.",
        )
        if run is None:
            return []
        runs_to_collect = [run]
        if run.parent_run_id is None:
            runs_to_collect = self._collect_run_subtree(run, account_id=run.account_id)
        artifacts: list[ArtifactRecord] = []
        seen_artifact_ids: set[str] = set()
        for subtree_run in runs_to_collect:
            for artifact in self._run_store.list_artifacts(
                subtree_run.id,
                account_id=run.account_id,
            ):
                if artifact.id in seen_artifact_ids:
                    continue
                seen_artifact_ids.add(artifact.id)
                artifacts.append(artifact)
        artifacts.sort(key=lambda item: (item.created_at, item.id), reverse=True)
        return [
            artifact.model_dump(mode="json")
            for artifact in artifacts
        ]

    def _collect_run_subtree(
        self,
        run: RunRecord,
        *,
        account_id: str | None = None,
    ) -> list[RunRecord]:
        """Return one run together with all descendants in tree order."""
        if self._run_store is None:
            return [run]
        ordered_runs: list[RunRecord] = [run]
        queue: list[RunRecord] = [run]
        while queue:
            current = queue.pop(0)
            children = sorted(
                self._run_store.list_runs(
                    account_id=account_id,
                    parent_run_id=current.id,
                ),
                key=lambda item: (item.created_at, item.id),
            )
            ordered_runs.extend(children)
            queue.extend(children)
        return ordered_runs

    def list_approval_requests(
        self,
        *,
        account_id: str | None = None,
        status: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> list[dict]:
        """List approval requests with optional filters."""
        if self._approval_store is None:
            return []
        requests = self._approval_store.list_requests(
            account_id=account_id,
            status=status,
            run_id=run_id,
        )
        if session_id is not None and self._run_store is not None:
            allowed_run_ids = {
                run.id
                for run in self._run_store.list_runs(
                    account_id=account_id,
                    session_id=session_id,
                )
            }
            requests = [request for request in requests if request.run_id in allowed_run_ids]
        return [request.model_dump(mode="json") for request in requests]

    def start_run(
        self,
        session_id: str,
        goal: str,
        *,
        account_id: str | None = None,
        parent_run_id: str | None = None,
        run_metadata: dict[str, object] | None = None,
        session_policy_override: SessionPolicyOverride | dict[str, object] | None = None,
    ) -> RunRecord:
        """Create and enqueue one durable run for a session."""
        if not self._run_manager:
            raise RuntimeError("Run manager not initialized.")

        session = self.get_session_info(
            session_id,
            account_id=account_id,
            strict=account_id is not None,
        )
        if not session:
            raise KeyError(f"Session not found: {session_id}")

        return self._run_manager.start_run(
            session_id,
            goal,
            parent_run_id=parent_run_id,
            run_metadata=run_metadata,
            session_policy_override=session_policy_override,
        )

    @staticmethod
    def _normalize_attachment_ids(attachment_ids: list[str] | None) -> list[str]:
        """Deduplicate attachment identifiers while preserving order."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in attachment_ids or []:
            attachment_id = str(item).strip()
            if not attachment_id or attachment_id in seen:
                continue
            normalized.append(attachment_id)
            seen.add(attachment_id)
        return normalized

    @staticmethod
    def _uploaded_file_block_from_record(upload: UploadRecord) -> dict[str, object]:
        """Serialize one upload record into a structured message block."""
        return UploadedFileContentBlock(
            upload_id=upload.id,
            original_name=upload.original_name,
            safe_name=upload.safe_name,
            relative_path=upload.relative_path,
            mime_type=upload.mime_type,
            size_bytes=upload.size_bytes,
            checksum=upload.checksum,
        ).model_dump(mode="python", exclude_none=True)

    def _build_chat_message_content(
        self,
        session_id: str,
        message: str,
        *,
        account_id: str | None = None,
        attachment_ids: list[str] | None = None,
    ) -> str | list[dict[str, object]]:
        """Build chat message content from text plus upload references."""
        normalized_message = message.strip()
        normalized_attachment_ids = self._normalize_attachment_ids(attachment_ids)

        blocks: list[dict[str, object]] = []
        if normalized_message:
            blocks.append(TextContentBlock(text=normalized_message).model_dump(mode="python"))

        for attachment_id in normalized_attachment_ids:
            upload = self.get_upload_info(
                attachment_id,
                account_id=account_id,
                strict=account_id is not None,
            )
            if upload is None:
                raise KeyError(f"Upload not found: {attachment_id}")
            if upload.session_id != session_id:
                raise ValueError("attachment_id does not belong to the target session.")
            blocks.append(self._uploaded_file_block_from_record(upload))

        if not blocks:
            raise ValueError("Chat message must include text or attachments.")

        if len(blocks) == 1 and blocks[0]["type"] == "text":
            return normalized_message
        return blocks

    def build_chat_message_content(
        self,
        session_id: str,
        message: str,
        *,
        account_id: str | None = None,
        attachment_ids: list[str] | None = None,
    ) -> str | list[dict[str, object]]:
        """构造可直接写入用户消息的标准内容结构。"""
        return self._build_chat_message_content(
            session_id,
            message,
            account_id=account_id,
            attachment_ids=attachment_ids,
        )

    def start_chat_run(
        self,
        session_id: str,
        message: str,
        *,
        account_id: str | None = None,
        attachment_ids: list[str] | None = None,
    ) -> RunRecord:
        """Create and enqueue one run for a chat turn."""
        user_message_content = self._build_chat_message_content(
            session_id,
            message,
            account_id=account_id,
            attachment_ids=attachment_ids,
        )
        goal = message_content_summary(user_message_content) or message.strip() or "处理已上传文件"

        run_metadata: dict[str, object] | None = None
        if not isinstance(user_message_content, str):
            run_metadata = {"user_message_content": user_message_content}

        return self.start_run(
            session_id,
            goal,
            account_id=account_id,
            run_metadata=run_metadata,
        )

    @staticmethod
    def _append_unique_identifier(
        identifiers: list[object] | None,
        value: str,
    ) -> list[str]:
        """Append one identifier while preserving order and removing blanks."""
        normalized: list[str] = []
        seen: set[str] = set()
        for item in identifiers or []:
            candidate = str(item).strip()
            if not candidate or candidate in seen:
                continue
            normalized.append(candidate)
            seen.add(candidate)

        candidate = value.strip()
        if candidate and candidate not in seen:
            normalized.append(candidate)
        return normalized

    def resolve_approval_request(
        self,
        approval_id: str,
        *,
        account_id: str | None = None,
        status: str,
        decision_notes: str = "",
        decision_scope: str = "once",
    ) -> dict:
        """Resolve one persisted approval request."""
        if self._approval_store is None:
            raise RuntimeError("Approval store not initialized.")

        approval_request = self._resolve_account_scoped_record(
            self._approval_store.get_request(approval_id, account_id=account_id),
            self._approval_store.get_request(approval_id) if account_id is not None else None,
            strict=account_id is not None,
            forbidden_message="Approval request does not belong to the current account.",
        )
        if approval_request is None:
            raise KeyError(f"Approval request not found: {approval_id}")
        if approval_request.status != "pending":
            raise ValueError(
                f"Approval request already resolved: {approval_request.status}"
            )

        normalized_scope = decision_scope.strip().lower() or "once"
        if normalized_scope not in {"once", "run", "template"}:
            raise ValueError(f"Unsupported approval decision scope: {decision_scope}")

        if status == "granted" and normalized_scope in {"run", "template"}:
            if self._run_store is None:
                raise RuntimeError("Run store not initialized.")

            run = self._run_store.get_run(approval_request.run_id, account_id=approval_request.account_id)
            if run is None:
                raise KeyError(f"Run not found: {approval_request.run_id}")

            if normalized_scope == "template":
                if self._agent_store is None:
                    raise RuntimeError("Agent store not initialized.")
                template = self._agent_store.get_agent_template_record(
                    run.agent_template_id,
                    account_id=approval_request.account_id,
                )
                if template is None:
                    raise KeyError(f"Agent template not found: {run.agent_template_id}")
                if template.is_system:
                    raise ValueError(
                        "Permanent approval updates are only supported for custom agent templates."
                    )

                updated_auto_approve_tools = self._append_unique_identifier(
                    template.approval_policy.auto_approve_tools,
                    approval_request.tool_name,
                )
                if updated_auto_approve_tools != template.approval_policy.auto_approve_tools:
                    self._agent_store.update_agent_template(
                        run.agent_template_id,
                        account_id=approval_request.account_id,
                        approval_policy=template.approval_policy.model_copy(
                            update={"auto_approve_tools": updated_auto_approve_tools}
                        ),
                    )

            updated_run_metadata = dict(run.run_metadata)
            updated_run_metadata["approval_auto_grant_tools"] = self._append_unique_identifier(
                updated_run_metadata.get("approval_auto_grant_tools"),
                approval_request.tool_name,
            )
            if updated_run_metadata != run.run_metadata:
                run = run.model_copy(update={"run_metadata": updated_run_metadata})
                self._run_store.update_run(run)

        resolved = approval_request.model_copy(
            update={
                "status": status,
                "resolved_at": utc_now_iso(),
                "decision_notes": decision_notes.strip(),
                "decision_scope": normalized_scope if status == "granted" else None,
            }
        )
        resolved = self._approval_store.update_request(resolved)
        if self._run_manager is not None:
            self._run_manager.notify_approval_resolved(resolved)
        return resolved.model_dump(mode="json")

    async def stream_run(
        self,
        run_id: str,
        *,
        account_id: str | None = None,
        after_event_id: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Stream events for one run."""
        if not self._run_manager:
            yield {"type": "error", "data": {"message": "Run manager not initialized."}}
            return
        if account_id is not None:
            self.get_run_info(run_id, account_id=account_id, strict=True)
        async for event in self._run_manager.stream_run(
            run_id,
            after_sequence=after_event_id,
        ):
            yield event

    async def cancel_run(self, run_id: str, *, account_id: str | None = None) -> dict | None:
        """Cancel one durable run by id."""
        if not self._run_manager:
            return None
        if account_id is not None:
            self.get_run_info(run_id, account_id=account_id, strict=True)
        run = self._run_manager.cancel_run(run_id)
        return run.model_dump(mode="json")

    async def resume_run(self, run_id: str, *, account_id: str | None = None) -> dict | None:
        """Resume one interrupted durable run by id."""
        if not self._run_manager:
            return None
        if account_id is not None:
            self.get_run_info(run_id, account_id=account_id, strict=True)
        run = self._run_manager.resume_run(run_id)
        return run.model_dump(mode="json")

    async def chat(
        self,
        session_id: str,
        message: str,
        *,
        account_id: str | None = None,
        attachment_ids: list[str] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Send a message to a session and stream back events.

        Args:
            session_id: Session ID
            message: User message

        Yields:
            AgentEvent dicts from Agent.run_stream()
        """
        try:
            run = self.start_chat_run(
                session_id,
                message,
                account_id=account_id,
                attachment_ids=attachment_ids,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            yield {"type": "error", "data": {"message": str(exc)}}
            return
        async for event in self.stream_run(run.id):
            yield event

    async def delete_session(self, session_id: str, *, account_id: str | None = None) -> bool:
        """Delete a session and clean up resources."""
        session = self._resolve_account_scoped_record(
            self._session_store.get_session_record(session_id, account_id=account_id),
            self._session_store.get_session_record(session_id) if account_id is not None else None,
            strict=account_id is not None,
            forbidden_message="Session does not belong to the current account.",
        )
        if session is None:
            return False
        await self.interrupt_session(session_id)
        self._runtime_registry.remove(session_id)

        deleted = self._session_store.delete_session(session_id, account_id=session.account_id)
        if session:
            workspace_dir = self._resolve_session_workspace_dir(session)
            shared_context_file = self._get_shared_context_path(
                workspace_dir,
                session_id,
            )
            shared_context_file.unlink(missing_ok=True)
            shutil.rmtree(
                self._get_upload_root(workspace_dir, session_id),
                ignore_errors=True,
            )

        return deleted

    def list_sessions(self, *, account_id: str | None = None) -> list[dict]:
        """List all active sessions with basic info."""
        return self._session_store.list_sessions(account_id=account_id)

    def list_learned_workflow_candidates(
        self,
        *,
        account_id: str | None = None,
        status: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """List reviewable learned-workflow candidates."""
        if self._learned_workflow_store is None:
            return []
        return [
            item.model_dump(mode="json")
            for item in self._learned_workflow_store.list_candidate_records(
                account_id=account_id,
                status=status,
                agent_template_id=agent_id,
                run_id=run_id,
                limit=limit,
            )
        ]

    def approve_learned_workflow_candidate(
        self,
        candidate_id: str,
        *,
        account_id: str | None = None,
        review_notes: str = "",
    ) -> dict:
        """Approve one learned-workflow candidate for later installation."""
        if self._learned_workflow_store is None:
            raise RuntimeError("Learned workflow store not initialized.")
        candidate = self._learned_workflow_store.update_candidate_status(
            candidate_id,
            account_id=account_id,
            status="approved",
            review_notes=review_notes,
        )
        return candidate.model_dump(mode="json")

    def reject_learned_workflow_candidate(
        self,
        candidate_id: str,
        *,
        account_id: str | None = None,
        review_notes: str = "",
    ) -> dict:
        """Reject one learned-workflow candidate."""
        if self._learned_workflow_store is None:
            raise RuntimeError("Learned workflow store not initialized.")
        candidate = self._learned_workflow_store.update_candidate_status(
            candidate_id,
            account_id=account_id,
            status="rejected",
            review_notes=review_notes,
        )
        return candidate.model_dump(mode="json")

    def install_learned_workflow_candidate(
        self,
        candidate_id: str,
        *,
        agent_id: str,
        account_id: str | None = None,
        skill_name: str | None = None,
    ) -> dict:
        """Install one approved learned-workflow candidate into an agent."""
        if self._learned_workflow_store is None or self._agent_store is None:
            raise RuntimeError("Learned workflow installation is unavailable.")

        candidate = self._learned_workflow_store.get_candidate_record(
            candidate_id,
            account_id=account_id,
        )
        if candidate is None:
            raise KeyError(f"Learned workflow candidate not found: {candidate_id}")
        if candidate.status != "approved":
            raise ValueError("Only approved learned workflow candidates can be installed.")

        template = self._agent_store.get_agent_template_record(
            agent_id,
            account_id=account_id,
        )
        if template is None:
            raise KeyError(f"Agent template not found: {agent_id}")
        if template.is_system:
            raise ValueError("Cannot install learned workflows into system agents.")

        normalized_skill_name = self._learned_workflow_store.normalize_skill_name(
            skill_name or candidate.suggested_skill_name,
            fallback=f"learned-workflow-{candidate.id[:8]}",
        )
        skills_dir = self._agent_store.get_agent_skills_dir(
            agent_id,
            account_id=template.account_id,
            is_system=template.is_system,
        )
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill_dir = skills_dir / normalized_skill_name
        skill_file = skill_dir / "SKILL.md"
        if skill_dir.exists():
            raise ValueError(f"Skill already exists: {normalized_skill_name}")

        skill_dir.mkdir(parents=True, exist_ok=False)
        skill_file.write_text(candidate.generated_skill_markdown, encoding="utf-8")
        self._agent_store.refresh_agent_skills_from_directory(
            agent_id,
            account_id=template.account_id,
        )
        updated = self._learned_workflow_store.update_candidate_status(
            candidate.id,
            account_id=candidate.account_id,
            status="installed",
            installed_agent_id=agent_id,
            installed_skill_path=str(skill_file.resolve()),
        )
        return updated.model_dump(mode="json")

    def list_skill_improvement_proposals(
        self,
        *,
        account_id: str | None = None,
        status: str | None = None,
        agent_id: str | None = None,
        skill_name: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        """List reviewable skill-improvement proposals."""
        if self._skill_improvement_store is None:
            return []
        return [
            item.model_dump(mode="json")
            for item in self._skill_improvement_store.list_proposal_records(
                account_id=account_id,
                status=status,
                agent_template_id=agent_id,
                skill_name=skill_name,
                run_id=run_id,
                limit=limit,
            )
        ]

    def approve_skill_improvement_proposal(
        self,
        proposal_id: str,
        *,
        account_id: str | None = None,
        review_notes: str = "",
    ) -> dict:
        """Approve one skill-improvement proposal for later application."""
        if self._skill_improvement_store is None:
            raise RuntimeError("Skill improvement store not initialized.")
        proposal = self._skill_improvement_store.update_proposal_status(
            proposal_id,
            account_id=account_id,
            status="approved",
            review_notes=review_notes,
        )
        return proposal.model_dump(mode="json")

    def reject_skill_improvement_proposal(
        self,
        proposal_id: str,
        *,
        account_id: str | None = None,
        review_notes: str = "",
    ) -> dict:
        """Reject one skill-improvement proposal."""
        if self._skill_improvement_store is None:
            raise RuntimeError("Skill improvement store not initialized.")
        proposal = self._skill_improvement_store.update_proposal_status(
            proposal_id,
            account_id=account_id,
            status="rejected",
            review_notes=review_notes,
        )
        return proposal.model_dump(mode="json")

    def apply_skill_improvement_proposal(
        self,
        proposal_id: str,
        *,
        account_id: str | None = None,
    ) -> dict:
        """Apply one approved skill-improvement proposal to its target SKILL.md."""
        if self._skill_improvement_store is None or self._agent_store is None:
            raise RuntimeError("Skill improvement application is unavailable.")

        proposal = self._skill_improvement_store.get_proposal_record(
            proposal_id,
            account_id=account_id,
        )
        if proposal is None:
            raise KeyError(f"Skill improvement proposal not found: {proposal_id}")
        if proposal.status != "approved":
            raise ValueError("Only approved skill improvement proposals can be applied.")

        template = self._agent_store.get_agent_template_record(
            proposal.agent_template_id,
            account_id=proposal.account_id,
        )
        if template is None:
            raise KeyError(f"Agent template not found: {proposal.agent_template_id}")
        if template.is_system:
            raise ValueError("Cannot mutate system-agent installed skills.")

        skill_file = (
            self._agent_store.get_agent_skills_dir(
                template.id,
                account_id=template.account_id,
                is_system=template.is_system,
            )
            / proposal.skill_name
            / "SKILL.md"
        )
        if not skill_file.exists():
            raise FileNotFoundError(f"Target skill file not found: {skill_file}")

        current_markdown = skill_file.read_text(encoding="utf-8")
        current_version = extract_skill_version(
            current_markdown,
            fallback_name=proposal.skill_name,
        )
        if current_version != proposal.base_version:
            raise ValueError(
                "Skill has changed since this proposal was generated. Please regenerate the proposal."
            )

        skill_file.write_text(proposal.proposed_skill_markdown, encoding="utf-8")
        self._agent_store.refresh_agent_skills_from_directory(
            template.id,
            account_id=template.account_id,
        )
        updated = self._skill_improvement_store.update_proposal_status(
            proposal.id,
            account_id=proposal.account_id,
            status="applied",
            applied_skill_path=str(skill_file.resolve()),
        )
        return updated.model_dump(mode="json")

    def search_session_history(
        self,
        query: str,
        *,
        account_id: str | None = None,
        session_id: str | None = None,
        agent_id: str | None = None,
        exclude_run_id: str | None = None,
        source_types: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
    ) -> list[dict]:
        """Search persisted session history with account-scoped filters."""
        return self._session_store.search_history(
            query,
            account_id=account_id,
            session_id=session_id,
            agent_id=agent_id,
            exclude_run_id=exclude_run_id,
            source_types=source_types,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
        )

    async def cleanup(self):
        """Clean up all sessions and MCP connections."""
        if self._integration_reply_dispatcher is not None:
            await self._integration_reply_dispatcher.shutdown()
        for session_id in self._runtime_registry.session_ids():
            await self.interrupt_session(session_id)
        self._runtime_registry.clear()
        try:
            from .tools.mcp_loader import cleanup_mcp_connections

            await cleanup_mcp_connections()
        except Exception:
            pass
        self._llm_client_cache.clear()

    async def reload_from_disk(self, config_path: str | Path | None = None) -> None:
        """Reload configuration/state after web-based setup updates."""
        await self.cleanup()
        self._config = None
        self._config_path = Path(config_path) if config_path else self._config_path
        self._shared_tools = None
        self._system_prompt = None
        self._llm_client = None
        self._llm_client_cache.clear()
        self._account_store = None
        self._session_store = None
        self._agent_store = None
        self._run_store = None
        self._trace_store = None
        self._approval_store = None
        self._upload_store = None
        self._memory_provider = None
        self._learned_workflow_store = None
        self._skill_improvement_store = None
        self._run_manager = None
        self._runtime_factory = None
        self._integration_reply_dispatcher = None
        self._initialized = False
        self._runtime_ready = False
        self._runtime_registry.clear()
        if self._config_path is None:
            self._config_path = Config.get_default_config_path()
        await self.initialize()


