"""Configuration management module.

Provides unified configuration loading and management functionality.
"""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from .llm_routing_models import LLMProfileOverride

PRIMARY_USER_CONFIG_DIR = ".clavi-agent"
PRIMARY_ROOT_PASSWORD_ENV = "CLAVI_AGENT_ROOT_PASSWORD"


class RetryConfig(BaseModel):
    """Retry configuration."""

    enabled: bool = True
    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0


class LLMConfig(BaseModel):
    """LLM configuration."""

    api_key: str
    api_base: str = "https://api.minimax.io"
    model: str = "MiniMax-M2"
    provider: str = "anthropic"  # "anthropic" or "openai"
    reasoning_enabled: bool = False
    planner_profile: LLMProfileOverride | None = None
    worker_profile: LLMProfileOverride | None = None
    retry: RetryConfig = Field(default_factory=RetryConfig)


class PromptMemoryConfig(BaseModel):
    """Prompt memory injection limits."""

    profile_char_limit: int = Field(default=600, ge=120)
    profile_summary_char_limit: int = Field(default=300, ge=60)
    profile_field_value_char_limit: int = Field(default=120, ge=20)
    preference_char_limit: int = Field(default=1200, ge=120)
    preference_entry_limit: int = Field(default=6, ge=1)
    preference_entry_char_limit: int = Field(default=180, ge=20)
    retrieved_context_char_limit: int = Field(default=1000, ge=120)
    retrieved_context_entry_limit: int = Field(default=4, ge=1)
    retrieved_context_entry_char_limit: int = Field(default=220, ge=20)
    project_context_char_limit: int = Field(default=1400, ge=120)
    project_context_file_char_limit: int = Field(default=420, ge=40)
    project_context_max_files: int = Field(default=4, ge=1)
    project_context_parent_depth: int = Field(default=4, ge=0)
    project_context_file_byte_limit: int = Field(default=24576, ge=512)
    agent_memory_char_limit: int = Field(default=800, ge=120)
    agent_memory_entry_limit: int = Field(default=4, ge=1)
    agent_memory_entry_char_limit: int = Field(default=160, ge=20)


class AgentConfig(BaseModel):
    """Agent configuration."""

    max_steps: int = 50
    max_concurrent_runs: int = Field(default=4, ge=1)
    run_timeout_seconds: int | None = Field(default=None, ge=1)
    parallel_delegate_limit: int = Field(default=4, ge=1)
    workspace_dir: str = "./workspace"
    system_prompt_path: str = "system_prompt.md"
    log_dir: str = f"~/{PRIMARY_USER_CONFIG_DIR}/log"
    session_store_path: str = "./workspace/.clavi_agent/sessions.db"
    agent_store_path: str = "./workspace/.clavi_agent/agents.db"
    prompt_memory: PromptMemoryConfig = Field(default_factory=PromptMemoryConfig)


class ToolsConfig(BaseModel):
    """Tools configuration."""

    enable_file_tools: bool = True
    enable_bash: bool = True
    enable_note: bool = True
    enable_skills: bool = True
    skills_dir: str = "./skills"
    enable_mcp: bool = True
    mcp_config_path: str = "mcp.json"
    node_bin: str | None = None
    npm_bin: str | None = None
    clawhub_bin: str | None = None


class MemoryProviderConfig(BaseModel):
    """Long-term memory provider configuration."""

    provider: str = "local"
    allow_fallback_to_local: bool = True
    inject_memories: bool = True
    expose_tools: bool = True
    sync_conversation_turns: bool = True
    mcp_server_name: str = "memory"


class FeatureFlagsConfig(BaseModel):
    """Feature rollout switches for phased environment enablement."""

    enable_durable_runs: bool = True
    enable_run_trace: bool = True
    enable_approval_flow: bool = True
    enable_supervisor_mode: bool = True
    enable_worker_model_routing: bool = True
    enable_compact_prompt_memory: bool = True
    enable_session_retrieval: bool = True
    enable_learned_workflow_generation: bool = True
    enable_external_memory_providers: bool = True

    def resolved(self) -> dict[str, bool]:
        """Return effective flags after applying feature dependencies."""
        durable_runs = bool(self.enable_durable_runs)
        return {
            "enable_durable_runs": durable_runs,
            "enable_run_trace": durable_runs and bool(self.enable_run_trace),
            "enable_approval_flow": durable_runs and bool(self.enable_approval_flow),
            "enable_supervisor_mode": bool(self.enable_supervisor_mode),
            "enable_worker_model_routing": bool(self.enable_worker_model_routing),
            "enable_compact_prompt_memory": bool(self.enable_compact_prompt_memory),
            "enable_session_retrieval": bool(self.enable_session_retrieval),
            "enable_learned_workflow_generation": bool(self.enable_learned_workflow_generation),
            "enable_external_memory_providers": bool(self.enable_external_memory_providers),
        }


class AuthConfig(BaseModel):
    """Local account and web-session bootstrap configuration."""

    auto_seed_root: bool = True
    root_username: str = "root"
    root_display_name: str = "Root"
    root_password: str | None = None
    root_password_env: str = PRIMARY_ROOT_PASSWORD_ENV
    web_session_cookie_name: str = "clavi_agent_session"
    web_session_ttl_hours: int = Field(default=12, ge=1, le=24 * 30)

    def resolve_root_password(self) -> str | None:
        """Resolve the root password from config first, then from env."""
        direct_value = str(self.root_password or "").strip()
        if direct_value:
            return direct_value

        env_name = str(self.root_password_env or "").strip()
        if not env_name:
            return None

        env_value = os.environ.get(env_name, "").strip()
        return env_value or None


class Config(BaseModel):
    """Main configuration class."""

    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    memory_provider: MemoryProviderConfig = Field(default_factory=MemoryProviderConfig)
    feature_flags: FeatureFlagsConfig = Field(default_factory=FeatureFlagsConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)

    def has_valid_api_key(self) -> bool:
        """Return whether the configured API key is usable for runtime startup."""
        normalized = str(self.llm.api_key or "").strip()
        return bool(normalized and normalized != "YOUR_API_KEY_HERE")

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        *,
        require_api_key: bool = True,
    ) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            raise ValueError("Configuration file is empty")

        api_key = str(data.get("api_key") or "").strip()
        if require_api_key and not api_key:
            raise ValueError("Configuration file missing required field: api_key")

        if require_api_key and api_key == "YOUR_API_KEY_HERE":
            raise ValueError("Please configure a valid API Key")

        retry_data = data.get("retry", {})
        retry_config = RetryConfig(
            enabled=retry_data.get("enabled", True),
            max_retries=retry_data.get("max_retries", 3),
            initial_delay=retry_data.get("initial_delay", 1.0),
            max_delay=retry_data.get("max_delay", 60.0),
            exponential_base=retry_data.get("exponential_base", 2.0),
        )

        llm_config = LLMConfig(
            api_key=api_key,
            api_base=data.get("api_base", "https://api.minimax.io"),
            model=data.get("model", "MiniMax-M2"),
            provider=data.get("provider", "anthropic"),
            reasoning_enabled=data.get("reasoning_enabled", False),
            planner_profile=(
                LLMProfileOverride.model_validate(data.get("planner_profile") or {})
                if data.get("planner_profile") is not None
                else None
            ),
            worker_profile=(
                LLMProfileOverride.model_validate(data.get("worker_profile") or {})
                if data.get("worker_profile") is not None
                else None
            ),
            retry=retry_config,
        )

        agent_data = data.get("agent", {})
        prompt_memory_data = agent_data.get("prompt_memory", {})

        agent_config = AgentConfig(
            max_steps=data.get("max_steps", 50),
            max_concurrent_runs=data.get("max_concurrent_runs", 4),
            run_timeout_seconds=data.get("run_timeout_seconds"),
            parallel_delegate_limit=data.get("parallel_delegate_limit", 4),
            workspace_dir=data.get("workspace_dir", "./workspace"),
            system_prompt_path=data.get("system_prompt_path", "system_prompt.md"),
            log_dir=data.get("log_dir", f"~/{PRIMARY_USER_CONFIG_DIR}/log"),
            session_store_path=data.get(
                "session_store_path",
                "./workspace/.clavi_agent/sessions.db",
            ),
            prompt_memory=PromptMemoryConfig(
                profile_char_limit=prompt_memory_data.get("profile_char_limit", 600),
                profile_summary_char_limit=prompt_memory_data.get(
                    "profile_summary_char_limit",
                    300,
                ),
                profile_field_value_char_limit=prompt_memory_data.get(
                    "profile_field_value_char_limit",
                    120,
                ),
                preference_char_limit=prompt_memory_data.get("preference_char_limit", 1200),
                preference_entry_limit=prompt_memory_data.get("preference_entry_limit", 6),
                preference_entry_char_limit=prompt_memory_data.get(
                    "preference_entry_char_limit",
                    180,
                ),
                retrieved_context_char_limit=prompt_memory_data.get(
                    "retrieved_context_char_limit",
                    1000,
                ),
                retrieved_context_entry_limit=prompt_memory_data.get(
                    "retrieved_context_entry_limit",
                    4,
                ),
                retrieved_context_entry_char_limit=prompt_memory_data.get(
                    "retrieved_context_entry_char_limit",
                    220,
                ),
                project_context_char_limit=prompt_memory_data.get(
                    "project_context_char_limit",
                    1400,
                ),
                project_context_file_char_limit=prompt_memory_data.get(
                    "project_context_file_char_limit",
                    420,
                ),
                project_context_max_files=prompt_memory_data.get(
                    "project_context_max_files",
                    4,
                ),
                project_context_parent_depth=prompt_memory_data.get(
                    "project_context_parent_depth",
                    4,
                ),
                project_context_file_byte_limit=prompt_memory_data.get(
                    "project_context_file_byte_limit",
                    24576,
                ),
                agent_memory_char_limit=prompt_memory_data.get("agent_memory_char_limit", 800),
                agent_memory_entry_limit=prompt_memory_data.get("agent_memory_entry_limit", 4),
                agent_memory_entry_char_limit=prompt_memory_data.get(
                    "agent_memory_entry_char_limit",
                    160,
                ),
            ),
        )

        tools_data = data.get("tools", {})
        tools_config = ToolsConfig(
            enable_file_tools=tools_data.get("enable_file_tools", True),
            enable_bash=tools_data.get("enable_bash", True),
            enable_note=tools_data.get("enable_note", True),
            enable_skills=tools_data.get("enable_skills", True),
            skills_dir=tools_data.get("skills_dir", "./skills"),
            enable_mcp=tools_data.get("enable_mcp", True),
            mcp_config_path=tools_data.get("mcp_config_path", "mcp.json"),
            node_bin=tools_data.get("node_bin"),
            npm_bin=tools_data.get("npm_bin"),
            clawhub_bin=tools_data.get("clawhub_bin"),
        )
        memory_provider_data = data.get("memory_provider", {})
        memory_provider = MemoryProviderConfig(
            provider=memory_provider_data.get("provider", "local"),
            allow_fallback_to_local=memory_provider_data.get(
                "allow_fallback_to_local",
                True,
            ),
            inject_memories=memory_provider_data.get("inject_memories", True),
            expose_tools=memory_provider_data.get("expose_tools", True),
            sync_conversation_turns=memory_provider_data.get(
                "sync_conversation_turns",
                True,
            ),
            mcp_server_name=memory_provider_data.get("mcp_server_name", "memory"),
        )
        feature_flags_data = data.get("feature_flags", {})
        feature_flags = FeatureFlagsConfig(
            enable_durable_runs=feature_flags_data.get("enable_durable_runs", True),
            enable_run_trace=feature_flags_data.get("enable_run_trace", True),
            enable_approval_flow=feature_flags_data.get("enable_approval_flow", True),
            enable_supervisor_mode=feature_flags_data.get("enable_supervisor_mode", True),
            enable_worker_model_routing=feature_flags_data.get(
                "enable_worker_model_routing",
                True,
            ),
            enable_compact_prompt_memory=feature_flags_data.get(
                "enable_compact_prompt_memory",
                True,
            ),
            enable_session_retrieval=feature_flags_data.get(
                "enable_session_retrieval",
                True,
            ),
            enable_learned_workflow_generation=feature_flags_data.get(
                "enable_learned_workflow_generation",
                True,
            ),
            enable_external_memory_providers=feature_flags_data.get(
                "enable_external_memory_providers",
                True,
            ),
        )
        auth_data = data.get("auth", {})
        auth_config = AuthConfig(
            auto_seed_root=auth_data.get("auto_seed_root", True),
            root_username=auth_data.get("root_username", "root"),
            root_display_name=auth_data.get("root_display_name", "Root"),
            root_password=auth_data.get("root_password"),
            root_password_env=auth_data.get(
                "root_password_env",
                PRIMARY_ROOT_PASSWORD_ENV,
            ),
            web_session_cookie_name=auth_data.get(
                "web_session_cookie_name",
                "clavi_agent_session",
            ),
            web_session_ttl_hours=auth_data.get("web_session_ttl_hours", 12),
        )

        if "agent_store_path" in agent_data:
            agent_config.agent_store_path = agent_data["agent_store_path"]

        return cls(
            llm=llm_config,
            agent=agent_config,
            tools=tools_config,
            memory_provider=memory_provider,
            feature_flags=feature_flags,
            auth=auth_config,
        )

    def get_feature_flags(self) -> dict[str, bool]:
        """Return effective feature flags for API consumers and runtime gates."""
        return self.feature_flags.resolved()

    def get_system_agents(self, available_skills: list[dict] | None = None) -> list[dict]:
        """Get the default built-in system agents for initialization."""
        prompt_path = self.find_config_file(self.agent.system_prompt_path)
        if prompt_path and prompt_path.exists():
            prompt_content = prompt_path.read_text(encoding="utf-8")
        else:
            prompt_content = (
                "You are Clavi Agent, an intelligent assistant powered by MiniMax M2 "
                "that can help users complete various tasks."
            )

        tools = []
        if self.tools.enable_bash:
            tools.extend(["BashTool", "BashOutputTool", "BashKillTool"])
        if self.tools.enable_file_tools:
            tools.extend(["ReadTool", "WriteTool", "EditTool"])
        if self.tools.enable_note:
            tools.extend(
                [
                    "SessionNoteTool",
                    "RecallNoteTool",
                    "SearchMemoryTool",
                    "SearchSessionHistoryTool",
                ]
            )

        tools.extend(["ShareContextTool", "ReadSharedContextTool"])

        return [
            {
                "id": "system-default-agent",
                "name": "Built-in Assistant",
                "description": (
                    "Default system agent with the standard workspace tools and any "
                    "configured installed skills."
                ),
                "system_prompt": prompt_content,
                "skills": available_skills or [],
                "tools": tools,
                "workspace_type": "isolated",
                "workspace_policy": {
                    "mode": "isolated",
                    "allow_session_override": True,
                    "readable_roots": [],
                    "writable_roots": [],
                    "read_only_tools": [],
                    "disabled_tools": [],
                    "allowed_shell_command_prefixes": [],
                    "allowed_network_domains": [],
                },
                "approval_policy": {
                    "mode": "default",
                    "require_approval_tools": [],
                    "auto_approve_tools": [],
                    "require_approval_risk_levels": [],
                    "require_approval_risk_categories": [],
                    "notes": "",
                },
                "run_policy": {
                    "timeout_seconds": self.agent.run_timeout_seconds,
                    "max_concurrent_runs": self.agent.max_concurrent_runs,
                },
                "delegation_policy": {
                    "mode": "prefer_delegate",
                    "require_delegate_for_write_actions": False,
                    "require_delegate_for_shell": False,
                    "require_delegate_for_stateful_mcp": False,
                    "allow_main_agent_read_tools": True,
                    "verify_worker_output": True,
                    "prefer_batch_delegate": True,
                },
            }
        ]

    @staticmethod
    def get_package_dir() -> Path:
        """Get the package installation directory."""
        return Path(__file__).parent

    @classmethod
    def find_config_file(cls, filename: str) -> Path | None:
        """Find a configuration file using the standard priority order."""
        dev_config = Path.cwd() / "clavi_agent" / "config" / filename
        if dev_config.exists():
            return dev_config

        user_config = Path.home() / PRIMARY_USER_CONFIG_DIR / "config" / filename
        if user_config.exists():
            return user_config

        package_config = cls.get_package_dir() / "config" / filename
        if package_config.exists():
            return package_config

        return None

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default config file path with priority search."""
        config_path = cls.find_config_file("config.yaml")
        if config_path:
            return config_path

        return cls.get_package_dir() / "config" / "config.yaml"

    @classmethod
    def get_bootstrap_config_path(cls) -> Path:
        """Return the preferred writable config path for first-run bootstrap."""
        dev_config_dir = Path.cwd() / "clavi_agent" / "config"
        if dev_config_dir.exists():
            return dev_config_dir / "config.yaml"

        return Path.home() / PRIMARY_USER_CONFIG_DIR / "config" / "config.yaml"

    @classmethod
    def ensure_bootstrap_config(cls, config_path: str | Path | None = None) -> Path:
        """Create a starter config from the bundled example when config.yaml is missing."""
        resolved_path = Path(config_path) if config_path else cls.get_bootstrap_config_path()
        if resolved_path.exists():
            return resolved_path

        example_path = cls.get_package_dir() / "config" / "config-example.yaml"
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        if example_path.exists():
            resolved_path.write_text(example_path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            resolved_path.write_text(
                "\n".join(
                    [
                        'api_key: ""',
                        'api_base: "https://api.minimax.io"',
                        'model: "MiniMax-M2"',
                        'provider: "anthropic"',
                        "reasoning_enabled: false",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        return resolved_path

    @classmethod
    def read_raw_config(cls, config_path: str | Path | None = None) -> tuple[Path, dict]:
        """Read the raw YAML mapping for config management flows."""
        resolved_path = Path(config_path) if config_path else cls.get_default_config_path()
        if not resolved_path.exists():
            resolved_path = cls.ensure_bootstrap_config(resolved_path)
        with open(resolved_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("Configuration file format is invalid.")
        return resolved_path, data

    @classmethod
    def write_raw_config(cls, config_path: str | Path, data: dict) -> Path:
        """Persist a raw YAML mapping back to disk."""
        resolved_path = Path(config_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
        return resolved_path

    @classmethod
    def get_tool_path_overrides(cls, config_path: str | Path | None = None) -> dict[str, str]:
        """Read persisted runtime tool paths without requiring a valid API key."""
        resolved_path = Path(config_path) if config_path else cls.get_default_config_path()
        if not resolved_path.exists():
            return {}

        try:
            with open(resolved_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return {}

        tools_data = data.get("tools", {})
        if not isinstance(tools_data, dict):
            return {}

        overrides: dict[str, str] = {}
        for key in ("node_bin", "npm_bin", "clawhub_bin"):
            value = tools_data.get(key)
            if not value:
                continue
            normalized = str(value).strip()
            if normalized:
                overrides[key] = normalized

        return overrides


