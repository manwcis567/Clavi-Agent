"""Tests for parallel_delegate_limit config parsing."""

from pathlib import Path

from clavi_agent.config import (
    AgentConfig,
    Config,
    FeatureFlagsConfig,
    MemoryProviderConfig,
    PromptMemoryConfig,
)


def test_agent_config_parallel_delegate_limit_default():
    config = AgentConfig()
    assert config.max_concurrent_runs == 4
    assert config.run_timeout_seconds is None
    assert config.parallel_delegate_limit == 4
    assert config.prompt_memory == PromptMemoryConfig()


def test_memory_provider_config_defaults():
    config = MemoryProviderConfig()
    assert config.provider == "local"
    assert config.allow_fallback_to_local is True
    assert config.inject_memories is True
    assert config.expose_tools is True
    assert config.sync_conversation_turns is True
    assert config.mcp_server_name == "memory"


def test_feature_flags_resolve_dependencies():
    flags = FeatureFlagsConfig(
        enable_durable_runs=False,
        enable_run_trace=True,
        enable_approval_flow=True,
    )

    assert flags.resolved() == {
        "enable_durable_runs": False,
        "enable_run_trace": False,
        "enable_approval_flow": False,
        "enable_supervisor_mode": True,
        "enable_worker_model_routing": True,
        "enable_compact_prompt_memory": True,
        "enable_session_retrieval": True,
        "enable_learned_workflow_generation": True,
        "enable_external_memory_providers": True,
    }


def test_config_from_yaml_parses_parallel_delegate_limit(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                "max_concurrent_runs: 3",
                "run_timeout_seconds: 45",
                "parallel_delegate_limit: 7",
                "tools:",
                "  enable_file_tools: false",
                "  enable_bash: false",
                "  enable_note: false",
                "  enable_skills: false",
                "  enable_mcp: false",
            ]
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_path)
    assert config.agent.max_concurrent_runs == 3
    assert config.agent.run_timeout_seconds == 45
    assert config.agent.parallel_delegate_limit == 7


def test_config_from_yaml_uses_parallel_delegate_limit_default(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("api_key: test-key\n", encoding="utf-8")

    config = Config.from_yaml(config_path)
    assert config.agent.max_concurrent_runs == 4
    assert config.agent.run_timeout_seconds is None
    assert config.agent.parallel_delegate_limit == 4
    assert config.agent.prompt_memory.agent_memory_entry_limit == 4
    assert config.agent.prompt_memory.project_context_max_files == 4


def test_config_from_yaml_parses_prompt_memory_limits(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                "agent:",
                "  prompt_memory:",
                "    profile_char_limit: 420",
                "    preference_entry_limit: 3",
                "    project_context_max_files: 2",
                "    agent_memory_entry_char_limit: 88",
            ]
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_path)
    assert config.agent.prompt_memory.profile_char_limit == 420
    assert config.agent.prompt_memory.preference_entry_limit == 3
    assert config.agent.prompt_memory.project_context_max_files == 2
    assert config.agent.prompt_memory.agent_memory_entry_char_limit == 88


def test_config_from_yaml_parses_feature_flags(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                "feature_flags:",
                "  enable_durable_runs: false",
                "  enable_run_trace: true",
                "  enable_approval_flow: true",
                "  enable_supervisor_mode: false",
                "  enable_worker_model_routing: false",
                "  enable_compact_prompt_memory: false",
                "  enable_session_retrieval: false",
                "  enable_learned_workflow_generation: false",
                "  enable_external_memory_providers: false",
            ]
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_path)
    assert config.get_feature_flags() == {
        "enable_durable_runs": False,
        "enable_run_trace": False,
        "enable_approval_flow": False,
        "enable_supervisor_mode": False,
        "enable_worker_model_routing": False,
        "enable_compact_prompt_memory": False,
        "enable_session_retrieval": False,
        "enable_learned_workflow_generation": False,
        "enable_external_memory_providers": False,
    }


def test_config_from_yaml_parses_memory_provider(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                "memory_provider:",
                "  provider: mcp",
                "  allow_fallback_to_local: false",
                "  inject_memories: false",
                "  expose_tools: false",
                "  sync_conversation_turns: false",
                "  mcp_server_name: legacy-memory",
            ]
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_path)
    assert config.memory_provider.provider == "mcp"
    assert config.memory_provider.allow_fallback_to_local is False
    assert config.memory_provider.inject_memories is False
    assert config.memory_provider.expose_tools is False
    assert config.memory_provider.sync_conversation_turns is False
    assert config.memory_provider.mcp_server_name == "legacy-memory"


def test_config_from_yaml_parses_reasoning_enabled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                'provider: "openai"',
                "reasoning_enabled: true",
            ]
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_path)
    assert config.llm.reasoning_enabled is True


def test_config_from_yaml_parses_runtime_tool_paths(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                "tools:",
                '  node_bin: "/usr/bin/node"',
                '  npm_bin: "/usr/bin/npm"',
                '  clawhub_bin: "/home/test/.local/bin/clawhub"',
            ]
        ),
        encoding="utf-8",
    )

    config = Config.from_yaml(config_path)
    assert config.tools.node_bin == "/usr/bin/node"
    assert config.tools.npm_bin == "/usr/bin/npm"
    assert config.tools.clawhub_bin == "/home/test/.local/bin/clawhub"


def test_get_tool_path_overrides_does_not_require_valid_api_key(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                'api_key: "YOUR_API_KEY_HERE"',
                "tools:",
                '  npm_bin: "/usr/bin/npm"',
                '  clawhub_bin: "/opt/bin/clawhub"',
            ]
        ),
        encoding="utf-8",
    )

    overrides = Config.get_tool_path_overrides(config_path)
    assert overrides == {
        "npm_bin": "/usr/bin/npm",
        "clawhub_bin": "/opt/bin/clawhub",
    }

