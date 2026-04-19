"""Tests for runtime dependency setup helpers."""

from pathlib import Path

import pytest

from clavi_agent.runtime_setup import (
    _parse_node_major_version,
    ensure_node_and_npm,
    persist_runtime_tool_paths,
)


def test_persist_runtime_tool_paths_inserts_managed_keys(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                "tools:",
                "  enable_file_tools: true",
                '  mcp_config_path: "mcp.json"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    persist_runtime_tool_paths(
        config_path,
        {
            "node_bin": "/usr/bin/node",
            "npm_bin": "/usr/bin/npm",
            "clawhub_bin": "/home/test/.local/bin/clawhub",
        },
    )

    updated = config_path.read_text(encoding="utf-8")
    assert "  enable_file_tools: true" in updated
    assert '  mcp_config_path: "mcp.json"' in updated
    assert "managed by clavi-agent-setup-runtime" in updated
    assert '  node_bin: "/usr/bin/node"' in updated
    assert '  npm_bin: "/usr/bin/npm"' in updated
    assert '  clawhub_bin: "/home/test/.local/bin/clawhub"' in updated


def test_persist_runtime_tool_paths_updates_existing_values(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "api_key: test-key",
                "tools:",
                '  node_bin: "/old/node"',
                '  clawhub_bin: "/old/clawhub"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    persist_runtime_tool_paths(
        config_path,
        {
            "node_bin": "/new/node",
            "clawhub_bin": "/new/clawhub",
        },
    )

    updated = config_path.read_text(encoding="utf-8")
    assert '  node_bin: "/new/node"' in updated
    assert '  clawhub_bin: "/new/clawhub"' in updated
    assert '  node_bin: "/old/node"' not in updated
    assert '  clawhub_bin: "/old/clawhub"' not in updated


def test_parse_node_major_version_accepts_standard_output():
    assert _parse_node_major_version("v22.15.1") == 22
    assert _parse_node_major_version("22.15.1") == 22


def test_ensure_node_and_npm_rejects_non_22_node(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("clavi_agent.runtime_setup.detect_node_binary", lambda preferred=None: "/usr/bin/node")
    monkeypatch.setattr("clavi_agent.runtime_setup.detect_npm_binary", lambda preferred=None: "/usr/bin/npm")
    monkeypatch.setattr("clavi_agent.runtime_setup._detect_node_major_version", lambda binary: (20, "v20.18.0"))

    with pytest.raises(RuntimeError, match=r"Detected Node\.js v20\.18\.0.*requires Node\.js 22"):
        ensure_node_and_npm()


