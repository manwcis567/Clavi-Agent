"""Runtime dependency bootstrap for node/npm/clawhub."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .config import Config
from .runtime_tools import detect_clawhub_binary, detect_node_binary, detect_npm_binary

RUNTIME_TOOL_KEYS = ("node_bin", "npm_bin", "clawhub_bin")
MANAGED_TOOLS_COMMENT = "  # External runtime tools (managed by clavi-agent-setup-runtime)"
REQUIRED_NODE_MAJOR = 22


def _command_display(command: list[str]) -> str:
    """Return a user-facing command string."""
    return " ".join(command)


def _run_command(command: list[str], description: str) -> None:
    """Run a setup command and stream output to the terminal."""
    print(f"[setup] {description}")
    print(f"[setup] -> {_command_display(command)}")
    subprocess.run(command, check=True)


def _command_exists(command: str) -> bool:
    """Return whether a command exists on PATH."""
    return shutil.which(command) is not None


def _read_node_version(node_binary: str) -> str | None:
    """Return the raw `node --version` output when available."""
    try:
        result = subprocess.run(
            [node_binary, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (FileNotFoundError, OSError, subprocess.CalledProcessError):
        return None

    version = result.stdout.strip() or result.stderr.strip()
    return version or None


def _parse_node_major_version(version_output: str | None) -> int | None:
    """Parse a Node.js major version from `node --version` output."""
    if not version_output:
        return None
    match = re.search(r"[vV]?(?P<major>\d+)(?:\.\d+){0,2}", version_output.strip())
    if not match:
        return None
    try:
        return int(match.group("major"))
    except (TypeError, ValueError):
        return None


def _detect_node_major_version(node_binary: str | None) -> tuple[int | None, str | None]:
    """Return the detected Node.js major version and raw version string."""
    if not node_binary:
        return None, None
    raw_version = _read_node_version(node_binary)
    return _parse_node_major_version(raw_version), raw_version


def _node_version_requirement_message(raw_version: str | None = None) -> str:
    """Build a user-facing Node.js version requirement error message."""
    if raw_version:
        return (
            f"Detected Node.js {raw_version}, but `clawhub` requires Node.js "
            f"{REQUIRED_NODE_MAJOR}. Install Node.js {REQUIRED_NODE_MAJOR} and rerun "
            "`clavi-agent-setup-runtime`."
        )
    return (
        f"Unable to determine the installed Node.js version, but `clawhub` requires "
        f"Node.js {REQUIRED_NODE_MAJOR}. Install Node.js {REQUIRED_NODE_MAJOR} and rerun "
        "`clavi-agent-setup-runtime`."
    )


def _with_optional_sudo(command: list[str]) -> list[str]:
    """Prefix a command with sudo when appropriate."""
    if platform.system() == "Windows":
        return command
    if os.environ.get("MINI_AGENT_SETUP_NO_SUDO", "").strip() == "1":
        return command
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() == 0:
        return command
    if _command_exists("sudo") or os.environ.get("SUDO_COMMAND"):
        return ["sudo", *command]
    return command


def _node_install_steps() -> list[tuple[str, list[str]]]:
    """Return platform-appropriate commands for installing node/npm."""
    system_name = platform.system()

    if system_name == "Windows":
        if _command_exists("winget"):
            return [
                (
                    "Installing Node.js LTS with winget",
                    [
                        "winget",
                        "install",
                        "-e",
                        "--id",
                        "OpenJS.NodeJS.LTS",
                        "--accept-source-agreements",
                        "--accept-package-agreements",
                    ],
                )
            ]
        if _command_exists("choco"):
            return [
                (
                    "Installing Node.js LTS with Chocolatey",
                    ["choco", "install", "nodejs-lts", "-y"],
                )
            ]
        return []

    if system_name == "Darwin":
        if _command_exists("brew"):
            return [("Installing Node.js with Homebrew", ["brew", "install", "node"])]
        return []

    if _command_exists("apt-get"):
        return [
            ("Refreshing apt package index", _with_optional_sudo(["apt-get", "update"])),
            (
                "Installing nodejs and npm with apt-get",
                _with_optional_sudo(["apt-get", "install", "-y", "nodejs", "npm"]),
            ),
        ]
    if _command_exists("dnf"):
        return [
            (
                "Installing nodejs and npm with dnf",
                _with_optional_sudo(["dnf", "install", "-y", "nodejs", "npm"]),
            )
        ]
    if _command_exists("yum"):
        return [
            (
                "Installing nodejs and npm with yum",
                _with_optional_sudo(["yum", "install", "-y", "nodejs", "npm"]),
            )
        ]
    if _command_exists("pacman"):
        return [
            (
                "Installing nodejs and npm with pacman",
                _with_optional_sudo(["pacman", "-Sy", "--noconfirm", "nodejs", "npm"]),
            )
        ]
    if _command_exists("zypper"):
        return [
            (
                "Installing nodejs and npm with zypper",
                _with_optional_sudo(["zypper", "--non-interactive", "install", "nodejs", "npm"]),
            )
        ]
    if _command_exists("brew"):
        return [("Installing Node.js with Homebrew", ["brew", "install", "node"])]

    return []


def _manual_node_install_message() -> str:
    """Return a platform-specific fallback message when automatic install is unavailable."""
    system_name = platform.system()
    if system_name == "Windows":
        return (
            f"Unable to auto-install Node.js {REQUIRED_NODE_MAJOR}. Install Node.js "
            f"{REQUIRED_NODE_MAJOR} first, then rerun `clavi-agent-setup-runtime`."
        )
    if system_name == "Darwin":
        return (
            f"Unable to auto-install Node.js {REQUIRED_NODE_MAJOR}. Install Homebrew and run "
            f"`brew install node@{REQUIRED_NODE_MAJOR}`, then rerun "
            "`clavi-agent-setup-runtime`."
        )
    return (
        f"Unable to auto-install Node.js {REQUIRED_NODE_MAJOR}. Install Node.js "
        f"{REQUIRED_NODE_MAJOR} and npm with your package manager, then rerun "
        "`clavi-agent-setup-runtime`."
    )


def ensure_node_and_npm(*, preferred_node: str | None = None, preferred_npm: str | None = None, install_missing: bool = True) -> tuple[str, str]:
    """Ensure node and npm are available, optionally installing them."""
    node_binary = detect_node_binary(preferred_node)
    npm_binary = detect_npm_binary(preferred_npm)
    detected_major, raw_version = _detect_node_major_version(node_binary)
    if node_binary and npm_binary and detected_major == REQUIRED_NODE_MAJOR:
        return node_binary, npm_binary
    if node_binary and detected_major != REQUIRED_NODE_MAJOR:
        raise RuntimeError(_node_version_requirement_message(raw_version))

    if not install_missing:
        raise RuntimeError(
            f"Node.js {REQUIRED_NODE_MAJOR} or npm is missing."
            if not node_binary
            else "npm is missing."
        )

    install_steps = _node_install_steps()
    if not install_steps:
        raise RuntimeError(_manual_node_install_message())

    for description, command in install_steps:
        _run_command(command, description)

    node_binary = detect_node_binary(preferred_node)
    npm_binary = detect_npm_binary(preferred_npm)
    detected_major, raw_version = _detect_node_major_version(node_binary)
    if not node_binary or not npm_binary:
        raise RuntimeError(
            "Node.js/npm installation finished, but the executables are still not visible. "
            "Open a new terminal and rerun `clavi-agent-setup-runtime`."
        )
    if detected_major != REQUIRED_NODE_MAJOR:
        raise RuntimeError(_node_version_requirement_message(raw_version))

    return node_binary, npm_binary


def ensure_clawhub(*, npm_binary: str, preferred_clawhub: str | None = None, install_missing: bool = True) -> str:
    """Ensure clawhub is available, optionally installing it with npm."""
    clawhub_binary = detect_clawhub_binary(
        clawhub_binary=preferred_clawhub,
        npm_binary=npm_binary,
    )
    if clawhub_binary:
        return clawhub_binary

    if not install_missing:
        raise RuntimeError("clawhub is missing.")

    package_name = os.environ.get("CLAWHUB_NPM_PACKAGE", "clawhub").strip() or "clawhub"
    install_commands = [
        ["npm", "install", "-g", package_name],
    ]
    if npm_binary.lower() != "npm":
        install_commands.insert(0, [npm_binary, "install", "-g", package_name])

    if platform.system() != "Windows":
        install_commands.append(_with_optional_sudo([npm_binary, "install", "-g", package_name]))

    last_error: Exception | None = None
    seen_commands: set[tuple[str, ...]] = set()
    for command in install_commands:
        command_tuple = tuple(command)
        if command_tuple in seen_commands:
            continue
        seen_commands.add(command_tuple)
        try:
            _run_command(command, "Installing clawhub with npm")
            last_error = None
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc

    clawhub_binary = detect_clawhub_binary(
        clawhub_binary=preferred_clawhub,
        npm_binary=npm_binary,
    )
    if clawhub_binary:
        return clawhub_binary

    raise RuntimeError(
        "Failed to install `clawhub`."
        if last_error is None
        else f"Failed to install `clawhub`: {last_error}"
    )


def _format_yaml_scalar(value: str) -> str:
    """Format a string as a YAML-safe scalar."""
    return json.dumps(value)


def persist_runtime_tool_paths(config_path: str | Path, tool_paths: dict[str, str]) -> None:
    """Persist runtime tool paths into the config file's tools section."""
    resolved_path = Path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {resolved_path}")

    raw_lines = resolved_path.read_text(encoding="utf-8").splitlines()
    tools_index = next(
        (idx for idx, line in enumerate(raw_lines) if line.strip() == "tools:" and not line.startswith(" ")),
        None,
    )

    updates = {key: value for key, value in tool_paths.items() if key in RUNTIME_TOOL_KEYS and value}

    if tools_index is None:
        if raw_lines and raw_lines[-1].strip():
            raw_lines.append("")
        raw_lines.append("tools:")
        raw_lines.append(MANAGED_TOOLS_COMMENT)
        for key in RUNTIME_TOOL_KEYS:
            if key in updates:
                raw_lines.append(f"  {key}: {_format_yaml_scalar(updates[key])}")
        resolved_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
        return

    section_end = len(raw_lines)
    for idx in range(tools_index + 1, len(raw_lines)):
        line = raw_lines[idx]
        stripped = line.strip()
        if not stripped:
            continue
        if not line.startswith(" ") and not line.startswith("#"):
            section_end = idx
            break

    for key in RUNTIME_TOOL_KEYS:
        if key not in updates:
            continue
        replacement = f"  {key}: {_format_yaml_scalar(updates[key])}"
        key_pattern = re.compile(rf"^\s{{2}}{re.escape(key)}\s*:")
        replaced = False
        for idx in range(tools_index + 1, section_end):
            if key_pattern.match(raw_lines[idx]):
                raw_lines[idx] = replacement
                replaced = True
                break
        if not replaced:
            insert_at = section_end
            if MANAGED_TOOLS_COMMENT not in raw_lines[tools_index + 1 : section_end]:
                raw_lines.insert(insert_at, MANAGED_TOOLS_COMMENT)
                section_end += 1
                insert_at += 1
            raw_lines.insert(insert_at, replacement)
            section_end += 1

    resolved_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")


def bootstrap_runtime_tools(config_path: str | Path, install_missing: bool = True) -> dict[str, str]:
    """Ensure runtime tools are installed and persist their absolute paths."""
    overrides = Config.get_tool_path_overrides(config_path)
    node_binary, npm_binary = ensure_node_and_npm(
        preferred_node=overrides.get("node_bin"),
        preferred_npm=overrides.get("npm_bin"),
        install_missing=install_missing,
    )
    clawhub_binary = ensure_clawhub(
        npm_binary=npm_binary,
        preferred_clawhub=overrides.get("clawhub_bin"),
        install_missing=install_missing,
    )

    resolved = {
        "node_bin": node_binary,
        "npm_bin": npm_binary,
        "clawhub_bin": clawhub_binary,
    }
    persist_runtime_tool_paths(config_path, resolved)
    return resolved


def parse_args() -> argparse.Namespace:
    """Parse runtime setup command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Install Clavi Agent runtime dependencies and persist their paths.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (defaults to the standard config search path)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate tool availability; do not attempt installation.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint for runtime dependency setup."""
    args = parse_args()
    config_path = Path(args.config) if args.config else Config.get_default_config_path()

    try:
        resolved = bootstrap_runtime_tools(config_path, install_missing=not args.check_only)
    except Exception as exc:
        print(f"[setup] Error: {exc}", file=sys.stderr)
        return 1

    print("[setup] Runtime tool bootstrap completed.")
    print(f"[setup] Config file: {config_path}")
    for key in RUNTIME_TOOL_KEYS:
        print(f"[setup] {key}: {resolved[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

