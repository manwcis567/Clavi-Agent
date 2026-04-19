"""Runtime tool discovery helpers shared by setup scripts and server code."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def _normalize_candidate_path(candidate: str) -> str:
    """Normalize a candidate executable path."""
    return str(Path(candidate).expanduser()).strip()


def _safe_home_path() -> Path | None:
    """Return the current user's home directory when it can be resolved."""
    try:
        return Path.home()
    except RuntimeError:
        return None


def _dedupe_candidates(candidates: list[str]) -> list[str]:
    """Deduplicate candidates while preserving order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate:
            continue
        normalized = _normalize_candidate_path(candidate)
        if not normalized or normalized in seen:
            continue
        deduped.append(normalized)
        seen.add(normalized)
    return deduped


def detect_node_binary(preferred: str | None = None) -> str | None:
    """Return the absolute node executable path when available."""
    direct_candidates: list[str] = []
    if preferred:
        direct_candidates.append(preferred)

    direct_candidates.extend(
        candidate
        for candidate in (
            shutil.which("node"),
            shutil.which("node.exe"),
        )
        if candidate
    )

    for candidate in _dedupe_candidates(direct_candidates):
        path = Path(candidate)
        if path.is_file():
            return str(path)

    return None


def detect_npm_binary(preferred: str | None = None) -> str | None:
    """Return the absolute npm executable path when available."""
    direct_candidates: list[str] = []
    if preferred:
        direct_candidates.append(preferred)

    direct_candidates.extend(
        candidate
        for candidate in (
            shutil.which("npm"),
            shutil.which("npm.cmd"),
            shutil.which("npm.exe"),
        )
        if candidate
    )

    for candidate in _dedupe_candidates(direct_candidates):
        path = Path(candidate)
        if path.is_file():
            return str(path)

    return None


def read_npm_global_prefix(npm_binary: str | None = None) -> str:
    """Return the configured npm global prefix, if npm is available."""
    npm_candidates = []
    explicit_npm = detect_npm_binary(npm_binary)
    if explicit_npm:
        npm_candidates.append(explicit_npm)

    if not npm_candidates:
        return ""

    for npm_command in npm_candidates:
        for args in (["config", "get", "prefix"], ["prefix", "-g"]):
            try:
                result = subprocess.run(
                    [npm_command, *args],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=True,
                )
            except (FileNotFoundError, OSError, subprocess.CalledProcessError):
                continue

            prefix = result.stdout.strip()
            if prefix and prefix.lower() not in {"undefined", "null"}:
                return prefix

    return ""


def candidate_clawhub_directories(
    *,
    system_name: str | None = None,
    npm_binary: str | None = None,
    npm_prefix: str | None = None,
) -> list[Path]:
    """Build platform-aware directories that may contain the clawhub executable."""
    directories: list[Path] = []
    seen: set[str] = set()
    home_path = _safe_home_path()

    def add_directory(path: Path | None) -> None:
        if path is None:
            return
        try:
            normalized = _normalize_candidate_path(str(path))
        except RuntimeError:
            return
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        directories.append(Path(normalized))

    configured_prefix = (npm_prefix or "").strip()
    if not configured_prefix:
        configured_prefix = os.environ.get("NPM_CONFIG_PREFIX", "").strip()
    if not configured_prefix:
        configured_prefix = read_npm_global_prefix(npm_binary)

    if configured_prefix:
        prefix_path = Path(configured_prefix).expanduser()
        add_directory(prefix_path)
        add_directory(prefix_path / "bin")

    current_system = system_name or platform.system()
    if current_system == "Windows":
        appdata = os.environ.get("APPDATA", "").strip()
        local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
        program_files = os.environ.get("ProgramFiles", "").strip()

        add_directory(Path(appdata) / "npm" if appdata else None)
        add_directory(home_path / "AppData" / "Roaming" / "npm" if home_path else None)
        add_directory(Path(local_appdata) / "Programs" / "nodejs" if local_appdata else None)
        add_directory(Path(program_files) / "nodejs" if program_files else None)
    else:
        add_directory(home_path / ".npm-global" / "bin" if home_path else None)
        add_directory(home_path / ".local" / "bin" if home_path else None)
        add_directory(Path("/usr/local/bin"))
        add_directory(Path("/usr/bin"))

    return directories


def detect_clawhub_binary(
    *,
    clawhub_binary: str | None = None,
    npm_binary: str | None = None,
) -> str | None:
    """Return the absolute clawhub executable path when available."""
    direct_candidates: list[str] = []
    if clawhub_binary:
        direct_candidates.append(clawhub_binary)

    direct_candidates.extend(
        candidate
        for candidate in (
            shutil.which("clawhub"),
            shutil.which("clawhub.cmd"),
            shutil.which("clawhub.ps1"),
            shutil.which("clawhub.exe"),
        )
        if candidate
    )

    for directory in candidate_clawhub_directories(npm_binary=npm_binary):
        for filename in ("clawhub", "clawhub.cmd", "clawhub.ps1", "clawhub.exe"):
            candidate = directory / filename
            if candidate.is_file():
                direct_candidates.append(str(candidate))

    for candidate in _dedupe_candidates(direct_candidates):
        path = Path(candidate)
        if path.is_file():
            return str(path)

    return None


def resolve_clawhub_command_prefix(
    *,
    clawhub_binary: str | None = None,
    npm_binary: str | None = None,
) -> list[str]:
    """Resolve the clawhub executable path with cross-platform fallbacks."""
    resolved_binary = detect_clawhub_binary(
        clawhub_binary=clawhub_binary,
        npm_binary=npm_binary,
    )
    if not resolved_binary:
        raise FileNotFoundError(
            "Unable to locate `clawhub`. Set the CLAWHUB_BIN environment variable "
            "or ensure the clawhub executable is available on PATH. If it was installed "
            "with npm, confirm the global npm prefix/bin directory is accessible."
        )

    if resolved_binary.lower().endswith(".ps1"):
        return [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            resolved_binary,
        ]

    return [resolved_binary]
