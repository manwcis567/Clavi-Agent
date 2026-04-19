"""Helpers for unified tool execution metadata, risk assessment, and artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import mimetypes
from pathlib import Path
import re
import time
from typing import Any

from .agent_template_models import ApprovalPolicy, DelegationPolicy, WorkspacePolicy
from .file_previews import guess_preview_kind
from .sqlite_schema import utc_now_iso
from .tools.base import Tool, ToolResult

_SHELL_WRITE_PATTERN = re.compile(
    r"(^|\s)(rm|mv|cp|mkdir|touch|tee|sed|git\s+commit|git\s+push|git\s+tag|git\s+reset|"
    r"Set-Content|Add-Content|Remove-Item|Move-Item|Copy-Item|New-Item|Out-File)(\s|$)|"
    r"(>>?)|(\|\s*tee\b)",
    re.IGNORECASE,
)
_NETWORK_PATTERN = re.compile(
    r"\b(curl|wget|Invoke-WebRequest|Invoke-RestMethod|git\s+clone|git\s+pull|git\s+fetch|"
    r"git\s+push|ssh|scp|sftp)\b",
    re.IGNORECASE,
)
_CREDENTIAL_PATTERN = re.compile(
    r"\b(token|api[_-]?key|secret|password|passwd|credential|aws_access_key|ssh-key|"
    r"private[_-]?key|gpg)\b",
    re.IGNORECASE,
)
_NETWORK_HOST_PATTERN = re.compile(
    r"(?:(?:https?|ssh|wss?)://|git@)([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?::\d+)?",
    re.IGNORECASE,
)
_PLAIN_HOST_PATTERN = re.compile(r"\b([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")
_DELIVERABLE_EXTENSIONS = frozenset(
    {
        "md",
        "markdown",
        "docx",
        "pdf",
        "pptx",
        "xlsx",
        "csv",
        "html",
        "htm",
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "svg",
    }
)
_DELIVERABLE_EXTENSION_PATTERN = "|".join(sorted(_DELIVERABLE_EXTENSIONS))
_SHELL_OUTPUT_FLAG_PATTERN = re.compile(
    rf"""
    (?:
        ^|[\s;|]
    )
    (?:
        -o
        |--?output
        |--?out
        |--?outfile
        |--?output-file
        |--?output_path
        |--?output-path
        |--?outputpath
        |--?outpath
        |--?export
        |--?export-file
        |--?destination
        |--?dest
        |--?save(?:-as)?
        |-OutputPath
        |-OutFile
    )
    \s*(?:=|\s+)
    (?P<path>
        "[^"]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})"
        |'[^']+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})'
        |[^\s"'`;|]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SHELL_REDIRECTION_PATTERN = re.compile(
    rf"""
    (?:^|[\s;|])
    >>?
    \s*
    (?P<path>
        "[^"]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})"
        |'[^']+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})'
        |[^\s"'`;|]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SHELL_OUTPUT_TEXT_PATTERN = re.compile(
    rf"""
    (?:
        saved
        |written
        |wrote
        |generated
        |exported
        |created
        |produced
        |rendered
        |converted
        |output(?:\s+file)?
        |report
        |result
    )
    (?:\s+\w+){{0,3}}
    \s+(?:to|at|as)\s+
    (?P<path>
        "[^"]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})"
        |'[^']+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})'
        |[^\s"'`;|]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SHELL_PATH_TOKEN_PATTERN = re.compile(
    rf"""
    (?P<path>
        "[^"]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})"
        |'[^']+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})'
        |[^\s"'`;|]+\.(?:{_DELIVERABLE_EXTENSION_PATTERN})
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
_SHELL_OUTPUT_VERB_PATTERN = re.compile(
    r"\b(generate|generated|export|exported|render|rendered|convert|converted|write|wrote|save|saved|create|created)\b",
    re.IGNORECASE,
)
_FILE_ACTION_PRIORITY = {
    "create_revised_file": 1,
    "export_deliverable": 2,
    "overwrite_uploaded_original": 3,
}


@dataclass(frozen=True, slots=True)
class UploadedFileTarget:
    upload_id: str
    upload_name: str
    absolute_path: str
    relative_path: str = ""


@dataclass(frozen=True, slots=True)
class ToolFileAction:
    kind: str
    parameter_summary: str
    impact_summary: str


@dataclass(slots=True)
class ToolArtifactHint:
    """Artifact candidate identified from a successful tool execution."""

    artifact_type: str
    uri: str
    display_name: str = ""
    role: str = "intermediate_file"
    format: str = ""
    mime_type: str = ""
    size_bytes: int | None = None
    source: str = "agent_generated"
    is_final: bool = False
    preview_kind: str = "none"
    parent_artifact_id: str | None = None
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialize one artifact hint for stream/runtime events."""
        return {
            "artifact_type": self.artifact_type,
            "uri": self.uri,
            "display_name": self.display_name,
            "role": self.role,
            "format": self.format,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "source": self.source,
            "is_final": self.is_final,
            "preview_kind": self.preview_kind,
            "parent_artifact_id": self.parent_artifact_id,
            "summary": self.summary,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ToolExecutionContext:
    """Normalized metadata captured around one tool call lifecycle."""

    tool_name: str
    tool_class: str
    tool_call_id: str
    arguments: dict[str, Any]
    parameter_summary: str
    risk_category: str
    risk_level: str
    requires_approval: bool
    approval_reason: str
    policy_allowed: bool
    policy_denied_reason: str
    impact_summary: str
    started_at: str = ""
    finished_at: str | None = None
    duration_ms: int | None = None
    artifacts: list[ToolArtifactHint] = field(default_factory=list)
    touched_paths: list[str] = field(default_factory=list)
    _started_perf_counter: float = field(default=0.0, repr=False)
    _tool: Tool | None = field(default=None, repr=False)

    def mark_started(self) -> None:
        """Record the start timestamp for this execution."""
        self.started_at = utc_now_iso()
        self._started_perf_counter = time.perf_counter()

    def mark_finished(self, result: ToolResult) -> None:
        """Record completion timestamps and derive artifact hints."""
        self.finished_at = utc_now_iso()
        self.duration_ms = max(0, int((time.perf_counter() - self._started_perf_counter) * 1000))
        self.artifacts = detect_tool_artifacts(
            tool=self._tool,
            function_name=self.tool_name,
            arguments=self.arguments,
            result=result,
        )
        self.touched_paths = [
            path.as_posix()
            for path in detect_tool_touched_paths(
                tool=self._tool,
                function_name=self.tool_name,
                arguments=self.arguments,
                result=result,
            )
        ]

    def tool_call_payload(self) -> dict[str, Any]:
        """Payload shape exposed by the public `tool_call` stream event."""
        return {
            "id": self.tool_call_id,
            "tool_call_id": self.tool_call_id,
            "name": self.tool_name,
            "arguments": dict(self.arguments),
            "parameter_summary": self.parameter_summary,
            "risk_category": self.risk_category,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "approval_reason": self.approval_reason,
            "policy_allowed": self.policy_allowed,
            "policy_denied_reason": self.policy_denied_reason,
            "impact_summary": self.impact_summary,
        }

    def runtime_start_payload(self) -> dict[str, Any]:
        """Payload shape exposed by `tool_started`/`delegate_started` events."""
        payload = self.tool_call_payload()
        payload["tool_class"] = self.tool_class
        payload["started_at"] = self.started_at
        return payload

    def runtime_finish_payload(self, result: ToolResult) -> dict[str, Any]:
        """Payload shape exposed by `tool_finished`/`delegate_finished` events."""
        payload = self.runtime_start_payload()
        payload.update(
            {
                "finished_at": self.finished_at,
                "duration_ms": self.duration_ms,
                "success": result.success,
                "content": result.content if result.success else None,
                "error": result.error if not result.success else None,
                "artifacts": [artifact.to_payload() for artifact in self.artifacts],
                "touched_paths": list(self.touched_paths),
            }
        )
        if result.metadata:
            payload["metadata"] = dict(result.metadata)
        return payload

    def tool_result_payload(self, result: ToolResult) -> dict[str, Any]:
        """Payload shape exposed by the public `tool_result` stream event."""
        payload = self.tool_call_payload()
        payload.update(
            {
                "success": result.success,
                "content": result.content if result.success else None,
                "error": result.error if not result.success else None,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "duration_ms": self.duration_ms,
                "artifacts": [artifact.to_payload() for artifact in self.artifacts],
                "touched_paths": list(self.touched_paths),
            }
        )
        if result.metadata:
            payload["metadata"] = dict(result.metadata)
        return payload


def prepare_tool_execution(
    *,
    tool: Tool | None,
    function_name: str,
    tool_call_id: str,
    arguments: dict[str, Any],
    approval_policy: ApprovalPolicy | None = None,
    workspace_policy: WorkspacePolicy | None = None,
    delegation_policy: DelegationPolicy | None = None,
    is_main_agent: bool = False,
    uploaded_file_targets: list[UploadedFileTarget] | None = None,
) -> ToolExecutionContext:
    """Build normalized execution metadata for one tool invocation."""
    tool_class = tool.__class__.__name__ if tool is not None else "UnknownTool"
    parameter_summary = summarize_tool_arguments(arguments)
    risk_category, risk_level, impact_summary = classify_tool_invocation(
        tool=tool,
        function_name=function_name,
        arguments=arguments,
    )
    file_action = _describe_tool_file_action(
        tool=tool,
        function_name=function_name,
        arguments=arguments,
        uploaded_file_targets=uploaded_file_targets or [],
    )
    if file_action is not None:
        parameter_summary = file_action.parameter_summary
        impact_summary = file_action.impact_summary
        if file_action.kind == "overwrite_uploaded_original":
            risk_category = "uploaded_original_overwrite"
            risk_level = "critical"

    requires_approval, approval_reason = requires_tool_approval(
        approval_policy=approval_policy,
        tool=tool,
        function_name=function_name,
        risk_category=risk_category,
        risk_level=risk_level,
    )
    if (
        file_action is not None
        and file_action.kind == "overwrite_uploaded_original"
        and not _has_explicit_auto_approve_policy(
            approval_policy=approval_policy,
            tool=tool,
            function_name=function_name,
        )
    ):
        requires_approval = True
        approval_reason = "uploaded_original_overwrite"

    context = ToolExecutionContext(
        tool_name=function_name,
        tool_class=tool_class,
        tool_call_id=tool_call_id,
        arguments=dict(arguments),
        parameter_summary=parameter_summary,
        risk_category=risk_category,
        risk_level=risk_level,
        requires_approval=requires_approval,
        approval_reason=approval_reason,
        policy_allowed=False,
        policy_denied_reason="",
        impact_summary=impact_summary,
        _tool=tool,
    )
    policy_allowed, policy_denied_reason = evaluate_tool_policy(
        tool=tool,
        function_name=function_name,
        arguments=arguments,
        workspace_policy=workspace_policy,
        delegation_policy=delegation_policy,
        is_main_agent=is_main_agent,
        risk_category=risk_category,
    )
    context.policy_allowed = policy_allowed
    context.policy_denied_reason = policy_denied_reason
    return context


def summarize_tool_arguments(arguments: dict[str, Any], *, max_length: int = 280) -> str:
    """Convert tool arguments into a compact, log-safe summary string."""
    sanitized: dict[str, Any] = {}
    for key, value in arguments.items():
        sanitized[key] = _truncate_argument_value(value)

    summary = json.dumps(sanitized, ensure_ascii=False, separators=(", ", ": "))
    if len(summary) <= max_length:
        return summary
    return f"{summary[: max_length - 3]}..."


def classify_tool_invocation(
    *,
    tool: Tool | None,
    function_name: str,
    arguments: dict[str, Any],
) -> tuple[str, str, str]:
    """Return `(risk_category, risk_level, impact_summary)` for a tool call."""
    tool_class = tool.__class__.__name__ if tool is not None else ""
    command = str(arguments.get("command", "")).strip()

    if tool_class == "BashTool" or function_name == "bash":
        if command and _CREDENTIAL_PATTERN.search(command):
            return (
                "credentials",
                "critical",
                "Shell command may access, print, or mutate credentials and other secrets.",
            )
        if command and _NETWORK_PATTERN.search(command):
            return (
                "external_network",
                "high",
                "Shell command may reach external systems or transfer data over the network.",
            )
        if command and _SHELL_WRITE_PATTERN.search(command):
            return (
                "shell_write",
                "high",
                "Shell command may mutate files, repository state, or local environment state.",
            )
        return (
            "shell_command",
            "medium",
            "Shell command executes arbitrary local instructions inside the workspace context.",
        )

    if tool_class in {"WriteTool", "EditTool"} or function_name in {"write_file", "edit_file"}:
        return (
            "filesystem_write",
            "high",
            "Tool may overwrite or mutate files in the workspace.",
        )

    if tool_class == "ReadTool" or function_name == "read_file":
        return (
            "filesystem_read",
            "low",
            "Tool reads existing workspace files without modifying them.",
        )

    if tool_class == "SendChannelFileTool" or function_name == "send_channel_file":
        return (
            "external_delivery",
            "medium",
            "Tool can upload one local file and deliver it back through the bound external channel.",
        )

    if tool_class == "BashKillTool" or function_name == "bash_kill":
        return (
            "process_control",
            "medium",
            "Tool can terminate or alter background process execution.",
        )

    if tool_class == "BashOutputTool" or function_name == "bash_output":
        return (
            "process_inspection",
            "low",
            "Tool only inspects output from an existing background process.",
        )

    if tool_class == "MCPTool":
        return (
            "external_network",
            "high",
            "MCP tool may cross process or network boundaries outside the local workspace.",
        )

    if function_name in {"delegate_task", "delegate_tasks"}:
        return (
            "delegate",
            "medium",
            "Tool can spawn sub-agents that continue acting within the same run context.",
        )

    return (
        "other",
        "low",
        "Tool risk classification not explicitly defined; treat as a low-risk custom action.",
    )


def requires_tool_approval(
    *,
    approval_policy: ApprovalPolicy | None,
    tool: Tool | None,
    function_name: str,
    risk_category: str,
    risk_level: str,
) -> tuple[bool, str]:
    """Whether the current template policy marks this tool call as approval-worthy."""
    if approval_policy is None:
        return False, ""

    identifiers = _collect_policy_identifiers(tool=tool, function_name=function_name)

    if _matches_policy_identifier(approval_policy.auto_approve_tools, identifiers):
        return False, ""

    if _matches_policy_identifier(approval_policy.require_approval_tools, identifiers):
        return True, "tool_rule"

    if risk_level in approval_policy.require_approval_risk_levels:
        return True, f"risk_level:{risk_level}"

    if risk_category in approval_policy.require_approval_risk_categories:
        return True, f"risk_category:{risk_category}"

    if approval_policy.mode == "strict":
        if risk_level in {"medium", "high", "critical"}:
            return True, f"strict:{risk_level}"

    return False, ""


def evaluate_tool_policy(
    *,
    tool: Tool | None,
    function_name: str,
    arguments: dict[str, Any],
    workspace_policy: WorkspacePolicy | None,
    delegation_policy: DelegationPolicy | None,
    is_main_agent: bool,
    risk_category: str,
) -> tuple[bool, str]:
    """Return whether the invocation is allowed by the current tool policy layer."""
    if tool is None:
        return False, f"Tool '{function_name}' is not registered in the current runtime."

    delegation_allowed, delegation_denied_reason = _evaluate_delegation_policy(
        tool=tool,
        function_name=function_name,
        delegation_policy=delegation_policy,
        is_main_agent=is_main_agent,
    )
    if not delegation_allowed:
        return delegation_allowed, delegation_denied_reason

    tool_class = tool.__class__.__name__
    identifiers = _collect_policy_identifiers(tool=tool, function_name=function_name)
    if (
        workspace_policy is not None
        and _matches_policy_identifier(workspace_policy.disabled_tools, identifiers)
    ):
        return (
            False,
            f"Tool policy blocked '{function_name}' because it is disabled by the current template policy.",
        )

    read_only_mode = (
        workspace_policy is not None
        and _matches_policy_identifier(workspace_policy.read_only_tools, identifiers)
    )

    if tool_class == "ReadTool" or function_name == "read_file":
        return _evaluate_file_read_policy(
            tool=tool,
            function_name=function_name,
            arguments=arguments,
            workspace_policy=workspace_policy,
        )

    if tool_class in {"WriteTool", "EditTool"} or function_name in {"write_file", "edit_file"}:
        if read_only_mode:
            return (
                False,
                f"Tool policy blocked '{function_name}' because it is restricted to read-only access.",
            )
        return _evaluate_file_write_policy(
            tool=tool,
            function_name=function_name,
            arguments=arguments,
            workspace_policy=workspace_policy,
        )

    if tool_class == "BashTool" or function_name == "bash":
        return _evaluate_shell_policy(
            function_name=function_name,
            arguments=arguments,
            workspace_policy=workspace_policy,
            risk_category=risk_category,
            read_only_mode=read_only_mode,
        )

    if tool_class == "BashKillTool" or function_name == "bash_kill":
        if read_only_mode:
            return (
                False,
                f"Tool policy blocked '{function_name}' because it is restricted to read-only access.",
            )
        return True, ""

    if tool_class == "MCPTool":
        if read_only_mode:
            return (
                False,
                f"Tool policy blocked '{function_name}' because MCP access is restricted to read-only tools.",
            )
        return _evaluate_network_access_policy(
            function_name=function_name,
            arguments=arguments,
            workspace_policy=workspace_policy,
        )

    if tool_class == "BashOutputTool" or function_name == "bash_output":
        return True, ""

    return True, ""


def _evaluate_delegation_policy(
    *,
    tool: Tool,
    function_name: str,
    delegation_policy: DelegationPolicy | None,
    is_main_agent: bool,
) -> tuple[bool, str]:
    """按主/子 agent 委派策略检查工具调用是否允许。"""
    if delegation_policy is None or not is_main_agent:
        return True, ""

    tool_class = tool.__class__.__name__
    mode = delegation_policy.mode
    read_only_tools = {
        "ReadTool",
        "RecallNoteTool",
        "SearchMemoryTool",
        "SearchSessionHistoryTool",
        "BashOutputTool",
    }
    always_allowed_tools = {
        "DelegateTool",
        "DelegateBatchTool",
        "ShareContextTool",
        "ReadSharedContextTool",
    }

    if tool_class in always_allowed_tools or function_name in {
        "delegate_task",
        "delegate_tasks",
        "share_context",
        "read_shared_context",
    }:
        return True, ""

    if (
        not delegation_policy.allow_main_agent_read_tools
        and (tool_class in read_only_tools or function_name == "read_file")
    ):
        return (
            False,
            "Main agent delegation policy forbids direct use of read-only tools. "
            "Use delegate_task/delegate_tasks or rely on shared context instead.",
        )

    if tool_class in {"WriteTool", "EditTool"} or function_name in {"write_file", "edit_file"}:
        if mode == "supervisor_only" or delegation_policy.require_delegate_for_write_actions:
            return (
                False,
                "Main agent delegation policy blocks direct file writes. "
                "Create or update files through delegate_task/delegate_tasks.",
            )

    if tool_class in {"BashTool", "BashKillTool"} or function_name in {"bash", "bash_kill"}:
        if mode == "supervisor_only" or delegation_policy.require_delegate_for_shell:
            return (
                False,
                "Main agent delegation policy blocks direct shell execution. "
                "Run shell commands through delegate_task/delegate_tasks.",
            )

    if tool_class == "MCPTool":
        if mode == "supervisor_only" or delegation_policy.require_delegate_for_stateful_mcp:
            return (
                False,
                "Main agent delegation policy blocks direct MCP execution. "
                "Delegate this action to a worker unless it has been downgraded to a read-only flow.",
            )

    if mode == "supervisor_only" and tool_class not in read_only_tools and function_name != "read_file":
        return (
            False,
            "Main agent is running in supervisor_only mode and may only plan, delegate, "
            "share context, or use explicitly allowed read-only tools.",
        )

    return True, ""


def _evaluate_file_read_policy(
    *,
    tool: Tool,
    function_name: str,
    arguments: dict[str, Any],
    workspace_policy: WorkspacePolicy | None,
) -> tuple[bool, str]:
    path_value = arguments.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return False, f"Tool policy blocked '{function_name}' because the target path is missing."

    if workspace_policy is None or not workspace_policy.readable_roots:
        return True, ""

    workspace_dir = getattr(tool, "workspace_dir", None)
    if workspace_dir is None:
        return (
            False,
            f"Tool policy blocked '{function_name}' because readable roots are configured but the tool has no workspace directory context.",
        )

    try:
        workspace_root = Path(workspace_dir).resolve()
    except Exception:
        return (
            False,
            f"Tool policy blocked '{function_name}' because the workspace root could not be resolved.",
        )

    target_path = _resolve_tool_path(tool, path_value)
    allowed_roots = _resolve_policy_roots(
        workspace_root=workspace_root,
        configured_roots=workspace_policy.readable_roots,
    )
    if any(_is_relative_to(target_path, allowed_root) for allowed_root in allowed_roots):
        return True, ""

    allowed_labels = ", ".join(str(root) for root in allowed_roots)
    return (
        False,
        (
            f"Tool policy blocked '{function_name}' for path '{target_path}' because it is outside the allowed readable roots: "
            f"{allowed_labels}"
        ),
    )


def detect_tool_artifacts(
    *,
    tool: Tool | None,
    function_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
) -> list[ToolArtifactHint]:
    """Identify standard artifact records from a successful tool invocation."""
    if not result.success:
        return []

    tool_class = tool.__class__.__name__ if tool is not None else ""
    artifacts: list[ToolArtifactHint] = []

    if tool_class in {"WriteTool", "EditTool"} or function_name in {"write_file", "edit_file"}:
        path_value = arguments.get("path")
        if isinstance(path_value, str) and path_value.strip():
            resolved_path = _resolve_tool_path(tool, path_value)
            operation = "write" if function_name == "write_file" or tool_class == "WriteTool" else "edit"
            artifacts.append(
                _build_workspace_file_artifact_hint(
                    resolved_path=resolved_path,
                    function_name=function_name,
                    operation=operation,
                    summary=(
                        f"{'Wrote' if operation == 'write' else 'Edited'} workspace file "
                        f"{resolved_path.name}"
                    ),
                    metadata={"operation": operation},
                )
            )

    if function_name == "bash" and not bool(arguments.get("run_in_background")):
        command = str(arguments.get("command", "")).strip()
        for resolved_path in _detect_shell_output_paths(tool=tool, command=command, result=result):
            artifacts.append(
                _build_workspace_file_artifact_hint(
                    resolved_path=resolved_path,
                    function_name=function_name,
                    operation="shell_command",
                    summary=f"Command produced workspace file {resolved_path.name}",
                    metadata={"artifact_detection": "shell_output_path"},
                )
            )

    bash_id = getattr(result, "bash_id", None)
    if isinstance(bash_id, str) and bash_id.strip():
        artifacts.append(
            ToolArtifactHint(
                artifact_type="background_process",
                uri=f"bash://{bash_id}",
                display_name=bash_id,
                role="supporting_output",
                source="system_generated",
                summary="Started background shell process",
                metadata={"source_tool": function_name},
            )
        )

    return artifacts


def detect_tool_touched_paths(
    *,
    tool: Tool | None,
    function_name: str,
    arguments: dict[str, Any],
    result: ToolResult,
) -> list[Path]:
    """Collect workspace paths a tool call inspected or mutated for prompt refresh."""
    del result
    tool_class = tool.__class__.__name__ if tool is not None else ""
    touched_paths: list[Path] = []
    seen: set[Path] = set()

    def add_path(path: Path | None) -> None:
        if path is None or path in seen:
            return
        seen.add(path)
        touched_paths.append(path)

    if tool_class in {"ReadTool", "WriteTool", "EditTool"} or function_name in {
        "read_file",
        "write_file",
        "edit_file",
    }:
        path_value = arguments.get("path")
        if isinstance(path_value, str) and path_value.strip():
            add_path(_resolve_tool_path(tool, path_value))

    if tool_class == "SendChannelFileTool" or function_name == "send_channel_file":
        path_value = arguments.get("path")
        if isinstance(path_value, str) and path_value.strip():
            add_path(_resolve_tool_path(tool, path_value))

    if function_name == "bash":
        command = str(arguments.get("command", "")).strip()
        if command:
            for resolved_path in sorted(
                _resolve_shell_command_paths(tool=tool, command=command),
                key=lambda item: item.as_posix(),
            ):
                add_path(resolved_path)
            for resolved_path in _detect_shell_output_candidate_paths(tool=tool, command=command):
                add_path(resolved_path)

    return touched_paths[:12]


def _evaluate_file_write_policy(
    *,
    tool: Tool,
    function_name: str,
    arguments: dict[str, Any],
    workspace_policy: WorkspacePolicy | None,
) -> tuple[bool, str]:
    path_value = arguments.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        return False, f"Tool policy blocked '{function_name}' because the target path is missing."

    if workspace_policy is None or not workspace_policy.writable_roots:
        return True, ""

    workspace_dir = getattr(tool, "workspace_dir", None)
    if workspace_dir is None:
        return (
            False,
            f"Tool policy blocked '{function_name}' because writable roots are configured but the tool has no workspace directory context.",
        )

    try:
        workspace_root = Path(workspace_dir).resolve()
    except Exception:
        return (
            False,
            f"Tool policy blocked '{function_name}' because the workspace root could not be resolved.",
        )

    target_path = _resolve_tool_path(tool, path_value)
    allowed_roots = _resolve_policy_roots(
        workspace_root=workspace_root,
        configured_roots=workspace_policy.writable_roots,
    )

    if any(_is_relative_to(target_path, allowed_root) for allowed_root in allowed_roots):
        return True, ""

    allowed_labels = ", ".join(str(root) for root in allowed_roots)
    return (
        False,
        (
            f"Tool policy blocked '{function_name}' for path '{target_path}' because it is outside the allowed writable roots: "
            f"{allowed_labels}"
        ),
    )


def _evaluate_shell_policy(
    *,
    function_name: str,
    arguments: dict[str, Any],
    workspace_policy: WorkspacePolicy | None,
    risk_category: str,
    read_only_mode: bool,
) -> tuple[bool, str]:
    command = str(arguments.get("command", "")).strip()
    if read_only_mode and risk_category in {"shell_write", "external_network", "credentials"}:
        return (
            False,
            f"Tool policy blocked '{function_name}' because the shell tool is restricted to read-only commands.",
        )

    if workspace_policy is None:
        return True, ""

    prefixes = [
        str(item).strip()
        for item in workspace_policy.allowed_shell_command_prefixes
        if str(item).strip()
    ]
    if prefixes:
        normalized_command = command.lstrip().lower()
        if not any(normalized_command.startswith(prefix.lower()) for prefix in prefixes):
            allowed = ", ".join(prefixes)
            return (
                False,
                f"Tool policy blocked '{function_name}' because the command is outside the allowed command prefixes: {allowed}",
            )

    if risk_category == "external_network":
        return _evaluate_network_access_policy(
            function_name=function_name,
            arguments=arguments,
            workspace_policy=workspace_policy,
        )

    return True, ""


def _evaluate_network_access_policy(
    *,
    function_name: str,
    arguments: dict[str, Any],
    workspace_policy: WorkspacePolicy | None,
) -> tuple[bool, str]:
    if workspace_policy is None or not workspace_policy.allowed_network_domains:
        return True, ""

    allowed_domains = [
        str(item).strip().lower()
        for item in workspace_policy.allowed_network_domains
        if str(item).strip()
    ]
    detected_domains = sorted(_extract_network_domains(arguments))
    if not detected_domains:
        return (
            False,
            f"Tool policy blocked '{function_name}' because network domains are restricted but no target domain could be inferred from the arguments.",
        )

    blocked = [
        domain
        for domain in detected_domains
        if not any(_domain_matches_allowlist(domain, allowed) for allowed in allowed_domains)
    ]
    if not blocked:
        return True, ""

    return (
        False,
        (
            f"Tool policy blocked '{function_name}' because the target domains are outside the allowed network domains: "
            f"{', '.join(allowed_domains)}. Blocked: {', '.join(blocked)}"
        ),
    )


def _resolve_policy_roots(*, workspace_root: Path, configured_roots: list[str]) -> list[Path]:
    allowed_roots: list[Path] = []
    for raw_root in configured_roots:
        root_value = str(raw_root).strip()
        if not root_value:
            continue
        root_path = Path(root_value)
        if not root_path.is_absolute():
            root_path = workspace_root / root_path
        allowed_roots.append(root_path.resolve())
    return allowed_roots


def _collect_policy_identifiers(*, tool: Tool | None, function_name: str) -> set[str]:
    identifiers = {function_name}
    if tool is not None:
        identifiers.add(tool.__class__.__name__)
        tool_name = getattr(tool, "name", "")
        if isinstance(tool_name, str) and tool_name.strip():
            identifiers.add(tool_name.strip())
    return {item.strip() for item in identifiers if item and item.strip()}


def _matches_policy_identifier(configured_items: list[str], identifiers: set[str]) -> bool:
    normalized_identifiers = {item.lower() for item in identifiers}
    return any(str(item).strip().lower() in normalized_identifiers for item in configured_items)


def _has_explicit_auto_approve_policy(
    *,
    approval_policy: ApprovalPolicy | None,
    tool: Tool | None,
    function_name: str,
) -> bool:
    if approval_policy is None:
        return False
    return _matches_policy_identifier(
        approval_policy.auto_approve_tools,
        _collect_policy_identifiers(tool=tool, function_name=function_name),
    )


def _extract_network_domains(arguments: dict[str, Any]) -> set[str]:
    domains: set[str] = set()
    for raw_value in arguments.values():
        for text_value in _flatten_string_values(raw_value):
            for match in _NETWORK_HOST_PATTERN.findall(text_value):
                domains.add(match.lower())
            if _NETWORK_PATTERN.search(text_value):
                for match in _PLAIN_HOST_PATTERN.findall(text_value):
                    lowered = match.lower()
                    if "." in lowered:
                        domains.add(lowered)
    return domains


def _flatten_string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_string_values(item))
        return flattened
    if isinstance(value, dict):
        flattened = []
        for item in value.values():
            flattened.extend(_flatten_string_values(item))
        return flattened
    return []


def _domain_matches_allowlist(domain: str, allowed_domain: str) -> bool:
    normalized_domain = domain.strip().lower()
    normalized_allowed = allowed_domain.strip().lower()
    return normalized_domain == normalized_allowed or normalized_domain.endswith(
        f".{normalized_allowed}"
    )


def _resolve_tool_path(tool: Tool | None, path_value: str) -> Path:
    file_path = Path(path_value)
    if file_path.is_absolute():
        return file_path.resolve()

    workspace_dir = getattr(tool, "workspace_dir", None)
    if workspace_dir:
        try:
            return (Path(workspace_dir) / file_path).resolve()
        except Exception:
            return file_path
    return file_path


def _guess_artifact_format(path: Path) -> str:
    return path.suffix.lstrip(".").lower()


def _guess_artifact_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or ""


def _build_workspace_file_artifact_hint(
    *,
    resolved_path: Path,
    function_name: str,
    operation: str,
    summary: str,
    metadata: dict[str, Any] | None = None,
) -> ToolArtifactHint:
    artifact_format = _guess_artifact_format(resolved_path)
    mime_type = _guess_artifact_mime_type(resolved_path)
    artifact_metadata = {"source_tool": function_name}
    if metadata:
        artifact_metadata.update(metadata)
    return ToolArtifactHint(
        artifact_type="workspace_file",
        uri=str(resolved_path),
        display_name=resolved_path.name,
        role="intermediate_file",
        format=artifact_format,
        mime_type=mime_type,
        size_bytes=_read_file_size(resolved_path),
        source="agent_generated",
        is_final=False,
        preview_kind=guess_preview_kind(
            artifact_format=artifact_format,
            mime_type=mime_type,
        ),
        summary=summary,
        metadata=artifact_metadata,
    )


def _describe_tool_file_action(
    *,
    tool: Tool | None,
    function_name: str,
    arguments: dict[str, Any],
    uploaded_file_targets: list[UploadedFileTarget],
) -> ToolFileAction | None:
    resolved_targets = _resolve_uploaded_file_targets(uploaded_file_targets)
    if not resolved_targets:
        return None

    tool_class = tool.__class__.__name__ if tool is not None else ""
    if tool_class in {"WriteTool", "EditTool"} or function_name in {"write_file", "edit_file"}:
        path_value = arguments.get("path")
        if not isinstance(path_value, str) or not path_value.strip():
            return None
        resolved_path = _resolve_tool_path(tool, path_value)
        return _describe_output_path_against_uploads(
            resolved_path=resolved_path,
            upload_targets=resolved_targets,
        )

    if function_name != "bash":
        return None

    command = str(arguments.get("command", "")).strip()
    if not command:
        return None

    output_paths = _detect_shell_output_candidate_paths(tool=tool, command=command)
    if output_paths:
        direct_match = _select_best_file_action(
            _describe_output_path_against_uploads(
                resolved_path=resolved_path,
                upload_targets=resolved_targets,
            )
            for resolved_path in output_paths
        )
        if direct_match is not None:
            return direct_match

    input_uploads = _detect_shell_input_uploads(
        tool=tool,
        command=command,
        upload_targets=resolved_targets,
    )
    if not input_uploads or not output_paths:
        return None

    return _select_best_file_action(
        _describe_output_path_against_uploads(
            resolved_path=resolved_path,
            upload_targets=input_uploads,
            allow_cross_directory=True,
        )
        for resolved_path in output_paths
    )


def _resolve_uploaded_file_targets(
    uploaded_file_targets: list[UploadedFileTarget],
) -> list[tuple[UploadedFileTarget, Path]]:
    resolved_targets: list[tuple[UploadedFileTarget, Path]] = []
    seen: set[tuple[str, Path]] = set()
    for target in uploaded_file_targets:
        try:
            resolved_path = Path(target.absolute_path).resolve()
        except Exception:
            continue
        key = (target.upload_id.strip(), resolved_path)
        if key in seen:
            continue
        seen.add(key)
        resolved_targets.append((target, resolved_path))
    return resolved_targets


def _describe_output_path_against_uploads(
    *,
    resolved_path: Path,
    upload_targets: list[tuple[UploadedFileTarget, Path]],
    allow_cross_directory: bool = False,
) -> ToolFileAction | None:
    return _select_best_file_action(
        _build_file_action(
            kind=_classify_path_against_upload(
                resolved_path=resolved_path,
                upload_path=upload_path,
                allow_cross_directory=allow_cross_directory,
            ),
            resolved_path=resolved_path,
            upload_target=upload_target,
            upload_path=upload_path,
        )
        for upload_target, upload_path in upload_targets
    )


def _select_best_file_action(actions: Any) -> ToolFileAction | None:
    best_action: ToolFileAction | None = None
    best_priority = -1
    for action in actions:
        if action is None:
            continue
        priority = _FILE_ACTION_PRIORITY.get(action.kind, 0)
        if priority > best_priority:
            best_action = action
            best_priority = priority
    return best_action


def _classify_path_against_upload(
    *,
    resolved_path: Path,
    upload_path: Path,
    allow_cross_directory: bool,
) -> str | None:
    if resolved_path == upload_path:
        return "overwrite_uploaded_original"

    same_directory = resolved_path.parent == upload_path.parent
    if same_directory:
        if (
            resolved_path.suffix
            and upload_path.suffix
            and resolved_path.suffix.lower() != upload_path.suffix.lower()
        ):
            return "export_deliverable"
        return "create_revised_file"

    if not allow_cross_directory or resolved_path == upload_path:
        return None

    if (
        resolved_path.suffix
        and upload_path.suffix
        and resolved_path.suffix.lower() != upload_path.suffix.lower()
    ):
        return "export_deliverable"
    if resolved_path.name != upload_path.name:
        return "create_revised_file"
    return None


def _build_file_action(
    *,
    kind: str | None,
    resolved_path: Path,
    upload_target: UploadedFileTarget,
    upload_path: Path,
) -> ToolFileAction | None:
    if not kind:
        return None

    source_name = upload_target.upload_name.strip() or upload_path.name or "已上传文件"
    target_name = resolved_path.name or resolved_path.as_posix()

    if kind == "overwrite_uploaded_original":
        return ToolFileAction(
            kind=kind,
            parameter_summary=f"覆盖已上传原文件：{source_name}",
            impact_summary=f"将直接修改已上传原文件“{source_name}”，需要审批确认。",
        )

    if kind == "create_revised_file":
        return ToolFileAction(
            kind=kind,
            parameter_summary=f"创建修订文件：{target_name}",
            impact_summary=(
                f"将基于已上传文件“{source_name}”创建修订副本“{target_name}”，保留原始文件不变。"
            ),
        )

    if kind == "export_deliverable":
        return ToolFileAction(
            kind=kind,
            parameter_summary=f"导出新交付文件：{target_name}",
            impact_summary=(
                f"将基于已上传文件“{source_name}”导出新的交付格式“{target_name}”，不会覆盖原始文件。"
            ),
        )

    return None


def _detect_shell_output_paths(
    *,
    tool: Tool | None,
    command: str,
    result: ToolResult,
) -> list[Path]:
    workspace_root = _resolve_workspace_root(tool)
    if workspace_root is None or not command:
        return []

    raw_candidates: list[str] = []
    for pattern in (_SHELL_OUTPUT_FLAG_PATTERN, _SHELL_REDIRECTION_PATTERN):
        raw_candidates.extend(match.group("path") for match in pattern.finditer(command))

    for text_value in _flatten_string_values(
        {
            "content": result.content,
            "stdout": getattr(result, "stdout", ""),
            "stderr": getattr(result, "stderr", ""),
        }
    ):
        raw_candidates.extend(match.group("path") for match in _SHELL_OUTPUT_TEXT_PATTERN.finditer(text_value))

    if not raw_candidates:
        fallback_paths = [
            match.group("path")
            for match in _SHELL_PATH_TOKEN_PATTERN.finditer(command)
        ]
        if len(fallback_paths) >= 2:
            raw_candidates.append(fallback_paths[-1])
        elif len(fallback_paths) == 1 and _SHELL_OUTPUT_VERB_PATTERN.search(command):
            raw_candidates.append(fallback_paths[0])

    resolved_candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_candidates:
        normalized_path = _normalize_shell_path_candidate(raw_path)
        if not normalized_path:
            continue
        resolved_path = _resolve_workspace_candidate_path(workspace_root, normalized_path)
        if resolved_path is None:
            continue
        if resolved_path.suffix.lstrip(".").lower() not in _DELIVERABLE_EXTENSIONS:
            continue
        if not resolved_path.exists() or not resolved_path.is_file():
            continue
        if resolved_path in seen:
            continue
        seen.add(resolved_path)
        resolved_candidates.append(resolved_path)
        if len(resolved_candidates) >= 6:
            break

    return resolved_candidates


def _detect_shell_output_candidate_paths(*, tool: Tool | None, command: str) -> list[Path]:
    workspace_root = _resolve_workspace_root(tool)
    if workspace_root is None or not command:
        return []

    raw_candidates: list[str] = []
    for pattern in (_SHELL_OUTPUT_FLAG_PATTERN, _SHELL_REDIRECTION_PATTERN):
        raw_candidates.extend(match.group("path") for match in pattern.finditer(command))

    if not raw_candidates:
        fallback_paths = [
            match.group("path")
            for match in _SHELL_PATH_TOKEN_PATTERN.finditer(command)
        ]
        if len(fallback_paths) >= 2:
            raw_candidates.append(fallback_paths[-1])
        elif len(fallback_paths) == 1 and _SHELL_OUTPUT_VERB_PATTERN.search(command):
            raw_candidates.append(fallback_paths[0])

    resolved_candidates: list[Path] = []
    seen: set[Path] = set()
    for raw_path in raw_candidates:
        normalized_path = _normalize_shell_path_candidate(raw_path)
        if not normalized_path:
            continue
        resolved_path = _resolve_workspace_candidate_path(workspace_root, normalized_path)
        if resolved_path is None or resolved_path in seen:
            continue
        seen.add(resolved_path)
        resolved_candidates.append(resolved_path)
    return resolved_candidates


def _detect_shell_input_uploads(
    *,
    tool: Tool | None,
    command: str,
    upload_targets: list[tuple[UploadedFileTarget, Path]],
) -> list[tuple[UploadedFileTarget, Path]]:
    resolved_command_paths = _resolve_shell_command_paths(tool=tool, command=command)
    if not resolved_command_paths:
        return []

    matched: list[tuple[UploadedFileTarget, Path]] = []
    seen: set[Path] = set()
    for upload_target, upload_path in upload_targets:
        if upload_path not in resolved_command_paths or upload_path in seen:
            continue
        seen.add(upload_path)
        matched.append((upload_target, upload_path))
    return matched


def _resolve_shell_command_paths(
    *,
    tool: Tool | None,
    command: str,
) -> set[Path]:
    workspace_root = _resolve_workspace_root(tool)
    if workspace_root is None or not command:
        return set()

    resolved_paths: set[Path] = set()
    raw_paths = [match.group("path") for match in _SHELL_PATH_TOKEN_PATTERN.finditer(command)]
    for raw_path in raw_paths:
        normalized_path = _normalize_shell_path_candidate(raw_path)
        if not normalized_path:
            continue
        resolved_path = _resolve_workspace_candidate_path(workspace_root, normalized_path)
        if resolved_path is not None:
            resolved_paths.add(resolved_path)
    return resolved_paths


def _resolve_workspace_root(tool: Tool | None) -> Path | None:
    workspace_dir = getattr(tool, "workspace_dir", None)
    if workspace_dir is None:
        return None
    try:
        return Path(workspace_dir).resolve()
    except Exception:
        return None


def _normalize_shell_path_candidate(raw_path: str) -> str:
    candidate = str(raw_path or "").strip()
    if not candidate:
        return ""
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in {'"', "'"}:
        candidate = candidate[1:-1]
    return candidate.rstrip(".,;:)]}")


def _resolve_workspace_candidate_path(workspace_root: Path, path_value: str) -> Path | None:
    try:
        candidate = Path(path_value)
        resolved = candidate.resolve() if candidate.is_absolute() else (workspace_root / candidate).resolve()
    except Exception:
        return None
    if not _is_relative_to(resolved, workspace_root):
        return None
    return resolved


def _read_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except OSError:
        return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _truncate_argument_value(value: Any, *, max_length: int = 120) -> Any:
    if isinstance(value, str):
        if len(value) <= max_length:
            return value
        return f"{value[: max_length - 3]}..."
    if isinstance(value, list):
        return [_truncate_argument_value(item, max_length=max_length) for item in value[:5]]
    if isinstance(value, dict):
        items = list(value.items())[:8]
        return {
            key: _truncate_argument_value(item_value, max_length=max_length)
            for key, item_value in items
        }
    return value
