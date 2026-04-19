"""Tests for tool approval and tool policy matching logic."""

from pathlib import Path

from clavi_agent.agent_template_models import ApprovalPolicy, DelegationPolicy, WorkspacePolicy
from clavi_agent.tool_execution import detect_tool_artifacts
from clavi_agent.tool_execution import detect_tool_touched_paths
from clavi_agent.tools.bash_tool import BashTool
from clavi_agent.tools.bash_tool import BashOutputResult
from clavi_agent.tool_execution import prepare_tool_execution
from clavi_agent.tool_execution import UploadedFileTarget
from clavi_agent.tools.base import Tool, ToolResult


class CustomReviewTool(Tool):
    """Simple custom tool used to validate class-name policy matching."""

    @property
    def name(self) -> str:
        return "custom_review"

    @property
    def description(self) -> str:
        return "Runs a custom review action."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self) -> ToolResult:
        return ToolResult(success=True, content="reviewed")


class CustomWriteTool(Tool):
    """Write-like tool with a workspace root used to validate writable-root policy checks."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Writes one file."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str) -> ToolResult:
        return ToolResult(success=True, content=f"wrote {path}")


class CustomReadTool(Tool):
    """Read-like tool with a workspace root used to validate readable-root policy checks."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Reads one file."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        }

    async def execute(self, path: str) -> ToolResult:
        return ToolResult(success=True, content=f"read {path}")


def test_prepare_tool_execution_requires_approval_for_explicit_function_name():
    execution = prepare_tool_execution(
        tool=None,
        function_name="write_file",
        tool_call_id="call-1",
        arguments={"path": "notes/output.md", "content": "hello"},
        approval_policy=ApprovalPolicy(
            mode="default",
            require_approval_tools=["write_file"],
        ),
    )

    assert execution.risk_category == "filesystem_write"
    assert execution.risk_level == "high"
    assert execution.requires_approval is True


def test_prepare_tool_execution_requires_approval_for_uploaded_original_overwrite(tmp_path):
    workspace_dir = tmp_path / "workspace"
    upload_path = workspace_dir / ".clavi_agent" / "uploads" / "session-1" / "upload-1" / "draft.md"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_text("# Draft\n", encoding="utf-8")

    execution = prepare_tool_execution(
        tool=CustomWriteTool(workspace_dir),
        function_name="write_file",
        tool_call_id="call-upload-overwrite",
        arguments={"path": ".clavi_agent/uploads/session-1/upload-1/draft.md", "content": "updated"},
        uploaded_file_targets=[
            UploadedFileTarget(
                upload_id="upload-1",
                upload_name="draft.md",
                absolute_path=str(upload_path),
                relative_path=".clavi_agent/uploads/session-1/upload-1/draft.md",
            )
        ],
    )

    assert execution.risk_category == "uploaded_original_overwrite"
    assert execution.risk_level == "critical"
    assert execution.requires_approval is True
    assert execution.approval_reason == "uploaded_original_overwrite"
    assert execution.parameter_summary == "覆盖已上传原文件：draft.md"
    assert "需要审批确认" in execution.impact_summary


def test_prepare_tool_execution_allows_explicit_auto_approve_for_uploaded_original_overwrite(tmp_path):
    workspace_dir = tmp_path / "workspace"
    upload_path = workspace_dir / ".clavi_agent" / "uploads" / "session-1" / "upload-1" / "draft.md"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_text("# Draft\n", encoding="utf-8")

    execution = prepare_tool_execution(
        tool=CustomWriteTool(workspace_dir),
        function_name="write_file",
        tool_call_id="call-upload-overwrite-auto",
        arguments={"path": ".clavi_agent/uploads/session-1/upload-1/draft.md", "content": "updated"},
        approval_policy=ApprovalPolicy(
            mode="default",
            auto_approve_tools=["write_file"],
        ),
        uploaded_file_targets=[
            UploadedFileTarget(
                upload_id="upload-1",
                upload_name="draft.md",
                absolute_path=str(upload_path),
                relative_path=".clavi_agent/uploads/session-1/upload-1/draft.md",
            )
        ],
    )

    assert execution.risk_category == "uploaded_original_overwrite"
    assert execution.requires_approval is False
    assert execution.parameter_summary == "覆盖已上传原文件：draft.md"


def test_prepare_tool_execution_describes_revised_copy_and_shell_export_for_uploaded_files(tmp_path):
    workspace_dir = tmp_path / "workspace"
    upload_path = workspace_dir / ".clavi_agent" / "uploads" / "session-1" / "upload-1" / "review.docx"
    upload_path.parent.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(b"docx")
    upload_target = UploadedFileTarget(
        upload_id="upload-1",
        upload_name="review.docx",
        absolute_path=str(upload_path),
        relative_path=".clavi_agent/uploads/session-1/upload-1/review.docx",
    )

    revised_execution = prepare_tool_execution(
        tool=CustomWriteTool(workspace_dir),
        function_name="write_file",
        tool_call_id="call-upload-revised",
        arguments={"path": ".clavi_agent/uploads/session-1/upload-1/review.revised.docx", "content": "updated"},
        uploaded_file_targets=[upload_target],
    )
    export_execution = prepare_tool_execution(
        tool=BashTool(str(workspace_dir)),
        function_name="bash",
        tool_call_id="call-upload-export",
        arguments={
            "command": (
                "python scripts/export.py "
                ".clavi_agent/uploads/session-1/upload-1/review.docx "
                "--output exports/review.pdf"
            )
        },
        uploaded_file_targets=[upload_target],
    )

    assert revised_execution.parameter_summary == "创建修订文件：review.revised.docx"
    assert "保留原始文件不变" in revised_execution.impact_summary
    assert export_execution.parameter_summary == "导出新交付文件：review.pdf"
    assert "不会覆盖原始文件" in export_execution.impact_summary


def test_prepare_tool_execution_requires_approval_for_explicit_tool_class_name():
    execution = prepare_tool_execution(
        tool=CustomReviewTool(),
        function_name="custom_review",
        tool_call_id="call-2",
        arguments={},
        approval_policy=ApprovalPolicy(
            mode="default",
            require_approval_tools=["CustomReviewTool"],
        ),
    )

    assert execution.risk_category == "other"
    assert execution.risk_level == "low"
    assert execution.requires_approval is True


def test_prepare_tool_execution_strict_mode_requires_medium_risk_delegate_calls():
    execution = prepare_tool_execution(
        tool=None,
        function_name="delegate_task",
        tool_call_id="call-3",
        arguments={"task": "inspect trace tree", "persona": "worker"},
        approval_policy=ApprovalPolicy(mode="strict"),
    )

    assert execution.risk_category == "delegate"
    assert execution.risk_level == "medium"
    assert execution.requires_approval is True


def test_prepare_tool_execution_strict_mode_keeps_low_risk_read_calls_auto_allowed():
    execution = prepare_tool_execution(
        tool=None,
        function_name="read_file",
        tool_call_id="call-4",
        arguments={"path": "task.md"},
        approval_policy=ApprovalPolicy(mode="strict"),
    )

    assert execution.risk_category == "filesystem_read"
    assert execution.risk_level == "low"
    assert execution.requires_approval is False


def test_prepare_tool_execution_auto_approve_tools_override_explicit_approval_requirement():
    execution = prepare_tool_execution(
        tool=None,
        function_name="write_file",
        tool_call_id="call-5",
        arguments={"path": "docs/approved.md", "content": "hello"},
        approval_policy=ApprovalPolicy(
            mode="default",
            require_approval_tools=["write_file"],
            auto_approve_tools=["write_file"],
        ),
    )

    assert execution.risk_category == "filesystem_write"
    assert execution.risk_level == "high"
    assert execution.requires_approval is False


def test_prepare_tool_execution_auto_approve_tools_override_strict_mode():
    execution = prepare_tool_execution(
        tool=None,
        function_name="delegate_task",
        tool_call_id="call-6",
        arguments={"task": "inspect trace tree", "persona": "worker"},
        approval_policy=ApprovalPolicy(
            mode="strict",
            auto_approve_tools=["delegate_task"],
        ),
    )

    assert execution.risk_category == "delegate"
    assert execution.risk_level == "medium"
    assert execution.requires_approval is False


def test_prepare_tool_execution_requires_approval_for_explicit_risk_level():
    execution = prepare_tool_execution(
        tool=None,
        function_name="write_file",
        tool_call_id="call-risk-level",
        arguments={"path": "docs/policy.md", "content": "hello"},
        approval_policy=ApprovalPolicy(
            mode="default",
            require_approval_risk_levels=["high"],
        ),
    )

    assert execution.requires_approval is True
    assert execution.approval_reason == "risk_level:high"


def test_prepare_tool_execution_requires_approval_for_explicit_risk_category(tmp_path):
    execution = prepare_tool_execution(
        tool=BashTool(str(tmp_path)),
        function_name="bash",
        tool_call_id="call-risk-category",
        arguments={"command": "curl https://api.example.com/health"},
        approval_policy=ApprovalPolicy(
            mode="default",
            require_approval_risk_categories=["external_network"],
        ),
    )

    assert execution.risk_category == "external_network"
    assert execution.requires_approval is True
    assert execution.approval_reason == "risk_category:external_network"


def test_prepare_tool_execution_blocks_write_outside_writable_roots(tmp_path):
    execution = prepare_tool_execution(
        tool=CustomWriteTool(tmp_path),
        function_name="write_file",
        tool_call_id="call-7",
        arguments={"path": "notes/output.md", "content": "hello"},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            writable_roots=["docs"],
        ),
    )

    assert execution.policy_allowed is False
    assert "allowed writable roots" in execution.policy_denied_reason


def test_prepare_tool_execution_allows_write_inside_writable_roots(tmp_path):
    execution = prepare_tool_execution(
        tool=CustomWriteTool(tmp_path),
        function_name="write_file",
        tool_call_id="call-8",
        arguments={"path": "docs/output.md", "content": "hello"},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            writable_roots=["docs"],
        ),
    )

    assert execution.policy_allowed is True
    assert execution.policy_denied_reason == ""


def test_prepare_tool_execution_blocks_read_outside_readable_roots(tmp_path):
    execution = prepare_tool_execution(
        tool=CustomReadTool(tmp_path),
        function_name="read_file",
        tool_call_id="call-9",
        arguments={"path": "notes/output.md"},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            readable_roots=["docs"],
        ),
    )

    assert execution.policy_allowed is False
    assert "allowed readable roots" in execution.policy_denied_reason


def test_prepare_tool_execution_blocks_read_only_tool_from_writing(tmp_path):
    execution = prepare_tool_execution(
        tool=CustomWriteTool(tmp_path),
        function_name="write_file",
        tool_call_id="call-10",
        arguments={"path": "docs/output.md", "content": "hello"},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            read_only_tools=["write_file"],
        ),
    )

    assert execution.policy_allowed is False
    assert "read-only access" in execution.policy_denied_reason


def test_prepare_tool_execution_blocks_disabled_tool(tmp_path):
    execution = prepare_tool_execution(
        tool=CustomReviewTool(),
        function_name="custom_review",
        tool_call_id="call-11",
        arguments={},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            disabled_tools=["custom_review"],
        ),
    )

    assert execution.policy_allowed is False
    assert "disabled by the current template policy" in execution.policy_denied_reason


def test_prepare_tool_execution_blocks_shell_command_outside_allowed_prefixes(tmp_path):
    execution = prepare_tool_execution(
        tool=BashTool(str(tmp_path)),
        function_name="bash",
        tool_call_id="call-12",
        arguments={"command": "git status"},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            allowed_shell_command_prefixes=["python -m pytest"],
        ),
    )

    assert execution.policy_allowed is False
    assert "allowed command prefixes" in execution.policy_denied_reason


def test_prepare_tool_execution_blocks_network_domain_outside_allowlist(tmp_path):
    execution = prepare_tool_execution(
        tool=BashTool(str(tmp_path)),
        function_name="bash",
        tool_call_id="call-13",
        arguments={"command": "curl https://evil.example.net/health"},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            allowed_network_domains=["example.com"],
        ),
    )

    assert execution.risk_category == "external_network"
    assert execution.policy_allowed is False
    assert "allowed network domains" in execution.policy_denied_reason


def test_prepare_tool_execution_blocks_main_agent_write_in_supervisor_only_mode(tmp_path):
    execution = prepare_tool_execution(
        tool=CustomWriteTool(tmp_path),
        function_name="write_file",
        tool_call_id="call-supervisor-write",
        arguments={"path": "docs/output.md", "content": "hello"},
        delegation_policy=DelegationPolicy(mode="supervisor_only"),
        is_main_agent=True,
    )

    assert execution.policy_allowed is False
    assert "blocks direct file writes" in execution.policy_denied_reason


def test_prepare_tool_execution_blocks_main_agent_shell_when_delegate_required(tmp_path):
    execution = prepare_tool_execution(
        tool=BashTool(str(tmp_path)),
        function_name="bash",
        tool_call_id="call-supervisor-shell",
        arguments={"command": "git status"},
        delegation_policy=DelegationPolicy(
            mode="prefer_delegate",
            require_delegate_for_shell=True,
        ),
        is_main_agent=True,
    )

    assert execution.policy_allowed is False
    assert "blocks direct shell execution" in execution.policy_denied_reason


def test_prepare_tool_execution_allows_worker_write_in_supervisor_only_mode(tmp_path):
    execution = prepare_tool_execution(
        tool=CustomWriteTool(tmp_path),
        function_name="write_file",
        tool_call_id="call-worker-write",
        arguments={"path": "docs/output.md", "content": "hello"},
        delegation_policy=DelegationPolicy(mode="supervisor_only"),
        is_main_agent=False,
    )

    assert execution.policy_allowed is True


def test_prepare_tool_execution_allows_network_domain_inside_allowlist(tmp_path):
    execution = prepare_tool_execution(
        tool=BashTool(str(tmp_path)),
        function_name="bash",
        tool_call_id="call-14",
        arguments={"command": "curl https://api.example.com/health"},
        workspace_policy=WorkspacePolicy(
            mode="isolated",
            allowed_network_domains=["example.com"],
        ),
    )

    assert execution.risk_category == "external_network"
    assert execution.policy_allowed is True


def test_detect_tool_artifacts_tracks_shell_output_files_with_explicit_output_flags(tmp_path):
    workspace_dir = tmp_path / "workspace"
    output_path = workspace_dir / "docs" / "report.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"%PDF-1.4\n% test pdf\n")

    artifacts = detect_tool_artifacts(
        tool=BashTool(str(workspace_dir)),
        function_name="bash",
        arguments={
            "command": 'python scripts/export.py --output "docs/report.pdf"',
            "run_in_background": False,
        },
        result=BashOutputResult(
            success=True,
            stdout="saved to docs/report.pdf",
            stderr="",
            exit_code=0,
        ),
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.artifact_type == "workspace_file"
    assert artifact.display_name == "report.pdf"
    assert artifact.format == "pdf"
    assert artifact.preview_kind == "pdf"
    assert artifact.mime_type == "application/pdf"
    assert artifact.metadata["source_tool"] == "bash"
    assert artifact.metadata["artifact_detection"] == "shell_output_path"


def test_detect_tool_artifacts_uses_stdout_to_avoid_treating_input_files_as_outputs(tmp_path):
    workspace_dir = tmp_path / "workspace"
    input_path = workspace_dir / "docs" / "draft.docx"
    output_path = workspace_dir / "docs" / "draft.reviewed.pdf"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"%PDF-1.4\n% reviewed pdf\n")

    artifacts = detect_tool_artifacts(
        tool=BashTool(str(workspace_dir)),
        function_name="bash",
        arguments={
            "command": "python scripts/convert.py docs/draft.docx docs/draft.reviewed.pdf",
            "run_in_background": False,
        },
        result=BashOutputResult(
            success=True,
            stdout="Successfully exported reviewed file to docs/draft.reviewed.pdf",
            stderr="",
            exit_code=0,
        ),
    )

    assert len(artifacts) == 1
    assert artifacts[0].display_name == "draft.reviewed.pdf"


def test_detect_tool_touched_paths_tracks_file_reads_and_shell_paths(tmp_path):
    workspace_dir = tmp_path / "workspace"
    feature_dir = workspace_dir / "modules" / "feature"
    feature_dir.mkdir(parents=True, exist_ok=True)
    target_file = feature_dir / "src" / "main.md"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("# main\n", encoding="utf-8")
    output_file = feature_dir / "report.md"
    output_file.write_text("# report\n", encoding="utf-8")

    read_paths = detect_tool_touched_paths(
        tool=CustomReadTool(workspace_dir),
        function_name="read_file",
        arguments={"path": "modules/feature/src/main.md"},
        result=ToolResult(success=True, content="ok"),
    )
    shell_paths = detect_tool_touched_paths(
        tool=BashTool(str(workspace_dir)),
        function_name="bash",
        arguments={
            "command": "python tools/export.py modules/feature/src/main.md --output modules/feature/report.md"
        },
        result=BashOutputResult(success=True, stdout="", stderr="", exit_code=0),
    )

    assert read_paths == [target_file.resolve()]
    assert target_file.resolve() in shell_paths
    assert output_file.resolve() in shell_paths

