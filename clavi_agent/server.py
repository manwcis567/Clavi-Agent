"""Clavi Agent Web Server - FastAPI + SSE for Web chat interface."""

import asyncio
import contextlib
from contextvars import ContextVar
from email.parser import BytesParser
from email.policy import default
import inspect
import json
import mimetypes
import os
import re
import shutil
import subprocess
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .account_models import AuthenticatedAccountSession
from .account_constants import ROOT_ACCOUNT_ID
from .agent_runtime import AgentRuntimeContext
from .config import Config
from .file_previews import (
    INLINE_FILE_PREVIEW_KINDS,
    TEXT_PREVIEW_BYTE_LIMIT,
    TEXT_PREVIEW_KINDS,
    guess_preview_kind,
    read_text_preview,
)
from .integration_models import prepare_bounded_json_payload
from .integrations import (
    IntegrationGateway,
    IntegrationGatewayError,
    IntegrationRouter,
    IntegrationRunBridge,
    IntegrationRunBridgeError,
    QuickAckIntent,
)
from .integrations.admin_service import IntegrationAdminService
from .integrations.feishu_long_connection import FeishuLongConnectionService
from .integrations.wechat_ilink import (
    WeChatILinkCredentials,
    fetch_login_qr_code,
    poll_login_status,
)
from .integrations.wechat_long_poll import WeChatLongPollService
from .runtime_tools import resolve_clawhub_command_prefix
from .session_models import SessionHistorySearchResult
from .scheduled_task_service import ScheduledTaskService
from .schema import Message
from .session import SessionManager
from .upload_models import UploadCreatePayload, UploadRecord

# --- Request / Response Models ---
class CreateSessionRequest(BaseModel):
    workspace_dir: str | None = None
    agent_id: str | None = None


class SessionAgentUpdateRequest(BaseModel):
    agent_id: str


class SessionSummary(BaseModel):
    session_id: str
    title: str
    workspace_dir: str
    agent_id: str | None = None
    created_at: str
    updated_at: str
    message_count: int
    last_message_preview: str


class CreateSessionResponse(SessionSummary):
    """Response for creating a session."""


class ChatRequest(BaseModel):
    message: str = ""
    attachment_ids: list[str] = Field(default_factory=list)


class CreateRunRequest(BaseModel):
    session_id: str
    goal: str
    parent_run_id: str | None = None
    workspace_policy: dict | None = None
    approval_policy: dict | None = None
    run_policy: dict | None = None
    delegation_policy: dict | None = None


class ApprovalDecisionRequest(BaseModel):
    decision_notes: str = ""
    decision_scope: Literal["once", "run", "template"] = "once"


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class SetupConfigRequest(BaseModel):
    name: str = "Default"
    api_key: str
    api_base: str = ""
    model: str = "MiniMax-M2"
    provider: Literal["anthropic", "openai"] = "anthropic"
    reasoning_enabled: bool = False
    llm_routing_policy: dict | None = None
    activate: bool = True


class ApiRoutingUpdateRequest(BaseModel):
    planner_api_config_id: str | None = None
    worker_api_config_id: str | None = None


class SetupStatusResponse(BaseModel):
    setup_required: bool
    runtime_ready: bool
    config_path: str
    api_key_configured: bool
    provider: str
    api_base: str
    model: str
    reasoning_enabled: bool
    config_count: int = 0
    active_config_id: str | None = None
    active_config_name: str | None = None
    message: str = ""


class ApiConfigSummaryResponse(BaseModel):
    id: str
    name: str
    provider: str
    api_base: str
    model: str
    reasoning_enabled: bool
    llm_routing_policy: dict = Field(default_factory=dict)
    is_active: bool
    masked_api_key: str
    created_at: str
    updated_at: str
    last_used_at: str | None = None


class AuthAccountResponse(BaseModel):
    id: str
    username: str
    display_name: str
    status: str
    is_root: bool
    created_at: str
    updated_at: str


class AuthSessionResponse(BaseModel):
    account: AuthAccountResponse
    expires_at: str


class UserProfileFieldMetaResponse(BaseModel):
    source: str
    confidence: float
    source_session_id: str | None = None
    source_run_id: str | None = None
    writer_type: str
    writer_id: str
    updated_at: str


class UserProfileResponse(BaseModel):
    user_id: str
    profile: dict[str, object] = Field(default_factory=dict)
    field_meta: dict[str, UserProfileFieldMetaResponse] = Field(default_factory=dict)
    normalized_profile: dict[str, object] = Field(default_factory=dict)
    summary: str = ""
    writer_type: str
    writer_id: str
    created_at: str
    updated_at: str


class UserMemoryEntryResponse(BaseModel):
    id: str
    user_id: str
    memory_type: str
    content: str
    summary: str = ""
    source_session_id: str | None = None
    source_run_id: str | None = None
    writer_type: str
    writer_id: str
    confidence: float
    created_at: str
    updated_at: str
    superseded_by: str | None = None


class MemoryAuditEventResponse(BaseModel):
    id: str
    user_id: str
    target_scope: str
    target_id: str
    action: str
    writer_type: str
    writer_id: str
    session_id: str | None = None
    run_id: str | None = None
    payload: dict[str, object] = Field(default_factory=dict)
    created_at: str


class UserProfileUpdateRequest(BaseModel):
    profile_updates: dict[str, object] = Field(default_factory=dict)
    remove_fields: list[str] = Field(default_factory=list)
    summary: str | None = None
    profile_source: str = "explicit"
    profile_confidence: float = 1.0


class UserMemoryUpdateRequest(BaseModel):
    memory_type: str | None = None
    content: str | None = None
    summary: str | None = None
    confidence: float | None = None


class MutationStatusResponse(BaseModel):
    status: str
    target_id: str


class MemoryProviderHealthResponse(BaseModel):
    configured_provider: str
    active_provider: str
    status: str
    fallback_active: bool
    inject_memories: bool
    expose_tools: bool
    sync_conversation_turns: bool
    capabilities: dict[str, bool] = Field(default_factory=dict)
    message: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


class LearnedWorkflowCandidateResponse(BaseModel):
    id: str
    account_id: str
    run_id: str
    session_id: str
    agent_template_id: str
    status: str
    title: str
    summary: str = ""
    description: str = ""
    signal_types: list[str] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    step_titles: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    suggested_skill_name: str = ""
    generated_skill_markdown: str = ""
    review_notes: str = ""
    installed_agent_id: str | None = None
    installed_skill_path: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    approved_at: str | None = None
    rejected_at: str | None = None
    installed_at: str | None = None


class LearnedWorkflowReviewRequest(BaseModel):
    review_notes: str = ""


class LearnedWorkflowInstallRequest(BaseModel):
    agent_id: str
    skill_name: str = ""


class SkillImprovementProposalResponse(BaseModel):
    id: str
    account_id: str
    run_id: str
    session_id: str
    agent_template_id: str
    skill_name: str
    target_skill_path: str
    status: str
    title: str
    summary: str = ""
    signal_types: list[str] = Field(default_factory=list)
    source_run_ids: list[str] = Field(default_factory=list)
    base_version: int
    proposed_version: int
    current_skill_markdown: str = ""
    proposed_skill_markdown: str = ""
    changelog_entry: str = ""
    review_notes: str = ""
    applied_skill_path: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    approved_at: str | None = None
    rejected_at: str | None = None
    applied_at: str | None = None


class SkillImprovementReviewRequest(BaseModel):
    review_notes: str = ""


# 账号管理请求/响应模型（仅 root 账号可用）
class AccountCreateRequest(BaseModel):
    """创建新账号的请求体。"""
    username: str
    password: str
    display_name: str = ""


class AccountUpdateRequest(BaseModel):
    """更新账号信息的请求体。"""
    display_name: str | None = None
    status: str | None = None  # active / disabled


class AccountResetPasswordRequest(BaseModel):
    """Root 重置其他账号密码的请求体。"""
    new_password: str


class AccountSummaryResponse(BaseModel):
    """账号列表/详情响应体。"""
    id: str
    username: str
    display_name: str
    status: str
    is_root: bool
    created_at: str
    updated_at: str


class BindingCreateRequest(BaseModel):
    tenant_id: str = ""
    chat_id: str
    thread_id: str = ""
    binding_scope: Literal["chat", "thread"]
    agent_id: str
    create_new_session: bool = False
    metadata: dict = Field(default_factory=dict)


class BindingUpdateRequest(BaseModel):
    tenant_id: str | None = None
    chat_id: str | None = None
    thread_id: str | None = None
    binding_scope: Literal["chat", "thread"] | None = None
    agent_id: str | None = None
    enabled: bool | None = None
    refresh_session: bool = False
    metadata: dict | None = None


class BindingResponse(BaseModel):
    id: str
    integration_id: str
    tenant_id: str = ""
    chat_id: str = ""
    thread_id: str = ""
    binding_scope: str
    agent_id: str
    session_id: str
    enabled: bool
    metadata: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str
    last_message_at: str | None = None
    session_key: str = ""


class IntegrationCredentialInput(BaseModel):
    credential_key: str
    storage_kind: Literal["env", "external_ref", "local_encrypted"] = "env"
    secret_ref: str = ""
    secret_value: str = ""
    metadata: dict = Field(default_factory=dict)


class IntegrationCredentialResponse(BaseModel):
    id: str
    credential_key: str
    storage_kind: str
    secret_ref: str = ""
    masked_value: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str


class IntegrationCreateRequest(BaseModel):
    name: str
    kind: str
    display_name: str = ""
    tenant_id: str = ""
    config: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    credentials: list[IntegrationCredentialInput] = Field(default_factory=list)
    enabled: bool | None = None


class IntegrationUpdateRequest(BaseModel):
    name: str | None = None
    kind: str | None = None
    display_name: str | None = None
    tenant_id: str | None = None
    config: dict | None = None
    metadata: dict | None = None
    credentials: list[IntegrationCredentialInput] | None = None


class WeChatSetupStatusResponse(BaseModel):
    integration_id: str = ""
    state: str = "idle"
    message: str = ""
    output: str = ""
    qr_text: str = ""
    qr_content: str = ""
    error: str = ""
    ilink_bot_id: str = ""
    ilink_user_id: str = ""
    base_url: str = ""
    updated_at: str | None = None


class IntegrationResponse(BaseModel):
    id: str
    name: str
    kind: str
    status: str
    display_name: str = ""
    tenant_id: str = ""
    webhook_path: str
    config: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    credentials: list[IntegrationCredentialResponse] = Field(default_factory=list)
    created_at: str
    updated_at: str
    last_verified_at: str | None = None
    last_error: str = ""
    binding_count: int = 0
    rule_count: int = 0
    last_event_at: str | None = None
    last_delivery_at: str | None = None
    deleted: bool = False
    setup_status: WeChatSetupStatusResponse | None = None


class IntegrationVerificationResponse(BaseModel):
    success: bool
    message: str
    integration: IntegrationResponse


class RoutingRuleCreateRequest(BaseModel):
    priority: int = 100
    match_type: Literal["integration_id", "chat_id", "thread_id"]
    match_value: str
    agent_id: str
    session_strategy: Literal["reuse", "chat", "thread"] = "reuse"
    enabled: bool = True
    metadata: dict = Field(default_factory=dict)


class RoutingRuleUpdateRequest(BaseModel):
    priority: int | None = None
    match_type: Literal["integration_id", "chat_id", "thread_id"] | None = None
    match_value: str | None = None
    agent_id: str | None = None
    session_strategy: Literal["reuse", "chat", "thread"] | None = None
    enabled: bool | None = None
    metadata: dict | None = None


class RoutingRuleResponse(BaseModel):
    id: str
    integration_id: str
    priority: int
    match_type: str
    match_value: str
    agent_id: str
    session_strategy: str
    enabled: bool
    metadata: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str


class InboundEventResponse(BaseModel):
    id: str
    integration_id: str
    provider_event_id: str = ""
    provider_message_id: str = ""
    provider_chat_id: str = ""
    provider_thread_id: str = ""
    provider_user_id: str = ""
    event_type: str
    received_at: str
    signature_valid: bool
    normalized_status: str
    normalized_error: str = ""
    metadata: dict = Field(default_factory=dict)


class DeliveryAttemptResponse(BaseModel):
    id: str
    delivery_id: str
    attempt_number: int
    status: str
    error_summary: str = ""
    started_at: str
    finished_at: str | None = None


class OutboundDeliveryResponse(BaseModel):
    id: str
    integration_id: str
    run_id: str
    session_id: str
    inbound_event_id: str | None = None
    provider_chat_id: str
    provider_thread_id: str = ""
    provider_message_id: str = ""
    delivery_type: str
    payload: dict | list | str | int | float | bool | None = Field(default_factory=dict)
    status: str
    attempt_count: int
    last_attempt_at: str | None = None
    error_summary: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str
    attempts: list[DeliveryAttemptResponse] = Field(default_factory=list)


static_dir = Path(__file__).parent / "web" / "static"


class SessionDetail(SessionSummary):
    messages: list[Message]


class SharedContextEntry(BaseModel):
    id: str
    timestamp: str
    source: str
    category: str
    title: str
    content: str


class SharedContextResponse(BaseModel):
    session_id: str
    entries: list[SharedContextEntry]


class SessionUploadsResponse(BaseModel):
    session_id: str
    uploads: list[UploadRecord]


class FilePreviewResponse(BaseModel):
    target_kind: Literal["upload", "artifact"]
    target_id: str
    display_name: str
    mime_type: str = "application/octet-stream"
    preview_kind: str = "none"
    size_bytes: int | None = None
    preview_supported: bool = False
    truncated: bool = False
    text_content: str | None = None
    inline_url: str = ""
    open_url: str = ""
    download_url: str = ""
    note: str = ""


class AgentCreateRequest(BaseModel):
    name: str
    description: str = ""
    system_prompt: str
    selected_skill_packages: list[str] = []
    mcp_configs: list[dict] = []
    workspace_type: str = "isolated"
    workspace_policy: dict | None = None
    approval_policy: dict | None = None
    run_policy: dict | None = None
    delegation_policy: dict | None = None


class AgentUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    selected_skill_packages: list[str] | None = None
    mcp_configs: list[dict] | None = None
    workspace_type: str | None = None
    workspace_policy: dict | None = None
    approval_policy: dict | None = None
    run_policy: dict | None = None
    delegation_policy: dict | None = None


class BrainstormRequest(BaseModel):
    name: str = ""
    description: str = ""
    system_prompt: str = ""


class SkillInstallRequest(BaseModel):
    package_names: list[str] = []


class ScheduledTaskCreateRequest(BaseModel):
    name: str
    cron_expression: str
    agent_id: str | None = None
    prompt: str
    integration_id: str | None = None
    target_chat_id: str = ""
    target_thread_id: str = ""
    reply_to_message_id: str = ""
    timezone: str = "server_local"
    enabled: bool = True
    metadata: dict = Field(default_factory=dict)


class ScheduledTaskUpdateRequest(BaseModel):
    name: str | None = None
    cron_expression: str | None = None
    agent_id: str | None = None
    prompt: str | None = None
    integration_id: str | None = None
    target_chat_id: str | None = None
    target_thread_id: str | None = None
    reply_to_message_id: str | None = None
    timezone: str | None = None
    enabled: bool | None = None
    metadata: dict | None = None


class ScheduledTaskResponse(BaseModel):
    id: str
    name: str
    cron_expression: str
    timezone: str
    agent_id: str
    prompt: str
    integration_id: str | None = None
    target_chat_id: str = ""
    target_thread_id: str = ""
    reply_to_message_id: str = ""
    enabled: bool
    session_id: str | None = None
    next_run_at: str | None = None
    last_scheduled_for: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str
    resolved_target_chat_id: str = ""
    resolved_target_thread_id: str = ""
    integration_status: str = ""
    integration_kind: str = ""
    integration_display_name: str = ""
    last_execution: dict | None = None


class ScheduledTaskExecutionResponse(BaseModel):
    id: str
    task_id: str
    trigger_kind: str
    scheduled_for: str | None = None
    run_id: str | None = None
    status: str
    error_summary: str = ""
    metadata: dict = Field(default_factory=dict)
    created_at: str
    updated_at: str
    run_status: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    run_error_summary: str = ""
    delivery_status: str = ""
    delivery_error_summary: str = ""
    delivery_count: int = 0
    session_id: str = ""
    task_name: str = ""
    integration_id: str | None = None


class ScheduledTaskExecutionDetailResponse(BaseModel):
    task: ScheduledTaskResponse | None = None
    execution: ScheduledTaskExecutionResponse
    run: dict | None = None
    steps: list[dict] = Field(default_factory=list)
    timeline: list[dict] = Field(default_factory=list)
    tree: dict | None = None
    tools: list[dict] = Field(default_factory=list)
    artifacts: list[dict] = Field(default_factory=list)
    deliveries: list[dict] = Field(default_factory=list)


CLAWHUB_REGISTRY = "https://cn.clawhub-mirror.com"
ACTIVE_SKILL_INSTALL_STATES = {"queued", "running"}
WECHAT_SETUP_ACTIVE_STATES = {"queued", "running", "waiting_scan", "scanned"}


AVAILABLE_TOOLS = [
    {"name": "BashTool", "description": "执行本地 Bash 终端指令"},
    {"name": "BashOutputTool", "description": "获取后台执行的 Bash 产出片段"},
    {"name": "BashKillTool", "description": "强制终止运行中的终端进程"},
    {"name": "ReadTool", "description": "读取文件内容"},
    {"name": "WriteTool", "description": "覆盖写入或创建新文件"},
    {"name": "EditTool", "description": "通过区块补丁局部修改文件"},
    {"name": "SessionNoteTool", "description": "管理并记忆跨会话的持续性笔记"},
    {"name": "RecallNoteTool", "description": "回忆结构化长期记忆和本地笔记摘要"},
    {"name": "SearchMemoryTool", "description": "搜索用户长期记忆和本地笔记明细"},
    {"name": "SearchSessionHistoryTool", "description": "检索当前账号的跨会话历史消息与运行结论"},
    {"name": "ShareContextTool", "description": "向其他 Agent 分发上下文情报"},
    {"name": "ReadSharedContextTool", "description": "读取所在工作区的汇总池情报"}
]

def _default_agent_tools(session_manager: SessionManager) -> list[str]:
    """Return the default tool set that every custom agent should receive."""
    if not session_manager._agent_store:
        return []

    default_agent = session_manager._agent_store.get_agent_template("system-default-agent")
    if not default_agent:
        return []

    return list(default_agent.get("tools", []))


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for install status payloads."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _resolve_clawhub_command_prefix() -> list[str]:
    """Resolve the clawhub executable path with config-aware fallbacks."""
    tool_overrides = Config.get_tool_path_overrides()
    configured_clawhub = os.environ.get("CLAWHUB_BIN", "").strip() or tool_overrides.get("clawhub_bin")
    configured_npm = os.environ.get("NPM_BIN", "").strip() or tool_overrides.get("npm_bin")
    return resolve_clawhub_command_prefix(
        clawhub_binary=configured_clawhub or None,
        npm_binary=configured_npm or None,
    )


def _tail_text(value: str, *, limit: int = 20000) -> str:
    normalized = str(value or "")
    if len(normalized) <= limit:
        return normalized
    return normalized[-limit:]


def _extract_terminal_qr(output: str) -> str:
    """Extract a QR-like unicode block from terminal output when available."""
    best_block: list[str] = []
    current_block: list[str] = []

    def is_qr_char(char: str) -> bool:
        if not char:
            return False
        codepoint = ord(char)
        return (
            char == " "
            or char == "█"
            or 0x2580 <= codepoint <= 0x259F
            or 0x25A0 <= codepoint <= 0x25FF
        )

    def flush_current() -> None:
        nonlocal best_block, current_block
        if len(current_block) >= 8:
            current_score = sum(len(line.rstrip()) for line in current_block)
            best_score = sum(len(line.rstrip()) for line in best_block)
            if current_score > best_score:
                best_block = list(current_block)
        current_block = []

    for raw_line in _strip_ansi_sequences(output).splitlines():
        line = raw_line.rstrip("\n")
        content = line.replace(" ", "")
        if len(content) >= 8 and all(is_qr_char(char) for char in content):
            current_block.append(line.rstrip())
            continue
        flush_current()

    flush_current()
    return "\n".join(best_block).strip()


def _parse_wechat_openclaw_version(output: str) -> str:
    normalized = _strip_ansi_sequences(output)
    match = re.search(r"OpenClaw(?:\s+版本)?[:：]\s*(\d+\.\d+\.\d+)", normalized)
    return match.group(1) if match else ""


def _parse_wechat_plugin_spec(output: str) -> str:
    normalized = _strip_ansi_sequences(output)
    match = re.search(r"(@tencent-weixin/openclaw-weixin@[^\s]+)", normalized)
    if not match:
        return ""
    return match.group(1).rstrip(".,;:!?)。…")


async def _run_wechat_installer(
    *,
    on_output=None,
    cwd: str | None = None,
) -> tuple[int, str]:
    """Run the official Tencent installer and stream combined output."""
    command = _resolve_wechat_installer_command()
    env = os.environ.copy()
    env.setdefault("NO_COLOR", "1")

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    chunks: list[str] = []
    try:
        while True:
            chunk = await process.stdout.read(1024) if process.stdout is not None else b""
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            chunks.append(text)
            if on_output is not None:
                maybe_awaitable = on_output(text)
                if inspect.isawaitable(maybe_awaitable):
                    await maybe_awaitable
        return_code = await process.wait()
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
            with contextlib.suppress(ProcessLookupError):
                await process.wait()
        raise

    normalized_output = _strip_ansi_sequences("".join(chunks))
    return return_code, normalized_output


def _decode_multipart_text(payload: bytes, charset: str | None) -> str:
    encoding = (charset or "utf-8").strip() or "utf-8"
    return payload.decode(encoding, errors="replace").strip()


def _build_quick_ack_response(intent: QuickAckIntent) -> Response:
    headers = {key: value for key, value in intent.headers.items()}
    if intent.body_type == "json":
        return JSONResponse(intent.body_json, status_code=intent.status_code, headers=headers)
    if intent.body_type == "text":
        return PlainTextResponse(intent.body_text, status_code=intent.status_code, headers=headers)
    return Response(status_code=intent.status_code, headers=headers)


async def _parse_upload_multipart_request(
    request: Request,
) -> tuple[list[UploadCreatePayload], str | None]:
    content_type = request.headers.get("content-type", "").strip()
    if not content_type.lower().startswith("multipart/form-data"):
        raise HTTPException(status_code=400, detail="Content-Type must be multipart/form-data")

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Upload body is empty")

    raw_message = (
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8")
        + body
    )
    message = BytesParser(policy=default).parsebytes(raw_message)
    if not message.is_multipart():
        raise HTTPException(status_code=400, detail="Invalid multipart upload payload")

    uploads: list[UploadCreatePayload] = []
    run_id: str | None = None

    for part in message.iter_parts():
        if part.get_content_disposition() != "form-data":
            continue

        field_name = str(part.get_param("name", header="content-disposition") or "").strip()
        payload = part.get_payload(decode=True) or b""
        filename = part.get_filename()

        if filename:
            uploads.append(
                UploadCreatePayload(
                    original_name=filename,
                    content_bytes=payload,
                    mime_type=part.get_content_type(),
                )
            )
            continue

        if field_name == "run_id":
            run_id = _decode_multipart_text(payload, part.get_content_charset()) or None

    if not uploads:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    return uploads, run_id


def _run_clawhub_command(args: list[str], cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run a clawhub command and capture text output."""
    command = _resolve_clawhub_command_prefix() + args
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def _classify_sse_event_name(event: dict) -> str:
    """Map one run event payload to a coarse SSE event class."""
    event_type = str(event.get("type", "")).strip().lower()
    if event_type in {"queued", "done", "error", "interrupted", "cancelled", "timed_out"}:
        return "state"
    return "ui"


def _encode_sse_message(
    *,
    data: str,
    event_name: str | None = None,
    event_id: int | None = None,
) -> str:
    """Serialize one SSE frame with optional id and event name."""
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event_name:
        lines.append(f"event: {event_name}")
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    lines.append("")
    return "\n".join(lines) + "\n"


def _parse_clawhub_search_output(stdout: str) -> list[dict]:
    """Parse `clawhub search` plain-text results into structured rows."""
    candidates: list[dict] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = re.split(r"\s{2,}", line)
        if len(parts) < 2:
            continue

        package_tokens = parts[0].split()
        if not package_tokens:
            continue

        package_name = package_tokens[0]
        version = package_tokens[1] if len(package_tokens) > 1 else ""
        description = parts[1].strip()
        score = ""
        if len(parts) > 2:
            score = parts[2].strip().strip("()")

        candidates.append(
            {
                "package_name": package_name,
                "version": version,
                "description": description,
                "score": score,
                "label": f"{package_name} {version}".strip(),
            }
        )

    return candidates


def _normalize_capability_dimensions(raw_dimensions: object) -> list[dict]:
    """Normalize brainstormed capability dimensions into stable search inputs."""
    if not isinstance(raw_dimensions, list):
        return []

    normalized: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for item in raw_dimensions:
        if isinstance(item, str):
            name = item.strip()
            keyword = name
            reason = ""
        elif isinstance(item, dict):
            name = str(
                item.get("name")
                or item.get("capability")
                or item.get("dimension")
                or item.get("keyword")
                or item.get("search_keyword")
                or ""
            ).strip()
            keyword = str(
                item.get("keyword")
                or item.get("search_keyword")
                or item.get("query")
                or name
            ).strip()
            reason = str(item.get("reason") or item.get("description") or "").strip()
        else:
            continue

        if not name or not keyword:
            continue

        dedupe_key = (name.lower(), keyword.lower())
        if dedupe_key in seen:
            continue

        normalized.append(
            {
                "name": name,
                "keyword": keyword,
                "reason": reason,
            }
        )
        seen.add(dedupe_key)

        if len(normalized) >= 4:
            break

    return normalized


def _extract_first_json_object(value: str) -> str | None:
    """Extract the first complete JSON object from free-form model output."""
    start_index: int | None = None
    depth = 0
    in_string = False
    escaping = False

    for index, char in enumerate(value):
        if start_index is None:
            if char == "{":
                start_index = index
                depth = 1
            continue

        if in_string:
            if escaping:
                escaping = False
            elif char == "\\":
                escaping = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start_index : index + 1]

    return None


def _parse_brainstorm_response_content(raw_content: str) -> dict:
    """Parse brainstorm JSON with tolerance for code fences and extra prose."""
    content = raw_content.strip()
    if not content:
        raise ValueError("LLM returned empty brainstorm content.")

    candidates: list[str] = [content]
    fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", content, flags=re.IGNORECASE | re.DOTALL)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    extracted = _extract_first_json_object(content)
    if extracted:
        candidates.append(extracted.strip())

    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        try:
            parsed = json.loads(normalized)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("LLM returned invalid JSON for brainstorm response.")


async def _search_capability_skill_groups(capability_dimensions: list[dict]) -> list[dict]:
    """Search skills concurrently for each capability dimension."""
    if not capability_dimensions:
        return []

    tasks = [
        _search_clawhub_skills(str(item.get("keyword", "")).strip())
        for item in capability_dimensions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    grouped_results: list[dict] = []
    for capability, result in zip(capability_dimensions, results, strict=False):
        group = {
            "capability_name": capability["name"],
            "keyword": capability["keyword"],
            "reason": capability.get("reason", ""),
            "skill_candidates": [],
            "skill_search_error": "",
        }
        if isinstance(result, Exception):
            group["skill_search_error"] = str(result)
        else:
            group["skill_candidates"] = result
        grouped_results.append(group)

    return grouped_results


async def _search_clawhub_skills(keyword: str) -> list[dict]:
    """Search the clawhub registry for skill candidates."""
    normalized = keyword.strip()
    if not normalized:
        return []

    result = await asyncio.to_thread(
        _run_clawhub_command,
        [
            "search",
            normalized,
            "--limit",
            "5",
            f"--registry={CLAWHUB_REGISTRY}",
        ],
    )
    return _parse_clawhub_search_output(result.stdout)


async def _install_skill_packages(
    session_manager: SessionManager,
    agent_id: str,
    package_names: list[str],
) -> list[dict]:
    """Install selected clawhub skills into the agent's dedicated skills directory."""
    if not session_manager._agent_store:
        return []

    selected: list[str] = []
    seen: set[str] = set()
    for item in package_names:
        package_name = item.strip()
        if not package_name or package_name in seen:
            continue
        selected.append(package_name)
        seen.add(package_name)

    if not selected:
        agent = session_manager._agent_store.get_agent_template(agent_id)
        return list(agent.get("skills", [])) if agent else []

    skills_dir = session_manager._agent_store.get_agent_skills_dir(agent_id)
    skills_dir.mkdir(parents=True, exist_ok=True)

    for package_name in selected:
        await asyncio.to_thread(
            _run_clawhub_command,
            [
                "install",
                package_name,
                "--workdir",
                str(skills_dir),
                f"--registry={CLAWHUB_REGISTRY}",
            ],
            str(skills_dir),
        )

    return session_manager._agent_store.collect_skills_metadata_from_directory(skills_dir)


def create_app(
    manager: SessionManager | None = None,
    *,
    enable_auth: bool | None = None,
) -> FastAPI:
    """Create the FastAPI app with an optional injected session manager."""
    session_manager = manager or SessionManager()
    auth_enabled = (manager is None) if enable_auth is None else bool(enable_auth)
    integration_gateway = IntegrationGateway(session_manager)
    integration_router = IntegrationRouter(session_manager)
    integration_run_bridge = IntegrationRunBridge(session_manager)
    integration_admin_service = IntegrationAdminService(session_manager)
    scheduled_task_service = ScheduledTaskService(session_manager)
    feishu_long_connection_manager = FeishuLongConnectionService(
        session_manager,
        integration_gateway=integration_gateway,
        integration_run_bridge=integration_run_bridge,
    )
    wechat_long_poll_manager = WeChatLongPollService(
        session_manager,
        integration_gateway=integration_gateway,
        integration_run_bridge=integration_run_bridge,
    )
    skill_install_statuses: dict[str, dict] = {}
    skill_install_tasks: dict[str, asyncio.Task] = {}
    wechat_setup_statuses: dict[str, dict] = {}
    wechat_setup_tasks: dict[str, asyncio.Task] = {}
    auth_session_var: ContextVar[AuthenticatedAccountSession | None] = ContextVar(
        "clavi_agent_auth_session",
        default=None,
    )

    def _feature_flags() -> dict[str, bool]:
        return session_manager.get_feature_flags()

    def _durable_runs_enabled() -> bool:
        return bool(_feature_flags().get("enable_durable_runs", True))

    def _run_trace_enabled() -> bool:
        return bool(_feature_flags().get("enable_run_trace", True))

    def _approval_flow_enabled() -> bool:
        return bool(_feature_flags().get("enable_approval_flow", True))

    def _session_retrieval_enabled() -> bool:
        return bool(_feature_flags().get("enable_session_retrieval", True))

    def _mask_api_key(api_key: str) -> str:
        normalized = str(api_key or "").strip()
        if not normalized:
            return ""
        if len(normalized) <= 8:
            return "*" * len(normalized)
        return f"{normalized[:4]}{'*' * max(4, len(normalized) - 8)}{normalized[-4:]}"

    def _api_config_summary_payload(record: dict) -> ApiConfigSummaryResponse:
        return ApiConfigSummaryResponse(
            id=str(record.get("id") or ""),
            name=str(record.get("name") or ""),
            provider=str(record.get("provider") or "anthropic"),
            api_base=str(record.get("api_base") or "https://api.minimax.io"),
            model=str(record.get("model") or "MiniMax-M2"),
            reasoning_enabled=bool(record.get("reasoning_enabled")),
            llm_routing_policy=dict(record.get("llm_routing_policy") or {}),
            is_active=bool(record.get("is_active")),
            masked_api_key=_mask_api_key(str(record.get("api_key") or "")),
            created_at=str(record.get("created_at") or ""),
            updated_at=str(record.get("updated_at") or ""),
            last_used_at=record.get("last_used_at"),
        )

    def _setup_status_payload(
        *,
        account_id: str | None = None,
        message: str = "",
    ) -> SetupStatusResponse:
        config = session_manager._config
        config_list = (
            session_manager.list_account_api_configs(account_id)
            if account_id
            else []
        )
        active_config = next(
            (item for item in config_list if bool(item.get("is_active"))),
            None,
        )
        api_key_configured = bool(active_config)
        if config is None:
            config_path = Config.get_bootstrap_config_path()
            return SetupStatusResponse(
                setup_required=not api_key_configured,
                runtime_ready=False,
                config_path=str(config_path),
                api_key_configured=api_key_configured,
                provider=str((active_config or {}).get("provider") or "anthropic"),
                api_base=str((active_config or {}).get("api_base") or "https://api.minimax.io"),
                model=str((active_config or {}).get("model") or "MiniMax-M2"),
                reasoning_enabled=bool((active_config or {}).get("reasoning_enabled")),
                config_count=len(config_list),
                active_config_id=str((active_config or {}).get("id") or "") or None,
                active_config_name=str((active_config or {}).get("name") or "") or None,
                message=message or "Clavi Agent is starting. Log in and bind an API Key to unlock execution.",
            )

        return SetupStatusResponse(
            setup_required=not api_key_configured,
            runtime_ready=session_manager.is_runtime_ready() and api_key_configured,
            config_path=str(session_manager._config_path or Config.get_default_config_path()),
            api_key_configured=api_key_configured,
            provider=str((active_config or {}).get("provider") or config.llm.provider or "anthropic"),
            api_base=str((active_config or {}).get("api_base") or config.llm.api_base or "https://api.minimax.io"),
            model=str((active_config or {}).get("model") or config.llm.model or "MiniMax-M2"),
            reasoning_enabled=bool(
                (active_config or {}).get("reasoning_enabled")
                if active_config is not None
                else config.llm.reasoning_enabled
            ),
            config_count=len(config_list),
            active_config_id=str((active_config or {}).get("id") or "") or None,
            active_config_name=str((active_config or {}).get("name") or "") or None,
            message=message
            or (
                ""
                if api_key_configured
                else (
                    "Bind an API Key for the current account to unlock chat and run features."
                    if account_id
                    else "Log in first, then bind an API Key for your account."
                )
            ),
        )

    def _auth_config():
        config = session_manager._config
        if config is None:
            raise RuntimeError("SessionManager 尚未加载配置。")
        return config.auth

    def _web_session_cookie_name() -> str:
        return _auth_config().web_session_cookie_name

    def _web_session_ttl_seconds() -> int:
        return int(_auth_config().web_session_ttl_hours) * 60 * 60

    def _next_web_session_expiry() -> str:
        return (
            datetime.now(timezone.utc) + timedelta(seconds=_web_session_ttl_seconds())
        ).isoformat(timespec="seconds")

    def _build_auth_account_response(auth_session: AuthenticatedAccountSession) -> AuthAccountResponse:
        account = auth_session.account
        return AuthAccountResponse(
            id=account.id,
            username=account.username,
            display_name=account.display_name,
            status=account.status,
            is_root=account.is_root,
            created_at=account.created_at,
            updated_at=account.updated_at,
        )

    def _optional_auth_session_from_request(request: Request) -> AuthenticatedAccountSession | None:
        if not auth_enabled:
            return None
        store = session_manager._account_store
        if store is None:
            return None
        raw_token = str(request.cookies.get(_web_session_cookie_name()) or "").strip()
        if not raw_token:
            return None
        return store.get_authenticated_session(raw_token)

    def _build_auth_session_response(
        auth_session: AuthenticatedAccountSession,
        *,
        expires_at: str | None = None,
    ) -> AuthSessionResponse:
        return AuthSessionResponse(
            account=_build_auth_account_response(auth_session),
            expires_at=expires_at or auth_session.web_session.expires_at,
        )

    def _set_auth_cookie(
        response: Response,
        *,
        request: Request,
        token: str,
        expires_at: str,
    ) -> None:
        expires_at_dt = datetime.fromisoformat(expires_at)
        max_age = max(
            1,
            int((expires_at_dt - datetime.now(timezone.utc)).total_seconds()),
        )
        response.set_cookie(
            key=_web_session_cookie_name(),
            value=token,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
            max_age=max_age,
            expires=expires_at_dt,
            path="/",
        )

    def _clear_auth_cookie(response: Response) -> None:
        response.delete_cookie(
            key=_web_session_cookie_name(),
            httponly=True,
            samesite="lax",
            path="/",
        )

    def _current_auth_session() -> AuthenticatedAccountSession:
        auth_session = auth_session_var.get()
        if auth_session is None:
            raise HTTPException(status_code=401, detail="Authentication required.")
        return auth_session

    def _current_account_id() -> str:
        if not auth_enabled:
            return ROOT_ACCOUNT_ID
        return _current_auth_session().account.id

    def _is_webhook_request(path: str) -> bool:
        return bool(
            re.fullmatch(r"/api/integrations/[^/]+/webhook", path)
            or re.fullmatch(r"/api/integrations/[^/]+/[^/]+/webhook", path)
        )

    def _requires_browser_auth(path: str) -> bool:
        if not auth_enabled:
            return False
        if not path.startswith("/api/"):
            return False
        if path in {
            "/api/features",
            "/api/auth/login",
            "/api/auth/logout",
            "/api/setup/status",
        }:
            return False
        if _is_webhook_request(path):
            return False
        protected_prefixes = (
            "/api/auth/me",
            "/api/auth/change-password",
            "/api/user-profile",
            "/api/user-memory",
            "/api/learned-workflows",
            "/api/sessions",
            "/api/runs",
            "/api/uploads",
            "/api/artifacts",
            "/api/approvals",
            "/api/agents",
            "/api/scheduled-tasks",
            "/api/scheduled-task-executions",
            "/api/integrations",
            "/api/bindings",
            "/api/routing-rules",
            "/api/outbound-deliveries",
            "/api/tools",
            "/api/skills",
            "/api/accounts",  # 账号管理接口
            "/api/setup/config",
            "/api/setup/configs",
        )
        return path.startswith(protected_prefixes)

    def _require_feature(enabled: bool, flag_name: str, capability_label: str) -> None:
        if enabled:
            return
        raise HTTPException(
            status_code=404,
            detail=(
                f"{capability_label} is disabled by feature flag "
                f"'{flag_name}'."
            ),
        )

    def _idle_skill_install_status() -> dict:
        return {
            "state": "idle",
            "message": "No skill installation in progress.",
            "packages": [],
            "error": "",
            "updated_at": None,
        }

    def _set_skill_install_status(
        agent_id: str,
        state: str,
        message: str,
        *,
        packages: list[str] | None = None,
        error: str = "",
    ) -> dict:
        status = {
            "state": state,
            "message": message,
            "packages": list(packages or []),
            "error": error,
            "updated_at": _utc_now_iso(),
        }
        skill_install_statuses[agent_id] = status
        return status

    def _get_skill_install_status(agent_id: str) -> dict:
        return dict(skill_install_statuses.get(agent_id, _idle_skill_install_status()))

    def _default_wechat_setup_status(integration_id: str) -> dict:
        return {
            "integration_id": integration_id,
            "state": "idle",
            "message": "保存默认 Agent 后，点击开始扫码连接。",
            "output": "",
            "qr_text": "",
            "error": "",
            "openclaw_version": "",
            "plugin_spec": "",
            "updated_at": None,
        }

    def _normalize_wechat_setup_status(integration_id: str, payload: dict | None) -> dict:
        normalized = dict(_default_wechat_setup_status(integration_id))
        if not isinstance(payload, dict):
            return normalized
        for key in normalized:
            value = payload.get(key)
            if value is None:
                continue
            normalized[key] = value
        normalized["integration_id"] = integration_id
        normalized["output"] = _tail_text(_strip_ansi_sequences(str(normalized.get("output") or "")))
        normalized["qr_text"] = str(normalized.get("qr_text") or "").strip()
        normalized["error"] = str(normalized.get("error") or "").strip()
        normalized["openclaw_version"] = str(normalized.get("openclaw_version") or "").strip()
        normalized["plugin_spec"] = str(normalized.get("plugin_spec") or "").strip()
        return normalized

    def _get_persisted_wechat_setup_status(record) -> dict:
        if record is None:
            return _default_wechat_setup_status("")
        raw_payload = record.metadata.get("wechat_setup")
        return _normalize_wechat_setup_status(record.id, raw_payload if isinstance(raw_payload, dict) else {})

    def _get_wechat_setup_status(integration_id: str, record=None) -> dict:
        live_status = wechat_setup_statuses.get(integration_id)
        if live_status is not None:
            return _normalize_wechat_setup_status(integration_id, live_status)
        if record is not None:
            return _get_persisted_wechat_setup_status(record)
        return _default_wechat_setup_status(integration_id)

    def _set_wechat_setup_status(
        integration_id: str,
        state: str,
        message: str,
        *,
        output_append: str = "",
        error: str | None = None,
        openclaw_version: str | None = None,
        plugin_spec: str | None = None,
        qr_text: str | None = None,
    ) -> dict:
        current = _get_wechat_setup_status(integration_id)
        if output_append:
            merged_output = _tail_text(current["output"] + _strip_ansi_sequences(output_append))
        else:
            merged_output = current["output"]

        next_payload = {
            **current,
            "integration_id": integration_id,
            "state": state,
            "message": message,
            "output": merged_output,
            "updated_at": _utc_now_iso(),
        }
        if error is not None:
            next_payload["error"] = error
        if openclaw_version is not None:
            next_payload["openclaw_version"] = openclaw_version
        if plugin_spec is not None:
            next_payload["plugin_spec"] = plugin_spec

        extracted_qr = _extract_terminal_qr(merged_output)
        if qr_text is not None:
            next_payload["qr_text"] = qr_text
        elif extracted_qr:
            next_payload["qr_text"] = extracted_qr

        wechat_setup_statuses[integration_id] = _normalize_wechat_setup_status(integration_id, next_payload)
        return dict(wechat_setup_statuses[integration_id])

    async def _persist_wechat_setup_status(integration_id: str, payload: dict) -> None:
        record = await integration_admin_service.get_integration(integration_id)
        if record is None:
            return
        metadata = dict(record.metadata)
        current = metadata.get("wechat_setup")
        normalized = _normalize_wechat_setup_status(
            integration_id,
            current if isinstance(current, dict) else {},
        )
        normalized.update(
            {
                key: value
                for key, value in payload.items()
                if key in normalized and value is not None
            }
        )
        metadata["wechat_setup"] = normalized
        try:
            await integration_admin_service.update_integration(
                integration_id,
                metadata=metadata,
            )
        except KeyError:
            return

    async def _append_wechat_setup_output(integration_id: str, text: str) -> dict:
        current = _get_wechat_setup_status(integration_id)
        output_so_far = current["output"] + _strip_ansi_sequences(text)
        normalized_output = _tail_text(output_so_far)
        lower_output = normalized_output.lower()
        next_state = current["state"] if current["state"] != "queued" else "running"
        next_message = (
            current["message"]
            if current["state"] != "queued"
            else "正在安装微信插件并准备扫码环境。"
        )
        if (
            "扫码" in normalized_output
            or "二维码" in normalized_output
            or "qr" in lower_output
            or "channels login" in lower_output
        ):
            next_state = "waiting_scan"
            next_message = "二维码已生成，请使用微信扫描完成配对。"

        return _set_wechat_setup_status(
            integration_id,
            next_state,
            next_message,
            output_append=text,
            openclaw_version=_parse_wechat_openclaw_version(normalized_output) or current["openclaw_version"],
            plugin_spec=_parse_wechat_plugin_spec(normalized_output) or current["plugin_spec"],
        )

    async def _run_wechat_setup_job(integration_id: str) -> None:
        try:
            record = await integration_admin_service.get_integration(integration_id)
            if record is None:
                _set_wechat_setup_status(
                    integration_id,
                    "failed",
                    "微信集成不存在，无法启动扫码。",
                    error="Integration not found.",
                )
                return
            if record.kind != "wechat":
                _set_wechat_setup_status(
                    integration_id,
                    "failed",
                    "当前集成不是微信类型。",
                    error="Integration kind mismatch.",
                )
                return

            default_agent_id = str(record.config.get("default_agent_id") or "").strip()
            if not default_agent_id:
                message = "请先为微信渠道选择默认 Agent，再开始扫码。"
                _set_wechat_setup_status(
                    integration_id,
                    "failed",
                    message,
                    error=message,
                )
                await _persist_wechat_setup_status(
                    integration_id,
                    {"state": "failed", "message": message, "error": message},
                )
                return

            _set_wechat_setup_status(
                integration_id,
                "running",
                "正在检查 OpenClaw 环境并安装微信插件。",
                error="",
                qr_text="",
            )
            await _persist_wechat_setup_status(
                integration_id,
                {"state": "running", "message": "正在检查 OpenClaw 环境并安装微信插件。", "error": ""},
            )

            return_code, output = await _run_wechat_installer(
                on_output=lambda text: _append_wechat_setup_output(integration_id, text),
                cwd=str(Path.cwd()),
            )
            current = _get_wechat_setup_status(integration_id)
            openclaw_version = _parse_wechat_openclaw_version(output) or current["openclaw_version"]
            plugin_spec = _parse_wechat_plugin_spec(output) or current["plugin_spec"]
            if return_code == 0:
                final_status = _set_wechat_setup_status(
                    integration_id,
                    "succeeded",
                    "微信插件已安装并完成扫码连接。",
                    openclaw_version=openclaw_version,
                    plugin_spec=plugin_spec,
                    error="",
                )
                await _persist_wechat_setup_status(
                    integration_id,
                    {
                        "state": "succeeded",
                        "message": final_status["message"],
                        "error": "",
                        "openclaw_version": openclaw_version,
                        "plugin_spec": plugin_spec,
                        "updated_at": final_status["updated_at"],
                    },
                )
                return

            output_lines = [line.strip() for line in current["output"].splitlines() if line.strip()]
            error_message = (
                current["error"]
                or (output_lines[-1] if output_lines else "")
                or "微信扫码安装失败，请检查 OpenClaw 和 Node.js 环境后重试。"
            )
            final_status = _set_wechat_setup_status(
                integration_id,
                "failed",
                error_message,
                error=error_message,
                openclaw_version=openclaw_version,
                plugin_spec=plugin_spec,
            )
            await _persist_wechat_setup_status(
                integration_id,
                {
                    "state": "failed",
                    "message": final_status["message"],
                    "error": final_status["error"],
                    "openclaw_version": openclaw_version,
                    "plugin_spec": plugin_spec,
                    "updated_at": final_status["updated_at"],
                },
            )
        except Exception as exc:
            final_status = _set_wechat_setup_status(
                integration_id,
                "failed",
                "微信扫码安装失败。",
                error=str(exc),
            )
            await _persist_wechat_setup_status(
                integration_id,
                {
                    "state": "failed",
                    "message": final_status["message"],
                    "error": final_status["error"],
                    "updated_at": final_status["updated_at"],
                },
            )
        finally:
            wechat_setup_tasks.pop(integration_id, None)

    def _queue_wechat_setup(integration_id: str) -> dict:
        existing_task = wechat_setup_tasks.get(integration_id)
        if existing_task and not existing_task.done():
            return _get_wechat_setup_status(integration_id)
        _set_wechat_setup_status(
            integration_id,
            "queued",
            "扫码任务已排队，正在启动官方微信安装器。",
            output_append="",
            error="",
            qr_text="",
        )
        wechat_setup_tasks[integration_id] = asyncio.create_task(_run_wechat_setup_job(integration_id))
        return _get_wechat_setup_status(integration_id)

    def _default_wechat_setup_status(integration_id: str) -> dict:
        return {
            "integration_id": integration_id,
            "state": "idle",
            "message": "Save a default agent, then start QR login.",
            "output": "",
            "qr_text": "",
            "qr_content": "",
            "error": "",
            "ilink_bot_id": "",
            "ilink_user_id": "",
            "base_url": "",
            "updated_at": None,
        }

    def _normalize_wechat_setup_status(integration_id: str, payload: dict | None) -> dict:
        normalized = dict(_default_wechat_setup_status(integration_id))
        if isinstance(payload, dict):
            for key in normalized:
                value = payload.get(key)
                if value is not None:
                    normalized[key] = value
        normalized["integration_id"] = integration_id
        normalized["output"] = _tail_text(str(normalized.get("output") or ""))
        normalized["qr_text"] = str(normalized.get("qr_text") or "").strip()
        normalized["qr_content"] = str(normalized.get("qr_content") or "").strip()
        normalized["error"] = str(normalized.get("error") or "").strip()
        normalized["ilink_bot_id"] = str(normalized.get("ilink_bot_id") or "").strip()
        normalized["ilink_user_id"] = str(normalized.get("ilink_user_id") or "").strip()
        normalized["base_url"] = str(normalized.get("base_url") or "").strip()
        return normalized

    def _get_persisted_wechat_setup_status(record) -> dict:
        if record is None:
            return _default_wechat_setup_status("")
        raw_payload = record.metadata.get("wechat_setup")
        return _normalize_wechat_setup_status(
            record.id,
            raw_payload if isinstance(raw_payload, dict) else {},
        )

    def _get_wechat_setup_status(integration_id: str, record=None) -> dict:
        live_status = wechat_setup_statuses.get(integration_id)
        if live_status is not None:
            return _normalize_wechat_setup_status(integration_id, live_status)
        if record is not None:
            return _get_persisted_wechat_setup_status(record)
        return _default_wechat_setup_status(integration_id)

    def _set_wechat_setup_status(
        integration_id: str,
        state: str,
        message: str,
        *,
        output_append: str = "",
        error: str | None = None,
        qr_text: str | None = None,
        qr_content: str | None = None,
        ilink_bot_id: str | None = None,
        ilink_user_id: str | None = None,
        base_url: str | None = None,
    ) -> dict:
        current = _get_wechat_setup_status(integration_id)
        next_payload = {
            **current,
            "integration_id": integration_id,
            "state": state,
            "message": message,
            "updated_at": _utc_now_iso(),
        }
        if output_append:
            next_payload["output"] = _tail_text(current["output"] + str(output_append))
        if error is not None:
            next_payload["error"] = error
        if qr_text is not None:
            next_payload["qr_text"] = qr_text
        if qr_content is not None:
            next_payload["qr_content"] = qr_content
        if ilink_bot_id is not None:
            next_payload["ilink_bot_id"] = ilink_bot_id
        if ilink_user_id is not None:
            next_payload["ilink_user_id"] = ilink_user_id
        if base_url is not None:
            next_payload["base_url"] = base_url
        wechat_setup_statuses[integration_id] = _normalize_wechat_setup_status(
            integration_id,
            next_payload,
        )
        return dict(wechat_setup_statuses[integration_id])

    async def _persist_wechat_setup_status(integration_id: str, payload: dict) -> None:
        record = await integration_admin_service.get_integration(integration_id)
        if record is None:
            return
        metadata = dict(record.metadata)
        current = metadata.get("wechat_setup")
        normalized = _normalize_wechat_setup_status(
            integration_id,
            current if isinstance(current, dict) else {},
        )
        normalized.update(
            {
                key: value
                for key, value in payload.items()
                if key in normalized and value is not None
            }
        )
        metadata["wechat_setup"] = normalized
        try:
            await integration_admin_service.update_integration(
                integration_id,
                metadata=metadata,
            )
        except KeyError:
            return

    async def _store_wechat_credentials(
        integration_id: str,
        credentials: WeChatILinkCredentials,
    ) -> None:
        current_credentials = await integration_admin_service.list_credentials(integration_id)
        next_wechat_values = {
            "bot_token": credentials.bot_token,
            "ilink_bot_id": credentials.ilink_bot_id,
            "base_url": credentials.base_url,
            "ilink_user_id": credentials.ilink_user_id,
        }
        preserved: list[dict] = []
        handled_keys: set[str] = set()

        for current in current_credentials:
            credential_key = str(current.credential_key or "").strip()
            if not credential_key:
                continue
            if credential_key in next_wechat_values:
                preserved.append(
                    {
                        "credential_key": credential_key,
                        "storage_kind": "local_encrypted",
                        "secret_value": next_wechat_values[credential_key],
                        "metadata": dict(current.metadata),
                    }
                )
                handled_keys.add(credential_key)
                continue

            item = {
                "credential_key": credential_key,
                "storage_kind": current.storage_kind,
                "metadata": dict(current.metadata),
            }
            if current.storage_kind == "local_encrypted":
                item["secret_value"] = current.secret_ciphertext
            else:
                item["secret_ref"] = current.secret_ref
            preserved.append(item)

        for credential_key, secret_value in next_wechat_values.items():
            if credential_key in handled_keys:
                continue
            preserved.append(
                {
                    "credential_key": credential_key,
                    "storage_kind": "local_encrypted",
                    "secret_value": secret_value,
                    "metadata": {},
                }
            )

        await integration_admin_service.update_integration(
            integration_id,
            credentials=preserved,
        )

    def _update_wechat_status_from_poll(
        integration_id: str,
        status: str,
        *,
        qr_content: str,
    ) -> dict:
        normalized_status = str(status or "").strip().lower()
        if normalized_status == "scaned":
            return _set_wechat_setup_status(
                integration_id,
                "scanned",
                "QR scanned. Confirm the login on your phone.",
                output_append="QR scanned. Waiting for confirmation.\n",
                error="",
                qr_text=qr_content,
                qr_content=qr_content,
            )
        if normalized_status == "confirmed":
            return _set_wechat_setup_status(
                integration_id,
                "running",
                "Login confirmed. Finishing iLink setup.",
                output_append="Login confirmed. Finalizing iLink credentials.\n",
                error="",
                qr_text=qr_content,
                qr_content=qr_content,
            )
        return _set_wechat_setup_status(
            integration_id,
            "waiting_scan",
            "Scan the QR code with WeChat to connect this channel.",
            qr_text=qr_content,
            qr_content=qr_content,
        )

    async def _run_wechat_setup_job(integration_id: str) -> None:
        try:
            record = await integration_admin_service.get_integration(integration_id)
            if record is None:
                final_status = _set_wechat_setup_status(
                    integration_id,
                    "failed",
                    "The WeChat integration no longer exists.",
                    error="Integration not found.",
                )
                await _persist_wechat_setup_status(integration_id, final_status)
                return
            if record.kind != "wechat":
                final_status = _set_wechat_setup_status(
                    integration_id,
                    "failed",
                    "The selected integration is not a WeChat channel.",
                    error="Integration kind mismatch.",
                )
                await _persist_wechat_setup_status(integration_id, final_status)
                return

            default_agent_id = str(record.config.get("default_agent_id") or "").strip()
            if not default_agent_id:
                final_status = _set_wechat_setup_status(
                    integration_id,
                    "failed",
                    "Choose a default agent before starting WeChat login.",
                    error="default_agent_id is required.",
                )
                await _persist_wechat_setup_status(integration_id, final_status)
                return

            running_status = _set_wechat_setup_status(
                integration_id,
                "running",
                "Requesting a new WeChat iLink QR code.",
                output_append="Starting native iLink QR login.\n",
                error="",
                qr_text="",
                qr_content="",
                ilink_bot_id="",
                ilink_user_id="",
                base_url="",
            )
            await _persist_wechat_setup_status(integration_id, running_status)

            qr_code = await fetch_login_qr_code()
            waiting_status = _set_wechat_setup_status(
                integration_id,
                "waiting_scan",
                "Scan the QR code with WeChat to connect this channel.",
                output_append="QR code generated. Waiting for scan.\n",
                error="",
                qr_text=qr_code.qr_content,
                qr_content=qr_code.qr_content,
            )
            await _persist_wechat_setup_status(integration_id, waiting_status)

            async def _on_login_status(status: str) -> None:
                updated_status = _update_wechat_status_from_poll(
                    integration_id,
                    status,
                    qr_content=qr_code.qr_content,
                )
                await _persist_wechat_setup_status(integration_id, updated_status)

            credentials = await poll_login_status(
                qr_code.qrcode,
                on_status=_on_login_status,
            )
            await _store_wechat_credentials(integration_id, credentials)

            final_status = _set_wechat_setup_status(
                integration_id,
                "succeeded",
                "WeChat channel connected through native iLink.",
                output_append="Native iLink credentials stored successfully.\n",
                error="",
                qr_text=qr_code.qr_content,
                qr_content=qr_code.qr_content,
                ilink_bot_id=credentials.ilink_bot_id,
                ilink_user_id=credentials.ilink_user_id,
                base_url=credentials.base_url,
            )
            await _persist_wechat_setup_status(integration_id, final_status)
            await integration_admin_service.set_integration_status(
                integration_id,
                status="active",
            )
            await _sync_wechat_long_poll(integration_id)
        except Exception as exc:
            final_status = _set_wechat_setup_status(
                integration_id,
                "failed",
                "Native WeChat iLink login failed.",
                output_append=f"{exc}\n",
                error=str(exc),
            )
            await _persist_wechat_setup_status(integration_id, final_status)
            await _sync_wechat_long_poll(integration_id)
        finally:
            wechat_setup_tasks.pop(integration_id, None)

    def _queue_wechat_setup(integration_id: str) -> dict:
        existing_task = wechat_setup_tasks.get(integration_id)
        if existing_task and not existing_task.done():
            return _get_wechat_setup_status(integration_id)
        queued_status = _set_wechat_setup_status(
            integration_id,
            "queued",
            "WeChat iLink login has been queued.",
            output_append="Queued native iLink QR login.\n",
            error="",
            qr_text="",
            qr_content="",
            ilink_bot_id="",
            ilink_user_id="",
            base_url="",
        )
        asyncio.create_task(_persist_wechat_setup_status(integration_id, queued_status))
        wechat_setup_tasks[integration_id] = asyncio.create_task(
            _run_wechat_setup_job(integration_id)
        )
        return queued_status

    def _serialize_agent_with_status(agent: dict | None) -> dict | None:
        if not agent:
            return None
        payload = dict(agent)
        payload.pop("llm_routing_policy", None)
        payload["skill_install_status"] = _get_skill_install_status(agent["id"])
        return payload

    def _serialize_binding(binding) -> BindingResponse:
        return BindingResponse(
            **binding.model_dump(),
            session_key=integration_router.build_session_key_for_binding(binding),
        )

    def _sanitize_public_json(payload: object) -> object:
        return prepare_bounded_json_payload(
            payload,
            max_bytes=512 * 1024,
        ).data

    def _serialize_credential(record) -> IntegrationCredentialResponse:
        return IntegrationCredentialResponse(
            id=record.id,
            credential_key=record.credential_key,
            storage_kind=record.storage_kind,
            secret_ref=record.secret_ref,
            masked_value=record.masked_value,
            metadata=record.metadata,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    def _serialize_routing_rule(record) -> RoutingRuleResponse:
        return RoutingRuleResponse(**record.model_dump())

    def _serialize_inbound_event(record) -> InboundEventResponse:
        return InboundEventResponse(
            id=record.id,
            integration_id=record.integration_id,
            provider_event_id=record.provider_event_id,
            provider_message_id=record.provider_message_id,
            provider_chat_id=record.provider_chat_id,
            provider_thread_id=record.provider_thread_id,
            provider_user_id=record.provider_user_id,
            event_type=record.event_type,
            received_at=record.received_at,
            signature_valid=record.signature_valid,
            normalized_status=record.normalized_status,
            normalized_error=record.normalized_error,
            metadata=_sanitize_public_json(record.metadata),
        )

    def _serialize_delivery_attempt(record) -> DeliveryAttemptResponse:
        return DeliveryAttemptResponse(
            id=record.id,
            delivery_id=record.delivery_id,
            attempt_number=record.attempt_number,
            status=record.status,
            error_summary=record.error_summary,
            started_at=record.started_at,
            finished_at=record.finished_at,
        )

    async def _serialize_delivery(record) -> OutboundDeliveryResponse:
        attempts = await integration_admin_service.list_delivery_attempts(
            record.id,
            account_id=record.account_id,
        )
        payload = record.model_dump()
        payload["payload"] = _sanitize_public_json(record.payload)
        payload["metadata"] = _sanitize_public_json(record.metadata)
        payload["attempts"] = [_serialize_delivery_attempt(attempt) for attempt in attempts]
        return OutboundDeliveryResponse(
            **payload,
        )

    async def _serialize_integration(record) -> IntegrationResponse:
        credentials = await integration_admin_service.list_credentials(
            record.id,
            account_id=record.account_id,
        )
        bindings = integration_router.list_bindings(
            account_id=record.account_id,
            integration_id=record.id,
        )
        routing_rules = await integration_admin_service.list_routing_rules(
            record.id,
            account_id=record.account_id,
            enabled=None,
        )
        events = await integration_admin_service.list_events(
            account_id=record.account_id,
            integration_id=record.id,
            limit=1,
        )
        deliveries = await integration_admin_service.list_deliveries(
            account_id=record.account_id,
            integration_id=record.id,
            limit=1,
        )
        setup_status = None
        if record.kind == "wechat":
            setup_status = WeChatSetupStatusResponse.model_validate(
                _get_wechat_setup_status(record.id, record=record)
            )
        return IntegrationResponse(
            id=record.id,
            name=record.name,
            kind=record.kind,
            status=record.status,
            display_name=record.display_name,
            tenant_id=record.tenant_id,
            webhook_path=record.webhook_path,
            config=_sanitize_public_json(record.config),
            metadata=_sanitize_public_json(record.metadata),
            credentials=[_serialize_credential(item) for item in credentials],
            created_at=record.created_at,
            updated_at=record.updated_at,
            last_verified_at=record.last_verified_at,
            last_error=record.last_error,
            binding_count=len(bindings),
            rule_count=len(routing_rules),
            last_event_at=events[0].received_at if events else None,
            last_delivery_at=deliveries[0].created_at if deliveries else None,
            deleted=integration_admin_service.is_deleted(record),
            setup_status=setup_status,
        )

    def _build_file_urls(target_kind: Literal["upload", "artifact"], target_id: str) -> dict[str, str]:
        base_path = (
            f"/api/uploads/{target_id}"
            if target_kind == "upload"
            else f"/api/artifacts/{target_id}"
        )
        return {
            "download_url": base_path,
            "open_url": f"{base_path}?disposition=inline",
            "inline_url": f"{base_path}?disposition=inline",
        }

    def _resolve_served_mime_type(file_path: Path, declared_mime_type: str) -> str:
        normalized_declared = declared_mime_type.strip()
        guessed_mime_type, _ = mimetypes.guess_type(str(file_path))
        if normalized_declared and normalized_declared != "application/octet-stream":
            return normalized_declared
        if guessed_mime_type:
            return guessed_mime_type
        if normalized_declared:
            return normalized_declared
        return "application/octet-stream"

    def _build_workspace_file_response(
        *,
        file_path: Path,
        display_name: str,
        mime_type: str,
        disposition: Literal["attachment", "inline"],
    ) -> FileResponse:
        resolved_name = display_name.strip() or file_path.name
        return FileResponse(
            path=file_path,
            media_type=_resolve_served_mime_type(file_path, mime_type),
            filename=resolved_name,
            content_disposition_type=disposition,
        )

    def _build_file_preview_response(
        *,
        target_kind: Literal["upload", "artifact"],
        target_id: str,
        file_path: Path,
        display_name: str,
        mime_type: str,
        size_bytes: int | None,
        format_hint: str = "",
    ) -> FilePreviewResponse:
        resolved_name = display_name.strip() or file_path.name
        resolved_mime_type = _resolve_served_mime_type(file_path, mime_type)
        preview_kind = guess_preview_kind(
            artifact_format=format_hint,
            mime_type=resolved_mime_type,
            filename=resolved_name,
        )
        urls = _build_file_urls(target_kind, target_id)
        payload: dict[str, object] = {
            "target_kind": target_kind,
            "target_id": target_id,
            "display_name": resolved_name,
            "mime_type": resolved_mime_type,
            "preview_kind": preview_kind,
            "size_bytes": size_bytes,
            "preview_supported": False,
            "truncated": False,
            "text_content": None,
            "inline_url": "",
            "open_url": urls["open_url"],
            "download_url": urls["download_url"],
            "note": "",
        }

        if preview_kind in TEXT_PREVIEW_KINDS:
            text_content, truncated = read_text_preview(file_path)
            payload["preview_supported"] = True
            payload["text_content"] = text_content
            payload["truncated"] = truncated
            if truncated:
                payload["note"] = (
                    f"预览内容已截断，仅显示前 {TEXT_PREVIEW_BYTE_LIMIT // 1024} KB。"
                )
            return FilePreviewResponse.model_validate(payload)

        if preview_kind in INLINE_FILE_PREVIEW_KINDS:
            payload["preview_supported"] = True
            payload["inline_url"] = urls["inline_url"]
            return FilePreviewResponse.model_validate(payload)

        if preview_kind == "office":
            payload["note"] = "当前 Office 文件暂不支持浏览器内预览，请直接打开原文件或下载。"
            return FilePreviewResponse.model_validate(payload)

        if preview_kind == "html":
            payload["note"] = "HTML 文件建议通过“打开原文件”在新标签页中查看。"
            return FilePreviewResponse.model_validate(payload)

        payload["note"] = "当前文件暂无可用预览，请直接打开原文件或下载。"
        return FilePreviewResponse.model_validate(payload)

    async def _run_skill_install_job(agent_id: str, package_names: list[str]) -> None:
        normalized = [item.strip() for item in package_names if item and item.strip()]
        _set_skill_install_status(
            agent_id,
            "running",
            "Installing selected skills in the background.",
            packages=normalized,
        )
        try:
            installed_skills = await _install_skill_packages(
                session_manager,
                agent_id,
                normalized,
            )
            if session_manager._agent_store:
                session_manager._agent_store.update_agent(
                    agent_id,
                    skills=installed_skills,
                    tools=_default_agent_tools(session_manager),
                )
            _set_skill_install_status(
                agent_id,
                "succeeded",
                "Skill installation completed.",
                packages=normalized,
            )
        except Exception as exc:
            _set_skill_install_status(
                agent_id,
                "failed",
                "Skill installation failed.",
                packages=normalized,
                error=str(exc),
            )
        finally:
            skill_install_tasks.pop(agent_id, None)

    def _normalize_skill_package_names(package_names: list[str] | None) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in package_names or []:
            package_name = item.strip()
            if not package_name or package_name in seen:
                continue
            normalized.append(package_name)
            seen.add(package_name)
        return normalized

    def _ensure_skill_install_available(agent_id: str, package_names: list[str] | None) -> list[str]:
        normalized = _normalize_skill_package_names(package_names)
        if not normalized:
            return []

        existing_task = skill_install_tasks.get(agent_id)
        if existing_task and not existing_task.done():
            raise RuntimeError("A skill installation job is already in progress for this agent.")
        return normalized

    def _ensure_skill_install_idle(agent_id: str) -> None:
        existing_task = skill_install_tasks.get(agent_id)
        if existing_task and not existing_task.done():
            raise RuntimeError("A skill installation job is already in progress for this agent.")

    def _queue_skill_install(agent_id: str, package_names: list[str] | None) -> dict:
        normalized = _ensure_skill_install_available(agent_id, package_names)
        if not normalized:
            return _get_skill_install_status(agent_id)

        _set_skill_install_status(
            agent_id,
            "queued",
            "Skill installation has been queued.",
            packages=normalized,
        )
        task = asyncio.create_task(_run_skill_install_job(agent_id, normalized))
        skill_install_tasks[agent_id] = task
        return _get_skill_install_status(agent_id)

    async def _stop_runtime_services(app: FastAPI) -> None:
        with contextlib.suppress(Exception):
            await app.state.scheduled_task_service.shutdown()
        with contextlib.suppress(Exception):
            await app.state.wechat_long_poll_manager.shutdown()
        with contextlib.suppress(Exception):
            await app.state.feishu_long_connection_manager.shutdown()

    async def _start_runtime_services_if_ready(app: FastAPI) -> bool:
        if not session_manager.is_runtime_ready():
            return False
        await app.state.feishu_long_connection_manager.start()
        await app.state.wechat_long_poll_manager.start()
        await app.state.scheduled_task_service.start()
        return True

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize session manager on startup, cleanup on shutdown."""
        await session_manager.initialize()
        await _start_runtime_services_if_ready(app)
        yield
        await _stop_runtime_services(app)
        for task in list(skill_install_tasks.values()):
            if task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        for task in list(wechat_setup_tasks.values()):
            if task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await session_manager.cleanup()

    app = FastAPI(title="Clavi Agent", version="0.1.0", lifespan=lifespan)
    app.state.integration_gateway = integration_gateway
    app.state.integration_router = integration_router
    app.state.integration_admin_service = integration_admin_service
    app.state.scheduled_task_service = scheduled_task_service
    app.state.feishu_long_connection_manager = feishu_long_connection_manager
    app.state.wechat_long_poll_manager = wechat_long_poll_manager

    @app.exception_handler(PermissionError)
    async def permission_error_handler(_request: Request, exc: PermissionError):
        return JSONResponse(
            status_code=403,
            content={"detail": str(exc) or "Forbidden."},
        )

    @app.middleware("http")
    async def browser_auth_middleware(request: Request, call_next):
        if not _requires_browser_auth(request.url.path):
            return await call_next(request)

        store = session_manager._account_store
        if store is None:
            return JSONResponse(
                status_code=500,
                content={"detail": "AccountStore is not initialized."},
            )

        raw_token = str(request.cookies.get(_web_session_cookie_name()) or "").strip()
        if not raw_token:
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
            )

        authenticated = store.get_authenticated_session(raw_token)
        if authenticated is None:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
            )
            _clear_auth_cookie(response)
            return response

        renewed_expires_at = _next_web_session_expiry()
        store.touch_web_session(
            authenticated.web_session.id,
            expires_at=renewed_expires_at,
        )
        refreshed = store.get_authenticated_session(raw_token)
        if refreshed is None:
            response = JSONResponse(
                status_code=401,
                content={"detail": "Authentication required."},
            )
            _clear_auth_cookie(response)
            return response

        token = auth_session_var.set(refreshed)
        try:
            response = await call_next(request)
        finally:
            auth_session_var.reset(token)

        _set_auth_cookie(
            response,
            request=request,
            token=raw_token,
            expires_at=renewed_expires_at,
        )
        return response

    @app.middleware("http")
    async def static_cache_control_middleware(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path or ""
        if path == "/" or path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

    async def _sync_feishu_long_connection(integration_id: str) -> None:
        manager = getattr(app.state, "feishu_long_connection_manager", None)
        if manager is None:
            return
        await manager.sync_integration(integration_id)

    async def _sync_wechat_long_poll(integration_id: str) -> None:
        manager = getattr(app.state, "wechat_long_poll_manager", None)
        if manager is None:
            return
        await manager.sync_integration(integration_id)

    @app.get("/api/features")
    async def get_feature_flags():
        """Expose effective rollout switches for frontend/runtime gating."""
        return _feature_flags()

    @app.get("/api/memory-provider/health", response_model=MemoryProviderHealthResponse)
    async def get_memory_provider_health():
        """Expose long-term memory provider health and fallback state."""
        await session_manager.initialize()
        return MemoryProviderHealthResponse(**session_manager.get_memory_provider_status())

    @app.get("/api/setup/status", response_model=SetupStatusResponse)
    async def get_setup_status(request: Request):
        """Expose whether the web runtime still needs API bootstrap configuration."""
        await session_manager.initialize()
        auth_session = _optional_auth_session_from_request(request)
        account_id = auth_session.account.id if auth_session is not None else None
        return _setup_status_payload(account_id=account_id)

    @app.post("/api/setup/config", response_model=SetupStatusResponse)
    async def save_setup_config(req: SetupConfigRequest):
        """Persist one account-owned API configuration and optionally activate it."""
        await session_manager.initialize()
        try:
            session_manager.save_account_api_config(
                _current_account_id(),
                name=req.name,
                api_key=req.api_key,
                provider=req.provider,
                api_base=req.api_base,
                model=req.model,
                reasoning_enabled=req.reasoning_enabled,
                llm_routing_policy=req.llm_routing_policy,
                activate=req.activate,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _setup_status_payload(
            account_id=_current_account_id(),
            message="API configuration saved for the current account.",
        )

    @app.get("/api/setup/configs", response_model=list[ApiConfigSummaryResponse])
    async def list_setup_configs():
        """List all account-owned API credential sets."""
        await session_manager.initialize()
        return [
            _api_config_summary_payload(record)
            for record in session_manager.list_account_api_configs(_current_account_id())
        ]

    @app.post("/api/setup/configs/{config_id}/activate", response_model=SetupStatusResponse)
    async def activate_setup_config(config_id: str):
        """Activate one stored API credential set for the current account."""
        await session_manager.initialize()
        try:
            session_manager.activate_account_api_config(_current_account_id(), config_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="API configuration not found.")
        return _setup_status_payload(
            account_id=_current_account_id(),
            message="Active API configuration switched.",
        )

    @app.delete("/api/setup/configs/{config_id}")
    async def delete_setup_config(config_id: str):
        """Delete one stored API credential set owned by the current account."""
        await session_manager.initialize()
        deleted = session_manager.delete_account_api_config(_current_account_id(), config_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="API configuration not found.")
        return {"status": "deleted", "id": config_id}

    @app.patch("/api/setup/configs/{config_id}/routing", response_model=ApiConfigSummaryResponse)
    async def update_setup_config_routing(config_id: str, req: ApiRoutingUpdateRequest):
        """Update one account-owned API config's main/worker routing selection."""
        await session_manager.initialize()
        try:
            record = session_manager.update_account_api_config_routing(
                _current_account_id(),
                config_id,
                llm_routing_policy=req.model_dump(exclude_none=True),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="API configuration not found.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _api_config_summary_payload(record)

    @app.post("/api/auth/login", response_model=AuthSessionResponse)
    async def login(req: LoginRequest, request: Request):
        store = session_manager._account_store
        if store is None:
            raise HTTPException(status_code=500, detail="AccountStore is not initialized.")

        authenticated_account = store.authenticate(req.username, req.password)
        if authenticated_account is None:
            raise HTTPException(status_code=401, detail="Invalid username or password.")

        expires_at = _next_web_session_expiry()
        _, raw_token = store.create_web_session(
            authenticated_account.id,
            expires_at=expires_at,
            user_agent=request.headers.get("user-agent", ""),
            ip=request.client.host if request.client is not None else "",
        )
        authenticated_session = store.get_authenticated_session(raw_token)
        if authenticated_session is None:
            raise HTTPException(status_code=500, detail="Failed to create authenticated session.")

        response = JSONResponse(
            _build_auth_session_response(
                authenticated_session,
                expires_at=expires_at,
            ).model_dump(mode="json")
        )
        _set_auth_cookie(
            response,
            request=request,
            token=raw_token,
            expires_at=expires_at,
        )
        return response

    @app.post("/api/auth/logout")
    async def logout(request: Request):
        store = session_manager._account_store
        if store is not None:
            raw_token = str(request.cookies.get(_web_session_cookie_name()) or "").strip()
            if raw_token:
                with contextlib.suppress(ValueError):
                    store.delete_web_session_by_token(raw_token)
        response = JSONResponse({"status": "logged_out"})
        _clear_auth_cookie(response)
        return response

    @app.get("/api/auth/me", response_model=AuthSessionResponse)
    async def get_current_account():
        return _build_auth_session_response(_current_auth_session())

    @app.post("/api/auth/change-password", response_model=AuthSessionResponse)
    async def change_password(req: ChangePasswordRequest):
        store = session_manager._account_store
        if store is None:
            raise HTTPException(status_code=500, detail="AccountStore is not initialized.")

        auth_session = _current_auth_session()
        verified_account = store.authenticate(
            auth_session.account.username,
            req.current_password,
        )
        if verified_account is None or verified_account.id != auth_session.account.id:
            raise HTTPException(status_code=400, detail="Current password is incorrect.")

        store.set_password(auth_session.account.id, req.new_password)
        latest = store.get_account_record(auth_session.account.id)
        if latest is None:
            raise HTTPException(status_code=500, detail="Account not found after password update.")
        return AuthSessionResponse(
            account=AuthAccountResponse(
                id=latest.id,
                username=latest.username,
                display_name=latest.display_name,
                status=latest.status,
                is_root=latest.is_root,
                created_at=latest.created_at,
                updated_at=latest.updated_at,
            ),
            expires_at=auth_session.web_session.expires_at,
        )

    # --- 账号管理接口（仅 root 账号可调用）---

    def _require_root_account() -> None:
        """断言当前会话属于 root 账号，否则抛出 403。"""
        auth_session = _current_auth_session()
        if not auth_session.account.is_root:
            raise HTTPException(status_code=403, detail="Only root account can manage accounts.")

    @app.get("/api/accounts", response_model=list[AccountSummaryResponse])
    async def list_accounts_api():
        """列出所有本地账号（仅 root 账号可访问）。"""
        _require_root_account()
        store = session_manager._account_store
        if store is None:
            raise HTTPException(status_code=500, detail="AccountStore is not initialized.")
        accounts = store.list_accounts()
        return [AccountSummaryResponse(**a) for a in accounts]

    @app.post("/api/accounts", response_model=AccountSummaryResponse, status_code=201)
    async def create_account_api(req: AccountCreateRequest):
        """创建新账号（仅 root 账号可操作）。"""
        _require_root_account()
        store = session_manager._account_store
        if store is None:
            raise HTTPException(status_code=500, detail="AccountStore is not initialized.")
        try:
            account = store.create_account(
                username=req.username,
                password=req.password,
                display_name=req.display_name or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return AccountSummaryResponse(**account)

    @app.get("/api/accounts/{account_id}", response_model=AccountSummaryResponse)
    async def get_account_api(account_id: str):
        """获取指定账号信息（仅 root 可访问所有账号，普通账号只能访问自身）。"""
        auth_session = _current_auth_session()
        # 普通账号只能查自己
        if not auth_session.account.is_root and auth_session.account.id != account_id:
            raise HTTPException(status_code=403, detail="Permission denied.")
        store = session_manager._account_store
        if store is None:
            raise HTTPException(status_code=500, detail="AccountStore is not initialized.")
        account = store.get_account(account_id)
        if account is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        return AccountSummaryResponse(**account)

    @app.patch("/api/accounts/{account_id}", response_model=AccountSummaryResponse)
    async def update_account_api(account_id: str, req: AccountUpdateRequest):
        """更新账号信息（display_name / status），仅 root 账号可操作。"""
        _require_root_account()
        store = session_manager._account_store
        if store is None:
            raise HTTPException(status_code=500, detail="AccountStore is not initialized.")
        try:
            account = store.update_account(
                account_id,
                display_name=req.display_name,
                status=req.status,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if account is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        return AccountSummaryResponse(**account)

    @app.post("/api/accounts/{account_id}/reset-password", response_model=AccountSummaryResponse)
    async def reset_account_password_api(account_id: str, req: AccountResetPasswordRequest):
        """Root 重置任意账号（含自身）的密码。"""
        _require_root_account()
        store = session_manager._account_store
        if store is None:
            raise HTTPException(status_code=500, detail="AccountStore is not initialized.")
        try:
            store.set_password(account_id, req.new_password)
        except KeyError:
            raise HTTPException(status_code=404, detail="Account not found.")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        account = store.get_account(account_id)
        if account is None:
            raise HTTPException(status_code=404, detail="Account not found.")
        return AccountSummaryResponse(**account)

    @app.get("/api/user-profile", response_model=UserProfileResponse)
    async def get_user_profile():
        """返回当前账号的结构化用户画像。"""
        profile = session_manager.get_user_profile_info(account_id=_current_account_id())
        if profile is None:
            raise HTTPException(status_code=404, detail="User profile not found")
        return UserProfileResponse(**profile)

    @app.patch("/api/user-profile", response_model=UserProfileResponse)
    async def update_user_profile(req: UserProfileUpdateRequest):
        """更新当前账号的结构化用户画像字段。"""
        if not req.profile_updates and not req.remove_fields and req.summary is None:
            raise HTTPException(
                status_code=400,
                detail="At least one profile update, removal, or summary change is required.",
            )
        try:
            payload = {
                "account_id": _current_account_id(),
                "profile_updates": req.profile_updates,
                "remove_fields": req.remove_fields,
                "profile_source": req.profile_source,
                "profile_confidence": req.profile_confidence,
            }
            if "summary" in req.model_fields_set:
                payload["summary"] = req.summary
            profile = session_manager.update_user_profile_info(**payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if profile is None:
            raise HTTPException(status_code=404, detail="User profile not found")
        return UserProfileResponse(**profile)

    @app.get("/api/user-memory", response_model=list[UserMemoryEntryResponse])
    async def list_user_memory(
        query: str | None = None,
        memory_type: str | None = None,
        include_superseded: bool = False,
        limit: int = 20,
    ):
        """列出或搜索当前账号的用户长期记忆。"""
        try:
            entries = session_manager.list_user_memory_entries_info(
                account_id=_current_account_id(),
                query=str(query or "").strip() or None,
                memory_types=[memory_type] if str(memory_type or "").strip() else None,
                include_superseded=include_superseded,
                limit=max(1, min(int(limit), 100)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return [UserMemoryEntryResponse(**item) for item in entries]

    @app.get("/api/user-memory/audit", response_model=list[MemoryAuditEventResponse])
    async def list_user_memory_audit(
        target_scope: Literal["user_profile", "user_memory"] | None = None,
        target_id: str | None = None,
        limit: int = 50,
    ):
        """列出当前账号最近的记忆写入审计轨迹。"""
        try:
            events = session_manager.list_user_memory_audit_events(
                account_id=_current_account_id(),
                target_scope=target_scope,
                target_id=str(target_id or "").strip() or None,
                limit=max(1, min(int(limit), 100)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return [MemoryAuditEventResponse(**item) for item in events]

    @app.get("/api/user-memory/{entry_id}", response_model=UserMemoryEntryResponse)
    async def get_user_memory_entry(entry_id: str):
        """返回当前账号的一条用户长期记忆。"""
        try:
            entry = session_manager.get_user_memory_entry_info(
                entry_id,
                account_id=_current_account_id(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if entry is None:
            raise HTTPException(status_code=404, detail="User memory entry not found")
        return UserMemoryEntryResponse(**entry)

    @app.patch("/api/user-memory/{entry_id}", response_model=UserMemoryEntryResponse)
    async def update_user_memory_entry(entry_id: str, req: UserMemoryUpdateRequest):
        """更新当前账号的一条用户长期记忆。"""
        if (
            req.memory_type is None
            and req.content is None
            and req.summary is None
            and req.confidence is None
        ):
            raise HTTPException(status_code=400, detail="At least one memory field update is required.")
        try:
            payload = {
                "account_id": _current_account_id(),
            }
            for field_name in ("memory_type", "content", "summary", "confidence"):
                if field_name in req.model_fields_set:
                    payload[field_name] = getattr(req, field_name)
            entry = session_manager.update_user_memory_entry_info(entry_id, **payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if entry is None:
            raise HTTPException(status_code=404, detail="User memory entry not found")
        return UserMemoryEntryResponse(**entry)

    @app.delete("/api/user-memory/{entry_id}", response_model=MutationStatusResponse)
    async def delete_user_memory_entry(entry_id: str, reason: str | None = None):
        """软删除当前账号的一条用户长期记忆。"""
        try:
            deleted = session_manager.delete_user_memory_entry_info(
                entry_id,
                account_id=_current_account_id(),
                reason=str(reason or "").strip() or None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not deleted:
            raise HTTPException(status_code=404, detail="User memory entry not found")
        return MutationStatusResponse(status="deleted", target_id=entry_id)

    @app.get(
        "/api/learned-workflows",
        response_model=list[LearnedWorkflowCandidateResponse],
    )
    async def list_learned_workflows(
        status: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ):
        """列出当前账号的 learned workflow 候选。"""
        return [
            LearnedWorkflowCandidateResponse(**item)
            for item in session_manager.list_learned_workflow_candidates(
                account_id=_current_account_id(),
                status=status,
                agent_id=agent_id,
                run_id=run_id,
                limit=limit,
            )
        ]

    @app.post(
        "/api/learned-workflows/{candidate_id}/approve",
        response_model=LearnedWorkflowCandidateResponse,
    )
    async def approve_learned_workflow(
        candidate_id: str,
        req: LearnedWorkflowReviewRequest | None = None,
    ):
        """批准一个 learned workflow 候选。"""
        try:
            payload = session_manager.approve_learned_workflow_candidate(
                candidate_id,
                account_id=_current_account_id(),
                review_notes=req.review_notes if req is not None else "",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return LearnedWorkflowCandidateResponse(**payload)

    @app.post(
        "/api/learned-workflows/{candidate_id}/reject",
        response_model=LearnedWorkflowCandidateResponse,
    )
    async def reject_learned_workflow(
        candidate_id: str,
        req: LearnedWorkflowReviewRequest | None = None,
    ):
        """拒绝一个 learned workflow 候选。"""
        try:
            payload = session_manager.reject_learned_workflow_candidate(
                candidate_id,
                account_id=_current_account_id(),
                review_notes=req.review_notes if req is not None else "",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return LearnedWorkflowCandidateResponse(**payload)

    @app.post(
        "/api/learned-workflows/{candidate_id}/install",
        response_model=LearnedWorkflowCandidateResponse,
    )
    async def install_learned_workflow(
        candidate_id: str,
        req: LearnedWorkflowInstallRequest,
    ):
        """把已审核候选安装到目标 Agent 技能目录。"""
        try:
            payload = session_manager.install_learned_workflow_candidate(
                candidate_id,
                agent_id=req.agent_id,
                account_id=_current_account_id(),
                skill_name=req.skill_name or None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return LearnedWorkflowCandidateResponse(**payload)

    @app.get(
        "/api/skill-improvements",
        response_model=list[SkillImprovementProposalResponse],
    )
    async def list_skill_improvements(
        status: str | None = None,
        agent_id: str | None = None,
        skill_name: str | None = None,
        run_id: str | None = None,
        limit: int | None = None,
    ):
        """列出当前账号的技能改进提案。"""
        return [
            SkillImprovementProposalResponse(**item)
            for item in session_manager.list_skill_improvement_proposals(
                account_id=_current_account_id(),
                status=status,
                agent_id=agent_id,
                skill_name=skill_name,
                run_id=run_id,
                limit=limit,
            )
        ]

    @app.post(
        "/api/skill-improvements/{proposal_id}/approve",
        response_model=SkillImprovementProposalResponse,
    )
    async def approve_skill_improvement(
        proposal_id: str,
        req: SkillImprovementReviewRequest | None = None,
    ):
        """批准一个技能改进提案。"""
        try:
            payload = session_manager.approve_skill_improvement_proposal(
                proposal_id,
                account_id=_current_account_id(),
                review_notes=req.review_notes if req is not None else "",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return SkillImprovementProposalResponse(**payload)

    @app.post(
        "/api/skill-improvements/{proposal_id}/reject",
        response_model=SkillImprovementProposalResponse,
    )
    async def reject_skill_improvement(
        proposal_id: str,
        req: SkillImprovementReviewRequest | None = None,
    ):
        """拒绝一个技能改进提案。"""
        try:
            payload = session_manager.reject_skill_improvement_proposal(
                proposal_id,
                account_id=_current_account_id(),
                review_notes=req.review_notes if req is not None else "",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return SkillImprovementProposalResponse(**payload)

    @app.post(
        "/api/skill-improvements/{proposal_id}/apply",
        response_model=SkillImprovementProposalResponse,
    )
    async def apply_skill_improvement(
        proposal_id: str,
    ):
        """把已审核技能改进提案应用到目标 SKILL.md。"""
        try:
            payload = session_manager.apply_skill_improvement_proposal(
                proposal_id,
                account_id=_current_account_id(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return SkillImprovementProposalResponse(**payload)

    @app.post("/api/sessions", response_model=CreateSessionResponse)
    async def create_session(req: CreateSessionRequest | None = None):
        """Create a new chat session."""
        workspace = req.workspace_dir if req else None
        agent_id = req.agent_id if req else None
        try:
            session_id = await session_manager.create_session(
                workspace_dir=workspace,
                agent_id=agent_id,
                account_id=_current_account_id(),
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=500, detail="Failed to create session")
        return CreateSessionResponse(**session)

    @app.get("/api/sessions", response_model=list[SessionSummary])
    async def list_sessions():
        """List all persisted sessions."""
        return [
            SessionSummary(**session)
            for session in session_manager.list_sessions(account_id=_current_account_id())
        ]

    @app.get("/api/session-history", response_model=list[SessionHistorySearchResult])
    async def search_session_history(
        query: str,
        session_id: str | None = None,
        agent_id: str | None = None,
        source_type: Literal[
            "all",
            "session_message",
            "run_goal",
            "run_completion",
            "run_failure",
            "shared_context",
        ] = "all",
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 10,
    ):
        """Search persisted history for the current account across sessions and runs."""
        _require_feature(
            _session_retrieval_enabled(),
            "enable_session_retrieval",
            "Session retrieval APIs",
        )
        try:
            source_types = None if source_type == "all" else [source_type]
            results = session_manager.search_session_history(
                query,
                account_id=_current_account_id(),
                session_id=str(session_id or "").strip() or None,
                agent_id=str(agent_id or "").strip() or None,
                source_types=source_types,
                date_from=str(date_from or "").strip() or None,
                date_to=str(date_to or "").strip() or None,
                limit=max(1, min(int(limit), 20)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return [SessionHistorySearchResult(**item) for item in results]

    @app.get("/api/sessions/{session_id}", response_model=SessionSummary)
    async def get_session(session_id: str):
        """Get session metadata."""
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionSummary(**session)

    @app.put("/api/sessions/{session_id}/agent", response_model=SessionSummary)
    async def switch_session_agent(session_id: str, req: SessionAgentUpdateRequest):
        """Switch the agent template bound to one existing session."""
        try:
            await session_manager.switch_session_agent(
                session_id,
                req.agent_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Session or agent not found")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionSummary(**session)

    @app.get("/api/sessions/{session_id}/messages", response_model=SessionDetail)
    async def get_session_messages(session_id: str):
        """Get session metadata and full message history."""
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = session_manager.get_session_messages(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        return SessionDetail(**session, messages=messages)

    @app.post(
        "/api/sessions/{session_id}/uploads",
        response_model=SessionUploadsResponse,
        status_code=201,
    )
    async def upload_session_files(session_id: str, request: Request):
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        uploads, run_id = await _parse_upload_multipart_request(request)
        try:
            created_uploads = session_manager.create_session_uploads(
                session_id,
                uploads,
                account_id=_current_account_id(),
                run_id=run_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return SessionUploadsResponse(session_id=session_id, uploads=created_uploads)

    @app.get("/api/sessions/{session_id}/uploads", response_model=SessionUploadsResponse)
    async def list_session_uploads(session_id: str):
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            uploads = session_manager.list_session_uploads(
                session_id,
                account_id=_current_account_id(),
                strict=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return SessionUploadsResponse(session_id=session_id, uploads=uploads)

    @app.get("/api/uploads/{upload_id}")
    async def download_upload(
        upload_id: str,
        disposition: Literal["attachment", "inline"] = "attachment",
    ):
        try:
            upload, upload_path = session_manager.resolve_upload_path(
                upload_id,
                account_id=_current_account_id(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return _build_workspace_file_response(
            file_path=upload_path,
            display_name=upload.original_name or upload.safe_name,
            mime_type=upload.mime_type,
            disposition=disposition,
        )

    @app.get("/api/uploads/{upload_id}/preview", response_model=FilePreviewResponse)
    async def preview_upload(upload_id: str):
        try:
            upload, upload_path = session_manager.resolve_upload_path(
                upload_id,
                account_id=_current_account_id(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return _build_file_preview_response(
            target_kind="upload",
            target_id=upload.id,
            file_path=upload_path,
            display_name=upload.original_name or upload.safe_name,
            mime_type=upload.mime_type,
            size_bytes=upload.size_bytes,
        )

    @app.get("/api/sessions/{session_id}/shared-context", response_model=SharedContextResponse)
    async def get_session_shared_context(session_id: str, limit: int = 200):
        """Get shared-context entries for one session."""
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        entries = session_manager.get_shared_context_entries(
            session_id,
            limit=limit,
            account_id=_current_account_id(),
            strict=True,
        )
        return SharedContextResponse(
            session_id=session_id,
            entries=[SharedContextEntry(**entry) for entry in entries],
        )

    @app.delete("/api/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session."""
        ok = await session_manager.delete_session(
            session_id,
            account_id=_current_account_id(),
        )
        if not ok:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"status": "deleted"}

    # --- Tool & Agent Configuration Endpoints ---
    
    @app.get("/api/tools")
    async def list_tools():
        """List all available standard tools."""
        return AVAILABLE_TOOLS

    @app.get("/api/agents")
    async def list_agents():
        """List all agent templates."""
        if not session_manager._agent_store:
            raise HTTPException(500, "AgentStore not initialized")
        return [
            _serialize_agent_with_status(agent)
            for agent in session_manager._agent_store.list_agent_templates(
                account_id=_current_account_id()
            )
        ]

    @app.get("/api/agents/{agent_id}")
    async def get_agent(agent_id: str):
        """Get agent template by ID."""
        if not session_manager._agent_store:
            raise HTTPException(500, "AgentStore not initialized")
        agent = session_manager._agent_store.get_agent_template(
            agent_id,
            account_id=_current_account_id(),
        )
        if agent is None and session_manager._agent_store.get_agent_template(agent_id) is not None:
            raise PermissionError("Agent template does not belong to the current account.")
        if not agent:
            raise HTTPException(404, "Agent not found")
        return _serialize_agent_with_status(agent)

    def _require_integration(integration_id: str):
        integration = integration_router.get_integration(
            integration_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if integration is None:
            raise HTTPException(status_code=404, detail="Integration not found")
        return integration

    def _require_binding(binding_id: str):
        binding = integration_router.get_binding(
            binding_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if binding is None:
            raise HTTPException(status_code=404, detail="Binding not found")
        return binding

    async def _require_routing_rule(rule_id: str):
        rule = await integration_admin_service.get_routing_rule(
            rule_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if rule is None:
            raise HTTPException(status_code=404, detail="Routing rule not found")
        return rule

    @app.get("/api/integrations", response_model=list[IntegrationResponse])
    async def list_integrations(
        kind: str | None = None,
        status: str | None = None,
        include_deleted: bool = False,
    ):
        records = await integration_admin_service.list_integrations(
            account_id=_current_account_id(),
            kind=kind,
            status=status,
            include_deleted=include_deleted,
        )
        return [await _serialize_integration(record) for record in records]

    @app.post("/api/integrations", response_model=IntegrationResponse, status_code=201)
    async def create_integration(req: IntegrationCreateRequest):
        try:
            record = await integration_admin_service.create_integration(
                account_id=_current_account_id(),
                name=req.name,
                kind=req.kind,
                display_name=req.display_name,
                tenant_id=req.tenant_id,
                config=req.config,
                metadata=req.metadata,
                credentials=[item.model_dump() for item in req.credentials],
                enabled=(
                    req.enabled
                    if req.enabled is not None
                    else str(req.kind or "").strip().lower() == "feishu"
                ),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        await _sync_feishu_long_connection(record.id)
        await _sync_wechat_long_poll(record.id)
        return await _serialize_integration(record)

    @app.patch("/api/integrations/{integration_id}", response_model=IntegrationResponse)
    async def update_integration(integration_id: str, req: IntegrationUpdateRequest):
        try:
            record = await integration_admin_service.update_integration(
                integration_id,
                account_id=_current_account_id(),
                name=req.name,
                kind=req.kind,
                display_name=req.display_name,
                tenant_id=req.tenant_id,
                config=req.config,
                metadata=req.metadata,
                credentials=None
                if req.credentials is None
                else [item.model_dump() for item in req.credentials],
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        await _sync_feishu_long_connection(record.id)
        await _sync_wechat_long_poll(record.id)
        return await _serialize_integration(record)

    @app.delete("/api/integrations/{integration_id}", response_model=IntegrationResponse)
    async def delete_integration(integration_id: str):
        try:
            record = await integration_admin_service.soft_delete_integration(
                integration_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        await _sync_feishu_long_connection(record.id)
        await _sync_wechat_long_poll(record.id)
        return await _serialize_integration(record)

    @app.post("/api/integrations/{integration_id}/verify", response_model=IntegrationVerificationResponse)
    async def verify_integration(integration_id: str):
        try:
            result = await integration_admin_service.verify_integration(
                integration_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        return IntegrationVerificationResponse(
            success=result.ok,
            message=result.message,
            integration=await _serialize_integration(result.integration),
        )

    @app.post("/api/integrations/{integration_id}/enable", response_model=IntegrationResponse)
    async def enable_integration(integration_id: str):
        try:
            record = await integration_admin_service.set_integration_status(
                integration_id,
                account_id=_current_account_id(),
                status="active",
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        await _sync_feishu_long_connection(record.id)
        await _sync_wechat_long_poll(record.id)
        return await _serialize_integration(record)

    @app.post("/api/integrations/{integration_id}/disable", response_model=IntegrationResponse)
    async def disable_integration(integration_id: str):
        try:
            record = await integration_admin_service.set_integration_status(
                integration_id,
                account_id=_current_account_id(),
                status="disabled",
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        await _sync_feishu_long_connection(record.id)
        await _sync_wechat_long_poll(record.id)
        return await _serialize_integration(record)

    @app.get(
        "/api/integrations/{integration_id}/wechat/setup",
        response_model=WeChatSetupStatusResponse,
    )
    async def get_wechat_setup_status(integration_id: str):
        record = _require_integration(integration_id)
        if record.kind != "wechat":
            raise HTTPException(status_code=400, detail="Integration is not a WeChat channel")
        return WeChatSetupStatusResponse.model_validate(
            _get_wechat_setup_status(integration_id, record=record)
        )

    @app.post(
        "/api/integrations/{integration_id}/wechat/setup",
        response_model=WeChatSetupStatusResponse,
    )
    async def start_wechat_setup(integration_id: str):
        record = _require_integration(integration_id)
        if record.kind != "wechat":
            raise HTTPException(status_code=400, detail="Integration is not a WeChat channel")
        status = _queue_wechat_setup(integration_id)
        return WeChatSetupStatusResponse.model_validate(status)

    @app.get("/api/integrations/{integration_id}/routing-rules", response_model=list[RoutingRuleResponse])
    async def list_integration_routing_rules(
        integration_id: str,
        enabled: bool | None = None,
    ):
        try:
            rules = await integration_admin_service.list_routing_rules(
                integration_id,
                account_id=_current_account_id(),
                enabled=enabled,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        return [_serialize_routing_rule(rule) for rule in rules]

    @app.post(
        "/api/integrations/{integration_id}/routing-rules",
        response_model=RoutingRuleResponse,
        status_code=201,
    )
    async def create_integration_routing_rule(
        integration_id: str,
        req: RoutingRuleCreateRequest,
    ):
        try:
            rule = await integration_admin_service.create_routing_rule(
                account_id=_current_account_id(),
                integration_id=integration_id,
                priority=req.priority,
                match_type=req.match_type,
                match_value=req.match_value,
                agent_id=req.agent_id,
                session_strategy=req.session_strategy,
                enabled=req.enabled,
                metadata=req.metadata,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _serialize_routing_rule(rule)

    @app.patch("/api/routing-rules/{rule_id}", response_model=RoutingRuleResponse)
    async def update_routing_rule(rule_id: str, req: RoutingRuleUpdateRequest):
        await _require_routing_rule(rule_id)
        try:
            rule = await integration_admin_service.update_routing_rule(
                rule_id,
                account_id=_current_account_id(),
                priority=req.priority,
                match_type=req.match_type,
                match_value=req.match_value,
                agent_id=req.agent_id,
                session_strategy=req.session_strategy,
                enabled=req.enabled,
                metadata=req.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _serialize_routing_rule(rule)

    @app.delete("/api/routing-rules/{rule_id}", response_model=RoutingRuleResponse)
    async def delete_routing_rule(rule_id: str):
        try:
            rule = await integration_admin_service.delete_routing_rule(
                rule_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Routing rule not found")
        return _serialize_routing_rule(rule)

    @app.get("/api/integrations/{integration_id}/events", response_model=list[InboundEventResponse])
    async def list_integration_events(
        integration_id: str,
        status: str | None = None,
        limit: int = 20,
    ):
        try:
            events = await integration_admin_service.list_events(
                account_id=_current_account_id(),
                integration_id=integration_id,
                status=status,
                limit=limit,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        return [_serialize_inbound_event(event) for event in events]

    @app.get(
        "/api/integrations/{integration_id}/deliveries",
        response_model=list[OutboundDeliveryResponse],
    )
    async def list_integration_deliveries(
        integration_id: str,
        status: str | None = None,
        limit: int = 20,
    ):
        try:
            deliveries = await integration_admin_service.list_deliveries(
                account_id=_current_account_id(),
                integration_id=integration_id,
                status=status,
                limit=limit,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Integration not found")
        return [await _serialize_delivery(delivery) for delivery in deliveries]

    @app.post("/api/outbound-deliveries/{delivery_id}/retry", response_model=OutboundDeliveryResponse)
    async def retry_outbound_delivery(delivery_id: str):
        try:
            delivery = await integration_admin_service.retry_delivery(
                delivery_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Outbound delivery not found")
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return await _serialize_delivery(delivery)

    @app.get("/api/integrations/{integration_id}/bindings", response_model=list[BindingResponse])
    async def list_integration_bindings(
        integration_id: str,
        tenant_id: str | None = None,
        chat_id: str | None = None,
        thread_id: str | None = None,
        binding_scope: Literal["chat", "thread"] | None = None,
        agent_id: str | None = None,
        enabled: bool | None = None,
    ):
        _require_integration(integration_id)
        bindings = integration_router.list_bindings(
            account_id=_current_account_id(),
            integration_id=integration_id,
            tenant_id=tenant_id,
            chat_id=chat_id,
            thread_id=thread_id,
            binding_scope=binding_scope,
            agent_id=agent_id,
            enabled=enabled,
        )
        return [_serialize_binding(binding) for binding in bindings]

    @app.post("/api/integrations/{integration_id}/bindings", response_model=BindingResponse, status_code=201)
    async def create_integration_binding(integration_id: str, req: BindingCreateRequest):
        _require_integration(integration_id)
        try:
            binding = await integration_router.upsert_binding(
                account_id=_current_account_id(),
                integration_id=integration_id,
                tenant_id=req.tenant_id,
                chat_id=req.chat_id,
                thread_id=req.thread_id,
                binding_scope=req.binding_scope,
                agent_id=req.agent_id,
                metadata={"binding_source": "api", **req.metadata},
                updated_at=_utc_now_iso(),
                force_new_session=req.create_new_session,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return _serialize_binding(binding)

    @app.patch("/api/bindings/{binding_id}", response_model=BindingResponse)
    async def update_binding(binding_id: str, req: BindingUpdateRequest):
        binding = _require_binding(binding_id)
        target_enabled = binding.enabled if req.enabled is None else req.enabled
        target_metadata = dict(binding.metadata) if req.metadata is None else dict(req.metadata)
        structural_change = any(
            value is not None
            for value in (
                req.tenant_id,
                req.chat_id,
                req.thread_id,
                req.binding_scope,
                req.agent_id,
                req.metadata,
            )
        )

        if not target_enabled and not structural_change and not req.refresh_session:
            disabled_binding = integration_router.disable_binding(
                binding_id,
                account_id=_current_account_id(),
                updated_at=_utc_now_iso(),
                reason="api_disable",
            )
            return _serialize_binding(disabled_binding)

        try:
            updated_binding = await integration_router.upsert_binding(
                account_id=_current_account_id(),
                integration_id=binding.integration_id,
                tenant_id=binding.tenant_id if req.tenant_id is None else req.tenant_id,
                chat_id=binding.chat_id if req.chat_id is None else req.chat_id,
                thread_id=binding.thread_id if req.thread_id is None else req.thread_id,
                binding_scope=binding.binding_scope if req.binding_scope is None else req.binding_scope,
                agent_id=binding.agent_id if req.agent_id is None else req.agent_id,
                metadata=target_metadata,
                updated_at=_utc_now_iso(),
                force_new_session=req.refresh_session or (
                    req.agent_id is not None and req.agent_id != binding.agent_id
                ),
                enabled=True,
                preferred_binding_id=binding.id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if not target_enabled:
            updated_binding = integration_router.disable_binding(
                updated_binding.id,
                account_id=_current_account_id(),
                updated_at=_utc_now_iso(),
                reason="api_disable",
            )
        return _serialize_binding(updated_binding)

    @app.delete("/api/bindings/{binding_id}", response_model=BindingResponse)
    async def delete_binding(binding_id: str):
        _require_binding(binding_id)
        disabled_binding = integration_router.disable_binding(
            binding_id,
            account_id=_current_account_id(),
            updated_at=_utc_now_iso(),
            reason="api_delete",
        )
        return _serialize_binding(disabled_binding)

    @app.get("/api/scheduled-tasks", response_model=list[ScheduledTaskResponse])
    async def list_scheduled_tasks(
        enabled: bool | None = None,
        agent_id: str | None = None,
        integration_id: str | None = None,
        limit: int | None = None,
    ):
        return await scheduled_task_service.list_tasks(
            account_id=_current_account_id(),
            enabled=enabled,
            agent_id=agent_id,
            integration_id=integration_id,
            limit=limit,
        )

    @app.post("/api/scheduled-tasks", response_model=ScheduledTaskResponse, status_code=201)
    async def create_scheduled_task(req: ScheduledTaskCreateRequest):
        try:
            return await scheduled_task_service.create_task(
                account_id=_current_account_id(),
                name=req.name,
                cron_expression=req.cron_expression,
                agent_id=req.agent_id,
                prompt=req.prompt,
                integration_id=req.integration_id,
                target_chat_id=req.target_chat_id,
                target_thread_id=req.target_thread_id,
                reply_to_message_id=req.reply_to_message_id,
                timezone_name=req.timezone,
                enabled=req.enabled,
                metadata=req.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get("/api/scheduled-tasks/{task_id}", response_model=ScheduledTaskResponse)
    async def get_scheduled_task(task_id: str):
        try:
            return await scheduled_task_service.get_task(
                task_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Scheduled task not found")

    @app.patch("/api/scheduled-tasks/{task_id}", response_model=ScheduledTaskResponse)
    async def update_scheduled_task(task_id: str, req: ScheduledTaskUpdateRequest):
        try:
            return await scheduled_task_service.update_task(
                task_id,
                account_id=_current_account_id(),
                name=req.name,
                cron_expression=req.cron_expression,
                agent_id=req.agent_id,
                prompt=req.prompt,
                integration_id=req.integration_id,
                target_chat_id=req.target_chat_id,
                target_thread_id=req.target_thread_id,
                reply_to_message_id=req.reply_to_message_id,
                timezone_name=req.timezone,
                enabled=req.enabled,
                metadata=req.metadata,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Scheduled task not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.delete("/api/scheduled-tasks/{task_id}", response_model=ScheduledTaskResponse)
    async def delete_scheduled_task(task_id: str):
        try:
            return await scheduled_task_service.delete_task(
                task_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Scheduled task not found")

    @app.post("/api/scheduled-tasks/{task_id}/enable", response_model=ScheduledTaskResponse)
    async def enable_scheduled_task(task_id: str):
        try:
            return await scheduled_task_service.set_task_enabled(
                task_id,
                account_id=_current_account_id(),
                enabled=True,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Scheduled task not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/scheduled-tasks/{task_id}/disable", response_model=ScheduledTaskResponse)
    async def disable_scheduled_task(task_id: str):
        try:
            return await scheduled_task_service.set_task_enabled(
                task_id,
                account_id=_current_account_id(),
                enabled=False,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Scheduled task not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.post("/api/scheduled-tasks/{task_id}/run", response_model=ScheduledTaskExecutionResponse)
    async def run_scheduled_task_now(task_id: str):
        try:
            return await scheduled_task_service.run_task_now(
                task_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Scheduled task not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @app.get(
        "/api/scheduled-tasks/{task_id}/executions",
        response_model=list[ScheduledTaskExecutionResponse],
    )
    async def list_scheduled_task_executions(task_id: str, limit: int = 20):
        try:
            return await scheduled_task_service.list_task_executions(
                task_id,
                account_id=_current_account_id(),
                limit=limit,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Scheduled task not found")

    @app.get(
        "/api/scheduled-task-executions/{execution_id}",
        response_model=ScheduledTaskExecutionDetailResponse,
    )
    async def get_scheduled_task_execution_detail(execution_id: str):
        try:
            return await scheduled_task_service.get_execution_detail(
                execution_id,
                account_id=_current_account_id(),
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail="Scheduled task execution not found",
            )

    @app.get("/api/skills/search")
    async def search_skills(keyword: str):
        """Search skill packages from the configured clawhub registry."""
        normalized = keyword.strip()
        if not normalized:
            raise HTTPException(400, "Keyword is required")
        try:
            return await _search_clawhub_skills(normalized)
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post("/api/agents")
    async def create_agent(req: AgentCreateRequest):
        """Create a new custom agent template."""
        if not session_manager._agent_store:
            raise HTTPException(500, "AgentStore not initialized")
        try:
            payload = req.model_dump()
            selected_skill_packages = payload.pop("selected_skill_packages", [])
            payload["tools"] = _default_agent_tools(session_manager)
            agent_id = session_manager._agent_store.generate_agent_id()
            payload["agent_id"] = agent_id
            payload["skills"] = []
            payload["account_id"] = _current_account_id()
            created = session_manager._agent_store.create_agent_template(**payload)
            _queue_skill_install(created["id"], selected_skill_packages)
            return _serialize_agent_with_status(created)
        except Exception as e:
            if "created" in locals():
                with contextlib.suppress(Exception):
                    session_manager._agent_store.delete_agent(created["id"])
            elif "agent_id" in locals():
                shutil.rmtree(
                    session_manager._agent_store.get_agent_dir(
                        agent_id,
                        account_id=_current_account_id(),
                    ),
                    ignore_errors=True,
                )
            raise HTTPException(400, str(e))

    @app.put("/api/agents/{agent_id}")
    async def update_agent(agent_id: str, req: AgentUpdateRequest):
        """Update a custom agent template."""
        if not session_manager._agent_store:
            raise HTTPException(500, "AgentStore not initialized")
        try:
            payload = req.model_dump(exclude_unset=True)
            selected_skill_packages = payload.pop("selected_skill_packages", None)
            normalized_skill_packages = _ensure_skill_install_available(
                agent_id,
                selected_skill_packages,
            )
            payload["tools"] = _default_agent_tools(session_manager)
            updated = session_manager._agent_store.update_agent_template(
                agent_id,
                account_id=_current_account_id(),
                **payload,
            )
            if not updated:
                if session_manager._agent_store.get_agent_template(agent_id) is not None:
                    raise PermissionError("Agent template does not belong to the current account.")
                raise HTTPException(404, "Agent not found")

            if normalized_skill_packages:
                _queue_skill_install(agent_id, normalized_skill_packages)

            return _serialize_agent_with_status(updated)
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.post("/api/agents/{agent_id}/skills/install")
    async def install_agent_skills(agent_id: str, req: SkillInstallRequest):
        """Queue skill installation for an existing custom agent."""
        if not session_manager._agent_store:
            raise HTTPException(500, "AgentStore not initialized")

        agent = session_manager._agent_store.get_agent_template(
            agent_id,
            account_id=_current_account_id(),
        )
        if agent is None and session_manager._agent_store.get_agent_template(agent_id) is not None:
            raise PermissionError("Agent template does not belong to the current account.")
        if not agent:
            raise HTTPException(404, "Agent not found")

        try:
            _queue_skill_install(agent_id, req.package_names)
            refreshed = session_manager._agent_store.get_agent_template(
                agent_id,
                account_id=_current_account_id(),
            )
            return _serialize_agent_with_status(refreshed)
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/agents/{agent_id}/skills/{skill_name}")
    async def delete_agent_skill(agent_id: str, skill_name: str):
        """Delete one installed skill from an existing custom agent."""
        if not session_manager._agent_store:
            raise HTTPException(500, "AgentStore not initialized")

        agent = session_manager._agent_store.get_agent_template(
            agent_id,
            account_id=_current_account_id(),
        )
        if agent is None and session_manager._agent_store.get_agent_template(agent_id) is not None:
            raise PermissionError("Agent template does not belong to the current account.")
        if not agent:
            raise HTTPException(404, "Agent not found")

        try:
            _ensure_skill_install_idle(agent_id)
            updated = session_manager._agent_store.delete_agent_skill(
                agent_id,
                skill_name,
                account_id=_current_account_id(),
            )
            if not updated:
                raise HTTPException(404, "Agent not found")
            return _serialize_agent_with_status(updated)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400, str(e))

    @app.delete("/api/agents/{agent_id}")
    async def delete_agent(agent_id: str):
        """Delete a custom agent template."""
        if not session_manager._agent_store:
            raise HTTPException(500, "AgentStore not initialized")
        task = skill_install_tasks.pop(agent_id, None)
        if task and not task.done():
            task.cancel()
        deleted = session_manager._agent_store.delete_agent(
            agent_id,
            account_id=_current_account_id(),
        )
        if not deleted:
            if session_manager._agent_store.get_agent_template(agent_id) is not None:
                raise PermissionError("Agent template does not belong to the current account.")
            raise HTTPException(404, "Agent not found or is system agent")
        skill_install_statuses.pop(agent_id, None)
        return {"status": "deleted"}

    @app.post("/api/agents/brainstorm")
    async def brainstorm_agent(req: BrainstormRequest):
        """Auto-complete agent configurations using the configured LLM."""
        try:
            llm_client = session_manager._get_account_llm_runtime(
                AgentRuntimeContext(
                    session_id="brainstorm",
                    account_id=_current_account_id(),
                    is_main_agent=True,
                ),
                None,
            ).client
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(500, f"Failed to initialize brainstorm client: {str(e)}")

        # Check if we have at least one field
        if not req.name.strip() and not req.description.strip() and not req.system_prompt.strip():
            raise HTTPException(400, "At least one property must be provided to brainstorm")

        prompt = f"""You are an expert AI agent architect.
Given the following partial configuration for a new agent, creatively fill in all missing fields (name, description, system_prompt) and decompose the agent into 2-4 independent capability dimensions that it will likely need.
Each capability dimension should represent one concrete ability area that can map to a different skill search direction, such as search/research, content creation, video production, data analysis, publishing, automation, etc.
Ensure the System Prompt is professional, actionable, and provides clear persona/rules.
For each capability dimension, provide:
- "name": a concise capability label
- "keyword": a short search keyword optimized for querying a skill registry
- "reason": one short sentence explaining why this capability is needed
Search keywords should prefer concise Chinese wording whenever possible and usually be 2-8 characters unless the concept is primarily known by an English proper noun.
If the task is clearly multi-stage, split it into multiple dimensions instead of collapsing everything into one keyword.
If fields are already provided, keep them or refine them to be more professional.
- Current Name: {req.name}
- Current Description: {req.description}
- Current System Prompt: {req.system_prompt}

Return the response purely as a valid JSON object with EXACT keys: "name", "description", "system_prompt", and "capability_dimensions".
"capability_dimensions" must be an array of 2-4 objects with EXACT keys: "name", "keyword", "reason".
Do not use markdown tags, just return raw JSON."""
        
        from .schema import Message
        msg = Message(role="user", content=prompt)
        try:
            resp = await llm_client.generate(messages=[msg])
            result = _parse_brainstorm_response_content(resp.content)
            result["name"] = str(result.get("name") or req.name).strip()
            result["description"] = str(result.get("description") or req.description).strip()
            result["system_prompt"] = str(result.get("system_prompt") or req.system_prompt).strip()
            capability_dimensions = _normalize_capability_dimensions(
                result.get("capability_dimensions")
            )
            if not capability_dimensions:
                legacy_keyword = str(result.get("recommended_skill_keyword", "")).strip()
                if legacy_keyword:
                    capability_dimensions = [
                        {
                            "name": "核心能力",
                            "keyword": legacy_keyword,
                            "reason": "Fallback capability dimension generated from the legacy keyword.",
                        }
                    ]

            result["capability_dimensions"] = capability_dimensions
            result["recommended_skill_keyword"] = (
                capability_dimensions[0]["keyword"] if capability_dimensions else ""
            )
            result["skill_capability_groups"] = await _search_capability_skill_groups(
                capability_dimensions
            )
            return result
        except Exception as e:
            raise HTTPException(500, f"Failed to generate brainstorm: {str(e)}")

    # --- Chat Endpoint ---

    def _build_run_stream_response(
        run_id: str,
        *,
        after_event_id: int | None = None,
        trace_offset: int | None = None,
    ) -> StreamingResponse:
        active_account_id = _current_account_id() if auth_enabled else None

        async def event_stream():
            queue: asyncio.Queue[dict | None] = asyncio.Queue()
            next_event_id = 0 if after_event_id is None else max(0, after_event_id + 1)

            async def produce_events():
                try:
                    async for event in session_manager.stream_run(
                        run_id,
                        account_id=active_account_id,
                        after_event_id=after_event_id,
                    ):
                        await queue.put(event)
                finally:
                    await queue.put(None)

            if trace_offset is not None:
                for trace_event in session_manager.list_run_trace(
                    run_id,
                    account_id=active_account_id,
                    offset=trace_offset,
                ):
                    payload = {"type": "trace", "data": trace_event}
                    yield _encode_sse_message(
                        data=json.dumps(payload, ensure_ascii=False),
                        event_name="trace",
                    )

            producer = asyncio.create_task(produce_events())
            try:
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # Keep SSE alive and reduce proxy buffering side effects.
                        yield ": ping\n\n"
                        continue

                    if event is None:
                        break
                    data = json.dumps(event, ensure_ascii=False)
                    yield _encode_sse_message(
                        data=data,
                        event_name=_classify_sse_event_name(event),
                        event_id=next_event_id,
                    )
                    next_event_id += 1
                yield _encode_sse_message(data="[DONE]", event_name="state")
            finally:
                if not producer.done():
                    producer.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await producer

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Run-ID": run_id,
            },
        )

    @app.post("/api/sessions/{session_id}/chat")
    async def chat(session_id: str, req: ChatRequest):
        """Send a message and stream back Agent events via SSE."""
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            run = session_manager.start_chat_run(
                session_id,
                req.message,
                account_id=_current_account_id(),
                attachment_ids=req.attachment_ids,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        return _build_run_stream_response(run.id)

    @app.post("/api/integrations/{integration_id}/webhook")
    async def integration_webhook(integration_id: str, request: Request):
        try:
            result = await integration_gateway.handle_http_request(integration_id, request)
        except IntegrationGatewayError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        try:
            await integration_run_bridge.bridge_gateway_result(result)
        except IntegrationRunBridgeError as exc:
            raise HTTPException(status_code=500, detail=exc.detail)
        return _build_quick_ack_response(result.quick_ack)

    @app.post("/api/integrations/{channel_kind}/{integration_id}/webhook")
    async def integration_webhook_by_kind(
        channel_kind: str,
        integration_id: str,
        request: Request,
    ):
        try:
            result = await integration_gateway.handle_http_request(
                integration_id,
                request,
                expected_kind=channel_kind,
            )
        except IntegrationGatewayError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)
        try:
            await integration_run_bridge.bridge_gateway_result(result)
        except IntegrationRunBridgeError as exc:
            raise HTTPException(status_code=500, detail=exc.detail)
        return _build_quick_ack_response(result.quick_ack)

    @app.post("/api/runs", status_code=201)
    async def create_run(req: CreateRunRequest):
        """Create one durable run without immediately attaching an SSE stream."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        try:
            session_policy_override = {
                key: value
                for key, value in {
                    "workspace_policy": req.workspace_policy,
                    "approval_policy": req.approval_policy,
                    "run_policy": req.run_policy,
                    "delegation_policy": req.delegation_policy,
                }.items()
                if value is not None
            }
            run = session_manager.start_run(
                req.session_id,
                req.goal,
                account_id=_current_account_id(),
                parent_run_id=req.parent_run_id,
                session_policy_override=session_policy_override or None,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        return run.model_dump(mode="json")

    @app.get("/api/runs")
    async def list_runs(
        session_id: str | None = None,
        status: str | None = None,
        parent_run_id: str | None = None,
        limit: int | None = None,
        failed_only: bool = False,
    ):
        """List durable runs with optional filters."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        effective_status = "failed" if failed_only else status
        return session_manager.list_runs(
            account_id=_current_account_id(),
            session_id=session_id,
            status=effective_status,
            parent_run_id=parent_run_id,
            limit=limit,
        )

    @app.get("/api/runs/failed")
    async def list_failed_runs(limit: int = 20):
        """List recent failed runs."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        return session_manager.list_runs(
            account_id=_current_account_id(),
            status="failed",
            limit=limit,
        )

    @app.get("/api/runs/metrics")
    async def get_run_metrics(session_id: str | None = None, limit: int | None = None):
        """Return aggregated durable-run metrics for diagnostics dashboards."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        return session_manager.get_run_metrics(
            account_id=_current_account_id(),
            session_id=session_id,
            limit=limit,
        )

    @app.get("/api/runs/metrics/export")
    async def export_run_metrics(
        session_id: str | None = None,
        limit: int | None = None,
        format: str = "prometheus",
    ):
        """Export aggregated run metrics for monitoring systems."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        normalized_format = format.strip().lower()
        try:
            payload = session_manager.export_run_metrics(
                format=normalized_format,
                account_id=_current_account_id(),
                session_id=session_id,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        if normalized_format == "prometheus":
            return PlainTextResponse(
                str(payload),
                media_type="text/plain; version=0.0.4; charset=utf-8",
            )
        return payload

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        """Get persisted metadata for one run."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return run

    @app.get("/api/runs/{run_id}/steps")
    async def get_run_steps(run_id: str):
        """List durable execution steps for one run."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return session_manager.list_run_steps(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )

    @app.get("/api/runs/{run_id}/trace")
    async def get_run_trace(run_id: str, offset: int | None = None):
        """List durable trace timeline events for one run."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if offset is not None and offset < 0:
            raise HTTPException(status_code=400, detail="offset must be >= 0")
        return session_manager.list_run_trace(
            run_id,
            account_id=_current_account_id(),
            strict=True,
            offset=offset,
        )

    @app.get("/api/runs/{run_id}/trace/timeline")
    async def get_run_trace_timeline(run_id: str):
        """Return a normalized timeline across the root run and all child runs."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            return session_manager.get_run_trace_timeline(
                run_id,
                account_id=_current_account_id(),
                strict=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/trace/tree")
    async def get_run_trace_tree(run_id: str):
        """Return the nested run tree for one run family."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            return session_manager.get_run_trace_tree(
                run_id,
                account_id=_current_account_id(),
                strict=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/trace/tools")
    async def get_run_trace_tools(run_id: str):
        """Return merged tool and delegate drill-down rows for one run family."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            return session_manager.get_run_tool_calls(
                run_id,
                account_id=_current_account_id(),
                strict=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/trace/export")
    async def export_run_trace(run_id: str):
        """Export one run family's trace, tree, tools, steps, and artifacts as JSON."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            return session_manager.export_run_trace(
                run_id,
                account_id=_current_account_id(),
                strict=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/trace/replay")
    async def replay_run_trace(run_id: str):
        """Return a developer-oriented replay view for one run family."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            return session_manager.replay_run_trace(
                run_id,
                account_id=_current_account_id(),
                strict=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/diagnostics")
    async def locate_run_diagnostics(run_id: str):
        """Return log and database locations that help debug one run tree."""
        _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        try:
            return session_manager.locate_run_diagnostics(
                run_id,
                account_id=_current_account_id(),
                strict=True,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    @app.get("/api/runs/{run_id}/artifacts")
    async def get_run_artifacts(run_id: str):
        """List materialized artifacts for one run."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return session_manager.list_run_artifacts(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )

    @app.get("/api/artifacts/{artifact_id}")
    async def download_artifact(
        artifact_id: str,
        disposition: Literal["attachment", "inline"] = "attachment",
    ):
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        try:
            artifact, artifact_path = session_manager.resolve_artifact_path(
                artifact_id,
                account_id=_current_account_id(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return _build_workspace_file_response(
            file_path=artifact_path,
            display_name=artifact.display_name or artifact_path.name,
            mime_type=artifact.mime_type,
            disposition=disposition,
        )

    @app.get("/api/artifacts/{artifact_id}/preview", response_model=FilePreviewResponse)
    async def preview_artifact(artifact_id: str):
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        try:
            artifact, artifact_path = session_manager.resolve_artifact_path(
                artifact_id,
                account_id=_current_account_id(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return _build_file_preview_response(
            target_kind="artifact",
            target_id=artifact.id,
            file_path=artifact_path,
            display_name=artifact.display_name or artifact_path.name,
            mime_type=artifact.mime_type,
            size_bytes=artifact.size_bytes,
            format_hint=artifact.format,
        )

    @app.get("/api/runs/{run_id}/events")
    async def stream_run_events(
        run_id: str,
        trace_offset: int | None = None,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ):
        """Reconnect to one durable run event stream."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        if trace_offset is not None:
            _require_feature(_run_trace_enabled(), "enable_run_trace", "Run trace diagnostics")
        run = session_manager.get_run_info(
            run_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        parsed_last_event_id: int | None = None
        if last_event_id is not None and last_event_id.strip():
            try:
                parsed_last_event_id = int(last_event_id.strip())
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID must be an integer event id",
                ) from exc
            if parsed_last_event_id < 0:
                raise HTTPException(status_code=400, detail="Last-Event-ID must be >= 0")
        if trace_offset is not None and trace_offset < 0:
            raise HTTPException(status_code=400, detail="trace_offset must be >= 0")
        return _build_run_stream_response(
            run_id,
            after_event_id=parsed_last_event_id,
            trace_offset=trace_offset,
        )

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        """Cancel one queued, running, or interrupted run."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        try:
            run = await session_manager.cancel_run(
                run_id,
                account_id=_current_account_id(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if not run:
            raise HTTPException(status_code=500, detail="Run manager not initialized")
        return run

    @app.post("/api/runs/{run_id}/resume")
    async def resume_run(run_id: str):
        """Resume one interrupted run from durable state."""
        _require_feature(_durable_runs_enabled(), "enable_durable_runs", "Durable run APIs")
        try:
            run = await session_manager.resume_run(
                run_id,
                account_id=_current_account_id(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not run:
            raise HTTPException(status_code=500, detail="Run manager not initialized")
        return run

    @app.get("/api/approvals")
    async def list_approvals(
        status: str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ):
        """List persisted approval requests."""
        _require_feature(_approval_flow_enabled(), "enable_approval_flow", "Approval APIs")
        return session_manager.list_approval_requests(
            account_id=_current_account_id(),
            status=status,
            run_id=run_id,
            session_id=session_id,
        )

    @app.post("/api/approvals/{approval_id}/grant")
    async def grant_approval(
        approval_id: str,
        req: ApprovalDecisionRequest | None = None,
    ):
        """Mark one approval request as granted."""
        _require_feature(_approval_flow_enabled(), "enable_approval_flow", "Approval APIs")
        try:
            return session_manager.resolve_approval_request(
                approval_id,
                account_id=_current_account_id(),
                status="granted",
                decision_notes=req.decision_notes if req is not None else "",
                decision_scope=req.decision_scope if req is not None else "once",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/approvals/{approval_id}/deny")
    async def deny_approval(
        approval_id: str,
        req: ApprovalDecisionRequest | None = None,
    ):
        """Mark one approval request as denied."""
        _require_feature(_approval_flow_enabled(), "enable_approval_flow", "Approval APIs")
        try:
            return session_manager.resolve_approval_request(
                approval_id,
                account_id=_current_account_id(),
                status="denied",
                decision_notes=req.decision_notes if req is not None else "",
                decision_scope=req.decision_scope if req is not None else "once",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/sessions/{session_id}/interrupt")
    async def interrupt_chat(session_id: str):
        """Interrupt the currently running agent task for a session."""
        session = session_manager.get_session_info(
            session_id,
            account_id=_current_account_id(),
            strict=True,
        )
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        interrupted = await session_manager.interrupt_session(session_id)
        return {
            "status": "interrupt_requested" if interrupted else "idle",
            "session_id": session_id,
        }

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """Serve the chat UI."""
        index_file = static_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=index_file.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Clavi Agent</h1><p>Frontend not found.</p>")

    return app


app = create_app()


# --- Entry point ---
def main():
    """Start the web server."""
    import uvicorn

    uvicorn.run(
        "clavi_agent.server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()


