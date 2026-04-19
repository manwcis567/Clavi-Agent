from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMProvider(str, Enum):
    """LLM provider types."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class FunctionCall(BaseModel):
    """Function call details."""

    name: str
    arguments: dict[str, Any]  # Function arguments as dict


class ToolCall(BaseModel):
    """Tool call structure."""

    id: str
    type: str  # "function"
    function: FunctionCall


class TextContentBlock(BaseModel):
    """Structured text block for one chat message."""

    type: Literal["text"] = "text"
    text: str = ""


class UploadedFileContentBlock(BaseModel):
    """Reference one uploaded file inside a structured user message."""

    type: Literal["uploaded_file"] = "uploaded_file"
    upload_id: str
    original_name: str = ""
    safe_name: str = ""
    relative_path: str = ""
    mime_type: str = ""
    size_bytes: int | None = Field(default=None, ge=0)
    checksum: str = ""


class ArtifactRefContentBlock(BaseModel):
    """Reference one existing artifact inside a structured user message."""

    type: Literal["artifact_ref"] = "artifact_ref"
    artifact_id: str
    run_id: str | None = None
    uri: str
    display_name: str = ""
    mime_type: str = ""
    role: str = ""
    summary: str = ""


class Message(BaseModel):
    """Chat message."""

    role: str  # "system", "user", "assistant", "tool"
    content: str | list[dict[str, Any]]  # Can be string or list of content blocks
    thinking: str | None = None  # Extended thinking content for assistant messages
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None  # For tool role


class LLMResponse(BaseModel):
    """LLM response."""

    content: str
    thinking: str | None = None  # Extended thinking blocks
    tool_calls: list[ToolCall] | None = None
    finish_reason: str


def normalize_message_content(
    content: str | list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]]:
    """Validate and normalize supported message content shapes."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise TypeError(f"Unsupported message content type: {type(content)!r}")

    normalized: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            raise TypeError(f"Unsupported content block type: {type(block)!r}")

        block_type = str(block.get("type", "")).strip()
        if block_type == "text":
            validated = TextContentBlock.model_validate(block)
        elif block_type == "uploaded_file":
            validated = UploadedFileContentBlock.model_validate(block)
        elif block_type == "artifact_ref":
            validated = ArtifactRefContentBlock.model_validate(block)
        else:
            raise ValueError(f"Unsupported message content block type: {block_type or '[missing]'}")

        normalized.append(validated.model_dump(mode="python", exclude_none=True))

    return normalized


def message_content_summary(content: str | list[dict[str, Any]] | None) -> str:
    """Build a concise user-facing summary for previews and titles."""
    normalized = normalize_message_content(content)
    if isinstance(normalized, str):
        return " ".join(normalized.split())

    text_parts: list[str] = []
    upload_names: list[str] = []
    artifact_names: list[str] = []

    for block in normalized:
        block_type = block["type"]
        if block_type == "text":
            text = " ".join(str(block.get("text", "")).split())
            if text:
                text_parts.append(text)
            continue

        if block_type == "uploaded_file":
            name = str(
                block.get("original_name")
                or block.get("safe_name")
                or block.get("upload_id")
                or ""
            ).strip()
            if name:
                upload_names.append(name)
            continue

        if block_type == "artifact_ref":
            name = str(
                block.get("display_name")
                or block.get("uri")
                or block.get("artifact_id")
                or ""
            ).strip()
            if name:
                artifact_names.append(name)

    summary_parts: list[str] = []
    if text_parts:
        summary_parts.append(" ".join(text_parts))
    if upload_names:
        suffix = " 等" if len(upload_names) > 3 else ""
        summary_parts.append(f"已附带上传文件：{', '.join(upload_names[:3])}{suffix}")
    if artifact_names:
        suffix = " 等" if len(artifact_names) > 3 else ""
        summary_parts.append(f"已引用产物：{', '.join(artifact_names[:3])}{suffix}")

    if summary_parts:
        return " ".join(summary_parts)
    if normalized:
        return "结构化消息"
    return ""


def render_message_content_for_model(content: str | list[dict[str, Any]] | None) -> str:
    """Render structured content into model-visible prompt text."""
    normalized = normalize_message_content(content)
    if isinstance(normalized, str):
        return normalized

    text_blocks: list[str] = []
    uploaded_file_blocks: list[UploadedFileContentBlock] = []
    artifact_blocks: list[ArtifactRefContentBlock] = []

    for block in normalized:
        block_type = block["type"]
        if block_type == "text":
            text = str(block.get("text", "")).strip()
            if text:
                text_blocks.append(text)
            continue

        if block_type == "uploaded_file":
            uploaded_file_blocks.append(UploadedFileContentBlock.model_validate(block))
            continue

        if block_type == "artifact_ref":
            artifact_blocks.append(ArtifactRefContentBlock.model_validate(block))

    sections: list[str] = []
    if text_blocks:
        sections.append("\n\n".join(text_blocks))

    if uploaded_file_blocks:
        lines = [
            "已附带上传文件：",
            "处理规则：先检查原文件内容；除非用户明确要求覆盖原文件，否则请在同一上传目录生成修订副本。",
        ]
        for index, block in enumerate(uploaded_file_blocks, start=1):
            name = block.original_name or block.safe_name or block.upload_id
            lines.append(f"{index}. 文件名：{name}")
            lines.append(f"   上传 ID：{block.upload_id}")
            if block.relative_path:
                lines.append(f"   工作区路径：{block.relative_path}")
            if block.mime_type:
                lines.append(f"   MIME：{block.mime_type}")
            if block.size_bytes is not None:
                lines.append(f"   大小：{block.size_bytes} bytes")
        sections.append("\n".join(lines))

    if artifact_blocks:
        lines = ["已引用产物："]
        for index, block in enumerate(artifact_blocks, start=1):
            name = block.display_name or block.uri or block.artifact_id
            lines.append(f"{index}. 名称：{name}")
            lines.append(f"   Artifact ID：{block.artifact_id}")
            if block.run_id:
                lines.append(f"   Run ID：{block.run_id}")
            lines.append(f"   URI：{block.uri}")
            if block.mime_type:
                lines.append(f"   MIME：{block.mime_type}")
            if block.role:
                lines.append(f"   角色：{block.role}")
            if block.summary:
                lines.append(f"   摘要：{block.summary}")
        sections.append("\n".join(lines))

    if sections:
        return "\n\n".join(sections)
    if normalized:
        return "结构化消息"
    return ""
