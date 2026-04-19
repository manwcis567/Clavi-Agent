"""Runtime tool for sending a workspace file back through the bound channel."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from .base import Tool, ToolResult

if TYPE_CHECKING:
    from ..agent_runtime import AgentRuntimeContext

RuntimeChannelFileSender = Callable[
    [Any, Path, str, str],
    Awaitable[dict[str, Any]],
]


class SendChannelFileTool(Tool):
    """Send a local workspace file to the user via the active bound channel."""

    def __init__(
        self,
        *,
        workspace_dir: str,
        runtime_context: "AgentRuntimeContext",
        sender: RuntimeChannelFileSender,
    ) -> None:
        self.workspace_dir = Path(workspace_dir).resolve()
        self.runtime_context = runtime_context
        self._sender = sender

    @property
    def name(self) -> str:
        return "send_channel_file"

    @property
    def description(self) -> str:
        return (
            "Send one local file in the current workspace back to the user through the "
            "bound Feishu conversation. Use this when you need to proactively return a "
            "generated or revised file before the run finishes."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or workspace-relative path to the file to send.",
                },
                "file_name": {
                    "type": "string",
                    "description": "Optional file name to display in Feishu. Defaults to the local file name.",
                },
                "text_fallback": {
                    "type": "string",
                    "description": "Optional fallback text used if the channel cannot send the file directly.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        path: str,
        file_name: str | None = None,
        text_fallback: str | None = None,
    ) -> ToolResult:
        try:
            resolved_path = self._resolve_path(path)
        except ValueError as exc:
            return ToolResult(success=False, content="", error=str(exc))

        if not resolved_path.exists():
            return ToolResult(success=False, content="", error=f"File not found: {path}")
        if not resolved_path.is_file():
            return ToolResult(success=False, content="", error=f"Path is not a file: {path}")

        outbound_name = str(file_name or resolved_path.name).strip() or resolved_path.name
        fallback_text = str(text_fallback or "").strip()

        try:
            delivery = await self._sender(
                self.runtime_context,
                resolved_path,
                outbound_name,
                fallback_text,
            )
        except Exception as exc:
            return ToolResult(success=False, content="", error=f"Failed to send file: {exc}")

        return ToolResult(
            success=True,
            content=f"Sent file via {self.runtime_context.channel_kind or 'bound channel'}: {outbound_name}",
            metadata=delivery,
        )

    def _resolve_path(self, raw_path: str) -> Path:
        normalized = str(raw_path or "").strip()
        if not normalized:
            raise ValueError("Parameter 'path' is required.")

        path_obj = Path(normalized)
        resolved = (
            path_obj.resolve()
            if path_obj.is_absolute()
            else (self.workspace_dir / path_obj).resolve()
        )
        try:
            resolved.relative_to(self.workspace_dir)
        except ValueError as exc:
            raise ValueError("send_channel_file only allows files inside the current workspace.") from exc
        return resolved
