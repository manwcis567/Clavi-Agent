"""Agent run logger."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .config import Config
from .schema import Message, ToolCall


class AgentLogger:
    """Agent run logger.

    Responsible for recording the complete interaction process of each agent run, including:
    - LLM requests and responses
    - Tool calls and results
    """

    def __init__(self, config: Config | None = None):
        """Initialize logger

        Logs are stored in the directory specified by config.agent.log_dir, defaulting to ~/.clavi-agent/log/
        """
        if config:
            # Expand user path if it contains ~
            log_dir_path = config.agent.log_dir.replace("~", str(Path.home()))
            # If the path is relative, make it relative to the project directory
            if not Path(log_dir_path).is_absolute():
                log_dir_path = Path.cwd() / log_dir_path
            self.log_dir = Path(log_dir_path)
        else:
            # Use default path if no config provided
            self.log_dir = Path.home() / ".clavi-agent" / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = None
        self.log_index = 0

    @staticmethod
    def _sanitize_path_token(value: str, *, max_length: int = 80) -> str:
        """Convert one identifier into a filename-safe token."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
        if not sanitized:
            return ""
        return sanitized[:max_length]

    def start_new_run(
        self,
        *,
        run_id: str | None = None,
        agent_name: str | None = None,
    ):
        """Start new run, create new log file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = ["agent_run", timestamp]
        safe_run_id = self._sanitize_path_token(run_id or "")
        safe_agent_name = self._sanitize_path_token(agent_name or "")
        if safe_run_id:
            filename_parts.append(safe_run_id)
        if safe_agent_name:
            filename_parts.append(safe_agent_name)
        log_filename = f"{'_'.join(filename_parts)}.log"
        self.log_file = self.log_dir / log_filename
        self.log_index = 0

        # Write log header
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"Agent Run Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Convert values into JSON-serializable structures."""
        if isinstance(value, BaseModel):
            return cls._normalize_value(value.model_dump(mode="python"))
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): cls._normalize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [cls._normalize_value(item) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)

    def log_event(self, event_type: str, payload: dict[str, Any] | None = None):
        """Log one structured runtime event."""
        self.log_index += 1
        content = json.dumps(
            {
                "event_type": event_type,
                "payload": self._normalize_value(payload or {}),
            },
            indent=2,
            ensure_ascii=False,
        )
        self._write_log("EVENT", content)

    def log_request(self, messages: list[Message], tools: list[Any] | None = None):
        """Log LLM request.

        Args:
            messages: Message list
            tools: Tool list (optional)
        """
        # Build complete request data structure
        request_data = {
            "messages": [],
            "tools": [],
        }

        # Convert messages to JSON serializable format
        for msg in messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
            }
            if msg.thinking:
                msg_dict["thinking"] = msg.thinking
            if msg.tool_calls:
                msg_dict["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]
            if msg.tool_call_id:
                msg_dict["tool_call_id"] = msg.tool_call_id
            if msg.name:
                msg_dict["name"] = msg.name

            request_data["messages"].append(msg_dict)

        # Only record tool names
        if tools:
            request_data["tools"] = [tool.name for tool in tools]

        self.log_event("llm_request", request_data)

    def log_response(
        self,
        content: str,
        thinking: str | None = None,
        tool_calls: list[ToolCall] | None = None,
        finish_reason: str | None = None,
    ):
        """Log LLM response.

        Args:
            content: Response content
            thinking: Thinking content (optional)
            tool_calls: Tool call list (optional)
            finish_reason: Finish reason (optional)
        """
        # Build complete response data structure
        response_data = {
            "content": content,
        }

        if thinking:
            response_data["thinking"] = thinking

        if tool_calls:
            response_data["tool_calls"] = [tc.model_dump() for tc in tool_calls]

        if finish_reason:
            response_data["finish_reason"] = finish_reason

        self.log_event("llm_response", response_data)

    def log_tool_result(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_success: bool,
        result_content: str | None = None,
        result_error: str | None = None,
    ):
        """Log tool execution result.

        Args:
            tool_name: Tool name
            arguments: Tool arguments
            result_success: Whether successful
            result_content: Result content (on success)
            result_error: Error message (on failure)
        """
        # Build complete tool execution result data structure
        tool_result_data = {
            "tool_name": tool_name,
            "arguments": arguments,
            "success": result_success,
        }

        if result_success:
            tool_result_data["result"] = result_content
        else:
            tool_result_data["error"] = result_error

        self.log_event("tool_finished", tool_result_data)

    def _write_log(self, log_type: str, content: str):
        """Write log entry.

        Args:
            log_type: Log type (REQUEST, RESPONSE, TOOL_RESULT)
            content: Log content
        """
        if self.log_file is None:
            return

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "-" * 80 + "\n")
            f.write(f"[{self.log_index}] {log_type}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}\n")
            f.write("-" * 80 + "\n")
            f.write(content + "\n")

    def get_log_file_path(self) -> Path:
        """Get current log file path."""
        return self.log_file

