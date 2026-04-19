"""Core Agent implementation."""

import asyncio
import inspect
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

import tiktoken

from .agent_template_models import ApprovalPolicy
from .llm import LLMClient
from .logger import AgentLogger
from .schema import LLMResponse, Message, normalize_message_content
from .tool_execution import ToolExecutionContext, prepare_tool_execution
from .tools.base import Tool, ToolResult
from .utils import calculate_display_width

if TYPE_CHECKING:
    from .agent_runtime import AgentRuntimeContext, AgentRuntimeHooks
    from .agent_template_models import AgentTemplateSnapshot


# ANSI color codes
class Colors:
    """Terminal color definitions"""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Foreground colors
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class AgentInterrupted(Exception):
    """Raised when an in-flight agent run is interrupted by the caller."""


class Agent:
    """Single agent with basic tools and MCP support."""

    def __init__(
        self,
        llm_client: LLMClient,
        system_prompt: str,
        tools: list[Tool],
        max_steps: int = 50,
        workspace_dir: str = "./workspace",
        token_limit: int = 80000,  # Summary triggered when tokens exceed this value
        config=None,  # Optional config parameter
    ):
        self.llm = llm_client
        self.tools = {tool.name: tool for tool in tools}
        self.max_steps = max_steps
        self.token_limit = token_limit
        self.workspace_dir = Path(workspace_dir)
        self.config = config

        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.system_prompt = self._compose_system_prompt(system_prompt)

        # Initialize message history
        self.messages: list[Message] = [Message(role="system", content=self.system_prompt)]

        # Initialize logger with config if available
        self.logger = AgentLogger(config)
        self._interrupt_requested = False
        self.runtime_context: AgentRuntimeContext | None = None
        self.runtime_hooks: AgentRuntimeHooks | None = None
        self.template_snapshot: AgentTemplateSnapshot | None = None
        self.manual_runtime_override = False
        self._runtime_auto_approve_tools: set[str] = set()
        self._startup_trace_data: dict[str, Any] = {}
        self.llm_fingerprint: str | None = None
        self.runtime_prompt_seed: str | None = None

    def add_user_message(self, content: str | list[dict[str, Any]]):
        """Add a user message to history."""
        self.messages.append(
            Message(role="user", content=normalize_message_content(content))
        )

    def bind_runtime(
        self,
        *,
        runtime_context: "AgentRuntimeContext | None" = None,
        runtime_hooks: "AgentRuntimeHooks | None" = None,
        template_snapshot: "AgentTemplateSnapshot | None" = None,
    ) -> None:
        """Attach run-scoped runtime metadata and hooks to this agent."""
        self.runtime_context = runtime_context
        self.runtime_hooks = runtime_hooks
        self.template_snapshot = template_snapshot

    def set_startup_trace_data(self, payload: dict[str, Any] | None) -> None:
        """Persist compact startup metadata for the next run_started trace event."""
        self._startup_trace_data = dict(payload or {})

    def _compose_system_prompt(self, system_prompt: str) -> str:
        """Normalize the runtime system prompt with workspace context."""
        normalized_prompt = str(system_prompt or "")
        if "Current Workspace" in normalized_prompt:
            return normalized_prompt
        workspace_info = (
            "\n\n## Current Workspace\n"
            f"You are currently working in: `{self.workspace_dir.absolute()}`\n"
            "All relative paths will be resolved relative to this directory."
        )
        return normalized_prompt + workspace_info

    def set_system_prompt(self, system_prompt: str) -> None:
        """Refresh the active system prompt and keep message history aligned."""
        normalized_prompt = self._compose_system_prompt(system_prompt)
        self.system_prompt = normalized_prompt
        if self.messages and self.messages[0].role == "system":
            self.messages[0] = Message(role="system", content=normalized_prompt)
            return
        self.messages.insert(0, Message(role="system", content=normalized_prompt))

    def _runtime_context_payload(self) -> dict[str, object]:
        """Serialize the current runtime context for structured events."""
        context = self.runtime_context
        if context is None:
            return {}

        payload: dict[str, object] = {
            "session_id": context.session_id,
            "run_id": context.run_id,
            "agent_name": context.agent_name,
            "is_main_agent": context.is_main_agent,
            "depth": context.depth,
        }
        if context.parent_run_id:
            payload["parent_run_id"] = context.parent_run_id
        if context.root_run_id:
            payload["root_run_id"] = context.root_run_id
        if context.template_snapshot is not None:
            payload["template_id"] = context.template_snapshot.template_id
            payload["template_version"] = context.template_snapshot.template_version
        return payload

    @staticmethod
    def _serialize_messages(messages: list[Message]) -> list[dict[str, object]]:
        """Convert message history into JSON-serializable dictionaries."""
        serialized: list[dict[str, object]] = []
        for message in messages:
            item: dict[str, object] = {
                "role": message.role,
                "content": message.content,
            }
            if message.thinking:
                item["thinking"] = message.thinking
            if message.tool_calls:
                item["tool_calls"] = [
                    tool_call.model_dump(mode="python") for tool_call in message.tool_calls
                ]
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if message.name:
                item["name"] = message.name
            serialized.append(item)
        return serialized

    @staticmethod
    def _serialize_response(response: LLMResponse) -> dict[str, object]:
        """Convert one model response into a structured payload."""
        payload: dict[str, object] = {
            "content": response.content,
            "finish_reason": response.finish_reason,
        }
        if response.thinking:
            payload["thinking"] = response.thinking
        if response.tool_calls:
            payload["tool_calls"] = [
                tool_call.model_dump(mode="python") for tool_call in response.tool_calls
            ]
        return payload

    @staticmethod
    def _runtime_tool_event_names(function_name: str) -> tuple[str, str]:
        """Return the runtime event names for one tool invocation."""
        if function_name in {"delegate_task", "delegate_tasks"}:
            return "delegate_started", "delegate_finished"
        return "tool_started", "tool_finished"

    async def _emit_runtime_event(
        self,
        event_type: str,
        data: dict[str, object] | None = None,
    ) -> None:
        """Emit one structured runtime event to logs and the optional trace sink."""
        event = {
            "type": event_type,
            "context": self._runtime_context_payload(),
            "data": dict(data or {}),
        }
        self.logger.log_event(event_type, event)

        if self.runtime_context is None or self.runtime_hooks is None:
            return
        if self.runtime_hooks.trace_sink is None:
            return

        maybe_awaitable = self.runtime_hooks.trace_sink(self.runtime_context, event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable

    @property
    def interrupt_requested(self) -> bool:
        """Whether a stop has been requested for the current run."""
        return self._interrupt_requested

    def request_interrupt(self) -> None:
        """Signal that the current run should stop as soon as possible."""
        self._interrupt_requested = True

    def reset_interrupt(self) -> None:
        """Clear a previously requested interrupt."""
        self._interrupt_requested = False

    def _raise_if_interrupted(self) -> None:
        """Abort the current run if an interrupt was requested."""
        if self._interrupt_requested:
            raise AgentInterrupted("Agent run interrupted by user.")

    def _is_parallel_delegate_call(self, function_name: str) -> bool:
        """Whether this tool call can be executed in parallel with peers."""
        return function_name in {"delegate_task", "delegate_tasks"}

    def _get_parallel_delegate_limit(self) -> int:
        """Get delegate concurrency limit from config with safe fallback."""
        default_limit = 4
        agent_config = getattr(self.config, "agent", None)
        if agent_config is None:
            return default_limit

        raw_limit = getattr(agent_config, "parallel_delegate_limit", default_limit)
        try:
            parsed_limit = int(raw_limit)
        except (TypeError, ValueError):
            parsed_limit = default_limit
        return max(1, parsed_limit)

    def _get_tool_instance_for_call(self, function_name: str) -> Tool | None:
        """Get a tool instance for one invocation.

        Tools may optionally provide a ``clone`` method to return a per-call
        instance, which avoids shared mutable state during concurrent runs.
        """
        tool = self.tools.get(function_name)
        if tool is None:
            return None

        clone = getattr(tool, "clone", None)
        if callable(clone):
            try:
                cloned_tool = clone()
                if isinstance(cloned_tool, Tool):
                    return cloned_tool
            except Exception:
                return tool

        return tool

    def _get_approval_policy(self):
        """Return the approval policy from the bound template snapshot if available."""
        policy: ApprovalPolicy | None = None
        if self.template_snapshot is not None:
            policy = self.template_snapshot.approval_policy
        context = self.runtime_context
        if policy is None and context is not None and context.template_snapshot is not None:
            policy = context.template_snapshot.approval_policy

        extra_tools = set(self._runtime_auto_approve_tools)
        if context is not None:
            extra_tools.update(context.approval_auto_grant_tools)

        if not extra_tools:
            return policy
        if policy is None:
            return ApprovalPolicy(auto_approve_tools=sorted(extra_tools))

        merged_tools = set(policy.auto_approve_tools)
        merged_tools.update(extra_tools)
        return policy.model_copy(update={"auto_approve_tools": sorted(merged_tools)})

    def _get_workspace_policy(self):
        """Return the workspace policy from the bound template snapshot if available."""
        policy = None
        if self.template_snapshot is not None:
            policy = self.template_snapshot.workspace_policy
        context = self.runtime_context
        if policy is None and context is not None and context.template_snapshot is not None:
            policy = context.template_snapshot.workspace_policy
        return policy

    def _get_delegation_policy(self):
        """Return the delegation policy from the bound template snapshot if available."""
        policy = None
        if self.template_snapshot is not None:
            policy = self.template_snapshot.delegation_policy
        context = self.runtime_context
        if policy is None and context is not None and context.template_snapshot is not None:
            policy = context.template_snapshot.delegation_policy
        return policy

    def _apply_approval_resolution(self, resolution: dict[str, Any]) -> None:
        """Apply granted approval scope updates to the in-memory agent runtime."""
        if str(resolution.get("status", "")).strip().lower() != "granted":
            return

        identifiers = resolution.get("auto_approve_tools", [])
        if not isinstance(identifiers, list):
            return

        normalized = {
            str(item).strip()
            for item in identifiers
            if str(item).strip()
        }
        if not normalized:
            return

        self._runtime_auto_approve_tools.update(normalized)
        if self.runtime_context is not None:
            existing = {
                str(item).strip()
                for item in self.runtime_context.approval_auto_grant_tools
                if str(item).strip()
            }
            existing.update(normalized)
            self.runtime_context.approval_auto_grant_tools = sorted(existing)

    async def _maybe_refresh_prompt_after_tool(
        self,
        *,
        execution: ToolExecutionContext,
        result: ToolResult,
    ) -> None:
        """Refresh prompt memory after tools touch new workspace paths."""
        if self.runtime_context is None or self.runtime_hooks is None:
            return
        if self.runtime_hooks.prompt_refresh_hook is None:
            return
        refresh_state = self.runtime_hooks.prompt_refresh_hook(
            self.runtime_context,
            self,
            execution.runtime_finish_payload(result),
        )
        if inspect.isawaitable(refresh_state):
            await refresh_state

    def _build_tool_execution_context(
        self,
        *,
        function_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> tuple[Tool | None, ToolExecutionContext]:
        """Prepare one per-call tool instance together with normalized execution metadata."""
        tool = self._get_tool_instance_for_call(function_name)
        execution = prepare_tool_execution(
            tool=tool,
            function_name=function_name,
            tool_call_id=tool_call_id,
            arguments=arguments,
            approval_policy=self._get_approval_policy(),
            workspace_policy=self._get_workspace_policy(),
            delegation_policy=self._get_delegation_policy(),
            is_main_agent=bool(
                self.runtime_context is not None and self.runtime_context.is_main_agent
            ),
            uploaded_file_targets=(
                list(self.runtime_context.uploaded_file_targets)
                if self.runtime_context is not None
                else []
            ),
        )
        return tool, execution

    async def _prepare_tool_approval(
        self,
        *,
        execution: ToolExecutionContext,
        step: int,
    ) -> dict[str, Any] | None:
        """Create one approval request descriptor when the tool requires approval."""
        if not execution.requires_approval:
            return None
        if self.runtime_context is None or self.runtime_hooks is None:
            return None
        if self.runtime_hooks.approval_hook is None:
            return None

        request_payload = {
            "step": step,
            "tool_call_id": execution.tool_call_id,
            "tool_name": execution.tool_name,
            "tool_class": execution.tool_class,
            "arguments": dict(execution.arguments),
            "parameter_summary": execution.parameter_summary,
            "risk_category": execution.risk_category,
            "risk_level": execution.risk_level,
            "approval_reason": execution.approval_reason,
            "impact_summary": execution.impact_summary,
            "requires_approval": execution.requires_approval,
        }
        approval_state = self.runtime_hooks.approval_hook(
            self.runtime_context,
            self,
            request_payload,
        )
        if inspect.isawaitable(approval_state):
            approval_state = await approval_state
        if approval_state is None:
            return None
        if not isinstance(approval_state, dict):
            approval_state = {"status": "granted"}

        event_data = approval_state.get("event_data")
        if not isinstance(event_data, dict):
            event_data = {
                key: value
                for key, value in approval_state.items()
                if key != "waiter"
            }
        if not event_data:
            event_data = dict(request_payload)
        event_data.setdefault("tool_call_id", execution.tool_call_id)
        event_data.setdefault("tool_name", execution.tool_name)
        event_data.setdefault("parameter_summary", execution.parameter_summary)
        event_data.setdefault("risk_level", execution.risk_level)
        event_data.setdefault("approval_reason", execution.approval_reason)
        event_data.setdefault("impact_summary", execution.impact_summary)
        event_data.setdefault("status", str(approval_state.get("status", "pending")))
        return {
            "event_data": event_data,
            "waiter": approval_state.get("waiter"),
            "status": str(approval_state.get("status", "pending")),
        }

    async def _await_tool_approval_decision(
        self,
        approval_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Wait for an approval decision descriptor returned by the runtime hook."""
        waiter = approval_state.get("waiter")
        if waiter is None:
            return {"status": str(approval_state.get("status", "granted"))}

        resolution = waiter
        if inspect.isawaitable(waiter):
            resolution = await waiter
        if isinstance(resolution, dict):
            return resolution
        return {"status": str(resolution)}

    @staticmethod
    def _build_approval_denied_result(
        execution: ToolExecutionContext,
        resolution: dict[str, Any],
    ) -> ToolResult:
        """Convert a denied approval decision into a synthetic tool failure."""
        approval_id = str(resolution.get("approval_request_id", "")).strip()
        decision_notes = str(resolution.get("decision_notes", "")).strip()
        message = f"Tool execution denied by approval flow for '{execution.tool_name}'."
        if approval_id:
            message = f"{message} Approval request: {approval_id}."
        if decision_notes:
            message = f"{message} Notes: {decision_notes}"
        return ToolResult(success=False, content="", error=message)

    @staticmethod
    def _build_policy_denied_result(execution: ToolExecutionContext) -> ToolResult:
        """Convert a policy block into a synthetic tool failure result."""
        message = execution.policy_denied_reason.strip() or (
            f"Tool execution blocked by policy for '{execution.tool_name}'."
        )
        return ToolResult(success=False, content="", error=message)

    @staticmethod
    def _build_tool_exception_result(e: Exception) -> ToolResult:
        """Convert a tool execution exception into ToolResult."""
        import traceback

        error_detail = f"{type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()
        return ToolResult(
            success=False,
            content="",
            error=f"Tool execution failed: {error_detail}\n\nTraceback:\n{error_trace}",
        )

    @staticmethod
    def _attach_tool_call_id_to_event(event: dict, tool_call_id: str) -> dict:
        """Attach parent tool_call_id to streamed tool events."""
        if not isinstance(event, dict):
            return event

        wrapped_event = dict(event)
        data = wrapped_event.get("data")
        if isinstance(data, dict):
            wrapped_data = dict(data)
            wrapped_data.setdefault("tool_call_id", tool_call_id)
            wrapped_event["data"] = wrapped_data
        return wrapped_event

    @staticmethod
    def _extract_final_response_from_stream_event(stream_event: dict) -> LLMResponse | None:
        """Extract LLMResponse from a provider stream event payload."""
        if not isinstance(stream_event, dict):
            return None
        if stream_event.get("type") != "final_response":
            return None

        data = stream_event.get("data")
        if not isinstance(data, dict):
            return None

        response = data.get("response")
        if isinstance(response, LLMResponse):
            return response
        if isinstance(response, dict):
            try:
                return LLMResponse(**response)
            except Exception:
                return None
        return None

    async def _execute_tool_call_nonstream(
        self,
        *,
        tool: Tool | None,
        function_name: str,
        arguments: dict,
    ) -> ToolResult:
        """Execute one tool call without streaming sub-events."""
        if tool is None:
            return ToolResult(
                success=False,
                content="",
                error=f"Unknown tool: {function_name}",
            )

        try:
            return await tool.execute(**arguments)
        except Exception as e:
            return self._build_tool_exception_result(e)

    async def _execute_tool_call_stream(
        self,
        *,
        tool: Tool | None,
        function_name: str,
        arguments: dict,
        tool_call_id: str,
        emit_event: Callable[[dict], Awaitable[None]],
    ) -> ToolResult:
        """Execute one tool call and emit streamed sub-events if supported."""
        if tool is None:
            return ToolResult(
                success=False,
                content="",
                error=f"Unknown tool: {function_name}",
            )

        try:
            if getattr(tool, "supports_stream", False):
                async for sub_event in tool.execute_stream(**arguments):
                    await emit_event(self._attach_tool_call_id_to_event(sub_event, tool_call_id))

                final_result = getattr(tool, "final_result", None)
                if isinstance(final_result, ToolResult):
                    return final_result
                return ToolResult(
                    success=False,
                    content="",
                    error=f"Streaming tool '{function_name}' did not provide a final result.",
                )

            return await tool.execute(**arguments)
        except Exception as e:
            return self._build_tool_exception_result(e)

    @staticmethod
    def _truncate_delegate_review_text(value: Any, limit: int) -> str:
        text = " ".join(str(value or "").split())
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _should_attach_delegate_review_guidance(self, function_name: str) -> bool:
        if function_name not in {"delegate_task", "delegate_tasks"}:
            return False
        if self.runtime_context is None or not self.runtime_context.is_main_agent:
            return False
        policy = self._get_delegation_policy()
        if policy is None:
            return False
        return bool(getattr(policy, "verify_worker_output", True))

    def _format_delegate_review_tool_message(self, result: ToolResult) -> str:
        metadata = result.metadata if isinstance(result.metadata, dict) else {}
        summary = self._truncate_delegate_review_text(
            metadata.get("summary") or result.content or result.error or "",
            400,
        )
        files_changed = metadata.get("files_changed", [])
        commands_run = metadata.get("commands_run", [])
        tests_run = metadata.get("tests_run", [])
        remaining_risks = metadata.get("remaining_risks", [])
        blockers = metadata.get("blockers", [])

        def render_items(label: str, items: Any, *, limit: int = 3) -> list[str]:
            if not isinstance(items, list):
                return []
            normalized = [
                self._truncate_delegate_review_text(item, 160)
                for item in items
                if str(item or "").strip()
            ][:limit]
            if not normalized:
                return []
            return [f"{label}: {', '.join(normalized)}"]

        lines = [
            "Worker result received.",
            f"summary: {summary}" if summary else "summary: (empty)",
            *render_items("files_changed", files_changed, limit=5),
            *render_items("commands_run", commands_run),
            *render_items("tests_run", tests_run),
            *render_items("remaining_risks", remaining_risks),
            *render_items("blockers", blockers),
            "Required review before final answer:",
            "1. Decide whether the original request is already satisfied.",
            "2. Decide whether more validation is still needed.",
            "3. Decide whether another worker must be delegated to close the gap.",
            "If blockers, remaining risks, or missing verification still exist, do not end the run yet.",
        ]
        if metadata:
            lines.extend(
                [
                    "Structured worker report:",
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ]
            )
        return "\n".join(lines)

    def _append_tool_message(self, tool_call_id: str, function_name: str, result: ToolResult):
        """Append one tool result message into conversation history."""
        content = result.content if result.success else f"Error: {result.error}"
        if self._should_attach_delegate_review_guidance(function_name):
            content = self._format_delegate_review_tool_message(result)
        tool_msg = Message(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=function_name,
        )
        self.messages.append(tool_msg)

    def repair_incomplete_tool_calls(
        self,
        reason: str = "Tool execution was interrupted before completion. Do not assume it succeeded.",
    ) -> int:
        """Insert synthetic tool results for any incomplete assistant tool calls.

        OpenAI-compatible chat APIs require every assistant message with
        ``tool_calls`` to be followed by a matching ``tool`` message for each
        ``tool_call_id`` before any later user/assistant turn. Interruptions can
        leave the history in an invalid state, so this method repairs it in-place.
        """
        repaired_messages: list[Message] = []
        inserted = 0
        index = 0

        while index < len(self.messages):
            message = self.messages[index]
            repaired_messages.append(message)
            index += 1

            if message.role != "assistant" or not message.tool_calls:
                continue

            responded_ids: set[str] = set()
            while index < len(self.messages) and self.messages[index].role == "tool":
                tool_message = self.messages[index]
                repaired_messages.append(tool_message)
                if tool_message.tool_call_id:
                    responded_ids.add(tool_message.tool_call_id)
                index += 1

            for tool_call in message.tool_calls:
                if tool_call.id in responded_ids:
                    continue

                repaired_messages.append(
                    Message(
                        role="tool",
                        content=f"Error: {reason}",
                        tool_call_id=tool_call.id,
                        name=tool_call.function.name,
                    )
                )
                inserted += 1

        if inserted:
            self.messages = repaired_messages

        return inserted

    def _estimate_tokens(self) -> int:
        """Accurately calculate token count for message history using tiktoken

        Uses cl100k_base encoder (GPT-4/Claude/M2 compatible)
        """
        try:
            # Use cl100k_base encoder (used by GPT-4 and most modern models)
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            # Fallback: if tiktoken initialization fails, use simple estimation
            return self._estimate_tokens_fallback()

        total_tokens = 0

        for msg in self.messages:
            # Count text content
            if isinstance(msg.content, str):
                total_tokens += len(encoding.encode(msg.content))
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        # Convert dict to string for calculation
                        total_tokens += len(encoding.encode(str(block)))

            # Count thinking
            if msg.thinking:
                total_tokens += len(encoding.encode(msg.thinking))

            # Count tool_calls
            if msg.tool_calls:
                total_tokens += len(encoding.encode(str(msg.tool_calls)))

            # Metadata overhead per message (approximately 4 tokens)
            total_tokens += 4

        return total_tokens

    def _estimate_tokens_fallback(self) -> int:
        """Fallback token estimation method (when tiktoken is unavailable)"""
        total_chars = 0
        for msg in self.messages:
            if isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict):
                        total_chars += len(str(block))

            if msg.thinking:
                total_chars += len(msg.thinking)

            if msg.tool_calls:
                total_chars += len(str(msg.tool_calls))

        # Rough estimation: average 2.5 characters = 1 token
        return int(total_chars / 2.5)

    async def _summarize_messages(self):
        """Message history summarization: summarize conversations between user messages when tokens exceed limit

        Strategy (Agent mode):
        - Keep all user messages (these are user intents)
        - Summarize content between each user-user pair (agent execution process)
        - If last round is still executing (has agent/tool messages but no next user), also summarize
        - Structure: system -> user1 -> summary1 -> user2 -> summary2 -> user3 -> summary3 (if executing)
        """
        estimated_tokens = self._estimate_tokens()

        # If not exceeded, no summary needed
        if estimated_tokens <= self.token_limit:
            return

        print(f"\n{Colors.BRIGHT_YELLOW}馃搳 Token estimate: {estimated_tokens}/{self.token_limit}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}馃攧 Triggering message history summarization...{Colors.RESET}")

        # Find all user message indices (skip system prompt)
        user_indices = [i for i, msg in enumerate(self.messages) if msg.role == "user" and i > 0]

        # Need at least 1 user message to perform summary
        if len(user_indices) < 1:
            print(f"{Colors.BRIGHT_YELLOW}鈿狅笍  Insufficient messages, cannot summarize{Colors.RESET}")
            return

        # Build new message list
        new_messages = [self.messages[0]]  # Keep system prompt
        summary_count = 0

        # Iterate through each user message and summarize the execution process after it
        for i, user_idx in enumerate(user_indices):
            # Add current user message
            new_messages.append(self.messages[user_idx])

            # Determine message range to summarize
            # If last user, go to end of message list; otherwise to before next user
            if i < len(user_indices) - 1:
                next_user_idx = user_indices[i + 1]
            else:
                next_user_idx = len(self.messages)

            # Extract execution messages for this round
            execution_messages = self.messages[user_idx + 1 : next_user_idx]

            # If there are execution messages in this round, summarize them
            if execution_messages:
                summary_text = await self._create_summary(execution_messages, i + 1)
                if summary_text:
                    summary_message = Message(
                        role="user",
                        content=f"[Assistant Execution Summary]\n\n{summary_text}",
                    )
                    new_messages.append(summary_message)
                    summary_count += 1

        # Replace message list
        self.messages = new_messages

        new_tokens = self._estimate_tokens()
        print(f"{Colors.BRIGHT_GREEN}鉁?Summary completed, tokens reduced from {estimated_tokens} to {new_tokens}{Colors.RESET}")
        print(f"{Colors.DIM}  Structure: system + {len(user_indices)} user messages + {summary_count} summaries{Colors.RESET}")

    async def _create_summary(self, messages: list[Message], round_num: int) -> str:
        """Create summary for one execution round

        Args:
            messages: List of messages to summarize
            round_num: Round number

        Returns:
            Summary text
        """
        if not messages:
            return ""

        # Build summary content
        summary_content = f"Round {round_num} execution process:\n\n"
        for msg in messages:
            if msg.role == "assistant":
                content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"Assistant: {content_text}\n"
                if msg.tool_calls:
                    tool_names = [tc.function.name for tc in msg.tool_calls]
                    summary_content += f"  鈫?Called tools: {', '.join(tool_names)}\n"
            elif msg.role == "tool":
                result_preview = msg.content if isinstance(msg.content, str) else str(msg.content)
                summary_content += f"  鈫?Tool returned: {result_preview}...\n"

        # Call LLM to generate concise summary
        try:
            summary_prompt = f"""Please provide a concise summary of the following Agent execution process:

{summary_content}

Requirements:
1. Focus on what tasks were completed and which tools were called
2. Keep key execution results and important findings
3. Be concise and clear, within 1000 words
4. Use English
5. Do not include "user" related content, only summarize the Agent's execution process"""

            summary_msg = Message(role="user", content=summary_prompt)
            response = await self.llm.generate(
                messages=[
                    Message(
                        role="system",
                        content="You are an assistant skilled at summarizing Agent execution processes.",
                    ),
                    summary_msg,
                ]
            )

            summary_text = response.content
            print(f"{Colors.BRIGHT_GREEN}鉁?Summary for round {round_num} generated successfully{Colors.RESET}")
            return summary_text

        except Exception as e:
            print(f"{Colors.BRIGHT_RED}鉁?Summary generation failed for round {round_num}: {e}{Colors.RESET}")
            # Use simple text summary on failure
            return summary_content

    async def run(self) -> str:
        """Execute agent loop until task is complete or max steps reached."""
        # Start new run, initialize log file
        runtime_context = self.runtime_context
        self.logger.start_new_run(
            run_id=runtime_context.run_id if runtime_context is not None else None,
            agent_name=runtime_context.agent_name if runtime_context is not None else None,
        )
        await self._emit_runtime_event(
            "run_started",
            {
                "max_steps": self.max_steps,
                "message_count": len(self.messages),
                **self._startup_trace_data,
            },
        )
        print(f"{Colors.DIM}馃摑 Log file: {self.logger.get_log_file_path()}{Colors.RESET}")

        step = 0

        while step < self.max_steps:
            # Check and summarize message history to prevent context overflow
            await self._summarize_messages()

            # Step header with proper width calculation
            BOX_WIDTH = 58
            step_text = f"{Colors.BOLD}{Colors.BRIGHT_CYAN}馃挱 Step {step + 1}/{self.max_steps}{Colors.RESET}"
            step_display_width = calculate_display_width(step_text)
            padding = max(0, BOX_WIDTH - 1 - step_display_width)  # -1 for leading space

            print(f"\n{Colors.DIM}{'-' * BOX_WIDTH}{Colors.RESET}")
            print(f"{Colors.DIM}|{Colors.RESET} {step_text}{' ' * padding}{Colors.DIM}|{Colors.RESET}")
            print(f"{Colors.DIM}{'-' * BOX_WIDTH}{Colors.RESET}")

            # Get tool list for LLM call
            tool_list = list(self.tools.values())

            await self._emit_runtime_event(
                "llm_request",
                {
                    "step": step + 1,
                    "messages": self._serialize_messages(self.messages),
                    "tools": [tool.name for tool in tool_list],
                },
            )

            try:
                response = await self.llm.generate(messages=self.messages, tools=tool_list)
            except Exception as e:
                # Check if it's a retry exhausted error
                from .retry import RetryExhaustedError

                if isinstance(e, RetryExhaustedError):
                    error_msg = f"LLM call failed after {e.attempts} retries\nLast error: {str(e.last_exception)}"
                    print(f"\n{Colors.BRIGHT_RED}鉂?Retry failed:{Colors.RESET} {error_msg}")
                else:
                    error_msg = f"LLM call failed: {str(e)}"
                    print(f"\n{Colors.BRIGHT_RED}鉂?Error:{Colors.RESET} {error_msg}")
                await self._emit_runtime_event(
                    "run_failed",
                    {
                        "step": step + 1,
                        "message": error_msg,
                        "stage": "llm_call",
                    },
                )
                return error_msg

            # Log LLM response
            await self._emit_runtime_event(
                "llm_response",
                {
                    "step": step + 1,
                    **self._serialize_response(response),
                },
            )

            # Add assistant message
            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)

            # Print thinking if present
            if response.thinking:
                print(f"\n{Colors.BOLD}{Colors.MAGENTA}馃 Thinking:{Colors.RESET}")
                print(f"{Colors.DIM}{response.thinking}{Colors.RESET}")

            # Print assistant response
            if response.content:
                print(f"\n{Colors.BOLD}{Colors.BRIGHT_BLUE}馃 Assistant:{Colors.RESET}")
                print(f"{response.content}")

            # Check if task is complete (no tool calls)
            if not response.tool_calls:
                await self._emit_runtime_event(
                    "run_completed",
                    {
                        "step": step + 1,
                        "content": response.content,
                        "finish_reason": response.finish_reason,
                    },
                )
                return response.content
            tool_contexts: list[dict[str, Any]] = []
            for tool_call in response.tool_calls:
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                tool, execution = self._build_tool_execution_context(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                )

                # Tool call header
                print(f"\n{Colors.BRIGHT_YELLOW}馃敡 Tool Call:{Colors.RESET} {Colors.BOLD}{Colors.CYAN}{function_name}{Colors.RESET}")

                # Arguments (formatted display)
                print(f"{Colors.DIM}   Arguments:{Colors.RESET}")
                # Truncate each argument value to avoid overly long output
                truncated_args = {}
                for key, value in arguments.items():
                    value_str = str(value)
                    if len(value_str) > 200:
                        truncated_args[key] = value_str[:200] + "..."
                    else:
                        truncated_args[key] = value
                args_json = json.dumps(truncated_args, indent=2, ensure_ascii=False)
                for line in args_json.split("\n"):
                    print(f"   {Colors.DIM}{line}{Colors.RESET}")

                tool_contexts.append(
                    {
                        "tool": tool,
                        "execution": execution,
                        "tool_call_id": tool_call_id,
                        "function_name": function_name,
                        "arguments": arguments,
                    }
                )

            results: list[ToolResult | None] = [None] * len(tool_contexts)
            index = 0
            while index < len(tool_contexts):
                context = tool_contexts[index]
                if not self._is_parallel_delegate_call(context["function_name"]):
                    execution: ToolExecutionContext = context["execution"]
                    if not execution.policy_allowed:
                        results[index] = self._build_policy_denied_result(execution)
                        index += 1
                        continue
                    approval_state = await self._prepare_tool_approval(
                        execution=execution,
                        step=step + 1,
                    )
                    if approval_state is not None:
                        resolution = await self._await_tool_approval_decision(approval_state)
                        if str(resolution.get("status", "pending")) != "granted":
                            results[index] = self._build_approval_denied_result(
                                execution,
                                resolution,
                            )
                            index += 1
                            continue
                        self._apply_approval_resolution(resolution)

                    start_event, _ = self._runtime_tool_event_names(context["function_name"])
                    execution.mark_started()
                    await self._emit_runtime_event(
                        start_event,
                        {
                            "step": step + 1,
                            **execution.runtime_start_payload(),
                        },
                    )
                    results[index] = await self._execute_tool_call_nonstream(
                        tool=context["tool"],
                        function_name=context["function_name"],
                        arguments=context["arguments"],
                    )
                    execution.mark_finished(results[index])
                    index += 1
                    continue

                batch_start = index
                while index < len(tool_contexts) and self._is_parallel_delegate_call(
                    tool_contexts[index]["function_name"]
                ):
                    index += 1
                batch_contexts = tool_contexts[batch_start:index]
                semaphore = asyncio.Semaphore(self._get_parallel_delegate_limit())

                async def execute_batch_context(batch_context: dict) -> ToolResult:
                    start_event, _ = self._runtime_tool_event_names(batch_context["function_name"])
                    execution: ToolExecutionContext = batch_context["execution"]
                    execution.mark_started()
                    await self._emit_runtime_event(
                        start_event,
                        {
                            "step": step + 1,
                            **execution.runtime_start_payload(),
                        },
                    )
                    async with semaphore:
                        result = await self._execute_tool_call_nonstream(
                            tool=batch_context["tool"],
                            function_name=batch_context["function_name"],
                            arguments=batch_context["arguments"],
                        )
                    execution.mark_finished(result)
                    return result

                batch_tasks = [execute_batch_context(batch_context) for batch_context in batch_contexts]
                batch_results = await asyncio.gather(*batch_tasks)
                for offset, batch_result in enumerate(batch_results):
                    results[batch_start + offset] = batch_result

            for context, result in zip(tool_contexts, results):
                if result is None:
                    result = ToolResult(
                        success=False,
                        content="",
                        error="Tool execution failed unexpectedly: missing result",
                    )

                function_name = context["function_name"]
                execution: ToolExecutionContext = context["execution"]

                _, finish_event = self._runtime_tool_event_names(function_name)
                await self._emit_runtime_event(
                    finish_event,
                    {
                        "step": step + 1,
                        **execution.runtime_finish_payload(result),
                    },
                )

                # Print result
                if result.success:
                    result_text = result.content
                    if len(result_text) > 300:
                        result_text = result_text[:300] + f"{Colors.DIM}...{Colors.RESET}"
                    print(f"{Colors.BRIGHT_GREEN}鉁?Result:{Colors.RESET} {result_text}")
                else:
                    print(f"{Colors.BRIGHT_RED}鉁?Error:{Colors.RESET} {Colors.RED}{result.error}{Colors.RESET}")

                self._append_tool_message(execution.tool_call_id, function_name, result)
                await self._maybe_refresh_prompt_after_tool(execution=execution, result=result)

            step += 1

        # Max steps reached
        error_msg = f"Task couldn't be completed after {self.max_steps} steps."
        print(f"\n{Colors.BRIGHT_YELLOW}鈿狅笍  {error_msg}{Colors.RESET}")
        await self._emit_runtime_event(
            "run_failed",
            {
                "step": step,
                "message": error_msg,
                "stage": "max_steps",
            },
        )
        return error_msg

    async def run_stream(self):
        """Execute agent loop, yielding structured events for each step.

        Yields dicts with keys:
            type: "thinking" | "content" | "tool_call" | "tool_result" | "done" | "error" | "step"
            data: event-specific payload
        """
        runtime_context = self.runtime_context
        self.logger.start_new_run(
            run_id=runtime_context.run_id if runtime_context is not None else None,
            agent_name=runtime_context.agent_name if runtime_context is not None else None,
        )
        await self._emit_runtime_event(
            "run_started",
            {
                "max_steps": self.max_steps,
                "message_count": len(self.messages),
            },
        )
        self.repair_incomplete_tool_calls()
        step = 0

        while step < self.max_steps:
            self._raise_if_interrupted()
            await self._summarize_messages()
            self._raise_if_interrupted()

            yield {"type": "step", "data": {"current": step + 1, "max": self.max_steps}}

            tool_list = list(self.tools.values())
            await self._emit_runtime_event(
                "llm_request",
                {
                    "step": step + 1,
                    "messages": self._serialize_messages(self.messages),
                    "tools": [tool.name for tool in tool_list],
                },
            )

            response: LLMResponse | None = None
            streamed_delta_emitted = False
            stream_error: Exception | None = None

            try:
                generate_stream = getattr(self.llm, "generate_stream", None)
                if callable(generate_stream):
                    try:
                        stream_iterator = generate_stream(messages=self.messages, tools=tool_list)
                        if not hasattr(stream_iterator, "__aiter__"):
                            if inspect.iscoroutine(stream_iterator):
                                stream_iterator.close()
                            raise NotImplementedError("LLM generate_stream is not an async iterator.")

                        async for stream_event in stream_iterator:
                            self._raise_if_interrupted()
                            final_response = self._extract_final_response_from_stream_event(stream_event)
                            if final_response is not None:
                                response = final_response
                                continue

                            if not isinstance(stream_event, dict):
                                continue
                            event_type = stream_event.get("type")
                            data = stream_event.get("data")
                            if not isinstance(data, dict):
                                data = {}

                            if event_type == "thinking_delta":
                                delta = data.get("delta")
                                if isinstance(delta, str) and delta:
                                    streamed_delta_emitted = True
                                    yield {"type": "thinking_delta", "data": {"delta": delta}}
                            elif event_type == "content_delta":
                                delta = data.get("delta")
                                if isinstance(delta, str) and delta:
                                    streamed_delta_emitted = True
                                    yield {"type": "content_delta", "data": {"delta": delta}}
                    except NotImplementedError:
                        response = None
                    except AgentInterrupted:
                        raise
                    except Exception as e:
                        # If incremental events already reached frontend, abort this turn
                        # to avoid duplicate/misaligned content from a fallback retry.
                        if streamed_delta_emitted:
                            await self._emit_runtime_event(
                                "run_failed",
                                {
                                    "step": step + 1,
                                    "message": f"LLM streaming failed: {e}",
                                    "stage": "llm_stream",
                                },
                            )
                            yield {"type": "error", "data": {"message": f"LLM streaming failed: {e}"}}
                            return
                        stream_error = e

                if response is None:
                    self._raise_if_interrupted()
                    response = await self.llm.generate(messages=self.messages, tools=tool_list)
            except AgentInterrupted:
                raise
            except Exception as e:
                from .retry import RetryExhaustedError

                actual_error = stream_error or e
                if isinstance(actual_error, RetryExhaustedError):
                    error_msg = (
                        f"LLM call failed after {actual_error.attempts} retries: "
                        f"{actual_error.last_exception}"
                    )
                else:
                    error_msg = f"LLM call failed: {actual_error}"
                await self._emit_runtime_event(
                    "run_failed",
                    {
                        "step": step + 1,
                        "message": error_msg,
                        "stage": "llm_call",
                    },
                )
                yield {"type": "error", "data": {"message": error_msg}}
                return

            if response is None:
                await self._emit_runtime_event(
                    "run_failed",
                    {
                        "step": step + 1,
                        "message": "LLM stream ended without final response.",
                        "stage": "llm_stream",
                    },
                )
                yield {"type": "error", "data": {"message": "LLM stream ended without final response."}}
                return

            self._raise_if_interrupted()
            await self._emit_runtime_event(
                "llm_response",
                {
                    "step": step + 1,
                    **self._serialize_response(response),
                },
            )

            assistant_msg = Message(
                role="assistant",
                content=response.content,
                thinking=response.thinking,
                tool_calls=response.tool_calls,
            )
            self.messages.append(assistant_msg)

            if response.thinking:
                yield {"type": "thinking", "data": {"content": response.thinking}}

            if response.content:
                yield {"type": "content", "data": {"content": response.content}}

            if not response.tool_calls:
                await self._emit_runtime_event(
                    "run_completed",
                    {
                        "step": step + 1,
                        "content": response.content,
                        "finish_reason": response.finish_reason,
                    },
                )
                yield {"type": "done", "data": {"content": response.content}}
                return

            tool_contexts: list[dict[str, Any]] = []
            for tool_call in response.tool_calls:
                self._raise_if_interrupted()
                tool_call_id = tool_call.id
                function_name = tool_call.function.name
                arguments = tool_call.function.arguments
                tool, execution = self._build_tool_execution_context(
                    function_name=function_name,
                    tool_call_id=tool_call_id,
                    arguments=arguments,
                )

                yield {
                    "type": "tool_call",
                    "data": execution.tool_call_payload(),
                }
                tool_contexts.append(
                    {
                        "tool": tool,
                        "execution": execution,
                        "tool_call_id": tool_call_id,
                        "function_name": function_name,
                        "arguments": arguments,
                    }
                )

            index = 0
            while index < len(tool_contexts):
                self._raise_if_interrupted()
                context = tool_contexts[index]
                if not self._is_parallel_delegate_call(context["function_name"]):
                    execution: ToolExecutionContext = context["execution"]
                    if not execution.policy_allowed:
                        result = self._build_policy_denied_result(execution)
                        yield {
                            "type": "tool_result",
                            "data": execution.tool_result_payload(result),
                        }
                        self._append_tool_message(
                            execution.tool_call_id,
                            context["function_name"],
                            result,
                        )
                        await self._maybe_refresh_prompt_after_tool(
                            execution=execution,
                            result=result,
                        )
                        index += 1
                        continue
                    approval_state = await self._prepare_tool_approval(
                        execution=execution,
                        step=step + 1,
                    )
                    if approval_state is not None:
                        yield {
                            "type": "approval_requested",
                            "data": approval_state["event_data"],
                        }
                        resolution = await self._await_tool_approval_decision(approval_state)
                        if str(resolution.get("status", "pending")) != "granted":
                            result = self._build_approval_denied_result(
                                execution,
                                resolution,
                            )
                            yield {
                                "type": "tool_result",
                                "data": execution.tool_result_payload(result),
                            }
                            self._append_tool_message(
                                execution.tool_call_id,
                                context["function_name"],
                                result,
                            )
                            await self._maybe_refresh_prompt_after_tool(
                                execution=execution,
                                result=result,
                            )
                            index += 1
                            continue
                        self._apply_approval_resolution(resolution)

                    start_event, finish_event = self._runtime_tool_event_names(
                        context["function_name"]
                    )
                    execution.mark_started()
                    await self._emit_runtime_event(
                        start_event,
                        {
                            "step": step + 1,
                            **execution.runtime_start_payload(),
                        },
                    )
                    result = await self._execute_tool_call_nonstream(
                        tool=context["tool"],
                        function_name=context["function_name"],
                        arguments=context["arguments"],
                    )
                    execution.mark_finished(result)
                    await self._emit_runtime_event(
                        finish_event,
                        {
                            "step": step + 1,
                            **execution.runtime_finish_payload(result),
                        },
                    )
                    yield {
                        "type": "tool_result",
                        "data": execution.tool_result_payload(result),
                    }
                    self._append_tool_message(
                        execution.tool_call_id,
                        context["function_name"],
                        result,
                    )
                    await self._maybe_refresh_prompt_after_tool(
                        execution=execution,
                        result=result,
                    )
                    index += 1
                    continue

                batch_start = index
                while index < len(tool_contexts) and self._is_parallel_delegate_call(
                    tool_contexts[index]["function_name"]
                ):
                    index += 1
                batch_contexts = tool_contexts[batch_start:index]
                queue: asyncio.Queue[tuple[str, int, dict | ToolResult]] = asyncio.Queue()
                batch_results: list[ToolResult | None] = [None] * len(batch_contexts)
                semaphore = asyncio.Semaphore(self._get_parallel_delegate_limit())

                async def run_one(batch_index: int, batch_context: dict):
                    async def emit_sub_event(event: dict):
                        await queue.put(("event", batch_index, event))

                    start_event, finish_event = self._runtime_tool_event_names(
                        batch_context["function_name"]
                    )
                    execution: ToolExecutionContext = batch_context["execution"]
                    execution.mark_started()
                    await self._emit_runtime_event(
                        start_event,
                        {
                            "step": step + 1,
                            **execution.runtime_start_payload(),
                        },
                    )
                    async with semaphore:
                        result = await self._execute_tool_call_stream(
                            tool=batch_context["tool"],
                            function_name=batch_context["function_name"],
                            arguments=batch_context["arguments"],
                            tool_call_id=batch_context["tool_call_id"],
                            emit_event=emit_sub_event,
                        )
                    execution.mark_finished(result)
                    await self._emit_runtime_event(
                        finish_event,
                        {
                            "step": step + 1,
                            **execution.runtime_finish_payload(result),
                        },
                    )
                    await queue.put(("done", batch_index, result))

                tasks = [
                    asyncio.create_task(run_one(batch_index, batch_context))
                    for batch_index, batch_context in enumerate(batch_contexts)
                ]

                try:
                    remaining = len(tasks)
                    while remaining > 0:
                        self._raise_if_interrupted()
                        item_type, batch_index, payload = await queue.get()
                        if item_type == "event":
                            yield payload
                            continue

                        if not isinstance(payload, ToolResult):
                            result = ToolResult(
                                success=False,
                                content="",
                                error="Tool execution failed unexpectedly: invalid result payload",
                            )
                        else:
                            result = payload
                        batch_context = batch_contexts[batch_index]
                        execution: ToolExecutionContext = batch_context["execution"]
                        yield {
                            "type": "tool_result",
                            "data": execution.tool_result_payload(result),
                        }
                        batch_results[batch_index] = result
                        remaining -= 1
                except (asyncio.CancelledError, AgentInterrupted):
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    raise

                await asyncio.gather(*tasks)
                for batch_index, batch_context in enumerate(batch_contexts):
                    self._raise_if_interrupted()
                    result = batch_results[batch_index]
                    if result is None:
                        result = ToolResult(
                            success=False,
                            content="",
                            error="Tool execution failed unexpectedly: missing result",
                        )
                    self._append_tool_message(
                        batch_context["tool_call_id"],
                        batch_context["function_name"],
                        result,
                    )
                    await self._maybe_refresh_prompt_after_tool(
                        execution=batch_context["execution"],
                        result=result,
                    )

            step += 1

        yield {
            "type": "error",
            "data": {"message": f"Task couldn't be completed after {self.max_steps} steps."},
        }
        await self._emit_runtime_event(
            "run_failed",
            {
                "step": step,
                "message": f"Task couldn't be completed after {self.max_steps} steps.",
                "stage": "max_steps",
            },
        )

    def get_history(self) -> list[Message]:
        """Get message history."""
        return self.messages.copy()

