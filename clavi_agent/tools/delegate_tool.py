"""Tools for delegating tasks to dynamic sub-agents (Supervisor-Worker pattern)."""

import asyncio
from typing import Any, AsyncGenerator, Callable

from .base import Tool, ToolResult


_STRUCTURED_TASK_TYPE_ENUM = ["execution", "exploration"]


def _normalize_optional_text(value: Any) -> str:
    """将可选文本字段规整为去首尾空白后的字符串。"""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_text_list(value: Any) -> list[str]:
    """将字符串或字符串列表规整为紧凑的文本条目列表。"""
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if not isinstance(value, list):
        return []

    normalized_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if normalized:
            normalized_items.append(normalized)
    return normalized_items


def _append_contract_list(lines: list[str], title: str, values: list[str]) -> None:
    """在存在内容时追加一个列表型任务合同分节。"""
    if not values:
        return
    lines.append(f"{title}:")
    for value in values:
        lines.append(f"- {value}")


def _dedupe_texts(values: list[str]) -> list[str]:
    """保持顺序地去重文本列表。"""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _normalize_path_text(value: Any) -> str:
    """统一路径分隔符，减少同一路径的重复记录。"""
    return str(value or "").strip().replace("\\", "/")


def _looks_like_test_command(command: str) -> bool:
    """粗略识别测试或验证命令，供 delegate 结果摘要使用。"""
    normalized = str(command or "").strip().lower()
    if not normalized:
        return False
    test_markers = (
        "pytest",
        "unittest",
        "nose",
        "vitest",
        "jest",
        "npm test",
        "pnpm test",
        "yarn test",
        "go test",
        "cargo test",
        "mvn test",
        "gradle test",
        "ctest",
    )
    return any(marker in normalized for marker in test_markers)


def _empty_worker_report(worker_index: int | None = None) -> dict[str, Any]:
    """初始化单个 worker 的结构化执行摘要。"""
    report: dict[str, Any] = {
        "summary": "",
        "files_changed": [],
        "commands_run": [],
        "tests_run": [],
        "remaining_risks": [],
        "blockers": [],
    }
    if worker_index is not None:
        report["worker_index"] = worker_index
    return report


def _update_worker_report(report: dict[str, Any], event: dict[str, Any]) -> None:
    """从 sub-agent 事件中提取结构化执行摘要。"""
    event_type = str(event.get("type", "")).strip()
    data = event.get("data", {})
    if not isinstance(data, dict):
        data = {}

    if event_type == "tool_call":
        if str(data.get("name", "")).strip() == "bash":
            arguments = data.get("arguments", {})
            if isinstance(arguments, dict):
                command = str(arguments.get("command", "")).strip()
                if command:
                    report["commands_run"].append(command)
                    if _looks_like_test_command(command):
                        report["tests_run"].append(command)
        return

    if event_type == "tool_result":
        touched_paths = data.get("touched_paths", [])
        if isinstance(touched_paths, list):
            report["files_changed"].extend(
                _normalize_path_text(item)
                for item in touched_paths
                if _normalize_path_text(item)
            )
        artifacts = data.get("artifacts", [])
        if isinstance(artifacts, list):
            report["files_changed"].extend(
                _normalize_path_text(item.get("uri", ""))
                for item in artifacts
                if isinstance(item, dict) and _normalize_path_text(item.get("uri", ""))
            )
        if not bool(data.get("success")):
            error_text = str(data.get("error", "")).strip()
            if error_text:
                report["blockers"].append(error_text)
        return

    if event_type == "done":
        report["summary"] = str(data.get("content", "")).strip()
        return

    if event_type == "error":
        message = str(data.get("message", "")).strip()
        if message:
            report["blockers"].append(message)
            if not str(report.get("summary", "")).strip():
                report["summary"] = message


def _finalize_worker_report(report: dict[str, Any]) -> dict[str, Any]:
    """收敛并规整单个 worker 的结构化执行摘要。"""
    normalized = dict(report)
    for key in ("files_changed", "commands_run", "tests_run", "remaining_risks", "blockers"):
        value = normalized.get(key, [])
        normalized[key] = _dedupe_texts(value if isinstance(value, list) else [])
    if not normalized["remaining_risks"] and normalized["blockers"]:
        normalized["remaining_risks"] = list(normalized["blockers"])
    normalized["summary"] = str(normalized.get("summary", "")).strip()
    return normalized


def _merge_worker_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合多个 worker 的结构化结果，生成上层 delegate 摘要。"""
    combined = _empty_worker_report()
    worker_summaries: list[str] = []
    normalized_workers: list[dict[str, Any]] = []
    for report in reports:
        normalized = _finalize_worker_report(report)
        normalized_workers.append(normalized)
        summary = str(normalized.get("summary", "")).strip()
        if summary:
            worker_summaries.append(summary)
        for key in ("files_changed", "commands_run", "tests_run", "remaining_risks", "blockers"):
            combined[key].extend(normalized.get(key, []))

    combined = _finalize_worker_report(combined)
    combined["workers"] = normalized_workers
    combined["summary"] = " | ".join(worker_summaries[:3]) if worker_summaries else ""
    combined["worker_count"] = len(normalized_workers)
    return combined


def _build_task_contract(arguments: dict[str, Any]) -> str:
    """将结构化 worker 合同字段渲染为委派任务正文。"""
    task = _normalize_optional_text(arguments.get("task"))
    task_type = _normalize_optional_text(arguments.get("task_type")).lower()
    if task_type not in _STRUCTURED_TASK_TYPE_ENUM:
        task_type = ""

    goal = _normalize_optional_text(arguments.get("goal"))
    scope = _normalize_optional_text(arguments.get("scope"))
    files_in_scope = _normalize_text_list(arguments.get("files_in_scope"))
    expected_changes = _normalize_text_list(arguments.get("expected_changes"))
    acceptance_criteria = _normalize_text_list(arguments.get("acceptance_criteria"))
    depends_on = _normalize_text_list(arguments.get("depends_on"))
    expected_outputs = _normalize_text_list(arguments.get("expected_outputs"))
    uncertainties = _normalize_text_list(arguments.get("uncertainties"))

    has_structured_contract = any(
        [
            task_type,
            goal,
            scope,
            files_in_scope,
            expected_changes,
            acceptance_criteria,
            depends_on,
            expected_outputs,
            uncertainties,
        ]
    )
    if not has_structured_contract:
        return task

    lines: list[str] = []
    if task:
        lines.extend(["Primary Task:", task, ""])

    lines.append("Structured Task Contract:")
    if task_type:
        lines.append(f"Task type: {task_type}")
    if goal:
        lines.append(f"Goal: {goal}")
    if scope:
        lines.append(f"Scope: {scope}")
    _append_contract_list(lines, "Files in scope", files_in_scope)
    _append_contract_list(lines, "Expected changes", expected_changes)
    _append_contract_list(lines, "Dependencies", depends_on)
    _append_contract_list(lines, "Expected outputs", expected_outputs)
    _append_contract_list(lines, "Acceptance criteria", acceptance_criteria)
    _append_contract_list(lines, "Open questions or boundaries", uncertainties)
    if task_type == "exploration" and not expected_changes:
        lines.append(
            "Exploration guidance: prioritize investigation, findings, and recommended next steps over code changes unless they are clearly required."
        )

    return "\n".join(lines).strip()


class DelegateTool(Tool):
    """Tool for spawning a sub-agent to handle one complex isolatable task."""

    def __init__(
        self,
        agent_factory: Callable[[str, int], Any],
        delegate_executor: Callable[[str, str, int], AsyncGenerator[dict, None]] | None = None,
    ):
        """Initialize with a factory function to create new Agent instances.

        Args:
            agent_factory: A callable that takes (system_prompt, max_steps) and returns an Agent.
        """
        self._agent_factory = agent_factory
        self._delegate_executor = delegate_executor
        self.final_result: ToolResult | None = None

    def clone(self) -> "DelegateTool":
        """Return a per-call tool instance to support concurrent executions safely."""
        return DelegateTool(
            agent_factory=self._agent_factory,
            delegate_executor=self._delegate_executor,
        )

    @property
    def name(self) -> str:
        return "delegate_task"

    @property
    def description(self) -> str:
        return (
            "Dynamically spawns a specialized sub-agent (worker) to execute delegated work. "
            "This is one of the main agent's default execution paths for implementation, verification, "
            "and other isolatable tasks that can be handed off cleanly. "
            "Provide a concrete persona and task contract so the worker can execute without extra clarification. "
            "You can pass structured contract fields such as goal, scope, files_in_scope, expected_changes, "
            "acceptance_criteria, depends_on, expected_outputs, uncertainties, and task_type to make the handoff precise. "
            "When coordinating multiple workers, publish requirements and decisions with share_context so "
            "other agents can read them via read_shared_context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "persona": {
                    "type": "string",
                    "description": "The custom system prompt defining the persona, expertise, and rules for the sub-agent (e.g., 'You are a senior React developer. Your task is to...').",
                },
                "task": {
                    "type": "string",
                    "description": "The specific, actionable task instruction for the sub-agent to execute. When structured contract fields are also present, this should be the concise primary task summary.",
                },
                "task_type": {
                    "type": "string",
                    "enum": _STRUCTURED_TASK_TYPE_ENUM,
                    "description": "Optional task contract type. Use execution for implementation/validation work and exploration for investigation/reporting work.",
                },
                "goal": {
                    "type": "string",
                    "description": "Optional explicit goal the worker must achieve.",
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope and boundaries for the worker.",
                },
                "files_in_scope": {
                    "type": "array",
                    "description": "Optional list of files or directories the worker should inspect or modify.",
                    "items": {"type": "string"},
                },
                "expected_changes": {
                    "type": "array",
                    "description": "Optional list of expected code or behavior changes. For exploration tasks this can be omitted.",
                    "items": {"type": "string"},
                },
                "acceptance_criteria": {
                    "type": ["array", "string"],
                    "description": "Optional acceptance criteria the worker result must satisfy.",
                    "items": {"type": "string"},
                },
                "depends_on": {
                    "type": ["array", "string"],
                    "description": "Optional upstream context, related tasks, or prerequisites the worker should account for.",
                    "items": {"type": "string"},
                },
                "expected_outputs": {
                    "type": ["array", "string"],
                    "description": "Optional expected artifacts, summaries, or verification outputs the worker should return.",
                    "items": {"type": "string"},
                },
                "uncertainties": {
                    "type": ["array", "string"],
                    "description": "Optional unknowns, risks, or boundaries the worker should surface rather than guess.",
                    "items": {"type": "string"},
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum allowed steps for the sub-agent before it forcefully stops. Keep it reasonable (default 20).",
                },
            },
            "required": ["persona", "task"],
        }

    @property
    def supports_stream(self) -> bool:
        """Indicate that this tool supports streaming its execution."""
        return True

    async def execute_stream(
        self,
        persona: str,
        task: str,
        max_steps: int = 20,
        **contract_fields: Any,
    ) -> AsyncGenerator[dict, None]:
        """Execute the sub-agent and stream its events back to the caller."""
        self.final_result = None
        worker_report = _empty_worker_report(worker_index=0)
        try:
            result_text = ""
            delegated_task = _build_task_contract({"task": task, **contract_fields})
            if self._delegate_executor is not None:
                event_stream = self._delegate_executor(persona, delegated_task, max_steps)
            else:
                sub_agent = self._agent_factory(persona, max_steps)
                sub_agent.add_user_message(delegated_task)
                event_stream = sub_agent.run_stream()

            async for event in event_stream:
                _update_worker_report(worker_report, event)
                if event["type"] in ["thinking", "content", "tool_call", "tool_result", "step", "error"]:
                    yield {"type": "sub_task", "data": {"worker_index": 0, "event": event}}

                if event["type"] == "done":
                    result_text = event["data"]["content"]
                elif event["type"] == "error":
                    result_text = f"Error: {event['data']['message']}"

            delegate_metadata = _merge_worker_reports([worker_report])
            if result_text and not str(delegate_metadata.get("summary", "")).strip():
                delegate_metadata["summary"] = result_text
            self.final_result = ToolResult(
                success=True,
                content=f"Sub-agent completed the task. Final result/summary:\n\n{result_text}",
                metadata=delegate_metadata,
            )
        except Exception as e:
            worker_report["blockers"].append(str(e))
            self.final_result = ToolResult(
                success=False,
                error=f"Sub-agent execution failed: {str(e)}",
                metadata=_merge_worker_reports([worker_report]),
            )

    async def execute(self, *args, **kwargs) -> ToolResult:
        """Fallback for non-streaming execution."""
        async for _ in self.execute_stream(*args, **kwargs):
            pass
        return self.final_result or ToolResult(
            success=False,
            content="",
            error="Sub-agent execution produced no result.",
        )


class DelegateBatchTool(Tool):
    """Tool for spawning multiple sub-agents concurrently in one call."""

    def __init__(
        self,
        agent_factory: Callable[[str, int], Any],
        max_parallel: int = 4,
        delegate_executor: Callable[[str, str, int], AsyncGenerator[dict, None]] | None = None,
    ):
        self._agent_factory = agent_factory
        self._max_parallel = max(1, int(max_parallel))
        self._delegate_executor = delegate_executor
        self.final_result: ToolResult | None = None

    def clone(self) -> "DelegateBatchTool":
        return DelegateBatchTool(
            agent_factory=self._agent_factory,
            max_parallel=self._max_parallel,
            delegate_executor=self._delegate_executor,
        )

    @property
    def name(self) -> str:
        return "delegate_tasks"

    @property
    def description(self) -> str:
        return (
            "Spawn multiple specialized sub-agents in parallel for independent subtasks. "
            "Prefer this when the main agent can hand off several independent worker tasks at once and merge their results later."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workers": {
                    "type": "array",
                    "description": "List of workers to launch in parallel.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "persona": {
                                "type": "string",
                                "description": "Worker system prompt/persona.",
                            },
                            "task": {
                                "type": "string",
                                "description": "Worker task instruction. When structured contract fields are present, this should stay as the concise primary task summary.",
                            },
                            "task_type": {
                                "type": "string",
                                "enum": _STRUCTURED_TASK_TYPE_ENUM,
                                "description": "Optional task contract type for this worker.",
                            },
                            "goal": {
                                "type": "string",
                                "description": "Optional explicit goal for this worker.",
                            },
                            "scope": {
                                "type": "string",
                                "description": "Optional scope and boundaries for this worker.",
                            },
                            "files_in_scope": {
                                "type": "array",
                                "description": "Optional list of files or directories relevant to this worker.",
                                "items": {"type": "string"},
                            },
                            "expected_changes": {
                                "type": "array",
                                "description": "Optional list of expected changes for this worker. Omit for pure exploration tasks.",
                                "items": {"type": "string"},
                            },
                            "acceptance_criteria": {
                                "type": ["array", "string"],
                                "description": "Optional acceptance criteria for this worker.",
                                "items": {"type": "string"},
                            },
                            "depends_on": {
                                "type": ["array", "string"],
                                "description": "Optional prerequisites or related context for this worker.",
                                "items": {"type": "string"},
                            },
                            "expected_outputs": {
                                "type": ["array", "string"],
                                "description": "Optional expected outputs for this worker.",
                                "items": {"type": "string"},
                            },
                            "uncertainties": {
                                "type": ["array", "string"],
                                "description": "Optional unknowns or boundaries for this worker.",
                                "items": {"type": "string"},
                            },
                            "max_steps": {
                                "type": "integer",
                                "description": "Optional max steps for this worker (default 20).",
                            },
                        },
                        "required": ["persona", "task"],
                    },
                    "minItems": 1,
                }
            },
            "required": ["workers"],
        }

    @property
    def supports_stream(self) -> bool:
        return True

    async def execute_stream(self, workers: list[dict[str, Any]]) -> AsyncGenerator[dict, None]:
        self.final_result = None
        if not workers:
            self.final_result = ToolResult(
                success=False,
                content="",
                error="delegate_tasks requires at least one worker item.",
            )
            return

        queue: asyncio.Queue[tuple[str, int, dict | ToolResult]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(self._max_parallel)
        worker_results: list[ToolResult | None] = [None] * len(workers)
        worker_reports: list[dict[str, Any]] = [
            _empty_worker_report(worker_index=worker_index)
            for worker_index in range(len(workers))
        ]

        async def run_one(worker_index: int, worker: dict[str, Any]) -> None:
            persona = worker.get("persona")
            task = worker.get("task")
            if not isinstance(persona, str) or not isinstance(task, str):
                await queue.put(
                    (
                        "done",
                        worker_index,
                        ToolResult(
                            success=False,
                            content="",
                            error=f"Worker {worker_index} is missing valid persona/task strings.",
                        ),
                    )
                )
                return

            max_steps_raw = worker.get("max_steps", 20)
            try:
                max_steps = int(max_steps_raw)
            except (TypeError, ValueError):
                max_steps = 20
            max_steps = max(1, max_steps)

            async with semaphore:
                try:
                    result_text = ""
                    delegated_task = _build_task_contract(worker)
                    if self._delegate_executor is not None:
                        event_stream = self._delegate_executor(persona, delegated_task, max_steps)
                    else:
                        sub_agent = self._agent_factory(persona, max_steps)
                        sub_agent.add_user_message(delegated_task)
                        event_stream = sub_agent.run_stream()

                    async for event in event_stream:
                        _update_worker_report(worker_reports[worker_index], event)
                        await queue.put(("event", worker_index, event))
                        if event["type"] == "done":
                            result_text = event["data"]["content"]
                        elif event["type"] == "error":
                            result_text = f"Error: {event['data']['message']}"

                    delegate_metadata = _merge_worker_reports([worker_reports[worker_index]])
                    if result_text and not str(delegate_metadata.get("summary", "")).strip():
                        delegate_metadata["summary"] = result_text
                    await queue.put(
                        (
                            "done",
                            worker_index,
                            ToolResult(
                                success=True,
                                content=result_text,
                                metadata=delegate_metadata,
                            ),
                        )
                    )
                except Exception as e:
                    worker_reports[worker_index]["blockers"].append(str(e))
                    await queue.put(
                        (
                            "done",
                            worker_index,
                            ToolResult(
                                success=False,
                                content="",
                                error=f"Worker {worker_index} failed: {str(e)}",
                                metadata=_merge_worker_reports([worker_reports[worker_index]]),
                            ),
                        )
                    )

        tasks = [
            asyncio.create_task(run_one(worker_index, worker))
            for worker_index, worker in enumerate(workers)
        ]

        try:
            remaining = len(tasks)
            while remaining > 0:
                item_type, worker_index, payload = await queue.get()
                if item_type == "event":
                    if isinstance(payload, dict):
                        yield {
                            "type": "sub_task",
                            "data": {
                                "worker_index": worker_index,
                                "event": payload,
                            },
                        }
                    continue

                if isinstance(payload, ToolResult):
                    worker_results[worker_index] = payload
                else:
                    worker_results[worker_index] = ToolResult(
                        success=False,
                        content="",
                        error=f"Worker {worker_index} returned an invalid result payload.",
                    )
                remaining -= 1
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        await asyncio.gather(*tasks)

        failures: list[str] = []
        summary_lines = ["Parallel sub-agent execution completed:"]
        completed_reports: list[dict[str, Any]] = []

        for worker_index, result in enumerate(worker_results):
            if result is None:
                failure = f"worker[{worker_index}] produced no result."
                failures.append(failure)
                summary_lines.append(f"- worker[{worker_index}]: {failure}")
                continue

            metadata = result.metadata if isinstance(result.metadata, dict) else {}
            if metadata:
                completed_reports.append(
                    {
                        **metadata,
                        "worker_index": metadata.get("worker_index", worker_index),
                    }
                )
            if result.success:
                summary_lines.append(f"- worker[{worker_index}] succeeded: {result.content}")
            else:
                error_text = result.error or "unknown error"
                failures.append(f"worker[{worker_index}] failed: {error_text}")
                summary_lines.append(f"- worker[{worker_index}] failed: {error_text}")

        aggregate_metadata = _merge_worker_reports(completed_reports or worker_reports)
        aggregate_metadata["summary"] = "\n".join(summary_lines[1:4]).strip()
        if failures:
            self.final_result = ToolResult(
                success=False,
                content="",
                error="\n".join(summary_lines),
                metadata=aggregate_metadata,
            )
        else:
            self.final_result = ToolResult(
                success=True,
                content="\n".join(summary_lines),
                metadata=aggregate_metadata,
            )

    async def execute(self, *args, **kwargs) -> ToolResult:
        async for _ in self.execute_stream(*args, **kwargs):
            pass
        return self.final_result or ToolResult(
            success=False,
            content="",
            error="delegate_tasks produced no result.",
        )
