"""RunManager integration tests through SessionManager."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from clavi_agent.agent import Agent
from clavi_agent.account_constants import ROOT_ACCOUNT_ID
from clavi_agent.config import AgentConfig, Config, FeatureFlagsConfig, LLMConfig, RetryConfig, ToolsConfig
from clavi_agent.run_models import RunCheckpointPayload, RunCheckpointRecord, RunRecord, RunStepRecord
from clavi_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from clavi_agent.session import SessionManager
from clavi_agent.tools.base import Tool, ToolResult
from clavi_agent.tools.bash_tool import BashOutputResult
from clavi_agent.tools.skill_tool import create_skill_tools
from clavi_agent.upload_models import UploadCreatePayload
from clavi_agent.user_memory_store import UserMemoryStore


def build_config(
    tmp_path: Path,
    *,
    max_concurrent_runs: int = 4,
    run_timeout_seconds: int | None = None,
    enable_file_tools: bool = False,
    enable_skills: bool = False,
    enable_learned_workflow_generation: bool = True,
) -> Config:
    return Config(
        llm=LLMConfig(
            api_key="test-key",
            api_base="https://example.com",
            model="test-model",
            provider="openai",
            retry=RetryConfig(enabled=False),
        ),
        agent=AgentConfig(
            max_steps=5,
            max_concurrent_runs=max_concurrent_runs,
            run_timeout_seconds=run_timeout_seconds,
            workspace_dir=str(tmp_path / "workspace"),
            system_prompt_path="system_prompt.md",
            log_dir=str(tmp_path / "logs"),
            session_store_path=str(tmp_path / "sessions.db"),
            agent_store_path=str(tmp_path / "agents.db"),
        ),
        tools=ToolsConfig(
            enable_file_tools=enable_file_tools,
            enable_bash=False,
            enable_note=False,
            enable_skills=enable_skills,
            enable_mcp=False,
        ),
        feature_flags=FeatureFlagsConfig(
            enable_learned_workflow_generation=enable_learned_workflow_generation,
        ),
    )


async def collect_events(generator):
    events = []
    async for event in generator:
        events.append(event)
    return events


def create_account_with_api_config(manager: SessionManager, username: str, display_name: str) -> dict:
    account = manager._account_store.create_account(
        username=username,
        password="Secret123!",
        display_name=display_name,
    )
    manager.save_account_api_config(
        account["id"],
        name=f"{display_name} Config",
        api_key="test-key",
        provider="openai",
        api_base="https://example.com",
        model="test-model",
        reasoning_enabled=False,
        activate=True,
    )
    return account


@patch("clavi_agent.session.LLMClient")
def test_run_started_trace_exposes_resolved_llm_routing(mock_llm_class, tmp_path: Path):
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    async def scenario() -> dict:
        config = build_config(tmp_path)
        config.llm.worker_profile = {"model": "global-worker-default", "reasoning_enabled": False}
        manager = SessionManager(config=config)
        await manager.initialize()

        account = create_account_with_api_config(manager, "trace-router", "Trace Router")
        manager.save_account_api_config(
            account["id"],
            name="Trace Router Config",
            api_key="test-key",
            provider="openai",
            api_base="https://example.com",
            model="planner-account",
            reasoning_enabled=True,
            activate=False,
        )
        worker_config = manager.save_account_api_config(
            account["id"],
            name="Trace Worker Config",
            api_key="worker-key",
            provider="openai",
            api_base="https://worker.example.com",
            model="worker-account",
            reasoning_enabled=False,
            activate=False,
        )
        manager.save_account_api_config(
            account["id"],
            name="Trace Router Config",
            api_key="test-key",
            provider="openai",
            api_base="https://example.com",
            model="planner-account",
            reasoning_enabled=True,
            llm_routing_policy={
                "worker_api_config_id": worker_config["id"],
            },
            activate=True,
        )

        agent_config = manager._agent_store.create_agent(
            name="Trace Router Agent",
            description="Trace router",
            system_prompt="You are a planning assistant.",
            tools=[],
            account_id=account["id"],
        )
        session_id = await manager.create_session(
            agent_id=agent_config["id"],
            account_id=account["id"],
        )
        await collect_events(manager.chat(session_id, "请输出一条最终答复"))

        run = manager._run_store.list_runs(session_id=session_id)[0]
        trace_events = manager._trace_store.list_events(run.id)
        run_started = next(event for event in trace_events if event.event_type == "run_started")
        payload = json.loads(run_started.payload_summary)
        await manager.cleanup()
        return payload

    payload = asyncio.run(scenario())

    assert payload["data"]["llm"]["profile_role"] == "planner"
    assert payload["data"]["llm"]["model"] == "planner-account"
    assert payload["data"]["llm"]["source"]["config_scope"] == "account_active"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_skips_learned_workflow_generation_when_flag_disabled(
    mock_llm_class,
    tmp_path: Path,
):
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="done", finish_reason="stop")
    )

    manager = SessionManager(
        config=build_config(
            tmp_path,
            enable_learned_workflow_generation=False,
        )
    )
    await manager.initialize()

    account = create_account_with_api_config(manager, "no-workflow-gen", "No Workflow Gen")
    session_id = await manager.create_session(account_id=account["id"])
    snapshot = manager._agent_store.snapshot_agent_template(
        "system-default-agent",
        account_id=account["id"],
    )

    assert snapshot is not None
    assert manager._run_manager is not None
    assert manager._learned_workflow_store is not None

    run = RunRecord(
        id="run-no-learned-workflow",
        session_id=session_id,
        account_id=account["id"],
        agent_template_id="system-default-agent",
        agent_template_snapshot=snapshot,
        status="completed",
        goal="生成发布说明",
        run_metadata={"user_endorsed_solution": True},
        created_at="2026-04-17T01:00:00+00:00",
        started_at="2026-04-17T01:00:01+00:00",
        finished_at="2026-04-17T01:00:05+00:00",
    )
    manager._run_store.create_run(run)
    manager._run_store.create_step(
        RunStepRecord(
            id="run-no-learned-workflow-completion",
            run_id=run.id,
            sequence=1,
            step_type="completion",
            status="completed",
            title="Run completed",
            output_summary="已完成发布说明生成。",
            started_at="2026-04-17T01:00:04+00:00",
            finished_at="2026-04-17T01:00:05+00:00",
        )
    )

    manager._run_manager._maybe_capture_learned_workflow_candidate(run)

    assert manager._learned_workflow_store.list_candidate_records(account_id=account["id"]) == []


class BlockingTool(Tool):
    """Tool stub that blocks until the run gets cancelled."""

    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    @property
    def name(self) -> str:
        return "blocking_tool"

    @property
    def description(self) -> str:
        return "Blocks until interrupted."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, *args, **kwargs) -> ToolResult:  # noqa: ANN002, ANN003
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return ToolResult(success=True, content="unexpected completion")


class GateTool(Tool):
    """Tool stub that waits on an external event before returning."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    @property
    def name(self) -> str:
        return "gate_tool"

    @property
    def description(self) -> str:
        return "Waits until the test opens the gate."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, *args, **kwargs) -> ToolResult:  # noqa: ANN002, ANN003
        self.started.set()
        await self.release.wait()
        return ToolResult(success=True, content="gate opened")


class SlowTool(Tool):
    """Tool stub that deliberately exceeds the configured run timeout."""

    def __init__(self):
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    @property
    def name(self) -> str:
        return "slow_tool"

    @property
    def description(self) -> str:
        return "Sleeps long enough to trigger run timeout."

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, *args, **kwargs) -> ToolResult:  # noqa: ANN002, ANN003
        self.started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return ToolResult(success=True, content="unexpected completion")


class SimpleWriteTool(Tool):
    """Write-like tool used to verify artifact persistence and approval metadata."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Writes a file."

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
        target = self.workspace_dir / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return ToolResult(success=True, content=f"wrote {target.name}")


class ShellExportTool(Tool):
    """Bash-like tool that exports a PDF into the session workspace."""

    def __init__(self, workspace_dir: Path):
        self.workspace_dir = workspace_dir

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return "Runs one export command."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
                "run_in_background": {"type": "boolean"},
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        timeout: int = 120,
        run_in_background: bool = False,
    ) -> ToolResult:
        del timeout, run_in_background
        output_path = self.workspace_dir / "exports" / "report.pdf"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"%PDF-1.4\n% exported report\n")
        return BashOutputResult(
            success=True,
            stdout=f"Command completed. Saved to {output_path.relative_to(self.workspace_dir).as_posix()}",
            stderr="",
            exit_code=0,
            content=f"{command}\nSaved to exports/report.pdf",
        )


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_chat_creates_completed_run_records(mock_llm_class, tmp_path: Path):
    """Session chat should create a durable run with steps, checkpoints, and trace."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))

    events = await collect_events(manager.chat(session_id, "finish the task"))

    assert [event["type"] for event in events if event["type"] in {"content", "done"}] == [
        "content",
        "done",
    ]

    runs = manager._run_store.list_runs(session_id=session_id)
    assert len(runs) == 1
    run = runs[0]
    assert run.goal == "finish the task"
    assert run.status == "completed"
    assert run.started_at is not None
    assert run.finished_at is not None
    assert run.current_step_index == 1
    assert run.last_checkpoint_at is not None

    steps = manager._run_store.list_steps(run.id)
    assert [step.step_type for step in steps] == ["llm_call", "completion"]
    assert [step.status for step in steps] == ["completed", "completed"]

    checkpoints = manager._run_store.list_checkpoints(run.id)
    assert {checkpoint.trigger for checkpoint in checkpoints} == {"llm_response", "run_finalizing"}

    trace_events = manager._trace_store.list_events(run.id)
    assert [event.event_type for event in trace_events] == [
        "run_started",
        "step",
        "llm_request",
        "llm_response",
        "content",
        "run_completed",
        "checkpoint_saved",
        "checkpoint_saved",
        "done",
    ]

    log_files = sorted((tmp_path / "logs").glob("agent_run_*.log"))
    assert log_files
    assert any(run.id in log_file.name for log_file in log_files)
    log_content = log_files[-1].read_text(encoding="utf-8")
    assert '"event_type": "llm_request"' in log_content
    assert '"event_type": "llm_response"' in log_content
    assert '"event_type": "run_completed"' in log_content


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_chat_interrupt_marks_run_interrupted(mock_llm_class, tmp_path: Path):
    """Interrupting an in-flight run should persist interrupted status and repaired history."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Using a tool first.",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="blocking_tool", arguments={}),
                )
            ],
        )
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    blocking_tool = BlockingTool()
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[blocking_tool],
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace-a"),
            config=manager._config,
        ),
    )

    chat_task = asyncio.create_task(collect_events(manager.chat(session_id, "please start working")))
    await asyncio.wait_for(blocking_tool.started.wait(), timeout=1)

    interrupted = await manager.interrupt_session(session_id)
    events = await asyncio.wait_for(chat_task, timeout=1)

    assert interrupted is True
    assert blocking_tool.cancelled.is_set()
    assert any(event["type"] == "interrupted" for event in events)

    runs = manager._run_store.list_runs(session_id=session_id)
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "interrupted"
    assert run.error_summary == "Agent run interrupted by user."
    assert run.finished_at is None

    checkpoints = manager._run_store.list_checkpoints(run.id)
    assert [checkpoint.trigger for checkpoint in checkpoints] == ["llm_response"]

    trace_events = manager._trace_store.list_events(run.id)
    assert [event.event_type for event in trace_events] == [
        "run_started",
        "step",
        "llm_request",
        "llm_response",
        "content",
        "checkpoint_saved",
        "tool_call",
        "tool_started",
        "run_interrupted",
        "interrupted",
    ]

    persisted_messages = manager.get_session_messages(session_id)
    assert [message.role for message in persisted_messages] == ["system", "user", "assistant", "tool"]
    assert persisted_messages[-1].tool_call_id == "call_1"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_factory_injects_context_into_main_and_sub_agents(
    mock_llm_class,
    tmp_path: Path,
):
    """Delegated workers should execute as child runs within the same run tree."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Delegating work.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(
                            name="delegate_task",
                            arguments={
                                "persona": "You are a worker.",
                                "task": "Finish this delegated task.",
                                "max_steps": 3,
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="sub-agent result", finish_reason="stop"),
            LLMResponse(content="main agent result", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))

    observed_contexts: list[dict[str, object]] = []
    original_run_stream = Agent.run_stream

    async def instrumented_run_stream(self):  # noqa: ANN001
        context = self.runtime_context
        observed_contexts.append(
            {
                "run_id": context.run_id if context else None,
                "session_id": context.session_id if context else None,
                "agent_name": context.agent_name if context else None,
                "depth": context.depth if context else None,
                "parent_run_id": context.parent_run_id if context else None,
                "root_run_id": context.root_run_id if context else None,
                "template_id": (
                    context.template_snapshot.template_id
                    if context and context.template_snapshot is not None
                    else None
                ),
                "has_hooks": self.runtime_hooks is not None,
            }
        )
        async for event in original_run_stream(self):
            yield event

    with patch.object(Agent, "run_stream", instrumented_run_stream):
        events = await collect_events(manager.chat(session_id, "delegate this task"))

    runs = manager._run_store.list_runs(session_id=session_id)
    assert len(runs) == 2
    root_run = next(run for run in runs if run.parent_run_id is None)
    child_run = next(run for run in runs if run.parent_run_id == root_run.id)

    assert any(event["type"] == "sub_task" for event in events)
    assert observed_contexts[0] == {
        "run_id": root_run.id,
        "session_id": session_id,
        "agent_name": "main",
        "depth": 0,
        "parent_run_id": None,
        "root_run_id": root_run.id,
        "template_id": root_run.agent_template_snapshot.template_id,
        "has_hooks": True,
    }
    assert any(
        item["run_id"] == child_run.id
        and item["session_id"] == session_id
        and str(item["agent_name"]).startswith("worker-")
        and item["depth"] == 1
        and item["parent_run_id"] == root_run.id
        and item["root_run_id"] == root_run.id
        and item["template_id"] == root_run.agent_template_snapshot.template_id
        and item["has_hooks"] is True
        for item in observed_contexts[1:]
    )

    assert child_run.run_metadata["kind"] == "delegate_child"
    assert child_run.run_metadata["root_run_id"] == root_run.id

    root_trace_event_types = [
        event.event_type for event in manager._trace_store.list_events(root_run.id)
    ]
    child_trace = manager._trace_store.list_events(child_run.id)
    child_trace_event_types = [event.event_type for event in child_trace]

    assert "delegate_started" in root_trace_event_types
    assert "delegate_finished" in root_trace_event_types
    assert "run_started" in root_trace_event_types
    assert "run_started" in child_trace_event_types
    assert "run_completed" in child_trace_event_types
    assert all(event.parent_run_id == root_run.id for event in child_trace)


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_delegate_results_capture_structured_summary_and_acceptance_trace(
    mock_llm_class,
    tmp_path: Path,
):
    """Delegate completions should persist structured worker summaries and acceptance review actions."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Delegating implementation.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="delegate_call_1",
                        type="function",
                        function=FunctionCall(
                            name="delegate_task",
                            arguments={
                                "persona": "You are a worker.",
                                "task": "Write the delegated file.",
                                "max_steps": 4,
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(
                content="Worker will write the file first.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="write_call_1",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/result.md",
                                "content": "delegated output",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="worker finished", finish_reason="stop"),
            LLMResponse(content="main accepted", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path, enable_file_tools=True))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))

    await collect_events(manager.chat(session_id, "delegate and accept"))

    runs = manager._run_store.list_runs(session_id=session_id)
    root_run = next(run for run in runs if run.parent_run_id is None)

    root_steps = manager._run_store.list_steps(root_run.id)
    assert [step.step_type for step in root_steps] == [
        "llm_call",
        "delegate",
        "llm_call",
        "delegate_review",
        "completion",
    ]
    assert root_steps[1].output_summary == "worker finished"
    assert root_steps[3].title == "Delegate review: accepted"
    assert "main accepted" in root_steps[3].output_summary

    trace_events = manager._trace_store.list_events(root_run.id)
    delegate_finished = next(
        event for event in trace_events if event.event_type == "delegate_finished"
    )
    delegate_finished_payload = json.loads(delegate_finished.payload_summary)
    delegate_metadata = delegate_finished_payload["data"]["metadata"]
    assert delegate_metadata["summary"] == "worker finished"
    assert "docs/result.md" in delegate_metadata["files_changed"]
    assert delegate_metadata["commands_run"] == []
    assert delegate_metadata["tests_run"] == []

    delegate_review = next(
        event for event in trace_events if event.event_type == "delegate_reviewed"
    )
    delegate_review_payload = json.loads(delegate_review.payload_summary)
    assert delegate_review.status == "accepted"
    assert delegate_review_payload["action"] == "accepted"
    assert delegate_review_payload["files_changed"] == ["docs/result.md"]
    assert delegate_review_payload["final_response"] == "main accepted"

    session_agent = manager.get_session(session_id)
    assert session_agent is not None
    delegate_tool_messages = [
        message
        for message in session_agent.messages
        if message.role == "tool" and message.name == "delegate_task"
    ]
    assert delegate_tool_messages
    delegate_tool_message = str(delegate_tool_messages[-1].content)
    assert "Required review before final answer:" in delegate_tool_message
    assert "Structured worker report:" in delegate_tool_message
    assert "files_changed:" in delegate_tool_message
    assert "docs/result.md" in delegate_tool_message


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_delegate_review_records_retry_before_followup_worker(
    mock_llm_class,
    tmp_path: Path,
):
    """Main-agent follow-up delegation should be recorded as a retry review action."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Delegate first attempt.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="delegate_call_1",
                        type="function",
                        function=FunctionCall(
                            name="delegate_task",
                            arguments={
                                "persona": "You are worker one.",
                                "task": "First delegated attempt.",
                                "max_steps": 3,
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="first worker result", finish_reason="stop"),
            LLMResponse(
                content="Need a second worker.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="delegate_call_2",
                        type="function",
                        function=FunctionCall(
                            name="delegate_task",
                            arguments={
                                "persona": "You are worker two.",
                                "task": "Second delegated attempt.",
                                "max_steps": 3,
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="second worker result", finish_reason="stop"),
            LLMResponse(content="final merged answer", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))

    await collect_events(manager.chat(session_id, "delegate twice"))

    root_run = next(
        run for run in manager._run_store.list_runs(session_id=session_id) if run.parent_run_id is None
    )
    review_steps = [
        step for step in manager._run_store.list_steps(root_run.id) if step.step_type == "delegate_review"
    ]
    assert [step.title for step in review_steps] == [
        "Delegate review: retry_delegated",
        "Delegate review: accepted",
    ]

    review_events = [
        event
        for event in manager._trace_store.list_events(root_run.id)
        if event.event_type == "delegate_reviewed"
    ]
    assert [event.status for event in review_events] == ["retry_delegated", "accepted"]
    first_review_payload = json.loads(review_events[0].payload_summary)
    assert first_review_payload["action"] == "retry_delegated"
    assert first_review_payload["trigger_tool_name"] == "delegate_task"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_factory_rebuilds_agent_when_run_snapshot_changes(
    mock_llm_class,
    tmp_path: Path,
):
    """Run startup should rebuild the main agent when the template snapshot version changes."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    agent_config = manager._agent_store.create_agent(
        name="Planner",
        description="Plans work",
        system_prompt="Original planner prompt.",
        tools=[],
    )
    session_id = await manager.create_session(agent_id=agent_config["id"])
    initial_agent = manager.get_session(session_id)

    updated_agent_config = manager._agent_store.update_agent(
        agent_config["id"],
        system_prompt="Updated planner prompt.",
    )
    assert updated_agent_config is not None

    await collect_events(manager.chat(session_id, "use the updated prompt"))

    rebuilt_agent = manager.get_session(session_id)
    run = manager._run_store.list_runs(session_id=session_id)[0]

    assert rebuilt_agent is not initial_agent
    assert rebuilt_agent is not None
    assert rebuilt_agent.template_snapshot is not None
    assert rebuilt_agent.template_snapshot.template_version == updated_agent_config["version"]
    assert run.agent_template_snapshot.template_version == updated_agent_config["version"]
    assert "Updated planner prompt." in rebuilt_agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_refreshes_injected_user_memory_when_reusing_existing_agent(
    mock_llm_class,
    tmp_path: Path,
):
    """Reused run agents should refresh compact user memory prompt sections."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    account = create_account_with_api_config(manager, "memory-refresh", "Memory Refresh")
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "zh-CN"},
        summary="初始摘要。",
    )

    agent_config = manager._agent_store.create_agent(
        name="Planner",
        description="Plans work",
        system_prompt="Original planner prompt.",
        tools=[],
        account_id=account["id"],
    )
    session_id = await manager.create_session(
        agent_id=agent_config["id"],
        account_id=account["id"],
    )
    session_agent = manager.get_session(session_id)

    assert session_agent is not None
    assert "初始摘要。" in session_agent.system_prompt

    await collect_events(manager.chat(session_id, "bootstrap durable run agent"))

    durable_agent = manager.get_session(session_id)

    assert durable_agent is not None
    assert durable_agent is not session_agent
    assert "初始摘要。" in durable_agent.system_prompt

    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "zh-CN", "response_length": "concise"},
        summary="更新后的摘要：中文且简洁。",
    )
    store.create_memory_entry(
        user_id=account["id"],
        memory_type="correction",
        content="所有文档和代码读写必须使用 UTF-8。",
        summary="UTF-8 是硬性要求。",
        confidence=0.98,
    )

    await collect_events(manager.chat(session_id, "use the refreshed memory"))

    rebuilt_agent = manager.get_session(session_id)

    assert rebuilt_agent is durable_agent
    assert "更新后的摘要：中文且简洁。" in rebuilt_agent.system_prompt
    assert "response_length: concise" in rebuilt_agent.system_prompt
    assert "UTF-8 是硬性要求。" in rebuilt_agent.system_prompt
    assert rebuilt_agent.messages[0].role == "system"
    assert "更新后的摘要：中文且简洁。" in rebuilt_agent.messages[0].content


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_started_trace_exposes_injected_prompt_memory_sections(
    mock_llm_class,
    tmp_path: Path,
):
    """run_started trace payload should expose injected prompt memory metadata."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    account = create_account_with_api_config(manager, "trace-memory", "Trace Memory")
    store = UserMemoryStore(tmp_path / "agents.db")
    store.upsert_user_profile(
        account["id"],
        profile={"preferred_language": "zh-CN"},
        summary="偏好中文。",
    )
    store.create_memory_entry(
        user_id=account["id"],
        memory_type="constraint",
        content="所有读写都必须使用 UTF-8。",
        summary="UTF-8 是硬性要求。",
        confidence=0.95,
    )

    session_id = await manager.create_session(account_id=account["id"])
    session_agent = manager.get_session(session_id)

    assert session_agent is not None

    workspace_dir = Path(session_agent.workspace_dir)
    (workspace_dir / ".agent_memory.json").write_text(
        json.dumps(
            [
                {
                    "timestamp": "2026-04-15T12:00:00+08:00",
                    "scope": "agent_memory",
                    "category": "workflow",
                    "summary": "提交前先跑相关测试。",
                    "content": "每次修改后先跑与变更相关的测试，再提交。",
                }
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    await collect_events(manager.chat(session_id, "verify injected prompt memory trace"))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    trace_events = manager._trace_store.list_events(run.id)
    run_started = next(event for event in trace_events if event.event_type == "run_started")
    payload = json.loads(run_started.payload_summary)
    memory_sections = payload["data"]["prompt"]["memory_sections"]
    section_keys = [section["key"] for section in memory_sections]

    assert section_keys == ["user_profile", "stable_preferences", "agent_memory"]
    assert payload["data"]["prompt"]["memory_section_keys"] == section_keys
    assert payload["data"]["prompt"]["profile_fields_used"] == ["preferred_language"]
    assert memory_sections[0]["title"] == "User Profile Summary"
    assert memory_sections[0]["metadata"]["fields"] == ["preferred_language"]
    assert memory_sections[1]["sources"][0].startswith("constraint:")
    assert memory_sections[2]["title"] == "Relevant Agent Memory"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_started_trace_respects_prompt_memory_section_char_limit(
    mock_llm_class,
    tmp_path: Path,
):
    """Prompt-memory trace should keep section bodies within the configured size cap."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    config = build_config(tmp_path)
    config.agent.prompt_memory.preference_char_limit = 110
    config.agent.prompt_memory.preference_entry_limit = 4
    config.agent.prompt_memory.preference_entry_char_limit = 90

    manager = SessionManager(config=config)
    await manager.initialize()

    account = create_account_with_api_config(manager, "prompt-bounds", "Prompt Bounds")
    store = UserMemoryStore(tmp_path / "agents.db")
    for index in range(4):
        store.create_memory_entry(
            user_id=account["id"],
            memory_type="preference",
            content=f"第 {index + 1} 条偏好：所有代码和文档读写统一使用 UTF-8，说明保持中文，并在提交前完成校验。",
            summary=f"偏好 {index + 1}：统一 UTF-8、中文说明、提交前校验。",
            confidence=0.9,
        )

    session_id = await manager.create_session(account_id=account["id"])
    await collect_events(manager.chat(session_id, "检查提示词裁剪"))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    trace_events = manager._trace_store.list_events(run.id)
    run_started = next(event for event in trace_events if event.event_type == "run_started")
    payload = json.loads(run_started.payload_summary)
    preference_section = next(
        section
        for section in payload["data"]["prompt"]["memory_sections"]
        if section["key"] == "stable_preferences"
    )

    assert preference_section["chars"] <= 110
    assert len(preference_section["body"]) <= 110


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_checkpoint_and_trace_record_memory_write_observability(
    mock_llm_class,
    tmp_path: Path,
):
    """Checkpoint metadata should record whether structured memory was written in the run."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="先记录约束。",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call-record-memory",
                        type="function",
                        function=FunctionCall(
                            name="record_note",
                            arguments={
                                "content": "所有读写必须使用 UTF-8。",
                                "scope": "user_memory",
                                "memory_type": "constraint",
                                "summary": "统一使用 UTF-8。",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="已记录。", finish_reason="stop"),
        ]
    )

    config = build_config(tmp_path)
    config.tools.enable_note = True
    manager = SessionManager(config=config)
    await manager.initialize()

    account = create_account_with_api_config(manager, "memory-write-trace", "Memory Write Trace")
    session_id = await manager.create_session(account_id=account["id"])
    await collect_events(manager.chat(session_id, "记住 UTF-8 约束"))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    checkpoints = manager._run_store.list_checkpoints(run.id)
    tool_checkpoint = next(item for item in checkpoints if item.trigger == "tool_completed")
    observability = tool_checkpoint.payload.metadata["observability"]

    assert observability["memory_written"] is True
    assert observability["memory_write_count"] == 1
    assert observability["memory_writes"][0]["scope"] == "user_memory"
    assert observability["memory_writes"][0]["action"] == "memory_create"
    assert observability["memory_writes"][0]["memory_type"] == "constraint"

    checkpoint_trace = next(
        json.loads(event.payload_summary)
        for event in manager._trace_store.list_events(run.id)
        if event.event_type == "checkpoint_saved"
        and json.loads(event.payload_summary)["trigger"] == "tool_completed"
    )
    assert checkpoint_trace["observability"]["memory_written"] is True
    assert checkpoint_trace["observability"]["memory_write_count"] == 1


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_started_trace_exposes_project_context_sources(
    mock_llm_class,
    tmp_path: Path,
):
    """run_started trace payload should expose loaded project-context file sources."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    workspace_dir = tmp_path / "project-context" / "workspace"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir.parent / "AGENTS.md").write_text(
        "代码与文档统一使用 UTF-8。\n页面改动尽量保持现有风格。",
        encoding="utf-8",
    )

    session_id = await manager.create_session(str(workspace_dir))
    await collect_events(manager.chat(session_id, "verify project context trace"))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    trace_events = manager._trace_store.list_events(run.id)
    run_started = next(event for event in trace_events if event.event_type == "run_started")
    payload = json.loads(run_started.payload_summary)
    project_context_section = next(
        section
        for section in payload["data"]["prompt"]["memory_sections"]
        if section["key"] == "project_context"
    )

    assert project_context_section["title"] == "Project Context"
    assert project_context_section["items"] == 1
    assert project_context_section["sources"] == [
        (workspace_dir.parent / "AGENTS.md").resolve().as_posix()
    ]


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_progressively_discovers_project_context_from_touched_paths(
    mock_llm_class,
    tmp_path: Path,
):
    """工具触达更深目录后，应在同一运行中刷新 Project Context。"""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="先读取目标文件。",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call-read-feature",
                        type="function",
                        function=FunctionCall(
                            name="read_file",
                            arguments={"path": "modules/feature/src/main.py"},
                        ),
                    )
                ],
            ),
            LLMResponse(content="已根据模块规则继续处理。", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path, enable_file_tools=True))
    await manager.initialize()

    workspace_dir = tmp_path / "workspace-progressive-context"
    feature_dir = workspace_dir / "modules" / "feature"
    feature_dir.mkdir(parents=True, exist_ok=True)
    (feature_dir / "AGENTS.md").write_text(
        "模块规则：该目录下的功能说明与提交说明统一使用中文。",
        encoding="utf-8",
    )
    target_file = feature_dir / "src" / "main.py"
    target_file.parent.mkdir(parents=True, exist_ok=True)
    target_file.write_text("print('feature')\n", encoding="utf-8")

    session_id = await manager.create_session(str(workspace_dir))
    initial_agent = manager.get_session(session_id)

    assert initial_agent is not None
    assert "模块规则" not in initial_agent.system_prompt

    await collect_events(manager.chat(session_id, "读取 feature 文件并继续"))

    refreshed_messages = mock_llm.generate.await_args_list[1].kwargs["messages"]
    refreshed_prompt = refreshed_messages[0].content
    final_agent = manager.get_session(session_id)

    assert "Project Context" in refreshed_prompt
    assert "[AGENTS.md] modules/feature/AGENTS.md" in refreshed_prompt
    assert "模块规则：该目录下的功能说明与提交说明统一使用中文。" in refreshed_prompt
    assert final_agent is not None
    assert "模块规则：该目录下的功能说明与提交说明统一使用中文。" in final_agent.system_prompt


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_injects_recent_retrieved_context_from_history_and_memory(
    mock_llm_class,
    tmp_path: Path,
):
    """Root runs should receive compact retrieval context from prior history and user memory."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    account = create_account_with_api_config(manager, "retrieval-memory", "Retrieval Memory")
    store = UserMemoryStore(tmp_path / "agents.db")
    store.create_memory_entry(
        user_id=account["id"],
        memory_type="project_fact",
        content="发布流程文档必须注明 UTF-8 编码。",
        summary="发布流程文档必须注明 UTF-8 编码。",
        confidence=0.92,
    )

    session_id = await manager.create_session(account_id=account["id"])
    snapshot = manager._agent_store.snapshot_agent_template(
        "system-default-agent",
        account_id=account["id"],
    )
    assert snapshot is not None

    manager._run_store.create_run(
        RunRecord(
            id="prior-run",
            session_id=session_id,
            account_id=account["id"],
            agent_template_id="system-default-agent",
            agent_template_snapshot=snapshot,
            status="completed",
            goal="UTF-8 发布流程",
            created_at="2026-04-15T12:00:00+00:00",
            started_at="2026-04-15T12:00:01+00:00",
            finished_at="2026-04-15T12:00:05+00:00",
        )
    )
    manager._run_store.create_step(
        RunStepRecord(
            id="prior-step",
            run_id="prior-run",
            sequence=1,
            step_type="completion",
            status="completed",
            title="Run completed",
            output_summary="发布流程统一使用 UTF-8，并且提交前先完成验证。",
            started_at="2026-04-15T12:00:04+00:00",
            finished_at="2026-04-15T12:00:05+00:00",
        )
    )

    await collect_events(manager.chat(session_id, "UTF-8 发布流程"))

    agent = manager.get_session(session_id)
    assert agent is not None
    assert "Recent Retrieved Context" in agent.system_prompt

    current_run = next(
        run for run in manager._run_store.list_runs(session_id=session_id) if run.id != "prior-run"
    )
    trace_events = manager._trace_store.list_events(current_run.id)
    run_started = next(event for event in trace_events if event.event_type == "run_started")
    payload = json.loads(run_started.payload_summary)
    retrieved_section = next(
        section
        for section in payload["data"]["prompt"]["memory_sections"]
        if section["key"] == "retrieved_context"
    )
    retrieval_meta = payload["data"]["prompt"]["retrieval"]

    assert any(source.startswith("history:") for source in retrieved_section["sources"])
    assert any(source.startswith("memory:project_fact:") for source in retrieved_section["sources"])
    assert retrieval_meta["query"] == "UTF-8 发布流程"
    assert retrieval_meta["history_hit_count"] == 1
    assert retrieval_meta["memory_hit_count"] == 1
    assert retrieval_meta["used_sources"] == retrieved_section["sources"]


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_skips_recent_retrieved_context_when_no_matches_exist(
    mock_llm_class,
    tmp_path: Path,
):
    """Prompt injection should stay compact when retrieval returns nothing."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    session_id = await manager.create_session()
    await collect_events(manager.chat(session_id, "Docker 部署"))

    agent = manager.get_session(session_id)
    assert agent is not None
    assert "Recent Retrieved Context" not in agent.system_prompt

    current_run = manager._run_store.list_runs(session_id=session_id)[0]
    trace_events = manager._trace_store.list_events(current_run.id)
    run_started = next(event for event in trace_events if event.event_type == "run_started")
    payload = json.loads(run_started.payload_summary)
    section_keys = [section["key"] for section in payload["data"]["prompt"]["memory_sections"]]
    retrieval_meta = payload["data"]["prompt"]["retrieval"]

    assert "retrieved_context" not in section_keys
    assert retrieval_meta["query"] == "Docker 部署"
    assert retrieval_meta["used_sources"] == []
    assert retrieval_meta["history_hit_count"] == 0
    assert retrieval_meta["memory_hit_count"] == 0


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_bounds_recent_retrieved_context_entries(
    mock_llm_class,
    tmp_path: Path,
):
    """Retrieved context should respect the configured top-k entry limit."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    config = build_config(tmp_path)
    config.agent.prompt_memory.retrieved_context_entry_limit = 1
    manager = SessionManager(config=config)
    await manager.initialize()

    session_id = await manager.create_session()
    snapshot = manager._agent_store.snapshot_agent_template("system-default-agent")
    assert snapshot is not None

    manager._run_store.create_run(
        RunRecord(
            id="older-run",
            session_id=session_id,
            account_id=ROOT_ACCOUNT_ID,
            agent_template_id="system-default-agent",
            agent_template_snapshot=snapshot,
            status="completed",
            goal="UTF-8",
            created_at="2026-04-14T12:00:00+00:00",
            started_at="2026-04-14T12:00:01+00:00",
            finished_at="2026-04-14T12:00:05+00:00",
        )
    )
    manager._run_store.create_step(
        RunStepRecord(
            id="older-step",
            run_id="older-run",
            sequence=1,
            step_type="completion",
            status="completed",
            title="Run completed",
            output_summary="较早的 UTF-8 结论。",
            started_at="2026-04-14T12:00:04+00:00",
            finished_at="2026-04-14T12:00:05+00:00",
        )
    )
    manager._run_store.create_run(
        RunRecord(
            id="newer-run",
            session_id=session_id,
            account_id=ROOT_ACCOUNT_ID,
            agent_template_id="system-default-agent",
            agent_template_snapshot=snapshot,
            status="completed",
            goal="UTF-8",
            created_at="2026-04-15T12:00:00+00:00",
            started_at="2026-04-15T12:00:01+00:00",
            finished_at="2026-04-15T12:00:05+00:00",
        )
    )
    manager._run_store.create_step(
        RunStepRecord(
            id="newer-step",
            run_id="newer-run",
            sequence=1,
            step_type="completion",
            status="completed",
            title="Run completed",
            output_summary="最新的 UTF-8 结论。",
            started_at="2026-04-15T12:00:04+00:00",
            finished_at="2026-04-15T12:00:05+00:00",
        )
    )

    await collect_events(manager.chat(session_id, "UTF-8"))

    agent = manager.get_session(session_id)
    assert agent is not None
    assert "Recent Retrieved Context" in agent.system_prompt

    current_run = next(
        run
        for run in manager._run_store.list_runs(session_id=session_id)
        if run.id not in {"older-run", "newer-run"}
    )
    trace_events = manager._trace_store.list_events(current_run.id)
    run_started = next(event for event in trace_events if event.event_type == "run_started")
    payload = json.loads(run_started.payload_summary)
    retrieved_section = next(
        section
        for section in payload["data"]["prompt"]["memory_sections"]
        if section["key"] == "retrieved_context"
    )

    assert retrieved_section["items"] == 1
    assert retrieved_section["sources"] == ["history:run_step:newer-step"]


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_agent_retrieves_shared_context_history(
    mock_llm_class,
    tmp_path: Path,
):
    """Recent retrieved context should include persisted shared board entries."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    session_id = await manager.create_session()
    agent = manager.get_session(session_id)
    assert agent is not None

    shared_result = await agent.tools["share_context"].execute(
        title="Encoding decision",
        content="Shared context confirms all code and docs should keep using UTF-8 encoding.",
        category="decision",
    )
    assert shared_result.success is True

    await collect_events(manager.chat(session_id, "UTF-8 encoding"))

    current_run = manager._run_store.list_runs(session_id=session_id)[0]
    trace_events = manager._trace_store.list_events(current_run.id)
    run_started = next(event for event in trace_events if event.event_type == "run_started")
    payload = json.loads(run_started.payload_summary)
    retrieved_section = next(
        section
        for section in payload["data"]["prompt"]["memory_sections"]
        if section["key"] == "retrieved_context"
    )

    assert any(source.startswith("history:shared_context:") for source in retrieved_section["sources"])


@patch("clavi_agent.session.LLMClient")
def test_memory_evaluation_same_user_keeps_preferences_across_agent_templates(
    mock_llm_class,
    tmp_path: Path,
):
    """评估场景：同一用户切换不同 agent template 后仍继承稳定偏好。"""
    async def scenario() -> None:
        mock_llm_class.return_value.generate = AsyncMock(
            return_value=LLMResponse(content="final answer", finish_reason="stop")
        )

        manager = SessionManager(config=build_config(tmp_path))
        await manager.initialize()

        account = create_account_with_api_config(manager, "eval-same-user", "Eval Same User")
        store = UserMemoryStore(tmp_path / "agents.db")
        store.upsert_user_profile(
            account["id"],
            profile={"preferred_language": "zh-CN", "response_length": "concise"},
            summary="用户偏好中文且简洁。",
        )
        store.create_memory_entry(
            user_id=account["id"],
            memory_type="preference",
            content="所有代码与文档的读写都必须使用 UTF-8。",
            summary="统一使用 UTF-8。",
            confidence=0.95,
        )

        reviewer_template = manager._agent_store.create_agent(
            name="Reviewer",
            description="Reviews changes",
            system_prompt="You are a strict reviewer.",
            tools=[],
            account_id=account["id"],
        )

        default_session_id = await manager.create_session(account_id=account["id"])
        reviewer_session_id = await manager.create_session(
            agent_id=reviewer_template["id"],
            account_id=account["id"],
        )

        default_agent = manager.get_session(default_session_id)
        reviewer_agent = manager.get_session(reviewer_session_id)

        assert default_agent is not None
        assert reviewer_agent is not None
        assert "用户偏好中文且简洁。" in default_agent.system_prompt
        assert "response_length: concise" in default_agent.system_prompt
        assert "统一使用 UTF-8。" in default_agent.system_prompt
        assert "You are a strict reviewer." in reviewer_agent.system_prompt
        assert "用户偏好中文且简洁。" in reviewer_agent.system_prompt
        assert "response_length: concise" in reviewer_agent.system_prompt
        assert "统一使用 UTF-8。" in reviewer_agent.system_prompt

    asyncio.run(scenario())


@patch("clavi_agent.session.LLMClient")
def test_memory_evaluation_user_correction_updates_stale_preference(
    mock_llm_class,
    tmp_path: Path,
):
    """评估场景：用户纠错后，旧画像偏好会被新的显式信号覆盖。"""
    async def scenario() -> None:
        mock_llm_class.return_value.generate = AsyncMock(
            return_value=LLMResponse(content="final answer", finish_reason="stop")
        )

        manager = SessionManager(config=build_config(tmp_path))
        await manager.initialize()

        account = create_account_with_api_config(manager, "eval-correction", "Eval Correction")
        store = UserMemoryStore(tmp_path / "agents.db")
        store.upsert_user_profile(
            account["id"],
            profile={"preferred_language": "en-US", "response_length": "verbose"},
            summary="旧画像：英文且详细。",
            profile_source="inferred",
            profile_confidence=0.55,
            writer_type="system",
            writer_id="profile_inference",
        )

        session_id = await manager.create_session(account_id=account["id"])
        initial_agent = manager.get_session(session_id)

        assert initial_agent is not None
        assert "preferred_language: en-US" in initial_agent.system_prompt
        assert "response_length: verbose" in initial_agent.system_prompt

        store.upsert_user_profile(
            account["id"],
            profile={"preferred_language": "zh-CN", "response_length": "concise"},
            summary="更正后：中文且简洁。",
            profile_source="explicit",
            profile_confidence=1.0,
            source_session_id=session_id,
            source_run_id="run-user-correction",
            writer_type="tool",
            writer_id="record_note",
        )

        await collect_events(manager.chat(session_id, "按最新偏好继续"))

        refreshed_agent = manager.get_session(session_id)
        inspection = store.inspect_user_profile(account["id"])

        assert refreshed_agent is not None
        assert "更正后：中文且简洁。" in refreshed_agent.system_prompt
        assert "preferred_language: zh-CN" in refreshed_agent.system_prompt
        assert "response_length: concise" in refreshed_agent.system_prompt
        assert "response_length: verbose" not in refreshed_agent.system_prompt
        assert inspection is not None
        assert inspection["normalized_profile"]["preferred_language"] == "zh-CN"
        assert inspection["field_meta"]["preferred_language"]["source"] == "explicit"
        assert inspection["field_meta"]["preferred_language"]["writer_id"] == "record_note"

    asyncio.run(scenario())


@patch("clavi_agent.session.LLMClient")
def test_memory_evaluation_retrieves_old_decision_from_another_session(
    mock_llm_class,
    tmp_path: Path,
):
    """评估场景：当前会话可以检索同一用户另一会话中的历史决策。"""
    async def scenario() -> None:
        mock_llm_class.return_value.generate = AsyncMock(
            return_value=LLMResponse(content="final answer", finish_reason="stop")
        )

        manager = SessionManager(config=build_config(tmp_path))
        await manager.initialize()

        account = create_account_with_api_config(manager, "eval-history", "Eval History")
        prior_session_id = await manager.create_session(account_id=account["id"])
        current_session_id = await manager.create_session(account_id=account["id"])
        snapshot = manager._agent_store.snapshot_agent_template(
            "system-default-agent",
            account_id=account["id"],
        )
        assert snapshot is not None

        manager._run_store.create_run(
            RunRecord(
                id="prior-decision-run",
                session_id=prior_session_id,
                account_id=account["id"],
                agent_template_id="system-default-agent",
                agent_template_snapshot=snapshot,
                status="completed",
                goal="确认 UTF-8 决策",
                created_at="2026-04-15T09:00:00+00:00",
                started_at="2026-04-15T09:00:01+00:00",
                finished_at="2026-04-15T09:00:05+00:00",
            )
        )
        manager._run_store.create_step(
            RunStepRecord(
                id="prior-decision-step",
                run_id="prior-decision-run",
                sequence=1,
                step_type="completion",
                status="completed",
                title="Run completed",
                output_summary="上周决定：源码、文档和导出文件统一使用 UTF-8 编码。",
                started_at="2026-04-15T09:00:04+00:00",
                finished_at="2026-04-15T09:00:05+00:00",
            )
        )

        history_results = manager.search_session_history(
            "UTF-8 决策",
            account_id=account["id"],
            limit=5,
        )
        await collect_events(manager.chat(current_session_id, "请回顾 UTF-8 决策"))

        current_run = manager._run_store.list_runs(session_id=current_session_id)[0]
        trace_events = manager._trace_store.list_events(current_run.id)
        run_started = next(event for event in trace_events if event.event_type == "run_started")
        payload = json.loads(run_started.payload_summary)
        retrieval_meta = payload["data"]["prompt"]["retrieval"]
        retrieved_section = next(
            (
                section
                for section in payload["data"]["prompt"]["memory_sections"]
                if section["key"] == "retrieved_context"
            ),
            None,
        )

        assert len(history_results) == 1
        assert history_results[0]["session_id"] == prior_session_id
        assert history_results[0]["run_id"] == "prior-decision-run"
        assert history_results[0]["source_type"] in {"run_goal", "run_completion"}
        assert retrieval_meta["query"] == "请回顾 UTF-8 决策"
        if retrieval_meta["history_hit_count"] >= 1:
            assert any(
                source.startswith("history:run_step:prior-decision-step")
                for source in retrieval_meta["used_sources"]
            )
        if retrieved_section is not None:
            assert "UTF-8" in retrieved_section["body"]

    asyncio.run(scenario())


@patch("clavi_agent.session.LLMClient")
def test_memory_evaluation_retrieves_project_convention_after_context_switch(
    mock_llm_class,
    tmp_path: Path,
):
    """评估场景：切换到新的项目子目录后，应补载该目录的项目约定。"""
    async def scenario() -> None:
        mock_llm = mock_llm_class.return_value
        mock_llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="先读取文档目录里的文件。",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call-read-docs-file",
                            type="function",
                            function=FunctionCall(
                                name="read_file",
                                arguments={"path": "docs/guides/intro.md"},
                            ),
                        )
                    ],
                ),
                LLMResponse(content="已按文档目录约定继续处理。", finish_reason="stop"),
            ]
        )

        manager = SessionManager(config=build_config(tmp_path, enable_file_tools=True))
        await manager.initialize()

        workspace_dir = tmp_path / "workspace-context-eval"
        docs_dir = workspace_dir / "docs"
        guides_dir = docs_dir / "guides"
        guides_dir.mkdir(parents=True, exist_ok=True)
        (docs_dir / "AGENTS.md").write_text(
            "文档子目录约定：新增功能说明、页面提示与提交说明统一使用中文。",
            encoding="utf-8",
        )
        (guides_dir / "intro.md").write_text("# intro\n", encoding="utf-8")

        session_id = await manager.create_session(str(workspace_dir))
        initial_agent = manager.get_session(session_id)

        assert initial_agent is not None
        assert "文档子目录约定" not in initial_agent.system_prompt

        await collect_events(manager.chat(session_id, "读取 docs/guides/intro.md 并继续"))

        refreshed_messages = mock_llm.generate.await_args_list[1].kwargs["messages"]
        refreshed_prompt = refreshed_messages[0].content
        refreshed_agent = manager.get_session(session_id)

        assert "Project Context" in refreshed_prompt
        assert "[AGENTS.md] docs/AGENTS.md" in refreshed_prompt
        assert "文档子目录约定：新增功能说明、页面提示与提交说明统一使用中文。" in refreshed_prompt
        assert refreshed_agent is not None
        assert "文档子目录约定：新增功能说明、页面提示与提交说明统一使用中文。" in refreshed_agent.system_prompt

    asyncio.run(scenario())


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_falls_back_when_session_agent_belongs_to_other_account(
    mock_llm_class,
    tmp_path: Path,
):
    """Run creation should not reuse an agent template outside the session account."""
    mock_llm_class.return_value.generate = AsyncMock(
        return_value=LLMResponse(content="final answer", finish_reason="stop")
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    account_a = create_account_with_api_config(manager, "alice", "Alice")
    account_b = create_account_with_api_config(manager, "bob", "Bob")
    manager._agent_store.create_agent(
        agent_id="foreign-agent",
        name="Foreign Agent",
        system_prompt="You are foreign.",
        account_id=account_b["id"],
    )

    session_id = await manager.create_session(account_id=account_a["id"])
    with manager._session_store._connect() as conn:
        conn.execute(
            "UPDATE sessions SET agent_id = ? WHERE session_id = ?",
            ("foreign-agent", session_id),
        )

    run = manager.start_chat_run(
        session_id,
        "use the tampered session agent",
        account_id=account_a["id"],
    )

    assert run.agent_template_id == "system-default-agent"
    assert run.agent_template_snapshot.template_id == "system-default-agent"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_persists_tool_artifacts_and_execution_metadata(
    mock_llm_class,
    tmp_path: Path,
):
    """Tool execution wrapper metadata should flow into durable trace and artifacts."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need to write a file first.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/report.md",
                                "content": "artifact body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(
                content="Write an index too.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write_index",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/index.md",
                                "content": "index body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="all done", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()
    llm_client, _ = manager._get_account_llm_runtime(ROOT_ACCOUNT_ID)

    template = manager._agent_store.create_agent(
        name="Writer",
        description="Writes files",
        system_prompt="You can write files.",
        tools=["WriteTool"],
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-a"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=5,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    await collect_events(manager.chat(session_id, "write the report"))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    artifacts = manager._run_store.list_artifacts(run.id)
    assert len(artifacts) == 1
    assert artifacts[0].artifact_type == "workspace_file"
    assert artifacts[0].uri.endswith("docs\\report.md") or artifacts[0].uri.endswith(
        "docs/report.md"
    )
    assert artifacts[0].display_name == "report.md"
    assert artifacts[0].role == "final_deliverable"
    assert artifacts[0].format == "md"
    assert artifacts[0].mime_type == "text/markdown"
    assert artifacts[0].source == "agent_generated"
    assert artifacts[0].preview_kind == "markdown"
    assert artifacts[0].is_final is True
    assert artifacts[0].metadata["source_tool"] == "write_file"
    assert artifacts[0].metadata["tool_call_id"] == "call_write"
    assert run.deliverable_manifest.primary_artifact_id == artifacts[0].id
    assert run.deliverable_manifest.items[0].artifact_id == artifacts[0].id
    assert run.deliverable_manifest.items[0].is_primary is True

    trace_events = manager._trace_store.list_events(run.id)
    tool_finished = next(event for event in trace_events if event.event_type == "tool_finished")
    assert tool_finished.duration_ms is not None
    assert tool_finished.duration_ms >= 0
    assert '"requires_approval": false' in tool_finished.payload_summary
    assert '"risk_category": "filesystem_write"' in tool_finished.payload_summary


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_generates_learned_workflow_candidate_from_successful_run(
    mock_llm_class,
    tmp_path: Path,
):
    """Successful workflow-like runs should create a reviewable learned-workflow candidate."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need to write a file first.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/report.md",
                                "content": "artifact body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(
                content="Write an index too.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write_index",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/index.md",
                                "content": "index body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="all done", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()
    llm_client, _ = manager._get_account_llm_runtime(ROOT_ACCOUNT_ID)

    template = manager._agent_store.create_agent(
        name="Writer",
        description="Writes files",
        system_prompt="You can write files.",
        tools=["WriteTool"],
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-a"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=5,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    await collect_events(manager.chat(session_id, "write the report"))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    candidates = manager.list_learned_workflow_candidates(
        account_id=ROOT_ACCOUNT_ID,
        run_id=run.id,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["status"] == "pending_review"
    assert candidate["run_id"] == run.id
    assert "successful_complex_run" in candidate["signal_types"]
    assert candidate["tool_names"] == ["write_file"]
    assert candidate["generated_skill_markdown"].startswith("---\nname:")


@patch("clavi_agent.session.LLMClient")
def test_run_manager_generates_skill_improvement_proposal_from_manual_flag(
    mock_llm_class,
    tmp_path: Path,
):
    """Successful manually flagged skill refinements should create a reviewable proposal."""
    async def scenario() -> None:
        mock_llm = mock_llm_class.return_value
        mock_llm.generate = AsyncMock(
            side_effect=[
                LLMResponse(
                    content="Read the installed skill first.",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call_skill",
                            type="function",
                            function=FunctionCall(
                                name="get_skill",
                                arguments={"skill_name": "report-skill"},
                            ),
                        )
                    ],
                ),
                LLMResponse(content="Refinement completed.", finish_reason="stop"),
            ]
        )

        manager = SessionManager(config=build_config(tmp_path, enable_skills=True))
        await manager.initialize()
        llm_client, _ = manager._get_account_llm_runtime(ROOT_ACCOUNT_ID)

        template = manager._agent_store.create_agent(
            name="Skill Maintainer",
            description="Maintains installed skills",
            system_prompt="Improve installed skills when needed.",
            tools=[],
        )
        skill_dir = manager._agent_store.get_agent_skills_dir(template["id"]) / "report-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: report-skill\n"
            "description: Generate reports.\n"
            "version: 1\n"
            "---\n\n"
            "# Report Skill\n",
            encoding="utf-8",
        )
        manager._agent_store.refresh_agent_skills_from_directory(template["id"])

        session_id = await manager.create_session(agent_id=template["id"])
        skill_tools, _ = create_skill_tools(
            str(manager._agent_store.get_agent_skills_dir(template["id"]))
        )
        manager.bind_session_agent(
            session_id,
            Agent(
                llm_client=llm_client,
                system_prompt="You maintain skills.",
                tools=skill_tools,
                max_steps=5,
                workspace_dir=str(tmp_path / "workspace-a"),
                config=manager._config,
            ),
        )

        run = manager.start_run(
            session_id,
            "Improve the installed report skill",
            run_metadata={
                "skill_improvement_targets": ["report-skill"],
                "skill_improvement_notes": {
                    "report-skill": "Add a pre-export checklist for report generation.",
                },
            },
        )
        await collect_events(manager.stream_run(run.id))

        proposals = manager.list_skill_improvement_proposals(
            account_id=ROOT_ACCOUNT_ID,
            run_id=run.id,
        )

        assert len(proposals) == 1
        proposal = proposals[0]
        assert proposal["status"] == "pending_review"
        assert proposal["skill_name"] == "report-skill"
        assert proposal["base_version"] == 1
        assert proposal["proposed_version"] == 2
        assert "manual_successful_refinement" in proposal["signal_types"]
        assert "version: 2" in proposal["proposed_skill_markdown"]

    asyncio.run(scenario())


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_promotes_shell_generated_pdf_as_deliverable(
    mock_llm_class,
    tmp_path: Path,
):
    """Shell-exported PDFs should be registered as previewable deliverables."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="先导出 PDF。",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_export_pdf",
                        type="function",
                        function=FunctionCall(
                            name="bash",
                            arguments={
                                "command": "python scripts/export_report.py --output exports/report.pdf",
                                "run_in_background": False,
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="PDF 已生成", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Shell Exporter",
        description="Exports reports with shell commands.",
        system_prompt="Export requested reports as PDFs.",
        tools=["BashTool"],
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-a"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[ShellExportTool(workspace_dir)],
            max_steps=5,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    await collect_events(manager.chat(session_id, "导出最终 PDF"))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    artifacts = manager._run_store.list_artifacts(run.id)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.display_name == "report.pdf"
    assert artifact.format == "pdf"
    assert artifact.preview_kind == "pdf"
    assert artifact.mime_type == "application/pdf"
    assert artifact.is_final is True
    assert artifact.role == "final_deliverable"
    assert artifact.metadata["source_tool"] == "bash"
    assert artifact.metadata["artifact_detection"] == "shell_output_path"
    assert run.deliverable_manifest.primary_artifact_id == artifact.id
    assert run.deliverable_manifest.items[0].artifact_id == artifact.id
    assert run.deliverable_manifest.items[0].format == "pdf"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_tracks_uploaded_revision_lineage_and_deliverable_manifest(
    mock_llm_class,
    tmp_path: Path,
):
    """Uploaded-file revisions should preserve lineage metadata and become formal deliverables."""
    mock_llm = mock_llm_class.return_value

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Reviser",
        description="Revises uploaded files",
        system_prompt="Revise uploaded files by creating a copy.",
        tools=["WriteTool"],
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-a"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=5,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    upload = manager.create_session_uploads(
        session_id,
        [
            UploadCreatePayload(
                original_name="draft.md",
                content_bytes=b"# Draft\n\nOriginal body",
                mime_type="text/markdown",
            )
        ],
    )[0]
    revised_path = Path(upload.relative_path).with_name("draft.revised.md").as_posix()
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="先生成修订版文件。",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": revised_path,
                                "content": "# Revised\n\nUpdated body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="修订完成", finish_reason="stop"),
        ]
    )

    await collect_events(manager.chat(session_id, "请修订这份草稿", attachment_ids=[upload.id]))

    run = manager._run_store.list_runs(session_id=session_id)[0]
    artifacts = manager._run_store.list_artifacts(run.id)
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.role == "revised_file"
    assert artifact.source == "agent_revised"
    assert artifact.is_final is True
    assert artifact.metadata["parent_upload_id"] == upload.id
    assert artifact.metadata["parent_upload_name"] == "draft.md"
    assert artifact.metadata["revision_mode"] == "copy_on_write"
    assert artifact.uri.endswith("draft.revised.md")
    assert run.deliverable_manifest.primary_artifact_id == artifact.id
    assert run.deliverable_manifest.items[0].artifact_id == artifact.id
    assert run.deliverable_manifest.items[0].role == "revised_file"
    assert run.deliverable_manifest.items[0].is_primary is True


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_uploaded_original_overwrite_requires_approval_by_default(
    mock_llm_class,
    tmp_path: Path,
):
    """Overwriting an uploaded original should trigger approval even without template policy."""
    mock_llm = mock_llm_class.return_value

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Overwrite Upload",
        description="Overwrites uploaded originals after approval",
        system_prompt="Only overwrite the uploaded original after approval.",
        tools=["WriteTool"],
    )
    workspace_dir = tmp_path / "workspace-a"
    session_id = await manager.create_session(
        workspace_dir=str(workspace_dir),
        agent_id=template["id"],
    )
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=5,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    upload = manager.create_session_uploads(
        session_id,
        [
            UploadCreatePayload(
                original_name="draft.md",
                content_bytes=b"# Draft\n\nOriginal body",
                mime_type="text/markdown",
            )
        ],
    )[0]
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="直接覆盖上传的原文件。",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_overwrite_upload",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": upload.relative_path,
                                "content": "# Draft\n\nUpdated body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="原文件覆盖完成", finish_reason="stop"),
        ]
    )

    run = manager.start_chat_run(session_id, "请直接覆盖这份草稿", attachment_ids=[upload.id])
    stream_task = asyncio.create_task(collect_events(manager.stream_run(run.id)))

    pending_request = None
    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        pending = manager._approval_store.list_requests(status="pending", run_id=run.id)
        if pending:
            pending_request = pending[0]
            break
        await asyncio.sleep(0.05)

    assert pending_request is not None
    assert pending_request.risk_level == "critical"
    assert pending_request.parameter_summary == "覆盖已上传原文件：draft.md"
    assert "需要审批确认" in pending_request.impact_summary

    waiting_run = manager._run_store.get_run(run.id)
    assert waiting_run is not None
    assert waiting_run.status == "waiting_approval"
    assert Path(upload.absolute_path).read_text(encoding="utf-8") == "# Draft\n\nOriginal body"

    manager.resolve_approval_request(
        pending_request.id,
        status="granted",
        decision_notes="允许覆盖原文件",
    )
    events = await asyncio.wait_for(stream_task, timeout=2)

    approval_event = next(event for event in events if event["type"] == "approval_requested")
    assert approval_event["data"]["parameter_summary"] == "覆盖已上传原文件：draft.md"
    assert approval_event["data"]["risk_level"] == "critical"

    persisted_run = manager._run_store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "completed"
    assert Path(upload.absolute_path).read_text(encoding="utf-8") == "# Draft\n\nUpdated body"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_waits_for_approval_and_resumes_after_grant(
    mock_llm_class,
    tmp_path: Path,
):
    """Approved tool calls should pause the run, then resume and complete in place."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need approval before writing.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/approved.md",
                                "content": "approved body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="write completed", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Approver",
        description="Needs approval before writes",
        system_prompt="Request approval before writing files.",
        tools=["WriteTool"],
        approval_policy={
            "mode": "default",
            "require_approval_tools": ["write_file"],
        },
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-a"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=5,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    run = manager.start_chat_run(session_id, "write with approval")
    stream_task = asyncio.create_task(collect_events(manager.stream_run(run.id)))

    pending_request = None
    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        pending = manager._approval_store.list_requests(status="pending", run_id=run.id)
        if pending:
            pending_request = pending[0]
            break
        await asyncio.sleep(0.05)

    assert pending_request is not None
    waiting_run = manager._run_store.get_run(run.id)
    assert waiting_run is not None
    assert waiting_run.status == "waiting_approval"

    resolved = manager.resolve_approval_request(
        pending_request.id,
        status="granted",
        decision_notes="approved for this run",
    )
    events = await asyncio.wait_for(stream_task, timeout=2)

    assert resolved["status"] == "granted"
    assert any(event["type"] == "approval_requested" for event in events)
    assert any(event["type"] == "tool_result" for event in events)
    assert any(event["type"] == "done" for event in events)

    persisted_run = manager._run_store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "completed"

    steps = manager._run_store.list_steps(run.id)
    assert [step.step_type for step in steps] == [
        "llm_call",
        "tool_call",
        "approval_wait",
        "llm_call",
        "completion",
    ]
    assert [step.status for step in steps] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "completed",
    ]

    approval_request = manager._approval_store.get_request(pending_request.id)
    assert approval_request is not None
    assert approval_request.status == "granted"
    assert approval_request.decision_notes == "approved for this run"
    assert approval_request.decision_scope == "once"

    trace_event_types = [event.event_type for event in manager._trace_store.list_events(run.id)]
    assert "approval_requested" in trace_event_types

    written_file = workspace_dir / "docs" / "approved.md"
    assert written_file.read_text(encoding="utf-8") == "approved body"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_scope_approval_grant_skips_future_same_tool_approvals(
    mock_llm_class,
    tmp_path: Path,
):
    """Run-scoped approval grants should auto-approve later calls to the same tool within the run."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="First write needs approval.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write_1",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/first.md",
                                "content": "first body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(
                content="Write once more in the same run.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write_2",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/second.md",
                                "content": "second body",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="all writes complete", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Run Scoped Approver",
        description="Needs approval before writes",
        system_prompt="Request approval before writing files.",
        tools=["WriteTool"],
        approval_policy={
            "mode": "default",
            "require_approval_tools": ["write_file"],
        },
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-run-scope"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=6,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    run = manager.start_chat_run(session_id, "write twice with one run approval")
    stream_task = asyncio.create_task(collect_events(manager.stream_run(run.id)))

    pending_request = None
    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        pending = manager._approval_store.list_requests(status="pending", run_id=run.id)
        if pending:
            pending_request = pending[0]
            break
        await asyncio.sleep(0.05)

    assert pending_request is not None
    resolved = manager.resolve_approval_request(
        pending_request.id,
        status="granted",
        decision_notes="approved for this run",
        decision_scope="run",
    )
    events = await asyncio.wait_for(stream_task, timeout=2)

    assert resolved["status"] == "granted"
    assert resolved["decision_scope"] == "run"
    assert any(event["type"] == "done" for event in events)

    persisted_run = manager._run_store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "completed"
    assert persisted_run.run_metadata["approval_auto_grant_tools"] == ["write_file"]

    approvals = manager._approval_store.list_requests(run_id=run.id)
    assert [approval.id for approval in approvals] == [pending_request.id]
    assert approvals[0].decision_scope == "run"

    first_file = workspace_dir / "docs" / "first.md"
    second_file = workspace_dir / "docs" / "second.md"
    assert first_file.read_text(encoding="utf-8") == "first body"
    assert second_file.read_text(encoding="utf-8") == "second body"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_continues_after_approval_denial(
    mock_llm_class,
    tmp_path: Path,
):
    """Denied tool approvals should fail the tool step and let the agent continue."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need approval before writing.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "docs/denied.md",
                                "content": "should not exist",
                            },
                        ),
                    )
                ],
            ),
            LLMResponse(content="approval denied handled", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    await manager.initialize()

    template = manager._agent_store.create_agent(
        name="Approver",
        description="Needs approval before writes",
        system_prompt="Request approval before writing files.",
        tools=["WriteTool"],
        approval_policy={
            "mode": "default",
            "require_approval_tools": ["write_file"],
        },
    )
    session_id = await manager.create_session(agent_id=template["id"])
    workspace_dir = tmp_path / "workspace-a"
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[SimpleWriteTool(workspace_dir)],
            max_steps=5,
            workspace_dir=str(workspace_dir),
            config=manager._config,
        ),
    )

    run = manager.start_chat_run(session_id, "write with approval")
    stream_task = asyncio.create_task(collect_events(manager.stream_run(run.id)))

    pending_request = None
    deadline = asyncio.get_running_loop().time() + 2
    while asyncio.get_running_loop().time() < deadline:
        pending = manager._approval_store.list_requests(status="pending", run_id=run.id)
        if pending:
            pending_request = pending[0]
            break
        await asyncio.sleep(0.05)

    assert pending_request is not None
    manager.resolve_approval_request(
        pending_request.id,
        status="denied",
        decision_notes="unsafe write",
    )
    events = await asyncio.wait_for(stream_task, timeout=2)

    denied_result = next(event for event in events if event["type"] == "tool_result")
    assert denied_result["data"]["success"] is False
    assert "unsafe write" in denied_result["data"]["error"]
    assert any(event["type"] == "done" for event in events)

    persisted_run = manager._run_store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "completed"

    steps = manager._run_store.list_steps(run.id)
    assert [step.step_type for step in steps] == [
        "llm_call",
        "tool_call",
        "approval_wait",
        "llm_call",
        "completion",
    ]
    assert [step.status for step in steps] == [
        "completed",
        "failed",
        "failed",
        "completed",
        "completed",
    ]

    denied_file = workspace_dir / "docs" / "denied.md"
    assert denied_file.exists() is False


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_queues_same_session_runs_until_previous_run_finishes(
    mock_llm_class,
    tmp_path: Path,
):
    """The in-process queue should serialize root runs within the same session."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need the gate tool.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="gate_tool", arguments={}),
                    )
                ],
            ),
            LLMResponse(content="first run finished", finish_reason="stop"),
            LLMResponse(content="second run finished", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path, max_concurrent_runs=2))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    gate_tool = GateTool()
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[gate_tool],
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace-a"),
            config=manager._config,
        ),
    )

    first_task = asyncio.create_task(collect_events(manager.chat(session_id, "first run")))
    await asyncio.wait_for(gate_tool.started.wait(), timeout=1)

    second_task = asyncio.create_task(collect_events(manager.chat(session_id, "second run")))
    await asyncio.sleep(0.05)

    runs = {run.goal: run for run in manager._run_store.list_runs(session_id=session_id)}
    first_run = runs["first run"]
    second_run = runs["second run"]

    assert first_run.status == "running"
    assert second_run.status == "queued"
    assert second_run.started_at is None

    gate_tool.release.set()
    first_events = await asyncio.wait_for(first_task, timeout=1)
    second_events = await asyncio.wait_for(second_task, timeout=1)

    assert any(event["type"] == "done" for event in first_events)
    assert second_events[0]["type"] == "queued"
    assert any(event["type"] == "done" for event in second_events)

    runs = {run.goal: run for run in manager._run_store.list_runs(session_id=session_id)}
    assert runs["first run"].status == "completed"
    assert runs["second run"].status == "completed"

    trace_events = manager._trace_store.list_events(runs["second run"].id)
    assert trace_events[0].event_type == "run_queued"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_respects_global_run_concurrency_limit(
    mock_llm_class,
    tmp_path: Path,
):
    """The dispatcher should queue excess runs once the global limit is exhausted."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Need the gate tool.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="gate_tool", arguments={}),
                    )
                ],
            ),
            LLMResponse(content="first session finished", finish_reason="stop"),
            LLMResponse(content="second session finished", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path, max_concurrent_runs=1))
    first_session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    second_session_id = await manager.create_session(str(tmp_path / "workspace-b"))

    gate_tool = GateTool()
    manager.bind_session_agent(
        first_session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[gate_tool],
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace-a"),
            config=manager._config,
        ),
    )

    first_task = asyncio.create_task(collect_events(manager.chat(first_session_id, "first session run")))
    await asyncio.wait_for(gate_tool.started.wait(), timeout=1)

    second_task = asyncio.create_task(
        collect_events(manager.chat(second_session_id, "second session run"))
    )
    await asyncio.sleep(0.05)

    first_run = manager._run_store.list_runs(session_id=first_session_id)[0]
    second_run = manager._run_store.list_runs(session_id=second_session_id)[0]

    assert first_run.status == "running"
    assert second_run.status == "queued"
    assert second_run.started_at is None

    gate_tool.release.set()
    first_events = await asyncio.wait_for(first_task, timeout=1)
    second_events = await asyncio.wait_for(second_task, timeout=1)

    assert any(event["type"] == "done" for event in first_events)
    assert second_events[0]["type"] == "queued"
    assert any(event["type"] == "done" for event in second_events)

    second_trace = manager._trace_store.list_events(second_run.id)
    assert second_trace[0].event_type == "run_queued"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_cancel_run_marks_running_run_cancelled(mock_llm_class, tmp_path: Path):
    """Cancelling an in-flight run should persist cancelled status and emit cancellation events."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Using a tool first.",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="blocking_tool", arguments={}),
                )
            ],
        )
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    blocking_tool = BlockingTool()
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[blocking_tool],
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace-a"),
            config=manager._config,
        ),
    )

    run = manager.start_chat_run(session_id, "cancel this run")
    stream_task = asyncio.create_task(collect_events(manager.stream_run(run.id)))
    await asyncio.wait_for(blocking_tool.started.wait(), timeout=1)

    cancelled_run = manager._run_manager.cancel_run(run.id)
    events = await asyncio.wait_for(stream_task, timeout=1)

    assert cancelled_run.id == run.id
    assert blocking_tool.cancelled.is_set()
    assert any(event["type"] == "cancelled" for event in events)

    persisted_run = manager._run_store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "cancelled"
    assert persisted_run.error_summary == "Agent run cancelled by user."
    assert persisted_run.finished_at is not None

    trace_events = manager._trace_store.list_events(run.id)
    assert trace_events[-2].event_type == "run_cancelled"
    assert trace_events[-1].event_type == "cancelled"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_resume_run_continues_interrupted_run(mock_llm_class, tmp_path: Path):
    """An interrupted run should resume in place and continue from repaired history."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        side_effect=[
            LLMResponse(
                content="Using a tool first.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="blocking_tool", arguments={}),
                    )
                ],
            ),
            LLMResponse(content="resumed after checkpoint", finish_reason="stop"),
        ]
    )

    manager = SessionManager(config=build_config(tmp_path))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    blocking_tool = BlockingTool()
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[blocking_tool],
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace-a"),
            config=manager._config,
        ),
    )

    run = manager.start_chat_run(session_id, "interrupt and resume")
    first_stream = asyncio.create_task(collect_events(manager.stream_run(run.id)))
    await asyncio.wait_for(blocking_tool.started.wait(), timeout=1)

    interrupted = await manager.interrupt_session(session_id)
    first_events = await asyncio.wait_for(first_stream, timeout=1)

    assert interrupted is True
    assert any(event["type"] == "interrupted" for event in first_events)

    resumed_run = await manager.resume_run(run.id)
    second_events = await collect_events(manager.stream_run(run.id))

    assert resumed_run is not None
    assert any(
        event["type"] == "content" and event["data"]["content"] == "resumed after checkpoint"
        for event in second_events
    )

    persisted_run = manager._run_store.get_run(run.id)
    assert persisted_run is not None
    assert persisted_run.status == "completed"

    trace_events = manager._trace_store.list_events(run.id)
    assert any(event.event_type == "run_resumed" for event in trace_events)


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_manager_recovers_running_runs_after_restart(mock_llm_class, tmp_path: Path):
    """Startup should recover persisted running runs and continue from the latest checkpointed state."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(content="recovered completion", finish_reason="stop")
    )

    config = build_config(tmp_path)
    setup_manager = SessionManager(config=config)
    session_id = await setup_manager.create_session(str(tmp_path / "workspace-a"))
    snapshot = setup_manager._agent_store.snapshot_agent_template("system-default-agent")
    assert snapshot is not None

    setup_manager._session_store.replace_messages(
        session_id,
        [
            Message(role="system", content="You are a recovered assistant."),
            Message(role="user", content="recover this run"),
            Message(
                role="assistant",
                content="Calling the blocking tool.",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        type="function",
                        function=FunctionCall(name="blocking_tool", arguments={}),
                    )
                ],
            ),
        ],
    )
    run = setup_manager._run_store.create_run(
        RunRecord(
            id="run-recover-1",
            session_id=session_id,
            agent_template_id="system-default-agent",
            agent_template_snapshot=snapshot,
            status="running",
            goal="recover this run",
            created_at="2026-04-09T12:01:00+00:00",
            started_at="2026-04-09T12:01:10+00:00",
            current_step_index=1,
        )
    )
    setup_manager._run_store.save_checkpoint(
        RunCheckpointRecord(
            id="checkpoint-recover-1",
            run_id=run.id,
            step_sequence=1,
            trigger="llm_response",
            payload=RunCheckpointPayload(current_step_index=1),
            created_at="2026-04-09T12:01:20+00:00",
        )
    )

    recovered_manager = SessionManager(config=config)
    await recovered_manager.initialize()

    async def wait_for_completion() -> RunRecord:
        deadline = asyncio.get_running_loop().time() + 2
        while True:
            recovered = recovered_manager._run_store.get_run(run.id)
            assert recovered is not None
            if recovered.status == "completed":
                return recovered
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("Recovered run did not complete in time.")
            await asyncio.sleep(0.05)

    recovered_run = await wait_for_completion()

    assert recovered_run.last_checkpoint_at is not None
    assert recovered_run.status == "completed"
    assert recovered_run.error_summary == ""

    persisted_messages = recovered_manager.get_session_messages(session_id)
    assert persisted_messages[-1].role == "assistant"
    assert persisted_messages[-1].content == "recovered completion"

    trace_events = recovered_manager._trace_store.list_events(run.id)
    assert trace_events[0].event_type == "run_resumed"


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_run_timeout_marks_run_timed_out(mock_llm_class, tmp_path: Path):
    """Runs exceeding the configured timeout should persist timed_out status."""
    mock_llm = mock_llm_class.return_value
    mock_llm.generate = AsyncMock(
        return_value=LLMResponse(
            content="Using a slow tool.",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="slow_tool", arguments={}),
                )
            ],
        )
    )

    manager = SessionManager(config=build_config(tmp_path, run_timeout_seconds=1))
    session_id = await manager.create_session(str(tmp_path / "workspace-a"))
    slow_tool = SlowTool()
    manager.bind_session_agent(
        session_id,
        Agent(
            llm_client=manager._llm_client,
            system_prompt="You are a test assistant.",
            tools=[slow_tool],
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace-a"),
            config=manager._config,
        ),
    )

    events = await collect_events(manager.chat(session_id, "trigger timeout"))

    assert slow_tool.started.is_set()
    assert slow_tool.cancelled.is_set()
    assert any(event["type"] == "timed_out" for event in events)

    run = manager._run_store.list_runs(session_id=session_id)[0]
    assert run.status == "timed_out"
    assert run.finished_at is not None
    assert run.error_summary == "Agent run timed out after 1 seconds."

    steps = manager._run_store.list_steps(run.id)
    assert [step.step_type for step in steps] == ["llm_call", "tool_call", "failure"]
    assert [step.status for step in steps] == ["completed", "failed", "failed"]

    trace_events = manager._trace_store.list_events(run.id)
    assert [event.event_type for event in trace_events][-2:] == [
        "run_timed_out",
        "timed_out",
    ]

