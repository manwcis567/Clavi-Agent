"""Tests for parallel delegate_task execution."""

import asyncio
import itertools
from pathlib import Path
from unittest.mock import patch

import pytest

from clavi_agent.agent import Agent
from clavi_agent.schema import FunctionCall, LLMResponse, ToolCall
from clavi_agent.tools.delegate_tool import DelegateBatchTool, DelegateTool


class StubLLMClient:
    """Minimal async LLM stub that returns predefined responses."""

    def __init__(self, responses: list[LLMResponse]):
        self._responses = responses
        self._index = 0

    async def generate(self, messages, tools):  # noqa: ANN001
        if self._index >= len(self._responses):
            raise AssertionError("No more stubbed responses available")
        response = self._responses[self._index]
        self._index += 1
        return response


class FakeSubAgent:
    """Sub-agent stub for validating concurrent execution behavior."""

    def __init__(self, tracker: dict[str, int], delay: float):
        self._tracker = tracker
        self._delay = delay

    def add_user_message(self, content: str):
        self._tracker["last_task_length"] = len(content)
        self._tracker["last_task"] = content
        self._tracker.setdefault("all_tasks", []).append(content)

    async def run_stream(self):
        self._tracker["active"] += 1
        self._tracker["max_active"] = max(self._tracker["max_active"], self._tracker["active"])
        try:
            await asyncio.sleep(self._delay)
            yield {"type": "content", "data": {"content": "working"}}
            yield {"type": "done", "data": {"content": "ok"}}
        finally:
            self._tracker["active"] -= 1


def _build_delegate_tool_calls() -> list[ToolCall]:
    return [
        ToolCall(
            id="call-1",
            type="function",
            function=FunctionCall(
                name="delegate_task",
                arguments={
                    "persona": "worker one",
                    "task": "task one",
                    "max_steps": 5,
                },
            ),
        ),
        ToolCall(
            id="call-2",
            type="function",
            function=FunctionCall(
                name="delegate_task",
                arguments={
                    "persona": "worker two",
                    "task": "task two",
                    "max_steps": 5,
                },
            ),
        ),
    ]


def _build_delegate_batch_tool_calls(worker_count: int = 2) -> list[ToolCall]:
    workers: list[dict[str, object]] = []
    for idx in range(worker_count):
        workers.append(
            {
                "persona": f"worker persona {idx + 1}",
                "task": f"task {idx + 1}",
                "max_steps": 5,
            }
        )

    return [
        ToolCall(
            id="call-batch-1",
            type="function",
            function=FunctionCall(
                name="delegate_tasks",
                arguments={"workers": workers},
            ),
        )
    ]


def _build_agent(
    tmp_path,
    tracker: dict[str, int],
    *,
    tool_calls: list[ToolCall] | None = None,
    parallel_limit: int = 4,
) -> Agent:
    responses = [
        LLMResponse(
            content="",
            tool_calls=tool_calls or _build_delegate_tool_calls(),
            finish_reason="tool_calls",
        ),
        LLMResponse(content="all done", tool_calls=None, finish_reason="stop"),
    ]
    llm = StubLLMClient(responses)

    def agent_factory(persona: str, max_steps: int):  # noqa: ARG001
        return FakeSubAgent(tracker=tracker, delay=0.05)

    class _Dummy:
        pass

    config = _Dummy()
    config.agent = _Dummy()
    config.agent.log_dir = str(tmp_path / "logs")
    config.agent.parallel_delegate_limit = parallel_limit

    return Agent(
        llm_client=llm,
        system_prompt="You are a supervisor.",
        tools=[
            DelegateTool(agent_factory=agent_factory),
            DelegateBatchTool(
                agent_factory=agent_factory,
                max_parallel=parallel_limit,
            ),
        ],
        max_steps=4,
        workspace_dir=str(tmp_path),
        config=config,
    )


@pytest.mark.asyncio
async def test_delegate_tasks_run_in_parallel_for_stream(tmp_path):
    tracker = {"active": 0, "max_active": 0, "last_task_length": 0}
    agent = _build_agent(tmp_path, tracker)
    agent.add_user_message("Run two workers")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    assert tracker["max_active"] >= 2

    sub_events = [event for event in events if event["type"] == "sub_task"]
    assert sub_events
    assert {event["data"].get("tool_call_id") for event in sub_events} == {"call-1", "call-2"}


@pytest.mark.asyncio
async def test_delegate_tasks_run_in_parallel_for_non_stream(tmp_path):
    tracker = {"active": 0, "max_active": 0, "last_task_length": 0}
    agent = _build_agent(tmp_path, tracker)
    agent.add_user_message("Run two workers")

    result = await agent.run()

    assert result == "all done"
    assert tracker["max_active"] >= 2


@pytest.mark.asyncio
async def test_delegate_tasks_batch_run_in_parallel_for_stream(tmp_path):
    tracker = {"active": 0, "max_active": 0, "last_task_length": 0}
    agent = _build_agent(
        tmp_path,
        tracker,
        tool_calls=_build_delegate_batch_tool_calls(worker_count=2),
    )
    agent.add_user_message("Run two workers in batch")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    assert tracker["max_active"] >= 2
    sub_events = [event for event in events if event["type"] == "sub_task"]
    assert sub_events
    assert {event["data"].get("worker_index") for event in sub_events} == {0, 1}


@pytest.mark.asyncio
async def test_delegate_tasks_batch_respects_parallel_limit(tmp_path):
    tracker = {"active": 0, "max_active": 0, "last_task_length": 0}
    agent = _build_agent(
        tmp_path,
        tracker,
        tool_calls=_build_delegate_batch_tool_calls(worker_count=5),
        parallel_limit=2,
    )
    agent.add_user_message("Run five workers with a limit")

    result = await agent.run()

    assert result == "all done"
    assert tracker["max_active"] <= 2


@pytest.mark.asyncio
async def test_delegate_tool_renders_structured_task_contract(tmp_path):
    tracker = {"active": 0, "max_active": 0, "last_task_length": 0}

    def agent_factory(persona: str, max_steps: int):  # noqa: ARG001
        return FakeSubAgent(tracker=tracker, delay=0.0)

    tool = DelegateTool(agent_factory=agent_factory)

    result = await tool.execute(
        persona="worker",
        task="Investigate the runtime delegation flow.",
        task_type="exploration",
        goal="Map the handoff path and identify the best hook.",
        scope="Only inspect runtime and delegate tool code paths.",
        files_in_scope=[
            "clavi_agent/agent_runtime.py",
            "clavi_agent/tools/delegate_tool.py",
        ],
        acceptance_criteria=[
            "Summarize the hook point.",
            "Call out any blocker clearly.",
        ],
        depends_on="Read existing shared context before concluding.",
        expected_outputs=["A concise findings summary."],
        uncertainties=["Do not edit files in this exploration pass."],
        max_steps=3,
    )

    assert result.success is True
    assert "Structured Task Contract:" in tracker["last_task"]
    assert "Task type: exploration" in tracker["last_task"]
    assert "Files in scope:" in tracker["last_task"]
    assert "- clavi_agent/agent_runtime.py" in tracker["last_task"]
    assert "Acceptance criteria:" in tracker["last_task"]
    assert "Exploration guidance:" in tracker["last_task"]


@pytest.mark.asyncio
async def test_delegate_batch_tool_renders_structured_worker_contracts(tmp_path):
    tracker = {"active": 0, "max_active": 0, "last_task_length": 0}

    def agent_factory(persona: str, max_steps: int):  # noqa: ARG001
        return FakeSubAgent(tracker=tracker, delay=0.0)

    tool = DelegateBatchTool(
        agent_factory=agent_factory,
        max_parallel=2,
    )

    result = await tool.execute(
        workers=[
            {
                "persona": "worker-1",
                "task": "Implement the runtime contract plumbing.",
                "goal": "Wire structured fields through the delegate tool.",
                "scope": "Limit changes to delegate tooling.",
                "files_in_scope": ["clavi_agent/tools/delegate_tool.py"],
                "expected_changes": ["Add structured contract support."],
                "acceptance_criteria": ["Existing delegate flows remain compatible."],
                "expected_outputs": ["Updated tool behavior and tests."],
                "max_steps": 3,
            }
        ]
    )

    assert result.success is True
    assert tracker["all_tasks"]
    assert "Structured Task Contract:" in tracker["all_tasks"][0]
    assert "Expected changes:" in tracker["all_tasks"][0]
    assert "Expected outputs:" in tracker["all_tasks"][0]


@patch("clavi_agent.session.LLMClient")
@pytest.mark.asyncio
async def test_parallel_sub_agents_get_unique_names(mock_llm_class, tmp_path: Path):
    """Concurrent sub_agent_factory calls must produce distinct agent names."""
    from clavi_agent.config import AgentConfig, Config, LLMConfig, RetryConfig, ToolsConfig
    from clavi_agent.session import SessionManager

    config = Config(
        llm=LLMConfig(
            api_key="test-key",
            api_base="https://example.com",
            model="test-model",
            provider="openai",
            retry=RetryConfig(enabled=False),
        ),
        agent=AgentConfig(
            max_steps=5,
            workspace_dir=str(tmp_path / "workspace"),
            system_prompt_path="system_prompt.md",
            log_dir=str(tmp_path / "logs"),
            session_store_path=str(tmp_path / "sessions.db"),
        ),
        tools=ToolsConfig(
            enable_file_tools=False,
            enable_bash=False,
            enable_note=False,
            enable_skills=False,
            enable_mcp=False,
        ),
    )

    manager = SessionManager(config=config)
    session_id = await manager.create_session(str(tmp_path / "workspace"))
    main_agent = manager.get_session(session_id)

    assert "delegate_tasks" in main_agent.tools

    delegate_tool = main_agent.tools["delegate_task"]
    factory = delegate_tool._agent_factory

    # Simulate concurrent factory calls via asyncio tasks
    names: list[str] = []

    async def call_factory(persona: str) -> None:
        sub = factory(persona, 3)
        assert "delegate_tasks" not in sub.tools
        # Extract the agent_name from its ShareContextTool
        share_tool = sub.tools.get("share_context")
        if share_tool is not None:
            names.append(share_tool.agent_name)

    await asyncio.gather(call_factory("worker A"), call_factory("worker B"))

    assert len(names) == 2, "Both factory calls should produce a sub-agent"
    assert names[0] != names[1], f"Concurrent workers got duplicate name: {names[0]!r}"


@pytest.mark.asyncio
async def test_shared_context_concurrent_write_no_data_loss(tmp_path: Path):
    """Concurrent asyncio writes to _SharedContextStore must not lose any entry."""
    from clavi_agent.tools.shared_context_tool import _SharedContextStore

    store = _SharedContextStore(str(tmp_path / "ctx.json"))
    n = 20

    async def write_one(i: int) -> None:
        await store.append_entry({"id": str(i), "content": f"item-{i}"})

    await asyncio.gather(*(write_one(i) for i in range(n)))

    entries = await store.load_entries()
    assert len(entries) == n, (
        f"Expected {n} entries but got {len(entries)}; concurrent writes lost data"
    )

