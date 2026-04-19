"""Agent streaming behavior tests."""

import pytest

from clavi_agent.agent import Agent
from clavi_agent.agent_template_models import AgentTemplateSnapshot, ApprovalPolicy
from clavi_agent.schema import FunctionCall, LLMResponse, Message, ToolCall
from clavi_agent.tools.base import Tool, ToolResult


class RecordingWriteTool(Tool):
    """Simple write-like tool used to validate execution metadata."""

    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.calls = 0

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write one file."

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
        self.calls += 1
        return ToolResult(success=True, content=f"wrote {path}")


class DeltaOnlyLLM:
    """LLM stub that streams deltas and returns final response."""

    async def generate_stream(self, messages, tools):  # noqa: ANN001
        assert isinstance(messages[-1], Message)
        yield {"type": "content_delta", "data": {"delta": "Hel"}}
        yield {"type": "content_delta", "data": {"delta": "lo"}}
        yield {
            "type": "final_response",
            "data": {"response": LLMResponse(content="Hello", finish_reason="stop")},
        }

    async def generate(self, messages, tools):  # noqa: ANN001
        raise AssertionError("generate() should not be called in this test")


class FallbackLLM:
    """LLM stub that does not support async-iterator streaming."""

    async def generate_stream(self, messages, tools):  # noqa: ANN001
        return None

    async def generate(self, messages, tools):  # noqa: ANN001
        return LLMResponse(content="fallback response", finish_reason="stop")


@pytest.mark.asyncio
async def test_run_stream_emits_content_delta_before_done(tmp_path):
    agent = Agent(
        llm_client=DeltaOnlyLLM(),
        system_prompt="You are a test assistant.",
        tools=[],
        max_steps=2,
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("Say hello")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    event_types = [event["type"] for event in events]
    assert "content_delta" in event_types
    assert "done" in event_types
    assert event_types.index("content_delta") < event_types.index("done")

    # Compatibility: full content event is still emitted.
    assert any(event["type"] == "content" and event["data"]["content"] == "Hello" for event in events)

    # Persisted message should be final aggregated content.
    assert agent.messages[-1].role == "assistant"
    assert agent.messages[-1].content == "Hello"


@pytest.mark.asyncio
async def test_run_stream_falls_back_to_generate_when_stream_not_supported(tmp_path):
    agent = Agent(
        llm_client=FallbackLLM(),
        system_prompt="You are a test assistant.",
        tools=[],
        max_steps=2,
        workspace_dir=str(tmp_path),
    )
    agent.add_user_message("Use fallback")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    assert any(
        event["type"] == "content" and event["data"]["content"] == "fallback response"
        for event in events
    )
    assert events[-1]["type"] == "done"


class RepairingLLM:
    """LLM stub that validates repaired tool-call history before generating."""

    async def generate(self, messages, tools):  # noqa: ANN001
        roles = [message.role for message in messages]
        assert roles == ["system", "user", "assistant", "tool", "user"]
        assert messages[3].tool_call_id == "call_1"
        assert str(messages[3].content).startswith("Error: Tool execution was interrupted")
        return LLMResponse(content="resumed safely", finish_reason="stop")


class WriteThenStopLLM:
    """LLM stub that issues one write tool call before returning a final answer."""

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, tools):  # noqa: ANN001
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="Writing the report.",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id="call_write",
                        type="function",
                        function=FunctionCall(
                            name="write_file",
                            arguments={
                                "path": "notes/output.md",
                                "content": "hello world",
                            },
                        ),
                    )
                ],
            )
        return LLMResponse(content="done", finish_reason="stop")


@pytest.mark.asyncio
async def test_run_stream_repairs_incomplete_tool_call_history_before_next_turn(tmp_path):
    agent = Agent(
        llm_client=RepairingLLM(),
        system_prompt="You are a test assistant.",
        tools=[],
        max_steps=2,
        workspace_dir=str(tmp_path),
    )
    agent.messages = [
        Message(role="system", content="You are a test assistant."),
        Message(role="user", content="Start the task"),
        Message(
            role="assistant",
            content="I'll use a tool.",
            tool_calls=[
                ToolCall(
                    id="call_1",
                    type="function",
                    function=FunctionCall(name="demo_tool", arguments={}),
                )
            ],
        ),
        Message(role="user", content="Stop. Change direction."),
    ]

    events = []
    async for event in agent.run_stream():
        events.append(event)

    assert any(
        event["type"] == "content" and event["data"]["content"] == "resumed safely"
        for event in events
    )
    assert [message.role for message in agent.messages[:5]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]


@pytest.mark.asyncio
async def test_run_stream_tool_events_include_risk_and_artifact_metadata(tmp_path):
    agent = Agent(
        llm_client=WriteThenStopLLM(),
        system_prompt="You are a test assistant.",
        tools=[RecordingWriteTool(tmp_path)],
        max_steps=3,
        workspace_dir=str(tmp_path),
    )
    agent.bind_runtime(
        template_snapshot=AgentTemplateSnapshot(
            template_id="template-1",
            template_version=1,
            captured_at="2026-04-10T00:00:00Z",
            name="Writer",
            system_prompt="You are a writer.",
            tools=["WriteTool"],
            approval_policy=ApprovalPolicy(
                mode="default",
                require_approval_tools=["write_file"],
            ),
        )
    )
    agent.add_user_message("Write the file")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    tool_call_event = next(event for event in events if event["type"] == "tool_call")
    tool_result_event = next(event for event in events if event["type"] == "tool_result")

    assert tool_call_event["data"]["parameter_summary"] == (
        '{"path": "notes/output.md", "content": "hello world"}'
    )
    assert tool_call_event["data"]["risk_category"] == "filesystem_write"
    assert tool_call_event["data"]["risk_level"] == "high"
    assert tool_call_event["data"]["requires_approval"] is True

    assert tool_result_event["data"]["success"] is True
    assert isinstance(tool_result_event["data"]["duration_ms"], int)
    assert tool_result_event["data"]["duration_ms"] >= 0
    assert tool_result_event["data"]["artifacts"][0]["artifact_type"] == "workspace_file"
    assert tool_result_event["data"]["artifacts"][0]["uri"].endswith("notes\\output.md") or tool_result_event["data"]["artifacts"][0]["uri"].endswith("notes/output.md")
    assert tool_result_event["data"]["artifacts"][0]["display_name"] == "output.md"
    assert tool_result_event["data"]["artifacts"][0]["format"] == "md"
    assert tool_result_event["data"]["artifacts"][0]["source"] == "agent_generated"
    assert tool_result_event["data"]["artifacts"][0]["preview_kind"] == "markdown"


@pytest.mark.asyncio
async def test_run_stream_blocks_tool_execution_when_writable_root_policy_denies_path(tmp_path):
    tool = RecordingWriteTool(tmp_path)
    agent = Agent(
        llm_client=WriteThenStopLLM(),
        system_prompt="You are a test assistant.",
        tools=[tool],
        max_steps=3,
        workspace_dir=str(tmp_path),
    )
    agent.bind_runtime(
        template_snapshot=AgentTemplateSnapshot(
            template_id="template-2",
            template_version=1,
            captured_at="2026-04-10T00:00:00Z",
            name="Locked Writer",
            system_prompt="You may only write under docs.",
            tools=["WriteTool"],
            workspace_policy={
                "mode": "isolated",
                "writable_roots": ["docs"],
            },
        )
    )
    agent.add_user_message("Write the file")

    events = []
    async for event in agent.run_stream():
        events.append(event)

    tool_call_event = next(event for event in events if event["type"] == "tool_call")
    tool_result_event = next(event for event in events if event["type"] == "tool_result")

    assert tool_call_event["data"]["policy_allowed"] is False
    assert "allowed writable roots" in tool_call_event["data"]["policy_denied_reason"]
    assert tool_result_event["data"]["success"] is False
    assert "allowed writable roots" in tool_result_event["data"]["error"]
    assert tool.calls == 0

