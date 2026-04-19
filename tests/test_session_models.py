"""Tests for typed session metadata and runtime registry."""

import asyncio

import pytest

from clavi_agent.session_models import SessionRuntimeRegistry


class StubAgent:
    """Minimal agent stub for runtime registry tests."""

    def __init__(self):
        self.interrupt_requested = False

    def request_interrupt(self):
        self.interrupt_requested = True


@pytest.mark.asyncio
async def test_session_runtime_registry_tracks_runtime_only_state():
    """Runtime registry should own active task state outside persisted sessions."""
    registry = SessionRuntimeRegistry()
    agent = StubAgent()

    registry.bind_agent("session-1", agent)
    registry.set_active_run("session-1", "run-1")

    assert registry.get_agent("session-1") is agent
    assert registry.get_active_run("session-1") == "run-1"
    assert registry.next_sub_agent_name("session-1") == "worker-1"
    assert registry.next_sub_agent_name("session-1") == "worker-2"

    task = asyncio.create_task(asyncio.sleep(60))
    registry.set_active_task("session-1", task)

    interrupted = registry.interrupt("session-1")

    assert interrupted is True
    assert registry.is_running("session-1") is True
    assert agent.interrupt_requested is True

    with pytest.raises(asyncio.CancelledError):
        await task

    registry.clear_active_task("session-1", task)
    registry.clear_active_run("session-1", "run-1")
    assert registry.is_running("session-1") is False
    assert registry.get_active_run("session-1") is None


def test_session_runtime_registry_remove_discards_transient_state():
    """Removing a session runtime should discard its in-memory state."""
    registry = SessionRuntimeRegistry()
    registry.bind_agent("session-1", StubAgent())

    registry.remove("session-1")

    assert registry.get_agent("session-1") is None
    assert registry.session_ids() == []

