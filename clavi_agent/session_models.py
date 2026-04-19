"""Session persistence models and in-memory runtime state."""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from .account_constants import ROOT_ACCOUNT_ID

if TYPE_CHECKING:
    from .agent import Agent


class SessionRecord(BaseModel):
    """Persisted session metadata without runtime-only fields."""

    session_id: str
    account_id: str = ROOT_ACCOUNT_ID
    title: str
    workspace_dir: str
    agent_id: str | None = None
    created_at: str
    updated_at: str
    message_count: int
    last_message_preview: str = ""
    ui_state: dict[str, Any] = Field(default_factory=dict)


SessionHistorySourceType = Literal[
    "session_message",
    "run_goal",
    "run_completion",
    "run_failure",
    "shared_context",
]


class SessionHistorySearchResult(BaseModel):
    """One matched item returned by cross-session history search."""

    source_key: str
    source_type: SessionHistorySourceType
    account_id: str = ROOT_ACCOUNT_ID
    session_id: str
    session_title: str = ""
    run_id: str | None = None
    message_seq: int | None = None
    role: str = ""
    title: str = ""
    content: str = ""
    snippet: str = ""
    created_at: str
    score: float = 0.0


@dataclass(slots=True)
class SessionRuntimeState:
    """In-memory runtime state for one session."""

    agent: "Agent"
    sub_agent_counter: itertools.count = field(default_factory=lambda: itertools.count(1))
    active_task: asyncio.Task | None = None
    active_run_id: str | None = None


class SessionRuntimeRegistry:
    """Owns transient runtime state that should not be persisted with sessions."""

    def __init__(self):
        self._runtime_states: dict[str, SessionRuntimeState] = {}

    def bind_agent(self, session_id: str, agent: "Agent") -> "Agent":
        """Register or replace the in-memory runtime agent for one session."""
        state = self._runtime_states.get(session_id)
        if state is None:
            self._runtime_states[session_id] = SessionRuntimeState(agent=agent)
            return agent

        state.agent = agent
        return agent

    def get_agent(self, session_id: str) -> "Agent | None":
        """Return the in-memory runtime agent for one session."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return None
        return state.agent

    def set_active_task(self, session_id: str, task: asyncio.Task | None):
        """Attach the active chat task to one session runtime."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return
        state.active_task = task

    def get_active_task(self, session_id: str) -> asyncio.Task | None:
        """Return the active task for one session, if any."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return None
        return state.active_task

    def set_active_run(self, session_id: str, run_id: str | None):
        """Attach the active run identifier to one session runtime."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return
        state.active_run_id = run_id

    def get_active_run(self, session_id: str) -> str | None:
        """Return the active run identifier for one session, if any."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return None
        return state.active_run_id

    def clear_active_run(self, session_id: str, run_id: str | None = None):
        """Clear the tracked active run if it still matches the provided id."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return
        if run_id is not None and state.active_run_id != run_id:
            return
        state.active_run_id = None

    def clear_active_task(self, session_id: str, task: asyncio.Task | None = None):
        """Clear the tracked task if it still matches the provided task."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return
        if task is not None and state.active_task is not task:
            return
        state.active_task = None

    def is_running(self, session_id: str) -> bool:
        """Whether the session currently has an unfinished task."""
        task = self.get_active_task(session_id)
        return task is not None and not task.done()

    def interrupt(self, session_id: str) -> bool:
        """Request interruption for one session and cancel its active task."""
        state = self._runtime_states.get(session_id)
        if state is None:
            return False

        state.agent.request_interrupt()
        task = state.active_task
        if task is None or task.done():
            return False

        task.cancel()
        return True

    def next_sub_agent_name(self, session_id: str) -> str:
        """Generate a unique sub-agent label within one session runtime."""
        state = self._runtime_states.get(session_id)
        if state is None:
            raise KeyError(f"Session runtime not found: {session_id}")
        return f"worker-{next(state.sub_agent_counter)}"

    def remove(self, session_id: str):
        """Drop all runtime state for one session."""
        self._runtime_states.pop(session_id, None)

    def session_ids(self) -> list[str]:
        """Return all session ids that currently have runtime state."""
        return list(self._runtime_states)

    def clear(self):
        """Drop all in-memory runtime state."""
        self._runtime_states.clear()
