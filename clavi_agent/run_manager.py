"""Run orchestration layer for durable session executions."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Awaitable, Callable

from .agent import Agent, AgentInterrupted
from .agent_runtime import AgentRuntimeContext
from .agent_store import AgentStore
from .agent_template_models import SessionPolicyOverride, resolve_template_snapshot_with_policies
from .approval_store import ApprovalStore
from .learned_workflow_store import LearnedWorkflowStore
from .run_models import (
    ApprovalRequestRecord,
    ArtifactRecord,
    CheckpointMessageSnapshot,
    PendingToolCallSnapshot,
    RunCheckpointPayload,
    RunCheckpointRecord,
    RunDeliverableManifest,
    RunDeliverableRef,
    RunRecord,
    RunStepRecord,
    TraceEventRecord,
)
from .run_store import RunStore
from .schema import Message, normalize_message_content
from .session_models import SessionRuntimeRegistry
from .session_store import SessionStore
from .skill_improvement_store import SkillImprovementStore
from .skill_improvement_utils import build_skill_improvement_payload
from .sqlite_schema import utc_now_iso
from .trace_store import TraceStore
from .upload_store import UploadStore

_DELIVERABLE_FORMATS: frozenset[str] = frozenset(
    {"md", "markdown", "docx", "pdf", "pptx", "xlsx", "csv", "html", "htm", "png", "jpg", "jpeg", "gif", "webp", "svg"}
)
_REVISION_NAME_MARKERS: tuple[str, ...] = (
    "revised",
    "reviewed",
    "updated",
    "final",
    "deliverable",
    "output",
    "result",
    "export",
    "v2",
    "v3",
)


@dataclass(slots=True)
class RunExecutionState:
    """In-memory execution state for one live or recently finished run."""

    run_id: str
    session_id: str
    goal: str
    task: asyncio.Task | None = None
    subscribers: set[asyncio.Queue[tuple[int | None, dict | None]]] = field(default_factory=set)
    history: list[tuple[int, dict]] = field(default_factory=list)
    next_event_sequence: int = 0
    next_trace_sequence: int = 0
    tool_step_ids: dict[str, str] = field(default_factory=dict)
    approval_steps: dict[str, RunStepRecord] = field(default_factory=dict)
    completed: bool = False
    queued: bool = True
    append_user_message: bool = True
    persist_session_messages: bool = True
    occupies_run_slot: bool = True
    stop_status: str = "interrupted"
    stop_message: str = "Agent run interrupted by user."
    memory_write_events: list[dict[str, object]] = field(default_factory=list)


class RunManager:
    """Create, execute, and stream durable runs for sessions."""

    _TRACE_PAYLOAD_SUMMARY_LIMIT = 4000
    _RUN_STARTED_TRACE_PAYLOAD_SUMMARY_LIMIT = 32000

    def __init__(
        self,
        *,
        run_store: RunStore,
        trace_store: TraceStore,
        approval_store: ApprovalStore,
        upload_store: UploadStore | None,
        learned_workflow_store: LearnedWorkflowStore | None,
        skill_improvement_store: SkillImprovementStore | None,
        session_store: SessionStore,
        agent_store: AgentStore,
        runtime_registry: SessionRuntimeRegistry,
        load_agent: Callable[[RunRecord], Agent | None],
        sync_session_snapshot: Callable[[str], None],
        sync_memory_provider_turn: Callable[[RunRecord, list[Message]], None] | None = None,
        terminal_run_handler: Callable[[RunRecord], Awaitable[None]] | None = None,
        max_concurrent_runs: int = 4,
        run_timeout_seconds: int | None = None,
        enable_learned_workflow_generation: bool = True,
    ):
        self._run_store = run_store
        self._trace_store = trace_store
        self._approval_store = approval_store
        self._upload_store = upload_store
        self._learned_workflow_store = learned_workflow_store
        self._skill_improvement_store = skill_improvement_store
        self._session_store = session_store
        self._agent_store = agent_store
        self._runtime_registry = runtime_registry
        self._load_agent = load_agent
        self._sync_session_snapshot = sync_session_snapshot
        self._sync_memory_provider_turn = sync_memory_provider_turn
        self._terminal_run_handler = terminal_run_handler
        self._max_concurrent_runs = max(1, int(max_concurrent_runs))
        self._run_timeout_seconds = (
            max(1, int(run_timeout_seconds))
            if run_timeout_seconds is not None
            else None
        )
        self._enable_learned_workflow_generation = bool(enable_learned_workflow_generation)
        # Session history and runtime agent state are still session-scoped, so only one
        # root run can execute per session until resume/merge semantics are introduced.
        self._max_session_concurrent_runs = 1
        self._execution_states: dict[str, RunExecutionState] = {}
        self._queued_run_ids: deque[str] = deque()
        self._running_run_ids: set[str] = set()
        self._session_running_counts: dict[str, int] = {}
        self._approval_waiters: dict[str, asyncio.Future[ApprovalRequestRecord]] = {}
        self._dispatch_task: asyncio.Task | None = None
        self._dispatch_lock = asyncio.Lock()

    def create_run(
        self,
        session_id: str,
        goal: str,
        *,
        parent_run_id: str | None = None,
        run_metadata: dict[str, object] | None = None,
        session_policy_override: SessionPolicyOverride | dict[str, object] | None = None,
    ) -> RunRecord:
        """Create a queued run record for one session."""
        session = self._session_store.get_session_record(session_id)
        if session is None:
            raise KeyError(f"Session not found: {session_id}")

        agent_id = session.agent_id or "system-default-agent"
        snapshot = self._agent_store.snapshot_agent_template(
            agent_id,
            account_id=session.account_id,
        )
        if snapshot is None and agent_id != "system-default-agent":
            agent_id = "system-default-agent"
            snapshot = self._agent_store.snapshot_agent_template(
                agent_id,
                account_id=session.account_id,
            )
        if snapshot is None:
            raise KeyError(f"Agent template not found: {agent_id}")

        override_model = SessionPolicyOverride.model_validate(
            session_policy_override or {}
        )
        if parent_run_id is not None and override_model.has_overrides():
            raise ValueError("Session policy overrides are only supported for root runs.")

        system_default_snapshot = self._agent_store.snapshot_agent_template(
            "system-default-agent",
            account_id=session.account_id,
        )
        snapshot = resolve_template_snapshot_with_policies(
            template_snapshot=snapshot,
            system_default_snapshot=system_default_snapshot,
            session_override=override_model if override_model.has_overrides() else None,
        )

        metadata = dict(run_metadata or {})
        if parent_run_id is None:
            metadata.setdefault("root_run_id", "")
            metadata.setdefault(
                "policy_hierarchy",
                {
                    "system_template_id": (
                        system_default_snapshot.template_id
                        if system_default_snapshot is not None
                        else None
                    ),
                    "template_id": agent_id,
                    "session_override": override_model.has_overrides(),
                    "runtime_approval_field": "approval_auto_grant_tools",
                },
            )
            if override_model.has_overrides():
                metadata["session_policy_override"] = override_model.to_metadata_payload()

        run = RunRecord(
            id=str(uuid.uuid4()),
            session_id=session_id,
            agent_template_id=agent_id,
            agent_template_snapshot=snapshot,
            status="queued",
            goal=goal,
            created_at=utc_now_iso(),
            parent_run_id=parent_run_id,
            run_metadata=metadata,
        )
        if parent_run_id is None:
            run = run.model_copy(
                update={
                    "run_metadata": {
                        **run.run_metadata,
                        "root_run_id": run.id,
                    }
                }
            )
        return self._run_store.create_run(run)

    def start_run(
        self,
        session_id: str,
        goal: str,
        *,
        parent_run_id: str | None = None,
        run_metadata: dict[str, object] | None = None,
        session_policy_override: SessionPolicyOverride | dict[str, object] | None = None,
    ) -> RunRecord:
        """Create a run and enqueue it for background execution."""
        run = self.create_run(
            session_id,
            goal,
            parent_run_id=parent_run_id,
            run_metadata=run_metadata,
            session_policy_override=session_policy_override,
        )
        self._enqueue_run(run, append_user_message=True)
        return run

    def get_run(self, run_id: str) -> RunRecord | None:
        """Return one run by id."""
        return self._run_store.get_run(run_id)

    def recover_pending_runs(self) -> list[RunRecord]:
        """Rebuild in-memory state for queued/running runs after a restart."""
        recovered: list[RunRecord] = []
        for status in ("queued", "running"):
            for run in self._run_store.list_runs(status=status):
                if run.id in self._execution_states and not self._execution_states[run.id].completed:
                    continue
                self._enqueue_run(
                    run,
                    append_user_message=(run.status == "queued"),
                    record_queued_event=(run.status == "queued"),
                )
                recovered.append(run)
        return recovered

    def resume_run(self, run_id: str) -> RunRecord:
        """Resume one interrupted run from the latest durable checkpoint."""
        run = self._run_store.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        if run.status != "interrupted":
            raise ValueError(f"Only interrupted runs can be resumed: {run.status}")

        state = self._execution_states.get(run.id)
        if state is None or state.completed:
            self._enqueue_run(run, append_user_message=False)
            return run

        if state.queued:
            return run
        if state.task is not None and not state.task.done():
            return run

        state.completed = False
        state.task = None
        state.queued = True
        state.append_user_message = False
        state.stop_status = "interrupted"
        state.stop_message = "Agent run interrupted by user."
        self._queued_run_ids.append(run.id)
        self._schedule_dispatch()
        return run

    def cancel_run(self, run_id: str) -> RunRecord:
        """Cancel a queued, running, or interrupted run."""
        run = self._run_store.get_run(run_id)
        if run is None:
            raise KeyError(f"Run not found: {run_id}")
        if run.is_terminal:
            return run

        state = self._execution_states.get(run.id)
        if run.status == "queued":
            run = self._cancel_without_active_task(
                run,
                state=state,
                message="Agent run cancelled by user.",
            )
            if state is not None:
                self._queued_run_ids = deque(
                    queued_run_id for queued_run_id in self._queued_run_ids if queued_run_id != run.id
                )
                self._complete_state(state)
            return run

        if run.status == "interrupted":
            return self._cancel_without_active_task(
                run,
                state=state,
                message="Agent run cancelled by user.",
            )

        if state is not None:
            state.stop_status = "cancelled"
            state.stop_message = "Agent run cancelled by user."
            if run.parent_run_id is not None and state.task is not None and not state.task.done():
                state.task.cancel()
                return run
        interrupted = self._runtime_registry.interrupt(run.session_id)
        if interrupted:
            return run

        return self._cancel_without_active_task(
            run,
            state=state,
            message="Agent run cancelled by user.",
        )

    async def stream_run(
        self,
        run_id: str,
        *,
        after_sequence: int | None = None,
    ) -> AsyncGenerator[dict, None]:
        """Subscribe to one run's event stream."""
        next_sequence = 0 if after_sequence is None else max(0, int(after_sequence) + 1)
        state = self._execution_states.get(run_id)
        if state is None:
            run = self._run_store.get_run(run_id)
            if run is None:
                yield {"type": "error", "data": {"message": f"Run not found: {run_id}"}}
                return
            if after_sequence is not None:
                return
            if run.status == "completed":
                yield {"type": "done", "data": {"content": run.error_summary or run.goal}}
            elif run.status == "failed":
                yield {"type": "error", "data": {"message": run.error_summary or "Run failed."}}
            elif run.status in {"interrupted", "cancelled", "timed_out"}:
                yield {
                    "type": run.status,
                    "data": {
                        "message": run.error_summary or f"Run {run.status}.",
                    },
                }
            return

        queue: asyncio.Queue[tuple[int | None, dict | None]] = asyncio.Queue()
        state.subscribers.add(queue)

        try:
            for sequence, event in list(state.history):
                if sequence < next_sequence:
                    continue
                yield event
                next_sequence = sequence + 1

            if state.completed:
                return

            while True:
                sequence, event = await queue.get()
                if sequence is None:
                    break
                if event is None or sequence < next_sequence:
                    continue
                yield event
                next_sequence = sequence + 1
        finally:
            state.subscribers.discard(queue)

    def interrupt_session(self, session_id: str) -> bool:
        """Interrupt the active run for one session."""
        run_id = self._runtime_registry.get_active_run(session_id)
        if run_id:
            state = self._execution_states.get(run_id)
            if state is not None:
                state.stop_status = "interrupted"
                state.stop_message = "Agent run interrupted by user."
        return self._runtime_registry.interrupt(session_id)

    async def _execute_run(self, state: RunExecutionState) -> None:
        run = self._run_store.get_run(state.run_id)
        if run is None:
            self._complete_state(state)
            return

        active_llm_step: RunStepRecord | None = None
        tool_steps: dict[str, RunStepRecord] = {}
        next_step_sequence = self._next_step_sequence(run.id)
        latest_assistant_content = ""
        pending_delegate_reviews: list[dict[str, object]] = []

        try:
            agent = self._load_agent(run)
            if agent is None:
                raise KeyError(f"Session not found: {state.session_id}")

            agent.reset_interrupt()
            if state.append_user_message:
                raw_user_message_content = run.run_metadata.get(
                    "user_message_content",
                    state.goal,
                )
                agent.add_user_message(raw_user_message_content)
                if state.persist_session_messages:
                    self._session_store.append_message(state.session_id, agent.messages[-1])

            run = self._prepare_run_for_execution(state=state, run=run)
            async def consume_events() -> None:
                nonlocal active_llm_step, latest_assistant_content, next_step_sequence, run

                async for event in agent.run_stream():
                    event_type = str(event.get("type", ""))
                    data = event.get("data", {})
                    if not isinstance(data, dict):
                        data = {}
                    latest_run = self._run_store.get_run(run.id)
                    if latest_run is not None:
                        run = latest_run

                    if event_type == "step":
                        latest_assistant_content = ""
                        active_llm_step, next_step_sequence = self._start_step(
                            run_id=run.id,
                            sequence=next_step_sequence,
                            step_type="llm_call",
                            title=f"LLM step {int(data.get('current', 0) or 0)}",
                            input_summary=state.goal if next_step_sequence == 0 else "",
                        )
                        run = run.model_copy(
                            update={"current_step_index": int(data.get("current", 0) or 0)}
                        )
                        run = self._run_store.update_run(run)
                    elif event_type in {"thinking", "content"}:
                        content = str(data.get("content", "")).strip()
                        if content:
                            latest_assistant_content = content
                    elif event_type in {"thinking_delta", "content_delta"}:
                        delta = str(data.get("delta", ""))
                        if delta:
                            latest_assistant_content = f"{latest_assistant_content}{delta}"
                    elif event_type == "tool_call":
                        tool_name = str(data.get("name", "tool"))
                        if pending_delegate_reviews:
                            review_action = (
                                "retry_delegated"
                                if self._is_delegate_tool_name(tool_name)
                                else "needs_followup"
                            )
                            next_step_sequence = self._record_delegate_review(
                                state=state,
                                run=run,
                                sequence=next_step_sequence,
                                pending_reviews=pending_delegate_reviews,
                                action=review_action,
                                trigger_tool_name=tool_name,
                            )
                            pending_delegate_reviews.clear()
                        if active_llm_step is not None and not active_llm_step.is_terminal:
                            active_llm_step = self._finish_step(
                                active_llm_step,
                                "completed",
                                output_summary=latest_assistant_content or tool_name,
                            )
                            run = self._save_checkpoint(
                                run,
                                agent=agent,
                                trigger="llm_response",
                                active_step_id=active_llm_step.id,
                            )

                        tool_step, next_step_sequence = self._start_step(
                            run_id=run.id,
                            sequence=next_step_sequence,
                            step_type=(
                                "delegate"
                                if self._is_delegate_tool_name(tool_name)
                                else "tool_call"
                            ),
                            title=tool_name,
                            input_summary=str(data.get("parameter_summary", "")).strip()
                            or self._safe_json(data.get("arguments", {})),
                        )
                        tool_call_id = str(data.get("id", ""))
                        tool_steps[tool_call_id] = tool_step
                        if tool_call_id:
                            state.tool_step_ids[tool_call_id] = tool_step.id
                    elif event_type == "tool_result":
                        tool_call_id = str(data.get("tool_call_id", ""))
                        tool_step = tool_steps.get(tool_call_id)
                        if tool_step is not None and not tool_step.is_terminal:
                            metadata = data.get("metadata", {})
                            if not isinstance(metadata, dict):
                                metadata = {}
                            if bool(data.get("success")):
                                output_summary = ""
                                if self._is_delegate_tool_name(str(data.get("name", ""))):
                                    output_summary = self._summarize_delegate_metadata(metadata)
                                if not output_summary:
                                    output_summary = str(data.get("content", "")).strip()
                                if not output_summary:
                                    output_summary = self._summarize_artifacts(
                                        data.get("artifacts", [])
                                    )
                                tool_steps[tool_call_id] = self._finish_step(
                                    tool_step,
                                    "completed",
                                    output_summary=output_summary,
                                )
                            else:
                                error_summary = str(data.get("error", "")).strip()
                                tool_steps[tool_call_id] = self._finish_step(
                                    tool_step,
                                    "failed",
                                    error_summary=error_summary,
                                )
                            run = self._save_checkpoint(
                                run,
                                agent=agent,
                                trigger="tool_completed",
                                active_step_id=tool_steps[tool_call_id].id,
                            )
                            self._persist_tool_artifacts(
                                run=run,
                                step=tool_steps[tool_call_id],
                                tool_name=str(data.get("name", "")),
                                tool_call_id=tool_call_id,
                                artifacts=data.get("artifacts", []),
                            )
                            if self._is_delegate_tool_name(str(data.get("name", ""))):
                                pending_delegate_reviews.append(
                                    {
                                        "tool_call_id": tool_call_id,
                                        "tool_name": str(data.get("name", "")).strip(),
                                        "success": bool(data.get("success")),
                                        "metadata": metadata,
                                    }
                                )
                        if tool_call_id:
                            state.tool_step_ids.pop(tool_call_id, None)
                    elif event_type == "approval_requested":
                        next_step_sequence = self._next_step_sequence(run.id)
                    elif event_type == "done":
                        if pending_delegate_reviews:
                            next_step_sequence = self._record_delegate_review(
                                state=state,
                                run=run,
                                sequence=next_step_sequence,
                                pending_reviews=pending_delegate_reviews,
                                action="accepted",
                                final_response=str(data.get("content", "")).strip(),
                            )
                            pending_delegate_reviews.clear()
                        if active_llm_step is not None and not active_llm_step.is_terminal:
                            active_llm_step = self._finish_step(
                                active_llm_step,
                                "completed",
                                output_summary=latest_assistant_content
                                or str(data.get("content", "")),
                            )
                            run = self._save_checkpoint(
                                run,
                                agent=agent,
                                trigger="llm_response",
                                active_step_id=active_llm_step.id,
                            )

                        self._create_instant_step(
                            run_id=run.id,
                            sequence=next_step_sequence,
                            step_type="completion",
                            title="Run completed",
                            output_summary=str(data.get("content", "")).strip(),
                        )
                        next_step_sequence += 1
                        run = run.transition_to("completed", changed_at=utc_now_iso())
                        run = self._run_store.update_run(run)
                        run = self._promote_primary_deliverable_if_missing(run)
                        self._maybe_capture_learned_workflow_candidate(run)
                        self._maybe_capture_skill_improvement_proposals(run)
                        run = self._save_checkpoint(
                            run,
                            agent=agent,
                            trigger="run_finalizing",
                        )
                    elif event_type == "error":
                        error_message = str(data.get("message", "")).strip()
                        if pending_delegate_reviews:
                            next_step_sequence = self._record_delegate_review(
                                state=state,
                                run=run,
                                sequence=next_step_sequence,
                                pending_reviews=pending_delegate_reviews,
                                action="rejected",
                                rejection_reason=error_message,
                            )
                            pending_delegate_reviews.clear()
                        if active_llm_step is not None and not active_llm_step.is_terminal:
                            active_llm_step = self._finish_step(
                                active_llm_step,
                                "failed",
                                error_summary=error_message,
                            )

                        self._create_instant_step(
                            run_id=run.id,
                            sequence=next_step_sequence,
                            step_type="failure",
                            title="Run failed",
                            error_summary=error_message,
                            status="failed",
                        )
                        next_step_sequence += 1
                        run = run.transition_to(
                            "failed",
                            changed_at=utc_now_iso(),
                            error_summary=error_message,
                        )
                        run = self._run_store.update_run(run)
                        self._maybe_capture_skill_improvement_proposals(run)

                    latest_run = self._run_store.get_run(run.id)
                    if latest_run is not None:
                        run = latest_run
                    await self._publish_stream_event(state, run, event)

                    if event_type in {"done", "error"}:
                        return

            if self._run_timeout_seconds is None:
                await consume_events()
            else:
                async with asyncio.timeout(self._run_timeout_seconds):
                    await consume_events()
        except AgentInterrupted:
            run = self._finalize_interrupted_run(
                state=state,
                run=run,
                agent=agent,
                active_llm_step=active_llm_step,
                tool_steps=tool_steps,
                next_step_sequence=next_step_sequence,
            )
            await self._publish_stream_event(
                state,
                run,
                {"type": run.status, "data": {"message": run.error_summary}},
            )
        except asyncio.CancelledError:
            run = self._finalize_interrupted_run(
                state=state,
                run=run,
                agent=agent,
                active_llm_step=active_llm_step,
                tool_steps=tool_steps,
                next_step_sequence=next_step_sequence,
            )
            await self._publish_stream_event(
                state,
                run,
                {"type": run.status, "data": {"message": run.error_summary}},
            )
        except TimeoutError:
            state.stop_status = "timed_out"
            state.stop_message = (
                f"Agent run timed out after {self._run_timeout_seconds} seconds."
            )
            run = self._finalize_interrupted_run(
                state=state,
                run=run,
                agent=agent,
                active_llm_step=active_llm_step,
                tool_steps=tool_steps,
                next_step_sequence=next_step_sequence,
            )
            await self._publish_stream_event(
                state,
                run,
                {"type": "timed_out", "data": {"message": run.error_summary}},
            )
        except Exception as exc:
            error_message = f"Run execution failed: {exc}"
            if active_llm_step is not None and not active_llm_step.is_terminal:
                self._finish_step(active_llm_step, "failed", error_summary=error_message)

            self._create_instant_step(
                run_id=run.id,
                sequence=next_step_sequence,
                step_type="failure",
                title="Run failed",
                error_summary=error_message,
                status="failed",
            )
            run = run.transition_to(
                "failed",
                changed_at=utc_now_iso(),
                error_summary=error_message,
            )
            run = self._run_store.update_run(run)
            self._maybe_capture_skill_improvement_proposals(run)
            self._record_trace(
                state=state,
                run=run,
                event_type="run_failed",
                status=run.status,
                payload_summary=error_message,
            )
            await self._publish_stream_event(
                state,
                run,
                {"type": "error", "data": {"message": error_message}},
            )
        finally:
            if "agent" in locals() and agent is not None:
                agent.reset_interrupt()
            if state.persist_session_messages:
                self._sync_session_snapshot(state.session_id)
            if (
                self._sync_memory_provider_turn is not None
                and "agent" in locals()
                and agent is not None
            ):
                latest_run = self._run_store.get_run(state.run_id)
                if latest_run is not None and latest_run.is_terminal:
                    try:
                        self._sync_memory_provider_turn(latest_run, agent.get_history())
                    except Exception as exc:
                        self._record_trace(
                            state=state,
                            run=latest_run,
                            event_type="memory_provider_sync_failed",
                            status=latest_run.status,
                            payload_summary=str(exc),
                        )
            if self._terminal_run_handler is not None:
                latest_run = self._run_store.get_run(state.run_id)
                if latest_run is not None and latest_run.is_terminal:
                    try:
                        await self._terminal_run_handler(latest_run)
                    except Exception as exc:
                        self._record_trace(
                            state=state,
                            run=latest_run,
                            event_type="terminal_run_handler_failed",
                            status=latest_run.status,
                            payload_summary=str(exc),
                        )
            if state.occupies_run_slot:
                self._runtime_registry.clear_active_task(state.session_id, state.task)
                self._runtime_registry.clear_active_run(state.session_id, state.run_id)
            self._release_run_slot(state)
            self._schedule_dispatch()
            self._complete_state(state)

    def _start_step(
        self,
        *,
        run_id: str,
        sequence: int,
        step_type: str,
        title: str,
        input_summary: str = "",
    ) -> tuple[RunStepRecord, int]:
        step = RunStepRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            sequence=sequence,
            step_type=step_type,
            title=title,
            input_summary=input_summary,
        )
        step = self._run_store.create_step(step)
        step = self._run_store.update_step(step.transition_to("running", changed_at=utc_now_iso()))
        return step, sequence + 1

    def _finish_step(
        self,
        step: RunStepRecord,
        status: str,
        *,
        output_summary: str | None = None,
        error_summary: str | None = None,
    ) -> RunStepRecord:
        updated = step.transition_to(
            status,
            changed_at=utc_now_iso(),
            output_summary=(output_summary or "")[:1000] if output_summary is not None else None,
            error_summary=(error_summary or "")[:1000] if error_summary is not None else None,
        )
        return self._run_store.update_step(updated)

    def _create_instant_step(
        self,
        *,
        run_id: str,
        sequence: int,
        step_type: str,
        title: str,
        output_summary: str = "",
        error_summary: str = "",
        status: str = "completed",
    ) -> RunStepRecord:
        now = utc_now_iso()
        step = RunStepRecord(
            id=str(uuid.uuid4()),
            run_id=run_id,
            sequence=sequence,
            step_type=step_type,
            status=status,
            title=title,
            output_summary=output_summary[:1000],
            error_summary=error_summary[:1000],
            started_at=now,
            finished_at=now,
        )
        return self._run_store.create_step(step)

    @staticmethod
    def _is_delegate_tool_name(tool_name: str) -> bool:
        """判断当前工具名是否属于委派工具。"""
        normalized = str(tool_name or "").strip()
        return normalized in {"delegate_task", "delegate_tasks"}

    @staticmethod
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

    def _summarize_delegate_metadata(self, metadata: object) -> str:
        """从 delegate 结构化结果中提取紧凑摘要。"""
        if not isinstance(metadata, dict):
            return ""
        summary = str(metadata.get("summary", "")).strip()
        if summary:
            return summary
        blockers = metadata.get("blockers", [])
        if isinstance(blockers, list):
            deduped = self._dedupe_texts([str(item) for item in blockers])
            if deduped:
                return "; ".join(deduped[:2])
        return ""

    def _record_delegate_review(
        self,
        *,
        state: RunExecutionState,
        run: RunRecord,
        sequence: int,
        pending_reviews: list[dict[str, object]],
        action: str,
        trigger_tool_name: str = "",
        final_response: str = "",
        rejection_reason: str = "",
    ) -> int:
        """把主 agent 对 delegate 结果的验收动作落到 step 与 trace 中。"""
        summaries: list[str] = []
        files_changed: list[str] = []
        commands_run: list[str] = []
        tests_run: list[str] = []
        remaining_risks: list[str] = []
        blockers: list[str] = []

        for review in pending_reviews:
            metadata = review.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            summary = self._summarize_delegate_metadata(metadata)
            if summary:
                summaries.append(summary)
            for key, target in (
                ("files_changed", files_changed),
                ("commands_run", commands_run),
                ("tests_run", tests_run),
                ("remaining_risks", remaining_risks),
                ("blockers", blockers),
            ):
                values = metadata.get(key, [])
                if isinstance(values, list):
                    target.extend(str(item) for item in values if str(item).strip())

        payload = {
            "action": action,
            "delegate_count": len(pending_reviews),
            "tool_call_ids": [
                str(review.get("tool_call_id", "")).strip()
                for review in pending_reviews
                if str(review.get("tool_call_id", "")).strip()
            ],
            "trigger_tool_name": trigger_tool_name.strip() or None,
            "final_response": final_response.strip() or None,
            "rejection_reason": rejection_reason.strip() or None,
            "summaries": self._dedupe_texts(summaries),
            "files_changed": self._dedupe_texts(files_changed),
            "commands_run": self._dedupe_texts(commands_run),
            "tests_run": self._dedupe_texts(tests_run),
            "remaining_risks": self._dedupe_texts(remaining_risks),
            "blockers": self._dedupe_texts(blockers),
        }

        review_lines = [f"action={action}", f"delegates={len(pending_reviews)}"]
        if payload["summaries"]:
            review_lines.append(f"summaries={'; '.join(payload['summaries'][:2])}")
        if payload["files_changed"]:
            review_lines.append(
                f"files_changed={', '.join(payload['files_changed'][:3])}"
            )
        if payload["tests_run"]:
            review_lines.append(f"tests_run={', '.join(payload['tests_run'][:2])}")
        if trigger_tool_name.strip():
            review_lines.append(f"trigger_tool={trigger_tool_name.strip()}")
        if rejection_reason.strip():
            review_lines.append(f"reason={rejection_reason.strip()}")
        if final_response.strip():
            review_lines.append(f"final_response={final_response.strip()[:160]}")
        review_summary = " | ".join(review_lines)

        step_status = "failed" if action == "rejected" else "completed"
        self._create_instant_step(
            run_id=run.id,
            sequence=sequence,
            step_type="delegate_review",
            title=f"Delegate review: {action}",
            output_summary=review_summary if step_status == "completed" else "",
            error_summary=review_summary if step_status == "failed" else "",
            status=step_status,
        )
        self._record_trace(
            state=state,
            run=run,
            event_type="delegate_reviewed",
            status=action,
            payload_summary=self._safe_json(payload),
        )
        return sequence + 1

    def _finalize_interrupted_run(
        self,
        *,
        state: RunExecutionState,
        run: RunRecord,
        agent: Agent,
        active_llm_step: RunStepRecord | None,
        tool_steps: dict[str, RunStepRecord],
        next_step_sequence: int,
    ) -> RunRecord:
        agent.repair_incomplete_tool_calls()

        message = state.stop_message
        if active_llm_step is not None and not active_llm_step.is_terminal:
            self._finish_step(active_llm_step, "failed", error_summary=message)
        for tool_step in tool_steps.values():
            if tool_step.is_terminal:
                continue
            self._finish_step(tool_step, "failed", error_summary=message)
        for approval_id, approval_step in list(state.approval_steps.items()):
            if not approval_step.is_terminal:
                self._finish_step(approval_step, "failed", error_summary=message)
            waiter = self._approval_waiters.pop(approval_id, None)
            if waiter is not None and not waiter.done():
                waiter.cancel()
        state.approval_steps.clear()
        state.tool_step_ids.clear()

        self._create_instant_step(
            run_id=run.id,
            sequence=next_step_sequence,
            step_type="failure",
            title=(
                "Run cancelled"
                if state.stop_status == "cancelled"
                else "Run timed out"
                if state.stop_status == "timed_out"
                else "Run interrupted"
            ),
            error_summary=message,
            status="failed",
        )
        target_status = (
            state.stop_status
            if state.stop_status in {"cancelled", "timed_out"}
            else "interrupted"
        )
        run = run.transition_to(
            target_status,
            changed_at=utc_now_iso(),
            error_summary=message,
        )
        run = self._run_store.update_run(run)
        self._record_trace(
            state=state,
            run=run,
            event_type=f"run_{target_status}",
            status=run.status,
            payload_summary=message,
        )
        return run

    def _save_checkpoint(
        self,
        run: RunRecord,
        *,
        agent: Agent,
        trigger: str,
        active_step_id: str | None = None,
    ) -> RunRecord:
        observability = self._build_checkpoint_observability(agent=agent, run=run)
        checkpoint = RunCheckpointRecord(
            id=str(uuid.uuid4()),
            run_id=run.id,
            step_sequence=run.current_step_index,
            trigger=trigger,
            payload=RunCheckpointPayload(
                message_snapshot=self._snapshot_messages(agent.messages),
                current_step_index=run.current_step_index,
                active_step_id=active_step_id,
                incomplete_tool_calls=self._snapshot_incomplete_tool_calls(agent.messages),
                sub_agent_states=[],
                shared_context_refs=[],
                metadata={"observability": observability},
            ),
            created_at=utc_now_iso(),
        )
        self._run_store.save_checkpoint(checkpoint)
        state = self._execution_states.get(run.id)
        if state is not None and state.persist_session_messages:
            self._session_store.replace_messages(run.session_id, agent.get_history())
        run = self._run_store.update_run(
            run.model_copy(update={"last_checkpoint_at": checkpoint.created_at})
        )
        if state is not None:
            self._record_trace(
                state=state,
                run=run,
                event_type="checkpoint_saved",
                status=run.status,
                payload_summary=self._safe_json(
                    {
                        "trigger": trigger,
                        "active_step_id": active_step_id,
                        "current_step_index": run.current_step_index,
                        "agent_name": (
                            agent.runtime_context.agent_name
                            if agent.runtime_context is not None
                            else "main"
                        ),
                        "observability": observability,
                    }
                ),
            )
        return run

    @staticmethod
    def _copy_jsonable(value: object) -> object:
        """Return a detached JSON-safe copy for trace/checkpoint payloads."""
        return json.loads(json.dumps(value, ensure_ascii=False))

    def _build_checkpoint_observability(
        self,
        *,
        agent: Agent,
        run: RunRecord,
    ) -> dict[str, object]:
        """Summarize prompt and memory-write signals for checkpoint diagnostics."""
        state = self._execution_states.get(run.id)
        prompt_observability: dict[str, object] = {}
        runtime_context = agent.runtime_context
        if runtime_context is not None:
            raw_prompt = runtime_context.prompt_trace_data.get("prompt", {})
            if isinstance(raw_prompt, dict):
                prompt_observability = {
                    "memory_section_keys": list(raw_prompt.get("memory_section_keys", [])),
                    "profile_fields_used": list(raw_prompt.get("profile_fields_used", [])),
                }
                raw_retrieval = raw_prompt.get("retrieval")
                if isinstance(raw_retrieval, dict):
                    prompt_observability["retrieval"] = self._copy_jsonable(raw_retrieval)

        memory_write_events = list(state.memory_write_events) if state is not None else []
        return {
            "prompt": prompt_observability,
            "memory_written": bool(memory_write_events),
            "memory_write_count": len(memory_write_events),
            "memory_writes": self._copy_jsonable(memory_write_events[-8:]),
        }

    def _build_execution_state(
        self,
        run: RunRecord,
        *,
        append_user_message: bool,
    ) -> RunExecutionState:
        trace_events = self._trace_store.list_events(run.id)
        next_trace_sequence = trace_events[-1].sequence + 1 if trace_events else 0
        is_root_run = run.parent_run_id is None
        return RunExecutionState(
            run_id=run.id,
            session_id=run.session_id,
            goal=run.goal,
            next_trace_sequence=next_trace_sequence,
            append_user_message=append_user_message,
            persist_session_messages=is_root_run,
            occupies_run_slot=is_root_run,
        )

    def _enqueue_run(
        self,
        run: RunRecord,
        *,
        append_user_message: bool,
        record_queued_event: bool = True,
    ) -> RunExecutionState:
        state = self._build_execution_state(run, append_user_message=append_user_message)
        self._execution_states[run.id] = state
        if not state.occupies_run_slot:
            self._start_execution_task(state)
            return state

        self._queued_run_ids.append(run.id)

        if record_queued_event and not self._can_start_run(state):
            self._record_queue_event(state=state, run=run)

        self._schedule_dispatch()
        return state

    def _prepare_run_for_execution(
        self,
        *,
        state: RunExecutionState,
        run: RunRecord,
    ) -> RunRecord:
        if run.status == "queued":
            run = run.transition_to("running", changed_at=utc_now_iso())
            run = self._run_store.update_run(run.model_copy(update={"error_summary": ""}))
            return run

        if run.status == "interrupted":
            run = run.transition_to("running", changed_at=utc_now_iso())
            run = self._run_store.update_run(run.model_copy(update={"error_summary": ""}))
            self._record_trace(
                state=state,
                run=run,
                event_type="run_resumed",
                status=run.status,
                payload_summary="Run resumed from the latest durable checkpoint.",
            )
            return run

        if run.status == "running":
            run = self._run_store.update_run(run.model_copy(update={"error_summary": ""}))
            self._record_trace(
                state=state,
                run=run,
                event_type="run_resumed",
                status=run.status,
                payload_summary="Run recovered after process restart.",
            )
            return run

        raise ValueError(f"Run cannot be executed from status: {run.status}")

    def _cancel_without_active_task(
        self,
        run: RunRecord,
        *,
        state: RunExecutionState | None,
        message: str,
    ) -> RunRecord:
        run = run.transition_to("cancelled", changed_at=utc_now_iso(), error_summary=message)
        run = self._run_store.update_run(run)
        next_step_sequence = self._next_step_sequence(run.id)
        self._create_instant_step(
            run_id=run.id,
            sequence=next_step_sequence,
            step_type="failure",
            title="Run cancelled",
            error_summary=message,
            status="failed",
        )
        record_state = state or self._build_execution_state(run, append_user_message=False)
        self._record_trace(
            state=record_state,
            run=run,
            event_type="run_cancelled",
            status=run.status,
            payload_summary=message,
        )
        if state is not None:
            self._append_state_history(
                state=state,
                run=run,
                event={"type": "cancelled", "data": {"message": message}},
            )
        return run

    def _next_step_sequence(self, run_id: str) -> int:
        steps = self._run_store.list_steps(run_id)
        if not steps:
            return 0
        return steps[-1].sequence + 1

    def _snapshot_messages(self, messages: list[Message]) -> list[CheckpointMessageSnapshot]:
        snapshots: list[CheckpointMessageSnapshot] = []
        for message in messages:
            tool_call_names = []
            if message.tool_calls:
                tool_call_names = [tool_call.function.name for tool_call in message.tool_calls]
            snapshots.append(
                CheckpointMessageSnapshot(
                    role=message.role,
                    content=normalize_message_content(message.content),
                    thinking=message.thinking or "",
                    tool_call_id=message.tool_call_id,
                    name=message.name,
                    tool_call_names=tool_call_names,
                )
            )
        return snapshots

    def _snapshot_incomplete_tool_calls(
        self,
        messages: list[Message],
    ) -> list[PendingToolCallSnapshot]:
        pending: list[PendingToolCallSnapshot] = []
        responded_ids: set[str] = {
            message.tool_call_id
            for message in messages
            if message.role == "tool" and message.tool_call_id
        }

        for issued_in_step, message in enumerate(messages):
            if message.role != "assistant" or not message.tool_calls:
                continue
            for tool_call in message.tool_calls:
                if tool_call.id in responded_ids:
                    continue
                pending.append(
                    PendingToolCallSnapshot(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.function.name,
                        arguments=tool_call.function.arguments,
                        issued_in_step_sequence=issued_in_step,
                    )
                )
        return pending

    async def _publish_stream_event(
        self,
        state: RunExecutionState,
        run: RunRecord,
        event: dict,
    ) -> None:
        sequence = state.next_event_sequence
        state.next_event_sequence += 1
        state.history.append((sequence, event))
        self._record_trace(
            state=state,
            run=run,
            event_type=str(event.get("type", "event")),
            status=run.status,
            payload_summary=self._summarize_event(event),
            duration_ms=self._extract_duration_ms(event.get("data")),
        )
        for subscriber in list(state.subscribers):
            await subscriber.put((sequence, event))

    def _schedule_dispatch(self) -> None:
        """Ensure the in-process run dispatcher is scheduled."""
        if self._dispatch_task is not None and not self._dispatch_task.done():
            return
        self._dispatch_task = asyncio.create_task(
            self._dispatch_queued_runs(),
            name="run-dispatcher",
        )

    async def _dispatch_queued_runs(self) -> None:
        """Drain the in-process queue while capacity remains available."""
        async with self._dispatch_lock:
            while self._queued_run_ids and self._has_global_capacity():
                queue_length = len(self._queued_run_ids)
                started_any = False

                for _ in range(queue_length):
                    run_id = self._queued_run_ids.popleft()
                    state = self._execution_states.get(run_id)
                    if state is None or state.completed or not state.queued:
                        continue
                    if not self._can_start_run(state):
                        self._queued_run_ids.append(run_id)
                        continue

                    self._start_execution_task(state)
                    started_any = True

                    if not self._has_global_capacity():
                        break

                if not started_any:
                    break

    def _has_global_capacity(self) -> bool:
        """Whether another root run can start right now."""
        return len(self._running_run_ids) < self._max_concurrent_runs

    def _can_start_run(self, state: RunExecutionState) -> bool:
        """Whether a queued run currently has both global and per-session capacity."""
        if not self._has_global_capacity():
            return False
        return self._session_running_counts.get(state.session_id, 0) < self._max_session_concurrent_runs

    def _start_execution_task(self, state: RunExecutionState) -> None:
        """Mark one queued run as active and hand it to the executor."""
        state.queued = False
        if state.occupies_run_slot:
            self._running_run_ids.add(state.run_id)
            self._session_running_counts[state.session_id] = (
                self._session_running_counts.get(state.session_id, 0) + 1
            )

        task = asyncio.create_task(self._execute_run(state), name=f"run:{state.run_id}")
        state.task = task
        if state.occupies_run_slot:
            self._runtime_registry.set_active_run(state.session_id, state.run_id)
            self._runtime_registry.set_active_task(state.session_id, task)

    def _release_run_slot(self, state: RunExecutionState) -> None:
        """Release the capacity held by one finished run."""
        if not state.occupies_run_slot:
            return
        self._running_run_ids.discard(state.run_id)

        session_running = self._session_running_counts.get(state.session_id, 0)
        if session_running <= 1:
            self._session_running_counts.pop(state.session_id, None)
        else:
            self._session_running_counts[state.session_id] = session_running - 1

    def _record_queue_event(self, *, state: RunExecutionState, run: RunRecord) -> None:
        """Emit a durable queued event for runs waiting on dispatcher capacity."""
        event = {
            "type": "queued",
            "data": {
                "run_id": run.id,
                "session_id": run.session_id,
                "message": (
                    "Run queued until a session slot and global worker slot are available."
                ),
            },
        }
        self._append_state_history(state=state, run=run, event=event, trace_event_type="run_queued")

    def _record_trace(
        self,
        *,
        state: RunExecutionState,
        run: RunRecord,
        event_type: str,
        status: str,
        payload_summary: str,
        duration_ms: int | None = None,
    ) -> None:
        sequence = state.next_trace_sequence
        state.next_trace_sequence += 1
        self._trace_store.create_event(
            TraceEventRecord(
                id=str(uuid.uuid4()),
                run_id=run.id,
                parent_run_id=run.parent_run_id,
                step_id=None,
                sequence=sequence,
                event_type=event_type,
                status=status,
                payload_summary=self._truncate_trace_payload_summary(
                    event_type=event_type,
                    payload_summary=payload_summary,
                ),
                duration_ms=duration_ms,
                created_at=utc_now_iso(),
            )
        )

    @classmethod
    def _truncate_trace_payload_summary(
        cls,
        *,
        event_type: str,
        payload_summary: str,
    ) -> str:
        normalized = str(payload_summary or "")
        limit = (
            cls._RUN_STARTED_TRACE_PAYLOAD_SUMMARY_LIMIT
            if str(event_type or "").strip() == "run_started"
            else cls._TRACE_PAYLOAD_SUMMARY_LIMIT
        )
        return normalized[:limit]

    def record_runtime_trace_event(
        self,
        context: AgentRuntimeContext,
        event: dict[str, object],
    ) -> None:
        """Persist a structured runtime event emitted by an agent instance."""
        run_id = context.run_id
        if not run_id:
            return

        run = self._run_store.get_run(run_id)
        if run is None:
            return

        state = self._execution_states.get(run_id)
        if state is None:
            state = self._build_execution_state(run, append_user_message=False)
            self._execution_states[run_id] = state

        event_data = dict(event.get("data", {}) or {})
        event_type = str(event.get("type", ""))
        if event_type == "run_started" and context.prompt_trace_data:
            for key, value in context.prompt_trace_data.items():
                event_data.setdefault(key, value)
        if event_type == "tool_finished":
            memory_write = self._extract_memory_write_event(event_data)
            if memory_write is not None:
                state.memory_write_events.append(memory_write)

        self._record_trace(
            state=state,
            run=run,
            event_type=event_type or "runtime_event",
            status=run.status,
            payload_summary=self._safe_json(
                {
                    "context": {
                        "session_id": context.session_id,
                        "run_id": context.run_id,
                        "agent_name": context.agent_name,
                        "is_main_agent": context.is_main_agent,
                        "depth": context.depth,
                        "parent_run_id": context.parent_run_id,
                        "root_run_id": context.root_run_id,
                    },
                    "data": event_data,
                }
            ),
            duration_ms=self._extract_duration_ms(event.get("data")),
        )

    @staticmethod
    def _extract_memory_write_event(event_data: dict[str, object]) -> dict[str, object] | None:
        """Recognize successful structured-memory writes from tool-finished payloads."""
        if str(event_data.get("name", "")).strip() != "record_note":
            return None
        if not bool(event_data.get("success")):
            return None

        content = str(event_data.get("content", "")).strip()
        if not content or content.startswith("Skipped saving"):
            return None

        arguments = event_data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        scope = ""
        action = ""
        if content.startswith("Updated user profile fields:"):
            scope = "user_profile"
            action = "profile_update"
        elif content.startswith("Saved user memory entry:"):
            scope = "user_memory"
            action = "memory_create"
        elif content.startswith("Updated existing user memory entry:"):
            scope = "user_memory"
            action = "memory_update"
        elif content.startswith("Merged into existing user memory entry:"):
            scope = "user_memory"
            action = "memory_merge"
        elif content.startswith("Recorded agent memory:"):
            scope = "agent_memory"
            action = "memory_record"
        else:
            return None

        observation: dict[str, object] = {
            "tool_call_id": str(event_data.get("tool_call_id", "")).strip(),
            "scope": scope,
            "action": action,
        }
        if scope == "user_profile":
            profile_updates = arguments.get("profile_updates")
            if isinstance(profile_updates, dict):
                observation["profile_fields"] = sorted(
                    str(key).strip()
                    for key in profile_updates
                    if str(key).strip()
                )
        else:
            memory_type = str(arguments.get("memory_type", "")).strip()
            if not memory_type and scope == "agent_memory":
                memory_type = str(arguments.get("category", "")).strip()
            if memory_type:
                observation["memory_type"] = memory_type
        return observation

    async def request_approval(
        self,
        context: AgentRuntimeContext,
        agent: Agent,
        payload: dict[str, object],
    ) -> dict[str, object] | None:
        """Persist an approval request and suspend the run until it is resolved."""
        run_id = context.run_id
        if not run_id:
            return None

        run = self._run_store.get_run(run_id)
        if run is None:
            return None

        state = self._execution_states.get(run_id)
        if state is None:
            state = self._build_execution_state(run, append_user_message=False)
            self._execution_states[run_id] = state

        tool_name = str(payload.get("tool_name", "")).strip() or "tool"
        tool_call_id = str(payload.get("tool_call_id", "")).strip()
        parameter_summary = str(payload.get("parameter_summary", "")).strip()
        impact_summary = str(payload.get("impact_summary", "")).strip()
        approval_reason = str(payload.get("approval_reason", "")).strip()
        risk_category = str(payload.get("risk_category", "")).strip()
        requested_at = utc_now_iso()

        approval_step, _ = self._start_step(
            run_id=run.id,
            sequence=self._next_step_sequence(run.id),
            step_type="approval_wait",
            title=f"Approval required: {tool_name}",
            input_summary=parameter_summary or impact_summary,
        )
        approval_request = self._approval_store.create_request(
            ApprovalRequestRecord(
                id=str(uuid.uuid4()),
                run_id=run.id,
                step_id=approval_step.id,
                tool_name=tool_name,
                risk_level=str(payload.get("risk_level", "")).strip(),
                status="pending",
                parameter_summary=parameter_summary,
                impact_summary=impact_summary,
                requested_at=requested_at,
            )
        )
        state.approval_steps[approval_request.id] = approval_step

        if run.status == "running":
            run = run.transition_to("waiting_approval", changed_at=requested_at)
            run = self._run_store.update_run(run)
        run = self._save_checkpoint(
            run,
            agent=agent,
            trigger="approval_wait",
            active_step_id=approval_step.id,
        )

        future: asyncio.Future[ApprovalRequestRecord] = asyncio.get_running_loop().create_future()
        self._approval_waiters[approval_request.id] = future

        async def wait_for_resolution() -> dict[str, object]:
            try:
                resolved_request = await future
            finally:
                self._approval_waiters.pop(approval_request.id, None)

            latest_run = self._run_store.get_run(run.id) or run
            approval_step_state = state.approval_steps.pop(approval_request.id, approval_step)

            if resolved_request.status == "granted":
                scope = resolved_request.decision_scope or "once"
                scope_summary = {
                    "once": "Approval granted for this tool call.",
                    "run": "Approval granted for this run. Matching tool calls will continue without re-approval.",
                    "template": "Approval granted and template policy updated for future runs.",
                }.get(scope, "Approval granted. Tool execution resumed.")
                self._finish_step(
                    approval_step_state,
                    "completed",
                    output_summary=scope_summary,
                )
            else:
                self._finish_step(
                    approval_step_state,
                    "failed",
                    error_summary=(
                        resolved_request.decision_notes.strip()
                        or f"Approval denied for tool '{tool_name}'."
                    ),
                )

            if latest_run.status == "waiting_approval" and not state.approval_steps:
                latest_run = latest_run.transition_to("running", changed_at=utc_now_iso())
                latest_run = self._run_store.update_run(latest_run)

            return {
                "approval_request_id": resolved_request.id,
                "status": resolved_request.status,
                "decision_notes": resolved_request.decision_notes,
                "resolved_at": resolved_request.resolved_at,
                "tool_name": resolved_request.tool_name,
                "tool_call_id": tool_call_id,
                "decision_scope": resolved_request.decision_scope,
                "auto_approve_tools": (
                    [resolved_request.tool_name]
                    if resolved_request.status == "granted"
                    and resolved_request.decision_scope in {"run", "template"}
                    else []
                ),
            }

        return {
            "status": "pending",
            "event_data": {
                "approval_request_id": approval_request.id,
                "run_id": run.id,
                "tool_call_id": tool_call_id,
                "tool_name": approval_request.tool_name,
                "risk_category": risk_category,
                "risk_level": approval_request.risk_level,
                "approval_reason": approval_reason,
                "parameter_summary": approval_request.parameter_summary,
                "impact_summary": approval_request.impact_summary,
                "requested_at": approval_request.requested_at,
                "status": approval_request.status,
            },
            "waiter": wait_for_resolution(),
        }

    def notify_approval_resolved(self, request: ApprovalRequestRecord) -> None:
        """Wake a waiting run after one approval request is resolved."""
        future = self._approval_waiters.get(request.id)
        if future is None or future.done():
            return
        future.set_result(request)

    @staticmethod
    def _summarize_event(event: dict) -> str:
        data = event.get("data", {})
        if not isinstance(data, dict):
            return ""
        return json.dumps(data, ensure_ascii=False)[:1000]

    @staticmethod
    def _safe_json(payload: object) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False)
        except TypeError:
            return str(payload)

    @staticmethod
    def _extract_duration_ms(payload: object) -> int | None:
        if not isinstance(payload, dict):
            return None
        value = payload.get("duration_ms")
        if value is None:
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _summarize_artifacts(artifacts: object) -> str:
        if not isinstance(artifacts, list):
            return ""
        summaries: list[str] = []
        for item in artifacts:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary", "")).strip()
            uri = str(item.get("uri", "")).strip()
            if summary:
                summaries.append(summary)
            elif uri:
                summaries.append(uri)
        return ", ".join(summaries[:3])

    def _persist_tool_artifacts(
        self,
        *,
        run: RunRecord,
        step: RunStepRecord,
        tool_name: str,
        tool_call_id: str,
        artifacts: object,
    ) -> None:
        if not isinstance(artifacts, list):
            return

        for artifact in artifacts:
            if not isinstance(artifact, dict):
                continue
            artifact_type = str(artifact.get("artifact_type", "")).strip()
            uri = str(artifact.get("uri", "")).strip()
            if not artifact_type or not uri:
                continue

            metadata = artifact.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {"value": metadata}

            persisted_artifact = ArtifactRecord(
                id=str(uuid.uuid4()),
                run_id=run.id,
                step_id=step.id,
                artifact_type=artifact_type,
                uri=uri,
                display_name=self._artifact_display_name(artifact),
                role=str(artifact.get("role", "intermediate_file")).strip() or "intermediate_file",
                format=str(artifact.get("format", "")).strip(),
                mime_type=str(artifact.get("mime_type", "")).strip(),
                size_bytes=self._artifact_size_bytes(artifact.get("size_bytes")),
                source=str(artifact.get("source", "agent_generated")).strip() or "agent_generated",
                is_final=self._artifact_is_final(artifact.get("is_final")),
                preview_kind=str(artifact.get("preview_kind", "none")).strip() or "none",
                parent_artifact_id=self._artifact_parent_id(artifact.get("parent_artifact_id")),
                summary=str(artifact.get("summary", "")).strip(),
                metadata={
                    **metadata,
                    "source_tool": tool_name,
                    "tool_call_id": tool_call_id,
                },
                created_at=utc_now_iso(),
            )
            persisted_artifact = self._normalize_artifact_for_run(
                run=run,
                artifact=persisted_artifact,
            )
            self._run_store.create_artifact(persisted_artifact)
            self._maybe_update_run_deliverable_manifest(run=run, artifact=persisted_artifact)

    @staticmethod
    def _artifact_display_name(artifact: dict) -> str:
        display_name = str(artifact.get("display_name", "")).strip()
        if display_name:
            return display_name
        uri = str(artifact.get("uri", "")).strip().replace("\\", "/").rstrip("/")
        if not uri:
            return ""
        return uri.rsplit("/", 1)[-1]

    @staticmethod
    def _artifact_size_bytes(value: object) -> int | None:
        if value is None or value == "":
            return None
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _artifact_is_final(value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "final"}
        if isinstance(value, (int, float)):
            return bool(value)
        return False

    @staticmethod
    def _artifact_parent_id(value: object) -> str | None:
        normalized = str(value or "").strip()
        return normalized or None

    def _normalize_artifact_for_run(
        self,
        *,
        run: RunRecord,
        artifact: ArtifactRecord,
    ) -> ArtifactRecord:
        metadata = dict(artifact.metadata)
        artifact_path = self._resolve_run_artifact_path(run, artifact.uri)
        if artifact_path is None:
            return artifact

        lineage = self._match_upload_lineage(run=run, artifact_path=artifact_path)
        if lineage is None:
            return artifact

        metadata.update(
            {
                "parent_upload_id": lineage["upload_id"],
                "parent_upload_name": lineage["upload_name"],
                "parent_upload_relative_path": lineage["relative_path"],
                "parent_upload_checksum": lineage["checksum"],
                "revision_mode": lineage["revision_mode"],
            }
        )
        role = artifact.role
        if role in {"", "intermediate_file", "supporting_output"}:
            role = "revised_file"

        return artifact.model_copy(
            update={
                "role": role,
                "source": "agent_revised",
                "is_final": True,
                "metadata": metadata,
            }
        )

    def _resolve_run_artifact_path(self, run: RunRecord, uri: str) -> Path | None:
        normalized_uri = str(uri).strip()
        if not normalized_uri or "://" in normalized_uri:
            return None

        session = self._session_store.get_session_record(run.session_id)
        if session is None:
            return None

        workspace_dir = Path(session.workspace_dir).resolve()
        candidate = Path(normalized_uri)
        if candidate.is_absolute():
            return candidate.resolve()
        return (workspace_dir / candidate).resolve()

    def _run_referenced_upload_ids(self, run: RunRecord) -> set[str]:
        raw_content = run.run_metadata.get("user_message_content")
        try:
            normalized = normalize_message_content(raw_content)
        except (TypeError, ValueError):
            return set()
        if isinstance(normalized, str):
            return set()

        referenced_upload_ids: set[str] = set()
        for block in normalized:
            if not isinstance(block, dict) or block.get("type") != "uploaded_file":
                continue
            upload_id = str(block.get("upload_id", "")).strip()
            if upload_id:
                referenced_upload_ids.add(upload_id)
        return referenced_upload_ids

    def _list_session_upload_descriptors(self, run: RunRecord) -> list[dict[str, object]]:
        if self._upload_store is None:
            return []

        session = self._session_store.get_session_record(run.session_id)
        if session is None:
            return []

        workspace_dir = Path(session.workspace_dir).resolve()
        referenced_upload_ids = self._run_referenced_upload_ids(run)
        descriptors: list[dict[str, object]] = []
        for upload in self._upload_store.list_uploads(run.session_id):
            try:
                absolute_path = (workspace_dir / Path(upload.relative_path)).resolve()
            except Exception:
                absolute_path = Path(upload.absolute_path).resolve()
            descriptors.append(
                {
                    "upload_id": upload.id,
                    "upload_name": upload.original_name or upload.safe_name or upload.id,
                    "relative_path": upload.relative_path,
                    "checksum": upload.checksum,
                    "absolute_path": absolute_path,
                    "referenced_in_run": upload.id in referenced_upload_ids,
                }
            )
        return descriptors

    @staticmethod
    def _stems_related(left: str, right: str) -> bool:
        normalized_left = left.strip().lower()
        normalized_right = right.strip().lower()
        if not normalized_left or not normalized_right:
            return False
        return (
            normalized_left == normalized_right
            or normalized_left in normalized_right
            or normalized_right in normalized_left
        )

    @staticmethod
    def _looks_like_revision_name(stem: str) -> bool:
        normalized = stem.strip().lower()
        return any(marker in normalized for marker in _REVISION_NAME_MARKERS)

    def _match_upload_lineage(
        self,
        *,
        run: RunRecord,
        artifact_path: Path,
    ) -> dict[str, object] | None:
        best_match: dict[str, object] | None = None
        best_score = 0
        artifact_suffix = artifact_path.suffix.lower()
        artifact_stem = artifact_path.stem

        for upload in self._list_session_upload_descriptors(run):
            upload_path = upload["absolute_path"]
            if not isinstance(upload_path, Path):
                continue

            if artifact_path == upload_path:
                score = 500
                revision_mode = "overwrite"
            else:
                score = 0
                if artifact_path.parent == upload_path.parent:
                    score += 220
                if artifact_suffix and artifact_suffix == upload_path.suffix.lower():
                    score += 30
                if self._stems_related(artifact_stem, upload_path.stem):
                    score += 45
                if self._looks_like_revision_name(artifact_stem):
                    score += 20
                revision_mode = "copy_on_write"

            if upload.get("referenced_in_run"):
                score += 15

            if score < 80 or score <= best_score:
                continue

            best_score = score
            best_match = {
                **upload,
                "revision_mode": revision_mode,
            }

        return best_match

    @staticmethod
    def _artifact_uri_key(artifact: ArtifactRecord) -> str:
        normalized_uri = artifact.uri.strip().replace("\\", "/")
        return normalized_uri or artifact.id

    @staticmethod
    def _artifact_counts_as_deliverable(artifact: ArtifactRecord) -> bool:
        return artifact.is_final or artifact.role == "final_deliverable"

    @staticmethod
    def _artifact_looks_like_deliverable_candidate(artifact: ArtifactRecord) -> bool:
        if artifact.artifact_type not in {"workspace_file", "document"}:
            return False
        if artifact.source == "system_generated":
            return False
        if "://" in artifact.uri:
            return False
        artifact_format = artifact.format.strip().lower()
        return artifact_format in _DELIVERABLE_FORMATS

    @staticmethod
    def _deliverable_score(artifact: ArtifactRecord) -> int:
        score = 0
        if artifact.metadata.get("parent_upload_id"):
            score += 500
        if artifact.source == "agent_revised":
            score += 400
        if artifact.role == "final_deliverable":
            score += 320
        if artifact.role == "revised_file":
            score += 280
        if artifact.is_final:
            score += 240
        if artifact.preview_kind and artifact.preview_kind != "none":
            score += 80
        if artifact.format.strip().lower() in _DELIVERABLE_FORMATS:
            score += 60
        if RunManager._looks_like_revision_name(Path(artifact.display_name or artifact.uri).stem):
            score += 30
        return score

    def _select_best_primary_artifact(
        self,
        artifacts: list[ArtifactRecord],
        *,
        allow_candidate_promotion: bool = False,
    ) -> ArtifactRecord | None:
        candidates: list[ArtifactRecord] = []
        for artifact in artifacts:
            if self._artifact_counts_as_deliverable(artifact):
                candidates.append(artifact)
                continue
            if allow_candidate_promotion and self._artifact_looks_like_deliverable_candidate(artifact):
                candidates.append(artifact)

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda item: (
                self._deliverable_score(item),
                item.created_at,
                item.id,
            ),
        )

    def _rebuild_run_deliverable_manifest(self, *, run: RunRecord) -> RunRecord:
        current_run = self._run_store.get_run(run.id) or run
        deduped: list[ArtifactRecord] = []
        seen_uris: set[str] = set()
        for artifact in self._run_store.list_artifacts(run.id):
            if not self._artifact_counts_as_deliverable(artifact):
                continue
            uri_key = self._artifact_uri_key(artifact)
            if uri_key in seen_uris:
                continue
            seen_uris.add(uri_key)
            deduped.append(artifact)

        primary_artifact = self._select_best_primary_artifact(deduped)
        if primary_artifact is None:
            manifest = RunDeliverableManifest()
        else:
            ordered = sorted(
                deduped,
                key=lambda item: (
                    self._deliverable_score(item),
                    item.created_at,
                    item.id,
                ),
                reverse=True,
            )
            manifest = RunDeliverableManifest(
                primary_artifact_id=primary_artifact.id,
                items=[
                    RunDeliverableRef(
                        artifact_id=item.id,
                        uri=item.uri,
                        display_name=item.display_name,
                        format=item.format,
                        mime_type=item.mime_type,
                        role=item.role,
                        is_primary=item.id == primary_artifact.id,
                    )
                    for item in ordered
                ],
            )

        if current_run.deliverable_manifest == manifest:
            run.deliverable_manifest = manifest
            return current_run

        updated_run = current_run.model_copy(update={"deliverable_manifest": manifest})
        self._run_store.update_run(updated_run)
        run.deliverable_manifest = manifest
        return updated_run

    def _promote_primary_deliverable_if_missing(self, run: RunRecord) -> RunRecord:
        current_run = self._rebuild_run_deliverable_manifest(run=run)
        if current_run.deliverable_manifest.primary_artifact_id:
            return current_run

        candidate = self._select_best_primary_artifact(
            self._run_store.list_artifacts(run.id),
            allow_candidate_promotion=True,
        )
        if candidate is None:
            return current_run

        updated_candidate = candidate.model_copy(
            update={
                "is_final": True,
                "role": (
                    candidate.role
                    if candidate.role not in {"", "intermediate_file", "supporting_output"}
                    else "final_deliverable"
                ),
            }
        )
        self._run_store.update_artifact(updated_candidate)
        return self._rebuild_run_deliverable_manifest(run=current_run)

    def _maybe_update_run_deliverable_manifest(
        self,
        *,
        run: RunRecord,
        artifact: ArtifactRecord,
    ) -> None:
        if not self._artifact_counts_as_deliverable(artifact):
            return
        self._rebuild_run_deliverable_manifest(run=run)

    @staticmethod
    def _normalize_goal_key(goal: str) -> str:
        normalized = re.sub(r"\s+", " ", str(goal or "").strip().lower())
        return normalized

    @staticmethod
    def _dedupe_strings(items: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            normalized.append(text)
            seen.add(text)
        return normalized

    def _collect_child_run_ids(self, root_run_id: str) -> list[str]:
        pending = [root_run_id]
        collected: list[str] = []
        seen: set[str] = {root_run_id}
        while pending:
            current_id = pending.pop(0)
            for child in self._run_store.list_runs(parent_run_id=current_id):
                if child.id in seen:
                    continue
                seen.add(child.id)
                collected.append(child.id)
                pending.append(child.id)
        return collected

    def _build_learned_workflow_skill_markdown(
        self,
        *,
        skill_name: str,
        title: str,
        summary: str,
        description: str,
        tool_names: list[str],
        step_titles: list[str],
        source_run_ids: list[str],
    ) -> str:
        lines = [
            "---",
            f"name: {skill_name}",
            f"description: {summary or title}",
            "---",
            "",
            f"# {title}",
            "",
        ]
        if summary:
            lines.extend([summary, ""])
        lines.extend(
            [
                "## When To Use",
                description or "Use this workflow when the task matches the original successful run.",
                "",
            ]
        )
        if step_titles:
            lines.append("## Workflow")
            for index, step_title in enumerate(step_titles, start=1):
                lines.append(f"{index}. {step_title}")
            lines.append("")
        if tool_names:
            lines.append("## Common Tools")
            for tool_name in tool_names:
                lines.append(f"- `{tool_name}`")
            lines.append("")
        lines.append("## Provenance")
        for run_id in source_run_ids:
            lines.append(f"- Source run: `{run_id}`")
        lines.append("")
        return "\n".join(lines)

    def _maybe_capture_learned_workflow_candidate(self, run: RunRecord) -> None:
        if (
            not self._enable_learned_workflow_generation
            or self._learned_workflow_store is None
            or run.parent_run_id is not None
        ):
            return

        steps = self._run_store.list_steps(run.id, account_id=run.account_id)
        artifacts = self._run_store.list_artifacts(run.id, account_id=run.account_id)
        child_run_ids = self._collect_child_run_ids(run.id)
        significant_steps = [
            step
            for step in steps
            if step.step_type in {"tool_call", "delegate", "approval_wait"}
        ]
        completion_step = next(
            (step for step in reversed(steps) if step.step_type == "completion"),
            None,
        )
        tool_names = self._dedupe_strings(
            [step.title for step in steps if step.step_type == "tool_call"]
        )
        step_titles = self._dedupe_strings(
            [
                step.title
                for step in steps
                if step.step_type in {"tool_call", "delegate", "approval_wait", "completion"}
            ]
        )
        signals: list[str] = []
        repeated_run_count = 0
        normalized_goal = self._normalize_goal_key(run.goal)
        if normalized_goal:
            for historical_run in self._run_store.list_runs(
                account_id=run.account_id,
                status="completed",
                limit=50,
            ):
                if historical_run.id == run.id or historical_run.parent_run_id is not None:
                    continue
                if self._normalize_goal_key(historical_run.goal) == normalized_goal:
                    repeated_run_count += 1
            if repeated_run_count > 0:
                signals.append("repeated_task_pattern")

        if (
            len(significant_steps) >= 2
            or bool(child_run_ids)
            or bool(run.deliverable_manifest.items)
        ):
            signals.append("successful_complex_run")

        if bool(run.run_metadata.get("user_endorsed_solution")):
            signals.append("user_endorsed_solution")

        signals = self._dedupe_strings(signals)
        if not signals:
            return

        completion_summary = ""
        if completion_step is not None:
            completion_summary = str(completion_step.output_summary or "").strip()
        artifact_names = self._dedupe_strings(
            [artifact.display_name or artifact.uri for artifact in artifacts if artifact.is_final]
        )
        title = run.goal.strip() or "Learned workflow candidate"
        summary_parts = []
        if completion_summary:
            summary_parts.append(completion_summary)
        if repeated_run_count > 0:
            summary_parts.append(f"Matched {repeated_run_count + 1} successful runs with the same goal pattern.")
        if artifact_names:
            summary_parts.append(f"Final outputs: {', '.join(artifact_names[:3])}.")
        summary = " ".join(summary_parts).strip() or title
        description_parts = [
            f"Goal pattern: {run.goal.strip() or title}.",
        ]
        if step_titles:
            description_parts.append(f"Key steps: {' -> '.join(step_titles[:6])}.")
        if child_run_ids:
            description_parts.append(f"Included delegated child runs: {', '.join(child_run_ids[:4])}.")
        description = " ".join(description_parts).strip()
        suggested_skill_name = self._learned_workflow_store.normalize_skill_name(
            run.goal,
            fallback=f"learned-workflow-{run.id[:8]}",
        )
        skill_markdown = self._build_learned_workflow_skill_markdown(
            skill_name=suggested_skill_name,
            title=title,
            summary=summary,
            description=description,
            tool_names=tool_names,
            step_titles=step_titles,
            source_run_ids=[run.id, *child_run_ids],
        )
        self._learned_workflow_store.upsert_candidate_for_run(
            account_id=run.account_id,
            run_id=run.id,
            session_id=run.session_id,
            agent_template_id=run.agent_template_id,
            title=title,
            summary=summary,
            description=description,
            signal_types=signals,
            source_run_ids=[run.id, *child_run_ids],
            tool_names=tool_names,
            step_titles=step_titles,
            artifact_ids=[artifact.id for artifact in artifacts],
            suggested_skill_name=suggested_skill_name,
            generated_skill_markdown=skill_markdown,
            metadata={
                "source_run_id": run.id,
                "root_run_id": run.run_metadata.get("root_run_id") or run.id,
                "completion_summary": completion_summary,
                "repeated_run_count": repeated_run_count + 1 if repeated_run_count else 0,
                "deliverable_count": len(run.deliverable_manifest.items),
                "significant_step_count": len(significant_steps),
            },
        )

    @staticmethod
    def _load_step_json(text: str) -> dict[str, object]:
        try:
            payload = json.loads(str(text or ""))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _extract_used_skill_names_from_steps(
        self,
        steps: list[RunStepRecord],
    ) -> list[str]:
        names: list[str] = []
        for step in steps:
            if step.step_type != "tool_call" or step.title != "get_skill":
                continue
            payload = self._load_step_json(step.input_summary)
            skill_name = str(payload.get("skill_name") or "").strip()
            if skill_name:
                names.append(skill_name)
        return self._dedupe_strings(names)

    def _extract_correction_note_texts(
        self,
        steps: list[RunStepRecord],
    ) -> list[str]:
        notes: list[str] = []
        for step in steps:
            if step.step_type != "tool_call" or step.title != "record_note":
                continue
            payload = self._load_step_json(step.input_summary)
            memory_type = str(
                payload.get("memory_type") or payload.get("category") or ""
            ).strip().lower()
            if memory_type != "correction":
                continue
            text = str(payload.get("summary") or payload.get("content") or "").strip()
            if text:
                notes.append(text)
        return self._dedupe_strings(notes)

    @staticmethod
    def _extract_run_failure_summary(
        run: RunRecord,
        steps: list[RunStepRecord],
    ) -> str:
        if run.error_summary.strip():
            return run.error_summary.strip()
        for step in reversed(steps):
            if step.step_type == "failure" and step.error_summary.strip():
                return step.error_summary.strip()
        return ""

    @staticmethod
    def _extract_run_completion_summary(steps: list[RunStepRecord]) -> str:
        for step in reversed(steps):
            if step.step_type == "completion" and step.output_summary.strip():
                return step.output_summary.strip()
        return ""

    def _extract_manual_skill_improvement_note(
        self,
        run: RunRecord,
        skill_name: str,
    ) -> tuple[bool, str]:
        raw_targets = run.run_metadata.get("skill_improvement_targets")
        targets: list[str]
        if isinstance(raw_targets, (list, tuple, set)):
            targets = self._dedupe_strings([str(item) for item in raw_targets])
        elif isinstance(raw_targets, str):
            targets = self._dedupe_strings([raw_targets])
        else:
            targets = []

        note = ""
        for key in ("skill_improvement_notes", "skill_improvement_note"):
            raw_note = run.run_metadata.get(key)
            if isinstance(raw_note, dict):
                note = str(raw_note.get(skill_name) or raw_note.get("*") or "").strip()
            elif isinstance(raw_note, str):
                note = raw_note.strip()
            if note:
                break

        flagged = run.status == "completed" and (
            skill_name in targets or (not targets and bool(note))
        )
        return flagged, note

    def _resolve_installed_skill_file(
        self,
        run: RunRecord,
        skill_name: str,
    ) -> Path | None:
        template = self._agent_store.get_agent_template_record(
            run.agent_template_id,
            account_id=run.account_id,
        )
        if template is None or template.is_system:
            return None
        skill_file = (
            self._agent_store.get_agent_skills_dir(
                template.id,
                account_id=template.account_id,
                is_system=template.is_system,
            )
            / skill_name
            / "SKILL.md"
        )
        if not skill_file.exists():
            return None
        return skill_file.resolve()

    def _collect_skill_history_signals(
        self,
        run: RunRecord,
        *,
        skill_name: str,
        current_steps: list[RunStepRecord],
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        correction_run_ids: list[str] = []
        correction_notes: list[str] = []
        failure_run_ids: list[str] = []
        failure_summaries: list[str] = []

        current_corrections = self._extract_correction_note_texts(current_steps)
        if current_corrections:
            correction_run_ids.append(run.id)
            correction_notes.extend(current_corrections[:2])

        current_failure_summary = self._extract_run_failure_summary(run, current_steps)
        if run.status == "failed":
            failure_run_ids.append(run.id)
            if current_failure_summary:
                failure_summaries.append(current_failure_summary)

        for historical_run in self._run_store.list_runs(account_id=run.account_id, limit=60):
            if historical_run.id == run.id or historical_run.parent_run_id is not None:
                continue
            historical_steps = self._run_store.list_steps(
                historical_run.id,
                account_id=run.account_id,
            )
            if skill_name not in self._extract_used_skill_names_from_steps(historical_steps):
                continue
            historical_corrections = self._extract_correction_note_texts(historical_steps)
            if historical_corrections:
                correction_run_ids.append(historical_run.id)
                correction_notes.extend(historical_corrections[:1])
            if historical_run.status == "failed":
                failure_run_ids.append(historical_run.id)
                summary = self._extract_run_failure_summary(historical_run, historical_steps)
                if summary:
                    failure_summaries.append(summary)

        return (
            self._dedupe_strings(correction_run_ids),
            self._dedupe_strings(correction_notes),
            self._dedupe_strings(failure_run_ids),
            self._dedupe_strings(failure_summaries),
        )

    def _maybe_capture_skill_improvement_proposals(self, run: RunRecord) -> None:
        if self._skill_improvement_store is None or run.parent_run_id is not None:
            return

        steps = self._run_store.list_steps(run.id, account_id=run.account_id)
        used_skills = self._extract_used_skill_names_from_steps(steps)
        if not used_skills:
            return

        completion_summary = self._extract_run_completion_summary(steps)
        for skill_name in used_skills:
            skill_file = self._resolve_installed_skill_file(run, skill_name)
            if skill_file is None:
                continue

            manual_flagged, manual_note = self._extract_manual_skill_improvement_note(
                run,
                skill_name,
            )
            (
                correction_run_ids,
                correction_notes,
                failure_run_ids,
                failure_summaries,
            ) = self._collect_skill_history_signals(
                run,
                skill_name=skill_name,
                current_steps=steps,
            )

            signal_types: list[str] = []
            if len(correction_run_ids) >= 2:
                signal_types.append("repeated_user_corrections")
            if len(failure_run_ids) >= 2:
                signal_types.append("repeated_run_failures")
            if manual_flagged:
                signal_types.append("manual_successful_refinement")
            signal_types = self._dedupe_strings(signal_types)
            if not signal_types:
                continue

            guidance_items: list[str] = []
            if manual_note:
                guidance_items.append(manual_note)
            if correction_notes:
                guidance_items.append(
                    f"Incorporate repeated correction feedback: {correction_notes[0]}"
                )
            if failure_summaries:
                guidance_items.append(
                    f"Document the recurring failure mode and its guardrail: {failure_summaries[0]}"
                )
            if completion_summary:
                guidance_items.append(
                    f"Preserve the successful refinement outcome: {completion_summary}"
                )

            summary = (
                manual_note
                or (
                    f"Refine `{skill_name}` to absorb repeated user corrections."
                    if correction_notes
                    else ""
                )
                or (
                    f"Refine `{skill_name}` to guard against repeated failures."
                    if failure_summaries
                    else ""
                )
                or f"Refine `{skill_name}` based on recent usage feedback."
            )
            source_run_ids = self._dedupe_strings(
                [run.id, *correction_run_ids, *failure_run_ids]
            )
            current_markdown = skill_file.read_text(encoding="utf-8")
            (
                base_version,
                proposed_version,
                proposed_markdown,
                changelog_entry,
            ) = build_skill_improvement_payload(
                current_markdown=current_markdown,
                skill_name=skill_name,
                summary=summary,
                signal_types=signal_types,
                guidance_items=guidance_items,
                source_run_ids=source_run_ids,
            )
            self._skill_improvement_store.upsert_proposal_for_skill(
                account_id=run.account_id,
                run_id=run.id,
                session_id=run.session_id,
                agent_template_id=run.agent_template_id,
                skill_name=skill_name,
                target_skill_path=str(skill_file),
                title=f"Improve skill: {skill_name}",
                summary=summary,
                signal_types=signal_types,
                source_run_ids=source_run_ids,
                base_version=base_version,
                proposed_version=proposed_version,
                current_skill_markdown=current_markdown,
                proposed_skill_markdown=proposed_markdown,
                changelog_entry=changelog_entry,
                metadata={
                    "current_run_status": run.status,
                    "current_completion_summary": completion_summary,
                    "current_failure_summary": self._extract_run_failure_summary(run, steps),
                    "correction_run_count": len(correction_run_ids),
                    "failure_run_count": len(failure_run_ids),
                    "manual_note": manual_note,
                },
            )

    def _append_state_history(
        self,
        *,
        state: RunExecutionState,
        run: RunRecord,
        event: dict,
        trace_event_type: str | None = None,
    ) -> None:
        """Append one event to in-memory history before any subscriber is attached."""
        sequence = state.next_event_sequence
        state.next_event_sequence += 1
        state.history.append((sequence, event))
        self._record_trace(
            state=state,
            run=run,
            event_type=trace_event_type or str(event.get("type", "event")),
            status=run.status,
            payload_summary=self._summarize_event(event),
        )

    def _complete_state(self, state: RunExecutionState) -> None:
        state.completed = True
        for subscriber in list(state.subscribers):
            subscriber.put_nowait((None, None))
