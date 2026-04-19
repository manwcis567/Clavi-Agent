"""Helpers for trace normalization, tree building, and diagnostics exports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .run_models import (
    ApprovalRequestRecord,
    ArtifactRecord,
    RunRecord,
    RunStepRecord,
    TraceEventRecord,
)
from .run_store import RunStore
from .sqlite_schema import utc_now_iso


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _duration_ms(started_at: str | None, finished_at: str | None) -> int | None:
    started = _parse_iso_datetime(started_at)
    finished = _parse_iso_datetime(finished_at)
    if started is None or finished is None:
        return None
    return max(0, int((finished - started).total_seconds() * 1000))


def _average_int(values: list[int]) -> int:
    if not values:
        return 0
    return int(round(sum(values) / len(values)))


def _average_float(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _event_category(event_type: str) -> str:
    if event_type.startswith("llm_"):
        return "llm"
    if event_type.startswith("tool_"):
        return "tool"
    if event_type.startswith("delegate_"):
        return "delegate"
    if event_type.startswith("run_"):
        return "run_lifecycle"
    if event_type == "checkpoint_saved":
        return "checkpoint"
    if event_type in {"content", "content_delta", "thinking", "thinking_delta"}:
        return "message"
    if event_type in {"done", "error", "queued", "cancelled", "interrupted", "timed_out"}:
        return "state"
    return "event"


def _peak_parallel_run_count(runs: list[RunRecord]) -> int:
    timeline_points: list[tuple[datetime, int]] = []
    for run in runs:
        started = _parse_iso_datetime(run.started_at or run.created_at)
        finished = _parse_iso_datetime(run.finished_at or run.started_at or run.created_at)
        if started is None or finished is None:
            continue
        timeline_points.append((started, 1))
        timeline_points.append((finished, -1))

    if not timeline_points:
        return 0

    timeline_points.sort(key=lambda item: (item[0], -item[1]))
    active = 0
    peak = 0
    for _, delta in timeline_points:
        active = max(0, active + delta)
        peak = max(peak, active)
    return peak


def _load_payload(payload_summary: str) -> tuple[object, dict[str, Any], dict[str, Any]]:
    raw = payload_summary.strip()
    if not raw:
        return "", {}, {}

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw, {}, {}

    if not isinstance(payload, dict):
        return payload, {}, {}

    context = payload.get("context", {})
    if not isinstance(context, dict):
        context = {}

    data: dict[str, Any]
    if isinstance(payload.get("data"), dict):
        data = dict(payload["data"])
    else:
        data = {key: value for key, value in payload.items() if key != "context"}

    return payload, context, data


def _run_root_id(run: RunRecord) -> str:
    root_run_id = run.run_metadata.get("root_run_id")
    if isinstance(root_run_id, str) and root_run_id.strip():
        return root_run_id.strip()
    return run.id if run.parent_run_id is None else run.parent_run_id


def _run_depth(run: RunRecord) -> int:
    if "depth" in run.run_metadata:
        return max(0, _coerce_int(run.run_metadata.get("depth"), 0))
    return 0 if run.parent_run_id is None else 1


def _run_agent_name(run: RunRecord) -> str:
    agent_name = run.run_metadata.get("agent_name")
    if isinstance(agent_name, str) and agent_name.strip():
        return agent_name.strip()
    return "main" if run.parent_run_id is None else "worker"


def _run_kind(run: RunRecord) -> str:
    kind = run.run_metadata.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    return "root" if run.parent_run_id is None else "child"


def find_root_run(run_store: RunStore, run_id: str) -> RunRecord | None:
    """Return the root run for the requested run id."""
    run = run_store.get_run(run_id)
    if run is None:
        return None

    if run.parent_run_id is None:
        return run

    root_run_id = run.run_metadata.get("root_run_id")
    if isinstance(root_run_id, str) and root_run_id.strip():
        root_run = run_store.get_run(root_run_id.strip())
        if root_run is not None:
            return root_run

    current = run
    while current.parent_run_id is not None:
        parent = run_store.get_run(current.parent_run_id)
        if parent is None:
            break
        current = parent
    return current


def collect_run_family(run_store: RunStore, run_id: str) -> tuple[RunRecord | None, list[RunRecord]]:
    """Collect one run together with all descendants in tree order."""
    root_run = find_root_run(run_store, run_id)
    if root_run is None:
        return None, []

    ordered_runs: list[RunRecord] = []
    queue: list[RunRecord] = [root_run]
    while queue:
        current = queue.pop(0)
        ordered_runs.append(current)
        children = sorted(
            run_store.list_runs(parent_run_id=current.id),
            key=lambda item: (item.created_at, item.id),
        )
        queue.extend(children)
    return root_run, ordered_runs


def normalize_trace_event(event: TraceEventRecord, run: RunRecord) -> dict[str, Any]:
    """Expand one persisted trace event into a stable API payload."""
    payload, context, data = _load_payload(event.payload_summary)
    tool_call_id = str(data.get("tool_call_id") or data.get("id") or "").strip() or None
    tool_name = str(data.get("name") or data.get("tool_name") or "").strip() or None
    return {
        "id": event.id,
        "run_id": event.run_id,
        "root_run_id": _run_root_id(run),
        "parent_run_id": event.parent_run_id,
        "step_id": event.step_id,
        "sequence": event.sequence,
        "timestamp": event.created_at,
        "event_type": event.event_type,
        "event_category": _event_category(event.event_type),
        "status": event.status,
        "payload_summary": event.payload_summary,
        "payload": payload,
        "context": context,
        "data": data,
        "duration_ms": event.duration_ms,
        "agent_name": str(context.get("agent_name") or data.get("agent_name") or _run_agent_name(run)),
        "run_goal": run.goal,
        "run_kind": _run_kind(run),
        "run_depth": _run_depth(run),
        "is_main_agent": bool(
            context.get("is_main_agent") if "is_main_agent" in context else run.parent_run_id is None
        ),
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
        "step_number": _coerce_int(data.get("step"), 0) or None,
        "active_step_id": str(data.get("active_step_id", "")).strip() or None,
    }


def build_trace_timeline(
    runs: list[RunRecord],
    trace_by_run_id: dict[str, list[TraceEventRecord]],
) -> list[dict[str, Any]]:
    """Return a normalized timeline across the root run and all descendants."""
    events: list[dict[str, Any]] = []
    for run in runs:
        for event in trace_by_run_id.get(run.id, []):
            events.append(normalize_trace_event(event, run))

    return sorted(
        events,
        key=lambda item: (
            str(item["timestamp"]),
            -_coerce_int(item["run_depth"], 0),
            str(item["run_id"]),
            _coerce_int(item["sequence"], 0),
        ),
    )


def build_run_tree(
    root_run: RunRecord,
    runs: list[RunRecord],
    trace_by_run_id: dict[str, list[TraceEventRecord]],
    artifacts_by_run_id: dict[str, list[ArtifactRecord]],
) -> dict[str, Any]:
    """Build a nested run tree annotated with trace and artifact counts."""
    children_by_parent: dict[str | None, list[RunRecord]] = {}
    for run in runs:
        children_by_parent.setdefault(run.parent_run_id, []).append(run)

    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: (item.created_at, item.id))

    def build_node(run: RunRecord) -> dict[str, Any]:
        trace_events = trace_by_run_id.get(run.id, [])
        latest_trace = trace_events[-1] if trace_events else None
        children = [build_node(child) for child in children_by_parent.get(run.id, [])]
        return {
            "id": run.id,
            "parent_run_id": run.parent_run_id,
            "root_run_id": _run_root_id(run),
            "goal": run.goal,
            "status": run.status,
            "agent_name": _run_agent_name(run),
            "run_kind": _run_kind(run),
            "depth": _run_depth(run),
            "created_at": run.created_at,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "error_summary": run.error_summary,
            "trace_event_count": len(trace_events),
            "artifact_count": len(artifacts_by_run_id.get(run.id, [])),
            "latest_trace_event": (
                {
                    "event_type": latest_trace.event_type,
                    "status": latest_trace.status,
                    "timestamp": latest_trace.created_at,
                    "sequence": latest_trace.sequence,
                }
                if latest_trace is not None
                else None
            ),
            "children": children,
        }

    return build_node(root_run)


def build_tool_call_drilldown(
    runs: list[RunRecord],
    trace_by_run_id: dict[str, list[TraceEventRecord]],
    steps_by_run_id: dict[str, list[RunStepRecord]],
) -> list[dict[str, Any]]:
    """Merge tool/delegate start and finish events into one drill-down list."""
    step_titles_by_run: dict[str, set[str]] = {
        run_id: {
            step.title
            for step in steps
            if step.step_type in {"tool_call", "delegate"}
        }
        for run_id, steps in steps_by_run_id.items()
    }
    indexed_runs = {run.id: run for run in runs}
    pending: dict[tuple[str, str], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, str]] = []

    for run in runs:
        normalized_events = [
            normalize_trace_event(event, run)
            for event in trace_by_run_id.get(run.id, [])
            if event.event_type in {"tool_started", "tool_finished", "delegate_started", "delegate_finished"}
        ]
        for event in normalized_events:
            tool_call_id = str(event.get("tool_call_id") or event["id"])
            key = (run.id, tool_call_id)
            data = event.get("data", {})
            if not isinstance(data, dict):
                data = {}

            entry = pending.get(key)
            if entry is None:
                ordered_keys.append(key)
                entry = {
                    "run_id": run.id,
                    "root_run_id": _run_root_id(run),
                    "parent_run_id": run.parent_run_id,
                    "agent_name": event["agent_name"],
                    "run_depth": event["run_depth"],
                    "event_category": event["event_category"],
                    "tool_call_id": event.get("tool_call_id"),
                    "tool_name": event.get("tool_name"),
                    "tool_class": str(data.get("tool_class", "")).strip(),
                    "parameter_summary": str(data.get("parameter_summary", "")).strip(),
                    "risk_category": str(data.get("risk_category", "")).strip(),
                    "risk_level": str(data.get("risk_level", "")).strip(),
                    "requires_approval": bool(data.get("requires_approval")),
                    "impact_summary": str(data.get("impact_summary", "")).strip(),
                    "started_at": None,
                    "finished_at": None,
                    "duration_ms": None,
                    "success": None,
                    "content": "",
                    "error": "",
                    "artifacts": [],
                    "step_number": event.get("step_number"),
                    "matches_run_step": bool(
                        event.get("tool_name") and event["tool_name"] in step_titles_by_run.get(run.id, set())
                    ),
                }
                pending[key] = entry

            if event["event_type"].endswith("_started"):
                entry["started_at"] = event["timestamp"]
            if event["event_type"].endswith("_finished"):
                entry["finished_at"] = event["timestamp"]
                entry["duration_ms"] = event["duration_ms"]
                entry["success"] = data.get("success")
                entry["content"] = str(data.get("content", "") or "")
                entry["error"] = str(data.get("error", "") or "")
                artifacts = data.get("artifacts", [])
                if isinstance(artifacts, list):
                    entry["artifacts"] = artifacts

    drilldown = [pending[key] for key in ordered_keys]
    return sorted(
        drilldown,
        key=lambda item: (
            str(item.get("started_at") or item.get("finished_at") or ""),
            _coerce_int(item.get("run_depth"), 0),
            str(item.get("run_id", "")),
            str(item.get("tool_call_id") or item.get("tool_name") or ""),
        ),
    )


def build_trace_export(
    *,
    root_run: RunRecord,
    runs: list[RunRecord],
    timeline: list[dict[str, Any]],
    run_tree: dict[str, Any],
    tool_calls: list[dict[str, Any]],
    steps_by_run_id: dict[str, list[RunStepRecord]],
    artifacts_by_run_id: dict[str, list[ArtifactRecord]],
) -> dict[str, Any]:
    """Build one export payload for trace diagnostics."""
    return {
        "exported_at": utc_now_iso(),
        "root_run": root_run.model_dump(mode="json"),
        "runs": [run.model_dump(mode="json") for run in runs],
        "timeline": timeline,
        "tree": run_tree,
        "tool_calls": tool_calls,
        "steps": {
            run_id: [step.model_dump(mode="json") for step in steps]
            for run_id, steps in steps_by_run_id.items()
        },
        "artifacts": {
            run_id: [artifact.model_dump(mode="json") for artifact in artifacts]
            for run_id, artifacts in artifacts_by_run_id.items()
        },
        "summary": {
            "run_count": len(runs),
            "trace_event_count": len(timeline),
            "tool_call_count": len(tool_calls),
            "artifact_count": sum(len(items) for items in artifacts_by_run_id.values()),
        },
    }


def _build_replay_title(event: dict[str, Any]) -> str:
    event_type = str(event.get("event_type") or "")
    tool_name = str(event.get("tool_name") or "").strip()

    if event_type == "run_started":
        return "Run started"
    if event_type == "run_completed":
        return "Run completed"
    if event_type == "run_failed":
        return "Run failed"
    if event_type == "run_resumed":
        return "Run resumed"
    if event_type == "run_cancelled":
        return "Run cancelled"
    if event_type == "run_timed_out":
        return "Run timed out"
    if event_type == "llm_request":
        return "LLM request"
    if event_type == "llm_response":
        return "LLM response"
    if event_type == "checkpoint_saved":
        return "Checkpoint saved"
    if event_type == "tool_started":
        return f"Tool started: {tool_name or 'unknown tool'}"
    if event_type == "tool_finished":
        return f"Tool finished: {tool_name or 'unknown tool'}"
    if event_type == "delegate_started":
        return f"Delegate started: {tool_name or 'delegate'}"
    if event_type == "delegate_finished":
        return f"Delegate finished: {tool_name or 'delegate'}"
    if event_type == "delegate_reviewed":
        return "Delegate reviewed"
    if event_type in {"content", "content_delta", "thinking", "thinking_delta"}:
        return event_type.replace("_", " ").title()
    return event_type.replace("_", " ").title() or "Trace event"


def _build_replay_summary(event: dict[str, Any]) -> str:
    data = event.get("data", {})
    if not isinstance(data, dict):
        data = {}

    for key in (
        "message",
        "content",
        "delta",
        "parameter_summary",
        "impact_summary",
        "error",
        "final_response",
        "rejection_reason",
        "action",
    ):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    payload = event.get("payload")
    if isinstance(payload, str) and payload.strip():
        return payload.strip()

    return ""


def build_trace_replay(
    *,
    requested_run_id: str,
    root_run: RunRecord,
    runs: list[RunRecord],
    timeline: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    artifacts_by_run_id: dict[str, list[ArtifactRecord]],
) -> dict[str, Any]:
    """Build a developer-friendly replay payload for one run tree."""
    base_timestamp = _parse_iso_datetime(
        str(timeline[0].get("timestamp")) if timeline else None
    )
    end_timestamp = _parse_iso_datetime(
        str(timeline[-1].get("timestamp")) if timeline else None
    )
    event_type_counts: dict[str, int] = {}
    tool_call_index = {
        (
            str(item.get("run_id") or ""),
            str(item.get("tool_call_id") or item.get("tool_name") or ""),
        ): item
        for item in tool_calls
    }
    frames: list[dict[str, Any]] = []

    for frame_index, event in enumerate(timeline):
        timestamp = _parse_iso_datetime(str(event.get("timestamp") or ""))
        relative_ms = None
        if base_timestamp is not None and timestamp is not None:
            relative_ms = max(0, int((timestamp - base_timestamp).total_seconds() * 1000))

        event_type = str(event.get("event_type") or "")
        event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1
        tool_lookup_key = (
            str(event.get("run_id") or ""),
            str(event.get("tool_call_id") or event.get("tool_name") or ""),
        )
        related_tool_call = tool_call_index.get(tool_lookup_key)

        frames.append(
            {
                "frame_index": frame_index,
                "timestamp": event.get("timestamp"),
                "relative_ms": relative_ms,
                "title": _build_replay_title(event),
                "summary": _build_replay_summary(event),
                "event_type": event_type,
                "event_category": event.get("event_category"),
                "status": event.get("status"),
                "run_id": event.get("run_id"),
                "root_run_id": event.get("root_run_id"),
                "parent_run_id": event.get("parent_run_id"),
                "agent_name": event.get("agent_name"),
                "run_depth": event.get("run_depth"),
                "run_kind": event.get("run_kind"),
                "tool_name": event.get("tool_name"),
                "tool_call_id": event.get("tool_call_id"),
                "step_id": event.get("step_id"),
                "step_number": event.get("step_number"),
                "duration_ms": event.get("duration_ms"),
                "data": event.get("data"),
                "tool_call": related_tool_call,
            }
        )

    playback_duration_ms = None
    if base_timestamp is not None and end_timestamp is not None:
        playback_duration_ms = max(0, int((end_timestamp - base_timestamp).total_seconds() * 1000))

    return {
        "generated_at": utc_now_iso(),
        "requested_run_id": requested_run_id,
        "root_run_id": root_run.id,
        "session_id": root_run.session_id,
        "playback": {
            "started_at": timeline[0].get("timestamp") if timeline else None,
            "finished_at": timeline[-1].get("timestamp") if timeline else None,
            "duration_ms": playback_duration_ms,
        },
        "summary": {
            "run_count": len(runs),
            "frame_count": len(frames),
            "tool_call_count": len(tool_calls),
            "artifact_count": sum(len(items) for items in artifacts_by_run_id.values()),
            "event_type_counts": dict(sorted(event_type_counts.items())),
        },
        "runs": [
            {
                "id": run.id,
                "parent_run_id": run.parent_run_id,
                "status": run.status,
                "goal": run.goal,
                "agent_name": _run_agent_name(run),
                "run_kind": _run_kind(run),
                "depth": _run_depth(run),
                "created_at": run.created_at,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
            }
            for run in runs
        ],
        "frames": frames,
    }


def build_run_metrics(
    *,
    root_runs: list[RunRecord],
    family_runs_by_root_id: dict[str, list[RunRecord]],
    trace_by_run_id: dict[str, list[TraceEventRecord]],
    approvals_by_run_id: dict[str, list[ApprovalRequestRecord]],
    session_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate durable-run metrics for product diagnostics and monitoring."""
    status_counts: dict[str, int] = {}
    failure_type_distribution: dict[str, int] = {}
    delegate_review_action_counts: dict[str, int] = {}
    llm_call_role_counts: dict[str, int] = {}
    duration_values: list[int] = []
    tool_call_counts: list[int] = []
    delegate_counts: list[int] = []
    batch_delegate_counts: list[int] = []
    child_run_counts: list[int] = []
    parallel_child_run_values: list[int] = []
    approval_wait_values: list[int] = []
    approval_request_count = 0
    pending_approval_count = 0
    completed_run_count = 0
    terminal_run_count = 0
    total_run_count = 0
    child_run_count = 0
    batch_delegate_call_count = 0
    forbidden_main_tool_attempt_count = 0
    reviewed_root_run_count = 0
    delegate_review_count = 0
    worker_first_pass_accept_count = 0
    worker_rework_count = 0
    total_llm_call_count = 0
    planner_llm_call_count = 0
    worker_llm_call_count = 0

    for root_run in root_runs:
        family_runs = family_runs_by_root_id.get(root_run.id, [root_run])
        child_runs = [run for run in family_runs if run.parent_run_id is not None]
        total_run_count += len(family_runs)
        child_run_count += len(child_runs)
        status_counts[root_run.status] = status_counts.get(root_run.status, 0) + 1
        child_run_counts.append(len(child_runs))
        parallel_child_run_values.append(_peak_parallel_run_count(child_runs))

        if root_run.status == "completed":
            completed_run_count += 1
        if root_run.status in {"completed", "failed", "cancelled", "timed_out"}:
            terminal_run_count += 1
        if root_run.status in {"failed", "cancelled", "timed_out", "interrupted"}:
            failure_type_distribution[root_run.status] = (
                failure_type_distribution.get(root_run.status, 0) + 1
            )

        run_duration_ms = _duration_ms(root_run.started_at, root_run.finished_at)
        if run_duration_ms is not None:
            duration_values.append(run_duration_ms)

        family_tool_calls = 0
        family_delegate_calls = 0
        family_batch_delegate_calls = 0
        family_review_actions: list[tuple[str, int, str]] = []
        family_rework_required = False
        for run in family_runs:
            for event in trace_by_run_id.get(run.id, []):
                _, context, data = _load_payload(event.payload_summary)
                is_main_agent = bool(
                    context.get("is_main_agent")
                    if "is_main_agent" in context
                    else run.parent_run_id is None
                )
                if event.event_type == "tool_started":
                    family_tool_calls += 1
                elif event.event_type == "delegate_started":
                    family_delegate_calls += 1
                    tool_name = str(data.get("name") or data.get("tool_name") or "").strip()
                    if tool_name == "delegate_tasks":
                        family_batch_delegate_calls += 1
                elif event.event_type == "tool_finished":
                    if not bool(data.get("policy_allowed", True)) and is_main_agent:
                        forbidden_main_tool_attempt_count += 1
                elif event.event_type == "llm_request":
                    role = "planner" if is_main_agent else "worker"
                    total_llm_call_count += 1
                    llm_call_role_counts[role] = llm_call_role_counts.get(role, 0) + 1
                    if is_main_agent:
                        planner_llm_call_count += 1
                    else:
                        worker_llm_call_count += 1
                elif event.event_type == "delegate_reviewed":
                    action = str(data.get("action") or event.status or "").strip() or "unknown"
                    delegate_review_count += 1
                    delegate_review_action_counts[action] = (
                        delegate_review_action_counts.get(action, 0) + 1
                    )
                    family_review_actions.append((event.created_at, event.sequence, action))
                    if action in {"retry_delegated", "needs_followup", "rejected"}:
                        family_rework_required = True

            for approval in approvals_by_run_id.get(run.id, []):
                approval_request_count += 1
                if approval.status == "pending":
                    pending_approval_count += 1
                wait_duration_ms = _duration_ms(approval.requested_at, approval.resolved_at)
                if wait_duration_ms is not None:
                    approval_wait_values.append(wait_duration_ms)

        tool_call_counts.append(family_tool_calls)
        delegate_counts.append(family_delegate_calls)
        batch_delegate_counts.append(family_batch_delegate_calls)
        batch_delegate_call_count += family_batch_delegate_calls

        if family_review_actions:
            family_review_actions.sort(key=lambda item: (item[0], item[1]))
            reviewed_root_run_count += 1
            if family_review_actions[0][2] == "accepted":
                worker_first_pass_accept_count += 1
            if family_rework_required:
                worker_rework_count += 1

    success_rate = _rate(completed_run_count, terminal_run_count)

    return {
        "generated_at": utc_now_iso(),
        "filters": {
            "session_id": session_id,
            "scope": "root_runs",
        },
        "summary": {
            "root_run_count": len(root_runs),
            "total_run_count": total_run_count,
            "child_run_count": child_run_count,
            "terminal_run_count": terminal_run_count,
            "completed_run_count": completed_run_count,
            "approval_request_count": approval_request_count,
            "pending_approval_count": pending_approval_count,
            "success_rate": success_rate,
            "average_duration_ms": _average_int(duration_values),
            "average_tool_call_count": _average_float(tool_call_counts),
            "average_delegate_count": _average_float(delegate_counts),
            "average_batch_delegate_count": _average_float(batch_delegate_counts),
            "average_approval_wait_ms": _average_int(approval_wait_values),
            "batch_delegate_call_count": batch_delegate_call_count,
            "delegate_batch_usage_rate": _rate(batch_delegate_call_count, sum(delegate_counts)),
            "forbidden_main_tool_attempt_count": forbidden_main_tool_attempt_count,
            "reviewed_root_run_count": reviewed_root_run_count,
            "delegate_review_count": delegate_review_count,
            "worker_first_pass_acceptance_rate": _rate(
                worker_first_pass_accept_count,
                reviewed_root_run_count,
            ),
            "worker_rework_rate": _rate(worker_rework_count, reviewed_root_run_count),
            "total_llm_call_count": total_llm_call_count,
            "planner_llm_call_count": planner_llm_call_count,
            "worker_llm_call_count": worker_llm_call_count,
            "planner_llm_call_share": _rate(planner_llm_call_count, total_llm_call_count),
            "worker_llm_call_share": _rate(worker_llm_call_count, total_llm_call_count),
            "average_child_run_count": _average_float(child_run_counts),
            "average_parallel_child_runs": _average_float(parallel_child_run_values),
            "duration_sample_count": len(duration_values),
            "approval_wait_sample_count": len(approval_wait_values),
        },
        "status_counts": dict(sorted(status_counts.items())),
        "failure_type_distribution": dict(sorted(failure_type_distribution.items())),
        "delegate_review_action_counts": dict(sorted(delegate_review_action_counts.items())),
        "llm_call_role_counts": dict(sorted(llm_call_role_counts.items())),
    }


def build_run_metrics_export(metrics: dict[str, Any]) -> str:
    """Serialize run metrics into a Prometheus-friendly text format."""
    summary = metrics.get("summary", {})
    status_counts = metrics.get("status_counts", {})
    failure_type_distribution = metrics.get("failure_type_distribution", {})
    filters = metrics.get("filters", {})
    session_id = filters.get("session_id")

    def escape_label_value(value: object) -> str:
        """Escape label values for Prometheus exposition format."""
        return str(value).replace("\\", "\\\\").replace('"', '\\"')

    def render_labels(extra_labels: dict[str, object]) -> str:
        labels = {
            key: str(value)
            for key, value in extra_labels.items()
            if value is not None and str(value).strip()
        }
        if session_id:
            labels["session_id"] = str(session_id)
        if not labels:
            return ""
        serialized = ",".join(
            f'{key}="{escape_label_value(value)}"'
            for key, value in sorted(labels.items())
        )
        return "{" + serialized + "}"

    def append_metric(
        lines: list[str],
        metric_name: str,
        metric_type: str,
        metric_help: str,
        value: int | float,
        labels: dict[str, object] | None = None,
    ) -> None:
        lines.append(f"# HELP {metric_name} {metric_help}")
        lines.append(f"# TYPE {metric_name} {metric_type}")
        rendered_labels = render_labels(labels or {})
        lines.append(f"{metric_name}{rendered_labels} {value}")

    lines: list[str] = [
        f"# Generated at {metrics.get('generated_at', utc_now_iso())}",
        "# Scope: durable root-run metrics",
    ]

    append_metric(
        lines,
        "clavi_agent_root_runs_total",
        "gauge",
        "Number of analyzed root runs.",
        int(summary.get("root_run_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_total_runs_total",
        "gauge",
        "Number of analyzed runs including child runs.",
        int(summary.get("total_run_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_child_runs_total",
        "gauge",
        "Number of analyzed child runs.",
        int(summary.get("child_run_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_run_success_rate",
        "gauge",
        "Completed root runs divided by terminal root runs.",
        float(summary.get("success_rate", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_run_duration_ms_avg",
        "gauge",
        "Average root run duration in milliseconds.",
        int(summary.get("average_duration_ms", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_tool_calls_avg",
        "gauge",
        "Average attempted tool calls per root run family.",
        float(summary.get("average_tool_call_count", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_delegate_calls_avg",
        "gauge",
        "Average attempted delegate calls per root run family.",
        float(summary.get("average_delegate_count", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_delegate_tasks_calls_total",
        "gauge",
        "Number of delegate_tasks batch calls across analyzed root run families.",
        int(summary.get("batch_delegate_call_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_delegate_tasks_usage_rate",
        "gauge",
        "Share of delegate calls that used delegate_tasks batching.",
        float(summary.get("delegate_batch_usage_rate", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_approval_wait_ms_avg",
        "gauge",
        "Average resolved approval wait time in milliseconds.",
        int(summary.get("average_approval_wait_ms", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_approval_requests_total",
        "gauge",
        "Number of analyzed approval requests.",
        int(summary.get("approval_request_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_pending_approvals_total",
        "gauge",
        "Number of pending approval requests.",
        int(summary.get("pending_approval_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_main_forbidden_tool_attempts_total",
        "gauge",
        "Number of main-agent tool attempts blocked by delegation policy.",
        int(summary.get("forbidden_main_tool_attempt_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_worker_first_pass_acceptance_rate",
        "gauge",
        "Share of reviewed root runs whose first delegate review accepted the worker result.",
        float(summary.get("worker_first_pass_acceptance_rate", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_worker_rework_rate",
        "gauge",
        "Share of reviewed root runs that required follow-up or retry after worker review.",
        float(summary.get("worker_rework_rate", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_planner_llm_calls_total",
        "gauge",
        "Number of planner-role LLM requests across analyzed root run families.",
        int(summary.get("planner_llm_call_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_worker_llm_calls_total",
        "gauge",
        "Number of worker-role LLM requests across analyzed root run families.",
        int(summary.get("worker_llm_call_count", 0)),
    )
    append_metric(
        lines,
        "clavi_agent_planner_llm_call_share",
        "gauge",
        "Share of LLM requests attributed to planner-role runs.",
        float(summary.get("planner_llm_call_share", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_worker_llm_call_share",
        "gauge",
        "Share of LLM requests attributed to worker-role runs.",
        float(summary.get("worker_llm_call_share", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_child_runs_avg",
        "gauge",
        "Average number of child runs per analyzed root run.",
        float(summary.get("average_child_run_count", 0.0)),
    )
    append_metric(
        lines,
        "clavi_agent_parallel_child_runs_avg",
        "gauge",
        "Average peak concurrent child runs per analyzed root run.",
        float(summary.get("average_parallel_child_runs", 0.0)),
    )

    if status_counts:
        lines.append("# HELP clavi_agent_run_status_total Root run count grouped by status.")
        lines.append("# TYPE clavi_agent_run_status_total gauge")
        for status, count in sorted(status_counts.items()):
            lines.append(
                f"clavi_agent_run_status_total{render_labels({'status': status})} {int(count)}"
            )

    if failure_type_distribution:
        lines.append("# HELP clavi_agent_run_failure_type_total Root run count grouped by failure type.")
        lines.append("# TYPE clavi_agent_run_failure_type_total gauge")
        for failure_type, count in sorted(failure_type_distribution.items()):
            lines.append(
                "clavi_agent_run_failure_type_total"
                f"{render_labels({'failure_type': failure_type})} {int(count)}"
            )

    delegate_review_action_counts = metrics.get("delegate_review_action_counts", {})
    if delegate_review_action_counts:
        lines.append(
            "# HELP clavi_agent_delegate_review_action_total Delegate review action count grouped by action."
        )
        lines.append("# TYPE clavi_agent_delegate_review_action_total gauge")
        for action, count in sorted(delegate_review_action_counts.items()):
            lines.append(
                "clavi_agent_delegate_review_action_total"
                f"{render_labels({'action': action})} {int(count)}"
            )

    return "\n".join(lines) + "\n"


def find_run_log_files(log_dir: Path, runs: list[RunRecord]) -> list[str]:
    """Locate log files for one run tree by filename and legacy content fallback."""
    if not log_dir.exists():
        return []

    run_ids = {run.id for run in runs}
    matched: list[str] = []
    candidates = sorted(log_dir.glob("agent_run_*.log"))

    for path in candidates:
        if any(run_id in path.name for run_id in run_ids):
            matched.append(str(path.resolve()))

    if matched:
        return matched

    needles = [f'"run_id": "{run_id}"' for run_id in run_ids]
    for path in candidates:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(needle in content for needle in needles):
            matched.append(str(path.resolve()))
    return matched


def build_run_location_payload(
    *,
    root_run: RunRecord,
    runs: list[RunRecord],
    session_db_path: Path,
    log_dir: Path,
    timeline: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    artifacts_by_run_id: dict[str, list[ArtifactRecord]],
) -> dict[str, Any]:
    """Build one compact diagnostics payload for locating run artifacts on disk."""
    return {
        "generated_at": utc_now_iso(),
        "run_id": root_run.id,
        "session_id": root_run.session_id,
        "root_run_id": root_run.id,
        "database_path": str(session_db_path.resolve()),
        "log_directory": str(log_dir.resolve()),
        "log_files": find_run_log_files(log_dir, runs),
        "run_ids": [run.id for run in runs],
        "trace_event_count": len(timeline),
        "tool_call_count": len(tool_calls),
        "artifact_count": sum(len(items) for items in artifacts_by_run_id.values()),
    }

