"""Application service for scheduled task CRUD, dispatch, and observability."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .integration_store import DeliveryStore, IntegrationStore
from .scheduled_task_models import (
    ScheduledTaskExecutionRecord,
    ScheduledTaskRecord,
)
from .scheduled_task_store import ScheduledTaskStore
from .sqlite_schema import utc_now_iso


_MONTH_NAME_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_DOW_NAME_MAP = {
    "sun": 0,
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
}
_DUE_TASK_BATCH_SIZE = 20
_CRON_LOOKAHEAD_MINUTES = 366 * 24 * 60


@dataclass(frozen=True)
class _CronField:
    raw: str
    values: set[int]
    restricted: bool


@dataclass(frozen=True)
class _CronSchedule:
    minute: _CronField
    hour: _CronField
    day_of_month: _CronField
    month: _CronField
    day_of_week: _CronField

    def matches(self, candidate: datetime) -> bool:
        cron_weekday = (candidate.weekday() + 1) % 7
        day_of_month_match = candidate.day in self.day_of_month.values
        day_of_week_match = cron_weekday in self.day_of_week.values
        if self.day_of_month.restricted and self.day_of_week.restricted:
            day_match = day_of_month_match or day_of_week_match
        else:
            day_match = day_of_month_match and day_of_week_match
        return (
            candidate.minute in self.minute.values
            and candidate.hour in self.hour.values
            and candidate.month in self.month.values
            and day_match
        )


def _parse_alias(value: str, alias_map: dict[str, int]) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("Cron field contains an empty value.")
    return str(alias_map.get(normalized, normalized))


def _parse_cron_value(
    raw_value: str,
    *,
    minimum: int,
    maximum: int,
    alias_map: dict[str, int] | None = None,
    normalize_day_of_week: bool = False,
) -> int:
    text = raw_value.strip()
    if alias_map is not None:
        text = _parse_alias(text, alias_map)
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid cron value: {raw_value}") from exc
    if normalize_day_of_week and value == 7:
        value = 0
    if value < minimum or value > maximum:
        raise ValueError(f"Cron value out of range: {raw_value}")
    return value


def _expand_cron_field(
    raw_field: str,
    *,
    minimum: int,
    maximum: int,
    alias_map: dict[str, int] | None = None,
    normalize_day_of_week: bool = False,
) -> _CronField:
    field = raw_field.strip()
    if not field:
        raise ValueError("Cron field cannot be empty.")

    values: set[int] = set()
    for part in field.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"Invalid cron list: {raw_field}")

        step = 1
        if "/" in token:
            base_token, step_token = token.split("/", 1)
            base_token = base_token.strip()
            step_token = step_token.strip()
            if not step_token:
                raise ValueError(f"Invalid cron step: {token}")
            try:
                step = int(step_token)
            except ValueError as exc:
                raise ValueError(f"Invalid cron step: {token}") from exc
            if step <= 0:
                raise ValueError(f"Cron step must be > 0: {token}")
        else:
            base_token = token

        if base_token in {"", "*"}:
            start = minimum
            end = maximum
        elif "-" in base_token:
            start_text, end_text = base_token.split("-", 1)
            start = _parse_cron_value(
                start_text,
                minimum=minimum,
                maximum=maximum,
                alias_map=alias_map,
                normalize_day_of_week=normalize_day_of_week,
            )
            end = _parse_cron_value(
                end_text,
                minimum=minimum,
                maximum=maximum,
                alias_map=alias_map,
                normalize_day_of_week=normalize_day_of_week,
            )
            if end < start:
                raise ValueError(f"Invalid cron range: {token}")
        else:
            start = end = _parse_cron_value(
                base_token,
                minimum=minimum,
                maximum=maximum,
                alias_map=alias_map,
                normalize_day_of_week=normalize_day_of_week,
            )

        values.update(range(start, end + 1, step))

    return _CronField(raw=field, values=values, restricted=field != "*")


def _parse_cron_expression(expression: str) -> _CronSchedule:
    normalized = " ".join(str(expression or "").strip().split())
    fields = normalized.split(" ")
    if len(fields) != 5:
        raise ValueError("Cron expression must contain exactly 5 fields.")
    return _CronSchedule(
        minute=_expand_cron_field(fields[0], minimum=0, maximum=59),
        hour=_expand_cron_field(fields[1], minimum=0, maximum=23),
        day_of_month=_expand_cron_field(fields[2], minimum=1, maximum=31),
        month=_expand_cron_field(
            fields[3],
            minimum=1,
            maximum=12,
            alias_map=_MONTH_NAME_MAP,
        ),
        day_of_week=_expand_cron_field(
            fields[4],
            minimum=0,
            maximum=7,
            alias_map=_DOW_NAME_MAP,
            normalize_day_of_week=True,
        ),
    )


class ScheduledTaskService:
    """Manage scheduled task definitions and dispatch runs when due."""

    def __init__(
        self,
        session_manager,
        *,
        poll_interval_seconds: float = 1.0,
        due_batch_size: int = _DUE_TASK_BATCH_SIZE,
    ):
        self._session_manager = session_manager
        self._poll_interval_seconds = max(0.1, float(poll_interval_seconds))
        self._due_batch_size = max(1, int(due_batch_size))
        self._task_store: ScheduledTaskStore | None = None
        self._integration_store: IntegrationStore | None = None
        self._delivery_store: DeliveryStore | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._dispatch_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the background poller that triggers due tasks."""
        await self._session_manager.initialize()
        self._ensure_stores()
        await self.dispatch_due_tasks()
        if self._poll_task is None or self._poll_task.done():
            self._stop_event.clear()
            self._poll_task = asyncio.create_task(
                self._run_poll_loop(),
                name="scheduled-task-poller",
            )

    async def shutdown(self) -> None:
        """Stop the background poller."""
        self._stop_event.set()
        if self._poll_task is None:
            return
        self._poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._poll_task
        self._poll_task = None

    async def _run_poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.dispatch_due_tasks()
            except Exception:
                pass
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                continue

    def _ensure_stores(self) -> None:
        if (
            self._task_store is not None
            and self._integration_store is not None
            and self._delivery_store is not None
        ):
            return
        config = self._session_manager._config
        if config is None:
            raise RuntimeError("SessionManager is not initialized.")
        db_path = Path(config.agent.session_store_path)
        if not db_path.is_absolute():
            db_path = Path.cwd() / db_path
        resolved_db_path = db_path.resolve()
        self._task_store = ScheduledTaskStore(resolved_db_path)
        self._integration_store = IntegrationStore(resolved_db_path)
        self._delivery_store = DeliveryStore(resolved_db_path)

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _as_utc_iso(value: datetime) -> str:
        normalized = value.astimezone(timezone.utc)
        return normalized.isoformat(timespec="seconds")

    @staticmethod
    def _parse_iso(value: str) -> datetime:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _server_local_tzinfo() -> tzinfo:
        return datetime.now().astimezone().tzinfo or timezone.utc

    def _resolve_task_tzinfo(self, timezone_name: str | None) -> tzinfo:
        normalized = str(timezone_name or "").strip()
        if not normalized or normalized == "server_local":
            return self._server_local_tzinfo()
        try:
            return ZoneInfo(normalized)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"Unsupported timezone: {normalized}") from exc

    def _compute_next_run_at(
        self,
        *,
        cron_expression: str,
        timezone_name: str,
        after: datetime | None = None,
    ) -> str:
        schedule = _parse_cron_expression(cron_expression)
        tz = self._resolve_task_tzinfo(timezone_name)
        base = (after or self._utc_now()).astimezone(tz).replace(second=0, microsecond=0)
        candidate = base + timedelta(minutes=1)
        for _ in range(_CRON_LOOKAHEAD_MINUTES):
            if schedule.matches(candidate):
                return self._as_utc_iso(candidate)
            candidate += timedelta(minutes=1)
        raise ValueError(
            f"Could not compute next run time for cron expression: {cron_expression}"
        )

    async def list_tasks(
        self,
        *,
        account_id: str | None = None,
        enabled: bool | None = None,
        agent_id: str | None = None,
        integration_id: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        await self._session_manager.initialize()
        self._ensure_stores()
        tasks = self._task_store.list_tasks(
            account_id=account_id,
            enabled=enabled,
            agent_id=agent_id,
            integration_id=integration_id,
            limit=limit,
        )
        return [self.serialize_task(task) for task in tasks]

    async def get_task(self, task_id: str, *, account_id: str | None = None) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        return self.serialize_task(self._require_task(task_id, account_id=account_id))

    async def create_task(
        self,
        *,
        account_id: str | None = None,
        name: str,
        cron_expression: str,
        agent_id: str | None = None,
        prompt: str,
        integration_id: str | None = None,
        target_chat_id: str = "",
        target_thread_id: str = "",
        reply_to_message_id: str = "",
        timezone_name: str = "server_local",
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        normalized_name = str(name or "").strip()
        normalized_prompt = str(prompt or "").strip()
        normalized_cron = " ".join(str(cron_expression or "").strip().split())
        normalized_timezone = str(timezone_name or "server_local").strip() or "server_local"
        normalized_integration_id = str(integration_id or "").strip() or None
        normalized_target_chat_id = str(target_chat_id or "").strip()
        normalized_target_thread_id = str(target_thread_id or "").strip()
        normalized_reply_to_message_id = str(reply_to_message_id or "").strip()
        if not normalized_name:
            raise ValueError("Task name is required.")
        if not normalized_cron:
            raise ValueError("Cron expression is required.")
        if not normalized_prompt:
            raise ValueError("Task prompt is required.")
        normalized_agent_id = self._resolve_task_agent_id(
            account_id=account_id,
            agent_id=agent_id,
            integration_id=normalized_integration_id,
        )
        _parse_cron_expression(normalized_cron)
        self._resolve_task_tzinfo(normalized_timezone)
        self._validate_task_delivery_target(
            account_id=account_id,
            integration_id=normalized_integration_id,
            target_chat_id=normalized_target_chat_id,
            target_thread_id=normalized_target_thread_id,
            reply_to_message_id=normalized_reply_to_message_id,
        )
        next_run_at = (
            self._compute_next_run_at(
                cron_expression=normalized_cron,
                timezone_name=normalized_timezone,
            )
            if enabled
            else None
        )
        now = utc_now_iso()
        record = ScheduledTaskRecord(
            id=str(uuid4()),
            account_id=account_id or ScheduledTaskRecord.model_fields["account_id"].default,
            name=normalized_name,
            cron_expression=normalized_cron,
            timezone=normalized_timezone,
            agent_id=normalized_agent_id,
            prompt=normalized_prompt,
            integration_id=normalized_integration_id,
            target_chat_id=normalized_target_chat_id,
            target_thread_id=normalized_target_thread_id,
            reply_to_message_id=normalized_reply_to_message_id,
            enabled=bool(enabled),
            next_run_at=next_run_at,
            metadata=dict(metadata or {}),
            created_at=now,
            updated_at=now,
        )
        self._task_store.create_task(record)
        return self.serialize_task(record)

    async def update_task(
        self,
        task_id: str,
        *,
        account_id: str | None = None,
        name: str | None = None,
        cron_expression: str | None = None,
        agent_id: str | None = None,
        prompt: str | None = None,
        integration_id: str | None = None,
        target_chat_id: str | None = None,
        target_thread_id: str | None = None,
        reply_to_message_id: str | None = None,
        timezone_name: str | None = None,
        enabled: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = self._require_task(task_id, account_id=account_id)
        next_name = existing.name if name is None else str(name or "").strip()
        next_prompt = existing.prompt if prompt is None else str(prompt or "").strip()
        next_cron = (
            existing.cron_expression
            if cron_expression is None
            else " ".join(str(cron_expression or "").strip().split())
        )
        next_timezone = (
            existing.timezone
            if timezone_name is None
            else str(timezone_name or "").strip() or "server_local"
        )
        next_enabled = existing.enabled if enabled is None else bool(enabled)
        next_integration_id = (
            existing.integration_id
            if integration_id is None
            else (str(integration_id).strip() or None)
        )
        next_target_chat_id = (
            existing.target_chat_id
            if target_chat_id is None
            else str(target_chat_id or "").strip()
        )
        next_target_thread_id = (
            existing.target_thread_id
            if target_thread_id is None
            else str(target_thread_id or "").strip()
        )
        next_reply_to_message_id = (
            existing.reply_to_message_id
            if reply_to_message_id is None
            else str(reply_to_message_id or "").strip()
        )
        next_agent_id = self._resolve_task_agent_id(
            account_id=existing.account_id,
            agent_id=agent_id,
            integration_id=next_integration_id,
            fallback_agent_id=(
                existing.agent_id
                if next_integration_id == existing.integration_id
                else ""
            ),
        )
        if not next_name:
            raise ValueError("Task name is required.")
        if not next_cron:
            raise ValueError("Cron expression is required.")
        if not next_prompt:
            raise ValueError("Task prompt is required.")
        _parse_cron_expression(next_cron)
        self._resolve_task_tzinfo(next_timezone)
        self._validate_task_delivery_target(
            account_id=existing.account_id,
            integration_id=next_integration_id,
            target_chat_id=next_target_chat_id,
            target_thread_id=next_target_thread_id,
            reply_to_message_id=next_reply_to_message_id,
        )
        next_session_id = existing.session_id
        if next_agent_id != existing.agent_id:
            next_session_id = None
        next_run_at = (
            self._compute_next_run_at(
                cron_expression=next_cron,
                timezone_name=next_timezone,
            )
            if next_enabled
            else None
        )
        updated = existing.model_copy(
            update={
                "name": next_name,
                "cron_expression": next_cron,
                "timezone": next_timezone,
                "agent_id": next_agent_id,
                "prompt": next_prompt,
                "integration_id": next_integration_id,
                "target_chat_id": next_target_chat_id,
                "target_thread_id": next_target_thread_id,
                "reply_to_message_id": next_reply_to_message_id,
                "enabled": next_enabled,
                "session_id": next_session_id,
                "next_run_at": next_run_at,
                "metadata": dict(existing.metadata) if metadata is None else dict(metadata),
                "updated_at": utc_now_iso(),
            }
        )
        self._task_store.update_task(updated)
        return self.serialize_task(updated)

    async def set_task_enabled(
        self,
        task_id: str,
        *,
        account_id: str | None = None,
        enabled: bool,
    ) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        existing = self._require_task(task_id, account_id=account_id)
        updated = existing.model_copy(
            update={
                "enabled": bool(enabled),
                "next_run_at": (
                    self._compute_next_run_at(
                        cron_expression=existing.cron_expression,
                        timezone_name=existing.timezone,
                    )
                    if enabled
                    else None
                ),
                "updated_at": utc_now_iso(),
            }
        )
        self._task_store.update_task(updated)
        return self.serialize_task(updated)

    async def delete_task(self, task_id: str, *, account_id: str | None = None) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        record = self._require_task(task_id, account_id=account_id)
        self._task_store.delete_task(task_id, account_id=record.account_id)
        return self.serialize_task(record)

    async def run_task_now(self, task_id: str, *, account_id: str | None = None) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        execution = await self._dispatch_task_execution(
            self._require_task(task_id, account_id=account_id),
            trigger_kind="manual",
            scheduled_for=None,
        )
        task = self._task_store.get_task(task_id, account_id=account_id)
        return self.serialize_execution(execution, task=task)

    async def dispatch_due_tasks(
        self,
        *,
        now: datetime | None = None,
    ) -> list[ScheduledTaskExecutionRecord]:
        await self._session_manager.initialize()
        self._ensure_stores()
        async with self._dispatch_lock:
            effective_now = now or self._utc_now()
            due_tasks = self._task_store.list_due_tasks(
                due_before=self._as_utc_iso(effective_now),
                limit=self._due_batch_size,
            )
            executions: list[ScheduledTaskExecutionRecord] = []
            for task in due_tasks:
                updated_task = task.model_copy(
                    update={
                        "last_scheduled_for": task.next_run_at,
                        "next_run_at": self._compute_next_run_at(
                            cron_expression=task.cron_expression,
                            timezone_name=task.timezone,
                            after=effective_now,
                        ),
                        "updated_at": utc_now_iso(),
                    }
                )
                self._task_store.update_task(updated_task)
                execution = await self._dispatch_task_execution(
                    updated_task,
                    trigger_kind="schedule",
                    scheduled_for=task.next_run_at,
                )
                executions.append(execution)
            return executions

    async def list_task_executions(
        self,
        task_id: str,
        *,
        account_id: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        await self._session_manager.initialize()
        self._ensure_stores()
        task = self._require_task(task_id, account_id=account_id)
        executions = self._task_store.list_executions(
            account_id=task.account_id,
            task_id=task_id,
            limit=limit,
        )
        return [self.serialize_execution(execution, task=task) for execution in executions]

    async def get_execution(self, execution_id: str, *, account_id: str | None = None) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        execution = self._require_execution(execution_id, account_id=account_id)
        task = self._task_store.get_task(execution.task_id, account_id=execution.account_id)
        return self.serialize_execution(execution, task=task)

    async def get_execution_detail(
        self,
        execution_id: str,
        *,
        account_id: str | None = None,
    ) -> dict[str, Any]:
        await self._session_manager.initialize()
        self._ensure_stores()
        execution = self._require_execution(execution_id, account_id=account_id)
        task = self._task_store.get_task(execution.task_id, account_id=execution.account_id)
        payload: dict[str, Any] = {
            "task": self.serialize_task(task) if task is not None else None,
            "execution": self.serialize_execution(execution, task=task),
            "run": None,
            "steps": [],
            "timeline": [],
            "tree": None,
            "tools": [],
            "artifacts": [],
            "deliveries": [],
        }
        if not execution.run_id:
            return payload

        run = self._session_manager.get_run_info(
            execution.run_id,
            account_id=execution.account_id,
        )
        if run is None:
            return payload

        payload["run"] = run
        payload["steps"] = self._session_manager.list_run_steps(
            execution.run_id,
            account_id=execution.account_id,
        )
        payload["artifacts"] = self._session_manager.list_run_artifacts(
            execution.run_id,
            account_id=execution.account_id,
        )
        if self._session_manager.is_feature_enabled("enable_run_trace"):
            with contextlib.suppress(KeyError, RuntimeError):
                payload["timeline"] = self._session_manager.get_run_trace_timeline(
                    execution.run_id,
                    account_id=execution.account_id,
                )
                payload["tree"] = self._session_manager.get_run_trace_tree(
                    execution.run_id,
                    account_id=execution.account_id,
                )
                payload["tools"] = self._session_manager.get_run_tool_calls(
                    execution.run_id,
                    account_id=execution.account_id,
                )

        deliveries = self._delivery_store.list_deliveries(
            account_id=execution.account_id,
            run_id=execution.run_id,
        )
        payload["deliveries"] = [
            delivery.model_dump(mode="json") for delivery in deliveries
        ]
        return payload

    async def _dispatch_task_execution(
        self,
        task: ScheduledTaskRecord,
        *,
        trigger_kind: str,
        scheduled_for: str | None,
    ) -> ScheduledTaskExecutionRecord:
        now = utc_now_iso()
        execution = ScheduledTaskExecutionRecord(
            id=str(uuid4()),
            task_id=task.id,
            account_id=task.account_id,
            trigger_kind=trigger_kind,
            scheduled_for=scheduled_for,
            status="queued",
            created_at=now,
            updated_at=now,
            metadata={
                "task_name": task.name,
                "agent_id": task.agent_id,
                "integration_id": task.integration_id,
            },
        )
        self._task_store.create_execution(execution)

        try:
            session_id, task = await self._ensure_task_session(task)
            run_metadata: dict[str, Any] = {
                "source_kind": "scheduled_task",
                "source_label": "scheduled_task",
                "scheduled_task_id": task.id,
                "scheduled_task_execution_id": execution.id,
                "scheduled_task_name": task.name,
                "scheduled_trigger_kind": trigger_kind,
            }
            if scheduled_for:
                run_metadata["scheduled_for"] = scheduled_for
            if task.integration_id:
                integration = self._require_integration(
                    task.integration_id,
                    account_id=task.account_id,
                )
                if integration.status != "active":
                    raise ValueError(
                        f"Target integration is not active: {integration.id}"
                    )
                delivery_target = self._resolve_delivery_target(task, integration.config)
                run_metadata.update(
                    {
                        "integration_id": integration.id,
                        "channel_kind": integration.kind,
                        "provider_chat_id": delivery_target["provider_chat_id"],
                        "provider_thread_id": delivery_target["provider_thread_id"],
                        "provider_message_id": delivery_target["reply_to_message_id"],
                    }
                )
            run = self._session_manager.start_run(
                session_id,
                task.prompt,
                account_id=task.account_id,
                run_metadata=run_metadata,
            )
        except Exception as exc:
            failed_execution = execution.model_copy(
                update={
                    "status": "dispatch_failed",
                    "error_summary": str(exc),
                    "updated_at": utc_now_iso(),
                }
            )
            return self._task_store.update_execution(failed_execution)

        queued_execution = execution.model_copy(
            update={
                "run_id": run.id,
                "status": run.status,
                "updated_at": utc_now_iso(),
            }
        )
        return self._task_store.update_execution(queued_execution)

    async def _ensure_task_session(
        self,
        task: ScheduledTaskRecord,
    ) -> tuple[str, ScheduledTaskRecord]:
        session_id = str(task.session_id or "").strip()
        if session_id:
            session = self._session_manager.get_session_info(
                session_id,
                account_id=task.account_id,
            )
            if session is not None and (session.get("agent_id") or "system-default-agent") == task.agent_id:
                return session_id, task

        session_id = await self._session_manager.create_session(
            agent_id=task.agent_id,
            account_id=task.account_id,
        )
        updated_task = task.model_copy(
            update={
                "session_id": session_id,
                "updated_at": utc_now_iso(),
            }
        )
        self._task_store.update_task(updated_task)
        return session_id, updated_task

    def _resolve_delivery_target(
        self,
        task: ScheduledTaskRecord,
        integration_config: dict[str, Any],
    ) -> dict[str, str]:
        return self._resolve_delivery_target_values(
            target_chat_id=task.target_chat_id,
            target_thread_id=task.target_thread_id,
            reply_to_message_id=task.reply_to_message_id,
            integration_config=integration_config,
        )

    def _resolve_delivery_target_values(
        self,
        *,
        target_chat_id: str = "",
        target_thread_id: str = "",
        reply_to_message_id: str = "",
        integration_config: dict[str, Any],
    ) -> dict[str, str]:
        provider_chat_id = (
            str(target_chat_id or "").strip()
            or str(integration_config.get("default_chat_id") or "").strip()
            or str(integration_config.get("default_target_id") or "").strip()
            or str(integration_config.get("target_id") or "").strip()
            or str(integration_config.get("receive_id") or "").strip()
        )
        provider_thread_id = (
            str(target_thread_id or "").strip()
            or str(integration_config.get("default_thread_id") or "").strip()
            or str(integration_config.get("thread_id") or "").strip()
        )
        reply_to_message_id = (
            str(reply_to_message_id or "").strip()
            or str(integration_config.get("default_reply_to_message_id") or "").strip()
            or str(integration_config.get("reply_to_message_id") or "").strip()
        )
        if not provider_chat_id:
            raise ValueError(
                "Scheduled task delivery target is missing. "
                "Configure integration.default_chat_id or provide target_chat_id."
            )
        return {
            "provider_chat_id": provider_chat_id,
            "provider_thread_id": provider_thread_id,
            "reply_to_message_id": reply_to_message_id,
        }

    def _validate_task_delivery_target(
        self,
        *,
        account_id: str | None,
        integration_id: str | None,
        target_chat_id: str = "",
        target_thread_id: str = "",
        reply_to_message_id: str = "",
    ) -> None:
        normalized_integration_id = str(integration_id or "").strip()
        if not normalized_integration_id:
            return
        integration = self._require_integration(
            normalized_integration_id,
            account_id=account_id,
        )
        self._resolve_delivery_target_values(
            target_chat_id=target_chat_id,
            target_thread_id=target_thread_id,
            reply_to_message_id=reply_to_message_id,
            integration_config=integration.config,
        )

    def serialize_task(self, task: ScheduledTaskRecord) -> dict[str, Any]:
        latest_execution = self._task_store.get_latest_execution(
            task.id,
            account_id=task.account_id,
        )
        payload = task.model_dump(mode="json")
        payload["resolved_target_chat_id"] = ""
        payload["resolved_target_thread_id"] = ""
        if task.integration_id:
            integration = self._integration_store.get_integration(
                task.integration_id,
                account_id=task.account_id,
            )
            if integration is not None:
                with contextlib.suppress(ValueError):
                    target = self._resolve_delivery_target(task, integration.config)
                    payload["resolved_target_chat_id"] = target["provider_chat_id"]
                    payload["resolved_target_thread_id"] = target["provider_thread_id"]
                payload["integration_status"] = integration.status
                payload["integration_kind"] = integration.kind
                payload["integration_display_name"] = integration.display_name or integration.name
            else:
                payload["integration_status"] = "missing"
                payload["integration_kind"] = ""
                payload["integration_display_name"] = task.integration_id
        else:
            payload["integration_status"] = ""
            payload["integration_kind"] = ""
            payload["integration_display_name"] = ""
        payload["last_execution"] = (
            self.serialize_execution(latest_execution, task=task)
            if latest_execution is not None
            else None
        )
        return payload

    def serialize_execution(
        self,
        execution: ScheduledTaskExecutionRecord,
        *,
        task: ScheduledTaskRecord | None = None,
    ) -> dict[str, Any]:
        run = (
            self._session_manager.get_run_info(
                execution.run_id,
                account_id=execution.account_id,
            )
            if execution.run_id
            else None
        )
        delivery_status = ""
        delivery_error_summary = ""
        delivery_count = 0
        if execution.run_id:
            deliveries = self._delivery_store.list_deliveries(
                account_id=execution.account_id,
                run_id=execution.run_id,
            )
            delivery_count = len(deliveries)
            delivery_status, delivery_error_summary = self._summarize_deliveries(deliveries)
        payload = execution.model_dump(mode="json")
        payload["status"] = (
            str(run.get("status") or "").strip()
            if run is not None
            else execution.status
        )
        payload["run_status"] = str(run.get("status") or "").strip() if run is not None else ""
        payload["started_at"] = run.get("started_at") if run is not None else None
        payload["finished_at"] = run.get("finished_at") if run is not None else None
        payload["run_error_summary"] = (
            str(run.get("error_summary") or "").strip()
            if run is not None
            else execution.error_summary
        )
        payload["delivery_status"] = delivery_status
        payload["delivery_error_summary"] = delivery_error_summary
        payload["delivery_count"] = delivery_count
        payload["session_id"] = (
            str(run.get("session_id") or "")
            if run is not None
            else str(task.session_id or "") if task is not None else ""
        )
        payload["task_name"] = task.name if task is not None else ""
        payload["integration_id"] = (
            task.integration_id
            if task is not None
            else str(run.get("run_metadata", {}).get("integration_id") or "")
            if run is not None
            else None
        )
        return payload

    @staticmethod
    def _summarize_deliveries(deliveries: list[Any]) -> tuple[str, str]:
        if not deliveries:
            return "", ""
        failed = next((delivery for delivery in deliveries if delivery.status == "failed"), None)
        if failed is not None:
            return "failed", str(failed.error_summary or "").strip()
        if all(delivery.status == "delivered" for delivery in deliveries):
            return "delivered", ""
        return "pending", ""

    def _require_task(
        self,
        task_id: str,
        *,
        account_id: str | None = None,
    ) -> ScheduledTaskRecord:
        task = self._task_store.get_task(task_id, account_id=account_id)
        if task is not None:
            return task
        if account_id is not None and self._task_store.get_task(task_id) is not None:
            raise PermissionError("Scheduled task does not belong to the current account.")
        task = self._task_store.get_task(task_id)
        if task is None:
            raise KeyError(f"Scheduled task not found: {task_id}")
        return task

    def _require_execution(
        self,
        execution_id: str,
        *,
        account_id: str | None = None,
    ) -> ScheduledTaskExecutionRecord:
        execution = self._task_store.get_execution(execution_id, account_id=account_id)
        if execution is not None:
            return execution
        if account_id is not None and self._task_store.get_execution(execution_id) is not None:
            raise PermissionError("Scheduled task execution does not belong to the current account.")
        execution = self._task_store.get_execution(execution_id)
        if execution is None:
            raise KeyError(f"Scheduled task execution not found: {execution_id}")
        return execution

    def _require_agent(self, agent_id: str, *, account_id: str | None = None) -> None:
        normalized = str(agent_id or "").strip()
        if not normalized:
            raise ValueError("Agent ID is required.")
        agent_store = self._session_manager._agent_store
        if agent_store is None or agent_store.get_agent_template(
            normalized,
            account_id=account_id,
        ) is None:
            raise ValueError(f"Agent not found: {normalized}")

    def _resolve_default_agent_id(self, integration_config: dict[str, Any]) -> str:
        for key in ("default_agent_id", "default_agent_template_id"):
            value = str(integration_config.get(key) or "").strip()
            if value:
                return value
        return ""

    def _resolve_task_agent_id(
        self,
        *,
        account_id: str | None,
        agent_id: str | None,
        integration_id: str | None,
        fallback_agent_id: str = "",
    ) -> str:
        normalized_agent_id = str(agent_id or "").strip()
        if normalized_agent_id:
            self._require_agent(normalized_agent_id, account_id=account_id)
            return normalized_agent_id

        normalized_integration_id = str(integration_id or "").strip()
        if normalized_integration_id:
            integration = self._require_integration(
                normalized_integration_id,
                account_id=account_id,
            )
            resolved_agent_id = self._resolve_default_agent_id(integration.config)
            if resolved_agent_id:
                self._require_agent(resolved_agent_id, account_id=integration.account_id)
                return resolved_agent_id

        normalized_fallback_agent_id = str(fallback_agent_id or "").strip()
        if normalized_fallback_agent_id:
            self._require_agent(normalized_fallback_agent_id, account_id=account_id)
            return normalized_fallback_agent_id

        raise ValueError(
            "Scheduled task agent is missing. "
            "Configure integration.default_agent_id or provide agent_id."
        )

    def _require_integration(self, integration_id: str, *, account_id: str | None = None):
        integration = self._integration_store.get_integration(integration_id, account_id=account_id)
        if integration is not None:
            return integration
        if account_id is not None and self._integration_store.get_integration(integration_id) is not None:
            raise PermissionError("Integration does not belong to the current account.")
        integration = self._integration_store.get_integration(integration_id)
        if integration is None:
            raise ValueError(f"Integration not found: {integration_id}")
        return integration
