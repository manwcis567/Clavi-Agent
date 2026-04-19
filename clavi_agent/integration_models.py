"""消息渠道集成相关的持久化模型。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from .account_constants import ROOT_ACCOUNT_ID

IntegrationStatus = Literal["active", "disabled", "error"]
CredentialStorageKind = Literal["env", "external_ref", "local_encrypted"]
InboundEventStatus = Literal[
    "received",
    "verified",
    "rejected",
    "deduplicated",
    "command_handled",
    "routed",
    "bridged",
    "completed",
    "failed",
]
OutboundDeliveryStatus = Literal["pending", "sending", "retrying", "delivered", "failed"]

INBOUND_EVENT_STATUS_TRANSITIONS: dict[InboundEventStatus, set[InboundEventStatus]] = {
    "received": {"verified", "rejected"},
    "verified": {"deduplicated", "command_handled", "routed"},
    "rejected": set(),
    "deduplicated": set(),
    "command_handled": set(),
    "routed": {"bridged", "failed"},
    "bridged": {"completed", "failed"},
    "completed": set(),
    "failed": set(),
}
OUTBOUND_DELIVERY_STATUS_TRANSITIONS: dict[
    OutboundDeliveryStatus, set[OutboundDeliveryStatus]
] = {
    "pending": {"sending"},
    "sending": {"delivered", "retrying", "failed"},
    "retrying": {"sending", "failed"},
    "delivered": set(),
    "failed": set(),
}

DEFAULT_MAX_INBOUND_HEADERS_BYTES = 16 * 1024
DEFAULT_MAX_INBOUND_PAYLOAD_BYTES = 128 * 1024
SENSITIVE_FIELD_KEYWORDS = frozenset(
    {
        "app_secret",
        "authorization",
        "encrypt_key",
        "password",
        "secret",
        "signature",
        "signing_secret",
        "token",
        "webhook_secret",
    }
)


def mask_secret(value: str, keep_prefix: int = 2, keep_suffix: int = 2) -> str:
    """生成脱敏后的展示值。"""
    normalized = str(value or "").strip()
    if not normalized:
        return ""
    if len(normalized) <= keep_prefix + keep_suffix:
        if len(normalized) <= 2:
            return "*" * len(normalized)
        return f"{normalized[:1]}{'*' * (len(normalized) - 2)}{normalized[-1:]}"
    hidden = "*" * max(4, len(normalized) - keep_prefix - keep_suffix)
    return f"{normalized[:keep_prefix]}{hidden}{normalized[-keep_suffix:]}"


def _is_sensitive_field(key: str) -> bool:
    normalized = key.strip().lower().replace("-", "_")
    return any(keyword in normalized for keyword in SENSITIVE_FIELD_KEYWORDS)


def _redact_sensitive_payload(payload: Any, path: str = "") -> tuple[Any, list[str]]:
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        redacted_fields: list[str] = []
        for key, value in payload.items():
            key_text = str(key)
            field_path = f"{path}.{key_text}" if path else key_text
            if _is_sensitive_field(key_text):
                sanitized[key_text] = mask_secret(str(value))
                redacted_fields.append(field_path)
                continue
            nested_value, nested_fields = _redact_sensitive_payload(value, field_path)
            sanitized[key_text] = nested_value
            redacted_fields.extend(nested_fields)
        return sanitized, redacted_fields

    if isinstance(payload, list):
        sanitized_items: list[Any] = []
        redacted_fields: list[str] = []
        for index, item in enumerate(payload):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            nested_value, nested_fields = _redact_sensitive_payload(item, item_path)
            sanitized_items.append(nested_value)
            redacted_fields.extend(nested_fields)
        return sanitized_items, redacted_fields

    return payload, []


def _build_truncated_payload(value: Any, size_bytes: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "_truncated": True,
        "_original_size_bytes": size_bytes,
        "_value_type": type(value).__name__,
    }
    if isinstance(value, dict):
        summary["_top_level_keys"] = list(value.keys())[:50]
    elif isinstance(value, list):
        summary["_item_count"] = len(value)
    elif value is not None:
        summary["_preview"] = str(value)[:120]
    return summary


class StoredJsonPayload(BaseModel):
    """用于落库的 JSON 负载包装。"""

    data: Any = Field(default_factory=dict)
    size_bytes: int = Field(default=0, ge=0)
    truncated: bool = False
    redacted_fields: list[str] = Field(default_factory=list)


def prepare_bounded_json_payload(
    payload: Any,
    *,
    max_bytes: int,
) -> StoredJsonPayload:
    """对 JSON 负载执行脱敏与体积裁剪。"""
    sanitized, redacted_fields = _redact_sensitive_payload(payload)

    import json

    raw_json = json.dumps(sanitized, ensure_ascii=False, sort_keys=True)
    size_bytes = len(raw_json.encode("utf-8"))
    if size_bytes <= max_bytes:
        return StoredJsonPayload(
            data=sanitized,
            size_bytes=size_bytes,
            truncated=False,
            redacted_fields=redacted_fields,
        )

    return StoredJsonPayload(
        data=_build_truncated_payload(sanitized, size_bytes),
        size_bytes=size_bytes,
        truncated=True,
        redacted_fields=redacted_fields,
    )


class IntegrationConfigRecord(BaseModel):
    """集成配置主记录。"""

    id: str
    account_id: str = ROOT_ACCOUNT_ID
    name: str
    kind: str
    status: IntegrationStatus = "disabled"
    display_name: str = ""
    tenant_id: str = ""
    webhook_path: str
    config: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    last_verified_at: str | None = None
    last_error: str = ""

    @model_validator(mode="after")
    def _fill_display_name(self) -> "IntegrationConfigRecord":
        if not self.display_name:
            self.display_name = self.name
        return self


class IntegrationCredentialRecord(BaseModel):
    """集成凭证引用或密文记录。"""

    id: str
    integration_id: str
    account_id: str = ROOT_ACCOUNT_ID
    credential_key: str
    storage_kind: CredentialStorageKind
    secret_ref: str = ""
    secret_ciphertext: str = ""
    masked_value: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    @model_validator(mode="after")
    def _validate_storage_fields(self) -> "IntegrationCredentialRecord":
        if self.storage_kind == "local_encrypted" and not self.secret_ciphertext:
            raise ValueError("local_encrypted 凭证必须提供密文。")
        if self.storage_kind in {"env", "external_ref"} and not self.secret_ref:
            raise ValueError(f"{self.storage_kind} 凭证必须提供 secret_ref。")
        if not self.masked_value:
            preview_source = self.secret_ref or self.secret_ciphertext
            self.masked_value = mask_secret(preview_source)
        return self


class InboundEventRecord(BaseModel):
    """入站事件审计记录。"""

    id: str
    integration_id: str
    account_id: str = ROOT_ACCOUNT_ID
    provider_event_id: str = ""
    provider_message_id: str = ""
    provider_chat_id: str = ""
    provider_thread_id: str = ""
    provider_user_id: str = ""
    event_type: str = "message"
    received_at: str
    signature_valid: bool = False
    dedup_key: str = ""
    raw_headers: Any = Field(default_factory=dict)
    raw_headers_size_bytes: int = Field(default=0, ge=0)
    raw_headers_truncated: bool = False
    raw_headers_redacted_fields: list[str] = Field(default_factory=list)
    raw_payload: Any = Field(default_factory=dict)
    raw_payload_size_bytes: int = Field(default=0, ge=0)
    raw_payload_truncated: bool = False
    raw_payload_redacted_fields: list[str] = Field(default_factory=list)
    normalized_status: InboundEventStatus = "received"
    normalized_error: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def can_transition_to(self, next_status: InboundEventStatus) -> bool:
        return next_status in INBOUND_EVENT_STATUS_TRANSITIONS[self.normalized_status]

    def transition_to(
        self,
        next_status: InboundEventStatus,
        *,
        error_message: str | None = None,
    ) -> "InboundEventRecord":
        if not self.can_transition_to(next_status):
            raise ValueError(
                f"非法的 InboundEvent 状态迁移: {self.normalized_status} -> {next_status}"
            )

        updates: dict[str, Any] = {"normalized_status": next_status}
        if error_message is not None:
            updates["normalized_error"] = error_message
        return self.model_copy(update=updates)


class ConversationBindingRecord(BaseModel):
    """渠道会话与 Agent 会话的绑定记录。"""

    id: str
    integration_id: str
    account_id: str = ROOT_ACCOUNT_ID
    tenant_id: str = ""
    chat_id: str = ""
    thread_id: str = ""
    binding_scope: str
    agent_id: str
    session_id: str
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str
    last_message_at: str | None = None


class RoutingRuleRecord(BaseModel):
    """基础路由规则记录。"""

    id: str
    integration_id: str
    account_id: str = ROOT_ACCOUNT_ID
    priority: int = Field(default=100)
    match_type: str
    match_value: str
    agent_id: str
    session_strategy: str = "reuse"
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str


class OutboundDeliveryRecord(BaseModel):
    """出站投递记录。"""

    id: str
    integration_id: str
    account_id: str = ROOT_ACCOUNT_ID
    run_id: str
    session_id: str
    inbound_event_id: str | None = None
    provider_chat_id: str
    provider_thread_id: str = ""
    provider_message_id: str = ""
    delivery_type: str
    payload: Any = Field(default_factory=dict)
    status: OutboundDeliveryStatus = "pending"
    attempt_count: int = Field(default=0, ge=0)
    last_attempt_at: str | None = None
    error_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    updated_at: str

    def can_transition_to(self, next_status: OutboundDeliveryStatus) -> bool:
        return next_status in OUTBOUND_DELIVERY_STATUS_TRANSITIONS[self.status]

    def transition_to(
        self,
        next_status: OutboundDeliveryStatus,
        *,
        changed_at: str | None = None,
        attempt_count: int | None = None,
        error_summary: str | None = None,
        provider_message_id: str | None = None,
    ) -> "OutboundDeliveryRecord":
        if not self.can_transition_to(next_status):
            raise ValueError(f"非法的 OutboundDelivery 状态迁移: {self.status} -> {next_status}")

        updates: dict[str, Any] = {"status": next_status}
        if changed_at is not None:
            updates["updated_at"] = changed_at
            updates["last_attempt_at"] = changed_at
        if attempt_count is not None:
            updates["attempt_count"] = attempt_count
        if error_summary is not None:
            updates["error_summary"] = error_summary
        if provider_message_id is not None:
            updates["provider_message_id"] = provider_message_id
        return self.model_copy(update=updates)


class DeliveryAttemptRecord(BaseModel):
    """单次出站投递尝试明细。"""

    id: str
    delivery_id: str
    account_id: str = ROOT_ACCOUNT_ID
    attempt_number: int = Field(ge=1)
    status: str
    request_payload: Any = Field(default_factory=dict)
    response_payload: Any = Field(default_factory=dict)
    error_summary: str = ""
    started_at: str
    finished_at: str | None = None
