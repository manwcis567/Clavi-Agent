"""Common models used by channel adapters."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ..integration_models import IntegrationConfigRecord

QuickAckBodyType = Literal["empty", "json", "text"]
AttachmentKind = Literal[
    "text",
    "image",
    "file",
    "audio",
    "media",
    "post",
    "interactive",
    "unknown",
]


class ChannelRequest(BaseModel):
    """Normalized webhook request envelope."""

    method: str = "POST"
    path: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    query_params: dict[str, str] = Field(default_factory=dict)
    body: bytes = Field(default=b"", repr=False)
    received_at: str
    remote_addr: str = ""

    def header(self, name: str, default: str = "") -> str:
        target = name.strip().lower()
        for key, value in self.headers.items():
            if key.strip().lower() == target:
                return str(value)
        return default


class ChannelContext(BaseModel):
    """Runtime adapter context."""

    integration: IntegrationConfigRecord
    credentials: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def integration_id(self) -> str:
        return self.integration.id

    @property
    def channel_kind(self) -> str:
        return self.integration.kind

    def get_secret(self, key: str, default: str = "") -> str:
        value = self.credentials.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
        config_value = self.integration.config.get(key)
        if config_value is not None and str(config_value).strip():
            return str(config_value).strip()
        return default

    def require_secret(self, key: str) -> str:
        value = self.get_secret(key)
        if not value:
            raise ValueError(f"Integration {self.integration.id} is missing required config: {key}")
        return value


class RequestVerificationResult(BaseModel):
    """Request verification result."""

    accepted: bool
    signature_valid: bool = False
    reason: str = ""
    body_json: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuickAckIntent(BaseModel):
    """Quick acknowledgement response intent."""

    status_code: int = 200
    body_type: QuickAckBodyType = "empty"
    body_text: str = ""
    body_json: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class NormalizedAdapterError(BaseModel):
    """Normalized channel error payload."""

    code: str = "unknown_error"
    message: str
    retryable: bool = False
    status_code: int | None = None
    raw_error: Any = None


class UploadBridgeHint(BaseModel):
    """Hints needed to bridge a channel attachment into uploads."""

    source_kind: Literal["remote_url", "provider_resource"] = "provider_resource"
    download_url: str = ""
    resource_type: str = "file"
    provider_file_id: str = ""
    provider_message_id: str = ""
    suggested_filename: str = ""
    mime_type: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MsgMention(BaseModel):
    """Normalized mention payload."""

    key: str = ""
    id: str = ""
    id_type: str = ""
    name: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MsgAttachment(BaseModel):
    """Normalized attachment reference."""

    kind: AttachmentKind = "unknown"
    provider_file_id: str = ""
    provider_message_id: str = ""
    name: str = ""
    mime_type: str = ""
    size_bytes: int | None = None
    download_url: str = ""
    upload_hint: UploadBridgeHint | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedInboundEvent(BaseModel):
    """Normalized inbound event parsed by one adapter."""

    integration_id: str
    channel_kind: str
    event_type: str
    provider_event_id: str = ""
    provider_message_id: str = ""
    provider_chat_id: str = ""
    provider_thread_id: str = ""
    provider_user_id: str = ""
    tenant_id: str = ""
    dedup_key: str = ""
    received_at: str
    signature_valid: bool = False
    message_type: str = "text"
    text: str = ""
    sender_name: str = ""
    is_group: bool = False
    locale: str = ""
    timezone: str = ""
    raw_headers: dict[str, str] = Field(default_factory=dict)
    raw_payload: Any = Field(default_factory=dict)
    attachments: list[MsgAttachment] = Field(default_factory=list)
    mentions: list[MsgMention] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MsgContextEnvelope(BaseModel):
    """Normalized business message context before routing."""

    integration_id: str
    channel_kind: str
    tenant_id: str = ""
    chat_id: str = ""
    thread_id: str = ""
    message_id: str = ""
    sender_id: str = ""
    sender_name: str = ""
    is_group: bool = False
    text: str = ""
    attachments: list[MsgAttachment] = Field(default_factory=list)
    mentions: list[MsgMention] = Field(default_factory=list)
    locale: str = ""
    timezone: str = ""
    received_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class OutboundMessage(BaseModel):
    """Normalized outbound message payload."""

    target_id: str = ""
    target_id_type: str = "chat_id"
    reply_to_message_id: str = ""
    thread_id: str = ""
    message_type: str = "text"
    text: str = ""
    content: dict[str, Any] = Field(default_factory=dict)
    dedup_key: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize_content(self) -> "OutboundMessage":
        if not self.content and self.message_type == "text":
            self.content = {"text": self.text}
        if not self.reply_to_message_id and not self.target_id:
            raise ValueError("OutboundMessage requires target_id or reply_to_message_id.")
        return self


class OutboundFile(BaseModel):
    """Normalized outbound file payload."""

    target_id: str = ""
    target_id_type: str = "chat_id"
    reply_to_message_id: str = ""
    thread_id: str = ""
    file_kind: Literal["image", "file", "audio", "media"] = "file"
    provider_file_id: str = ""
    file_name: str = ""
    url: str = ""
    text_fallback: str = ""
    dedup_key: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_source(self) -> "OutboundFile":
        if not self.reply_to_message_id and not self.target_id:
            raise ValueError("OutboundFile requires target_id or reply_to_message_id.")
        if not self.provider_file_id and not self.url:
            raise ValueError("OutboundFile requires provider_file_id or url.")
        return self


class OutboundReaction(BaseModel):
    """Unified outbound emoji reaction request."""

    message_id: str
    reaction_type: str
    dedup_key: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_payload(self) -> "OutboundReaction":
        self.message_id = str(self.message_id or "").strip()
        self.reaction_type = str(self.reaction_type or "").strip()
        if not self.message_id:
            raise ValueError("OutboundReaction requires message_id.")
        if not self.reaction_type:
            raise ValueError("OutboundReaction requires reaction_type.")
        return self


class OutboundSendResult(BaseModel):
    """Normalized outbound send result."""

    ok: bool
    provider_message_id: str = ""
    provider_chat_id: str = ""
    provider_thread_id: str = ""
    http_status: int | None = None
    raw_response: Any = None
    error: NormalizedAdapterError | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplyAttachmentRef(BaseModel):
    """Normalized artifact reference used by reply dispatch."""

    artifact_id: str = ""
    file_kind: Literal["image", "file", "audio", "media"] = "file"
    display_name: str = ""
    mime_type: str = ""
    download_url: str = ""
    text_fallback: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplyEnvelope(BaseModel):
    """Normalized terminal reply envelope for one completed run."""

    integration_id: str
    channel_kind: str
    run_id: str
    session_id: str
    inbound_event_id: str = ""
    binding_id: str = ""
    provider_chat_id: str
    provider_thread_id: str = ""
    reply_to_message_id: str = ""
    run_status: str
    text: str = ""
    attachments: list[ReplyAttachmentRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
