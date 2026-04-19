"""本地联调用的 Mock 渠道适配器。"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from .adapter_base import ChannelAdapter, decode_json_body
from .models import (
    ChannelContext,
    ChannelRequest,
    MsgAttachment,
    MsgContextEnvelope,
    MsgMention,
    NormalizedAdapterError,
    OutboundFile,
    OutboundMessage,
    OutboundReaction,
    OutboundSendResult,
    ParsedInboundEvent,
    QuickAckIntent,
    RequestVerificationResult,
    UploadBridgeHint,
)


class MockChannelAdapter(ChannelAdapter):
    """用于本地调试与自动化测试的渠道适配器。"""

    kind = "mock"

    def __init__(self):
        self.sent_payloads: list[dict[str, Any]] = []

    @property
    def display_name(self) -> str:
        return "Mock Channel"

    def verify_request(
        self,
        context: ChannelContext,
        request: ChannelRequest,
    ) -> RequestVerificationResult:
        payload = decode_json_body(request.body)
        expected_token = (
            context.get_secret("verify_token") or context.get_secret("verification_token")
        )
        request_token = str(payload.get("token") or payload.get("verification_token") or "").strip()
        token_valid = not expected_token or request_token == expected_token
        if not token_valid:
            return RequestVerificationResult(
                accepted=False,
                signature_valid=False,
                reason="Mock 渠道 verify token 不匹配。",
                body_json=payload,
            )

        signing_secret = context.get_secret("signing_secret")
        if signing_secret:
            timestamp = request.header("x-mock-timestamp")
            signature = request.header("x-mock-signature")
            expected_signature = self._build_signature(
                signing_secret,
                timestamp,
                request.body,
            )
            if not timestamp or not signature or not hmac.compare_digest(signature, expected_signature):
                return RequestVerificationResult(
                    accepted=False,
                    signature_valid=False,
                    reason="Mock 渠道签名校验失败。",
                    body_json=payload,
                )

        return RequestVerificationResult(
            accepted=True,
            signature_valid=True,
            body_json=payload,
        )

    def parse_inbound_event(
        self,
        context: ChannelContext,
        request: ChannelRequest,
        verification: RequestVerificationResult,
    ) -> ParsedInboundEvent:
        payload = verification.body_json if verification.body_json is not None else decode_json_body(request.body)
        message = payload.get("message") or {}
        event_type = str(payload.get("event_type") or payload.get("type") or "message")
        attachments = [self._parse_attachment(item, message.get("message_id", "")) for item in message.get("attachments", [])]
        mentions = [self._parse_mention(item) for item in message.get("mentions", [])]
        provider_message_id = str(message.get("message_id") or "")
        provider_event_id = str(payload.get("event_id") or provider_message_id or "")
        provider_chat_id = str(message.get("chat_id") or "")
        provider_thread_id = str(message.get("thread_id") or "")
        provider_user_id = str(message.get("sender_id") or payload.get("sender_id") or "")

        return ParsedInboundEvent(
            integration_id=context.integration_id,
            channel_kind=self.kind,
            event_type=event_type,
            provider_event_id=provider_event_id,
            provider_message_id=provider_message_id,
            provider_chat_id=provider_chat_id,
            provider_thread_id=provider_thread_id,
            provider_user_id=provider_user_id,
            tenant_id=str(payload.get("tenant_id") or ""),
            dedup_key=str(payload.get("dedup_key") or provider_message_id or provider_event_id),
            received_at=request.received_at,
            signature_valid=verification.signature_valid,
            message_type=str(message.get("message_type") or "text"),
            text=str(message.get("text") or ""),
            sender_name=str(message.get("sender_name") or ""),
            is_group=bool(message.get("is_group")),
            locale=str(message.get("locale") or ""),
            timezone=str(message.get("timezone") or ""),
            raw_headers=request.headers,
            raw_payload=payload,
            attachments=attachments,
            mentions=mentions,
            metadata={
                "mock_event_kind": event_type,
                "challenge": str(payload.get("challenge") or ""),
                "is_from_self": bool(
                    message.get("is_from_self")
                    or payload.get("is_from_self")
                ),
            },
        )

    def build_msg_context(
        self,
        context: ChannelContext,
        event: ParsedInboundEvent,
    ) -> MsgContextEnvelope:
        return MsgContextEnvelope(
            integration_id=context.integration_id,
            channel_kind=self.kind,
            tenant_id=event.tenant_id,
            chat_id=event.provider_chat_id,
            thread_id=event.provider_thread_id,
            message_id=event.provider_message_id,
            sender_id=event.provider_user_id,
            sender_name=event.sender_name,
            is_group=event.is_group,
            text=event.text,
            attachments=event.attachments,
            mentions=event.mentions,
            locale=event.locale,
            timezone=event.timezone,
            received_at=event.received_at,
            metadata={
                "event_type": event.event_type,
                "message_type": event.message_type,
            },
        )

    def emit_quick_ack(
        self,
        context: ChannelContext,
        event: ParsedInboundEvent,
    ) -> QuickAckIntent:
        if event.event_type == "url_verification":
            return QuickAckIntent(
                body_type="json",
                body_json={"challenge": event.metadata.get("challenge", "")},
            )
        return QuickAckIntent(body_type="json", body_json={"ok": True})

    async def send_outbound_message(
        self,
        context: ChannelContext,
        message: OutboundMessage,
    ) -> OutboundSendResult:
        provider_message_id = f"mock-msg-{len(self.sent_payloads) + 1}"
        record = {
            "kind": "message",
            "integration_id": context.integration_id,
            "message_id": provider_message_id,
            "payload": message.model_dump(mode="json"),
        }
        self.sent_payloads.append(record)
        return OutboundSendResult(
            ok=True,
            provider_message_id=provider_message_id,
            provider_chat_id=message.target_id,
            provider_thread_id=message.thread_id,
            raw_response={"ok": True},
            metadata={"record_index": len(self.sent_payloads) - 1},
        )

    async def send_outbound_file(
        self,
        context: ChannelContext,
        file: OutboundFile,
    ) -> OutboundSendResult:
        provider_message_id = f"mock-file-{len(self.sent_payloads) + 1}"
        record = {
            "kind": "file",
            "integration_id": context.integration_id,
            "message_id": provider_message_id,
            "payload": file.model_dump(mode="json"),
        }
        self.sent_payloads.append(record)
        return OutboundSendResult(
            ok=True,
            provider_message_id=provider_message_id,
            provider_chat_id=file.target_id,
            provider_thread_id=file.thread_id,
            raw_response={"ok": True},
            metadata={"record_index": len(self.sent_payloads) - 1},
        )

    async def send_outbound_reaction(
        self,
        context: ChannelContext,
        reaction: OutboundReaction,
    ) -> OutboundSendResult:
        provider_message_id = f"mock-reaction-{len(self.sent_payloads) + 1}"
        record = {
            "kind": "reaction",
            "integration_id": context.integration_id,
            "message_id": provider_message_id,
            "payload": reaction.model_dump(mode="json"),
        }
        self.sent_payloads.append(record)
        return OutboundSendResult(
            ok=True,
            provider_message_id=provider_message_id,
            raw_response={"ok": True},
            metadata={"record_index": len(self.sent_payloads) - 1},
        )

    def normalize_error(self, error: Exception | dict[str, Any]) -> NormalizedAdapterError:
        if isinstance(error, dict):
            return NormalizedAdapterError(
                code=str(error.get("code") or "mock_error"),
                message=str(error.get("message") or "Mock 渠道异常"),
                retryable=bool(error.get("retryable", False)),
                raw_error=error,
            )
        return NormalizedAdapterError(
            code="mock_error",
            message=str(error),
            raw_error={"type": type(error).__name__},
        )

    @staticmethod
    def _build_signature(secret: str, timestamp: str, body: bytes) -> str:
        message = f"{timestamp}.{body.decode('utf-8')}".encode("utf-8")
        return hmac.new(secret.encode("utf-8"), message, hashlib.sha256).hexdigest()

    @staticmethod
    def _parse_attachment(raw_attachment: dict[str, Any], message_id: str) -> MsgAttachment:
        download_url = str(raw_attachment.get("download_url") or "")
        upload_hint = None
        if download_url:
            upload_hint = UploadBridgeHint(
                source_kind="remote_url",
                download_url=download_url,
                resource_type=str(raw_attachment.get("kind") or "file"),
                provider_file_id=str(raw_attachment.get("provider_file_id") or ""),
                provider_message_id=message_id,
                suggested_filename=str(raw_attachment.get("name") or ""),
                mime_type=str(raw_attachment.get("mime_type") or ""),
            )
        return MsgAttachment(
            kind=str(raw_attachment.get("kind") or "unknown"),
            provider_file_id=str(raw_attachment.get("provider_file_id") or ""),
            provider_message_id=message_id,
            name=str(raw_attachment.get("name") or ""),
            mime_type=str(raw_attachment.get("mime_type") or ""),
            size_bytes=raw_attachment.get("size_bytes"),
            download_url=download_url,
            upload_hint=upload_hint,
            metadata=dict(raw_attachment.get("metadata") or {}),
        )

    @staticmethod
    def _parse_mention(raw_mention: dict[str, Any]) -> MsgMention:
        return MsgMention(
            key=str(raw_mention.get("key") or ""),
            id=str(raw_mention.get("id") or ""),
            id_type=str(raw_mention.get("id_type") or ""),
            name=str(raw_mention.get("name") or ""),
            metadata=dict(raw_mention.get("metadata") or {}),
        )
