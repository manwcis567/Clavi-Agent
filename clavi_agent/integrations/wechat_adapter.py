"""Native WeChat iLink adapter."""

from __future__ import annotations

import uuid
from typing import Any

import httpx

from .adapter_base import ChannelAdapter, decode_json_body
from .models import (
    ChannelContext,
    ChannelRequest,
    MsgAttachment,
    MsgContextEnvelope,
    NormalizedAdapterError,
    OutboundFile,
    OutboundMessage,
    OutboundSendResult,
    ParsedInboundEvent,
    QuickAckIntent,
    RequestVerificationResult,
)
from .wechat_ilink import WeChatILinkClient, WeChatILinkCredentials

USER_MESSAGE_TYPE = 1
BOT_MESSAGE_TYPE = 2
MESSAGE_STATE_FINISH = 2

ITEM_TYPE_TEXT = 1
ITEM_TYPE_IMAGE = 2
ITEM_TYPE_VOICE = 3
ITEM_TYPE_FILE = 4
ITEM_TYPE_VIDEO = 5

_WEBHOOK_UNSUPPORTED_REASON = (
    "WeChat uses native iLink long polling in Clavi Agent; public webhook ingress is not available."
)


class WeChatAdapter(ChannelAdapter):
    """Adapter that handles internal WeChat iLink events and outbound text replies."""

    kind = "wechat"

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._transport = transport

    @property
    def display_name(self) -> str:
        return "WeChat"

    def verify_request(
        self,
        context: ChannelContext,
        request: ChannelRequest,
    ) -> RequestVerificationResult:
        payload = decode_json_body(request.body)
        if self._is_internal_ilink_request(request):
            return RequestVerificationResult(
                accepted=True,
                signature_valid=True,
                body_json=payload,
                metadata={
                    "connection_mode": "ilink_poll",
                    "verification_source": "internal_bridge",
                },
            )
        return RequestVerificationResult(
            accepted=False,
            signature_valid=False,
            reason=_WEBHOOK_UNSUPPORTED_REASON,
            body_json=payload,
            metadata={"connection_mode": "ilink_poll"},
        )

    def parse_inbound_event(
        self,
        context: ChannelContext,
        request: ChannelRequest,
        verification: RequestVerificationResult,
    ) -> ParsedInboundEvent:
        payload = verification.body_json if verification.body_json is not None else decode_json_body(request.body)
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, dict):
            raise RuntimeError("WeChat iLink payload is missing the message object.")

        provider_message_id = str(message.get("message_id") or "").strip()
        provider_event_id = str(message.get("client_id") or provider_message_id).strip()
        from_user_id = str(message.get("from_user_id") or "").strip()
        to_user_id = str(message.get("to_user_id") or "").strip()
        context_token = str(message.get("context_token") or "").strip()
        message_type_code = int(message.get("message_type") or 0)
        message_state = int(message.get("message_state") or 0)
        is_from_self = (
            message_type_code == BOT_MESSAGE_TYPE
            or from_user_id == context.get_secret("ilink_bot_id")
        )
        sender_id = to_user_id if is_from_self else from_user_id
        sender_type = "bot" if is_from_self else "user"
        text, attachments = self._extract_message_payload(message)

        return ParsedInboundEvent(
            integration_id=context.integration_id,
            channel_kind=self.kind,
            event_type="message",
            provider_event_id=provider_event_id or provider_message_id,
            provider_message_id=provider_message_id,
            provider_chat_id=sender_id,
            provider_thread_id="",
            provider_user_id=sender_id,
            tenant_id="",
            dedup_key=provider_message_id or provider_event_id or context_token,
            received_at=request.received_at,
            signature_valid=verification.signature_valid,
            message_type="text" if text else "media",
            text=text,
            sender_name=sender_id,
            is_group=False,
            locale="zh-CN",
            timezone="Asia/Shanghai",
            raw_headers=request.headers,
            raw_payload=payload,
            attachments=attachments,
            metadata={
                "provider_event_type": "ilink.getupdates",
                "context_token": context_token,
                "sender_type": sender_type,
                "is_from_self": is_from_self,
                "message_type_code": message_type_code,
                "message_state": message_state,
                "to_user_id": to_user_id,
                "from_user_id": from_user_id,
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
                "provider_event_type": event.metadata.get("provider_event_type", ""),
                "message_type": event.message_type,
                "context_token": event.metadata.get("context_token", ""),
                "sender_type": event.metadata.get("sender_type", ""),
            },
        )

    def emit_quick_ack(
        self,
        context: ChannelContext,
        event: ParsedInboundEvent,
    ) -> QuickAckIntent:
        return QuickAckIntent(status_code=200)

    async def send_outbound_message(
        self,
        context: ChannelContext,
        message: OutboundMessage,
    ) -> OutboundSendResult:
        try:
            client = self._build_client(context)
        except Exception as exc:
            return OutboundSendResult(ok=False, error=self.normalize_error(exc))

        target_id = str(message.target_id or context.integration.config.get("default_chat_id") or "").strip()
        if not target_id:
            return OutboundSendResult(
                ok=False,
                error=self.normalize_error(
                    {
                        "code": "missing_target_id",
                        "message": "WeChat outbound delivery requires a target user id.",
                    }
                ),
            )

        text = str(message.text or message.content.get("text") or "").strip()
        if not text:
            return OutboundSendResult(
                ok=False,
                error=self.normalize_error(
                    {
                        "code": "missing_text",
                        "message": "WeChat outbound message text cannot be empty.",
                    }
                ),
            )

        payload = {
            "msg": {
                "from_user_id": client.bot_id,
                "to_user_id": target_id,
                "client_id": str(message.dedup_key or uuid.uuid4().hex),
                "message_type": BOT_MESSAGE_TYPE,
                "message_state": MESSAGE_STATE_FINISH,
                "item_list": [
                    {
                        "type": ITEM_TYPE_TEXT,
                        "text_item": {"text": text},
                    }
                ],
                "context_token": str(message.metadata.get("context_token") or "").strip(),
            },
            "base_info": {},
        }

        try:
            response_payload = await client.send_message(payload)
        except Exception as exc:
            return OutboundSendResult(ok=False, error=self.normalize_error(exc))

        if int(response_payload.get("ret") or 0) != 0:
            return OutboundSendResult(
                ok=False,
                raw_response=response_payload,
                error=self.normalize_error(response_payload),
            )

        return OutboundSendResult(
            ok=True,
            provider_chat_id=target_id,
            provider_thread_id=str(message.thread_id or ""),
            raw_response=response_payload,
        )

    async def send_outbound_file(
        self,
        context: ChannelContext,
        file: OutboundFile,
    ) -> OutboundSendResult:
        fallback_text = str(file.text_fallback or "").strip()
        if not fallback_text:
            fallback_text = f"A file was generated: {file.file_name or file.provider_file_id or 'artifact'}"
        return await self.send_outbound_message(
            context,
            OutboundMessage(
                target_id=file.target_id,
                reply_to_message_id=file.reply_to_message_id,
                thread_id=file.thread_id,
                message_type="text",
                text=fallback_text,
                dedup_key=file.dedup_key or uuid.uuid4().hex,
                metadata=dict(file.metadata),
            ),
        )

    def normalize_error(self, error: Exception | dict[str, Any]) -> NormalizedAdapterError:
        if isinstance(error, dict):
            return NormalizedAdapterError(
                code=str(error.get("code") or error.get("errcode") or "wechat_error"),
                message=str(error.get("message") or error.get("errmsg") or "WeChat request failed."),
                retryable=bool(error.get("retryable", False)),
                status_code=error.get("status_code"),
                raw_error=error,
            )
        return NormalizedAdapterError(
            code="wechat_error",
            message=str(error) or "WeChat request failed.",
            retryable=False,
            raw_error=repr(error),
        )

    def _build_client(self, context: ChannelContext) -> WeChatILinkClient:
        credentials = WeChatILinkCredentials(
            bot_token=context.require_secret("bot_token"),
            ilink_bot_id=context.require_secret("ilink_bot_id"),
            base_url=context.get_secret("base_url") or "https://ilinkai.weixin.qq.com",
            ilink_user_id=context.get_secret("ilink_user_id"),
        )
        return WeChatILinkClient(credentials, transport=self._transport)

    @staticmethod
    def _is_internal_ilink_request(request: ChannelRequest) -> bool:
        return (
            request.method in {"ILINK_POLL", "LONG_POLL"}
            and request.header("x-wechat-connection-mode").strip().lower() == "ilink_poll"
        )

    def _extract_message_payload(self, message: dict[str, Any]) -> tuple[str, list[MsgAttachment]]:
        texts: list[str] = []
        attachments: list[MsgAttachment] = []
        for item in list(message.get("item_list") or []):
            if not isinstance(item, dict):
                continue
            item_type = int(item.get("type") or 0)
            if item_type == ITEM_TYPE_TEXT:
                text_item = item.get("text_item") or {}
                text = str(text_item.get("text") or "").strip()
                if text:
                    texts.append(text)
                continue
            if item_type == ITEM_TYPE_VOICE:
                voice_item = item.get("voice_item") or {}
                voice_text = str(voice_item.get("text") or "").strip()
                if voice_text:
                    texts.append(voice_text)
                else:
                    texts.append("[Voice message]")
                attachments.append(
                    MsgAttachment(
                        kind="audio",
                        metadata={"voice_item": voice_item},
                    )
                )
                continue
            if item_type == ITEM_TYPE_IMAGE:
                attachments.append(
                    MsgAttachment(
                        kind="image",
                        metadata={"image_item": item.get("image_item") or {}},
                    )
                )
                if not texts:
                    texts.append("[Image message]")
                continue
            if item_type == ITEM_TYPE_VIDEO:
                attachments.append(
                    MsgAttachment(
                        kind="media",
                        metadata={"video_item": item.get("video_item") or {}},
                    )
                )
                if not texts:
                    texts.append("[Video message]")
                continue
            if item_type == ITEM_TYPE_FILE:
                file_item = item.get("file_item") or {}
                attachments.append(
                    MsgAttachment(
                        kind="file",
                        name=str(file_item.get("file_name") or "").strip(),
                        metadata={"file_item": file_item},
                    )
                )
                if not texts:
                    texts.append("[File message]")

        normalized_text = "\n".join(part for part in texts if part).strip()
        return normalized_text, attachments

