"""飞书渠道适配器。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import mimetypes
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

import httpx

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

_LARK_LOCALE_PATTERN = re.compile(r"LarkLocale/([A-Za-z_]+)")
_FEISHU_LOCALE_KEYS = ("zh_cn", "en_us", "ja_jp")
_SHARE_CARD_TYPE_LABELS = {
    "share_chat": "[shared chat]",
    "share_user": "[shared user]",
    "share_calendar_event": "[shared calendar event]",
    "system": "[system message]",
    "merge_forward": "[merged forward messages]",
}


class FeishuAdapter(ChannelAdapter):
    """飞书事件订阅与回写适配器。"""

    kind = "feishu"
    DEFAULT_CONNECTION_MODE = "long_connection"
    _REPLY_CONTEXT_MAX_LEN = 200

    def __init__(self, *, transport: httpx.AsyncBaseTransport | None = None, timeout_seconds: float = 10.0):
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._tenant_token_cache: dict[str, tuple[str, float]] = {}

    @property
    def display_name(self) -> str:
        return "Feishu"

    def verify_request(
        self,
        context: ChannelContext,
        request: ChannelRequest,
    ) -> RequestVerificationResult:
        payload = decode_json_body(request.body)
        if self._is_internal_long_connection_request(request):
            return RequestVerificationResult(
                accepted=True,
                signature_valid=True,
                body_json=payload,
                metadata={
                    "connection_mode": "long_connection",
                    "verification_source": "internal_bridge",
                },
            )
        encrypt_key = context.get_secret("encrypt_key")
        signature = request.header("X-Lark-Signature")
        timestamp = request.header("X-Lark-Request-Timestamp")
        nonce = request.header("X-Lark-Request-Nonce")

        signature_valid = True
        if signature:
            if not encrypt_key:
                return RequestVerificationResult(
                    accepted=False,
                    signature_valid=False,
                    reason="飞书签名校验需要配置 encrypt_key。",
                    body_json=payload,
                )
            expected_signature = self._build_signature(
                timestamp=timestamp,
                nonce=nonce,
                encrypt_key=encrypt_key,
                body=request.body,
            )
            signature_valid = bool(timestamp and nonce) and hmac.compare_digest(
                signature,
                expected_signature,
            )
            if not signature_valid:
                return RequestVerificationResult(
                    accepted=False,
                    signature_valid=False,
                    reason="飞书请求签名不合法。",
                    body_json=payload,
                )

            max_age_seconds = self._signature_max_age_seconds(context)
            if max_age_seconds is not None and not self._within_time_window(
                request.received_at,
                timestamp,
                max_age_seconds,
            ):
                return RequestVerificationResult(
                    accepted=False,
                    signature_valid=False,
                    reason="飞书请求时间戳超出允许窗口。",
                    body_json=payload,
                )

        resolved_payload = payload
        if "encrypt" in payload:
            try:
                resolved_payload = self._decrypt_payload(payload["encrypt"], encrypt_key)
            except ValueError as exc:
                return RequestVerificationResult(
                    accepted=False,
                    signature_valid=signature_valid,
                    reason=str(exc),
                    body_json=payload,
                )

        expected_token = context.get_secret("verification_token") or context.get_secret("verify_token")
        request_token = str(
            resolved_payload.get("header", {}).get("token")
            or resolved_payload.get("token")
            or ""
        ).strip()
        token_valid = not expected_token or request_token == expected_token
        if not token_valid:
            return RequestVerificationResult(
                accepted=False,
                signature_valid=False,
                reason="飞书 verification token 不匹配。",
                body_json=resolved_payload,
            )

        return RequestVerificationResult(
            accepted=True,
            signature_valid=token_valid and signature_valid,
            body_json=resolved_payload,
            metadata={"encrypted": "encrypt" in payload},
        )

    def parse_inbound_event(
        self,
        context: ChannelContext,
        request: ChannelRequest,
        verification: RequestVerificationResult,
    ) -> ParsedInboundEvent:
        payload = verification.body_json if verification.body_json is not None else decode_json_body(request.body)
        if str(payload.get("type") or "") == "url_verification":
            challenge = str(payload.get("challenge") or "")
            provider_event_id = challenge or "url_verification"
            return ParsedInboundEvent(
                integration_id=context.integration_id,
                channel_kind=self.kind,
                event_type="url_verification",
                provider_event_id=provider_event_id,
                dedup_key=f"url_verification:{provider_event_id}",
                received_at=request.received_at,
                signature_valid=verification.signature_valid,
                raw_headers=request.headers,
                raw_payload=payload,
                metadata={"challenge": challenge},
            )

        header = payload.get("header") or {}
        event = payload.get("event") or {}
        message = event.get("message") or {}
        sender = event.get("sender") or {}
        raw_event_type = str(header.get("event_type") or "")
        event_type = self._normalize_event_type(raw_event_type)
        sender_id_value, sender_id_type = self._pick_user_id(sender.get("sender_id") or {})
        raw_mentions = list(message.get("mentions", []) or [])
        mentions = [self._parse_mention(item) for item in raw_mentions]
        content_payload = self._parse_message_content(message.get("content"))
        message_type = str(message.get("message_type") or "text")
        text, attachments = self._extract_message_payload(
            context=context,
            message_id=str(message.get("message_id") or ""),
            message_type=message_type,
            content_payload=content_payload,
            raw_mentions=raw_mentions,
        )
        locale = self._extract_locale(str(message.get("user_agent") or ""))
        provider_message_id = str(message.get("message_id") or "")
        provider_event_id = str(header.get("event_id") or provider_message_id)
        provider_chat_id = str(message.get("chat_id") or "")
        provider_thread_id = str(message.get("thread_id") or message.get("root_id") or "")

        return ParsedInboundEvent(
            integration_id=context.integration_id,
            channel_kind=self.kind,
            event_type=event_type,
            provider_event_id=provider_event_id,
            provider_message_id=provider_message_id,
            provider_chat_id=provider_chat_id,
            provider_thread_id=provider_thread_id,
            provider_user_id=sender_id_value,
            tenant_id=str(header.get("tenant_key") or sender.get("tenant_key") or ""),
            dedup_key=provider_message_id or provider_event_id,
            received_at=request.received_at,
            signature_valid=verification.signature_valid,
            message_type=message_type,
            text=text,
            sender_name=str(sender.get("sender_name") or sender.get("name") or ""),
            is_group=str(message.get("chat_type") or "") == "group",
            locale=locale,
            timezone="",
            raw_headers=request.headers,
            raw_payload=payload,
            attachments=attachments,
            mentions=mentions,
            metadata={
                "sender_id_type": sender_id_type,
                "sender_type": str(sender.get("sender_type") or ""),
                "chat_type": str(message.get("chat_type") or ""),
                "root_id": str(message.get("root_id") or ""),
                "parent_id": str(message.get("parent_id") or ""),
                "provider_event_type": raw_event_type or event_type,
                "user_agent": str(message.get("user_agent") or ""),
                "app_id": str(header.get("app_id") or ""),
                "content_payload": content_payload,
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
                "root_id": event.metadata.get("root_id", ""),
                "parent_id": event.metadata.get("parent_id", ""),
                "sender_id_type": event.metadata.get("sender_id_type", ""),
            },
        )

    async def prepare_msg_context(
        self,
        context: ChannelContext,
        msg_context: MsgContextEnvelope,
    ) -> MsgContextEnvelope:
        parent_id = str(msg_context.metadata.get("parent_id") or "").strip()
        if not parent_id:
            return msg_context

        reply_context = await self._get_reply_context_text(context, parent_id)
        if not reply_context:
            return msg_context

        normalized_text = str(msg_context.text or "").strip()
        next_text = reply_context if not normalized_text else f"{reply_context}\n{normalized_text}"
        metadata = dict(msg_context.metadata)
        metadata["reply_context"] = reply_context
        return msg_context.model_copy(update={"text": next_text, "metadata": metadata})

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
        return QuickAckIntent()

    def build_quick_reaction(
        self,
        context: ChannelContext,
        msg_context: MsgContextEnvelope,
    ) -> OutboundReaction | None:
        if not self._quick_reaction_enabled(context):
            return None
        if not context.get_secret("app_id") or not context.get_secret("app_secret"):
            return None
        message_id = str(msg_context.message_id or "").strip()
        event_type = str(msg_context.metadata.get("event_type") or "").strip().lower()
        if not message_id or event_type != "message":
            return None
        reaction_type = self._quick_reaction_emoji_type(context)
        if not reaction_type:
            return None
        return OutboundReaction(
            message_id=message_id,
            reaction_type=reaction_type,
            dedup_key=f"{context.integration_id}:{message_id}:{reaction_type}:quick-reaction",
            metadata={
                "provider_chat_id": str(msg_context.chat_id or ""),
                "provider_thread_id": str(msg_context.thread_id or ""),
            },
        )

    async def prepare_upload_download(
        self,
        context: ChannelContext,
        attachment: MsgAttachment,
    ) -> UploadBridgeHint | None:
        hint = attachment.upload_hint
        if hint is None:
            return None

        auth_scheme = str(hint.metadata.get("auth_scheme") or "").strip().lower()
        if auth_scheme != "tenant_access_token":
            return hint

        access_token = await self._get_tenant_access_token(context)
        headers = dict(hint.headers)
        headers["Authorization"] = f"Bearer {access_token}"
        return hint.model_copy(update={"headers": headers})

    async def send_outbound_message(
        self,
        context: ChannelContext,
        message: OutboundMessage,
    ) -> OutboundSendResult:
        try:
            access_token = await self._get_tenant_access_token(context)
        except Exception as exc:
            return OutboundSendResult(
                ok=False,
                error=self.normalize_error(exc),
            )
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }
        payload = {
            "content": json.dumps(message.content, ensure_ascii=False),
            "msg_type": message.message_type,
        }
        if message.dedup_key:
            payload["uuid"] = self._normalize_message_uuid(message.dedup_key)

        reply_target_message_id = self._resolve_reply_target_message_id(context, message)
        if reply_target_message_id:
            path = f"/open-apis/im/v1/messages/{reply_target_message_id}/reply"
            if message.thread_id:
                payload["reply_in_thread"] = True
            return await self._post_feishu_api(
                context=context,
                path=path,
                headers=headers,
                payload=payload,
                default_chat_id=message.target_id,
                default_thread_id=message.thread_id,
            )

        payload["receive_id"] = message.target_id
        return await self._post_feishu_api(
            context=context,
            path="/open-apis/im/v1/messages",
            headers=headers,
            payload=payload,
            params={"receive_id_type": message.target_id_type},
            default_chat_id=message.target_id,
            default_thread_id=message.thread_id,
        )

    async def send_outbound_file(
        self,
        context: ChannelContext,
        file: OutboundFile,
    ) -> OutboundSendResult:
        if not file.provider_file_id:
            try:
                file = await self._prepare_generated_file_for_send(context, file)
            except Exception as exc:
                return OutboundSendResult(
                    ok=False,
                    error=self.normalize_error(exc),
                )
        if not file.provider_file_id:
            text = file.text_fallback or self._build_file_fallback_text(file)
            return await self.send_outbound_message(
                context,
                OutboundMessage(
                    target_id=file.target_id,
                    target_id_type=file.target_id_type,
                    reply_to_message_id=file.reply_to_message_id,
                    thread_id=file.thread_id,
                    message_type="text",
                    text=text,
                    dedup_key=file.dedup_key,
                ),
            )

        send_kind = file.file_kind
        if send_kind != "file" and str(file.metadata.get("provider_origin") or "").strip() == "generated_upload":
            send_kind = "file"

        if send_kind == "image":
            content = {"image_key": file.provider_file_id}
            message_type = "image"
        elif send_kind == "media":
            image_key = str(file.metadata.get("image_key") or "")
            if not image_key:
                return OutboundSendResult(
                    ok=False,
                    error=NormalizedAdapterError(
                        code="feishu_media_requires_image_key",
                        message="飞书视频消息需要同时提供 image_key。",
                        retryable=False,
                    ),
                )
            content = {
                "file_key": file.provider_file_id,
                "image_key": image_key,
                "file_name": file.file_name,
            }
            message_type = "media"
        else:
            content = {"file_key": file.provider_file_id}
            message_type = "audio" if send_kind == "audio" else "file"

        return await self.send_outbound_message(
            context,
            OutboundMessage(
                target_id=file.target_id,
                target_id_type=file.target_id_type,
                reply_to_message_id=file.reply_to_message_id,
                thread_id=file.thread_id,
                message_type=message_type,
                content=content,
                dedup_key=file.dedup_key,
            ),
        )

    async def _prepare_generated_file_for_send(
        self,
        context: ChannelContext,
        file: OutboundFile,
    ) -> OutboundFile:
        local_path = self._resolve_local_outbound_file_path(file)
        if not local_path and not str(file.url or "").strip():
            return file

        file_name, content_bytes, mime_type = await self._load_outbound_file_bytes(
            context=context,
            file=file,
            local_path=local_path,
        )
        if not content_bytes:
            raise ValueError("Feishu outbound file upload produced empty content.")

        access_token = await self._get_tenant_access_token(context)
        provider_file_id = await self._upload_outbound_file_bytes(
            context=context,
            access_token=access_token,
            file_name=file_name,
            mime_type=mime_type,
            file_kind=file.file_kind,
            content_bytes=content_bytes,
        )
        metadata = dict(file.metadata)
        metadata["provider_origin"] = "generated_upload"
        if local_path:
            metadata["local_path"] = local_path
        return file.model_copy(
            update={
                "provider_file_id": provider_file_id,
                "file_name": file_name,
                "metadata": metadata,
            }
        )

    async def _load_outbound_file_bytes(
        self,
        *,
        context: ChannelContext,
        file: OutboundFile,
        local_path: str,
    ) -> tuple[str, bytes, str]:
        if local_path:
            path_obj = Path(local_path)
            content_bytes = path_obj.read_bytes()
            file_name = file.file_name or path_obj.name
            mime_type = (
                str(file.metadata.get("mime_type") or "").strip()
                or mimetypes.guess_type(file_name)[0]
                or "application/octet-stream"
            )
            return file_name, content_bytes, mime_type

        download_url = self._resolve_outbound_download_url(context, file.url)
        if not download_url:
            raise ValueError("Feishu outbound file is missing a downloadable URL.")

        async with httpx.AsyncClient(
            timeout=self._timeout_seconds,
            transport=self._transport,
            follow_redirects=True,
        ) as client:
            response = await client.get(download_url)
        response.raise_for_status()

        content_type = str(response.headers.get("content-type") or "").split(";")[0].strip()
        file_name = (
            file.file_name
            or self._filename_from_content_disposition(
                str(response.headers.get("content-disposition") or "")
            )
            or unquote(Path(urlparse(download_url).path).name)
            or "artifact"
        )
        mime_type = (
            str(file.metadata.get("mime_type") or "").strip()
            or content_type
            or mimetypes.guess_type(file_name)[0]
            or "application/octet-stream"
        )
        return file_name, response.content, mime_type

    async def _upload_outbound_file_bytes(
        self,
        *,
        context: ChannelContext,
        access_token: str,
        file_name: str,
        mime_type: str,
        file_kind: str,
        content_bytes: bytes,
    ) -> str:
        api_base = self._api_base_url(context)
        file_type = self._resolve_upload_file_type(
            file_name=file_name,
            mime_type=mime_type,
            file_kind=file_kind,
        )
        headers = {"Authorization": f"Bearer {access_token}"}
        files = {
            "file": (file_name, content_bytes, mime_type or "application/octet-stream"),
        }
        data = {
            "file_type": file_type,
            "file_name": file_name,
        }

        async with httpx.AsyncClient(
            base_url=api_base,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            response = await client.post(
                "/open-apis/im/v1/files",
                headers=headers,
                data=data,
                files=files,
            )

        payload = self._safe_json(response)
        if response.status_code >= 400 or int(payload.get("code") or 0) != 0:
            normalized = self._normalize_error_payload(payload, response.status_code)
            raise RuntimeError(
                f"{normalized.message} (file_name={file_name}, file_type={file_type}, status={response.status_code})"
            )

        file_key = str((payload.get("data") or {}).get("file_key") or "").strip()
        if not file_key:
            raise RuntimeError("Feishu upload response missing file_key.")
        return file_key

    async def send_outbound_reaction(
        self,
        context: ChannelContext,
        reaction: OutboundReaction,
    ) -> OutboundSendResult:
        try:
            access_token = await self._get_tenant_access_token(context)
        except Exception as exc:
            return OutboundSendResult(
                ok=False,
                error=self.normalize_error(exc),
            )

        return await self._post_feishu_api(
            context=context,
            path=f"/open-apis/im/v1/messages/{reaction.message_id}/reactions",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            payload={
                "reaction_type": {
                    "emoji_type": reaction.reaction_type,
                }
            },
        )

    def normalize_error(self, error: Exception | dict[str, Any]) -> NormalizedAdapterError:
        if isinstance(error, httpx.TimeoutException):
            return NormalizedAdapterError(
                code="timeout",
                message="飞书接口请求超时。",
                retryable=True,
                raw_error={"type": type(error).__name__},
            )
        if isinstance(error, httpx.HTTPError):
            response = error.response
            payload = None
            if response is not None:
                try:
                    payload = response.json()
                except ValueError:
                    payload = {"body": response.text}
            return self._normalize_error_payload(payload or {}, response.status_code if response else None)
        if isinstance(error, dict):
            return self._normalize_error_payload(error, error.get("status_code"))
        return NormalizedAdapterError(
            code="unknown_error",
            message=str(error),
            raw_error={"type": type(error).__name__},
        )

    async def _post_feishu_api(
        self,
        *,
        context: ChannelContext,
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        params: dict[str, Any] | None = None,
        default_chat_id: str = "",
        default_thread_id: str = "",
    ) -> OutboundSendResult:
        api_base = self._api_base_url(context)
        async with httpx.AsyncClient(
            base_url=api_base,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            response = await client.post(path, headers=headers, params=params, json=payload)

        response_payload = self._safe_json(response)
        if response.status_code >= 400:
            return OutboundSendResult(
                ok=False,
                http_status=response.status_code,
                raw_response=response_payload,
                error=self._normalize_error_payload(response_payload, response.status_code),
            )

        if isinstance(response_payload, dict) and int(response_payload.get("code") or 0) != 0:
            return OutboundSendResult(
                ok=False,
                http_status=response.status_code,
                raw_response=response_payload,
                error=self._normalize_error_payload(response_payload, response.status_code),
            )

        data = {}
        if isinstance(response_payload, dict):
            data = response_payload.get("data") or {}
        return OutboundSendResult(
            ok=True,
            provider_message_id=str(data.get("message_id") or ""),
            provider_chat_id=str(data.get("chat_id") or default_chat_id),
            provider_thread_id=str(data.get("thread_id") or default_thread_id),
            http_status=response.status_code,
            raw_response=response_payload,
            metadata={
                "root_id": str(data.get("root_id") or ""),
                "parent_id": str(data.get("parent_id") or ""),
            },
        )

    async def _get_tenant_access_token(self, context: ChannelContext) -> str:
        cache_key = context.integration_id
        cached = self._tenant_token_cache.get(cache_key)
        if cached is not None and cached[1] > time.time() + 60:
            return cached[0]

        app_id = context.require_secret("app_id")
        app_secret = context.require_secret("app_secret")
        api_base = self._api_base_url(context)
        async with httpx.AsyncClient(
            base_url=api_base,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            response = await client.post(
                "/open-apis/auth/v3/tenant_access_token/internal",
                headers={"Content-Type": "application/json; charset=utf-8"},
                json={"app_id": app_id, "app_secret": app_secret},
            )

        payload = self._safe_json(response)
        if response.status_code >= 400 or int(payload.get("code") or 0) != 0:
            normalized = self._normalize_error_payload(payload, response.status_code)
            raise RuntimeError(normalized.message)

        access_token = str(payload.get("tenant_access_token") or "")
        if not access_token:
            raise RuntimeError("飞书 tenant_access_token 响应缺少 access token。")
        expire_seconds = int(payload.get("expire") or 7200)
        self._tenant_token_cache[cache_key] = (access_token, time.time() + expire_seconds)
        return access_token

    def _extract_message_payload(
        self,
        *,
        context: ChannelContext,
        message_id: str,
        message_type: str,
        content_payload: Any,
        raw_mentions: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[MsgAttachment]]:
        if message_type == "text":
            if isinstance(content_payload, dict):
                text = str(content_payload.get("text") or content_payload.get("raw_text") or "")
            else:
                text = str(content_payload or "")
            return self._resolve_mentions_text(text, raw_mentions), []

        if message_type == "image":
            image_key = str((content_payload or {}).get("image_key") or "")
            attachment = self._build_message_resource_attachment(
                context=context,
                message_id=message_id,
                resource_key=image_key,
                resource_type="image",
                kind="image",
            )
            return "", [attachment] if attachment is not None else []

        if message_type in {"file", "audio"}:
            file_key = str((content_payload or {}).get("file_key") or "")
            attachment = self._build_message_resource_attachment(
                context=context,
                message_id=message_id,
                resource_key=file_key,
                resource_type="file",
                kind="audio" if message_type == "audio" else "file",
                file_name=str((content_payload or {}).get("file_name") or ""),
            )
            return "", [attachment] if attachment is not None else []

        if message_type == "media":
            file_key = str((content_payload or {}).get("file_key") or "")
            attachment = self._build_message_resource_attachment(
                context=context,
                message_id=message_id,
                resource_key=file_key,
                resource_type="file",
                kind="media",
                file_name=str((content_payload or {}).get("file_name") or ""),
                extra_metadata={"image_key": str((content_payload or {}).get("image_key") or "")},
            )
            return "", [attachment] if attachment is not None else []

        if message_type == "post":
            return self._extract_post_payload(context, message_id, content_payload or {})

        if message_type == "interactive":
            return self._extract_share_card_content(content_payload or {}, message_type), []

        if message_type in _SHARE_CARD_TYPE_LABELS:
            return self._extract_share_card_content(content_payload or {}, message_type), []

        return f"[飞书消息类型 {message_type}]", []

    def _extract_post_payload(
        self,
        context: ChannelContext,
        message_id: str,
        content_payload: dict[str, Any],
    ) -> tuple[str, list[MsgAttachment]]:
        post_block = self._select_post_block(content_payload)
        if post_block is None:
            return "", []

        texts: list[str] = []
        attachments: list[MsgAttachment] = []
        title = str(post_block.get("title") or "").strip()
        if title:
            texts.append(title)

        for row in post_block.get("content", []) or []:
            for item in row:
                tag = str(item.get("tag") or "")
                if tag == "text":
                    texts.append(str(item.get("text") or ""))
                elif tag == "a":
                    link_text = str(item.get("text") or item.get("href") or "").strip()
                    if link_text:
                        texts.append(link_text)
                elif tag == "at":
                    mention_label = str(item.get("user_name") or item.get("user_id") or "").strip()
                    if mention_label:
                        texts.append(f"@{mention_label}")
                elif tag == "img":
                    attachment = self._build_message_resource_attachment(
                        context=context,
                        message_id=message_id,
                        resource_key=str(item.get("image_key") or ""),
                        resource_type="image",
                        kind="image",
                    )
                    if attachment is not None:
                        attachments.append(attachment)
                elif tag == "media":
                    attachment = self._build_message_resource_attachment(
                        context=context,
                        message_id=message_id,
                        resource_key=str(item.get("file_key") or ""),
                        resource_type="file",
                        kind="media",
                        extra_metadata={"image_key": str(item.get("image_key") or "")},
                    )
                    if attachment is not None:
                        attachments.append(attachment)
                elif tag == "code_block":
                    code_text = str(item.get("text") or "").strip()
                    language = str(item.get("language") or "").strip()
                    if code_text:
                        fence = f"```{language}" if language else "```"
                        texts.append(f"{fence}\n{code_text}\n```")
        return "\n".join(part for part in texts if part).strip(), attachments

    def _build_message_resource_attachment(
        self,
        *,
        context: ChannelContext,
        message_id: str,
        resource_key: str,
        resource_type: str,
        kind: str,
        file_name: str = "",
        extra_metadata: dict[str, Any] | None = None,
    ) -> MsgAttachment | None:
        if not resource_key:
            return None
        download_url = (
            f"{self._api_base_url(context)}/open-apis/im/v1/messages/"
            f"{message_id}/resources/{resource_key}?type={resource_type}"
        )
        upload_hint = UploadBridgeHint(
            source_kind="provider_resource",
            download_url=download_url,
            resource_type=resource_type,
            provider_file_id=resource_key,
            provider_message_id=message_id,
            suggested_filename=file_name,
            metadata={"auth_scheme": "tenant_access_token"},
        )
        return MsgAttachment(
            kind=kind if kind in {"image", "file", "audio", "media"} else "unknown",
            provider_file_id=resource_key,
            provider_message_id=message_id,
            name=file_name,
            download_url=download_url,
            upload_hint=upload_hint,
            metadata=extra_metadata or {},
        )

    @staticmethod
    def _parse_message_content(content: Any) -> Any:
        if isinstance(content, dict):
            return content
        if not isinstance(content, str):
            return {}
        normalized = content.strip()
        if not normalized:
            return {}
        try:
            return json.loads(normalized)
        except ValueError:
            return {"raw_text": normalized}

    async def _get_reply_context_text(
        self,
        context: ChannelContext,
        message_id: str,
    ) -> str:
        try:
            access_token = await self._get_tenant_access_token(context)
        except Exception:
            return ""

        api_base = self._api_base_url(context)
        async with httpx.AsyncClient(
            base_url=api_base,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = await client.get(
                    f"/open-apis/im/v1/messages/{message_id}",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
            except httpx.HTTPError:
                return ""

        payload = self._safe_json(response)
        if response.status_code >= 400 or int(payload.get("code") or 0) != 0:
            return ""

        message = self._extract_message_from_get_payload(payload)
        if not message:
            return ""

        msg_type = str(message.get("msg_type") or message.get("message_type") or "").strip()
        raw_content = message.get("content")
        if raw_content is None:
            body = message.get("body") or {}
            raw_content = body.get("content")
        content_payload = self._parse_message_content(raw_content)
        text = self._extract_reply_text(msg_type, content_payload).strip()
        if not text:
            return ""
        if len(text) > self._REPLY_CONTEXT_MAX_LEN:
            text = f"{text[: self._REPLY_CONTEXT_MAX_LEN]}..."
        return f"[Reply to: {text}]"

    def _resolve_reply_target_message_id(
        self,
        context: ChannelContext,
        message: OutboundMessage,
    ) -> str:
        direct_reply_message_id = str(message.reply_to_message_id or "").strip()
        root_message_id = str(
            message.metadata.get("provider_root_message_id")
            or message.metadata.get("root_id")
            or ""
        ).strip()
        if str(message.thread_id or "").strip():
            return root_message_id or direct_reply_message_id
        if self._reply_to_message_enabled(context):
            return direct_reply_message_id
        return ""

    def _reply_to_message_enabled(self, context: ChannelContext) -> bool:
        raw_value = context.integration.config.get("reply_to_message")
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value or "").strip().lower()
        return normalized in {"1", "true", "yes", "on"}

    def _quick_reaction_enabled(self, context: ChannelContext) -> bool:
        raw_value = context.integration.config.get("quick_reaction_enabled")
        if raw_value is None:
            return True
        if isinstance(raw_value, bool):
            return raw_value
        normalized = str(raw_value or "").strip().lower()
        if not normalized:
            return True
        return normalized in {"1", "true", "yes", "on"}

    def _quick_reaction_emoji_type(self, context: ChannelContext) -> str:
        return str(context.integration.config.get("quick_reaction_emoji_type") or "DONE").strip()

    @classmethod
    def connection_mode(cls, config: dict[str, Any] | None) -> str:
        raw_value = str((config or {}).get("connection_mode") or "").strip().lower()
        normalized = raw_value.replace("-", "_")
        if normalized in {"", "long_connection", "longconnection", "ws", "websocket"}:
            return cls.DEFAULT_CONNECTION_MODE
        if normalized in {"webhook", "event_subscription"}:
            return "webhook"
        return cls.DEFAULT_CONNECTION_MODE

    @classmethod
    def uses_long_connection(cls, config: dict[str, Any] | None) -> bool:
        return cls.connection_mode(config) == cls.DEFAULT_CONNECTION_MODE

    @staticmethod
    def _is_internal_long_connection_request(request: ChannelRequest) -> bool:
        return (
            str(request.method or "").strip().upper() == "LONG_CONNECTION"
            and request.header("x-feishu-connection-mode").strip().lower() == "long_connection"
        )

    @staticmethod
    def _normalize_event_type(raw_event_type: str) -> str:
        normalized = str(raw_event_type or "").strip()
        if normalized.startswith("im.message.receive"):
            return "message"
        return normalized or "message"

    @staticmethod
    def _resolve_mentions_text(
        text: str,
        raw_mentions: list[dict[str, Any]] | None,
    ) -> str:
        normalized = str(text or "")
        if not normalized or not raw_mentions:
            return normalized

        resolved = normalized
        for raw_mention in raw_mentions:
            key = str(raw_mention.get("key") or "").strip()
            if not key or key not in resolved:
                continue
            name = str(raw_mention.get("name") or key).strip() or key
            mention_id = raw_mention.get("id") or {}
            open_id = str(mention_id.get("open_id") or "").strip()
            user_id = str(mention_id.get("user_id") or "").strip()
            if open_id and user_id:
                replacement = f"@{name} ({open_id}, user id: {user_id})"
            elif open_id:
                replacement = f"@{name} ({open_id})"
            else:
                replacement = f"@{name}"
            resolved = resolved.replace(key, replacement)
        return resolved

    @staticmethod
    def _select_post_block(content_payload: dict[str, Any]) -> dict[str, Any] | None:
        root = content_payload
        if isinstance(root, dict) and isinstance(root.get("post"), dict):
            root = root["post"]
        if not isinstance(root, dict):
            return None

        if isinstance(root.get("content"), list):
            return root

        for key in _FEISHU_LOCALE_KEYS:
            block = root.get(key)
            if isinstance(block, dict) and isinstance(block.get("content"), list):
                return block

        for value in root.values():
            if isinstance(value, dict) and isinstance(value.get("content"), list):
                return value
        return None

    def _extract_reply_text(
        self,
        message_type: str,
        content_payload: Any,
    ) -> str:
        normalized_message_type = str(message_type or "").strip()
        if normalized_message_type == "text":
            if isinstance(content_payload, dict):
                return str(content_payload.get("text") or content_payload.get("raw_text") or "")
            return str(content_payload or "")
        if normalized_message_type == "post":
            post_block = self._select_post_block(content_payload or {})
            if post_block is None:
                return ""
            texts: list[str] = []
            title = str(post_block.get("title") or "").strip()
            if title:
                texts.append(title)
            for row in post_block.get("content", []) or []:
                if not isinstance(row, list):
                    continue
                for item in row:
                    if not isinstance(item, dict):
                        continue
                    tag = str(item.get("tag") or "")
                    if tag in {"text", "a"}:
                        text = str(item.get("text") or item.get("href") or "").strip()
                        if text:
                            texts.append(text)
                    elif tag == "at":
                        label = str(item.get("user_name") or item.get("user_id") or "").strip()
                        if label:
                            texts.append(f"@{label}")
                    elif tag == "code_block":
                        code_text = str(item.get("text") or "").strip()
                        if code_text:
                            texts.append(code_text)
            return "\n".join(part for part in texts if part).strip()
        if normalized_message_type == "interactive":
            return self._extract_share_card_content(content_payload or {}, normalized_message_type)
        return ""

    @staticmethod
    def _extract_message_from_get_payload(payload: dict[str, Any]) -> dict[str, Any]:
        data = payload.get("data") or {}
        items = data.get("items")
        if isinstance(items, list) and items:
            first_item = items[0]
            if isinstance(first_item, dict):
                return first_item
        message = data.get("message")
        if isinstance(message, dict):
            return message
        return data if isinstance(data, dict) else {}

    def _extract_share_card_content(
        self,
        content_payload: Any,
        message_type: str,
    ) -> str:
        parts: list[str] = []
        if message_type == "interactive":
            parts.extend(self._extract_interactive_content(content_payload))
        else:
            label = _SHARE_CARD_TYPE_LABELS.get(message_type, f"[{message_type}]")
            parts.append(label)
            if isinstance(content_payload, dict):
                for key in ("chat_id", "user_id", "event_key"):
                    value = str(content_payload.get(key) or "").strip()
                    if value:
                        parts.append(value)
                        break
        return "\n".join(part for part in parts if part).strip() or f"[{message_type}]"

    def _extract_interactive_content(self, content_payload: Any) -> list[str]:
        if isinstance(content_payload, str):
            try:
                content_payload = json.loads(content_payload)
            except ValueError:
                normalized = content_payload.strip()
                return [normalized] if normalized else []
        if not isinstance(content_payload, dict):
            return []

        parts: list[str] = []
        title = content_payload.get("title")
        if isinstance(title, dict):
            title_text = str(title.get("content") or title.get("text") or "").strip()
            if title_text:
                parts.append(f"title: {title_text}")
        elif isinstance(title, str):
            normalized_title = title.strip()
            if normalized_title:
                parts.append(f"title: {normalized_title}")

        elements = content_payload.get("elements")
        if isinstance(elements, list):
            for element in elements:
                parts.extend(self._extract_interactive_element_content(element))

        card = content_payload.get("card")
        if isinstance(card, dict):
            parts.extend(self._extract_interactive_content(card))

        header = content_payload.get("header")
        if isinstance(header, dict):
            header_title = header.get("title")
            if isinstance(header_title, dict):
                header_text = str(header_title.get("content") or header_title.get("text") or "").strip()
                if header_text:
                    parts.append(f"title: {header_text}")

        return parts

    def _extract_interactive_element_content(self, element: Any) -> list[str]:
        if not isinstance(element, dict):
            return []

        parts: list[str] = []
        tag = str(element.get("tag") or "").strip()
        if tag in {"markdown", "lark_md"}:
            content = str(element.get("content") or "").strip()
            if content:
                parts.append(content)
        elif tag == "div":
            text = element.get("text")
            if isinstance(text, dict):
                content = str(text.get("content") or text.get("text") or "").strip()
                if content:
                    parts.append(content)
            elif isinstance(text, str):
                normalized_text = text.strip()
                if normalized_text:
                    parts.append(normalized_text)
            fields = element.get("fields")
            if isinstance(fields, list):
                for field in fields:
                    if not isinstance(field, dict):
                        continue
                    field_text = field.get("text")
                    if isinstance(field_text, dict):
                        content = str(field_text.get("content") or field_text.get("text") or "").strip()
                        if content:
                            parts.append(content)
        elif tag == "a":
            href = str(element.get("href") or "").strip()
            text = str(element.get("text") or "").strip()
            if text:
                parts.append(text)
            elif href:
                parts.append(f"link: {href}")
        elif tag == "button":
            text = element.get("text")
            if isinstance(text, dict):
                content = str(text.get("content") or text.get("text") or "").strip()
                if content:
                    parts.append(content)
            url = str(
                element.get("url")
                or (element.get("multi_url") or {}).get("url")
                or ""
            ).strip()
            if url:
                parts.append(f"link: {url}")
        elif tag == "img":
            alt = element.get("alt")
            if isinstance(alt, dict):
                alt_text = str(alt.get("content") or "").strip()
                parts.append(alt_text or "[image]")
            else:
                parts.append("[image]")
        elif tag == "note":
            note_elements = element.get("elements")
            if isinstance(note_elements, list):
                for note_element in note_elements:
                    parts.extend(self._extract_interactive_element_content(note_element))
        elif tag == "column_set":
            columns = element.get("columns")
            if isinstance(columns, list):
                for column in columns:
                    if not isinstance(column, dict):
                        continue
                    elements = column.get("elements")
                    if isinstance(elements, list):
                        for nested in elements:
                            parts.extend(self._extract_interactive_element_content(nested))
        elif tag == "plain_text":
            content = str(element.get("content") or "").strip()
            if content:
                parts.append(content)
        else:
            nested_elements = element.get("elements")
            if isinstance(nested_elements, list):
                for nested in nested_elements:
                    parts.extend(self._extract_interactive_element_content(nested))
        return parts

    @staticmethod
    def _parse_mention(raw_mention: dict[str, Any]) -> MsgMention:
        user_id, id_type = FeishuAdapter._pick_user_id(raw_mention.get("id") or {})
        return MsgMention(
            key=str(raw_mention.get("key") or ""),
            id=user_id,
            id_type=id_type,
            name=str(raw_mention.get("name") or ""),
            metadata={"tenant_key": str(raw_mention.get("tenant_key") or "")},
        )

    @staticmethod
    def _pick_user_id(user_id_bundle: dict[str, Any]) -> tuple[str, str]:
        for key in ("open_id", "user_id", "union_id"):
            value = str(user_id_bundle.get(key) or "").strip()
            if value:
                return value, key
        return "", ""

    @staticmethod
    def _extract_locale(user_agent: str) -> str:
        match = _LARK_LOCALE_PATTERN.search(user_agent)
        if not match:
            return ""
        return match.group(1).replace("_", "-")

    @staticmethod
    def _build_signature(
        *,
        timestamp: str,
        nonce: str,
        encrypt_key: str,
        body: bytes,
    ) -> str:
        builder = f"{timestamp}{nonce}{encrypt_key}{body.decode('utf-8')}"
        return hashlib.sha256(builder.encode("utf-8")).hexdigest()

    @staticmethod
    def _within_time_window(received_at: str, timestamp: str, max_age_seconds: int) -> bool:
        try:
            request_time = int(timestamp)
            if request_time > 10**12:
                request_time = request_time // 1000
            received_time = datetime.fromisoformat(received_at)
            if received_time.tzinfo is None:
                received_time = received_time.replace(tzinfo=timezone.utc)
            delta = abs(received_time.timestamp() - request_time)
            return delta <= max_age_seconds
        except (TypeError, ValueError):
            return False

    def _signature_max_age_seconds(self, context: ChannelContext) -> int | None:
        value = context.integration.config.get("signature_max_age_seconds")
        if value in {None, ""}:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_json(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = {"code": response.status_code, "msg": response.text}
        if isinstance(payload, dict):
            return payload
        return {"code": response.status_code, "msg": str(payload)}

    @staticmethod
    def _normalize_error_payload(
        payload: dict[str, Any],
        status_code: int | None,
    ) -> NormalizedAdapterError:
        code = str(payload.get("code") or payload.get("error_code") or "feishu_error")
        message = str(payload.get("msg") or payload.get("message") or "飞书接口调用失败。")
        retryable_codes = {"230020", "230049"}
        retryable = bool(status_code and status_code >= 500) or code in retryable_codes
        return NormalizedAdapterError(
            code=code,
            message=message,
            retryable=retryable,
            status_code=status_code,
            raw_error=payload,
        )

    @staticmethod
    def _build_file_fallback_text(file: OutboundFile) -> str:
        file_label = file.file_name or "文件"
        if file.url:
            return f"{file_label}\n{file.url}"
        return file_label

    @staticmethod
    def _resolve_local_outbound_file_path(file: OutboundFile) -> str:
        for candidate in (
            str(file.metadata.get("local_path") or "").strip(),
            str(file.metadata.get("uri") or "").strip(),
        ):
            if not candidate:
                continue
            if candidate.startswith("file://"):
                parsed = urlparse(candidate)
                candidate = unquote(parsed.path or "")
                if re.fullmatch(r"/[A-Za-z]:/.*", candidate):
                    candidate = candidate[1:]
            if "://" in candidate:
                continue
            path_obj = Path(candidate)
            if path_obj.is_file():
                return str(path_obj)
        return ""

    @staticmethod
    def _resolve_outbound_download_url(context: ChannelContext, url: str) -> str:
        normalized = str(url or "").strip()
        if not normalized:
            return ""
        if normalized.startswith(("http://", "https://")):
            return normalized
        if normalized.startswith("/"):
            public_base_url = str(
                context.integration.config.get("public_base_url")
                or os.environ.get("MINI_AGENT_PUBLIC_BASE_URL")
                or ""
            ).strip()
            if public_base_url:
                return urljoin(f"{public_base_url.rstrip('/')}/", normalized.lstrip("/"))
            return ""
        return normalized

    @staticmethod
    def _filename_from_content_disposition(content_disposition: str) -> str:
        if not content_disposition:
            return ""
        match = re.search(r"filename\*=UTF-8''([^;]+)", content_disposition, re.IGNORECASE)
        if match:
            return unquote(match.group(1).strip().strip('"'))
        match = re.search(r'filename="?([^";]+)"?', content_disposition, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    @staticmethod
    def _resolve_upload_file_type(*, file_name: str, mime_type: str, file_kind: str) -> str:
        suffix = Path(file_name or "").suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".doc", ".docx"}:
            return "doc"
        if suffix in {".xls", ".xlsx"}:
            return "xls"
        if suffix in {".ppt", ".pptx"}:
            return "ppt"
        return "stream"

    @staticmethod
    def _normalize_message_uuid(dedup_key: str) -> str:
        """Fit message dedup keys into Feishu's uuid length constraints."""
        normalized = str(dedup_key or "").strip()
        if len(normalized) <= 50:
            return normalized
        return f"ma-{hashlib.sha1(normalized.encode('utf-8')).hexdigest()}"

    @staticmethod
    def _api_base_url(context: ChannelContext) -> str:
        return str(context.integration.config.get("api_base_url") or "https://open.feishu.cn").rstrip("/")

    @staticmethod
    def _decrypt_payload(encrypted_payload: str, encrypt_key: str) -> dict[str, Any]:
        if not encrypt_key:
            raise ValueError("飞书加密事件体缺少 encrypt_key。")
        try:
            from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        except ImportError as exc:
            raise ValueError("当前环境缺少 cryptography，暂不支持飞书加密事件体。") from exc

        encrypted_bytes = base64.b64decode(encrypted_payload)
        if len(encrypted_bytes) < 16:
            raise ValueError("飞书加密事件体格式不合法。")
        iv = encrypted_bytes[:16]
        ciphertext = encrypted_bytes[16:]
        if len(ciphertext) % 16 != 0:
            raise ValueError("飞书加密事件体长度不合法。")
        key = hashlib.sha256(encrypt_key.encode("utf-8")).digest()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        start = plaintext.find(b"{")
        end = plaintext.rfind(b"}")
        if start < 0 or end < 0 or end < start:
            raise ValueError("飞书加密事件体解密后不是合法 JSON。")
        return json.loads(plaintext[start : end + 1].decode("utf-8"))
