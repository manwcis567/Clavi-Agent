"""渠道适配器基类与公共工具。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from .models import (
    ChannelContext,
    ChannelRequest,
    MsgAttachment,
    MsgContextEnvelope,
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


def decode_json_body(body: bytes) -> Any:
    """按 UTF-8 解码 JSON 请求体。"""
    if not body:
        return {}
    return json.loads(body.decode("utf-8"))


class ChannelAdapter(ABC):
    """统一的渠道适配器接口。"""

    kind: str

    @property
    def display_name(self) -> str:
        return self.kind

    @abstractmethod
    def verify_request(
        self,
        context: ChannelContext,
        request: ChannelRequest,
    ) -> RequestVerificationResult:
        """校验 webhook 请求是否合法。"""

    @abstractmethod
    def parse_inbound_event(
        self,
        context: ChannelContext,
        request: ChannelRequest,
        verification: RequestVerificationResult,
    ) -> ParsedInboundEvent:
        """把 webhook 请求解析为统一入站事件。"""

    @abstractmethod
    def build_msg_context(
        self,
        context: ChannelContext,
        event: ParsedInboundEvent,
    ) -> MsgContextEnvelope:
        """把入站事件转换为业务链路可消费的 MsgContext。"""

    @abstractmethod
    def emit_quick_ack(
        self,
        context: ChannelContext,
        event: ParsedInboundEvent,
    ) -> QuickAckIntent:
        """生成快速 ACK 响应。"""

    @abstractmethod
    async def send_outbound_message(
        self,
        context: ChannelContext,
        message: OutboundMessage,
    ) -> OutboundSendResult:
        """发送文本或结构化消息。"""

    @abstractmethod
    async def send_outbound_file(
        self,
        context: ChannelContext,
        file: OutboundFile,
    ) -> OutboundSendResult:
        """发送文件或图片。"""

    def build_quick_reaction(
        self,
        context: ChannelContext,
        msg_context: MsgContextEnvelope,
    ) -> OutboundReaction | None:
        return None

    async def send_outbound_reaction(
        self,
        context: ChannelContext,
        reaction: OutboundReaction,
    ) -> OutboundSendResult:
        return OutboundSendResult(
            ok=False,
            error=NormalizedAdapterError(
                code="unsupported_operation",
                message=f"{self.display_name} does not support outbound reactions.",
                retryable=False,
            ),
        )

    @abstractmethod
    def normalize_error(self, error: Exception | dict[str, Any]) -> NormalizedAdapterError:
        """把渠道异常归一化。"""

    async def prepare_upload_download(
        self,
        context: ChannelContext,
        attachment: MsgAttachment,
    ) -> UploadBridgeHint | None:
        return attachment.upload_hint

    async def prepare_msg_context(
        self,
        context: ChannelContext,
        msg_context: MsgContextEnvelope,
    ) -> MsgContextEnvelope:
        """在进入业务链路前异步补充消息上下文。"""
        return msg_context


class ChannelAdapterRegistry:
    """渠道适配器注册表。"""

    def __init__(self, adapters: Iterable[ChannelAdapter] | None = None):
        self._adapters: dict[str, ChannelAdapter] = {}
        for adapter in adapters or []:
            self.register(adapter)

    def register(self, adapter: ChannelAdapter) -> ChannelAdapter:
        self._adapters[adapter.kind] = adapter
        return adapter

    def get(self, kind: str) -> ChannelAdapter:
        normalized = kind.strip().lower()
        if normalized not in self._adapters:
            raise KeyError(f"未注册的渠道适配器: {kind}")
        return self._adapters[normalized]

    def has(self, kind: str) -> bool:
        return kind.strip().lower() in self._adapters

    def kinds(self) -> list[str]:
        return sorted(self._adapters.keys())
