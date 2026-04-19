"""渠道适配层导出。"""

from .adapter_base import ChannelAdapter, ChannelAdapterRegistry
from .dispatcher import IntegrationReplyDispatcher
from .feishu_adapter import FeishuAdapter
from .feishu_long_connection import FeishuLongConnectionError, FeishuLongConnectionService
from .gateway import (
    GatewayCommandResult,
    GatewayProcessResult,
    IntegrationGateway,
    IntegrationGatewayError,
)
from .mock_adapter import MockChannelAdapter
from .wechat_adapter import WeChatAdapter
from .wechat_long_poll import WeChatLongPollService
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
    ReplyAttachmentRef,
    ReplyEnvelope,
    RequestVerificationResult,
    UploadBridgeHint,
)
from .router import IntegrationRouter, ROOT_THREAD_ID, RouteResolution
from .runtime_bridge import (
    IntegrationRunBridge,
    IntegrationRunBridgeError,
    IntegrationRunBridgeResult,
)


def create_default_adapter_registry() -> ChannelAdapterRegistry:
    """创建内置适配器注册表。"""
    return ChannelAdapterRegistry([MockChannelAdapter(), FeishuAdapter(), WeChatAdapter()])


__all__ = [
    "ChannelAdapter",
    "ChannelAdapterRegistry",
    "ChannelContext",
    "ChannelRequest",
    "FeishuAdapter",
    "FeishuLongConnectionError",
    "FeishuLongConnectionService",
    "GatewayCommandResult",
    "GatewayProcessResult",
    "IntegrationGateway",
    "IntegrationGatewayError",
    "IntegrationReplyDispatcher",
    "IntegrationRouter",
    "IntegrationRunBridge",
    "IntegrationRunBridgeError",
    "IntegrationRunBridgeResult",
    "MockChannelAdapter",
    "MsgAttachment",
    "MsgContextEnvelope",
    "MsgMention",
    "NormalizedAdapterError",
    "OutboundFile",
    "OutboundMessage",
    "OutboundReaction",
    "OutboundSendResult",
    "ParsedInboundEvent",
    "QuickAckIntent",
    "ReplyAttachmentRef",
    "ReplyEnvelope",
    "RequestVerificationResult",
    "ROOT_THREAD_ID",
    "RouteResolution",
    "UploadBridgeHint",
    "WeChatAdapter",
    "WeChatLongPollService",
    "create_default_adapter_registry",
]
