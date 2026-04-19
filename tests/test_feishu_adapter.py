import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from clavi_agent.integration_models import IntegrationConfigRecord
from clavi_agent.integrations import (
    ChannelContext,
    ChannelRequest,
    FeishuAdapter,
    MockChannelAdapter,
    OutboundFile,
    OutboundMessage,
    OutboundReaction,
)


def build_integration(kind: str, config: dict | None = None) -> IntegrationConfigRecord:
    return IntegrationConfigRecord(
        id=f"{kind}-integration",
        name=f"{kind}-integration",
        kind=kind,
        status="active",
        webhook_path=f"/api/integrations/{kind}/webhook",
        config=config or {},
        created_at="2026-04-13T12:00:00+00:00",
        updated_at="2026-04-13T12:00:00+00:00",
    )


def test_mock_channel_adapter_round_trip():
    adapter = MockChannelAdapter()
    context = ChannelContext(
        integration=build_integration("mock", {"verify_token": "mock-token"}),
        credentials={"signing_secret": "mock-secret"},
    )
    payload = {
        "event_id": "mock-evt-1",
        "event_type": "message",
        "tenant_id": "tenant-1",
        "token": "mock-token",
        "message": {
            "message_id": "mock-msg-1",
            "chat_id": "chat-1",
            "thread_id": "thread-1",
            "sender_id": "user-1",
            "sender_name": "测试用户",
            "is_group": True,
            "message_type": "text",
            "text": "hello mock",
            "locale": "zh-CN",
            "timezone": "Asia/Shanghai",
            "attachments": [
                {
                    "kind": "file",
                    "provider_file_id": "file-1",
                    "name": "demo.txt",
                    "mime_type": "text/plain",
                    "download_url": "https://example.com/demo.txt",
                }
            ],
            "mentions": [{"id": "user-2", "name": "Tom", "id_type": "mock_user"}],
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    timestamp = "1713000000"
    request = ChannelRequest(
        headers={
            "x-mock-timestamp": timestamp,
            "x-mock-signature": MockChannelAdapter._build_signature("mock-secret", timestamp, body),
        },
        body=body,
        received_at="2026-04-13T12:00:00+00:00",
    )

    verification = adapter.verify_request(context, request)
    assert verification.accepted is True
    assert verification.signature_valid is True

    event = adapter.parse_inbound_event(context, request, verification)
    assert event.provider_message_id == "mock-msg-1"
    assert event.attachments[0].upload_hint is not None
    assert event.attachments[0].upload_hint.download_url == "https://example.com/demo.txt"

    msg_context = adapter.build_msg_context(context, event)
    assert msg_context.text == "hello mock"
    assert msg_context.sender_name == "测试用户"

    send_result = asyncio.run(
        adapter.send_outbound_message(
            context,
            OutboundMessage(target_id="chat-1", text="收到"),
        )
    )
    assert send_result.ok is True

    file_result = asyncio.run(
        adapter.send_outbound_file(
            context,
            OutboundFile(target_id="chat-1", provider_file_id="file-2", file_name="report.txt"),
        )
    )
    assert file_result.ok is True
    assert [item["kind"] for item in adapter.sent_payloads] == ["message", "file"]


def test_feishu_adapter_verifies_signed_request_and_builds_context():
    adapter = FeishuAdapter()
    received_at = "2026-04-13T12:00:00+00:00"
    timestamp = str(int(datetime.fromisoformat(received_at).timestamp()))
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {
                "app_id": "cli_xxx",
                "verification_token": "verify-token",
                "signature_max_age_seconds": 60,
            },
        ),
        credentials={"app_secret": "app-secret", "encrypt_key": "encrypt-key"},
    )
    payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-1",
            "event_type": "im.message.receive_v1",
            "create_time": "1713000000000",
            "token": "verify-token",
            "app_id": "cli_xxx",
            "tenant_key": "tenant-key-1",
        },
        "event": {
            "sender": {
                "sender_id": {
                    "open_id": "ou_sender",
                    "user_id": "u_sender",
                    "union_id": "on_sender",
                },
                "sender_type": "user",
                "tenant_key": "tenant-key-1",
            },
            "message": {
                "message_id": "om_message_1",
                "chat_id": "oc_chat_1",
                "thread_id": "omt_thread_1",
                "chat_type": "group",
                "message_type": "file",
                "content": json.dumps(
                    {"file_key": "file_v2_123", "file_name": "demo.txt"},
                    ensure_ascii=False,
                ),
                "mentions": [
                    {
                        "key": "@_user_1",
                        "id": {"open_id": "ou_mention_1"},
                        "name": "Tom",
                        "tenant_key": "tenant-key-1",
                    }
                ],
                "user_agent": "Lark/7.0.0 LarkLocale/en_US",
            },
        },
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = ChannelRequest(
        headers={
            "X-Lark-Request-Timestamp": timestamp,
            "X-Lark-Request-Nonce": "nonce-1",
            "X-Lark-Signature": FeishuAdapter._build_signature(
                timestamp=timestamp,
                nonce="nonce-1",
                encrypt_key="encrypt-key",
                body=body,
            ),
        },
        body=body,
        received_at=received_at,
    )

    verification = adapter.verify_request(context, request)
    assert verification.accepted is True
    assert verification.signature_valid is True

    event = adapter.parse_inbound_event(context, request, verification)
    assert event.event_type == "message"
    assert event.metadata["provider_event_type"] == "im.message.receive_v1"
    assert event.provider_event_id == "evt-1"
    assert event.provider_user_id == "ou_sender"
    assert event.locale == "en-US"
    assert event.attachments[0].provider_file_id == "file_v2_123"
    assert (
        event.attachments[0].upload_hint.download_url
        == "https://open.feishu.cn/open-apis/im/v1/messages/om_message_1/resources/file_v2_123?type=file"
    )

    msg_context = adapter.build_msg_context(context, event)
    assert msg_context.chat_id == "oc_chat_1"
    assert msg_context.thread_id == "omt_thread_1"
    assert msg_context.mentions[0].id == "ou_mention_1"


def test_feishu_adapter_resolves_text_mentions_and_wrapped_post_payload():
    adapter = FeishuAdapter()
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {"app_id": "cli_xxx", "verification_token": "verify-token"},
        ),
        credentials={"app_secret": "app-secret"},
    )

    text_payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-text",
            "event_type": "im.message.receive_v1",
            "token": "verify-token",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou_sender"}},
            "message": {
                "message_id": "om_text_1",
                "chat_id": "oc_chat_1",
                "chat_type": "p2p",
                "message_type": "text",
                "content": json.dumps({"text": "hello @_user_1"}, ensure_ascii=False),
                "mentions": [
                    {
                        "key": "@_user_1",
                        "id": {"open_id": "ou_mention_1", "user_id": "u_mention_1"},
                        "name": "Tom",
                    }
                ],
            },
        },
    }
    text_request = ChannelRequest(
        headers={},
        body=json.dumps(text_payload, ensure_ascii=False).encode("utf-8"),
        received_at="2026-04-13T12:00:00+00:00",
    )
    text_verification = adapter.verify_request(context, text_request)
    text_event = adapter.parse_inbound_event(context, text_request, text_verification)
    assert text_event.text == "hello @Tom (ou_mention_1, user id: u_mention_1)"

    post_payload = {
        "schema": "2.0",
        "header": {
            "event_id": "evt-post",
            "event_type": "im.message.receive_v1",
            "token": "verify-token",
        },
        "event": {
            "sender": {"sender_id": {"open_id": "ou_sender"}},
            "message": {
                "message_id": "om_post_1",
                "chat_id": "oc_chat_1",
                "chat_type": "group",
                "message_type": "post",
                "content": json.dumps(
                    {
                        "post": {
                            "zh_cn": {
                                "title": "日报",
                                "content": [
                                    [
                                        {"tag": "text", "text": "今天完成了联调"},
                                        {"tag": "img", "image_key": "img_1"},
                                    ],
                                    [
                                        {
                                            "tag": "code_block",
                                            "language": "python",
                                            "text": "print('hello')",
                                        }
                                    ],
                                ],
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
            },
        },
    }
    post_request = ChannelRequest(
        headers={},
        body=json.dumps(post_payload, ensure_ascii=False).encode("utf-8"),
        received_at="2026-04-13T12:00:01+00:00",
    )
    post_verification = adapter.verify_request(context, post_request)
    post_event = adapter.parse_inbound_event(context, post_request, post_verification)
    assert "日报" in post_event.text
    assert "今天完成了联调" in post_event.text
    assert "print('hello')" in post_event.text
    assert post_event.attachments[0].provider_file_id == "img_1"


def test_feishu_adapter_emits_url_verification_ack():
    adapter = FeishuAdapter()
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {"app_id": "cli_xxx", "verification_token": "verify-token"},
        ),
        credentials={"app_secret": "app-secret"},
    )
    payload = {"type": "url_verification", "token": "verify-token", "challenge": "challenge-1"}
    request = ChannelRequest(
        headers={},
        body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        received_at="2026-04-13T12:00:00+00:00",
    )

    verification = adapter.verify_request(context, request)
    event = adapter.parse_inbound_event(context, request, verification)
    ack = adapter.emit_quick_ack(context, event)

    assert verification.accepted is True
    assert ack.body_type == "json"
    assert ack.body_json == {"challenge": "challenge-1"}


def test_feishu_adapter_prepare_msg_context_prepends_reply_context():
    recorded_requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages/om_parent":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "items": [
                            {
                                "msg_type": "text",
                                "body": {
                                    "content": json.dumps({"text": "original question"}, ensure_ascii=False)
                                },
                            }
                        ]
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {"app_id": "cli_xxx", "verification_token": "verify-token"},
        ),
        credentials={"app_secret": "app-secret"},
    )
    request = ChannelRequest(
        headers={},
        body=json.dumps(
            {
                "schema": "2.0",
                "header": {
                    "event_id": "evt-reply",
                    "event_type": "im.message.receive_v1",
                    "token": "verify-token",
                },
                "event": {
                    "sender": {"sender_id": {"open_id": "ou_sender"}},
                    "message": {
                        "message_id": "om_message_1",
                        "chat_id": "oc_chat_1",
                        "chat_type": "p2p",
                        "message_type": "text",
                        "parent_id": "om_parent",
                        "content": json.dumps({"text": "my answer"}, ensure_ascii=False),
                    },
                },
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        received_at="2026-04-13T12:00:00+00:00",
    )
    verification = adapter.verify_request(context, request)
    event = adapter.parse_inbound_event(context, request, verification)
    msg_context = adapter.build_msg_context(context, event)

    enriched = asyncio.run(adapter.prepare_msg_context(context, msg_context))
    assert enriched.text.startswith("[Reply to: original question]")
    assert "my answer" in enriched.text
    assert enriched.metadata["reply_context"] == "[Reply to: original question]"
    assert recorded_requests == [
        ("POST", "/open-apis/auth/v3/tenant_access_token/internal"),
        ("GET", "/open-apis/im/v1/messages/om_parent"),
    ]


def test_feishu_adapter_send_message_defaults_to_create_when_reply_disabled():
    recorded_requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["receive_id"] == "oc_chat_1"
            assert payload["msg_type"] == "text"
            assert "reply_in_thread" not in payload
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"message_id": "om_send_1", "chat_id": "oc_chat_1"},
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {"app_id": "cli_xxx", "verification_token": "verify-token"},
        ),
        credentials={"app_secret": "app-secret"},
    )

    result = asyncio.run(
        adapter.send_outbound_message(
            context,
            OutboundMessage(
                target_id="oc_chat_1",
                reply_to_message_id="om_source",
                text="普通发送",
            ),
        )
    )

    assert result.ok is True
    assert result.provider_message_id == "om_send_1"
    assert recorded_requests == [
        ("POST", "/open-apis/auth/v3/tenant_access_token/internal"),
        ("POST", "/open-apis/im/v1/messages"),
    ]


def test_feishu_adapter_hashes_overlong_uuid_before_sending():
    recorded_uuids: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            payload = json.loads(request.content.decode("utf-8"))
            recorded_uuids.append(payload["uuid"])
            assert len(payload["uuid"]) <= 50
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"message_id": "om_send_1", "chat_id": "oc_chat_1"},
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {"app_id": "cli_xxx", "verification_token": "verify-token"},
        ),
        credentials={"app_secret": "app-secret"},
    )
    long_dedup_key = "b61050fd-274f-470c-9837-5507516bf427:tool-file:317a93b28da04dd2a9c698a87e352759"

    result = asyncio.run(
        adapter.send_outbound_message(
            context,
            OutboundMessage(
                target_id="oc_chat_1",
                text="普通发送",
                dedup_key=long_dedup_key,
            ),
        )
    )

    assert result.ok is True
    assert recorded_uuids == [FeishuAdapter._normalize_message_uuid(long_dedup_key)]
    assert recorded_uuids[0] != long_dedup_key


def test_feishu_adapter_send_message_and_file_uses_token_cache():
    recorded_requests: list[tuple[str, str, dict, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        recorded_requests.append((request.method, str(request.url), dict(request.headers), body))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages/om_source/reply":
            payload = json.loads(body.decode("utf-8"))
            assert payload["msg_type"] == "text"
            assert json.loads(payload["content"]) == {"text": "已收到"}
            assert payload["reply_in_thread"] is True
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "message_id": "om_reply_1",
                        "chat_id": "oc_chat_1",
                        "thread_id": "omt_thread_1",
                    },
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            assert request.url.params["receive_id_type"] == "chat_id"
            payload = json.loads(body.decode("utf-8"))
            assert payload["msg_type"] == "file"
            assert json.loads(payload["content"]) == {"file_key": "file_v2_123"}
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "message_id": "om_file_1",
                        "chat_id": "oc_chat_1",
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {"app_id": "cli_xxx", "verification_token": "verify-token"},
        ),
        credentials={"app_secret": "app-secret"},
    )

    message_result = asyncio.run(
        adapter.send_outbound_message(
            context,
            OutboundMessage(
                reply_to_message_id="om_source",
                thread_id="omt_thread_1",
                text="已收到",
            ),
        )
    )
    assert message_result.ok is True
    assert message_result.provider_message_id == "om_reply_1"

    file_result = asyncio.run(
        adapter.send_outbound_file(
            context,
            OutboundFile(
                target_id="oc_chat_1",
                provider_file_id="file_v2_123",
                file_name="demo.txt",
            ),
        )
    )
    assert file_result.ok is True
    assert file_result.provider_message_id == "om_file_1"

    auth_requests = [item for item in recorded_requests if "/tenant_access_token/internal" in item[1]]
    assert len(auth_requests) == 1


def test_feishu_adapter_uploads_generated_file_before_sending(tmp_path):
    recorded_requests: list[tuple[str, str, bytes]] = []
    artifact_path = tmp_path / "report.md"
    artifact_path.write_text("# report\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        recorded_requests.append((request.method, request.url.path, body))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/files":
            assert request.headers["Authorization"] == "Bearer t-123"
            assert b'name="file_type"' in body
            assert b"stream" in body
            assert b'name="file_name"' in body
            assert b"report.md" in body
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"file_key": "file_v2_uploaded"},
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            payload = json.loads(body.decode("utf-8"))
            assert payload["msg_type"] == "file"
            assert json.loads(payload["content"]) == {"file_key": "file_v2_uploaded"}
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "message_id": "om_file_uploaded",
                        "chat_id": "oc_chat_1",
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration("feishu", {"app_id": "cli_xxx"}),
        credentials={"app_secret": "app-secret"},
    )

    result = asyncio.run(
        adapter.send_outbound_file(
            context,
            OutboundFile(
                target_id="oc_chat_1",
                file_name="report.md",
                url="/api/artifacts/artifact-1",
                metadata={"local_path": str(artifact_path)},
            ),
        )
    )

    assert result.ok is True
    assert result.provider_message_id == "om_file_uploaded"
    assert [path for _, path, _ in recorded_requests] == [
        "/open-apis/auth/v3/tenant_access_token/internal",
        "/open-apis/im/v1/files",
        "/open-apis/im/v1/messages",
    ]


def test_feishu_adapter_uploads_csv_as_stream_and_sends_plain_file_message(tmp_path):
    recorded_requests: list[tuple[str, str, bytes]] = []
    artifact_path = tmp_path / "table.csv"
    artifact_path.write_text("a,b\n1,2\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content
        recorded_requests.append((request.method, request.url.path, body))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/files":
            assert b'name="file_type"' in body
            assert b"stream" in body
            assert b"table.csv" in body
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"file_key": "file_v2_csv"},
                },
            )
        if request.url.path == "/open-apis/im/v1/messages":
            payload = json.loads(body.decode("utf-8"))
            assert payload["msg_type"] == "file"
            assert json.loads(payload["content"]) == {"file_key": "file_v2_csv"}
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "message_id": "om_file_csv",
                        "chat_id": "oc_chat_1",
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration("feishu", {"app_id": "cli_xxx"}),
        credentials={"app_secret": "app-secret"},
    )

    result = asyncio.run(
        adapter.send_outbound_file(
            context,
            OutboundFile(
                target_id="oc_chat_1",
                file_name="table.csv",
                url="/api/artifacts/artifact-2",
                metadata={"local_path": str(artifact_path)},
            ),
        )
    )

    assert result.ok is True
    assert result.provider_message_id == "om_file_csv"


def test_feishu_adapter_send_outbound_reaction():
    recorded_requests: list[tuple[str, str, bytes]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path, request.content))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages/om_source/reactions":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload == {
                "reaction_type": {
                    "emoji_type": "DONE",
                }
            }
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {},
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {
                "app_id": "cli_xxx",
                "app_secret": "app-secret",
                "verification_token": "verify-token",
            },
        ),
        credentials={},
    )

    result = asyncio.run(
        adapter.send_outbound_reaction(
            context,
            OutboundReaction(
                message_id="om_source",
                reaction_type="DONE",
            ),
        )
    )

    assert result.ok is True
    assert recorded_requests == [
        ("POST", "/open-apis/auth/v3/tenant_access_token/internal", b'{"app_id":"cli_xxx","app_secret":"app-secret"}'),
        ("POST", "/open-apis/im/v1/messages/om_source/reactions", b'{"reaction_type":{"emoji_type":"DONE"}}'),
    ]


def test_feishu_adapter_send_message_uses_root_id_for_thread_reply():
    recorded_requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append((request.method, request.url.path))
        if request.url.path == "/open-apis/auth/v3/tenant_access_token/internal":
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "tenant_access_token": "t-123",
                    "expire": 7200,
                },
            )
        if request.url.path == "/open-apis/im/v1/messages/om_root/reply":
            payload = json.loads(request.content.decode("utf-8"))
            assert payload["reply_in_thread"] is True
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "message_id": "om_reply_root",
                        "chat_id": "oc_chat_1",
                        "thread_id": "omt_thread_1",
                    },
                },
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    adapter = FeishuAdapter(transport=httpx.MockTransport(handler))
    context = ChannelContext(
        integration=build_integration(
            "feishu",
            {"app_id": "cli_xxx", "verification_token": "verify-token"},
        ),
        credentials={"app_secret": "app-secret"},
    )

    result = asyncio.run(
        adapter.send_outbound_message(
            context,
            OutboundMessage(
                target_id="oc_chat_1",
                reply_to_message_id="om_source",
                thread_id="omt_thread_1",
                text="线程回复",
                metadata={"provider_root_message_id": "om_root"},
            ),
        )
    )

    assert result.ok is True
    assert result.provider_message_id == "om_reply_root"
    assert recorded_requests == [
        ("POST", "/open-apis/auth/v3/tenant_access_token/internal"),
        ("POST", "/open-apis/im/v1/messages/om_root/reply"),
    ]

