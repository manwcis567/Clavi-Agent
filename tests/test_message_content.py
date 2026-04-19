"""Tests for structured message content helpers and LLM conversions."""

from clavi_agent.llm.anthropic_client import AnthropicClient
from clavi_agent.llm.openai_client import OpenAIClient
from clavi_agent.schema import Message, message_content_summary, render_message_content_for_model


STRUCTURED_USER_CONTENT = [
    {"type": "text", "text": "请修订这份草稿"},
    {
        "type": "uploaded_file",
        "upload_id": "upload-1",
        "original_name": "draft.md",
        "safe_name": "draft.md",
        "relative_path": ".clavi_agent/uploads/session-1/upload-1/draft.md",
        "mime_type": "text/markdown",
        "size_bytes": 128,
        "checksum": "abc123",
    },
    {
        "type": "artifact_ref",
        "artifact_id": "artifact-1",
        "run_id": "run-1",
        "uri": "reports/final.md",
        "display_name": "final.md",
        "mime_type": "text/markdown",
        "role": "final_deliverable",
        "summary": "上一版导出的结果",
    },
]


def test_message_content_helpers_render_summary_and_model_text():
    summary = message_content_summary(STRUCTURED_USER_CONTENT)
    rendered = render_message_content_for_model(STRUCTURED_USER_CONTENT)

    assert "请修订这份草稿" in summary
    assert "draft.md" in summary
    assert "final.md" in summary
    assert "处理规则" in rendered
    assert "工作区路径" in rendered
    assert "Artifact ID" in rendered
    assert "draft.md" in rendered


def test_openai_convert_messages_flattens_structured_user_content():
    client = OpenAIClient(
        api_key="test-key",
        api_base="https://example.com/v1",
        model="test-model",
    )

    _, api_messages = client._convert_messages(
        [Message(role="user", content=STRUCTURED_USER_CONTENT)]
    )

    assert isinstance(api_messages[0]["content"], str)
    assert "已附带上传文件" in api_messages[0]["content"]
    assert "draft.md" in api_messages[0]["content"]
    assert "Artifact ID" in api_messages[0]["content"]


def test_anthropic_convert_messages_flattens_structured_user_content():
    client = AnthropicClient(
        api_key="test-key",
        api_base="https://example.com/anthropic",
        model="test-model",
    )

    _, api_messages = client._convert_messages(
        [Message(role="user", content=STRUCTURED_USER_CONTENT)]
    )

    assert isinstance(api_messages[0]["content"], str)
    assert "已附带上传文件" in api_messages[0]["content"]
    assert "draft.md" in api_messages[0]["content"]
    assert "Artifact ID" in api_messages[0]["content"]

