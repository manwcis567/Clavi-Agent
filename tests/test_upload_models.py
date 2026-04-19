import pytest

from clavi_agent.session import SessionManager
from clavi_agent.upload_models import sanitize_upload_filename


def test_sanitize_upload_filename_strips_path_segments_and_reserved_chars():
    assert sanitize_upload_filename(r"..\unsafe\report?.md") == "report_.md"


def test_sanitize_upload_filename_falls_back_for_empty_names():
    assert sanitize_upload_filename(" .. ") == "upload.bin"


def test_resolve_session_workspace_path_keeps_inside_paths_and_blocks_escape(tmp_path):
    workspace_dir = (tmp_path / "workspace").resolve()
    workspace_dir.mkdir()

    inside_path = SessionManager._resolve_session_workspace_path(
        workspace_dir,
        ".clavi_agent/uploads/session-1/upload-1/draft.md",
        label="Upload",
    )

    assert inside_path == (
        workspace_dir / ".clavi_agent/uploads/session-1/upload-1/draft.md"
    ).resolve()

    with pytest.raises(ValueError, match="escapes the session workspace"):
        SessionManager._resolve_session_workspace_path(
            workspace_dir,
            "../outside.md",
            label="Upload",
        )

    with pytest.raises(ValueError, match="escapes the session workspace"):
        SessionManager._resolve_session_workspace_path(
            workspace_dir,
            str((tmp_path / "outside.md").resolve()),
            label="Upload",
        )

