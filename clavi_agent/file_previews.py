from __future__ import annotations

from pathlib import Path


TEXT_PREVIEW_BYTE_LIMIT = 128 * 1024
TEXT_PREVIEW_KINDS = frozenset({"markdown", "text", "json"})
INLINE_FILE_PREVIEW_KINDS = frozenset({"image", "pdf"})


def guess_preview_kind(
    *,
    artifact_format: str = "",
    mime_type: str = "",
    filename: str = "",
) -> str:
    normalized_format = artifact_format.strip().lower()
    if not normalized_format and filename:
        normalized_format = Path(filename).suffix.lstrip(".").lower()

    normalized_mime = mime_type.strip().lower()

    if normalized_format in {"md", "markdown"} or normalized_mime == "text/markdown":
        return "markdown"
    if normalized_mime.startswith("text/") or normalized_format in {"txt", "csv", "log"}:
        return "text"
    if normalized_format == "json" or normalized_mime == "application/json":
        return "json"
    if normalized_format in {"html", "htm"} or normalized_mime == "text/html":
        return "html"
    if normalized_mime.startswith("image/") or normalized_format in {
        "png",
        "jpg",
        "jpeg",
        "gif",
        "webp",
        "svg",
    }:
        return "image"
    if normalized_format == "pdf" or normalized_mime == "application/pdf":
        return "pdf"
    if normalized_format in {"doc", "docx", "ppt", "pptx", "xls", "xlsx"}:
        return "office"
    return "none"


def read_text_preview(
    path: Path,
    *,
    max_bytes: int = TEXT_PREVIEW_BYTE_LIMIT,
) -> tuple[str, bool]:
    content = path.read_bytes()
    truncated = len(content) > max_bytes
    snippet = content[:max_bytes]
    return snippet.decode("utf-8", errors="replace"), truncated
