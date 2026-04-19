from __future__ import annotations

import hashlib
import mimetypes
import re
from typing import Final

from pydantic import BaseModel, Field

from .account_constants import ROOT_ACCOUNT_ID


MAX_UPLOAD_SIZE_BYTES: Final[int] = 25 * 1024 * 1024
ALLOWED_UPLOAD_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".csv",
        ".doc",
        ".docx",
        ".gif",
        ".htm",
        ".html",
        ".jpeg",
        ".jpg",
        ".json",
        ".markdown",
        ".md",
        ".pdf",
        ".png",
        ".ppt",
        ".pptx",
        ".rtf",
        ".svg",
        ".text",
        ".tsv",
        ".txt",
        ".webp",
        ".xls",
        ".xlsx",
        ".xml",
        ".yaml",
        ".yml",
    }
)
BLOCKED_UPLOAD_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {
        ".app",
        ".bat",
        ".cmd",
        ".com",
        ".cpl",
        ".dll",
        ".dmg",
        ".exe",
        ".hta",
        ".js",
        ".jar",
        ".msi",
        ".ps1",
        ".psm1",
        ".py",
        ".pyz",
        ".reg",
        ".scr",
        ".sh",
        ".vb",
        ".vbs",
    }
)

_INVALID_UPLOAD_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]+')


class UploadCreatePayload(BaseModel):
    original_name: str
    content_bytes: bytes = Field(repr=False)
    mime_type: str = ""


class UploadRecord(BaseModel):
    id: str
    session_id: str
    account_id: str = ROOT_ACCOUNT_ID
    run_id: str | None = None
    original_name: str
    safe_name: str
    relative_path: str
    absolute_path: str
    mime_type: str = "application/octet-stream"
    size_bytes: int = Field(ge=0)
    checksum: str
    created_at: str
    created_by: str = "user"


def sanitize_upload_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/").split("/")[-1].strip()
    normalized = _INVALID_UPLOAD_NAME_CHARS.sub("_", normalized).strip(" .")
    if not normalized or normalized in {".", ".."}:
        return "upload.bin"
    stem, dot, suffix = normalized.rpartition(".")
    if not dot:
        return normalized[:200]

    safe_stem = stem[: max(1, 200 - len(suffix) - 1)].rstrip(" .")
    if not safe_stem:
        safe_stem = "upload"
    return f"{safe_stem}.{suffix}"


def upload_extension(filename: str) -> str:
    if "." not in filename:
        return ""
    return f".{filename.rsplit('.', 1)[-1].lower()}"


def resolve_upload_mime_type(filename: str, declared_mime_type: str | None) -> str:
    normalized_declared = (declared_mime_type or "").strip()
    guessed_mime_type, _ = mimetypes.guess_type(filename)
    if normalized_declared and normalized_declared != "application/octet-stream":
        return normalized_declared
    if guessed_mime_type:
        return guessed_mime_type
    if normalized_declared:
        return normalized_declared
    return "application/octet-stream"


def compute_upload_checksum(content_bytes: bytes) -> str:
    return hashlib.sha256(content_bytes).hexdigest()
