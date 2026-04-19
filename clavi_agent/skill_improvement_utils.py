"""Helpers for parsing and versioning SKILL.md content."""

from __future__ import annotations

import re
from collections.abc import Iterable

import yaml

from .sqlite_schema import utc_now_iso

_FRONTMATTER_PATTERN = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)


def split_skill_markdown(
    markdown: str,
    *,
    fallback_name: str,
    fallback_description: str,
) -> tuple[dict[str, object], str]:
    text = str(markdown or "").replace("\r\n", "\n").strip("\n")
    match = _FRONTMATTER_PATTERN.match(text)
    frontmatter: dict[str, object] = {}
    body = text
    if match is not None:
        body = match.group(2).strip()
        try:
            loaded = yaml.safe_load(match.group(1))
        except yaml.YAMLError:
            loaded = {}
        if isinstance(loaded, dict):
            frontmatter = dict(loaded)

    name = str(frontmatter.get("name") or fallback_name).strip() or fallback_name
    description = (
        str(frontmatter.get("description") or fallback_description).strip()
        or fallback_description
    )
    frontmatter["name"] = name
    frontmatter["description"] = description
    return frontmatter, body


def coerce_skill_version(value: object) -> int:
    try:
        version = int(value)
    except (TypeError, ValueError):
        return 1
    return max(1, version)


def extract_skill_version(markdown: str, *, fallback_name: str = "skill") -> int:
    frontmatter, _ = split_skill_markdown(
        markdown,
        fallback_name=fallback_name,
        fallback_description=fallback_name,
    )
    return coerce_skill_version(frontmatter.get("version"))


def render_skill_markdown(frontmatter: dict[str, object], body: str) -> str:
    yaml_text = yaml.safe_dump(
        frontmatter,
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    normalized_body = body.strip()
    if normalized_body:
        return f"---\n{yaml_text}\n---\n\n{normalized_body}\n"
    return f"---\n{yaml_text}\n---\n"


def build_skill_improvement_payload(
    *,
    current_markdown: str,
    skill_name: str,
    summary: str,
    signal_types: list[str],
    guidance_items: Iterable[str],
    source_run_ids: list[str],
) -> tuple[int, int, str, str]:
    frontmatter, body = split_skill_markdown(
        current_markdown,
        fallback_name=skill_name,
        fallback_description=summary or f"Improvement proposal for {skill_name}.",
    )
    base_version = coerce_skill_version(frontmatter.get("version"))
    proposed_version = base_version + 1
    frontmatter["version"] = proposed_version

    timestamp = utc_now_iso()[:10]
    normalized_guidance = [str(item).strip() for item in guidance_items if str(item).strip()]
    normalized_signals = [str(item).strip() for item in signal_types if str(item).strip()]
    normalized_sources = [str(item).strip() for item in source_run_ids if str(item).strip()]

    lines = [
        f"## Maintainer Update v{proposed_version}",
        "",
        summary or f"Refine `{skill_name}` based on recent usage feedback.",
        "",
    ]
    if normalized_signals:
        lines.append("### Signals")
        for signal in normalized_signals:
            lines.append(f"- `{signal}`")
        lines.append("")
    if normalized_guidance:
        lines.append("### Guidance")
        for item in normalized_guidance:
            lines.append(f"- {item}")
        lines.append("")
    if normalized_sources:
        lines.append("### Provenance")
        for run_id in normalized_sources:
            lines.append(f"- Source run: `{run_id}`")
        lines.append("")

    note_block = "\n".join(lines).strip()
    merged_body = body.rstrip()
    if merged_body:
        merged_body = f"{merged_body}\n\n{note_block}"
    else:
        merged_body = note_block

    changelog_entry = (
        f"v{proposed_version} ({timestamp}): "
        f"{summary or f'Refined {skill_name} after recent usage feedback.'}"
    )
    return (
        base_version,
        proposed_version,
        render_skill_markdown(frontmatter, merged_body),
        changelog_entry,
    )
