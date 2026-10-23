"""Parse raw text into conversation turns, when it looks like a conversation.

Treating a chat transcript as a flat document (just splitting on blank
lines) throws away the one structure that actually matters for
summarization: who said what, and which turn was most recent. This module
recognizes two common shapes and falls back to ``None`` — "not a
conversation" — for anything else, so callers can use a plain-document
strategy instead.

Recognized formats
-------------------
1. JSON: a list of ``{"role": ..., "content": ...}`` objects, the shape
   used by most chat/completions APIs. ``content`` may also be a list of
   content blocks (``[{"type": "text", "text": "..."}]``), which is
   flattened to plain text.
2. Speaker-prefixed plain text: lines like ``User: ...`` or
   ``Assistant: ...`` that start a new turn, with following lines (until
   the next speaker prefix) treated as that turn's content. Detection
   requires at least two speaker-prefixed lines from at least two distinct
   speakers, to avoid misfiring on an ordinary document that happens to
   contain one "Label: text" line.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

_SPEAKER_LINE_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _-]{0,24}):\s?(.*)$")
_MIN_SPEAKER_LINES = 2
_MIN_DISTINCT_SPEAKERS = 2
_MAX_SPEAKER_NAME_WORDS = 3


@dataclass
class Turn:
    role: str
    content: str


def parse_conversation(text: str) -> list[Turn] | None:
    """Return the conversation's turns, or ``None`` if ``text`` doesn't look
    like a conversation."""
    turns = _parse_json_messages(text)
    if turns is not None:
        return turns
    return _parse_speaker_prefixed(text)


def _parse_json_messages(text: str) -> list[Turn] | None:
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    try:
        data = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, list) or not data:
        return None

    turns: list[Turn] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = item.get("content")
        if role is None or content is None:
            return None
        if isinstance(content, list):
            content = "\n".join(
                block.get("text", "") for block in content if isinstance(block, dict) and "text" in block
            )
        if not isinstance(content, str):
            return None
        turns.append(Turn(role=str(role), content=content))
    return turns if turns else None


def _parse_speaker_prefixed(text: str) -> list[Turn] | None:
    lines = text.splitlines()
    turns: list[Turn] = []
    current_role: str | None = None
    current_lines: list[str] = []
    speaker_line_count = 0
    speakers_seen: set[str] = set()

    def flush() -> None:
        if current_role is not None:
            content = "\n".join(current_lines).strip()
            if content:
                turns.append(Turn(role=current_role, content=content))

    for line in lines:
        m = _SPEAKER_LINE_RE.match(line)
        if m and len(m.group(1).split()) <= _MAX_SPEAKER_NAME_WORDS:
            flush()
            current_role = m.group(1).strip()
            current_lines = [m.group(2)] if m.group(2) else []
            speaker_line_count += 1
            speakers_seen.add(current_role.lower())
        elif current_role is not None:
            current_lines.append(line)
    flush()

    if speaker_line_count < _MIN_SPEAKER_LINES or len(speakers_seen) < _MIN_DISTINCT_SPEAKERS:
        return None
    return turns if turns else None
