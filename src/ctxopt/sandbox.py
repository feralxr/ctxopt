"""Sandbox noisy tool output: collapse, dedupe, and budget output so it
doesn't blow out an LLM's context window.

Design
------
Most tool output (logs, test runners, build tools, stack traces, diffs) is
line-oriented, so a single pipeline handles it:

1. Detect a content profile (``generic`` / ``json`` / ``stacktrace`` /
   ``diff``), automatically or forced via ``profile=``. The profile only
   changes which lines are always preserved (``keep_patterns``) and the
   default head/tail window sizes — detection is deliberately conservative
   and falls back to ``generic`` when unsure.
2. Collapse near-duplicate lines: lines that are identical *after*
   normalizing volatile substrings (timestamps, hex addresses, UUIDs, bare
   numbers) are folded into one representative line plus a repeat count.
   This is what actually tames things like recursive stack traces or
   progress spam, where only a counter or address differs between lines.
3. Elide the "boring middle": keep the first/last N lines, and within the
   elided region still preserve any line matching ``keep_patterns``
   (error/exception/fail/traceback by default, plus structural markers for
   the ``diff``/``stacktrace`` profiles) up to a cap, so signal isn't
   silently dropped along with the noise.
4. Fit to a token budget: if ``max_tokens`` is given, the head/tail/keep
   windows shrink in a bounded loop until the result actually fits, rather
   than trusting a single line-count guess to land under budget.

JSON is the one case that needs a different approach, since it's structured
data rather than lines of text: large arrays and long strings are truncated
at the value level so the result stays valid, readable JSON.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .safety import compile_safe_pattern
from .tokens import count_tokens

_TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?")
_TIME_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}\b")
_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
# Only standalone numeric tokens (e.g. " 42 ", "#10000") are treated as
# volatile. Digits embedded in an identifier (like "line1" or "v2") are left
# alone, since those are usually meaningful content, not a counter.
_NUMBER_RE = re.compile(r"\b\d+\b")

_DEFAULT_KEEP_PATTERNS = (
    r"\berror\b",
    r"\bexception\b",
    r"\bfail(?:ed|ure)?\b",
    r"\btraceback\b",
    r"\bfatal\b",
    r"\bpanic\b",
)

_PROFILE_PRESETS: dict[str, dict] = {
    "generic": {"extra_keep_patterns": (), "head_lines": 40, "tail_lines": 40},
    "stacktrace": {
        "extra_keep_patterns": (r"^traceback", r"^caused by:?"),
        "head_lines": 15,
        "tail_lines": 30,
    },
    "diff": {
        "extra_keep_patterns": (r"^diff --git", r"^--- ", r"^\+\+\+ ", r"^@@"),
        "head_lines": 60,
        "tail_lines": 20,
    },
}


@dataclass
class SandboxResult:
    text: str
    original_tokens: int
    final_tokens: int
    lines_elided: int
    profile: str
    fit_to_budget: bool = False


def detect_profile(text: str) -> str:
    """Best-effort guess at the shape of ``text``: ``json``, ``stacktrace``,
    ``diff``, or ``generic``. Conservative by design — when unsure, falls
    back to ``generic``, which is always a safe default."""
    stripped = text.strip()
    if stripped and stripped[0] in "{[":
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass

    lines = stripped.splitlines()
    sample = lines[:60]

    diff_markers = sum(1 for line in sample if line.startswith(("+++ ", "--- ", "@@ ", "diff --git ")))
    if diff_markers >= 2:
        return "diff"

    frame_markers = sum(
        1
        for line in sample
        if re.search(r"^\s*at .+\(.*\)\s*$", line)
        or re.search(r'File "[^"]+", line \d+', line)
        or line.strip().lower().startswith("traceback (most recent call last)")
    )
    if frame_markers >= 3:
        return "stacktrace"

    return "generic"


def _normalize_for_dedup(line: str) -> str:
    """Strip volatile substrings so near-identical lines compare equal."""
    line = _TIMESTAMP_RE.sub("<ts>", line)
    line = _TIME_RE.sub("<time>", line)
    line = _UUID_RE.sub("<uuid>", line)
    line = _HEX_RE.sub("<hex>", line)
    line = _NUMBER_RE.sub("<n>", line)
    return line


def _collapse_repeats(lines: list[str], threshold: int) -> list[str]:
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        key = _normalize_for_dedup(lines[i])
        j = i
        while j < n and _normalize_for_dedup(lines[j]) == key:
            j += 1
        run_len = j - i
        if run_len >= threshold:
            out.append(lines[i])
            out.append(f"... [{run_len - 1} near-identical line(s) collapsed] ...")
        else:
            out.extend(lines[i:j])
        i = j
    return out


def _elide_middle(
    lines: list[str],
    head_lines: int,
    tail_lines: int,
    keep_regexes: list[re.Pattern],
    max_kept_matches: int,
) -> tuple[list[str], int]:
    if len(lines) <= head_lines + tail_lines:
        return lines, 0

    head = lines[:head_lines] if head_lines else []
    tail = lines[-tail_lines:] if tail_lines else []
    middle_end = len(lines) - tail_lines if tail_lines else len(lines)
    middle = lines[head_lines:middle_end]

    kept: list[str] = []
    if keep_regexes:
        for line in middle:
            if len(kept) >= max_kept_matches:
                break
            if any(rx.search(line) for rx in keep_regexes):
                kept.append(line)

    elided_count = len(middle) - len(kept)
    marker = f"... [{elided_count} lines elided"
    if kept:
        marker += f", {len(kept)} matching line(s) preserved below"
    marker += "] ..."

    return [*head, marker, *kept, *tail], elided_count


def _fit_to_budget(
    lines: list[str],
    head_lines: int,
    tail_lines: int,
    keep_regexes: list[re.Pattern],
    max_kept_matches: int,
    max_tokens: int,
) -> tuple[list[str], int, bool]:
    """Shrink head/tail/keep windows in a bounded loop until the rendered
    text fits ``max_tokens``, or we hit the floor and give up gracefully."""
    h, t, k = head_lines, tail_lines, max_kept_matches
    result, elided = _elide_middle(lines, h, t, keep_regexes, k)
    shrunk = False
    for _ in range(15):
        if count_tokens("\n".join(result)) <= max_tokens:
            return result, elided, shrunk
        if h <= 2 and t <= 2 and k <= 1:
            break
        h, t, k = max(2, int(h * 0.7)), max(2, int(t * 0.7)), max(1, int(k * 0.7))
        result, elided = _elide_middle(lines, h, t, keep_regexes, k)
        shrunk = True
    return result, elided, shrunk


def _sandbox_json(text: str, max_items: int, max_string_len: int) -> str | None:
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    truncated = _truncate_json_value(data, max_items, max_string_len)
    return json.dumps(truncated, indent=2, ensure_ascii=False)


def _truncate_json_value(value, max_items: int, max_string_len: int, depth: int = 0, max_depth: int = 8):
    if depth >= max_depth:
        return "... [max depth reached] ..."
    if isinstance(value, str):
        if len(value) > max_string_len:
            return value[:max_string_len] + f"... [{len(value) - max_string_len} chars truncated]"
        return value
    if isinstance(value, list):
        items = [_truncate_json_value(v, max_items, max_string_len, depth + 1, max_depth) for v in value]
        if len(items) > max_items:
            half = max(1, max_items // 2)
            omitted = len(items) - 2 * half
            return [*items[:half], f"... [{omitted} items omitted] ...", *items[-half:]]
        return items
    if isinstance(value, dict):
        return {k: _truncate_json_value(v, max_items, max_string_len, depth + 1, max_depth) for k, v in value.items()}
    return value


def sandbox_text(
    text: str,
    *,
    max_tokens: int | None = None,
    max_lines: int = 200,
    head_lines: int | None = None,
    tail_lines: int | None = None,
    collapse_repeats: bool = True,
    repeat_threshold: int = 3,
    keep_patterns: list[str] | None = None,
    max_kept_matches: int = 50,
    profile: str = "auto",
    json_max_items: int = 20,
    json_max_string_len: int = 500,
) -> SandboxResult:
    """Collapse long, repetitive, or noisy tool output into a compact form
    that preserves signal (errors, structure) while discarding bulk.

    Parameters
    ----------
    max_tokens:
        If set, the output is shrunk until it fits this token budget,
        overriding the line-count windows as needed. This is the
        recommended way to call ``sandbox_text`` inside an agent loop,
        since tokens — not lines — are what actually fill a context window.
    max_lines / head_lines / tail_lines:
        Line-count based control, used when ``max_tokens`` isn't set.
        ``head_lines``/``tail_lines`` default to profile-specific values
        when left as ``None``.
    keep_patterns:
        Extra regexes (in addition to the error/exception/fail/traceback
        defaults, and any profile-specific structural patterns) whose
        matching lines are never elided, up to ``max_kept_matches``.
    profile:
        ``"auto"`` (default) to detect content type, or force one of
        ``"generic"``, ``"json"``, ``"stacktrace"``, ``"diff"``.
    json_max_items / json_max_string_len:
        For the ``json`` profile: max array items and string length kept
        before truncating, at every nesting level.
    """
    original_tokens = count_tokens(text)
    resolved_profile = detect_profile(text) if profile == "auto" else profile

    if resolved_profile == "json":
        json_text = _sandbox_json(text, json_max_items, json_max_string_len)
        if json_text is not None:
            return SandboxResult(
                text=json_text,
                original_tokens=original_tokens,
                final_tokens=count_tokens(json_text),
                lines_elided=0,
                profile="json",
            )
        resolved_profile = "generic"  # looked like JSON but didn't parse

    preset = _PROFILE_PRESETS.get(resolved_profile, _PROFILE_PRESETS["generic"])
    effective_head = head_lines if head_lines is not None else preset["head_lines"]
    effective_tail = tail_lines if tail_lines is not None else preset["tail_lines"]

    # Patterns ctxopt authors itself (defaults, profile presets) are trusted
    # and compiled directly; user-supplied keep_patterns go through the
    # ReDoS guard, since they're the one place arbitrary external regex
    # reaches this code.
    trusted_patterns = list(_DEFAULT_KEEP_PATTERNS) + list(preset["extra_keep_patterns"])
    keep_regexes = [re.compile(p, re.IGNORECASE) for p in trusted_patterns]
    keep_regexes += [compile_safe_pattern(p, flags=re.IGNORECASE) for p in (keep_patterns or [])]

    lines = text.splitlines()
    if collapse_repeats:
        lines = _collapse_repeats(lines, repeat_threshold)

    if max_tokens is not None:
        result_lines, lines_elided, fit = _fit_to_budget(
            lines, effective_head, effective_tail, keep_regexes, max_kept_matches, max_tokens
        )
    else:
        fit = False
        if len(lines) > max_lines:
            result_lines, lines_elided = _elide_middle(
                lines, effective_head, effective_tail, keep_regexes, max_kept_matches
            )
        else:
            result_lines, lines_elided = lines, 0

    result_text = "\n".join(result_lines)
    return SandboxResult(
        text=result_text,
        original_tokens=original_tokens,
        final_tokens=count_tokens(result_text),
        lines_elided=lines_elided,
        profile=resolved_profile,
        fit_to_budget=fit,
    )
