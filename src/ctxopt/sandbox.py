"""Sandbox noisy tool output by collapsing repetitive or oversized blocks."""
from __future__ import annotations

from dataclasses import dataclass

from .tokens import count_tokens


@dataclass
class SandboxResult:
    text: str
    original_tokens: int
    final_tokens: int
    lines_elided: int


def sandbox_text(
    text: str,
    *,
    max_lines: int = 200,
    head_lines: int = 40,
    tail_lines: int = 40,
    collapse_repeats: bool = True,
    repeat_threshold: int = 3,
) -> SandboxResult:
    """Collapse long or repetitive tool output into a compact form.

    Keeps the first ``head_lines`` and last ``tail_lines`` lines intact and
    replaces everything in between with a single marker line noting how many
    lines were elided. Consecutive duplicate lines (e.g. repeated log spam)
    are also collapsed when ``collapse_repeats`` is True.
    """
    original_tokens = count_tokens(text)
    lines = text.splitlines()

    if collapse_repeats:
        lines = _collapse_repeats(lines, threshold=repeat_threshold)

    lines_elided = 0
    if len(lines) > max_lines:
        head = lines[:head_lines]
        tail = lines[-tail_lines:] if tail_lines else []
        lines_elided = max(0, len(lines) - len(head) - len(tail))
        marker = f"... [{lines_elided} lines elided] ..."
        lines = head + [marker] + tail

    result_text = "\n".join(lines)
    return SandboxResult(
        text=result_text,
        original_tokens=original_tokens,
        final_tokens=count_tokens(result_text),
        lines_elided=lines_elided,
    )


def _collapse_repeats(lines: list[str], threshold: int = 3) -> list[str]:
    """Collapse runs of ``threshold`` or more identical lines into one line
    plus a note of how many additional repeats were removed."""
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i
        while j < n and lines[j] == lines[i]:
            j += 1
        run_len = j - i
        if run_len >= threshold:
            out.append(lines[i])
            out.append(f"... [line repeated {run_len - 1} more times] ...")
        else:
            out.extend(lines[i:j])
        i = j
    return out
