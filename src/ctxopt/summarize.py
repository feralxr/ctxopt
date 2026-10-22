"""Summarize long conversations or logs down to a compact digest."""
from __future__ import annotations

from dataclasses import dataclass

from .tokens import count_tokens

_KEYWORDS = ("error", "decided", "result", "conclusion", "important", "todo", "fix", "bug")


@dataclass
class SummaryResult:
    text: str
    original_tokens: int
    final_tokens: int


def summarize_text(text: str, *, target_ratio: float = 0.25, min_tokens: int = 50) -> SummaryResult:
    """Produce a local, dependency-free extractive summary.

    This selects a representative subset of paragraphs rather than calling
    out to an LLM, so it works offline with zero API cost and is fully
    deterministic. For higher-quality abstractive summaries, pipe the output
    of this function into your own LLM call (see README for an example).
    """
    original_tokens = count_tokens(text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return SummaryResult(text="", original_tokens=original_tokens, final_tokens=0)

    target_tokens = max(min_tokens, int(original_tokens * target_ratio))

    scored = sorted(paragraphs, key=lambda p: _paragraph_score(p, paragraphs), reverse=True)

    kept: list[str] = []
    running_tokens = 0
    for p in scored:
        p_tokens = count_tokens(p)
        if running_tokens + p_tokens > target_tokens and kept:
            continue
        kept.append(p)
        running_tokens += p_tokens
        if running_tokens >= target_tokens:
            break

    # Restore original order so the digest still reads naturally.
    kept_set = set(kept)
    ordered = [p for p in paragraphs if p in kept_set]
    result_text = "\n\n".join(ordered)

    return SummaryResult(
        text=result_text,
        original_tokens=original_tokens,
        final_tokens=count_tokens(result_text),
    )


def _paragraph_score(paragraph: str, all_paragraphs: list[str]) -> float:
    """Simple, transparent heuristic: longer and keyword-dense paragraphs,
    plus the first and last paragraph of the document, score higher."""
    length_score = min(len(paragraph) / 500, 1.0)
    lowered = paragraph.lower()
    keyword_hits = sum(lowered.count(kw) for kw in _KEYWORDS)
    position_bonus = 1.0 if paragraph in (all_paragraphs[0], all_paragraphs[-1]) else 0.0
    return length_score + 0.3 * keyword_hits + 0.2 * position_bonus
