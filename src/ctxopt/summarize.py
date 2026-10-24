"""Summarize long conversations or documents down to a compact digest.

Design
------
1. Detect structure: a transcript parsed by :mod:`ctxopt.conversation` is
   summarized turn by turn; anything else falls back to paragraph-based
   segmentation. Either way, the unit being scored, filtered, and selected
   is called a "segment" below.
2. Signal preservation: segments matching ``keep_patterns`` (error/
   exception/decided/important by default) and the most recent
   ``keep_last_n_turns`` segments are always kept, regardless of budget —
   the same philosophy as :mod:`ctxopt.sandbox`'s keep patterns.
3. Score the rest by a transparent heuristic (length, keyword density,
   first/last position) and greedily add the highest-scoring segments until
   the token budget is met, skipping any segment that's near-duplicate
   (Jaccard-similar) to one already kept.
4. Restore chronological order so the digest still reads naturally.

This local pipeline (``llm_call=None``, the default) is fully offline,
deterministic, and costs nothing. For abstractive summaries, pass a
``llm_call: str -> str`` function (see :func:`ctxopt.llm.anthropic_backend`
for a ready-made one) — segments are grouped into token-sized chunks,
summarized chunk by chunk, and reduced once more if multiple chunks remain,
i.e. a small map-reduce rather than one unbounded call.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from .conversation import parse_conversation
from .safety import compile_safe_pattern
from .tokens import count_tokens

_SCORE_KEYWORDS = (
    "error",
    "exception",
    "decided",
    "decision",
    "result",
    "conclusion",
    "important",
    "todo",
    "fix",
    "bug",
    "action item",
    "deadline",
)

# Conservative on purpose: these are *always* kept regardless of budget, so
# the default list stays small to avoid a noisy transcript blowing past the
# target size. Softer signals still help via _SCORE_KEYWORDS above.
_DEFAULT_KEEP_PATTERNS = (r"\berror\b", r"\bexception\b", r"\bdecided\b", r"\bimportant\b")

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")


@dataclass
class SummaryResult:
    text: str
    original_tokens: int
    final_tokens: int
    turns_total: int | None = None
    turns_kept: int | None = None
    backend: str = "local"


@dataclass
class _Segment:
    index: int
    role: str | None
    content: str


def _build_segments(text: str) -> tuple[list[_Segment], bool]:
    turns = parse_conversation(text)
    if turns:
        return [_Segment(i, t.role, t.content) for i, t in enumerate(turns)], True
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [_Segment(i, None, p) for i, p in enumerate(paragraphs)], False


def _render_segment(seg: _Segment) -> str:
    return f"{seg.role}: {seg.content}" if seg.role is not None else seg.content


def _word_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _similarity(a: str, b: str) -> float:
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    union = len(wa | wb)
    return len(wa & wb) / union if union else 0.0


def _segment_score(content: str, is_first: bool, is_last: bool) -> float:
    length_score = min(len(content) / 500, 1.0)
    lowered = content.lower()
    keyword_hits = sum(lowered.count(kw) for kw in _SCORE_KEYWORDS)
    position_bonus = 1.0 if (is_first or is_last) else 0.0
    return length_score + 0.3 * keyword_hits + 0.2 * position_bonus


def _extractive_reduce(
    segments: list[_Segment],
    target_tokens: int,
    keep_regexes: list[re.Pattern],
    keep_last_n: int,
    redundancy_threshold: float,
) -> list[_Segment]:
    if not segments:
        return []
    n = len(segments)

    forced_idx: set[int] = set()
    if keep_last_n > 0:
        forced_idx.update(range(max(0, n - keep_last_n), n))
    if keep_regexes:
        for seg in segments:
            if any(rx.search(seg.content) for rx in keep_regexes):
                forced_idx.add(seg.index)

    kept = [s for s in segments if s.index in forced_idx]
    candidates = [s for s in segments if s.index not in forced_idx]

    running_tokens = sum(count_tokens(_render_segment(s)) for s in kept)
    kept_contents = [s.content for s in kept]

    scored = sorted(
        candidates,
        key=lambda s: _segment_score(s.content, s.index == 0, s.index == n - 1),
        reverse=True,
    )

    for seg in scored:
        if running_tokens >= target_tokens and kept:
            break
        if any(_similarity(seg.content, kc) >= redundancy_threshold for kc in kept_contents):
            continue
        seg_tokens = count_tokens(_render_segment(seg))
        if running_tokens + seg_tokens > target_tokens and kept:
            continue
        kept.append(seg)
        kept_contents.append(seg.content)
        running_tokens += seg_tokens

    kept.sort(key=lambda s: s.index)
    return kept


def _llm_reduce(
    segments: list[_Segment],
    is_conversation: bool,
    target_tokens: int,
    llm_call: Callable[[str], str],
    llm_chunk_tokens: int,
    original_tokens: int,
) -> SummaryResult:
    rendered = [_render_segment(s) for s in segments]

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for piece in rendered:
        piece_tokens = count_tokens(piece)
        if current and current_tokens + piece_tokens > llm_chunk_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(piece)
        current_tokens += piece_tokens
    if current:
        chunks.append("\n\n".join(current))

    chunk_summaries = [llm_call(chunk) for chunk in chunks]

    if len(chunk_summaries) == 1:
        final_text = chunk_summaries[0]
    else:
        combined = "\n\n".join(chunk_summaries)
        final_text = llm_call(combined) if count_tokens(combined) > target_tokens else combined

    return SummaryResult(
        text=final_text,
        original_tokens=original_tokens,
        final_tokens=count_tokens(final_text),
        turns_total=len(segments) if is_conversation else None,
        turns_kept=None,
        backend="llm",
    )


def summarize_text(
    text: str,
    *,
    target_ratio: float = 0.25,
    max_tokens: int | None = None,
    min_tokens: int = 50,
    keep_patterns: list[str] | None = None,
    keep_last_n_turns: int = 2,
    redundancy_threshold: float = 0.85,
    llm_call: Callable[[str], str] | None = None,
    llm_chunk_tokens: int = 1500,
) -> SummaryResult:
    """Condense ``text`` into a shorter digest.

    Parameters
    ----------
    target_ratio:
        Target size as a fraction of the original, used when ``max_tokens``
        isn't set.
    max_tokens:
        Target size in tokens directly. Takes priority over ``target_ratio``.
    keep_patterns:
        Extra regexes (on top of the error/exception/decided/important
        defaults) whose matching segments are always kept.
    keep_last_n_turns:
        Always keep this many of the most recent segments, regardless of
        score — only meaningful when ``text`` parses as a conversation (see
        :mod:`ctxopt.conversation`); ignored for plain documents, where
        "most recent" isn't a meaningful idea.
    redundancy_threshold:
        Jaccard word-overlap above which a candidate segment is treated as
        a near-duplicate of one already kept, and skipped.
    llm_call:
        Optional ``str -> str`` function for abstractive summarization
        (e.g. :func:`ctxopt.llm.anthropic_backend`). When set, the local
        extractive heuristic above is bypassed in favor of a small
        map-reduce over ``llm_call``.
    llm_chunk_tokens:
        Only used with ``llm_call``: max size of each chunk sent to it.
    """
    original_tokens = count_tokens(text)
    segments, is_conversation = _build_segments(text)
    if not segments:
        return SummaryResult(text="", original_tokens=original_tokens, final_tokens=0, turns_total=0, turns_kept=0)

    target_tokens = max_tokens if max_tokens is not None else max(min_tokens, int(original_tokens * target_ratio))

    if llm_call is not None:
        return _llm_reduce(segments, is_conversation, target_tokens, llm_call, llm_chunk_tokens, original_tokens)

    # As in sandbox.py: our own default patterns are trusted and compiled
    # directly; user-supplied keep_patterns go through the ReDoS guard.
    keep_regexes = [re.compile(p, re.IGNORECASE) for p in _DEFAULT_KEEP_PATTERNS]
    keep_regexes += [compile_safe_pattern(p, flags=re.IGNORECASE) for p in (keep_patterns or [])]
    effective_keep_last = keep_last_n_turns if is_conversation else 0

    kept = _extractive_reduce(segments, target_tokens, keep_regexes, effective_keep_last, redundancy_threshold)
    result_text = "\n\n".join(_render_segment(s) for s in kept)

    return SummaryResult(
        text=result_text,
        original_tokens=original_tokens,
        final_tokens=count_tokens(result_text),
        turns_total=len(segments) if is_conversation else None,
        turns_kept=len(kept) if is_conversation else None,
        backend="local",
    )
