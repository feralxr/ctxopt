"""Guardrails for compiling and running user-supplied regular expressions.

``ctxopt`` accepts user-supplied regex in a few places — ``sandbox``'s and
``summarize``'s ``keep_patterns``. A pattern like ``(a+)+$`` matched against
a crafted input can take catastrophically long (classic ReDoS) due to
exponential backtracking in Python's ``re`` engine, which has no built-in
timeout or cancellation. Two independent layers of defense, applied only to
*user*-supplied patterns — the small set of patterns ctxopt authors itself
(defaults, profile presets) are trusted and compiled normally:

1. A static heuristic that rejects the most common catastrophic-backtracking
   shape — a quantified group nested inside another quantifier, e.g.
   ``(a+)+``, ``(a*)*``. This is deterministic and catches the common case,
   but isn't exhaustive: plenty of dangerous patterns (e.g. ambiguous
   alternation like ``(a|a)*``) don't match this shape.
2. A smoke test at *registration* time: the compiled pattern is run once
   against a short adversarial probe string, with a wall-clock timeout. If
   it doesn't finish in time, the pattern is rejected before it's ever used
   on real data. This catches shapes the static check misses, without
   paying a timeout-wrapper's overhead on every real line scanned
   afterwards — real matching, once a pattern passes this check, runs at
   normal `re` speed with no per-call thread overhead.

The smoke test is a heuristic, not a proof: a pattern can still pass it and
turn out to be slow on some other input. It catches the common, deliberately
or accidentally catastrophic cases without materially slowing down normal
use.
"""
from __future__ import annotations

import multiprocessing
import re

MAX_PATTERN_LENGTH = 200
SMOKE_TEST_TIMEOUT = 0.5

# A quantified group nested inside another quantifier -- (x+)+, (x*)*,
# (x+)*, (x*)+, and the same with a character class or "." in place of a
# literal -- is the classic catastrophic-backtracking trigger.
_NESTED_QUANTIFIER_RE = re.compile(r"\([^()]*[+*][^()]*\)[+*]")

# A short string engineered to trigger exponential backtracking in common
# ReDoS shapes (repeated character with no valid terminator) without being
# so large that a *safe* pattern takes meaningfully longer to run on it.
_SMOKE_PROBE = "a" * 24 + "1" * 8 + "!"


class UnsafePatternError(ValueError):
    """Raised when a user-supplied regex pattern is rejected as unsafe."""


def compile_safe_pattern(pattern: str, *, flags: int = 0) -> re.Pattern:
    """Compile ``pattern``, rejecting it first if it's too long, matches a
    known catastrophic-backtracking shape, or fails a runtime smoke test.

    Raises :class:`UnsafePatternError` (a ``ValueError`` subclass) rather
    than a bare ``re.error``, so callers can tell "this pattern is
    dangerous or malformed" apart from other errors.
    """
    if len(pattern) > MAX_PATTERN_LENGTH:
        raise UnsafePatternError(
            f"Pattern is {len(pattern)} characters, over the {MAX_PATTERN_LENGTH}-character limit."
        )
    if _NESTED_QUANTIFIER_RE.search(pattern):
        raise UnsafePatternError(
            f"Pattern {pattern!r} contains a nested quantifier (e.g. a group "
            "like `(x+)+`), a common cause of catastrophic backtracking. "
            "Rewrite it to avoid a quantified group inside another "
            "quantified group."
        )

    try:
        compiled = re.compile(pattern, flags)
    except re.error as exc:
        raise UnsafePatternError(f"Invalid regex {pattern!r}: {exc}") from exc

    if timed_search(pattern, _SMOKE_PROBE, flags=flags, timeout=SMOKE_TEST_TIMEOUT) is None:
        raise UnsafePatternError(
            f"Pattern {pattern!r} took too long matching a short test string "
            "and was rejected as a likely catastrophic-backtracking risk."
        )

    return compiled


def _match_worker(pattern: str, flags: int, text: str, result_queue) -> None:
    try:
        compiled = re.compile(pattern, flags)
        result_queue.put(bool(compiled.search(text)))
    except re.error:
        result_queue.put(False)


def timed_search(pattern: str, text: str, *, flags: int = 0, timeout: float = 0.5):
    """Run ``re.search(pattern, text)`` with a hard wall-clock timeout,
    in a separate process.

    A thread-based timeout *cannot* actually protect against a
    catastrophically slow regex in CPython: the GIL isn't released during
    the C-level backtracking loop, so a ``threading.Thread.join(timeout)``
    call will itself block for the full duration of the match before it
    gets a chance to notice the timeout elapsed — the "timeout" is a no-op
    in practice. Only OS-level process termination can actually preempt it,
    so this spawns a subprocess and kills it if it overruns.

    Returns ``True``/``False`` for a match/no-match that completed in time,
    or ``None`` if it didn't finish within ``timeout`` seconds (the
    subprocess is terminated in that case) or if a subprocess genuinely
    couldn't be started in this environment (e.g. a sandboxed runtime that
    disables process creation) — callers should treat ``None`` as "could
    not verify this pattern is safe" either way.
    """
    try:
        ctx: multiprocessing.context.BaseContext
        try:
            ctx = multiprocessing.get_context("fork")  # fast; Linux/macOS only
        except ValueError:
            ctx = multiprocessing.get_context("spawn")  # portable fallback (e.g. Windows)
        result_queue = ctx.Queue()
        proc = ctx.Process(target=_match_worker, args=(pattern, flags, text, result_queue))
        proc.start()
    except Exception:
        return None

    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return None
    try:
        return result_queue.get_nowait()
    except Exception:
        return None
