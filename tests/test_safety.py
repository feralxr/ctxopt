import re

import pytest

from ctxopt.safety import MAX_PATTERN_LENGTH, UnsafePatternError, compile_safe_pattern, timed_search


def test_safe_pattern_compiles_and_matches():
    rx = compile_safe_pattern(r"payment_id=\w+")
    assert rx.search("payment_id=abc123")
    assert not rx.search("nothing here")


def test_flags_are_applied():
    rx = compile_safe_pattern(r"error", flags=re.IGNORECASE)
    assert rx.search("ERROR: something broke")


def test_overlong_pattern_is_rejected():
    with pytest.raises(UnsafePatternError, match="character limit"):
        compile_safe_pattern("a" * (MAX_PATTERN_LENGTH + 1))


def test_nested_quantifier_is_rejected_by_static_check():
    with pytest.raises(UnsafePatternError, match="nested quantifier"):
        compile_safe_pattern(r"(a+)+$")


def test_nested_star_quantifier_is_rejected():
    with pytest.raises(UnsafePatternError, match="nested quantifier"):
        compile_safe_pattern(r"(a*)*b")


def test_ambiguous_alternation_blowup_is_rejected_by_smoke_test():
    # Not caught by the static nested-quantifier heuristic (no quantifier
    # inside the group), but catastrophic in practice -- must be caught by
    # the runtime smoke test instead.
    with pytest.raises(UnsafePatternError, match="took too long"):
        compile_safe_pattern(r"(a|a)*b")


def test_malformed_regex_raises_unsafe_pattern_error_not_re_error():
    with pytest.raises(UnsafePatternError, match="Invalid regex"):
        compile_safe_pattern(r"(unclosed")


def test_simple_safe_patterns_are_not_false_positives():
    # A handful of realistic, legitimate keep-patterns should all compile
    # fine and quickly -- the guard shouldn't be so aggressive it rejects
    # ordinary usage.
    patterns = [
        r"trace_id=\w+",
        r"\bCUSTOM_TAG\b",
        r"^ERROR:",
        r"user_id:\s*\d+",
        r"[A-Z]{3,5}-\d+",  # ticket-style IDs like ABC-123
    ]
    for p in patterns:
        rx = compile_safe_pattern(p)
        assert rx is not None


def test_timed_search_returns_true_for_matching_safe_pattern():
    assert timed_search(r"hello", "hello world", timeout=1.0) is True


def test_timed_search_returns_false_for_non_matching_safe_pattern():
    assert timed_search(r"xyz", "hello world", timeout=1.0) is False


def test_timed_search_returns_none_on_timeout():
    # Directly exercise the timeout path with a known-catastrophic pattern,
    # independent of compile_safe_pattern's higher-level wrapping.
    result = timed_search(r"(a|a)*b", "a" * 28 + "!", timeout=0.2)
    assert result is None
