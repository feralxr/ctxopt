import json
import string

from ctxopt.sandbox import detect_profile, sandbox_text


def _idx_to_word(i: int) -> str:
    """Encode an integer as a base-26 letter string instead of digits, so
    generated test lines stay genuinely distinct after number-normalization
    (which intentionally treats standalone digits as volatile noise)."""
    s = ""
    while True:
        s = string.ascii_lowercase[i % 26] + s
        i //= 26
        if i == 0:
            break
        i -= 1
    return s


# ---- basic behavior -------------------------------------------------------


def test_short_text_is_untouched():
    text = "alpha\nbeta\ngamma"
    result = sandbox_text(text, max_lines=200)
    assert result.text == text
    assert result.lines_elided == 0


def test_long_generic_text_is_elided():
    lines = [f"unique line about {_idx_to_word(i)} topic with no repeats" for i in range(500)]
    text = "\n".join(lines)
    result = sandbox_text(text, max_lines=100, head_lines=10, tail_lines=10, profile="generic")
    assert "lines elided" in result.text
    assert result.final_tokens < result.original_tokens
    assert result.profile == "generic"


def test_exact_repeated_lines_are_collapsed():
    text = "\n".join(["same line"] * 10 + ["different"])
    result = sandbox_text(text, collapse_repeats=True, repeat_threshold=3)
    assert "collapsed" in result.text
    assert result.text.count("same line") == 1


# ---- near-duplicate (normalized) collapsing -------------------------------


def test_near_duplicate_lines_with_changing_counters_collapse():
    lines = [f"Processing item {i} of 10000" for i in range(50)]
    text = "\n".join(lines)
    result = sandbox_text(text, collapse_repeats=True, repeat_threshold=3)
    assert "collapsed" in result.text
    # Only the first representative line plus the marker should remain.
    assert result.text.count("Processing item") == 1


def test_near_duplicate_lines_with_changing_timestamps_collapse():
    lines = [f"2026-01-01T12:00:{i:02d}Z heartbeat ok" for i in range(10)]
    text = "\n".join(lines)
    result = sandbox_text(text, collapse_repeats=True, repeat_threshold=3)
    assert "collapsed" in result.text
    assert result.text.count("heartbeat ok") == 1


def test_lines_with_hex_addresses_collapse():
    lines = [f"allocated block at 0x{i:08x}" for i in range(20)]
    text = "\n".join(lines)
    result = sandbox_text(text, collapse_repeats=True, repeat_threshold=3)
    assert result.text.count("allocated block") == 1


# ---- keep-pattern preservation ---------------------------------------------


def test_error_lines_are_preserved_inside_elided_region():
    noise = [f"noise line {i}" for i in range(300)]
    noise.insert(150, "CRITICAL: Exception occurred in worker thread")
    text = "\n".join(noise)
    result = sandbox_text(text, max_lines=50, head_lines=5, tail_lines=5)
    assert "Exception occurred in worker thread" in result.text


def test_custom_keep_pattern_is_preserved():
    noise = [f"noise line {i}" for i in range(300)]
    noise.insert(150, "CUSTOM_MARKER: important business event")
    text = "\n".join(noise)
    result = sandbox_text(
        text, max_lines=50, head_lines=5, tail_lines=5, keep_patterns=[r"CUSTOM_MARKER"]
    )
    assert "CUSTOM_MARKER: important business event" in result.text


def test_max_kept_matches_caps_preserved_lines():
    noise = [f"error line {i}" for i in range(200)]
    text = "\n".join(noise)
    result = sandbox_text(text, max_lines=10, head_lines=2, tail_lines=2, max_kept_matches=5)
    assert result.text.count("error line") <= 2 + 5 + 2  # head + kept cap + tail


# ---- token-budget fitting ---------------------------------------------------


def test_max_tokens_shrinks_output_to_fit():
    lines = [f"this is a moderately long diagnostic line about {_idx_to_word(i)} with some words" for i in range(2000)]
    text = "\n".join(lines)
    result = sandbox_text(text, max_tokens=200)
    assert result.final_tokens <= 220  # small slack for the marker line itself
    assert result.fit_to_budget is True


def test_max_tokens_with_errors_still_respects_budget_roughly():
    lines = [f"noise {i}" for i in range(1000)]
    for idx in (100, 300, 500, 700, 900):
        lines[idx] = f"error: something broke at step {idx}"
    text = "\n".join(lines)
    result = sandbox_text(text, max_tokens=150)
    assert result.final_tokens <= 300  # errors can push it over target, but not wildly


# ---- profile detection -----------------------------------------------------


def test_detect_profile_json():
    assert detect_profile('{"a": 1, "b": [1, 2, 3]}') == "json"


def test_detect_profile_diff():
    diff_text = (
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1,3 +1,3 @@\n"
        "-old line\n"
        "+new line\n"
    )
    assert detect_profile(diff_text) == "diff"


def test_detect_profile_stacktrace():
    trace = (
        "Traceback (most recent call last):\n"
        '  File "app.py", line 10, in <module>\n'
        "    main()\n"
        '  File "app.py", line 5, in main\n'
        "    raise ValueError('boom')\n"
        "ValueError: boom\n"
    )
    assert detect_profile(trace) == "stacktrace"


def test_detect_profile_generic_fallback():
    assert detect_profile("just a normal line of text\nanother line\n") == "generic"


# ---- JSON profile: structural truncation -----------------------------------


def test_json_profile_truncates_large_array():
    data = {"items": list(range(500))}
    text = json.dumps(data)
    result = sandbox_text(text, json_max_items=10)
    assert result.profile == "json"
    parsed = json.loads(result.text)
    assert len(parsed["items"]) <= 11  # 10 kept + 1 omission marker string
    assert any(isinstance(x, str) and "omitted" in x for x in parsed["items"])


def test_json_profile_truncates_long_strings():
    data = {"blob": "x" * 5000}
    text = json.dumps(data)
    result = sandbox_text(text, json_max_string_len=100)
    parsed = json.loads(result.text)
    assert len(parsed["blob"]) < 200
    assert "truncated" in parsed["blob"]


def test_json_profile_output_is_valid_json():
    data = {"a": [1, 2, 3] * 100, "b": {"nested": "value", "deep": {"x": list(range(50))}}}
    text = json.dumps(data)
    result = sandbox_text(text)
    json.loads(result.text)  # must not raise


def test_non_parsable_json_like_text_falls_back_to_generic():
    text = "{ this looks like json but is not valid"
    result = sandbox_text(text, max_lines=1)
    assert result.profile == "generic"


# ---- diff profile ------------------------------------------------------------


def test_diff_profile_keeps_file_headers_when_eliding():
    hunk_lines = [f"+added line {i}" for i in range(300)]
    diff_text = "diff --git a/big.py b/big.py\n--- a/big.py\n+++ b/big.py\n@@ -1,300 +1,300 @@\n" + "\n".join(
        hunk_lines
    )
    result = sandbox_text(diff_text, max_lines=20, profile="diff")
    assert "diff --git a/big.py b/big.py" in result.text
    assert "--- a/big.py" in result.text
    assert "+++ b/big.py" in result.text


# ---- token accounting --------------------------------------------------------


def test_token_counts_are_reported_and_consistent():
    text = "hello world\n" * 300
    result = sandbox_text(text, max_lines=50, head_lines=5, tail_lines=5)
    assert result.original_tokens > 0
    assert result.final_tokens > 0
    assert result.final_tokens <= result.original_tokens
