from ctxopt.sandbox import sandbox_text


def test_short_text_is_untouched():
    text = "line1\nline2\nline3"
    result = sandbox_text(text, max_lines=200)
    assert result.text == text
    assert result.lines_elided == 0


def test_long_text_is_collapsed():
    lines = [f"line {i}" for i in range(500)]
    text = "\n".join(lines)
    result = sandbox_text(text, max_lines=100, head_lines=10, tail_lines=10)
    assert "lines elided" in result.text
    assert result.lines_elided == 500 - 20
    assert result.final_tokens < result.original_tokens


def test_repeated_lines_are_collapsed():
    text = "\n".join(["same line"] * 10 + ["different"])
    result = sandbox_text(text, collapse_repeats=True, repeat_threshold=3)
    assert "repeated" in result.text
    assert result.text.count("same line") == 1


def test_token_counts_are_reported():
    text = "hello world\n" * 300
    result = sandbox_text(text, max_lines=50, head_lines=5, tail_lines=5)
    assert result.original_tokens > 0
    assert result.final_tokens > 0
    assert result.final_tokens <= result.original_tokens
