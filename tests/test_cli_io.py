from pathlib import Path

from ctxopt.cli import _read_bounded_file, _read_capped_stdin, _read_input


def _write_file(path: Path, head: str, middle_line: str, middle_count: int, tail: str) -> int:
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n")
        for _ in range(middle_count):
            f.write(middle_line + "\n")
        f.write(tail + "\n")
    return path.stat().st_size


# ---- bounded file reading ----------------------------------------------------


def test_small_file_is_read_in_full_without_bounding(tmp_path):
    path = tmp_path / "small.log"
    path.write_text("line one\nline two\nline three\n", encoding="utf-8")
    text = _read_input(path, max_input_bytes=20_000_000)
    assert text == "line one\nline two\nline three\n"
    assert "truncated" not in text and "too large" not in text


def test_large_file_is_bounded_via_head_and_tail(tmp_path):
    path = tmp_path / "huge.log"
    size = _write_file(path, "HEAD_MARKER", "filler " * 20, 50_000, "TAIL_MARKER")
    assert size > 1_000_000  # sanity check it's actually large

    text = _read_bounded_file(path, max_bytes=200_000)
    assert "HEAD_MARKER" in text
    assert "TAIL_MARKER" in text
    assert "too large to read in full" in text
    # Bounded means meaningfully smaller than the original, not just equal.
    assert len(text.encode("utf-8")) < size


def test_bounded_file_read_respects_byte_budget_roughly(tmp_path):
    path = tmp_path / "huge.log"
    _write_file(path, "HEAD", "x" * 100, 100_000, "TAIL")
    max_bytes = 50_000
    text = _read_bounded_file(path, max_bytes=max_bytes)
    # head + tail + marker shouldn't wildly exceed the requested budget.
    assert len(text.encode("utf-8")) < max_bytes + 1000


def test_read_input_dispatches_to_bounded_read_only_when_over_the_cap(tmp_path):
    path = tmp_path / "medium.log"
    path.write_text("x" * 1000, encoding="utf-8")
    # Cap is larger than the file -- should read it in full, untouched.
    text = _read_input(path, max_input_bytes=10_000)
    assert text == "x" * 1000

    # Cap is smaller than the file -- should go through the bounded path.
    text = _read_input(path, max_input_bytes=100)
    assert "too large" in text


# ---- capped stdin reading -----------------------------------------------------


def test_capped_stdin_reads_short_input_in_full(monkeypatch):
    import io

    data = b"hello world\n"
    monkeypatch.setattr("sys.stdin", type("Stdin", (), {"buffer": io.BytesIO(data)})())
    text = _read_capped_stdin(max_bytes=1000)
    assert text == "hello world\n"
    assert "truncated" not in text


def test_capped_stdin_truncates_long_input(monkeypatch):
    import io

    data = b"A" * 10_000
    monkeypatch.setattr("sys.stdin", type("Stdin", (), {"buffer": io.BytesIO(data)})())
    text = _read_capped_stdin(max_bytes=100)
    assert text.startswith("A" * 100)
    assert "truncated at 100 bytes" in text
    assert "9900 more bytes discarded" in text


def test_capped_stdin_handles_exactly_at_the_boundary(monkeypatch):
    import io

    data = b"B" * 500
    monkeypatch.setattr("sys.stdin", type("Stdin", (), {"buffer": io.BytesIO(data)})())
    text = _read_capped_stdin(max_bytes=500)
    assert text == "B" * 500
    assert "truncated" not in text
