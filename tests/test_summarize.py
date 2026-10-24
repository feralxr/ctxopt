from ctxopt.summarize import summarize_text

# ---- basic behavior ---------------------------------------------------------


def test_empty_text_returns_empty_summary():
    result = summarize_text("")
    assert result.text == ""
    assert result.final_tokens == 0


def test_summary_is_shorter_than_original():
    paragraphs = [f"This is paragraph number {i} with some extra filler words here." for i in range(20)]
    text = "\n\n".join(paragraphs)
    result = summarize_text(text, target_ratio=0.2)
    assert result.final_tokens <= result.original_tokens
    assert result.final_tokens > 0
    assert result.backend == "local"


def test_summary_preserves_paragraph_order():
    paragraphs = [f"Paragraph {i} discusses an important decided result." for i in range(10)]
    text = "\n\n".join(paragraphs)
    result = summarize_text(text, target_ratio=0.5)
    indices = [paragraphs.index(p) for p in result.text.split("\n\n") if p in paragraphs]
    assert indices == sorted(indices)


def test_max_tokens_overrides_ratio():
    paragraphs = [f"Filler paragraph {i} with enough words to matter here." for i in range(30)]
    text = "\n\n".join(paragraphs)
    result = summarize_text(text, target_ratio=0.9, max_tokens=40)
    assert result.final_tokens <= 60  # some slack for forced/position-bonus segments


def test_plain_document_has_no_turn_counts():
    text = "\n\n".join(f"Paragraph {i}." for i in range(5))
    result = summarize_text(text)
    assert result.turns_total is None
    assert result.turns_kept is None


# ---- conversation awareness --------------------------------------------------


def _make_conversation(n_turns: int) -> str:
    lines = []
    for i in range(n_turns):
        speaker = "User" if i % 2 == 0 else "Assistant"
        lines.append(f"{speaker}: message number {i} with some generic filler content in it.")
    return "\n".join(lines)


def test_conversation_is_detected_and_turn_counted():
    text = _make_conversation(20)
    result = summarize_text(text, target_ratio=0.3)
    assert result.turns_total == 20
    assert result.turns_kept is not None
    assert result.turns_kept <= 20


def test_keep_last_n_turns_always_preserved():
    text = _make_conversation(30)
    result = summarize_text(text, max_tokens=10, keep_last_n_turns=4)
    # Even with a tiny budget, the last 4 turns must survive.
    for i in range(26, 30):
        assert f"message number {i}" in result.text


def test_keep_last_n_turns_ignored_for_plain_documents():
    # "most recent" isn't meaningful for a non-conversation document.
    paragraphs = [f"Paragraph {i} with filler words to pad it out nicely." for i in range(10)]
    text = "\n\n".join(paragraphs)
    result = summarize_text(text, max_tokens=5, keep_last_n_turns=4)
    # Should not crash or force-include the last 4 paragraphs just because
    # keep_last_n_turns was set; turns_total is None so it's a no-op.
    assert result.turns_total is None


# ---- signal preservation (keep_patterns) -------------------------------------


def test_default_keep_patterns_preserve_error_segments():
    paragraphs = [f"Boring filler paragraph number {i} that says nothing much." for i in range(40)]
    paragraphs.insert(20, "A critical error occurred during the deployment process.")
    text = "\n\n".join(paragraphs)
    result = summarize_text(text, max_tokens=30)
    assert "critical error occurred" in result.text


def test_custom_keep_pattern_is_preserved():
    paragraphs = [f"Boring filler paragraph number {i} that says nothing much." for i in range(40)]
    paragraphs.insert(20, "CUSTOM_TAG: this line must survive no matter what.")
    text = "\n\n".join(paragraphs)
    result = summarize_text(text, max_tokens=30, keep_patterns=[r"CUSTOM_TAG"])
    assert "CUSTOM_TAG" in result.text


# ---- redundancy filtering -----------------------------------------------------


def test_near_duplicate_paragraphs_are_not_both_kept():
    base = "The deployment finished successfully with no errors reported anywhere important."
    near_dup = "The deployment finished successfully with no errors reported anywhere notable."
    distinct = "A completely unrelated paragraph about something else entirely different."
    text = "\n\n".join([base, near_dup, distinct] * 3)
    result = summarize_text(text, target_ratio=0.9, redundancy_threshold=0.8)
    # The near-duplicate pair should not both appear even at a generous ratio.
    assert not (base in result.text and near_dup in result.text)


# ---- LLM-backed (mocked) map-reduce -------------------------------------------


def test_llm_call_is_used_when_provided():
    calls = []

    def fake_llm(chunk: str) -> str:
        calls.append(chunk)
        return f"[summary of {len(chunk)} chars]"

    text = "\n\n".join(f"Paragraph {i} with some content." for i in range(5))
    result = summarize_text(text, llm_call=fake_llm)

    assert result.backend == "llm"
    assert len(calls) == 1  # small input fits in a single chunk
    assert result.text.startswith("[summary of")


def test_llm_call_map_reduce_for_large_input():
    calls = []

    def fake_llm(chunk: str) -> str:
        calls.append(chunk)
        return "summary chunk " * 10  # deliberately not tiny, so concatenating several exceeds target_tokens

    # Force multiple map chunks with a tiny llm_chunk_tokens budget, and a
    # tiny max_tokens so the concatenated chunk summaries still exceed
    # budget and trigger the final reduce call.
    text = "\n\n".join(f"Paragraph {i} with quite a bit of padding content in it to use tokens." for i in range(50))
    result = summarize_text(text, llm_call=fake_llm, llm_chunk_tokens=50, max_tokens=20)

    assert result.backend == "llm"
    map_calls = len(calls)
    assert map_calls >= 3  # several map calls over the chunks
    # One additional reduce call should have been made, bringing the total
    # above the number of chunks, and its (mocked) output is the final text.
    assert calls[-1].count("summary chunk") > 1  # the reduce call saw concatenated chunk summaries
    assert result.text == "summary chunk " * 10
