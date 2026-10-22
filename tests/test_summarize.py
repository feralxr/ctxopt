from ctxopt.summarize import summarize_text


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


def test_summary_preserves_paragraph_order():
    paragraphs = [f"Paragraph {i} discusses an important decided result." for i in range(10)]
    text = "\n\n".join(paragraphs)
    result = summarize_text(text, target_ratio=0.5)

    indices = [paragraphs.index(p) for p in result.text.split("\n\n") if p in paragraphs]
    assert indices == sorted(indices)


def test_short_text_below_min_tokens_is_kept_whole_ish():
    text = "Just one short paragraph."
    result = summarize_text(text, target_ratio=0.1, min_tokens=5)
    assert result.text != ""
