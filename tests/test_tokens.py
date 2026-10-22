from ctxopt.tokens import count_tokens


def test_empty_string_is_zero_tokens():
    assert count_tokens("") == 0


def test_nonempty_text_has_positive_tokens():
    assert count_tokens("hello world, this is a test") > 0


def test_longer_text_has_more_tokens():
    short = "hello"
    long = "hello " * 100
    assert count_tokens(long) > count_tokens(short)
