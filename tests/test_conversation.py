import json

from ctxopt.conversation import Turn, parse_conversation


def test_plain_document_is_not_a_conversation():
    text = "This is just a regular paragraph of text.\n\nAnd another one."
    assert parse_conversation(text) is None


def test_single_label_line_is_not_a_conversation():
    # One "Label: text" line shouldn't misfire as a two-speaker conversation.
    text = "Note: this is an important caveat about the system.\nMore detail follows here."
    assert parse_conversation(text) is None


def test_speaker_prefixed_text_is_parsed():
    text = (
        "User: What's the weather like?\n"
        "Assistant: It's sunny today.\n"
        "User: Great, thanks!\n"
        "Assistant: You're welcome."
    )
    turns = parse_conversation(text)
    assert turns is not None
    assert len(turns) == 4
    assert turns[0] == Turn(role="User", content="What's the weather like?")
    assert turns[-1] == Turn(role="Assistant", content="You're welcome.")


def test_speaker_prefixed_text_accumulates_multiline_content():
    text = "User: line one\nline two\nAssistant: reply"
    turns = parse_conversation(text)
    assert turns is not None
    assert turns[0].content == "line one\nline two"
    assert turns[1].content == "reply"


def test_json_message_list_is_parsed():
    data = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]
    turns = parse_conversation(json.dumps(data))
    assert turns == [Turn(role="user", content="hello"), Turn(role="assistant", content="hi there")]


def test_json_with_content_blocks_is_flattened():
    data = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "part one"}, {"type": "text", "text": "part two"}]},
    ]
    turns = parse_conversation(json.dumps(data))
    assert turns[1].content == "part one\npart two"


def test_malformed_json_list_falls_back_to_none():
    # Looks like JSON but isn't a valid message list.
    assert parse_conversation('[{"not_role": 1}]') is None


def test_empty_text_is_not_a_conversation():
    assert parse_conversation("") is None
