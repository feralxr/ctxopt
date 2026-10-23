import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from ctxopt.llm import anthropic_backend


def test_factory_does_not_require_api_key_up_front():
    # Building the backend should never fail just because no key is set yet.
    backend = anthropic_backend(api_key=None)
    assert callable(backend)


def test_missing_api_key_raises_at_call_time(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    backend = anthropic_backend(api_key=None)
    with pytest.raises(RuntimeError, match="No Anthropic API key"):
        backend("some text")


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_successful_call_extracts_text_from_response():
    backend = anthropic_backend(api_key="fake-key")
    fake_payload = {"content": [{"type": "text", "text": "This is the summary."}]}

    with patch("urllib.request.urlopen", return_value=_FakeResponse(fake_payload)) as mock_urlopen:
        result = backend("long text to summarize")

    assert result == "This is the summary."
    assert mock_urlopen.called
    request = mock_urlopen.call_args[0][0]
    assert request.get_header("X-api-key") == "fake-key"
    body = json.loads(request.data.decode("utf-8"))
    assert "long text to summarize" in body["messages"][0]["content"]


def test_env_var_is_used_when_api_key_not_passed(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    backend = anthropic_backend()
    fake_payload = {"content": [{"type": "text", "text": "ok"}]}

    with patch("urllib.request.urlopen", return_value=_FakeResponse(fake_payload)):
        result = backend("text")
    assert result == "ok"


def test_http_error_is_wrapped_in_runtime_error():
    backend = anthropic_backend(api_key="fake-key")
    http_error = urllib.error.HTTPError(
        url="https://api.anthropic.com/v1/messages",
        code=401,
        msg="Unauthorized",
        hdrs=None,
        fp=io.BytesIO(b'{"error": "invalid api key"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(RuntimeError, match="401"):
            backend("text")


def test_multiple_text_blocks_are_concatenated():
    backend = anthropic_backend(api_key="fake-key")
    fake_payload = {"content": [{"type": "text", "text": "part one "}, {"type": "text", "text": "part two"}]}

    with patch("urllib.request.urlopen", return_value=_FakeResponse(fake_payload)):
        result = backend("text")
    assert result == "part one part two"
