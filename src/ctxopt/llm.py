"""Optional LLM-backed summarization.

``summarize_text`` works entirely offline by default — nothing in this
library makes a network call unless you explicitly opt in. This module is
one building block for when you do want real abstractive summaries: it
wraps the Anthropic Messages API behind the ``Callable[[str], str]`` shape
``summarize_text(..., llm_call=...)`` expects, using only the standard
library (``urllib``), so using it costs no extra dependency and *not* using
it costs nothing at all.

You're not required to use this specific backend — any ``str -> str``
callable works as ``llm_call``, so you can point it at OpenAI, a local
model, or anything else.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Callable

_API_URL = "https://api.anthropic.com/v1/messages"
_DEFAULT_MODEL = "claude-haiku-5-5"
_DEFAULT_PROMPT = (
    "Summarize the following content concisely. Preserve key facts, "
    "decisions, and anything that looks like an error or action item. "
    "Respond with only the summary, no preamble or meta-commentary.\n\n{text}"
)


def anthropic_backend(
    *,
    api_key: str | None = None,
    model: str = _DEFAULT_MODEL,
    max_tokens: int = 1024,
    prompt_template: str = _DEFAULT_PROMPT,
    timeout: float = 60.0,
) -> Callable[[str], str]:
    """Return an ``llm_call`` function backed by the Anthropic Messages API.

    Reads ``ANTHROPIC_API_KEY`` from the environment when ``api_key`` isn't
    passed explicitly. The key is only checked when the returned function is
    actually called, not when this factory runs, so building the backend
    never fails just because the env var isn't set yet (e.g. while wiring up
    a CLI's ``--llm`` flag before the user has configured anything).
    """
    resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    def _call(text: str) -> str:
        if not resolved_key:
            raise RuntimeError(
                "No Anthropic API key found. Pass api_key=, or set the "
                "ANTHROPIC_API_KEY environment variable."
            )
        body = json.dumps(
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt_template.format(text=text)}],
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            _API_URL,
            data=body,
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": resolved_key,
                "anthropic-version": "2023-06-01",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"Anthropic API error {exc.code}: {detail}") from exc
        text_blocks = [
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        ]
        return "".join(text_blocks).strip()

    return _call
