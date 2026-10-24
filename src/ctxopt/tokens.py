"""Token counting utilities."""
from __future__ import annotations

from typing import Any, Optional

_ENCODING: Optional[Any] = None
try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken is optional at runtime
    pass


def count_tokens(text: str) -> int:
    """Return an approximate token count for ``text``.

    Uses tiktoken's cl100k_base encoding when available, falling back to a
    whitespace-based heuristic (~0.75 tokens per word) otherwise so the rest
    of the library keeps working even without tiktoken installed.
    """
    if not text:
        return 0
    if _ENCODING is not None:
        return len(_ENCODING.encode(text))
    words = text.split()
    return max(1, int(len(words) / 0.75))
