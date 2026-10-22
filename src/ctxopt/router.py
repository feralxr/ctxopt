"""Heuristic complexity scoring and model routing."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .tokens import count_tokens

_CODE_PATTERN = re.compile(r"```|def |class |function |import |SELECT |#include")
_COMPLEX_KEYWORDS = (
    "architecture",
    "design",
    "refactor",
    "algorithm",
    "prove",
    "optimi",
    "concurren",
    "distributed",
    "security",
    "vulnerabilit",
    "debug",
    "analyze",
    "compare",
    "trade-off",
    "tradeoff",
)
_SIMPLE_KEYWORDS = (
    "what is",
    "define",
    "spell",
    "translate",
    "capital of",
    "convert",
    "list",
    "rename",
    "format",
)


@dataclass
class RouteDecision:
    score: float
    tier: str
    model: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class RouterConfig:
    """Maps complexity tiers to model names and sets the score thresholds
    between them. Override ``tiers`` to point at whichever models/providers
    you use."""

    tiers: dict[str, str] = field(
        default_factory=lambda: {
            "simple": "claude-haiku-5-5",
            "standard": "claude-sonnet-5",
            "complex": "claude-opus-5-5",
        }
    )
    simple_max_score: float = 0.3
    complex_min_score: float = 0.55


def score_prompt(prompt: str) -> tuple[float, list[str]]:
    """Return a 0-1 complexity score for ``prompt`` plus the reasons behind it."""
    reasons: list[str] = []
    score = 0.0

    tokens = count_tokens(prompt)
    length_component = min(tokens / 800, 1.0)
    score += 0.4 * length_component
    if length_component > 0.3:
        reasons.append(f"long input (~{tokens} tokens)")

    if _CODE_PATTERN.search(prompt):
        score += 0.25
        reasons.append("contains code or code-like syntax")

    lowered = prompt.lower()
    complex_hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in lowered)
    if complex_hits:
        score += min(0.55, 0.12 * complex_hits)
        reasons.append(f"{complex_hits} complexity keyword(s) matched")

    simple_hits = sum(1 for kw in _SIMPLE_KEYWORDS if kw in lowered)
    if simple_hits and complex_hits == 0:
        score -= 0.2
        reasons.append(f"{simple_hits} simple-query keyword(s) matched")

    score = max(0.0, min(1.0, score))
    return score, reasons


def route(prompt: str, config: RouterConfig | None = None) -> RouteDecision:
    """Score ``prompt`` and pick a model tier from ``config``."""
    config = config or RouterConfig()
    score, reasons = score_prompt(prompt)

    if score <= config.simple_max_score:
        tier = "simple"
    elif score >= config.complex_min_score:
        tier = "complex"
    else:
        tier = "standard"

    model = config.tiers.get(tier, config.tiers["standard"])
    return RouteDecision(score=score, tier=tier, model=model, reasons=reasons)
