"""Heuristic complexity scoring and model routing.

Design
------
``score_prompt`` scores a prompt on 0-1 using a few transparent, auditable
signals (never a black box): input length, presence of code, density of
complexity/simplicity keywords, multi-step instruction phrasing, and —
optionally — how much conversation history has already accumulated, since a
long-running session often warrants a more capable model even when the
latest message looks simple on its own.

Keyword density is normalized per 100 tokens rather than counted raw, so a
long prompt doesn't get flagged "complex" just for containing a handful of
matches relative to its size — a 2000-token prompt with 3 keyword hits is
not nearly as clearly complex as a 50-token prompt with 3 hits.

``route`` turns that score into a tier + model using ``RouterConfig``, and
layers in two things that a routing heuristic actually needs in production:

- **Cost awareness**: each tier can carry per-token economics
  (``RouterConfig.economics``); ``route`` reports an estimated cost and
  latency, and — if ``max_cost_per_call`` is set — will downgrade to a
  cheaper tier that fits the budget, noting why in ``reasons``.
- **Escalation**: ``looks_like_refusal_or_shallow`` is a cheap heuristic for
  "did the model dodge this or phone it in", and ``escalate`` bumps a
  decision one tier up so a caller can retry — closing the loop that a
  static one-shot router can't.

``RoutingLog`` is an optional, append-only JSONL audit trail of routing
decisions (tier distribution, scores, costs) for inspecting how the router
is actually behaving over time; it's opt-in and stores only a prompt hash
by default, not the prompt text.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .tokens import count_tokens

_CODE_PATTERN = re.compile(r"```|def |class |function |import |SELECT |#include")
_MULTISTEP_PATTERN = re.compile(r"\b(first|firstly|then|next|finally|lastly|step \d|after that)\b", re.IGNORECASE)

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

_REFUSAL_PATTERNS = (
    r"\bi can'?t help with that\b",
    r"\bi cannot assist\b",
    r"\bi'?m not able to\b",
    r"\bi don'?t have enough information\b",
    r"\bi'?m sorry,? but\b",
    r"\bas an ai\b",
)
_REFUSAL_REGEXES = [re.compile(p, re.IGNORECASE) for p in _REFUSAL_PATTERNS]


@dataclass
class ScoringWeights:
    """Tunable weights for each scoring signal, and the keyword lists
    themselves — override either to adapt the heuristic to a different
    language or domain without touching the scoring logic."""

    length_weight: float = 0.3
    code_weight: float = 0.2
    keyword_weight: float = 0.55
    multistep_weight: float = 0.1
    history_weight: float = 0.15
    simple_penalty: float = 0.2
    complex_keywords: tuple = _COMPLEX_KEYWORDS
    simple_keywords: tuple = _SIMPLE_KEYWORDS


@dataclass
class TierEconomics:
    """Per-tier cost and latency estimates, used to compute
    ``RouteDecision.estimated_cost_usd``/``estimated_latency_ms`` and to
    enforce ``RouterConfig.max_cost_per_call``."""

    input_cost_per_mtok: float = 0.0  # USD per 1M input tokens
    output_cost_per_mtok: float = 0.0  # USD per 1M output tokens
    avg_latency_ms: float = 0.0


_DEFAULT_ECONOMICS = {
    "simple": TierEconomics(input_cost_per_mtok=0.8, output_cost_per_mtok=4.0, avg_latency_ms=400),
    "standard": TierEconomics(input_cost_per_mtok=3.0, output_cost_per_mtok=15.0, avg_latency_ms=1200),
    "complex": TierEconomics(input_cost_per_mtok=15.0, output_cost_per_mtok=75.0, avg_latency_ms=3000),
}


@dataclass
class RouteDecision:
    score: float
    tier: str
    model: str
    reasons: list[str] = field(default_factory=list)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost_usd: float | None = None
    estimated_latency_ms: float | None = None
    downgraded_for_budget: bool = False


@dataclass
class RouterConfig:
    """Maps complexity tiers to model names, sets the score thresholds
    between them, and (optionally) the economics used for cost-aware
    routing. ``tiers`` keeps the simple ``{tier: model_name}`` shape so
    existing configs keep working unchanged."""

    tiers: dict[str, str] = field(
        default_factory=lambda: {
            "simple": "claude-haiku-5-5",
            "standard": "claude-sonnet-5",
            "complex": "claude-opus-5-5",
        }
    )
    tier_order: tuple = ("simple", "standard", "complex")
    simple_max_score: float = 0.3
    complex_min_score: float = 0.55
    economics: dict = field(default_factory=lambda: dict(_DEFAULT_ECONOMICS))
    max_cost_per_call: float | None = None
    weights: ScoringWeights = field(default_factory=ScoringWeights)


def score_prompt(
    prompt: str,
    *,
    history_tokens: int = 0,
    weights: ScoringWeights | None = None,
) -> tuple[float, list[str]]:
    """Return a 0-1 complexity score for ``prompt`` plus the reasons behind
    it. ``history_tokens`` — the accumulated size of the conversation so
    far — nudges the score up for long-running sessions even if the latest
    message is short."""
    weights = weights or ScoringWeights()
    reasons: list[str] = []
    score = 0.0

    tokens = count_tokens(prompt)
    length_component = min(tokens / 800, 1.0)
    score += weights.length_weight * length_component
    if length_component > 0.3:
        reasons.append(f"long input (~{tokens} tokens)")

    if _CODE_PATTERN.search(prompt):
        score += weights.code_weight
        reasons.append("contains code or code-like syntax")

    lowered = prompt.lower()
    complex_hits = sum(1 for kw in weights.complex_keywords if kw in lowered)
    if complex_hits:
        # Hits per 100 tokens, so density — not raw count — drives the
        # score; a handful of matches in a huge prompt shouldn't dominate.
        density = complex_hits / max(1, tokens) * 100
        score += min(weights.keyword_weight, 0.08 * density)
        reasons.append(f"{complex_hits} complexity keyword(s) matched")

    simple_hits = sum(1 for kw in weights.simple_keywords if kw in lowered)
    if simple_hits and complex_hits == 0:
        score -= weights.simple_penalty
        reasons.append(f"{simple_hits} simple-query keyword(s) matched")

    if _MULTISTEP_PATTERN.search(prompt):
        score += weights.multistep_weight
        reasons.append("multi-step instructions detected")

    if history_tokens > 0:
        history_component = min(1.0, history_tokens / 4000)
        if history_component > 0.1:
            score += weights.history_weight * history_component
            reasons.append(f"long conversation history (~{history_tokens} tokens)")

    score = max(0.0, min(1.0, score))
    return score, reasons


def _tier_for_score(score: float, config: RouterConfig) -> str:
    if score <= config.simple_max_score:
        return "simple"
    if score >= config.complex_min_score:
        return "complex"
    return "standard"


def _estimate(tier: str, config: RouterConfig, input_tokens: int, output_tokens: int) -> tuple:
    econ = config.economics.get(tier)
    if econ is None:
        return None, None
    input_cost = (input_tokens / 1_000_000) * econ.input_cost_per_mtok
    output_cost = (output_tokens / 1_000_000) * econ.output_cost_per_mtok
    return input_cost + output_cost, econ.avg_latency_ms


def _apply_budget_constraint(
    tier: str,
    config: RouterConfig,
    input_tokens: int,
    output_tokens: int,
    reasons: list[str],
) -> tuple:
    cost, latency = _estimate(tier, config, input_tokens, output_tokens)
    if config.max_cost_per_call is None or cost is None or cost <= config.max_cost_per_call:
        return tier, cost, latency, False

    order = config.tier_order
    start_idx = order.index(tier) if tier in order else len(order) - 1
    for idx in range(start_idx - 1, -1, -1):
        candidate = order[idx]
        c_cost, c_latency = _estimate(candidate, config, input_tokens, output_tokens)
        if c_cost is None or c_cost <= config.max_cost_per_call:
            reasons.append(f"downgraded to '{candidate}' to stay within max_cost_per_call=${config.max_cost_per_call}")
            return candidate, c_cost, c_latency, True

    cheapest = order[0]
    c_cost, c_latency = _estimate(cheapest, config, input_tokens, output_tokens)
    reasons.append(f"budget too low for any tier; using cheapest available ('{cheapest}')")
    return cheapest, c_cost, c_latency, True


def route(
    prompt: str,
    config: RouterConfig | None = None,
    *,
    history_tokens: int = 0,
    estimated_output_tokens: int = 500,
) -> RouteDecision:
    """Score ``prompt`` and pick a model tier from ``config``.

    ``history_tokens`` lets a caller factor in accumulated conversation
    size (see :func:`score_prompt`). ``estimated_output_tokens`` feeds the
    cost estimate — tune it if your typical responses run much shorter or
    longer than 500 tokens.
    """
    config = config or RouterConfig()
    score, reasons = score_prompt(prompt, history_tokens=history_tokens, weights=config.weights)
    tier = _tier_for_score(score, config)

    input_tokens = count_tokens(prompt) + history_tokens
    tier, cost, latency, downgraded = _apply_budget_constraint(
        tier, config, input_tokens, estimated_output_tokens, reasons
    )

    model = config.tiers.get(tier) or next(iter(config.tiers.values()))
    return RouteDecision(
        score=score,
        tier=tier,
        model=model,
        reasons=reasons,
        estimated_input_tokens=input_tokens,
        estimated_output_tokens=estimated_output_tokens,
        estimated_cost_usd=cost,
        estimated_latency_ms=latency,
        downgraded_for_budget=downgraded,
    )


def looks_like_refusal_or_shallow(response: str, *, min_words: int = 8) -> bool:
    """Cheap heuristic: does ``response`` look like a refusal, or a
    suspiciously short/low-effort answer that might warrant escalating to a
    stronger tier and retrying? Not a substitute for real evaluation, but
    useful as an automatic trigger in an agent loop."""
    if not response or not response.strip():
        return True
    if any(rx.search(response) for rx in _REFUSAL_REGEXES):
        return True
    return len(response.split()) < min_words


def escalate(decision: RouteDecision, config: RouterConfig | None = None) -> RouteDecision:
    """Return a new :class:`RouteDecision` one tier above ``decision``, for
    retrying after :func:`looks_like_refusal_or_shallow` flags a response.
    Returns ``decision`` unchanged if it's already at the top tier."""
    config = config or RouterConfig()
    order = config.tier_order
    if decision.tier not in order:
        return decision
    idx = order.index(decision.tier)
    if idx >= len(order) - 1:
        return decision

    new_tier = order[idx + 1]
    model = config.tiers.get(new_tier, decision.model)
    cost, latency = _estimate(new_tier, config, decision.estimated_input_tokens, decision.estimated_output_tokens)
    escalation_note = f"escalated from '{decision.tier}': response looked like a refusal or was too shallow"
    return RouteDecision(
        score=decision.score,
        tier=new_tier,
        model=model,
        reasons=[*decision.reasons, escalation_note],
        estimated_input_tokens=decision.estimated_input_tokens,
        estimated_output_tokens=decision.estimated_output_tokens,
        estimated_cost_usd=cost,
        estimated_latency_ms=latency,
        downgraded_for_budget=False,
    )


class RoutingLog:
    """Append-only JSONL audit log of routing decisions, for inspecting
    tier distribution and catching systematic misrouting over time. Stores
    a prompt hash, not the prompt text, unless ``store_prompt_text=True``.
    """

    def __init__(self, path, *, store_prompt_text: bool = False):
        self.path = Path(path)
        self.store_prompt_text = store_prompt_text

    def record(self, prompt: str, decision: RouteDecision) -> None:
        entry = {
            "timestamp": time.time(),
            "tier": decision.tier,
            "model": decision.model,
            "score": decision.score,
            "reasons": decision.reasons,
            "estimated_cost_usd": decision.estimated_cost_usd,
            "prompt_tokens": count_tokens(prompt),
        }
        if self.store_prompt_text:
            entry["prompt"] = prompt
        else:
            entry["prompt_hash"] = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def summary(self) -> dict:
        """Tier distribution and average score/cost across all logged
        decisions. Returns ``{"total": 0}`` if nothing's been logged yet."""
        if not self.path.exists():
            return {"total": 0}

        counts: dict = {}
        total_score = 0.0
        total_cost = 0.0
        cost_count = 0
        n = 0
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                n += 1
                counts[entry["tier"]] = counts.get(entry["tier"], 0) + 1
                total_score += entry.get("score", 0.0)
                if entry.get("estimated_cost_usd") is not None:
                    total_cost += entry["estimated_cost_usd"]
                    cost_count += 1

        return {
            "total": n,
            "tier_counts": counts,
            "avg_score": total_score / n if n else 0.0,
            "avg_cost_usd": total_cost / cost_count if cost_count else None,
        }
