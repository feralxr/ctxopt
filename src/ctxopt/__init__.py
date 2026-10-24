"""ctxopt: a lightweight toolkit for taming LLM context windows."""

from .conversation import Turn, parse_conversation
from .llm import anthropic_backend
from .router import (
    RouteDecision,
    RouterConfig,
    RoutingLog,
    ScoringWeights,
    TierEconomics,
    escalate,
    looks_like_refusal_or_shallow,
    route,
    score_prompt,
)
from .safety import UnsafePatternError
from .sandbox import SandboxResult, detect_profile, sandbox_text
from .summarize import SummaryResult, summarize_text
from .tokens import count_tokens

__version__ = "0.5.0"

__all__ = [
    "RouteDecision",
    "RouterConfig",
    "RoutingLog",
    "SandboxResult",
    "ScoringWeights",
    "SummaryResult",
    "TierEconomics",
    "Turn",
    "UnsafePatternError",
    "__version__",
    "anthropic_backend",
    "count_tokens",
    "detect_profile",
    "escalate",
    "looks_like_refusal_or_shallow",
    "parse_conversation",
    "route",
    "sandbox_text",
    "score_prompt",
    "summarize_text",
]
