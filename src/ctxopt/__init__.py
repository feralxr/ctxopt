"""ctxopt: a lightweight toolkit for taming LLM context windows."""

from .router import RouteDecision, RouterConfig, route, score_prompt
from .sandbox import SandboxResult, detect_profile, sandbox_text
from .summarize import SummaryResult, summarize_text
from .tokens import count_tokens

__version__ = "0.2.0"

__all__ = [
    "count_tokens",
    "sandbox_text",
    "SandboxResult",
    "detect_profile",
    "summarize_text",
    "SummaryResult",
    "route",
    "score_prompt",
    "RouteDecision",
    "RouterConfig",
    "__version__",
]
