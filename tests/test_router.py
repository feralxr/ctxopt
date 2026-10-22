from ctxopt.router import RouterConfig, route, score_prompt


def test_simple_prompt_scores_low():
    score, _ = score_prompt("What is the capital of France?")
    assert score < 0.4


def test_complex_prompt_scores_high():
    prompt = (
        "Analyze the trade-offs of this distributed systems architecture, "
        "debug the race condition below, and refactor for better concurrency:\n"
        "```python\ndef handler():\n    pass\n```"
    )
    score, reasons = score_prompt(prompt)
    assert score > 0.5
    assert reasons


def test_route_picks_simple_tier_for_simple_prompt():
    decision = route("Define photosynthesis in one sentence.")
    assert decision.tier == "simple"
    assert decision.model == RouterConfig().tiers["simple"]


def test_route_picks_complex_tier_for_complex_prompt():
    prompt = (
        "Design a fault-tolerant, distributed architecture for a payment system, "
        "analyze security vulnerabilities, and optimize for concurrency. "
        "Also debug this algorithm and compare trade-offs."
    )
    decision = route(prompt)
    assert decision.tier == "complex"


def test_custom_router_config_overrides_models():
    config = RouterConfig(tiers={"simple": "my-small-model", "standard": "my-mid-model", "complex": "my-big-model"})
    decision = route("What is 2 + 2?", config)
    assert decision.model == "my-small-model"
