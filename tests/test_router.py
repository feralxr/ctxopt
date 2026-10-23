import json

from ctxopt.router import (
    RouterConfig,
    RoutingLog,
    TierEconomics,
    escalate,
    looks_like_refusal_or_shallow,
    route,
    score_prompt,
)


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


# ---- keyword density normalization -------------------------------------------


def test_keyword_density_distinguishes_short_dense_from_long_diluted():
    short_dense = "Debug this algorithm and analyze the security architecture trade-offs."
    padding = " ".join(["filler content that adds length without adding complexity"] * 60)
    long_diluted = short_dense + " " + padding

    dense_score, _ = score_prompt(short_dense)
    diluted_score, _ = score_prompt(long_diluted)

    # Same keyword hits, but far lower density in the diluted version.
    assert dense_score > diluted_score


# ---- conversation history awareness -------------------------------------------


def test_history_tokens_increase_score_for_simple_looking_prompt():
    prompt = "Can you also check that last part?"
    low, _ = score_prompt(prompt, history_tokens=0)
    high, reasons = score_prompt(prompt, history_tokens=6000)
    assert high > low
    assert any("history" in r for r in reasons)


def test_route_accepts_history_tokens():
    decision = route("What's next?", history_tokens=5000)
    assert decision.estimated_input_tokens > 5000


# ---- cost/latency estimation and budget constraint ----------------------------


def test_route_reports_cost_and_latency_estimates():
    decision = route("Explain recursion simply.")
    assert decision.estimated_cost_usd is not None
    assert decision.estimated_cost_usd >= 0
    assert decision.estimated_latency_ms is not None


def test_max_cost_per_call_downgrades_tier():
    # A complex-scoring prompt, but with a budget too small for the complex tier.
    prompt = (
        "Design a fault-tolerant, distributed architecture for a payment system, "
        "analyze security vulnerabilities, and optimize for concurrency. "
        "Also debug this algorithm and compare trade-offs."
    )
    config = RouterConfig(max_cost_per_call=0.0005)
    decision = route(prompt, config)
    assert decision.tier != "complex"
    assert decision.downgraded_for_budget is True
    assert any("downgraded" in r or "budget" in r for r in decision.reasons)


def test_impossibly_low_budget_falls_back_to_cheapest_tier():
    config = RouterConfig(max_cost_per_call=0.0000001)
    decision = route("Write a full microservices platform from scratch.", config)
    assert decision.tier == config.tier_order[0]
    assert decision.downgraded_for_budget is True


def test_no_economics_means_no_cost_estimate():
    config = RouterConfig(economics={})
    decision = route("Hello there.", config)
    assert decision.estimated_cost_usd is None
    assert decision.estimated_latency_ms is None


# ---- escalation -----------------------------------------------------------------


def test_looks_like_refusal_detects_refusal_phrases():
    assert looks_like_refusal_or_shallow("I'm sorry, but I can't help with that request.")
    assert looks_like_refusal_or_shallow("I cannot assist with this.")


def test_looks_like_refusal_detects_short_answers():
    assert looks_like_refusal_or_shallow("Not sure.")
    assert looks_like_refusal_or_shallow("")


def test_looks_like_refusal_false_for_substantive_answer():
    answer = "Recursion is when a function calls itself, breaking a problem into smaller subproblems until a base case is reached."
    assert not looks_like_refusal_or_shallow(answer)


def test_escalate_moves_up_one_tier():
    decision = route("What is 2 + 2?")  # starts at "simple"
    assert decision.tier == "simple"
    escalated = escalate(decision)
    assert escalated.tier == "standard"
    assert "escalated" in escalated.reasons[-1]


def test_escalate_at_top_tier_is_a_no_op():
    config = RouterConfig()
    decision = route("Design a distributed fault-tolerant payment architecture.", config)
    # Escalate repeatedly; should never go past the top tier.
    for _ in range(5):
        decision = escalate(decision, config)
    assert decision.tier == config.tier_order[-1]


# ---- routing log ------------------------------------------------------------------


def test_routing_log_records_and_summarizes(tmp_path):
    log = RoutingLog(tmp_path / "routing.jsonl")
    decision1 = route("What is the capital of Japan?")
    decision2 = route("Design a distributed fault-tolerant payment architecture.")
    log.record("What is the capital of Japan?", decision1)
    log.record("Design a distributed fault-tolerant payment architecture.", decision2)

    summary = log.summary()
    assert summary["total"] == 2
    assert sum(summary["tier_counts"].values()) == 2


def test_routing_log_does_not_store_prompt_text_by_default(tmp_path):
    path = tmp_path / "routing.jsonl"
    log = RoutingLog(path)
    decision = route("a very secret prompt")
    log.record("a very secret prompt", decision)

    raw = path.read_text()
    assert "a very secret prompt" not in raw
    entry = json.loads(raw.strip())
    assert "prompt_hash" in entry


def test_routing_log_can_store_prompt_text_when_opted_in(tmp_path):
    path = tmp_path / "routing.jsonl"
    log = RoutingLog(path, store_prompt_text=True)
    decision = route("store me please")
    log.record("store me please", decision)

    entry = json.loads(path.read_text().strip())
    assert entry["prompt"] == "store me please"


def test_routing_log_summary_empty_when_no_file(tmp_path):
    log = RoutingLog(tmp_path / "does_not_exist.jsonl")
    assert log.summary() == {"total": 0}
