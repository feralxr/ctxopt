# ctxopt

**A lightweight CLI for taming LLM context windows.**

`ctxopt` helps you keep conversations with LLMs — and the agents built on
them — lean. It does three things, each independently useful:

- 🧹 **Sandbox** noisy tool output (logs, stack traces, diffs, JSON blobs) down
  to a compact form that fits an actual token budget, instead of dumping
  thousands of lines into your context.
- 📝 **Summarize** long conversations or transcripts into a shorter digest,
  locally and deterministically — no API calls required.
- 🧭 **Route** a prompt to the right model tier based on an estimate of how
  complex it actually is, so you're not paying frontier-model prices for
  "what's the capital of France?"

No heavy dependencies, no required API keys, no telemetry. It's a small tool
that does a small number of things well.

## Install

```bash
pip install ctxopt
```

Or from source:

```bash
git clone https://github.com/feralxr/ctxopt.git
cd ctxopt
pip install -e .
```

Requires Python 3.9+.

## Usage

### Sandbox noisy output

This is the core command: it collapses long, repetitive, or oversized tool
output while preserving the signal that actually matters.

```bash
ctxopt sandbox build.log
# or pipe it in:
cat build.log | ctxopt sandbox
```

**Token-budget fitting** (recommended — tokens, not lines, are what fill a
context window):

```bash
ctxopt sandbox giant_stacktrace.log --max-tokens 300
```

```
  File "app.py", line 14, in recurse
    return recurse(n-1)
RecursionError: maximum recursion depth exceeded
profile=stacktrace  5346 -> 118 tokens (5228 saved, 981 lines elided, fit to budget)
```

**Near-duplicate collapsing.** Lines that are identical after stripping
volatile bits (timestamps, hex addresses, UUIDs, standalone counters) get
folded together — this is what actually tames recursive stack traces and
progress spam, where only a counter or timestamp changes line to line.
Digits embedded *inside* an identifier (like `line1` or `v2`) are left
alone, so distinct content never silently merges:

```bash
ctxopt sandbox big_diff.patch
```

```
diff --git a/big_module.py b/big_module.py
--- a/big_module.py
+++ b/big_module.py
@@ -1,300 +1,300 @@
+added line 0
... [299 near-identical line(s) collapsed] ...
profile=diff  1216 -> 28 tokens (1188 saved, 0 lines elided)
```

**Content profiles.** `ctxopt sandbox` auto-detects the shape of the input
(`--profile auto` by default) and adjusts behavior accordingly:

| Profile | Detected when | Behavior |
|---|---|---|
| `generic` | default fallback | Head/tail elision + near-dup collapsing |
| `json` | input parses as JSON | Structural truncation: long strings and large arrays are trimmed *at the value level*, so the output stays valid JSON |
| `stacktrace` | 3+ frame-like lines (`File "...", line N` / `at foo(...)`) | Tighter head window, `Traceback`/`Caused by` lines always preserved |
| `diff` | 2+ diff markers (`+++`, `---`, `@@`, `diff --git`) | File headers and hunk markers always preserved even when the body is elided |

JSON example — a 300-user payload with 800-character bios collapses to a
fraction of its size while staying parseable:

```bash
ctxopt sandbox users.json
```

```json
{
  "users": [
    { "id": 0, "name": "user_0", "bio": "xxxx...xx... [300 chars truncated]" },
    ...
  ]
}
```
```
profile=json  2401 -> 306 tokens (2095 saved, 0 lines elided)
```

**Signal preservation.** Lines matching `error`, `exception`, `fail(ed/ure)`,
`traceback`, `fatal`, or `panic` are never elided, even inside the collapsed
middle — up to a cap (`max_kept_matches` as a library param) so a
pathological log full of "error" lines can't defeat the budget. Add your own
patterns with `--keep`:

```bash
ctxopt sandbox app.log --keep "payment_id=\w+" --max-tokens 500
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--max-tokens` | — | Shrink output to fit this token budget (shrinks head/tail/keep windows until it fits) |
| `--max-lines` | 200 | Collapse once output exceeds this many lines (used when `--max-tokens` isn't set) |
| `--head` / `--tail` | profile-specific | Lines to keep from the start/end |
| `--profile` | `auto` | `auto`, `generic`, `json`, `stacktrace`, or `diff` |
| `--keep` | — | Extra regex to always preserve (repeatable) |
| `--out` | stdout | Write result to a file instead |

### Summarize a conversation

This is conversation-*aware*, not just a document shortener. It detects two
transcript formats — `Role: message` plain text, and JSON message lists
(`[{"role": ..., "content": ...}]`, the shape most chat APIs use) — and
summarizes turn by turn. Anything else falls back to paragraph-based
segmentation.

```bash
ctxopt summarize support_thread.txt --max-tokens 120
```

A 47-turn transcript — a real bug report, 39 repetitive "any update?"
check-ins, and a resolution — collapses to the parts that matter:

```
User: Hey, the deploy script is failing on staging.

Assistant: Can you share the error output?

User: Sure, here it is: ValueError: missing config key DATABASE_URL

Assistant: Looks like the staging .env file is missing that key. I will check.

Assistant: still looking into it, item 10 is in progress.

Assistant: Found it, the key was renamed in the last config migration. We
decided to add a fallback default for it.

User: Perfect, that fixed it. Thanks!

Assistant: Glad it works now. Let me know if anything else comes up.

backend=local  640 -> 120 tokens  turns 8/47
```

The problem report, the diagnosis, the decision, and the resolution all
survived; the 39 near-identical "checking in" turns didn't.

**How segments are chosen**, same philosophy as `sandbox`'s keep patterns:

- Lines matching `error`, `exception`, `decided`, or `important` are always
  kept, plus any `--keep <regex>` you add, however small the budget.
- The most recent `--keep-last` turns (default 2) are always kept — only
  meaningful for conversations, ignored for plain documents.
- Everything else is scored by length, keyword density, and position, then
  greedily added until the budget is met — skipping any candidate that's a
  near-duplicate (word-overlap above `redundancy_threshold`) of something
  already kept, so repetitive check-ins don't all survive just because each
  one scores decently on its own.

This local mode is fully offline and deterministic — no network calls, no
cost, same input always gives the same output.

**Abstractive summaries (optional).** Pass `--llm` to use a real model
instead of the local heuristic. It reads `ANTHROPIC_API_KEY` from your
environment and does a small map-reduce: segments are grouped into
token-sized chunks, each chunk gets summarized by the model, and — only if
the combined chunk summaries still exceed the target — one final call
reduces them into a single digest.

```bash
export ANTHROPIC_API_KEY=sk-...
ctxopt summarize huge_transcript.txt --llm --max-tokens 300
```

As a library, `llm_call` is just `Callable[[str], str]` — point it at
anything, not just Anthropic:

```python
from ctxopt import summarize_text, anthropic_backend

result = summarize_text(transcript, llm_call=anthropic_backend(), max_tokens=300)
print(result.backend, result.text)  # "llm", "..."
```

### Route a prompt to a model tier

Score a prompt's complexity and get a model recommendation, with cost and
latency estimates:

```bash
ctxopt route "What is the capital of France?"
```

```
Score         0.00
Tier          simple
Model         claude-haiku-5-5
Est. cost     $0.00201
Est. latency  400 ms
Reasons       2 simple-query keyword(s) matched
```

```bash
ctxopt route "Design a fault-tolerant distributed architecture for this payment system and debug the race condition below: ..."
```

```
Score         0.76
Tier          complex
Model         claude-opus-5-5
Est. cost     $0.03789
Est. latency  3000 ms
Reasons       contains code or code-like syntax; 4 complexity keyword(s) matched
```

**Keyword density, not raw count.** Complexity keyword matches are scored
per 100 tokens, not as a flat count — so a short, keyword-dense request
scores clearly complex, while the same keywords diluted across a long,
mostly-filler prompt don't:

```python
from ctxopt import score_prompt

score_prompt("Debug this algorithm and analyze the security architecture trade-offs.")
# -> (0.55, ['6 complexity keyword(s) matched'])

score_prompt("Debug this algorithm and analyze the security architecture "
             "trade-offs. " + "filler content that adds length. " * 60)
# -> (0.27, ['long input (~412 tokens)', '6 complexity keyword(s) matched'])
# same keywords, far lower density -> lower score despite the length bonus
```

**Conversation history matters too.** A short message late in a long
session often still needs a capable model to stay coherent — pass
`--history-tokens` (or `history_tokens=` as a library) to factor that in:

```bash
ctxopt route "Can you also check that last part?" --history-tokens 6000
```

**Cost-aware routing.** Every tier can carry cost/latency economics; set a
budget and the router downgrades to a tier that fits it, noting why:

```bash
ctxopt route "Design a fault-tolerant distributed payment architecture..." --max-cost 0.0005
```

```
Reasons  9 complexity keyword(s) matched; budget too low for any tier;
         using cheapest available ('simple')
```

**Escalation.** A one-shot router can't know if its pick actually worked —
`looks_like_refusal_or_shallow` is a cheap heuristic for "this response
dodged the question or phoned it in", and `escalate` bumps a decision up a
tier so you can retry:

```python
from ctxopt import route, escalate, looks_like_refusal_or_shallow

decision = route(prompt)
response = call_model(decision.model, prompt)  # your own call
if looks_like_refusal_or_shallow(response):
    decision = escalate(decision)
    response = call_model(decision.model, prompt)
```

**Audit trail.** `RoutingLog` appends routing decisions to a JSONL file
(prompt hash by default, not the prompt text) and summarizes tier
distribution and average cost over time:

```python
from ctxopt import RoutingLog

log = RoutingLog("routing.jsonl")
log.record(prompt, decision)
log.summary()  # {"total": 142, "tier_counts": {...}, "avg_cost_usd": ...}
```

Tiers, model names, score thresholds, keyword lists, and per-tier economics
are all configurable via `RouterConfig` — see below.

## Using it as a library

Every command is backed by a plain Python function with no CLI coupling:

```python
from ctxopt import sandbox_text, summarize_text, route, RouterConfig

# Collapse noisy output to fit an actual token budget
result = sandbox_text(log_text, max_tokens=500)
print(result.profile, result.text, result.original_tokens, result.final_tokens)

# Always preserve your own markers (e.g. a request/trace id), on top of the
# built-in error/exception/fail defaults
result = sandbox_text(log_text, max_tokens=500, keep_patterns=[r"trace_id=\w+"])

# Summarize a transcript (conversation-aware, local and free)
digest = summarize_text(transcript_text, max_tokens=300, keep_last_n_turns=3)
print(digest.turns_kept, "of", digest.turns_total, "turns kept")

# Route to a model tier, with your own model names and a cost budget
config = RouterConfig(
    tiers={"simple": "gpt-4o-mini", "standard": "gpt-4o", "complex": "o1"},
    max_cost_per_call=0.05,
)
decision = route(user_prompt, config, history_tokens=conversation_size)
print(decision.tier, decision.model, decision.estimated_cost_usd, decision.reasons)
```

## Why this exists

Agent and chat pipelines accumulate context fast: verbose tool output,
sprawling multi-turn history, and routing every request — simple or
complex — through the same expensive model. `ctxopt` is a small set of
composable primitives for the three most common sources of that bloat. It's
not a framework and doesn't ask you to adopt an architecture — pipe things
in, get leaner things out.

## Development

```bash
git clone https://github.com/feralxr/ctxopt.git
cd ctxopt
pip install -e ".[dev]"
pytest
```

Every push/PR to `main` runs the test suite across Python 3.9–3.12 via
[`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Publishing

Releases are published to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no PyPI API token is ever stored as a secret. One-time setup on
PyPI's side, under **Your project → Publishing → Add a new pending publisher**:

| Field | Value |
|---|---|
| PyPI Project Name | `ctxopt` |
| Owner | `feralxr` |
| Repository name | `ctxopt` |
| Workflow name | `workflow.yml` |
| Environment name | `pypi` |

After that's linked, cutting a release is just:

```bash
# bump the version in pyproject.toml and src/ctxopt/__init__.py first
git tag v0.2.0
git push origin v0.2.0
```

[`.github/workflows/workflow.yml`](.github/workflows/workflow.yml) builds the
sdist/wheel, checks the tag matches `pyproject.toml`'s version, and publishes.

## License

MIT — see [LICENSE](LICENSE).
