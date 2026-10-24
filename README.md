# ctxopt

**A lightweight CLI and Python library for taming LLM context windows.**

`ctxopt` helps you keep conversations with LLMs — and the agents built on
them — lean. It does three things, each independently useful and each
usable on its own without adopting the other two:

- **Sandbox** noisy tool output (logs, stack traces, diffs, JSON blobs) down
  to a compact form that fits an actual token budget, instead of dumping
  thousands of lines into your context.
- **Summarize** long conversations or transcripts into a shorter digest,
  locally and deterministically by default, with an optional path to real
  LLM-backed abstractive summarization.
- **Route** a prompt to the right model tier based on an estimate of how
  complex it actually is — including cost and latency estimates, a budget
  constraint, and an escalation path for when the cheap model's answer
  wasn't good enough.

No heavy dependencies, no required API keys, no telemetry. It's a small
tool that does a small number of things well, not a framework you adopt.

## Table of contents

- [Why this exists](#why-this-exists)
- [Install](#install)
- [Usage: sandbox](#sandbox-noisy-output)
- [Usage: summarize](#summarize-a-conversation)
- [Usage: route](#route-a-prompt-to-a-model-tier)
- [Using it as a library](#using-it-as-a-library)
- [Design principles](#design-principles)
- [Known limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

## Why this exists

Agent and chat pipelines accumulate context fast, and it tends to come from
the same three places every time:

1. **Tool output.** A build log, a stack trace, a database query result, a
   file listing — these get pasted into the context wholesale, and most of
   it is noise. A 900-line log might carry 20 lines of actual signal.
2. **Conversation history.** Long-running sessions or support threads
   accumulate turns that mostly restate or check in on something already
   said, but a naive "keep the last N messages" strategy will just as
   happily cut the one message that mattered as the fifteen that didn't.
3. **Using one model for everything.** Routing every request — "what's 2+2"
   and "design a distributed consensus protocol" alike — through the same
   expensive, slow model wastes money and latency on the easy 80%.

`ctxopt` is a small set of composable primitives for those three problems
specifically. It doesn't ask you to restructure your pipeline around it —
each function takes text (or a prompt) in and gives you something leaner
back out, and you decide where that fits.

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

Requires Python 3.9 or later. The only required third-party dependencies
are Typer and Rich (for the CLI) and tiktoken (for accurate token counts —
the library still works without it, falling back to a word-count heuristic,
if installing it isn't an option in your environment).

## Sandbox noisy output

This is the core command: it collapses long, repetitive, or oversized tool
output while preserving the signal that actually matters, rather than
blindly truncating.

```bash
ctxopt sandbox build.log
# or pipe it in:
cat build.log | ctxopt sandbox
```

### Token-budget fitting

Tokens, not lines, are what actually fill a context window, so this is the
recommended way to call it: give it a budget, and it shrinks its internal
head/tail/keep windows in a bounded loop until the result actually fits,
rather than trusting a single line-count guess.

```bash
ctxopt sandbox giant_stacktrace.log --max-tokens 300
```

```
  File "app.py", line 14, in recurse
    return recurse(n-1)
RecursionError: maximum recursion depth exceeded
profile=stacktrace  5346 -> 118 tokens (5228 saved, 981 lines elided, fit to budget)
```

### Near-duplicate collapsing

Lines that are identical after stripping volatile substrings — timestamps,
hex addresses, UUIDs, standalone numeric counters — get folded into one
representative line plus a repeat count. This is what actually tames
recursive stack traces and progress spam, where only a counter or
timestamp differs line to line. Digits embedded *inside* an identifier
(like `line1` or `v2`) are deliberately left alone, so distinct content
never silently merges just because it contains a number:

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

### Content profiles

`ctxopt sandbox` auto-detects the shape of the input (`--profile auto` by
default, or force one explicitly) and adjusts its behavior:

| Profile | Detected when | Behavior |
|---|---|---|
| `generic` | default fallback | Head/tail elision plus near-duplicate collapsing |
| `json` | input parses as JSON | Structural truncation: long strings and large arrays are trimmed *at the value level*, so the output stays valid, parseable JSON instead of being treated as flat lines |
| `stacktrace` | 3 or more frame-like lines (`File "...", line N` / `at foo(...)`) | Tighter head window, `Traceback`/`Caused by` lines always preserved |
| `diff` | 2 or more diff markers (`+++`, `---`, `@@`, `diff --git`) | File headers and hunk markers always preserved even when the hunk body is elided |

JSON example — a 300-user payload with 800-character bios collapses to a
fraction of its size while staying parseable:

```bash
ctxopt sandbox users.json
```

```json
{
  "users": [
    { "id": 0, "name": "user_0", "bio": "xxxx...xx... [300 chars truncated]" }
  ]
}
```
```
profile=json  2401 -> 306 tokens (2095 saved, 0 lines elided)
```

### Signal preservation

Lines matching `error`, `exception`, `fail(ed/ure)`, `traceback`, `fatal`,
or `panic` are never elided, even inside the collapsed middle — up to a cap
(`max_kept_matches`, a library parameter) so a pathological log full of
"error" lines can't single-handedly defeat the budget. Add your own
patterns with `--keep`:

```bash
ctxopt sandbox app.log --keep "payment_id=\w+" --max-tokens 500
```

### Guardrails on user-supplied patterns

`--keep` patterns (here and in `summarize`) run through a ReDoS
(regex-denial-of-service) guard before they're used: a pattern like
`(a+)+$` matched against a crafted input can take catastrophically long in
Python's regex engine. The guard rejects the common nested-quantifier shape
outright, and separately smoke-tests every pattern against a short
adversarial probe string with a hard timeout before it's ever run on real
data — both checks apply only to patterns you supply; ctxopt's own built-in
defaults are trusted and skip them.

```bash
ctxopt sandbox app.log --keep '(a|a)*b'
```

```
Pattern '(a|a)*b' took too long matching a short test string and was rejected as
a likely catastrophic-backtracking risk.
```

This is a heuristic, not a proof — see Known limitations below.

### Flags

| Flag | Default | Description |
|---|---|---|
| `--max-tokens` | — | Shrink output to fit this token budget (recommended; shrinks head/tail/keep windows until it fits) |
| `--max-lines` | 200 | Collapse once output exceeds this many lines (used only when `--max-tokens` isn't set) |
| `--head` / `--tail` | profile-specific | Lines to keep from the start/end |
| `--profile` | `auto` | `auto`, `generic`, `json`, `stacktrace`, or `diff` |
| `--keep` | — | Extra regex pattern to always preserve (repeatable) |
| `--out` | stdout | Write result to a file instead |
| `--max-input-bytes` | 20 MB | Read at most this many bytes from a huge file/stdin (head+tail) instead of loading it all — see Known limitations |

## Summarize a conversation

This is conversation-*aware*, not just a document shortener. It detects two
transcript formats automatically:

- Plain `Role: message` speaker-prefixed text (`User: ...` / `Assistant: ...`)
- JSON message lists (`[{"role": ..., "content": ...}]`), the shape most
  chat/completions APIs use, including content-block arrays

Anything that doesn't match either format falls back to paragraph-based
segmentation, so it still works as a general document shortener.

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

### How segments are chosen

Same philosophy as `sandbox`'s keep patterns — preserve signal first, fill
the remaining budget by score, and don't let near-duplicates crowd out
distinct content:

1. Lines matching `error`, `exception`, `decided`, or `important` are
   always kept, plus any `--keep <regex>` you add — however small the
   budget, these never get cut.
2. The most recent `--keep-last` turns (default 2) are always kept — only
   meaningful for detected conversations; ignored for plain documents,
   where "most recent" isn't a well-defined idea.
3. Everything else is scored by length, keyword density, and position
   (first/last get a small bonus), then greedily added in score order
   until the budget is met — skipping any candidate that's a near-duplicate
   (word-overlap above `redundancy_threshold`, default 0.85) of something
   already kept. This is why the 39 "checking in" turns above collapse to
   effectively one: they'd otherwise each score decently on their own.
4. The final selection is re-sorted back into chronological order, so the
   digest still reads as a coherent narrative rather than a shuffled
   highlight reel.

This local mode is fully offline and deterministic: no network calls, no
cost, and the same input always produces the same output — useful when you
need reproducible behavior in tests or CI.

### Abstractive summaries (optional)

Pass `--llm` to use a real model instead of the local heuristic. It reads
`ANTHROPIC_API_KEY` from your environment and runs a small map-reduce:
segments are grouped into token-sized chunks, each chunk gets summarized by
the model independently, and — only if the combined chunk summaries still
exceed the target size — one additional call reduces them into a single
final digest. For small inputs that fit in one chunk, that's just a single
API call.

```bash
export ANTHROPIC_API_KEY=sk-...
ctxopt summarize huge_transcript.txt --llm --max-tokens 300
```

As a library, `llm_call` is just `Callable[[str], str]` — point it at
anything, not only Anthropic's API:

```python
from ctxopt import summarize_text, anthropic_backend

result = summarize_text(transcript, llm_call=anthropic_backend(), max_tokens=300)
print(result.backend, result.text)  # "llm", "..."
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--max-tokens` | — | Target size in tokens directly (overrides `--ratio`) |
| `--ratio` | 0.25 | Target size as a fraction of the original, used when `--max-tokens` isn't set |
| `--keep` | — | Extra regex pattern to always preserve (repeatable) |
| `--keep-last` | 2 | Always keep this many of the most recent turns (conversations only) |
| `--llm` | off | Use an LLM for abstractive summarization instead of the local heuristic |
| `--model` | `claude-haiku-5-5` | Model to use with `--llm` |
| `--out` | stdout | Write result to a file instead |
| `--max-input-bytes` | 20 MB | Read at most this many bytes from a huge file/stdin (head+tail) instead of loading it all |

## Route a prompt to a model tier

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

### Keyword density, not raw count

Complexity keyword matches are scored per 100 tokens, not as a flat count —
a short, keyword-dense request scores clearly complex, while the same
keywords diluted across a long, mostly-filler prompt don't dominate the
score the way a naive count would let them:

```python
from ctxopt import score_prompt

score_prompt("Debug this algorithm and analyze the security architecture trade-offs.")
# -> (0.55, ['6 complexity keyword(s) matched'])

score_prompt("Debug this algorithm and analyze the security architecture "
             "trade-offs. " + "filler content that adds length. " * 60)
# -> (0.27, ['long input (~412 tokens)', '6 complexity keyword(s) matched'])
# same keywords, far lower density -> lower score despite the length bonus
```

### Conversation history matters too

A short message late in a long session often still needs a capable model to
stay coherent with everything that came before it. Pass `--history-tokens`
(or `history_tokens=` as a library) to factor accumulated context size into
the score, on top of the current message alone:

```bash
ctxopt route "Can you also check that last part?" --history-tokens 6000
```

### Cost-aware routing

Every tier can carry cost and latency economics (`RouterConfig.economics`).
Set a budget and the router downgrades to the cheapest tier that still fits
it — or, if nothing fits, falls back to the cheapest available tier and
says so explicitly rather than silently picking something over budget:

```bash
ctxopt route "Design a fault-tolerant distributed payment architecture..." --max-cost 0.0005
```

```
Reasons  9 complexity keyword(s) matched; budget too low for any tier;
         using cheapest available ('simple')
```

### Escalation

A one-shot router can't know in advance whether its pick actually produced
a good answer. `looks_like_refusal_or_shallow` is a cheap heuristic for "did
this response dodge the question or phone it in" (refusal-phrase matching,
plus a minimum word-count floor), and `escalate` bumps a decision up one
tier so you can retry against a stronger model:

```python
from ctxopt import route, escalate, looks_like_refusal_or_shallow

decision = route(prompt)
response = call_model(decision.model, prompt)  # your own call
if looks_like_refusal_or_shallow(response):
    decision = escalate(decision)
    response = call_model(decision.model, prompt)
```

`escalate` is a no-op at the top tier, so this loop is always safe to run
without an extra bounds check.

### Audit trail

`RoutingLog` appends routing decisions to an append-only JSONL file — a
prompt hash by default, not the prompt text itself, unless you explicitly
opt into storing it — and can summarize tier distribution and average cost
across everything it's logged:

```python
from ctxopt import RoutingLog

log = RoutingLog("routing.jsonl")
log.record(prompt, decision)
log.summary()  # {"total": 142, "tier_counts": {...}, "avg_cost_usd": ...}
```

### Flags

| Flag | Default | Description |
|---|---|---|
| `--history-tokens` | 0 | Accumulated conversation size so far, if any |
| `--output-tokens` | 500 | Estimated response length, used for the cost estimate |
| `--max-cost` | — | Budget in USD; downgrades the tier if the initial pick exceeds it |

Tiers, model names, score thresholds, keyword lists, and per-tier economics
are all configurable via `RouterConfig` — see below.

## Using it as a library

Every CLI command is backed by a plain Python function with no CLI
coupling, so you can call any of them directly:

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

See each module's usage section above for the full parameter list, or read
the docstrings directly — `sandbox_text`, `summarize_text`, and `route` are
each documented in full at the top of `src/ctxopt/sandbox.py`,
`summarize.py`, and `router.py` respectively.

## Design principles

A few choices that apply across all three modules, on purpose:

- **Budget in tokens, not lines or a ratio, whenever possible.** Tokens are
  what actually fill a context window. `max_tokens` is the recommended way
  to call `sandbox` and `summarize`; line counts and ratios exist as a
  simpler fallback, not the primary interface.
- **Preserve signal before optimizing for size.** Every module has some
  notion of "never drop this" (`keep_patterns` in sandbox and summarize,
  `keep_last_n_turns` in summarize) that's checked before the general
  scoring/elision logic runs, not after. Hitting the budget exactly matters
  less than not silently discarding the one line that actually mattered.
- **Transparent heuristics over black boxes.** Every scoring decision comes
  with a plain-English `reasons` list. None of the three modules uses a
  trained model to make its core decision — "what to keep, what to cut" is
  always something you can read and reason about, not something you have
  to trust.
- **Local and free by default; networked and paid is opt-in.** Nothing in
  `ctxopt` makes a network call unless you explicitly ask it to
  (`summarize(..., llm_call=...)`). Pure local functions are easy to test,
  easy to run in CI, and have no surprise cost.
- **Zero required dependencies beyond the CLI itself.** The optional LLM
  backend (`ctxopt.llm.anthropic_backend`) is built on `urllib` from the
  standard library specifically so that using it doesn't pull in an extra
  package, and *not* using it costs nothing at all.
- **Untrusted input gets guardrails, not just features.** User-supplied
  regex (`--keep` in `sandbox` and `summarize`) is screened for
  catastrophic-backtracking risk before it's run, and both CLI commands
  read huge files/stdin through a bounded head+tail path rather than
  loading everything into memory unconditionally. These exist because a
  CLI that works great on curated examples but falls over or hangs on
  adversarial or merely huge real-world input isn't actually done.

## Known limitations

Documented honestly rather than left for you to discover:

- **Near-duplicate collapsing is consecutive-only.** It folds together runs
  of adjacent near-identical lines, but doesn't detect a *cyclic* pattern
  (e.g. frames `A, B, C, A, B, C, ...` repeating with no two adjacent lines
  identical). Such input still gets handled by the head/tail elision step,
  just not by the dedup step specifically.
- **Conversation detection is heuristic, not guaranteed.** Speaker-prefixed
  detection requires at least two speaker-labeled lines from at least two
  distinct speakers; an ordinary document that happens to contain a couple
  of `Label: text` lines from different labels could misfire as a
  conversation. In practice this is rare, but it's a real false-positive
  surface, not a theoretical one.
- **The router's scoring is a heuristic, not a classifier.** It's
  deliberately transparent and auditable (see Design principles above)
  rather than trained on labeled data, which means it can be wrong in ways
  a learned model wouldn't be — it's a reasonable default, not a
  guarantee of correct routing for every prompt.
- **Cost/latency estimates are only as accurate as the economics you
  configure.** The default `RouterConfig.economics` values are illustrative
  placeholders, not live pricing — set them to your actual provider's
  current rates for `estimated_cost_usd` to mean anything precise.
- **JSON profile truncation is lossy by design.** Once a long string or
  large array has been truncated, the original content is gone from the
  output — this is meant for sandboxing output you're about to discard
  anyway (feeding a model, logging), not as a reversible compression
  format.
- **The ReDoS guard is a heuristic, not a proof of safety.** The
  nested-quantifier check and the smoke-test timeout catch the common
  catastrophic-backtracking shapes, but a sufficiently creative pattern
  could in principle pass both and still be slow on some input the smoke
  test didn't happen to probe. It meaningfully raises the bar; it doesn't
  make arbitrary user-supplied regex provably safe.
- **Bounded large-file reads see a head and a tail, not the whole file.**
  When a file exceeds `--max-input-bytes` (default 20 MB), `sandbox` and
  `summarize` read only a head and tail slice of it — a keep-pattern match
  or an error that happens to fall entirely in the untouched middle of a
  huge file won't be seen. This is a deliberate trade-off (bounded memory
  and near-instant reads via seeking, instead of loading the whole file)
  that matters once a file is far larger than any reasonable token budget
  could use anyway; raise `--max-input-bytes` if you need to see more of a
  specific large file.

## Contributing

Issues and pull requests are welcome. For development setup, the test
suite, CI, and the release process, see [MAINTAINING.md](MAINTAINING.md) —
that file covers maintainer-facing operational detail that doesn't belong
in this README.

## License

MIT — see [LICENSE](LICENSE).
