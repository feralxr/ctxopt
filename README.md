# ctxopt

**A lightweight CLI for taming LLM context windows.**

`ctxopt` helps you keep conversations with LLMs — and the agents built on
them — lean. It does three things, each independently useful:

- 🧹 **Sandbox** noisy tool output (logs, stack traces, command results) down
  to a compact form instead of dumping thousands of lines into your context.
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

Collapse long or repetitive tool/log output, keeping the head and tail
intact and eliding the noisy middle:

```bash
ctxopt sandbox build.log
# or pipe it in:
cat build.log | ctxopt sandbox
```

```
$ npm install
added package-0
added package-1
added package-2
added package-3
... [295 lines elided] ...
added package-299
added 842 packages in 12s
npm WARN deprecated some-package@1.0.0
... [line repeated 598 more times] ...
Build succeeded in 4.2s

4016 -> 50 tokens (3966 saved, 295 lines elided)
```

Flags:

| Flag | Default | Description |
|---|---|---|
| `--max-lines` | 200 | Collapse once output exceeds this many lines |
| `--head` | 40 | Lines to keep from the start |
| `--tail` | 40 | Lines to keep from the end |
| `--out` | stdout | Write result to a file instead |

### Summarize a conversation

Condense a long transcript or log into a shorter digest using a fast,
transparent extractive heuristic (paragraph length, keyword density, and
position all factor into what gets kept):

```bash
ctxopt summarize transcript.txt --ratio 0.2
```

```
638 -> 126 tokens
```

This is intentionally local and deterministic rather than LLM-backed, so it
costs nothing and behaves predictably. If you want an abstractive summary,
pipe the condensed output into your own LLM call — you'll be sending it far
fewer tokens than the original.

### Route a prompt to a model tier

Score a prompt's complexity and get a model recommendation back:

```bash
ctxopt route "What is the capital of France?"
```

```
Score    0.00
Tier     simple
Model    claude-haiku-5-5
Reasons  2 simple-query keyword(s) matched
```

```bash
ctxopt route "Design a fault-tolerant distributed architecture for this payment system and debug the race condition below: ..."
```

```
Score    0.74
Tier     complex
Model    claude-opus-5-5
Reasons  contains code or code-like syntax; 4 complexity keyword(s) matched
```

The scoring heuristic weighs input length, presence of code, and
complexity/simplicity keyword signals. Tiers and model names are fully
configurable — see below.

## Using it as a library

Every command is backed by a plain Python function with no CLI coupling:

```python
from ctxopt import sandbox_text, summarize_text, route, RouterConfig

# Collapse noisy output
result = sandbox_text(log_text, max_lines=200)
print(result.text, result.original_tokens, result.final_tokens)

# Summarize a transcript
digest = summarize_text(transcript_text, target_ratio=0.25)

# Route to a model tier, with your own model names
config = RouterConfig(tiers={
    "simple": "gpt-4o-mini",
    "standard": "gpt-4o",
    "complex": "o1",
})
decision = route(user_prompt, config)
print(decision.tier, decision.model, decision.reasons)
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

## License

MIT — see [LICENSE](LICENSE).
