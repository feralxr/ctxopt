# Maintaining ctxopt

This file is for people maintaining this repository, not for people using
the package. If you're looking for how to use `ctxopt`, see
[README.md](README.md).

## Development setup

```bash
git clone https://github.com/feralxr/ctxopt.git
cd ctxopt
pip install -e ".[dev]"
pytest
```

The package is installed in editable mode (`-e`), so changes to
`src/ctxopt/` are picked up immediately without reinstalling.

## Project layout

```
src/ctxopt/
  sandbox.py       - collapse/truncate noisy tool output
  summarize.py     - conversation-aware local + LLM-backed summarization
  conversation.py  - transcript parsing (plain speaker-prefixed / JSON)
  router.py        - complexity scoring and model-tier routing
  llm.py           - optional Anthropic backend (stdlib-only, no required dep)
  tokens.py        - token counting (tiktoken, with a fallback heuristic)
  cli.py           - the `ctxopt` command-line entry point
tests/              - one test file per module above
.github/workflows/  - CI (tests.yml) and release (workflow.yml)
```

## Continuous integration

Every push and pull request to `main` runs the full test suite across
Python 3.9–3.12 via [`.github/workflows/tests.yml`](.github/workflows/tests.yml).
A PR shouldn't be merged with a red check.

## Versioning

The version lives in two places and they must match exactly, or the release
workflow refuses to publish:

- `pyproject.toml` → `[project] version = "X.Y.Z"`
- `src/ctxopt/__init__.py` → `__version__ = "X.Y.Z"`

This project doesn't promise strict semver yet (pre-1.0), but the rough
convention so far: a bump to the middle number (`0.X.0`) accompanies a
meaningful rebuild of one of the three modules (sandbox/summarize/router);
patch bumps are for fixes that don't change behavior people rely on.

## Release process

Releases publish to PyPI via [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC) — no PyPI API token is stored as a secret anywhere, in this repo or
on PyPI's side.

### One-time setup (already done for this repo, documented here in case it
### ever needs to be redone — e.g. after transferring repo ownership)

On PyPI, under **Your project → Publishing → Add a new pending publisher**:

| Field | Value |
|---|---|
| PyPI Project Name | `ctxopt` |
| Owner | `feralxr` |
| Repository name | `ctxopt` |
| Workflow name | `workflow.yml` |
| Environment name | `pypi` |

All five fields must match exactly — case-sensitive, including the `.yml`
extension on the workflow name — or publishing fails with PyPI's
`invalid-publisher` error. If the workflow file is ever renamed, this entry
has to be updated (or re-created) to match.

### Cutting a release

1. Bump the version in both `pyproject.toml` and `src/ctxopt/__init__.py`
   (see Versioning above) in a normal commit, and push it to `main`.
2. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
3. [`.github/workflows/workflow.yml`](.github/workflows/workflow.yml) picks
   up the tag push, builds the sdist/wheel, verifies the tag matches the
   version in `pyproject.toml` (and fails loudly if it doesn't — this is
   the one safety check standing between a typo and a bad release), and
   publishes.
4. Confirm it landed: `pip index versions ctxopt` or check
   `https://pypi.org/project/ctxopt/`. PyPI's index can take a minute or two
   to reflect a just-published release even after the workflow reports
   success.

### If a release needs to be redone

Move the existing tag rather than bumping to a new version for something
like a typo fix in CI config:

```bash
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z
git tag vX.Y.Z
git push origin vX.Y.Z
```

This re-triggers the workflow against whatever `main` currently points to.
PyPI does not allow re-uploading the *same* version number with different
file contents, so this only works before that version has actually
published successfully — once a version is live on PyPI, ship a new patch
version instead.

## Credential handling

Nothing in this project's CI/release pipeline requires a long-lived secret:
GitHub Actions' built-in `GITHUB_TOKEN` and PyPI's Trusted Publishing OIDC
exchange cover everything. If you ever find yourself about to add a
`PYPI_API_TOKEN` or similar to repo secrets, stop and check whether Trusted
Publishing already covers the case — it almost certainly does.
