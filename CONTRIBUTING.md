# Contributing

This is a small personal tool kept deliberately small. Issues and pull requests are
welcome; so is a fork, if what you want is a different tool.

## Setup

```bash
python -m venv .venv && . .venv/bin/activate    # Python 3.10 or newer
pip install -e ".[dev,mcp]"
```

Three runtime dependencies (PyYAML, jsonschema, httpx), two dev ones (pytest, ruff),
one optional extra (mcp). A pull request that adds a fourth runtime dependency needs
to argue for it in the description.

## Tests and lint

```bash
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests    # if you want the formatter's opinion
```

Both must be green before a pull request. **No test in this repository makes a network
call or reads an API key.** Everything runs against the fake adapter in
`src/evalmine/adapters/fake.py`, which is deterministic by construction: token counts,
latency and text all derive from the request's content hash. If you find yourself
wanting to hit a real provider in a test, that is a sign the thing you are testing
belongs in a pure function in `metrics.py` instead.

The producer-side TypeScript harness kit has its own zero-network suite:

```bash
npm ci --prefix packages/harness-kit
npm test --prefix packages/harness-kit
```

Its runtime dependency list must remain empty. Synthetic bundle fixtures are round-tripped
through the Python importer in CI so the TypeScript profile cannot silently drift from the
authoritative import schema.

CI runs {ubuntu, macos, windows} x {3.10, 3.13} — six legs, every one running every
test including the MCP ones. Windows is in the matrix for real reasons (path handling,
the pointer file that stands in for a symlink), so check the Windows leg before
assuming a failure is a fluke.

## No secrets, ever

Keys come from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, and
`OPENROUTER_API_KEY` in the environment and from nowhere else. There is no config-file
key path, and `evalmine run` refuses to start if a suite file contains a string matching
a known key prefix.

CI enforces this twice: a scan over the working tree, and **a scan over the entire git
history** (`git log -p --all`). A key-shaped string in any commit fails the build, and
the only fix is rewriting history, which is expensive. Two early commits in this repo
carried a key-shaped *test fixture* — not a real key — and had to be rewritten for
exactly this reason; both fixtures are now assembled at runtime. So: never paste a
key, never paste a plausible-looking fake one, and if you have already committed one,
say so in the pull request rather than force-pushing quietly.

Nothing that could carry a key is ever written to a cache entry, a report, or a log
line.

## Changes start in the spec

[`docs/spec.md`](docs/spec.md) is the contract this code is written against, not a
document that describes it after the fact. Any change to behaviour — a metric
definition, an exit code, the cache key, the suite schema, a default, a new MCP tool —
lands in the spec first, in the same pull request, above the code that implements it.
If the spec and the code disagree, the spec is right and the code is a bug.

Bug fixes, tests, docs, typos and internal refactors that change no documented
behaviour do not need a spec change.

Four rules that are settled and would need a strong argument to reopen, each with its
reasoning written out in the spec: an unknown model is a hard failure and never a
`$0.00` (§6.3); a schema failure excludes a pair from the win-rate rather than scoring
it a loss (§7.2, ruling O-3); the cost cap is enforced in `core.run_suite()` and never
in a caller (§6.4); a failed execution check is evidence shown to the judge and the
human, never an exclusion and never an automatic loss (§6.6, ruling O-4).

## Commits and pull requests

Small commits, present tense, saying what changed and why the change is the right one.
The tests that prove it belong in the same commit as the behaviour. If a pull request
changes what a number means, say so in the description in one sentence — that is the
kind of change a user of this tool has to notice.
