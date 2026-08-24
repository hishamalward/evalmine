# evalmine

Leaderboard numbers have never predicted how a model change lands on the forty-odd
tasks you actually run. One tops a benchmark, you swap it in, and it is quietly worse
at the job you depend on.

evalmine answers one question about a model change, on your tasks: did it help, hurt,
or cost more for the same result? You write a YAML suite of your tasks. It runs them
across two or more models, schema-checks each answer, times it, and has an LLM judge
compare answers pairwise in both orders, so the judge's preference for whichever it
sees first cancels out. It scores that judge against your preference labels with
Cohen's kappa, and refuses to headline a win-rate when it cannot show the judge agrees
with you. Cost comes from a price table pinned to a date; an unknown model fails the
run rather than costing $0. Reports are versioned by suite hash; a three-tool MCP
server lets an agent run the evals mid-task.

Result, on the example suite here against the fake adapter: kappa 0.25 over 12 labels
is below the 0.40 floor, so the 0.463 win-rate prints flagged, not headlined. That
refusal is the tool working:

```
$ evalmine run examples/everyday-eight.yaml \
    --models anthropic/claude-haiku-4-5,google/gemini-2.5-flash --fake

run 20260823T210009Z_c4545e4e_dbc76614  (everyday-eight)
  report: reports/everyday-eight/20260823T210009Z_c4545e4e_dbc76614/report.md
  calibration: below_floor - kappa 0.25 (fair) over 12 labels - headline eligible: false
  google/gemini-2.5-flash vs anthropic/claude-haiku-4-5: win-rate 0.463 (UNCALIBRATED) [0.325-0.613] over schema-passing pairs only, n=20 - flips 3 - excluded 0
  cost: $0.0658 this run (answers $0.0081, judge $0.0578); if uncached $0.0658
```

The fake adapter is deterministic, so those figures reproduce exactly on a clean
checkout. Nothing above contacted a provider or spent a cent.

![evalmine: validate the suite, run it against the fake adapter, read the calibration and win-rate sections of the report it wrote](docs/demo.gif)

Every frame of that is a real run. Re-record it with `vhs docs/demo.tape`
([vhs](https://github.com/charmbracelet/vhs), `brew install vhs`).

**Status.** v0.1.0, pre-release. The core, the three provider adapters, execution
checks and the MCP surface are built and tested; the price table is verified against
each provider's public pricing page on its pinned date. No decision-log entry exists
yet — see [Not yet](#not-yet).

Specification: [docs/spec.md](docs/spec.md). It is the contract the code is written
against and it wins over this README wherever the two disagree.
How it works, in depth: [docs/learning/how-it-works.md](docs/learning/how-it-works.md) ([styled HTML rendering](docs/learning/how-it-works.html)).

## Quickstart

```bash
git clone https://github.com/hishamalward/evalmine.git && cd evalmine
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"          # add ,mcp -> ".[dev,mcp]" for the MCP server
```

Python 3.10 or newer. Three runtime dependencies: PyYAML, jsonschema, httpx.

**Check a suite without spending anything.** `validate` parses the file, applies the
JSON Schema, renders every prompt (an unmatched `{{placeholder}}` is a hard error),
and resolves every model string against the price table. Zero network calls.

```bash
evalmine validate examples/everyday-eight.yaml
# ok: examples/everyday-eight.yaml - 8 tasks, 20 cases, 12 labels; every prompt
# rendered; 3 model strings resolved against prices-2026-08-23.yaml
```

**Run it against the fake adapter.** `--fake` routes every model string to a built-in
deterministic adapter: no key, no network, no spend. The two model strings below are
the ones the example suite's twelve human labels refer to, so this run exercises the
calibration path end to end.

```bash
evalmine run examples/everyday-eight.yaml \
  --models anthropic/claude-haiku-4-5,google/gemini-2.5-flash --fake
```

**Run it for real.** Keys come from the environment and from nowhere else. Copy
`.env.example`, fill it in outside the repo, and export what you need.

```bash
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...

evalmine run examples/everyday-eight.yaml \
  --models anthropic/claude-haiku-4-5,google/gemini-2.5-flash \
  --max-cost 0.50
```

A pre-flight estimate runs before the first live call. If it exceeds `--max-cost`
the run is refused (exit 4) and nothing is spent. Without a cap anywhere, the CLI
default is $2.00. Every call is cached on disk by content hash, so a rerun is free
and a report is reproducible; `--no-cache` forces fresh calls and still writes them.

Other commands: `evalmine prices [--for suite.yaml]`, `evalmine last suite.yaml`,
`evalmine report <run-id>`, `evalmine compare <report_a> <report_b>`.

## The suite file

One YAML file holds your tasks, the judge config, and your labels. The shipped
example is [examples/everyday-eight.yaml](examples/everyday-eight.yaml): eight
invented tasks (rewrite, extract, classify, explain, small code change) over twenty
cases, three of them carrying an output schema, with twelve preference labels. The
full schema is spec §5; the shape is:

```yaml
suite: everyday-eight
version: 1

defaults: { temperature: 0, max_tokens: 700, timeout_s: 60 }
limits:   { max_cost_usd: 1.50 }

judge:
  model: anthropic/claude-sonnet-4-6
  rubric: |
    Prefer the answer that a competent colleague would ship without editing.
    ...
  calibration: { min_kappa: 0.40, min_labels: 10, on_below_floor: flag }

tasks:
  - id: ticket-triage
    kind: classify                # a free label, used only to group report rows
    prompt: |
      Classify this support ticket. Return JSON only.

      Ticket:
      {{ticket}}
    schema: { type: object, required: [category, severity], ... }
    rubric: |                     # appended to the suite rubric for this task
      In addition to the suite rubric: ...
    cases:
      - id: charged-twice
        vars: { ticket: "I was charged twice this month..." }

labels:
  - { task: ticket-triage, case: charged-twice,
      baseline: anthropic/claude-haiku-4-5,
      candidate: google/gemini-2.5-flash,
      prefer: candidate, note: "team-wide lockout is high, not medium" }
```

Three things about this file that are deliberate:

- **Templating is not Jinja.** Exactly `{{name}}`, substituted once, no expressions
  and no filters. A placeholder with no matching var is a hard error at load time,
  because a silently empty variable is the easiest way to make an eval quietly
  meaningless.
- **Unknown keys are errors,** at every level. A typo'd `rubrik:` that gets ignored
  produces a report that looks fine and means nothing.
- **`labels` is where the tool's credibility comes from.** They are your judgements,
  recorded before you see the win-rate, and the judge is scored against them. A suite
  with no labels still runs; it just cannot produce a headline number.

Replace the example with your own tasks. That is the entire point of the tool.

### Execution checks for code tasks

Prose is a bad proxy for code that runs. A case can declare a `check`: a bash
snippet that gets the answer's code (`$ANSWER` is a file, `$ANSWER_TEXT` the
text) and exits 0 if it works. It runs in a fresh temp dir, under a timeout,
with secrets stripped from the environment, and is never cached. Every fenced
block in the answer runs, in order, each on its own fixture; the final block is
the verdict and the earlier ones are recorded beside it, so an answer that
retracts a wrong block and writes a second one is scored on the second and
shows the retraction.

```yaml
- id: jq-remote
  vars: { task: "Write a jq filter ... the JSON is in postings.json" }
  check:
    setup: 'printf "[{\"t\":\"a\",\"remote\":true}]" > postings.json'
    run: 'jq -r "$(cat "$ANSWER")" postings.json | grep -q a'
```

The result — pass/fail, exit code, output — sits beside the answer in
`answers.jsonl`, the scorecard, and the HTML pair view, and the judge is shown
it with one fixed rule: an answer whose check failed cannot beat one that
passed. Spec §6.6.

## How to read a report

`reports/<suite>/<run-id>/report.md` alongside `report.json`, `report.html`,
`answers.jsonl` and `pairs.jsonl`. Read it in this order.

**1. Calibration, first.** It is printed above the win-rates on purpose. You want
Cohen's kappa between the judge's verdicts and your labels, with its Landis-Koch band
name attached, and the 3x3 confusion matrix under it. Kappa rather than plain
agreement because agreement is inflated the moment one category dominates, and it
will: judges learn that ties are safe. The matrix tells you *how* the judge is wrong,
which matters — a judge that never says "tie" when you do is a different problem from
one that systematically prefers whatever is new. Under it, a per-task breakdown tells
you *where*: one kappa can hide a judge that is excellent on your rewrite task and
useless on your triage task, and the average is the finding you would lose.

**2. A win-rate you should not trust.** Three conditions, any one of which is enough:

- `headline_eligible: false` — kappa is below the floor, there are too few labels, or
  kappa is undefined because both raters used one category throughout. The report
  bans the number from being a headline, flags every figure with a dagger, and the
  JSON and every MCP response carry the same flag, so an agent reading the summary
  cannot quote the number without the caveat.
- **Flip rate above 0.30.** A flip is a pair where the judge changed its answer when
  the two answers changed places. Above about a third, the win-rate is measuring
  presentation order, not quality. The report says so in the same table.
- **A small `n`, or a shrinking one.** The win-rate is computed over
  *schema-passing pairs only*: a pair where either side failed to parse or failed its
  schema is excluded rather than scored a loss, so that a model bad at emitting JSON
  does not lose a *quality* comparison for a *formatting* failure. The cost is that
  `n` shrinks, which is why the section is titled "over schema-passing pairs only,
  n=…" and why `n` is never printed without the schema-pass rate on the same screen.

**Before you publish a number, raise `min_kappa` to 0.60.** The shipped default is
0.40 — the conventional bottom of fair-to-moderate agreement, low enough that a first
suite with a dozen labels can plausibly clear it. That is the floor for *using a
number yourself*, with your own memory of how the labelling went. 0.60 —
"substantial" — is the floor for *telling someone else a number*, where that memory
does not travel. The tool ships permissive so a first suite is worth running twice;
this recommendation exists so a first suite is not what ends up in a blog post.

**3. Then the scorecard, and read cost with quality, never after it.** Schema-pass
rate (labelled `native` or `prompted`, because a provider that enforces a schema for
you and one that was merely asked nicely are not the same measurement), the exec pass
rate with its `n` where a task declares execution checks, p50 and p95 latency with
their `n`, cost this run and cost if uncached. A candidate that wins
0.55 for triple the money is a different decision from one that wins 0.55 for half.

**4. The per-task table, sorted worst-first.** A headline win-rate that did not move
while three tasks moved 0.4 in opposite directions is the finding you would otherwise
miss. `evalmine compare A B` prints exactly those movers between two runs.

**5. `report.html`, and the labelling flow.** Every run also writes one
self-contained page — no server, no dependencies, opens off a `file://` path. Same
sections, plus each judged pair side by side with **the model names hidden and the
judge's verdict folded away**, so you read the answers exactly as the judge read
them. *Prefer A · Tie · Prefer B* under each one, then **copy labels YAML** hands you
the `labels:` entries to paste back into your suite: ten minutes of clicking instead
of half an hour of hand-editing, which is the difference between a calibration set
that grows and one that does not.

The report contains no adjectives and makes no recommendation. Judgement goes in
[DECISIONS.md](DECISIONS.md), worded from your verdict — the report pre-fills the
template for you at the bottom of every run.

## MCP

`evalmine-mcp` is a stdio MCP server exposing exactly three tools, which call the
same `core.py` functions the CLI calls:

| tool | does | spends |
|---|---|---|
| `run_suite(suite_path, models, max_cost, baseline, no_cache)` | runs the suite, returns the summary and the report paths | up to the cap |
| `compare(report_a, report_b)` | the delta between two reports | nothing |
| `last_report(suite_path)` | the most recent report for a suite | nothing |

Register it by copying [.mcp.json.example](.mcp.json.example) to `.mcp.json`. Install
the extra first: `pip install -e ".[mcp]"`.

The point is that an agent can run your evals *mid-task* — "before you swap the model
in this file, run the suite and tell me the win-rate" — instead of a person reading a
report afterwards.

Three tools rather than the whole CLI because an agent-facing surface should be the
smallest set of verbs that supports the decision, and every extra tool is another way
to spend money nobody authorised.

**The caps, and why the agent's default is lower than yours.** The cap is a parameter
of `core.run_suite()`, not a CLI flag that MCP re-implements: there is exactly one
place where money can be spent, and it is capped there. If the agent supplies
`max_cost` it is used, but a request above `EVALMINE_MCP_MAX_COST_CEILING` ($5.00 by
default) is refused outright rather than clamped and run. If the agent omits it, the
cap is `min(suite.limits.max_cost_usd, EVALMINE_MCP_MAX_COST)`, default **$1.00** —
half the CLI's $2.00, because the human at the CLI typed the number and the agent did
not. An over-cap run returns a structured refusal, spends nothing, and is never
silently truncated to fit; a truncated run produces a smaller number that looks like
a complete one.

`run_suite` returns the summary and the paths, never raw provider responses. Those
stay in `answers.jsonl` on disk. A tool that streams every answer back into an
agent's context costs the caller more than the eval did, and turns an eval harness
into an exfiltration path for whatever is in your prompts. `suite_path` must also
resolve inside `EVALMINE_MCP_SUITE_ROOT` (default: the server's working directory).

## Prior art

[promptfoo](https://www.promptfoo.dev/) and [Braintrust](https://www.braintrust.dev/)
are the obvious tools here, and both are more capable than this one.

promptfoo has far more assertion types, a web viewer, red-teaming, and provider
coverage that is not three. Braintrust is a hosted platform: tracing, datasets built
from production logs, a real UI, collaboration, and the operational maturity that
comes with being someone's product. If you want breadth, or a team looking at the
same numbers, use one of those.

evalmine exists for three narrower reasons.

- **The judge is calibrated against you, or its number does not print.** Both of the
  above can score with an LLM judge. Neither makes calibration to *your* labels the
  gate on whether a win-rate is quotable. That inversion — the refusal being the
  default — is the whole thesis, and it is not a feature you can bolt on to a tool
  that ships the number regardless.
- **The decision log is a first-class artefact.** The output of an eval is not a
  number, it is a decision you have to defend in six months. `DECISIONS.md` is
  pre-filled by the report and written by a human, and it lives in your repo next to
  the code the decision was about.
- **The surface is small enough to read in a sitting.** Roughly 6,000 lines including
  four adapters, the reports and the execution checks. No LLM framework, no provider SDKs — three
  hand-written POSTs to documented JSON endpoints. That cost is real and worth
  stating: when a provider changes its API, we find out by breaking, not by
  upgrading.

If those three do not matter to you, the honest recommendation is promptfoo.

## Not yet

Out of scope for v0.1.0, and the README says so rather than leaving you to discover
it: RAG or retrieval eval; agent or multi-turn trajectories; fine-tuning anything; a
web UI; anything hosted; more than three providers; rubric auto-generation; MCP tools
beyond the three above.

Every number in this README comes from the fake adapter on the invented example
suite. No labelled run on a real suite has yet produced a calibrated number or a
`DECISIONS.md` entry; that comes before any v0.1.0 tag.

## Development

```bash
pip install -e ".[dev,mcp]"
python -m pytest -q          # 310 tests, none of which make a network call
python -m ruff check src tests
```

CI runs {ubuntu, macos, windows} x {3.10, 3.13}, every leg running every test, plus a
secret scan over the working tree *and* the full git history. No API key belongs in
this repo, and `evalmine run` refuses to start if a suite file contains a string
matching a known key prefix.

See [CONTRIBUTING.md](CONTRIBUTING.md). Changes start in `docs/spec.md`.

## License

MIT. See [LICENSE](LICENSE).
