# evalmine — specification (v0.1.0, pre-code)

**Status:** approved, with the three open questions in §14 ruled on. This document
is the contract the code is written against.

**Date:** 2026-08-23 (rulings O-1, O-2, O-3 and the name applied the same day)

---

## 1. What this is

> A public, Python eval harness that scores a model change against your own
> tasks — pairwise LLM-judge win-rates calibrated to your labels, schema-pass
> rate, latency and cost — and writes a versioned report.

Every model release ships with leaderboard numbers that have never predicted how
the model behaves on the forty-odd tasks you actually run. This tool answers one
question about a model change: **did it help me, hurt me, or cost me more for the
same result?** It answers it on your tasks, with your preferences as the
yardstick, and it refuses to give you a headline number when it cannot show that
its judge agrees with you.

### 1.1 Name

**`evalmine`**, ruled by the owner on 2026-08-23. It reads two ways, both of them
the thesis: *eval mine* — evaluate **my** tasks, not a leaderboard's — and *eval
mine*, the seam you dig your own numbers out of.

- PyPI distribution: `evalmine`
- Import package: `evalmine`
- CLI: `evalmine`
- MCP entry point: `evalmine-mcp`
- Environment variables: `EVALMINE_*`

The pre-ruling shortlist was `nof1bench` (from the clinical *n-of-1 trial*),
`ownbench` (rejected: an existing GitHub namesake in the same problem space) and
`swapjudge` (rejected: names the mechanism, not the thesis). The ruling replaced
`nof1bench`, which this document was originally written against.

**Re-checked 2026-08-23 (round 2):** `evalmine` is still free.
`https://pypi.org/pypi/evalmine/json` returns 404 (no such distribution).
`gh api "search/repositories?q=evalmine+in:name"` returns `total_count: 0` (no
GitHub repository named `evalmine`, under any owner), and a web search for
`"evalmine" github repository` surfaces no meaningful collision - only
unrelated eval-tooling repos (`openai/evals`, `mlfoundations/evalchemy`,
etc.) and an unrelated `evalmee` GitHub organisation, none of which share the
name. No name is reserved by this check alone; re-check both again
immediately before the first release, since a check made during development
can go stale.

### 1.2 Clean-room statement

The *pattern* in this spec comes from a private, product-specific eval harness in
a closed codebase (Listenality). Before writing this spec I read that harness —
its CLI, its config, its scoring script, its decision log, its blinded-review
sheet — **for pattern only**. What carried over is structural and, in most cases,
independently obvious once you have run evals for a while:

- separating *run* from *score* from *report*;
- a decision log as the place where judgment lives, kept out of the report;
- blinding the models from the human labeller;
- cost and reliability reported next to quality, never after it;
- pinning prices to a date, because a cost number without a date is a lie (that
  harness carries a correction banner over a whole quarter of under-counted
  figures for precisely this reason — the lesson is baked into §6.3 here).

What was **deliberately not carried over**:

- no code, in any language — that harness is TypeScript, this is Python written
  from this spec;
- none of its prompts, system prompts, rubrics, or JSON schemas;
- none of its task definitions, task IDs (T1–T5), stack names, or model stacks;
- none of its data: no fixtures, no ground-truth files, no track/library data, no
  sample outputs, no vocabularies (mood/genre/language lists);
- none of its domain concepts — this tool has no notion of a product pipeline,
  write-back, database snapshots, batch enrichment, or per-stack configuration;
- none of its report copy, HTML, or CSS;
- no decision-log entries; the template in §9.4 is written from scratch and
  contains no figures from it.

The example suite in §5.4 is invented for this repository. It is not that
harness's tasks, not Hisham's real tasks, and not anyone's production workload.

### 1.3 Scope

**In (v0.1.0).** A CLI that runs a YAML suite across two or more model strings;
schema-pass rate, latency, cost, and pairwise judge win-rate against a baseline
with position swap and ties; judge calibration against human labels with a floor
below which win-rates are not printed as a headline; disk caching by content
hash; Markdown + JSON reports with a "what changed" section; three provider
adapters (Anthropic, OpenAI, Google) behind one small interface, plus a fake
adapter; an MCP server exposing exactly three tools.

**Out — and the README says so.** RAG or retrieval eval; agent or multi-turn
trajectories; fine-tuning; a web UI; anything hosted; more than three providers;
rubric auto-generation; MCP tools beyond the three in §11.

### 1.4 Blast radius

A new, empty, public repository at `~/public_repos/evalmine`. Nothing outside it
is created, read, or modified. At runtime the tool writes only under the working
directory: the cache directory (default `.evalmine-cache/`), the reports
directory (default `reports/`), and `DECISIONS.md` if and only if the user asks
for it. It reads API keys from the environment and from nowhere else — no config
file may contain a key, and `evalmine run` refuses to start if a suite file contains
a string matching a known key prefix.

---

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **suite** | one YAML file: defaults, a judge config, tasks, and optional human labels |
| **task** | one prompt template with an optional schema and rubric |
| **case** | one set of `vars` substituted into a task's prompt. A task has ≥1 case |
| **call** | one request to one model for one (task, case) |
| **answer** | the text a call returned, plus usage, latency, and schema verdict |
| **pair** | (task, case, baseline model, candidate model) — the unit that gets judged |
| **pass** | one judge call on a pair in one presentation order. Every pair gets two |
| **run** | one execution of one suite over one model list, producing one report |
| **model string** | `provider/model-id`, e.g. `anthropic/claude-sonnet-4-6`. The prefix selects the adapter; the whole string is the price-table key |

---

## 3. Repository layout

```
evalmine/
  README.md              MIT; first 200 words are problem -> approach -> result
  LICENSE                MIT
  CONTRIBUTING.md        one page
  DECISIONS.md           the user's decision log (starts with a header only)
  .env.example           key names, no values
  .mcp.json.example      §11.5
  pyproject.toml
  docs/
    spec.md              this file
    learning/how-it-works.md    (step 4 of the plan; not in v0.1.0 scope here)
  examples/
    everyday-eight.yaml  §5.4
  prices/
    prices-2026-08-23.yaml      §6.3
  src/evalmine/
    __init__.py
    cli.py               argparse only; no logic
    core.py              run_suite(), compare(), last_report() — the library API
    suite.py             load + validate a suite; render prompts
    adapters/
      base.py            the Protocol, Request, Response, errors
      anthropic.py  openai.py  google.py  fake.py
    cache.py             §6.5
    judge.py             §7.1–7.2
    metrics.py           §6, §7 — pure functions, no I/O
    prices.py            §6.3
    report.py            §9
    mcp_server.py        §11  (optional extra, Python >= 3.10)
  tests/
```

---

## 4. CLI surface

```
evalmine run <suite.yaml> --models <m1,m2[,m3...]>
         [--baseline <model>] [--judge <model>] [--repeats N]
         [--max-cost USD] [--no-cache] [--cache-dir PATH] [--out DIR]
         [--fake] [--fail-under-calibration] [--json] [-v]

evalmine validate <suite.yaml>          # parse, schema-check, render every prompt,
                                    # resolve every model against the price table.
                                    # Zero network calls. Exit 0 or 1.
evalmine prices [--table PATH] [--for <suite.yaml>]
                                    # print the pinned table; with --for, assert
                                    # every model the suite could use resolves.
evalmine report <run-id | report.json>  # re-render report.md from report.json
evalmine last <suite.yaml>              # print the most recent run-id for this suite
evalmine compare <report_a> <report_b>  # print the §9.3 delta between two reports
```

Flag notes:

- `--models` is ordered. `--baseline` defaults to the **first** model listed;
  every other model is a candidate compared against it. Two models is the normal
  case; three or more produce one win-rate column per candidate, never a
  round-robin (a round-robin is `k(k-1)/2` times the judge spend and nobody asked
  for it).
- `--judge` overrides `judge.model` in the suite. The judge may be one of the
  models under test; the report says so in a warning, because it is a known
  self-preference risk.
- `--repeats N` (default 1) runs each call N times and treats each repeat as a
  separate case for schema/latency, and each repeat-index as a separate pair for
  judging. Only useful at temperature > 0.
- `--max-cost` overrides `limits.max_cost_usd`. See §6.4.
- `--no-cache` ignores existing cache entries but still **writes** the fresh
  results into the cache. There is no flag that disables writing; a run you
  cannot re-report from is not worth having.
- `--fake` forces every model string onto the fake adapter regardless of prefix.
  Costs nothing, contacts nothing, is deterministic. Used by tests and by anyone
  who wants to see what a report looks like.
- `--fail-under-calibration` turns a below-floor calibration result into exit 3
  instead of a flagged report. Off by default; intended for CI.

**Exit codes.** `0` ok · `1` usage or suite error (bad YAML, unknown model,
missing key, unrenderable prompt) · `2` provider or runtime failure that stopped
the run · `3` calibration below floor **and** `--fail-under-calibration` · `4`
refused before spending: pre-flight estimate exceeds `--max-cost` · `5` aborted
mid-run on the live cost ceiling, partial report written.

---

## 5. The suite file

### 5.1 Top level

| Key | Type | Required | Meaning |
|---|---|---|---|
| `suite` | string | yes | identifier; must match `^[a-z0-9][a-z0-9-]{0,63}$`. Names the report directory |
| `version` | int | yes | suite-format version. `1` for now. An unknown version is a hard error, not a warning |
| `description` | string | no | free text, reproduced in the report header |
| `defaults` | object | no | per-call defaults (§5.2) |
| `limits` | object | no | `max_cost_usd` (float) |
| `judge` | object | yes | §5.3 |
| `tasks` | array | yes | ≥1 task |
| `labels` | array | no | human preference labels (§5.5) |
| `labels_path` | string | no | path to a YAML/JSON file holding `labels`, relative to the suite file. Mutually exclusive with `labels` |

### 5.2 `defaults` and per-task overrides

`temperature` (float, default `0`), `max_tokens` (int, default `700`),
`timeout_s` (int, default `60`), `top_p` (float, optional), `stop` (array of
strings, optional), `system` (string, optional). Any of them may be set on an
individual task, where it wins.

### 5.3 `judge`

| Key | Type | Required | Meaning |
|---|---|---|---|
| `model` | model string | yes | who judges |
| `rubric` | string | yes | the default rubric, shown to the judge on every pair |
| `max_tokens` | int | no | default `400` |
| `temperature` | float | no | default `0` |
| `calibration.min_kappa` | float | no | default `0.40` (§8) |
| `calibration.min_labels` | int | no | default `10` |
| `calibration.on_below_floor` | `flag` \| `fail` | no | default `flag` |

### 5.4 `tasks[]` and `cases[]`

A task:

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | unique in the suite; `^[a-z0-9][a-z0-9-]{0,63}$` |
| `prompt` | string | yes | the user message. `{{var}}` placeholders are substituted from the case's `vars` |
| `system` | string | no | system message |
| `kind` | string | no | a free label (`rewrite`, `extract`, …) used only to group rows in the report |
| `schema` | object | no | inline JSON Schema (Draft 2020-12) the parsed answer must satisfy |
| `schema_path` | string | no | path to a schema file; mutually exclusive with `schema` |
| `rubric` | string | no | appended to the suite rubric for this task's pairs |
| `judge` | bool | no | default `true`. `false` means schema/latency/cost only — no judge calls, no win-rate contribution |
| `cases` | array | yes | ≥1 case |
| `check` | object | no | execution-check defaults for every case (§6.6): `setup`, `timeout_s`, and optionally `run` |
| *(any `defaults` key)* | | no | overrides the suite default |

A case: `id` (string, unique within the task), `vars` (object of string →
scalar), and optionally `check` (§6.6: `run`, plus `setup` / `timeout_s`
overriding the task's). A task with no placeholders may declare `vars: {}`.

**Templating** is deliberately not Jinja: exactly `{{name}}`, whitespace inside
the braces allowed, substituted once, no expressions, no filters, no loops. A
placeholder with no matching var is a **hard error at load time** (exit 1) — a
silently empty variable is the single easiest way to make an eval quietly
meaningless. `evalmine validate` renders every prompt to catch this without spending
anything.

**Suite validation** is by a JSON Schema shipped at
`src/evalmine/suite.schema.json` and applied with `jsonschema`.
`additionalProperties: false` at every level: an unrecognised key is a hard
error, not a silent no-op, because a typo'd `rubrik:` that gets ignored produces
a report that looks fine and means nothing.

### 5.5 `labels[]`

One object per human judgement:

| Key | Type | Required |
|---|---|---|
| `task` | task id | yes |
| `case` | case id | yes |
| `baseline` | model string | yes |
| `candidate` | model string | yes |
| `prefer` | `baseline` \| `candidate` \| `tie` | yes |
| `note` | string | no |

A label whose `(task, case)` does not exist is a hard error. A label whose model
pair is not in the current run is **ignored**, and the report states how many
labels were ignored and for which pairs — otherwise you get a "calibrated"
report whose calibration came from labels about models that were not in it.

### 5.6 The complete example suite

The block below is `examples/everyday-eight.yaml` **verbatim**. The file is the
source of truth; a test asserts that this section and the file are byte-identical
so the spec cannot drift away from a suite that actually parses.

<!-- BEGIN examples/everyday-eight.yaml -->
```yaml
# everyday-eight — the example suite that ships with evalmine.
#
# Eight small, invented tasks of the kind a working developer actually runs on a
# model: rewrite, extract, classify, explain, and a small code change. Nothing
# here is a real workload; the suite exists so that `evalmine run` has something to
# run and so the schema below has a complete worked example. Replace it with your
# own tasks — that is the entire point of the tool.
#
# Suite schema: docs/spec.md §5.

suite: everyday-eight
version: 1
description: >
  Eight everyday LLM tasks (rewrite, extract, classify, explain, small code
  change) invented as a demonstration suite. Twenty cases across eight tasks,
  three of the tasks carrying an output schema, twelve human preference labels
  for judge calibration.

defaults:
  temperature: 0
  max_tokens: 700
  timeout_s: 60

limits:
  # Refuse the run if the pre-flight estimate exceeds this. --max-cost overrides.
  max_cost_usd: 1.50

judge:
  model: anthropic/claude-sonnet-4-6
  max_tokens: 400
  rubric: |
    Prefer the answer that a competent colleague would ship without editing.
    Weigh, in order: (1) does it do what was asked, exactly; (2) is every claim
    in it supported by the input; (3) is it as short as it can be while still
    complete. Do not reward length, hedging, formatting flourishes, or a
    friendlier tone. If the two answers are equally shippable, say "tie" —
    ties are expected and are not a failure of the judge.
  calibration:
    min_kappa: 0.40
    min_labels: 10
    on_below_floor: flag

tasks:
  - id: changelog-line
    kind: rewrite
    prompt: |
      Rewrite this commit message as one changelog line for end users of a
      desktop note-taking app. One sentence, present tense, no issue numbers,
      no internal module names, under 20 words.

      Commit message:
      {{commit}}
    rubric: |
      In addition to the suite rubric: the line must be understandable by
      someone who has never seen the codebase, must not exceed 20 words, and
      must not invent a benefit the commit does not support.
    cases:
      - id: debounce
        vars:
          commit: |
            fix(editor): debounce autosave to 800ms

            The autosave timer fired on every keystroke, which meant a 40-page
            note re-serialised ~12x/second on a fast typist and pinned a core.
      - id: sync-conflict
        vars:
          commit: |
            feat(sync): last-writer-wins conflict resolution for note bodies

            Previously a conflict silently dropped the remote copy. Now the newer
            mtime wins and the loser is kept in .conflicts/ for 30 days.
      - id: dep-bump
        vars:
          commit: |
            chore(deps): bump sqlite bindings 3.44 -> 3.46

            Picks up the FTS5 trigram tokenizer. No user-visible change yet.

  - id: receipt-fields
    kind: extract
    prompt: |
      Extract the fields below from this receipt text. Use null for anything the
      receipt does not state. Do not compute or infer values that are not
      printed. Return JSON only.

      Receipt:
      {{receipt}}
    schema:
      type: object
      additionalProperties: false
      required: [merchant, date, currency, total, tax, line_items]
      properties:
        merchant: { type: [string, "null"] }
        date: { type: [string, "null"], description: "ISO 8601 date as printed" }
        currency: { type: [string, "null"] }
        total: { type: [number, "null"] }
        tax: { type: [number, "null"] }
        line_items:
          type: array
          items:
            type: object
            additionalProperties: false
            required: [description, amount]
            properties:
              description: { type: string }
              amount: { type: number }
    cases:
      - id: cafe
        vars:
          receipt: |
            THE SECOND CUP
            14 Mill Lane
            2026-03-04  09:12
            Flat white          3.40
            Almond croissant    2.95
            ------------------------
            SUBTOTAL            6.35
            VAT 20%             1.27
            TOTAL GBP           7.62
            CARD ****4417
      - id: hardware-messy
        vars:
          receipt: |
            HARDWARE DEPOT #221 --- thank you!!
            03/04/26
            2 x M6 bolt @ 0.40 ......... 0.80
            wood glue .................. 5.49
            *** MEMBER DISCOUNT -1.00 ***
            TOTAL              5.29
            (tax included, rate not shown)
      - id: no-total
        vars:
          receipt: |
            Kiosk 7 — Provisional slip
            Two coffees, one water
            Amounts to be confirmed at the counter.

  - id: ticket-triage
    kind: classify
    prompt: |
      Classify this support ticket. Choose exactly one category and one
      severity. Severity "critical" is reserved for data loss, a security
      problem, or a total outage. Return JSON only.

      Ticket:
      {{ticket}}
    schema:
      type: object
      additionalProperties: false
      required: [category, severity, needs_human]
      properties:
        category:
          type: string
          enum: [billing, bug, feature_request, how_to, account_access, abuse]
        severity:
          type: string
          enum: [low, medium, high, critical]
        needs_human: { type: boolean }
    cases:
      - id: charged-twice
        vars:
          ticket: |
            I was charged twice this month, once on the 2nd and once on the 3rd,
            same amount. I only have one seat. Can you refund one of them?
      - id: cant-login
        vars:
          ticket: |
            Password reset emails never arrive. I've tried three addresses and
            checked spam. This is the fourth day. My whole team is locked out.
      - id: polite-feature
        vars:
          ticket: |
            Love the app. Any chance of a dark theme? Not urgent at all, just
            would be easier on the eyes at night.

  - id: stacktrace-explain
    kind: explain
    prompt: |
      Explain this error to a developer in their first year. Say what went
      wrong, why it went wrong, and the single most likely fix. Three short
      paragraphs at most. Do not paste the traceback back at them.

      {{trace}}
    rubric: |
      In addition to the suite rubric: the diagnosis must be correct, the fix
      must be the most likely one rather than a list of possibilities, and the
      explanation must not assume knowledge the reader was told not to have.
    cases:
      - id: none-attr
        vars:
          trace: |
            Traceback (most recent call last):
              File "app/report.py", line 41, in build
                rows = load(path).records
            AttributeError: 'NoneType' object has no attribute 'records'
      - id: unicode
        vars:
          trace: |
            Traceback (most recent call last):
              File "ingest.py", line 12, in read
                return open(p).read()
            UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa3 in position
            1041: invalid start byte

  - id: meeting-actions
    kind: extract
    prompt: |
      Pull the action items out of these notes. An action item has an owner and
      something to be done. If a due date is not stated, use null — do not guess
      one from context. Return JSON only.

      Notes:
      {{notes}}
    schema:
      type: object
      additionalProperties: false
      required: [actions]
      properties:
        actions:
          type: array
          items:
            type: object
            additionalProperties: false
            required: [owner, action, due]
            properties:
              owner: { type: string }
              action: { type: string }
              due: { type: [string, "null"] }
    cases:
      - id: standup
        vars:
          notes: |
            Priya: staging is still on the old image, she'll redeploy after lunch.
            Marcus asked whether we still need the CSV export — nobody knew, he'll
            check with support by Friday. General agreement that the onboarding
            copy is confusing. Someone should rewrite it.
      - id: no-actions
        vars:
          notes: |
            Short one today. Everyone's blocked on the vendor, nothing moved
            since yesterday, we'll pick it up when they reply.

  - id: regex-repair
    kind: code
    prompt: |
      This function is meant to return every hashtag in a string, without the
      leading '#', lowercased, in order of first appearance and without
      duplicates. It does not. Return the corrected function and nothing else.

      ```python
      {{code}}
      ```
    rubric: |
      In addition to the suite rubric: the returned function must actually
      satisfy the stated contract, including the de-duplication and the
      ordering. Prefer the answer that changes less of the original.
    cases:
      - id: dup-and-case
        vars:
          code: |
            import re

            def hashtags(text):
                return re.findall(r"#\w+", text)
      - id: greedy
        vars:
          code: |
            import re

            def hashtags(text):
                found = re.findall(r"#(.*)", text)
                return [f.lower() for f in found]

  - id: blurb-tighten
    kind: rewrite
    prompt: |
      Rewrite this product blurb in 25 words or fewer. Keep every concrete fact.
      Remove every superlative and every claim you cannot support from the text
      itself. Return only the rewritten blurb.

      {{blurb}}
    rubric: |
      In addition to the suite rubric: a rewrite over 25 words loses outright,
      however good it reads. A rewrite that drops a concrete fact (a number, a
      capability, a constraint) loses to one that keeps it.
    cases:
      - id: superlatives
        vars:
          blurb: |
            The world's most powerful and revolutionary task manager, trusted by
            millions, delivers an unparalleled experience. Sync across up to 5
            devices, share lists with 20 collaborators, and work fully offline.
      - id: already-short
        vars:
          blurb: |
            Offline-first task manager. Syncs to 5 devices. Shares lists with up
            to 20 people. No account required.
      - id: vague
        vars:
          blurb: |
            A next-generation platform that empowers teams to unlock their
            potential through seamless, AI-driven collaboration at scale.

  - id: sql-from-question
    kind: code
    prompt: |
      Write one SQL query answering the question, against this schema. Standard
      SQL, no vendor extensions. Return only the query.

      Schema:
        orders(id, customer_id, placed_at DATE, status TEXT, total_cents INT)
        customers(id, name TEXT, country TEXT, created_at DATE)
        refunds(id, order_id, amount_cents INT, refunded_at DATE)

      Question: {{question}}
    rubric: |
      In addition to the suite rubric: the query must be correct against the
      stated schema, including the treatment of orders with no refund. Prefer
      the simpler query when both are correct; do not reward CTEs for their own
      sake.
    cases:
      - id: net-revenue
        vars:
          question: >
            For each country, net revenue in 2025 — order totals minus refunds —
            counting only orders with status 'complete', highest first.
      - id: never-ordered
        vars:
          question: >
            How many customers created in 2025 have never placed an order?

# Human preference labels. Each one says which answer YOU preferred on a given
# case for a given baseline/candidate pair. The judge is scored against these:
# see docs/spec.md §7 (Cohen's kappa) and §8 (calibration floor). Labels whose
# model pair is not in the current run are ignored and reported as such.
labels:
  - { task: changelog-line,    case: debounce,      baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: candidate, note: "haiku kept the module name" }
  - { task: changelog-line,    case: sync-conflict, baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: baseline,  note: "gemini invented a UI for the conflict folder" }
  - { task: changelog-line,    case: dep-bump,      baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: tie }
  - { task: receipt-fields,    case: hardware-messy, baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: baseline,  note: "gemini computed a tax figure the receipt never printed" }
  - { task: receipt-fields,    case: no-total,      baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: tie }
  - { task: ticket-triage,     case: cant-login,    baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: candidate, note: "team-wide lockout is high, not medium" }
  - { task: ticket-triage,     case: polite-feature, baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: tie }
  - { task: stacktrace-explain, case: none-attr,    baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: baseline,  note: "gemini listed four possible causes instead of one fix" }
  - { task: meeting-actions,   case: standup,       baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: candidate, note: "haiku assigned the unowned rewrite to Marcus" }
  - { task: regex-repair,      case: dup-and-case,  baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: candidate }
  - { task: blurb-tighten,     case: superlatives,  baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: baseline,  note: "gemini came in at 31 words" }
  - { task: sql-from-question, case: net-revenue,   baseline: anthropic/claude-haiku-4-5, candidate: google/gemini-2.5-flash, prefer: tie,       note: "both handled the missing-refund join correctly" }
```
<!-- END examples/everyday-eight.yaml -->

---

## 6. Metrics, exactly as computed

All of §6 and §7 live in `metrics.py` as pure functions over plain dicts: no
network, no filesystem, no clock. That is what makes them unit-testable, and the
unit tests are the point.

### 6.1 Schema-pass rate

Only for tasks that declare a schema. For each answer:

1. `json.loads(text)` on the whole response. If that succeeds, that is the
   parsed value.
2. If it fails, take the **first** fenced block delimited by ` ```json ` or
   ` ``` ` and `json.loads` that. Nothing else — no brace-matching, no repair, no
   retry. A model that cannot emit JSON on request is telling you something and
   the harness should not hide it.
3. If both fail: `parse_fail`.
4. Otherwise validate against the schema with `jsonschema` Draft 2020-12. Valid →
   `pass`; invalid → `schema_fail`, and the first validation error message is
   recorded.

`schema_pass_rate = pass / (pass + schema_fail + parse_fail)` per model, per
task, and overall. `parse_fail` and `schema_fail` are also reported separately;
they are different bugs.

**Recorded alongside every schema task:** `schema_mode` ∈ `native` (the provider
enforced the schema for us) | `prompted` (the schema was appended to the prompt
as an instruction because the provider or model does not support enforcement).
Comparing a `native` model's schema-pass rate to a `prompted` model's is
comparing two different things, and the report labels the column rather than
silently averaging them.

### 6.2 Latency

`latency_ms` is wall-clock around the adapter's HTTP call, measured once, stored
in the cache entry with the answer. Statistics are computed over stored values,
so a fully cached rerun reproduces the same latency numbers as the original run
rather than reporting ~0 ms. Every latency block carries `live_fraction` — the
share of the calls in it that were made in *this* run — so a reader can see when
they are looking at last week's timings.

- `p50` = median. Even n: mean of the two middle values.
- `p95` = **nearest-rank**: sort ascending, take the value at 1-indexed position
  `ceil(0.95 * n)`. For n < 20 this is the maximum; the report prints `p95 (n=k)`
  so nobody mistakes a max for a percentile.

Judge-call latency is tracked but never mixed into a model's latency.

### 6.3 Cost

The price table is a file in the repo, pinned by date:
`prices/prices-YYYY-MM-DD.yaml`. `evalmine run` uses the newest file in `prices/`
unless `--prices` names one, and the report header states which file it used.

```yaml
pinned: 2026-08-23
currency: USD
notes: >
  Prices per million tokens, taken from each provider's public pricing page on
  the pinned date. Illustrative values shown here; every shipped row carries the
  URL it came from and the date it was read.
models:
  - model: anthropic/claude-sonnet-4-6
    input_per_mtok: 3.00
    output_per_mtok: 15.00
    cached_input_per_mtok: 0.30
    source: https://www.anthropic.com/pricing
    read_on: 2026-08-23
```

For one call:

```
cost = input_tokens        / 1e6 * input_per_mtok
     + output_tokens       / 1e6 * output_per_mtok
     + cached_input_tokens / 1e6 * cached_input_per_mtok   # 0 if absent
```

Rules that are not negotiable:

- **Unknown model is a hard failure.** Model strings are resolved against the
  table during planning, *before the first provider call*. Any unresolved string
  raises `UnknownModelError` and exits 1, naming the string and the table file.
  There is no fallback price, no `$0.00`, no warning-and-continue. A silent zero
  is how a cost comparison becomes a lie, and this project exists to make cost
  comparisons that hold up.
- **Missing usage is not zero.** If a provider returns no token counts, that
  call's cost is `null`, the model's cost total is reported as
  `>= $X (k calls missing usage)`, and `report.cost_incomplete = true`.
- **Reasoning/thinking tokens bill at the output rate** and are included in
  `output_tokens` whenever the provider reports them. Where a provider reports
  them separately, they are added to `output_tokens` and also recorded as
  `reasoning_tokens` so the report can show the split. Not counting them
  under-reports the cost of reasoning models by multiples.
- **Judge spend is separate.** `cost.answers`, `cost.judge`, `cost.total`. The
  judge is a real expense of running this tool and hiding it inside the models'
  numbers would flatter both.
- Cache hits cost `0` in this run and the report shows both `cost_this_run` and
  `cost_if_uncached`.
- **An unverified table says so, in the report.** Each row may carry
  `source` and `read_on`; the table may carry `verified: true|false`. A table with
  `verified: false` — which is what the repo ships until someone has re-read every
  provider's pricing page on the pinned date — makes the report header print
  `price table: <file> (UNVERIFIED — figures are placeholders)` and sets
  `report.prices_verified = false`. The alternative is a repo full of plausible
  numbers nobody checked, which is the same failure as a silent `$0.00` wearing
  better clothes.

### 6.4 The cost guard (`--max-cost`)

Enforced in `core.run_suite()`, **not** in `cli.py`. The CLI parses a flag and
hands it to the library; the MCP server hands the same argument to the same
function. There is exactly one place where money can be spent without a cap, and
it does not exist.

**Pre-flight estimate**, computed after planning and before the first live call,
over calls that are not already cache hits:

```
est_input_tokens  = ceil(len(rendered_prompt + system) / 4)     # chars/4 heuristic
est_output_tokens = max_tokens                                   # worst case
est_call_cost     = est_input_tokens/1e6*in_price + est_output_tokens/1e6*out_price

judge call: est_input = ceil((len(rubric) + 2*prompt) / 4) + 2*max_tokens
            est_output = judge.max_tokens
```

The heuristic is documented in the report as a heuristic. It over-estimates
(output is assumed maximal), which is the right direction for a guard.

If `estimate > max_cost`: **refuse**, exit 4, print the estimate, the cap, and
the per-model breakdown. Nothing is spent. If no cap is set anywhere, the default
is `$2.00` for the CLI and `$1.00` for MCP (§11.4).

**Live ceiling.** Actual spend accumulates as calls return. If it crosses
`max_cost` mid-run, the run stops, writes a report marked
`aborted_over_budget: true` with everything completed so far, and exits 5. The
report is never presented as complete when it is not.

### 6.5 The cache key

```python
key = sha256(canonical(payload)).hexdigest()

payload = {
  "v": 1,                       # cache-format version
  "kind": "answer" | "judge",
  "provider": "anthropic",
  "model": "anthropic/claude-sonnet-4-6",
  "system": <string or null>,
  "prompt": <the fully rendered prompt string>,
  "params": {"temperature":…, "max_tokens":…, "top_p":…, "stop":[…], "seed":…},
  "schema": <the schema dict, or null>,
  "schema_mode": "native" | "prompted",
  "adapter_version": 1,         # bumped when an adapter changes what it sends
  "repeat": 0,                  # repeat index when --repeats > 1
}

canonical(o) = json.dumps(o, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8")
```

**Not in the key**, deliberately: the date, the run id, the suite hash, the suite
file path, the price table, the API key, and any environment. Two runs a month
apart with the same prompt and params hit the same entry — that is the feature.

Judge passes hash the same way with `kind: "judge"`; because a swapped pass has
the two answers in the other order, its rendered prompt differs and it gets its
own key. Nothing special is needed to keep the two passes distinct.

Path: `<cache-dir>/<provider>/<key[:2]>/<key>.json`. Entry:

```json
{"key":"…","created_at":"2026-08-23T09:14:02Z","model":"…","text":"…",
 "input_tokens":812,"output_tokens":140,"cached_input_tokens":0,
 "reasoning_tokens":0,"latency_ms":2140,"finish_reason":"stop",
 "adapter_version":1,"tool_version":"0.1.0"}
```

A corrupt or unreadable entry is treated as a miss and overwritten. `--no-cache`
skips the read and performs the write.

---

### 6.6 Execution checks

A code task judged on its prose rewards code that reads well and does not run.
A case may therefore declare a **check**: a bash snippet that is handed the
answer's code and exits `0` if the code satisfies the contract.

```yaml
- id: small-code-daily
  kind: code
  prompt: "{{task}}"
  check:
    timeout_s: 20                      # task-level defaults
  cases:
    - id: jq-remote-postings
      vars: { task: "... the JSON is in postings.json ..." }
      check:
        setup: |                       # lays down fixtures in a fresh temp dir
          cat > postings.json <<'EOF'
          [ ... ]
          EOF
        run: |                         # exit 0 = pass
          jq -r "$(cat "$ANSWER")" postings.json > out.txt
          diff out.txt expected.txt
```

Exactly as computed:

1. **Code extraction.** Every fenced block in the answer, whatever its language
   tag, with the fences removed and empty blocks dropped; an answer with no fence
   is one block, used whole, stripped. Each block runs on its own fresh fixture,
   in order. Past five blocks only the last five run, and the record says so by
   index.
2. **Environment.** A fresh temporary directory per answer, removed afterwards.
   `ANSWER` is the path of a file holding the code, `ANSWER_TEXT` is the code
   itself, `EVALMINE_CHECK=1`. Every environment variable whose name contains
   `KEY`, `TOKEN`, `SECRET`, `PASSWORD` or `CREDENTIAL` is stripped: what runs
   is model-written code.
3. **`setup`** runs first in that directory. A non-zero exit is recorded as
   `error` — the fixture is broken, which says nothing about the answer — with
   the output prefixed `setup failed:`.
4. **`run`** then runs. Exit `0` → `pass`; anything else → `fail` with the exit
   code. Exceeding `timeout_s` (default 30) → `fail` with no exit code. Combined
   stdout and stderr (the last 4000 characters) are recorded with the answer.
5. **Verdict.** The final block's result is the answer's `check_status`,
   `check_exit` and `check_output`. Every block that ran is recorded in order
   under `check_blocks` (index, status, exit code, output). An answer that
   retracts a wrong block and offers a second is scored on the second, and the
   record shows it took two.
6. `check_pass_rate = pass / (pass + fail + error)` per model, per task, and
   overall, over answers that had a check. `error` counts against the rate so a
   broken fixture shows up as a number rather than as a quietly smaller `n`.
   `multi_block` counts the answers whose check ran more than one block.

Checks are **never cached**: they are local and cheap, and editing one must
re-evaluate every answer without an API call. They run on cached answers too.

A failed check is **not an exclusion** (contrast ruling O-3). The pair is still
judged, and the judge is shown both results — status, exit code, output, and for
a multi-block answer the sequence of block verdicts — under a fixed rule: *an
answer whose check failed cannot beat an answer whose check passed; if both
passed or both failed, decide on the rubric and on what the output shows; an
answer judged on its final block still carries its earlier blocks, and reaching
a passing block after a failing one is not the same as being right the first
time.* The human sees the same in `report.html`, beside each answer, block by
block.
A check that ran is part of the record on every surface: `answers.jsonl`
(`check_status`, `check_exit`, `check_output`), the scorecard, the per-task
table, and the failures section.

Checks need `bash`: on `PATH`, or Git for Windows' bash on Windows (the
`System32` `bash.exe` is the WSL launcher and is never used), or whatever
`EVALMINE_BASH` names. They are the one place evalmine executes something it did
not write; keep fixtures synthetic.

## 7. The judge

### 7.1 Protocol

For every pair — (task, case, baseline B, candidate C) — the judge is called
**twice**:

- **pass 1** presents B as "Answer 1" and C as "Answer 2";
- **pass 2** presents C as "Answer 1" and B as "Answer 2".

The judge never sees a model name, a provider, a price, a latency, or which
answer is the baseline. It sees the original task prompt, the suite rubric plus
the task rubric, the two answers, and — for a case with an execution check
(§6.6) — each answer's check result, swapped along with the answers in pass 2.
It is asked for JSON:

```json
{"winner": "1" | "2" | "tie", "reason": "<one sentence>"}
```

parsed by the §6.1 rules. A judge response that will not parse is retried once;
if it still will not parse, that **pass** is recorded as `judge_unparseable` and
the pair is excluded from the win-rate (counted in `excluded_pairs`).

Position swap exists because LLM judges have a measurable preference for the
first answer they are shown. Running one order and reporting the number is the
most common way to publish a win-rate that is partly an artefact of ordering.

### 7.2 Pair scoring

Each pass maps to a verdict in `{baseline, candidate, tie}`. The two verdicts
combine into one pair score `s ∈ [0, 1]`, from the candidate's point of view:

| pass 1 | pass 2 | s | recorded as |
|---|---|---|---|
| candidate | candidate | **1.0** | consistent win |
| baseline | baseline | **0.0** | consistent loss |
| tie | tie | **0.5** | tie |
| candidate | tie *(either order)* | **0.75** | soft win |
| baseline | tie *(either order)* | **0.25** | soft loss |
| candidate | baseline *(either order)* | **0.5** | **flip** — counted in `position_flips` |

A flip means the judge changed its mind when the answers changed places; the two
passes cancel to 0.5 rather than being thrown away, and the flip rate is printed
next to the win-rate. **A win-rate with a flip rate above 0.30 is not a
measurement, and the report says so in the same table.**

**Exclusions, decided before any judge call** (ruling O-3, 2026-08-23):

A pair in which **either** side did not produce a usable answer is **excluded from
the win-rate**. It is not an automatic loss. No judge call is made and no judge
money is spent.

| Condition | Reason recorded |
|---|---|
| baseline `parse_fail` or `schema_fail`, candidate fine | `baseline_schema_fail` |
| candidate `parse_fail` or `schema_fail`, baseline fine | `candidate_schema_fail` |
| both sides `parse_fail` or `schema_fail` | `both_schema_fail` |
| either side a provider hard error after retries | `provider_error` (run is flagged) |
| both judge passes unparseable after one retry | `judge_unparseable` |

Every excluded pair is counted in `excluded_pairs` and **broken out by reason** in
both `report.json` and the report's exclusions section.

Why exclusion and not a loss: the win-rate is supposed to answer "which answer is
better", and a model that is merely bad at emitting JSON would otherwise lose a
quality comparison for a formatting failure, tangling two findings in one number.
The formatting failure is not forgiven — it is already the headline
**schema-pass rate** of §6.1, which is reported per model next to the win-rate.
The cost of this choice is that a model that fails schema often is judged on the
subset of cases where it did not, so the win-rate's `n` shrinks and the reader
must read the two numbers together. That is why the win-rate section is titled
**"over schema-passing pairs only, n=…"** and why `n` is never printed without
the schema-pass rate on the same screen.

### 7.3 Win-rate

```
win_rate(C vs B) = mean(s over all included pairs)
```

reported per candidate overall and per (candidate, task), each with `n` — the
number of included pairs — beside it. Never a percentage without its `n`. Because
of the §7.2 exclusions, `n` is the number of **schema-passing** pairs, and the
report's win-rate heading says so verbatim: *"over schema-passing pairs only,
n=…"*.

**Interval.** A percentile bootstrap over the pair scores: 10,000 resamples of
size `n` with replacement, 2.5th and 97.5th percentiles. The RNG is seeded
deterministically from the suite hash (`seed = int(suite_hash[:8], 16)`) so the
same inputs give the same interval. For `n < 8` the interval is suppressed and
the report prints `CI: n too small`. Bootstrap rather than a binomial interval
because pair scores are not Bernoulli — they take five values.

### 7.4 Cohen's kappa (judge vs human)

Over the labelled pairs that were actually run. Both raters are collapsed to the
same three categories:

- human: `prefer` as given;
- judge: `s > 0.5` → candidate, `s < 0.5` → baseline, `s == 0.5` → tie.

With `N` labelled pairs, `a_k` = count where the judge said category `k`, `b_k` =
count where the human said `k`, and `agree` = count where they matched:

```
po    = agree / N
pe    = Σ_k (a_k / N) * (b_k / N)          over k in {baseline, candidate, tie}
kappa = (po - pe) / (1 - pe)
```

Edge cases, all of which are reported rather than smoothed:

- `N == 0` → `kappa: null`, status `no_labels`.
- `pe == 1` (both raters used one and the same single category throughout) →
  `kappa: null`, status `undefined_pe_1`. Treated as **below the floor**: a judge
  that only ever says "tie" agrees with a human who only ever says "tie" 100% of
  the time and has demonstrated nothing.
- `po < pe` → kappa is negative and printed as such. Worse than chance is a real
  finding.
- `N < min_labels` → kappa is computed and printed but status is
  `insufficient_labels`.

**Every kappa value printed anywhere — report, JSON summary, MCP response, the
decision-log template — carries its Landis-Koch band name** (ruling O-2), e.g.
`0.43 (moderate)`. A bare number invites the reader to supply their own scale,
and most readers' scale for "0.43" is "nearly half right", which it is not.

| kappa | band |
|---|---|
| < 0.00 | `poor` |
| 0.00 – 0.20 | `slight` |
| 0.21 – 0.40 | `fair` |
| 0.41 – 0.60 | `moderate` |
| 0.61 – 0.80 | `substantial` |
| 0.81 – 1.00 | `almost perfect` |

Bands are upper-inclusive at each boundary as written above; `kappa: null` prints
as `null (undefined)`. `report.json` carries `kappa_band` next to `kappa`.

Plain agreement `po` is printed too, always next to kappa, never instead of it:
agreement is inflated whenever one category dominates, which it does the moment
your judge learns that ties are safe.

The 3×3 confusion matrix is written to `report.json` and rendered in `report.md`.
It is the fastest way to see *how* the judge is wrong — a judge that never says
"tie" when you do is a different problem from one that systematically prefers the
new model.

**Per-task agreement.** The overall kappa says *how much* the
judge agrees with you. It does not say *where* it stops. A judge tuned to one
task's taste can be excellent at that task and useless at the next one, and a
single number averages the second finding away. So alongside the overall kappa,
`calibration.per_task_agreement` carries one row per task that has at least one
usable label:

| Field | Meaning |
|---|---|
| `task` | task id |
| `n` | labels on this task that matched a scored pair |
| `agree` | of those, how many the judge got right |
| `agreement` | `agree / n` — plain agreement, never a substitute for kappa |
| `kappa` | Cohen's kappa for this task, or `null` when `n < min_n` |
| `kappa_band` | the §7.4 band name, `undefined` when kappa is null |
| `low_n` | `true` when kappa was suppressed for want of labels |
| `min_n` | the suppression threshold, **default 5** |
| `status` | `computed` · `low_n` · `undefined_pe_1` |
| `confusion` | this task's 3×3 matrix |

Rows are sorted by `agreement` ascending — worst first, the same discipline as
the per-task win-rate table of §9.2: the task the judge is furthest from you on
is the first row on screen.

Kappa is suppressed below `min_n` rather than printed small, because a
chance-corrected statistic over three observations is a number with a decimal
point and no content, and printing one invites it to be quoted. Those rows carry
plain agreement, flagged, and plain agreement over three labels is not evidence
either — it is a prompt to go label that task some more. The same rows are
rendered in `report.md` under the confusion matrix and in `report.html`, with
the same field names.

---

## 8. Calibration floor

`judge.calibration.min_kappa`, **default `0.40`**, with
`min_labels` default `10`.

`calibration.status` is one of `ok` · `below_floor` · `insufficient_labels` ·
`no_labels` · `undefined_pe_1`. `headline_eligible` is `true` only when status is
`ok`.

When `headline_eligible` is false:

- `report.md` prints the calibration block **before** the win-rate block, with a
  banner naming the reason;
- the win-rate section is titled **"Win-rates (UNCALIBRATED — not a headline)"**
  and every win-rate figure is suffixed `†`;
- the JSON summary and every MCP response carry `headline_eligible: false` and
  the reason string, so an agent reading the summary cannot report the number as
  established without also carrying the caveat;
- the exit code is still `0` unless `--fail-under-calibration` (exit 3) or
  `judge.calibration.on_below_floor: fail`.

Why 0.40: it is the conventional bottom of "fair-to-moderate" agreement, low
enough that a first suite with a dozen labels can plausibly clear it and high
enough that a coin-flip judge cannot. It is a default, not a law — it lives in
the suite file precisely so it can be argued with.

**Ruling O-2 (2026-08-23): the default stays 0.40, and two things follow from
that.** First, every kappa the tool prints carries its Landis-Koch band name
(§7.4), so a reader never has to decide unaided what 0.43 means. Second, **the
README recommends raising `min_kappa` to 0.60 — "substantial" — before publishing
any result as a claim.** 0.40 is the floor for *using a number yourself*, with
your own memory of how the labelling went; 0.60 is the floor for *telling someone
else a number*, where that memory does not travel. The tool ships permissive so
that a first suite is worth running twice; the recommendation exists so that the
first suite is not what ends up in a blog post.

---

## 9. Reports

### 9.1 Files

```
reports/<suite>/<run-id>/report.json     the record; every number in report.md
reports/<suite>/<run-id>/report.md       the human view, rendered from the JSON
reports/<suite>/<run-id>/report.html     the browser view + the labelling surface (S9.5)
reports/<suite>/<run-id>/answers.jsonl   one line per answer (text + usage)
reports/<suite>/<run-id>/pairs.jsonl     one line per pair (both passes + reasons)
reports/<suite>/latest.json              {"run_id": "...", "path": "..."}
```

`report.html` is written by `run`, not by `evalmine report`:
it embeds the answers themselves, which `report.json` does not carry, so it is
built from the run rather than re-rendered from the record.

`run-id = <UTC yyyymmddThhmmssZ>_<suite_hash[:8]>_<models_hash[:8]>` where
`suite_hash = sha256(raw suite file bytes)` and
`models_hash = sha256("\n".join(models_in_cli_order))`. Editing the suite — labels
included — changes the id, which is the intent.

`latest.json` is a pointer file, not a symlink, because CI runs on Windows.

### 9.2 `report.md` layout, in order

1. **Header** — suite name and description, suite hash, models (baseline marked),
   judge model, date, tool version, price table file, cache hit rate,
   `live_fraction`, total cost split answers/judge.
2. **Calibration** — status, kappa, plain agreement, `n` labels used, `n` labels
   ignored and why, the 3×3 confusion matrix, the §7.4 per-task agreement table
  , and the banner if not eligible. Deliberately above the
   win-rates.
3. **Win-rates** — one row per candidate: win-rate, CI, `n`, consistent
   wins/losses, ties, soft wins/losses, **flip rate**, excluded pairs.
4. **Per-model scorecard** — schema-pass rate (with `schema_mode`), parse fails,
   schema fails, execution-check pass rate with its `n` (§6.6), p50, p95
   (with n), cost this run, cost if uncached, cost per 1k calls.
5. **Per-task table** — for each task: per-model schema pass, per-model
   execution-check pass (only when some task has a check), per-candidate
   win-rate and `n`, median latency, cost. Sorted by candidate win-rate ascending
   so the tasks the new model is worst at are the first thing on screen.
6. **What changed** — §9.3, present only when a previous report exists.
7. **Failures and exclusions** — every excluded pair with its reason; every
   provider error; every failed or errored execution check with its exit code
   and first output line; every unparseable judge response.
8. **Reproduce** — the exact command, the price file, the cache directory, and a
   note of whether the run was live or cached.
9. **Decision log entry** — the §9.4 template, pre-filled with this run's
   numbers, in a copy-pasteable fenced block.

The report contains no adjectives. No "impressively", no "surprisingly", no
recommendation. Judgment goes in `DECISIONS.md`, worded from the owner's verdict.

### 9.3 "What changed"

Compared against the most recent previous report **for the same suite name**.

- If the suite hash differs, the section opens with **"Suite changed since
  <run-id> — task-level deltas are not comparable"** and shows only: tasks added
  / removed / modified (by per-task hash), models added / removed, price table
  change, and the labels diff. No metric deltas.
- If the suite hash matches, deltas for: win-rate per candidate (Δ and both
  values), kappa, plain agreement, schema-pass per model, p50, p95, cost per
  model, flip rate.
- **Movers**: every (candidate, task) whose win-rate moved by more than `0.15`
  (configurable `report.mover_threshold`), listed with old → new and `n`. A
  headline win-rate that did not move while three tasks moved 0.4 in opposite
  directions is the finding you would otherwise miss.
- Cache note: how much of the current report is unchanged cached answers, since a
  delta computed entirely from cache means only the *judge* or the *scoring*
  changed.

`evalmine compare A B` prints exactly this structure for any two reports.

### 9.4 Decision-log template

Appended by hand to `DECISIONS.md` (the tool prints it; the tool does not write
it — this is the one artefact a human must author):

```markdown
## <YYYY-MM-DD> — <suite> — <baseline> vs <candidate>

- **Run:** <run-id> · report: <path>
- **Question:** <the decision this run was supposed to inform>
- **Numbers:** win-rate <x> [<ci_lo>–<ci_hi>], n=<n>, flips <f> ·
  kappa <k> (agreement <po>, <N> labels) · schema pass <b>% → <c>% ·
  p95 <b>ms → <c>ms · cost/run $<b> → $<c>
- **Decision:** adopt | reject | inconclusive | adopt-for-<subset>
- **Why:** <two sentences, in terms of the numbers above>
- **What would change this:** <the result that would reverse the decision>
- **Not measured:** <what this suite does not cover that the decision assumes>
```

### 9.5 `report.html` and the labelling flow

One self-contained file per run: inline CSS, inline JS, no external asset, no
framework, no network call, light and dark from `prefers-color-scheme` with the
full token set defined in both. It opens off a `file://` path and it will still
open in five years, which is the whole reason it has no dependencies. Wide
tables scroll inside their own container; long answers scroll inside their pane;
the page body never scrolls sideways. Every string in it — answers included — is
HTML-escaped, because an answer is untrusted text.

It carries the §9.2 sections in the same order, and then the two things markdown
cannot do.

**Side-by-side answer pairs.** For each judged pair: the task prompt (folded),
and answer A and answer B in two columns. **The model names are hidden by
default**, behind a "reveal models" toggle, and so is the judge's verdict and
its one-line reason. You read the answers exactly as the judge read them —
blind. Reading "this one is from the new model" first is how a calibration set
ends up recording your expectations instead of your judgement, and a calibration
set that records your expectations makes kappa a measure of nothing.

Which answer lands in slot A is **randomised per pair, seeded from the pair id**
— `sha256(task|case|baseline|candidate)` — so it is stable across reloads and
across regenerations of the same run, and a half-finished labelling session
picks up where it left off instead of silently swapping sides underneath you.

**The labelling flow.** Under each pair: *Prefer A* · *Tie* · *Prefer B*, and an
optional one-line "why". A sticky footer counts progress and offers **copy
labels YAML**, which emits exactly the §5.5 entries — `task`, `case`,
`baseline`, `candidate`, `prefer`, `note` — ready to paste into the suite file.
Selections persist in `localStorage`, keyed by run id, every read and write
wrapped in `try`/`catch`; a browser with storage disabled loses persistence and
nothing else. Nothing is sent anywhere: there is no server.

**The mapping is computed in Python, not in the browser.** Going from "I clicked
Prefer A" back to `prefer: baseline` or `prefer: candidate` is the one piece of
this that can be wrong without anyone noticing — a flipped label does not crash,
it quietly poisons the kappa the tool's credibility rests on. So the generator
bakes a `{"A": ..., "B": ..., "tie": "tie"}` lookup table into each pair, and
the page's JavaScript only reads that table. The function that builds it is a
tested pure function; there is no untested mapping code anywhere in the feature.

---

## 10. Provider adapters

One `Protocol`, four implementations, no framework, no LLM library:

```python
class Adapter(Protocol):
    name: str          # "anthropic" | "openai" | "google" | "fake"
    version: int       # part of the cache key; bump when the request changes

    def complete(self, req: Request) -> Response: ...
```

`Request`: `model_id`, `system`, `prompt`, `max_tokens`, `temperature`, `top_p`,
`stop`, `schema`, `timeout_s`.
`Response`: `text`, `input_tokens`, `output_tokens`, `cached_input_tokens`,
`reasoning_tokens`, `latency_ms`, `finish_reason`, `schema_mode`.

Errors: `AdapterError(retryable: bool)`. Retries: up to 2, delay
`2 ** attempt + uniform(0, 1)` seconds, on timeouts, 429, and 5xx only. A
non-retryable error (401, 400) fails the run immediately — a missing key should
stop you in the first two seconds, not after forty calls.

Structured output: where the provider supports schema enforcement the adapter
uses it and reports `schema_mode: native`; otherwise it appends the schema to the
prompt with a fixed instruction and reports `schema_mode: prompted`. The mode is
in the cache key and in the report.

The fake adapter returns deterministic text derived from the cache key, with
deterministic token counts and latency. It is what the whole test suite runs
against, and `--fake` exposes it to users.

**Keys** come from `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY` only.
Nothing else is read. `.env.example` lists the names. Keys are never written to a
cache entry, a report, or a log line.

---

## 11. MCP surface

One module, `src/evalmine/mcp_server.py`, stdio transport, official Python MCP
SDK, console script `evalmine-mcp`. It calls the same `core.py` functions the CLI
calls and contains no evaluation logic of its own.

**Why it exists:** so an agent can run your evals *mid-task* — "before you swap
the model in this file, run the suite and tell me the win-rate" — instead of a
person reading a report afterwards. **Why three tools and not the CLI:** an
agent-facing surface should be the smallest set of verbs that supports the
decision, and every extra tool is another way to spend money you did not
authorise.

### 11.1 `run_suite`

```
in : suite_path: str
     models: list[str]                 (>= 2, first is baseline unless baseline given)
     max_cost: float | null            (USD; see 11.4)
     baseline: str | null
     no_cache: bool = false
out: {
  "run_id": str, "report_path": str, "report_md_path": str,
  "headline_eligible": bool,
  "calibration": {"status": str, "kappa": float|null, "agreement": float|null,
                  "n_labels": int, "reason": str|null},
  "per_model": [{"model": str, "role": "baseline"|"candidate",
                 "win_rate": float|null, "ci": [float,float]|null, "n_pairs": int,
                 "flip_rate": float|null, "schema_pass": float|null,
                 "p50_ms": int, "p95_ms": int, "cost_usd": float|null}],
  "totals": {"cost_usd": float, "cost_answers_usd": float, "cost_judge_usd": float,
             "live_calls": int, "cache_hits": int, "excluded_pairs": int},
  "warnings": [str]
}
```

It returns the summary and the paths. It **never** returns raw provider
responses — those stay in `answers.jsonl` on disk. A tool that streams every
answer back into an agent's context is a tool that costs the caller more than the
eval did, and it makes an eval harness into an exfiltration path for whatever is
in your prompts.

### 11.2 `compare`

```
in : report_a: str, report_b: str          (run-ids or paths)
out: the §9.3 delta object as JSON, plus {"comparable": bool, "reason": str|null}
```

### 11.3 `last_report`

```
in : suite_path: str
out: {"found": true, "run_id": str, "report_path": str, "generated_at": str,
      "summary": <the same shape as run_suite's summary fields>}
   | {"found": false}
```

Reads from disk only. Zero spend, always.

### 11.4 Cost enforcement

- The cap is a parameter of `core.run_suite(..., max_cost: float)`. The MCP tool
  is a thin caller; it cannot reach the provider layer any other way.
- If the agent supplies `max_cost`, it is used — but clamped to
  `EVALMINE_MCP_MAX_COST_CEILING` (default **$5.00**). A request above the ceiling is
  **refused**, not clamped-and-run.
- If the agent omits it, the effective cap is
  `min(suite.limits.max_cost_usd, EVALMINE_MCP_MAX_COST)`, default **$1.00** — lower
  than the CLI's $2.00, because the human at the CLI typed the number and the
  agent did not.
- An over-cap run returns a **structured refusal**, spends nothing, and is not an
  exception:
  `{"refused": true, "reason": "estimate_exceeds_cap", "estimate_usd": 3.41, "cap_usd": 1.00, "hint": "raise max_cost, cut --models, or run a subset"}`.
- The run is never silently truncated to fit the cap. Truncation produces a
  smaller number that looks like a complete one, which is worse than a refusal.
- `suite_path` must resolve inside `EVALMINE_MCP_SUITE_ROOT` (default: the server's
  working directory). A path outside it is refused. The server never invents a
  suite.

### 11.5 `.mcp.json.example`

```json
{
  "mcpServers": {
    "evalmine": {
      "command": "evalmine-mcp",
      "args": [],
      "env": {
        "ANTHROPIC_API_KEY": "${ANTHROPIC_API_KEY}",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}",
        "GOOGLE_API_KEY": "${GOOGLE_API_KEY}",
        "EVALMINE_MCP_MAX_COST": "1.00",
        "EVALMINE_MCP_MAX_COST_CEILING": "5.00",
        "EVALMINE_MCP_SUITE_ROOT": "${PWD}"
      }
    }
  }
}
```

---

## 12. Dependencies and Python policy

**Runtime, core:**

| Package | Constraint | Why |
|---|---|---|
| `PyYAML` | `>=6.0` | the suite format |
| `jsonschema` | `>=4.18` | Draft 2020-12 validation of both the suite and task outputs |
| `httpx` | `>=0.27` | one HTTP client for all three providers |

That is three. **No provider SDKs**: three SDKs is three dependency trees, three
release cadences, and three abstractions between the reader and the request. The
stated point of the adapter layer is that it is small enough to read in one
sitting, and a hand-written `POST` to a documented JSON endpoint is smaller than
an SDK wrapper. The cost of this choice is real and should be stated in the
README: when a provider changes its API, we notice by breaking, not by upgrading.

**Runtime, `[mcp]` extra:** `mcp >= 2.0`.

**Dev:** `pytest`, `ruff`. Nothing else.

**Python versions.** Ruling O-1 (2026-08-23): **the floor is 3.10 everywhere.**

- `pyproject.toml` sets `requires-python = ">=3.10"`. There is no separate
  syntax-discipline rule for the core: PEP 604 unions and builtin generics are
  available and may be used.
- **MCP stays an optional extra**, `pip install evalmine[mcp]`, declared
  `mcp>=2.0`. It needs no version marker and no test skips, because the extra's
  floor and the package's floor are now the same number.
- **CI matrix:** {ubuntu, macos, windows} × {3.10, 3.13}. Six legs, all of which
  can run every test including the MCP ones.

The reason: Python 3.9 reached end-of-life in October 2025, the official MCP
Python SDK requires 3.10+, and supporting 3.9 would have bought permanent syntax
discipline in the core plus an MCP surface that half the CI matrix could not
exercise — in exchange for users on an unsupported runtime.

---

## 13. Success criteria (verifiable)

Each of these is a test or a command whose output can be checked.

**Unit tests, all against pure functions in `metrics.py`, no network:**

1. **Pair scoring** covers all six rows of the §7.2 table, both orders of the
   asymmetric ones, and asserts the exact score.
2. **Flip counting**: a fixture with a known number of disagreeing pass-pairs
   produces exactly that `position_flips`, and the flip rate crosses the 0.30
   warning threshold at the boundary.
3. **Ties**: a suite where the judge ties every pair produces `win_rate == 0.5`,
   `n` equal to the pair count, and — because `pe == 1` against an all-tie human
   — `kappa: null` with status `undefined_pe_1` and `headline_eligible: false`.
4. **Win-rate arithmetic**: hand-computed expected value for a mixed fixture
   (consistent wins, soft wins, flips, exclusions), asserted to 1e-9.
5. **Bootstrap CI** is deterministic: the same fixture and suite hash produce a
   byte-identical interval across runs and across platforms; `n < 8` suppresses
   it.
6. **Kappa**: a worked 3-category example checked against a hand calculation;
   `po == pe` → 0; `po < pe` → negative; `N == 0` → null/`no_labels`; `pe == 1` →
   null/`undefined_pe_1`.
7. **Cost lookup**: known model computes the hand-checked figure including cached
   input tokens; **unknown model raises `UnknownModelError` before any adapter is
   constructed** (asserted by a fake adapter whose constructor fails the test if
   called); missing usage yields `null` cost and `cost_incomplete`.
8. **Cost guard**: a plan whose estimate exceeds the cap exits 4 with zero
   adapter calls; a plan under the cap proceeds; the live ceiling stops a run
   mid-way and writes `aborted_over_budget: true`.
9. **Cache key**: identical inputs → identical key; changing each of prompt,
   system, model, each param, schema, `schema_mode`, `adapter_version`, and
   `repeat` changes it; changing the date, run id, suite path, price table, and
   environment does **not**; the canonical-JSON encoding is stable across
   platforms and Python versions.
10. **Schema verdicts**: raw JSON passes; fenced JSON passes; prose-wrapped JSON
    with no fence is `parse_fail`; valid JSON violating the schema is
    `schema_fail`; a single-side failure **excludes** the pair from the win-rate
    with reason `baseline_schema_fail` / `candidate_schema_fail` and makes no
    judge call at all (asserted by a judge double that fails the test if called);
    the excluded pair appears in `excluded_pairs` and in the by-reason breakdown,
    and the win-rate's `n` counts only the surviving pairs.
11. **Suite loading**: unknown top-level key → exit 1; unknown `version` → exit 1;
    a `{{placeholder}}` with no var → exit 1 naming task, case, and variable;
    a label pointing at a non-existent case → exit 1; a label for a model pair
    not in the run → ignored and counted.

**Integration test:** `evalmine run examples/everyday-eight.yaml --models
fake/a,fake/b --fake` runs the whole example suite end to end against the fake
adapter and asserts on `report.json`: 20 cases × 2 models = 40 answers, 20 pairs,
40 judge passes, every expected key present, cost > 0 (the fake has a price row),
calibration block populated from the 12 labels, `report.md` renders, and a second
identical run is a 100% cache hit that produces an identical `report.json` modulo
`run_id` and timestamps. A third run with a modified suite produces a "what
changed" section that names the modified task.

**CI:** green on {ubuntu, macos, windows} × {3.10, 3.13}, every leg running every
test including the MCP ones. Three MCP tool tests against the fake adapter,
including one asserting a structured refusal when `max_cost` is below the
estimate and that no adapter call occurred.

**Repo hygiene:** MIT licence; README whose first 200 words are problem →
approach → result; a 30-second demo GIF; `CONTRIBUTING.md`; `.env.example`; a
secret-scan step in CI; no key anywhere in the repo or its history.

**The one that actually matters:** one **real** run on Hisham's own suite (~40
tasks with labels, supplied after this spec is approved) comparing two model
versions, producing the first report and the first `DECISIONS.md` entry. Until
that exists, this is a tool that has never been used.


---

## 14. Rulings — 2026-08-23

The three questions this spec stopped on, and how they were decided. Each ruling
is written into the section it governs; this is the index, not the authority.

**O-1. Python 3.9, or 3.10? → 3.10, everywhere.** `requires-python = ">=3.10"`,
one CI matrix of {ubuntu, macos, windows} × {3.10, 3.13}, no version skips, no
3.9 syntax discipline. MCP remains an optional extra (`evalmine[mcp]`) for
dependency-weight reasons, not version reasons. See §12.

**O-2. Is `min_kappa: 0.40` the right default? → Yes, kept — with a band name on
every kappa and a stricter recommendation for publication.** The floor stays
0.40; every printed kappa carries its Landis-Koch band (§7.4); the README
recommends 0.60 before any result is published as a claim. See §8.

**O-3. Should a schema failure be a loss, or an exclusion? → An exclusion.** Any
pair where either side is `parse_fail` or `schema_fail` is excluded from the
win-rate, with the reason recorded, counted in `excluded_pairs`, and broken out
by reason in the report. Schema-pass rate remains its own headline metric, and
the win-rate is labelled "over schema-passing pairs only, n=…". See §7.2, §7.3,
and §13.10.

**O-4. Should a failed execution check be a loss, an exclusion, or evidence? →
Evidence, with a rule.** A pair is still judged when a check fails; the judge and
the human both see the result and the output, and the judge is told a failed
check cannot beat a passed one. Excluding would throw away the most informative
pairs — the ones where one side works and the other does not — and an automatic
loss would hide a check whose fixture is wrong. An answer with several code
blocks is judged on its final block, with every earlier block on the record: the
retraction is evidence too. See §6.6.

**The name** was ruled at the same time: `evalmine`, replacing `nof1bench`. See
§1.1.

---

## 15. Build order (after approval)

1. `suite.py`, `metrics.py`, `prices.py`, `cache.py`, `adapters/fake.py` + every
   unit test in §13. No network, no keys, no spend.
2. `core.py`, `report.py`, `cli.py` + the integration test.
3. `adapters/{anthropic,openai,google}.py`, then the example suite run once
   against each to prove the plumbing — nothing more.
4. `mcp_server.py` + its three tests.
5. CI, README, GIF, `CONTRIBUTING.md`, `DECISIONS.md` header.
6. The real run on the real suite; the first decision-log entry, written with a
   human.
7. `docs/learning/how-it-works.md`, then `v0.1.0`.
