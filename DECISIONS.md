# Decisions

The log of model decisions made with this tool. One entry per decision, newest last.

This file exists because the output of an eval is not a number, it is a decision you
have to defend in six months, to someone who was not there and cannot re-derive it
from a directory of reports. The report says what the run measured; **this file says
what you did about it and why.** They are deliberately separate: `report.md` contains
no adjectives and makes no recommendation, and this file is where the judgement goes.

Entries are written by a person. `evalmine run` prints a pre-filled template at the
bottom of every report — the numbers, the run id and the report path are already in
it — and you paste that here and fill in the four prose fields. The tool never writes
to this file itself.

Fill in every field, including the awkward two. **"What would change this"** is the
result that would reverse the decision, written *before* you have it, so that six
months from now you know whether the new report is that result or just a different
one. **"Not measured"** is what the suite does not cover but the decision assumes; it
is where the honest caveat lives, and it is usually the field that turns out to matter.

If the decision is `inconclusive`, write the entry anyway. A run that did not settle
anything is worth knowing about, not least because it is evidence about the suite.

## Template

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

## Entries

## 2026-08-23 — my-tasks (private) — anthropic/claude-opus-5 vs anthropic/claude-sonnet-5

*Drafted from the run by the session that ran it; the owner's wording is pending. The
suite and its reports are private and live outside this repository; this entry carries
counts and numbers only.*

- **Run:** 20260824T001654Z_cb783334_b0ab3852 · report: private (not in this repository)
- **Question:** a shakedown — does the harness survive a real suite against current
  models, and is Sonnet 5 good enough to replace Opus 5 on my everyday tasks?
- **Numbers:** win-rate 0.611 [0.43–0.79] (UNCALIBRATED, 0 labels), n=18, flips 2 ·
  kappa undefined (0 labels) · schema pass 100% → 100% · exec pass 50% → 75% (n=4) ·
  p95 9848ms → 5771ms · cost/run $0.166 → $0.067 (if uncached)
- **Decision:** inconclusive
- **Why:** No pair was labelled, so the judge is uncalibrated and the win-rate is not a
  number I may quote. The run's yield was three harness fixes — sampling parameters
  rejected on the Claude 5 models, default-on thinking spending the answer budget
  (three of the first eight answers, 18 to 506 visible characters), and execution
  checks for the code task — not a model decision.
- **What would change this:** 15–20 labels spread across the tasks with kappa at or
  above 0.40, on a suite built for a decision I actually have to make, and a win-rate
  whose interval excludes 0.5.
- **Not measured:** thinking-on behaviour (the adapter disables it on both sides);
  anything with tools or multiple turns; the 18 cases trimmed out; whether a Sonnet 4.6
  judge can grade the code and reasoning tasks at all — the per-task kappa would say.
