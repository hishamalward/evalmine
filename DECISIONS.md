# Decisions

The log of model decisions made with this tool. One entry per decision, newest last.

This file exists because the output of an eval is not a number, it is a decision you
have to defend in six months, to someone who was not there and cannot re-derive it
from a directory of reports. The report says what the run measured; **this file says
what you did about it and why.** They are deliberately separate: `report.md` contains
no adjectives and makes no recommendation, and this file is where the judgement goes.

Entries are worded from the owner's verdict. `evalmine run` prints a pre-filled
template at the bottom of every report — the numbers, the run id and the report path
are already in it. The owner says what they decided and why; the four prose fields are
filled from that, by the owner or by an agent working with them. `evalmine` itself
never writes to this file.

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

*No entries yet.*
