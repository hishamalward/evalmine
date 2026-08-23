# LinkedIn post draft (not posted)

Draft only. Every `[BRACKETED]` span is a number or a name that comes out of the first
real run on the real suite; nothing else should need rewriting once those land. The
shape is fixed: what I expected, what the benchmark said, what my tasks said, one
number, what I decided.

Do not post this until the run exists and the `DECISIONS.md` entry is written. A post
about a result you have not got is the exact failure this tool was built against.

---

I kept making model decisions on other people's numbers, so I built something to make
them on mine.

**What I expected.** [CANDIDATE MODEL] to be a straight upgrade on [BASELINE MODEL].
The release notes said so and nothing in my week argued with it.

**What the benchmark said.** [BENCHMARK NAME]: [BASELINE SCORE] to [CANDIDATE SCORE].
A clear win.

**What my tasks said.** I ran a suite of [N] tasks I actually do, [N CASES] cases,
[N LABELLED] of them carrying a preference I had written down before I saw any
results. Each pair of answers goes to a judge twice, with the two answers swapped the
second time, so the judge's habit of liking whichever answer it reads first cancels
out. Then the judge gets scored against my labels: Cohen's kappa [KAPPA] ([BAND]).
Below the floor it would have refused to print a headline number at all.

**One number.** [WIN RATE] over [N PAIRS] pairs, 95% CI [CI LOW] to [CI HIGH].
[ONE SENTENCE ON WHERE IT WENT: which task kinds it won, which it lost, and whether
schema-pass or latency moved with it.]

**What I decided.** [ADOPT / REJECT / ADOPT FOR A SUBSET / INCONCLUSIVE].
[TWO SENTENCES, IN TERMS OF THE NUMBERS ABOVE, INCLUDING COST: $[BASELINE COST] to
$[CANDIDATE COST] per run.] [ONE SENTENCE: what result would reverse this.]

The harness is public and MIT: [REPO URL]. It does one thing, which is to tell you
whether a model change helped you on your own work, and it will not give you a
headline win-rate when it cannot show that its judge agrees with you.

---

## Notes for filling this in

- Every number above appears in `report.md` and in the pre-filled decision-log block at
  the bottom of it. Copy them from there rather than retyping.
- Kappa never goes out without its band name. If the band is below "substantial", say
  so in the post rather than quietly rounding up: spec §8 recommends `min_kappa: 0.60`
  before a number is published as a claim, and this is publishing.
- If the real result is boring, post the boring result. "The benchmark was right and I
  switched" is a shorter post and a true one.
- If the result is inconclusive, that is also a post, and the honest version of it is
  about the suite rather than the models.
- Nothing about Listenality's harness, its tasks, its numbers, or its clients.
