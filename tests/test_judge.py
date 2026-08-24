"""The two-pass judge protocol. Spec S7.1-S7.2, ruling O-3.

Every test here runs against a judge double, so nothing is called and nothing is
spent.
"""

from __future__ import annotations

import json

import pytest

from evalmine.judge import (
    BASELINE_SCHEMA_FAIL,
    BOTH_SCHEMA_FAIL,
    CANDIDATE_SCHEMA_FAIL,
    JUDGE_UNPARSEABLE,
    Judge,
    JudgeCall,
    build_prompt,
    compose_rubric,
    exclusion_reason,
    verdict_from_winner,
)
from evalmine.suite import CallParams, Case, JudgeConfig, Task

CONFIG = JudgeConfig(model="fake/judge", rubric="Prefer the shippable answer.")

TASK = Task(
    id="t",
    prompt_template="Rewrite {{x}}.",
    kind="rewrite",
    schema=None,
    rubric="Also: under 20 words.",
    judge=True,
    params=CallParams(),
    cases=(Case(id="c", vars={"x": "this"}, prompt="Rewrite this.", system=None),),
    hash="0" * 64,
)


class ScriptedJudge:
    """Returns the scripted texts in order, and records every prompt it saw."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts: list[str] = []
        self.bypassed: list[bool] = []

    def __call__(self, prompt, system, bypass_cache=False):
        self.prompts.append(prompt)
        self.bypassed.append(bypass_cache)
        text = self.texts.pop(0)
        return JudgeCall(text=text, cost=0.001, cached=False, latency_ms=100)


def verdict(winner: str) -> str:
    return json.dumps({"winner": winner, "reason": "because"})


def run(texts):
    call = ScriptedJudge(texts)
    judge = Judge(CONFIG, call)
    pair = judge.judge_pair(
        task=TASK,
        case_id="c",
        repeat=0,
        baseline="fake/a",
        candidate="fake/b",
        baseline_prompt="Rewrite this.",
        baseline_text="BASELINE ANSWER",
        candidate_text="CANDIDATE ANSWER",
    )
    return pair, call


def test_the_two_passes_swap_the_answers():
    pair, call = run([verdict("1"), verdict("1")])
    first, second = call.prompts
    assert first.index("BASELINE ANSWER") < first.index("CANDIDATE ANSWER")
    assert second.index("CANDIDATE ANSWER") < second.index("BASELINE ANSWER")
    assert pair.judge_calls == 2


def test_the_judge_never_sees_a_model_name_or_a_price():
    _, call = run([verdict("tie"), verdict("tie")])
    for prompt in call.prompts:
        assert "fake/a" not in prompt and "fake/b" not in prompt
        assert "baseline" not in prompt.lower().split("=== rubric ===")[0]
        assert "$" not in prompt


def test_winner_one_in_both_orders_is_a_flip_scored_half():
    pair, _ = run([verdict("1"), verdict("1")])
    # order 1 winner "1" = baseline; order 2 winner "1" = candidate
    assert [p.verdict for p in pair.passes] == ["baseline", "candidate"]
    assert pair.category == "flip"
    assert pair.score == 0.5
    assert pair.is_flip


def test_a_consistent_candidate_win_scores_one():
    pair, _ = run([verdict("2"), verdict("1")])
    assert [p.verdict for p in pair.passes] == ["candidate", "candidate"]
    assert pair.score == 1.0
    assert pair.category == "consistent_win"


def test_a_consistent_baseline_win_scores_zero():
    pair, _ = run([verdict("1"), verdict("2")])
    assert pair.score == 0.0
    assert pair.category == "consistent_loss"


def test_two_ties_score_half():
    pair, _ = run([verdict("tie"), verdict("tie")])
    assert pair.score == 0.5
    assert pair.category == "tie"


def test_a_soft_win():
    pair, _ = run([verdict("2"), verdict("tie")])
    assert pair.score == 0.75
    assert pair.category == "soft_win"


def test_verdict_mapping_both_directions():
    assert verdict_from_winner("1", 1) == "baseline"
    assert verdict_from_winner("2", 1) == "candidate"
    assert verdict_from_winner("1", 2) == "candidate"
    assert verdict_from_winner("2", 2) == "baseline"
    assert verdict_from_winner("tie", 1) == "tie"
    assert verdict_from_winner("tie", 2) == "tie"


def test_an_unparseable_pass_is_retried_once_bypassing_the_cache():
    pair, call = run(["not json at all", verdict("2"), verdict("1")])
    assert call.bypassed == [False, True, False]
    assert pair.score == 1.0
    assert pair.passes[0].retried is True
    assert pair.judge_calls == 3


def test_an_unparseable_pass_that_stays_unparseable_excludes_the_pair():
    pair, _ = run(["nope", "still nope", verdict("1"), verdict("1")])
    assert pair.excluded is True
    assert pair.reason == JUDGE_UNPARSEABLE
    assert pair.score is None


def test_a_verdict_outside_the_enum_is_treated_as_unparseable():
    bad = json.dumps({"winner": "answer 1", "reason": "x"})
    pair, _ = run([bad, bad, verdict("1"), verdict("1")])
    assert pair.excluded is True
    assert pair.reason == JUDGE_UNPARSEABLE


def test_judge_cost_accumulates_over_both_passes():
    pair, _ = run([verdict("1"), verdict("2")])
    assert pair.judge_cost == pytest.approx(0.002)


def test_exclusion_reasons_by_which_side_failed():
    assert exclusion_reason(False, False) is None
    assert exclusion_reason(True, False) == BASELINE_SCHEMA_FAIL
    assert exclusion_reason(False, True) == CANDIDATE_SCHEMA_FAIL
    assert exclusion_reason(True, True) == BOTH_SCHEMA_FAIL


def test_rubrics_compose_suite_then_task():
    composed = compose_rubric(CONFIG, TASK)
    assert composed.startswith("Prefer the shippable answer.")
    assert composed.endswith("Also: under 20 words.")


def test_prompt_carries_the_task_the_rubric_and_both_answers():
    prompt = build_prompt("Do the thing.", "The rubric.", "one", "two")
    for fragment in ("Do the thing.", "The rubric.", "=== ANSWER 1 ===", "=== ANSWER 2 ==="):
        assert fragment in prompt


# --------------------------------------------------------------------------
# S6.6 execution checks in the judge prompt
# --------------------------------------------------------------------------

PASSED = {"status": "pass", "exit_code": 0, "output": "ok-out"}
FAILED = {"status": "fail", "exit_code": 1, "output": "boom"}


def test_prompt_has_no_execution_section_without_checks():
    assert "EXECUTION CHECK" not in build_prompt("t", "r", "one", "two")


def test_prompt_carries_both_execution_results_and_the_rule():
    prompt = build_prompt("t", "r", "one", "two", PASSED, FAILED)
    assert "=== EXECUTION CHECK ===" in prompt
    assert "Answer 1: PASS (exit 0)" in prompt
    assert "ok-out" in prompt
    assert "Answer 2: FAIL (exit 1)" in prompt
    assert "boom" in prompt
    assert "cannot beat" in prompt
    assert prompt.index("=== ANSWER 2 ===") < prompt.index("=== EXECUTION CHECK ===")
    assert prompt.index("=== EXECUTION CHECK ===") < prompt.index("=== YOUR VERDICT ===")


def test_an_unchecked_side_is_labelled_not_checked():
    prompt = build_prompt("t", "r", "one", "two", PASSED, None)
    assert "Answer 2: not checked" in prompt


def test_a_timed_out_check_has_no_exit_code_in_the_prompt():
    timed_out = {"status": "fail", "exit_code": None, "output": ""}
    prompt = build_prompt("t", "r", "one", "two", timed_out, None)
    assert "Answer 1: FAIL\n" in prompt
    assert "(no output)" in prompt


def test_several_blocks_are_reported_in_order_with_the_final_as_verdict():
    fumbled = {
        "status": "pass", "exit_code": 0, "output": "5 a.txt",
        "blocks": [
            {"index": 1, "status": "fail", "exit_code": 1, "output": "1 a.txt"},
            {"index": 2, "status": "pass", "exit_code": 0, "output": "5 a.txt"},
        ],
    }
    prompt = build_prompt("t", "r", "one", "two", fumbled, PASSED)
    assert "Answer 1: PASS (exit 0)" in prompt
    assert "Answer 1 submitted 2 code blocks; the verdict above is the final block's." in prompt
    assert "block 1: FAIL (exit 1), block 2: PASS (exit 0)" in prompt
    assert "not the same as being right the first time" in prompt
    assert "Answer 2 submitted" not in prompt


def test_the_swapped_pass_swaps_the_checks_with_the_answers():
    from evalmine.judge import Judge, JudgeCall
    from evalmine.suite import CallParams, JudgeConfig, Task

    prompts: list[str] = []

    def call(prompt: str, system: str | None, bypass_cache: bool = False) -> JudgeCall:
        prompts.append(prompt)
        return JudgeCall(text='{"winner": "1", "reason": "r"}')

    task = Task(
        id="code", prompt_template="p", kind="code", schema=None, rubric=None,
        judge=True, params=CallParams(), cases=(), hash="h",
    )
    judge = Judge(JudgeConfig(model="fake/judge", rubric="r"), call)
    judge.judge_pair(
        task=task, case_id="c", repeat=0, baseline="b", candidate="c",
        baseline_prompt="p", baseline_text="base-text", candidate_text="cand-text",
        baseline_check=PASSED, candidate_check=FAILED,
    )
    first, second = prompts
    assert first.index("base-text") < first.index("cand-text")
    assert "Answer 1: PASS (exit 0)" in first and "Answer 2: FAIL (exit 1)" in first
    assert second.index("cand-text") < second.index("base-text")
    assert "Answer 1: FAIL (exit 1)" in second and "Answer 2: PASS (exit 0)" in second
