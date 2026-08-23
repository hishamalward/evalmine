"""The pure metric functions. Spec S6-S7, success criteria S13.1-S13.6 and S13.10."""

from __future__ import annotations

import pytest

from evalmine.metrics import (
    FLIP_RATE_WARNING,
    bootstrap_ci,
    calibration_status,
    cohens_kappa,
    extract_json,
    format_kappa,
    judge_category,
    kappa_band,
    latency_stats,
    median,
    p95_nearest_rank,
    schema_pass_rate,
    schema_verdict,
    score_pair,
    seed_from_suite_hash,
    win_rate,
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


# --------------------------------------------------------------------------
# S13.1 pair scoring - all six rows, both orders of the asymmetric ones
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("pass1", "pass2", "expected_score", "expected_category"),
    [
        ("candidate", "candidate", 1.0, "consistent_win"),
        ("baseline", "baseline", 0.0, "consistent_loss"),
        ("tie", "tie", 0.5, "tie"),
        ("candidate", "tie", 0.75, "soft_win"),
        ("tie", "candidate", 0.75, "soft_win"),
        ("baseline", "tie", 0.25, "soft_loss"),
        ("tie", "baseline", 0.25, "soft_loss"),
        ("candidate", "baseline", 0.5, "flip"),
        ("baseline", "candidate", 0.5, "flip"),
    ],
)
def test_pair_scoring_table(pass1, pass2, expected_score, expected_category):
    score, category = score_pair(pass1, pass2)
    assert score == expected_score
    assert category == expected_category


def test_pair_scoring_rejects_an_unknown_verdict():
    with pytest.raises(ValueError):
        score_pair("winner", "tie")


# --------------------------------------------------------------------------
# S13.2 flip counting and the 0.30 warning boundary
# --------------------------------------------------------------------------


def test_flip_counting_and_warning_boundary():
    verdicts = [
        ("candidate", "baseline"),  # flip
        ("baseline", "candidate"),  # flip
        ("candidate", "baseline"),  # flip
        ("candidate", "candidate"),
        ("tie", "tie"),
        ("baseline", "baseline"),
        ("candidate", "tie"),
        ("baseline", "tie"),
        ("tie", "tie"),
        ("candidate", "candidate"),
    ]
    scored = [score_pair(a, b) for a, b in verdicts]
    flips = sum(1 for _, category in scored if category == "flip")
    assert flips == 3
    flip_rate = flips / len(scored)
    assert flip_rate == 0.30
    # 0.30 itself is not "above 0.30"; one more flip is.
    assert not flip_rate > FLIP_RATE_WARNING
    assert (flips + 1) / len(scored) > FLIP_RATE_WARNING


# --------------------------------------------------------------------------
# S13.3 an all-tie judge
# --------------------------------------------------------------------------


def test_all_ties_gives_half_a_win_rate_and_an_undefined_kappa():
    scores = [score_pair("tie", "tie")[0] for _ in range(12)]
    assert win_rate(scores) == 0.5
    assert len(scores) == 12

    kappa = cohens_kappa([("tie", "tie")] * 12)
    assert kappa["kappa"] is None
    assert kappa["status"] == "undefined_pe_1"

    calibration = calibration_status(kappa, min_kappa=0.40, min_labels=10)
    assert calibration["status"] == "undefined_pe_1"
    assert calibration["headline_eligible"] is False
    assert "single category" in calibration["reason"]


# --------------------------------------------------------------------------
# S13.4 win-rate arithmetic
# --------------------------------------------------------------------------


def test_win_rate_is_the_mean_of_included_pair_scores():
    scores = [1.0, 1.0, 0.75, 0.5, 0.5, 0.25, 0.0]  # excluded pairs simply absent
    assert win_rate(scores) == pytest.approx(4.0 / 7.0, abs=1e-9)
    assert win_rate([]) is None


# --------------------------------------------------------------------------
# S13.5 bootstrap CI
# --------------------------------------------------------------------------

MIXED = [1.0, 1.0, 0.75, 0.5, 0.5, 0.25, 0.0, 0.75, 1.0, 0.25]


def test_bootstrap_is_deterministic_for_a_given_seed():
    seed = seed_from_suite_hash("c4545e4e" + "0" * 56)
    first = bootstrap_ci(MIXED, seed)
    second = bootstrap_ci(list(MIXED), seed)
    assert first == second
    assert first is not None
    lo, hi = first
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_interval_is_pinned():
    """Guards against an accidental change to the resampling procedure."""
    assert bootstrap_ci(MIXED, seed=305419896) == (0.4, 0.8)


def test_bootstrap_depends_on_the_seed():
    """Two seeds can agree - at n=10 the resample means are coarse - but not always."""
    intervals = {bootstrap_ci(MIXED, seed=s) for s in range(8)}
    assert len(intervals) > 1


def test_bootstrap_brackets_the_point_estimate():
    lo, hi = bootstrap_ci(MIXED, seed=305419896)
    assert lo <= win_rate(MIXED) <= hi


def test_bootstrap_is_suppressed_below_eight_pairs():
    assert bootstrap_ci([1.0] * 7, seed=1) is None
    assert bootstrap_ci([1.0] * 8, seed=1) is not None


def test_seed_comes_from_the_first_eight_hex_of_the_suite_hash():
    assert seed_from_suite_hash("deadbeefcafe") == 0xDEADBEEF


# --------------------------------------------------------------------------
# S13.6 kappa
# --------------------------------------------------------------------------


def test_kappa_worked_example_matches_the_hand_calculation():
    pairs = (
        [("candidate", "candidate")] * 4
        + [("baseline", "baseline")] * 3
        + [("candidate", "baseline")] * 1
        + [("tie", "tie")] * 2
    )
    result = cohens_kappa(pairs)
    assert result["n"] == 10
    assert result["po"] == pytest.approx(0.9)
    # pe = (5/10)(4/10) + (3/10)(4/10) + (2/10)(2/10) = 0.36
    assert result["pe"] == pytest.approx(0.36)
    assert result["kappa"] == pytest.approx(0.54 / 0.64, abs=1e-12)
    assert result["kappa_band"] == "almost perfect"
    assert result["confusion"]["baseline"]["candidate"] == 1
    assert result["confusion"]["candidate"]["candidate"] == 4


def test_kappa_is_zero_when_agreement_equals_chance():
    result = cohens_kappa([("candidate", "candidate"), ("candidate", "baseline")])
    assert result["po"] == 0.5
    assert result["pe"] == pytest.approx(0.5)
    assert result["kappa"] == pytest.approx(0.0)


def test_kappa_is_negative_when_worse_than_chance():
    result = cohens_kappa([("candidate", "baseline"), ("baseline", "candidate")])
    assert result["kappa"] == pytest.approx(-1.0)
    assert result["kappa_band"] == "poor"


def test_kappa_with_no_labels():
    result = cohens_kappa([])
    assert result["kappa"] is None
    assert result["status"] == "no_labels"
    assert calibration_status(result, 0.40, 10)["headline_eligible"] is False


def test_kappa_rejects_an_unknown_category():
    with pytest.raises(ValueError):
        cohens_kappa([("winner", "tie")])


@pytest.mark.parametrize(
    ("value", "band"),
    [
        (-0.01, "poor"),
        (0.0, "slight"),
        (0.20, "slight"),
        (0.21, "fair"),
        (0.40, "fair"),
        (0.43, "moderate"),
        (0.60, "moderate"),
        (0.61, "substantial"),
        (0.80, "substantial"),
        (0.81, "almost perfect"),
        (1.0, "almost perfect"),
        (None, "undefined"),
    ],
)
def test_landis_koch_bands(value, band):
    assert kappa_band(value) == band


def test_kappa_is_always_printed_with_its_band():
    assert format_kappa(0.43) == "0.43 (moderate)"
    assert format_kappa(None) == "null (undefined)"


def test_judge_category_collapses_the_five_pair_scores():
    assert judge_category(1.0) == "candidate"
    assert judge_category(0.75) == "candidate"
    assert judge_category(0.5) == "tie"
    assert judge_category(0.25) == "baseline"
    assert judge_category(0.0) == "baseline"


def test_calibration_statuses_in_order_of_precedence():
    below = cohens_kappa(
        [("candidate", "candidate")] * 5
        + [("candidate", "baseline")] * 4
        + [("baseline", "baseline")] * 1
    )
    assert calibration_status(below, min_kappa=0.90, min_labels=5)["status"] == "below_floor"
    assert calibration_status(below, min_kappa=0.01, min_labels=5)["status"] == "ok"
    assert (
        calibration_status(below, min_kappa=0.01, min_labels=50)["status"]
        == "insufficient_labels"
    )
    assert calibration_status(below, min_kappa=0.01, min_labels=5)["headline_eligible"] is True


# --------------------------------------------------------------------------
# S13.10 schema verdicts
# --------------------------------------------------------------------------


def test_raw_json_passes():
    assert schema_verdict('{"ok": true}', SCHEMA).status == "pass"


def test_fenced_json_passes():
    text = 'Here you go:\n```json\n{"ok": false}\n```\n'
    assert schema_verdict(text, SCHEMA).status == "pass"


def test_unlabelled_fence_passes():
    assert schema_verdict('```\n{"ok": true}\n```', SCHEMA).status == "pass"


def test_prose_wrapped_json_without_a_fence_is_a_parse_fail():
    verdict = schema_verdict('Sure! The answer is {"ok": true} - hope that helps.', SCHEMA)
    assert verdict.status == "parse_fail"
    assert verdict.failed


def test_valid_json_violating_the_schema_is_a_schema_fail():
    verdict = schema_verdict('{"ok": "yes"}', SCHEMA)
    assert verdict.status == "schema_fail"
    assert verdict.error and "ok" in verdict.error
    assert verdict.failed


def test_no_schema_is_not_applicable():
    verdict = schema_verdict("free text", None)
    assert verdict.status == "not_applicable"
    assert not verdict.failed


def test_extract_json_prefers_the_whole_response():
    ok, value = extract_json('{"a": 1}')
    assert ok and value == {"a": 1}
    ok, value = extract_json("```json\n{\"a\": 2}\n```")
    assert ok and value == {"a": 2}
    assert extract_json("not json at all") == (False, None)


def test_schema_pass_rate_counts_the_two_failure_kinds_separately():
    verdicts = ["pass", "pass", "pass", "schema_fail", "parse_fail"]
    rate = schema_pass_rate(verdicts)
    assert rate == {"n": 5, "pass": 3, "schema_fail": 1, "parse_fail": 1, "rate": 0.6}
    assert schema_pass_rate([])["rate"] is None


# --------------------------------------------------------------------------
# S6.2 latency
# --------------------------------------------------------------------------


def test_median_of_even_and_odd_samples():
    assert median([3, 1, 2]) == 2
    assert median([4, 1, 2, 3]) == 2.5
    assert median([]) is None


def test_p95_is_nearest_rank():
    values = list(range(1, 21))  # n=20 -> ceil(0.95*20) = 19 -> 19
    assert p95_nearest_rank(values) == 19
    # n < 20: nearest rank is the maximum, which is why n is always printed
    assert p95_nearest_rank([5, 1, 9]) == 9
    assert p95_nearest_rank([]) is None


def test_latency_stats_carry_the_live_fraction():
    stats = latency_stats([100, 200, 300, 400], live_count=1)
    assert stats["p50_ms"] == 250
    assert stats["live_fraction"] == 0.25
    assert latency_stats([], 0)["live_fraction"] is None
