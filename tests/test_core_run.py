"""run_suite: planning, the two-stage cost guard, exclusions, the cache.

Spec S6.4, S7.2; success criteria S13.7 and S13.8. Every call in here goes to
the fake adapter or to a double; nothing contacts a provider.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import yaml
from conftest import EXAMPLE_SUITE, PRICES_DIR

from evalmine.adapters import FakeAdapter, FakeFailure, Request, Response
from evalmine.core import CostRefused, RunError, UsageError, run_suite
from evalmine.prices import UnknownModelError

MODELS = ["anthropic/claude-haiku-4-5", "google/gemini-2.5-flash"]
WHEN = datetime(2026, 8, 23, 9, 14, 2, tzinfo=timezone.utc)


class ExplodingAdapter:
    """Fails the test if it is ever constructed."""

    name = "fake"
    version = 1

    def __init__(self, *args, **kwargs):
        raise AssertionError("an adapter was constructed when none should have been")


class SilentAdapter(FakeAdapter):
    """Constructible, but fails the test if it is ever called."""

    def complete(self, req: Request) -> Response:
        raise AssertionError("a provider call was made when none should have been")


def run(tmp_path, **overrides):
    kwargs = dict(
        suite_path=EXAMPLE_SUITE,
        models=list(MODELS),
        fake=True,
        prices_dir=PRICES_DIR,
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "reports",
        now=WHEN,
        retry_sleep=lambda _: None,
        command="evalmine run (test)",
    )
    kwargs.update(overrides)
    suite_path = kwargs.pop("suite_path")
    models = kwargs.pop("models")
    return run_suite(suite_path, models, **kwargs)


# --------------------------------------------------------------------------
# arguments
# --------------------------------------------------------------------------


def test_at_least_two_models_are_required(tmp_path):
    with pytest.raises(UsageError):
        run(tmp_path, models=["fake/a"])


def test_duplicate_models_are_rejected(tmp_path):
    with pytest.raises(UsageError):
        run(tmp_path, models=["fake/a", "fake/a"])


def test_baseline_must_be_one_of_the_models(tmp_path):
    with pytest.raises(UsageError):
        run(tmp_path, baseline="fake/c")


def test_baseline_defaults_to_the_first_model(tmp_path):
    result = run(tmp_path)
    assert result.report["baseline"] == MODELS[0]
    assert result.report["candidates"] == [MODELS[1]]


# --------------------------------------------------------------------------
# S13.7 unknown model, before any adapter exists
# --------------------------------------------------------------------------


def test_unknown_model_raises_before_any_adapter_is_constructed(tmp_path):
    with pytest.raises(UnknownModelError) as exc:
        run(
            tmp_path,
            models=["anthropic/claude-haiku-4-5", "openai/does-not-exist"],
            adapter_factory=lambda provider, fake: ExplodingAdapter(),
        )
    assert "openai/does-not-exist" in str(exc.value)


def test_unknown_judge_model_is_also_caught_before_adapters(tmp_path):
    with pytest.raises(UnknownModelError):
        run(
            tmp_path,
            judge_model="openai/no-such-judge",
            adapter_factory=lambda provider, fake: ExplodingAdapter(),
        )


def test_missing_usage_gives_a_null_cost_and_flags_the_report(tmp_path):
    result = run(tmp_path, adapter_factory=lambda p, f: FakeAdapter(drop_usage=True))
    assert result.report["cost_incomplete"] is True
    assert result.report["totals"]["missing_usage_calls"] == 40
    assert result.report["per_model"][MODELS[0]]["cost"]["incomplete"] is True


# --------------------------------------------------------------------------
# S13.8 the cost guard
# --------------------------------------------------------------------------


def test_a_plan_over_the_cap_is_refused_with_no_calls(tmp_path):
    with pytest.raises(CostRefused) as exc:
        run(tmp_path, max_cost=0.0001, adapter_factory=lambda p, f: SilentAdapter())
    refusal = exc.value
    assert refusal.estimate > refusal.cap
    assert refusal.breakdown
    assert "Nothing was spent" in str(refusal)
    assert not (tmp_path / "reports").exists()


def test_a_plan_under_the_cap_proceeds(tmp_path):
    result = run(tmp_path, max_cost=5.00)
    assert result.exit_code == 0
    assert result.report["totals"]["answers"] == 40


def test_the_suite_limit_is_the_cap_when_no_flag_is_given(tmp_path):
    result = run(tmp_path)
    assert result.report["run"]["max_cost_usd"] == 1.50  # limits.max_cost_usd


def test_max_cost_overrides_the_suite_limit(tmp_path):
    result = run(tmp_path, max_cost=3.0)
    assert result.report["run"]["max_cost_usd"] == 3.0


class ExpensiveAdapter(FakeAdapter):
    """A provider that bills far more than the pre-flight heuristic predicts."""

    def complete(self, req: Request) -> Response:
        response = super().complete(req)
        return Response(
            text=response.text,
            input_tokens=5_000_000,
            output_tokens=5_000_000,
            cached_input_tokens=0,
            reasoning_tokens=0,
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
            schema_mode=response.schema_mode,
        )


def test_the_live_ceiling_aborts_mid_run_and_writes_a_partial_report(tmp_path):
    result = run(tmp_path, max_cost=20.0, adapter_factory=lambda p, f: ExpensiveAdapter())
    assert result.exit_code == 5
    report = result.report
    assert report["aborted_over_budget"] is True
    assert report["totals"]["answers"] < 40
    assert report["totals"]["pairs"] == 0
    assert result.report_path.is_file()
    assert any("aborted mid-run" in w for w in report["warnings"])


# --------------------------------------------------------------------------
# provider failures
# --------------------------------------------------------------------------


def test_a_non_retryable_provider_error_stops_the_run(tmp_path):
    with pytest.raises(RunError):
        run(
            tmp_path,
            adapter_factory=lambda p, f: FakeAdapter(failures=[FakeFailure("auth_error")]),
        )


def test_a_persistent_retryable_error_excludes_the_pair_and_flags_the_run(tmp_path):
    def factory(provider, fake):
        return FakeAdapter(
            failures=[FakeFailure("server_error", prompt_contains="Rewrite this commit message")]
        )

    result = run(tmp_path, adapter_factory=factory)
    report = result.report
    assert report["errors"], "the failed calls should be recorded"
    reasons = report["win_rates"][MODELS[1]]["excluded_by_reason"]
    assert reasons["provider_error"] == 3  # three cases in changelog-line
    assert any("provider call failed" in w for w in report["warnings"])


# --------------------------------------------------------------------------
# ruling O-3: schema failures exclude, they do not lose
# --------------------------------------------------------------------------


def test_a_schema_failure_excludes_the_pair_rather_than_losing_it(tmp_path):
    def factory(provider, fake):
        # only the candidate's answers violate their schema
        return FakeAdapter(failures=[FakeFailure("schema_violation", model="gemini-2.5-flash")])

    result = run(tmp_path, adapter_factory=factory)
    report = result.report
    win = report["win_rates"][MODELS[1]]
    # 8 schema cases across three schema-carrying tasks
    assert win["excluded_by_reason"]["candidate_schema_fail"] == 8
    assert win["n"] == 12
    assert report["per_model"][MODELS[1]]["schema"]["rate"] == 0.0
    assert report["per_model"][MODELS[0]]["schema"]["rate"] == 1.0
    # a schema failure is not a loss: no pair scored 0.0 for that reason
    assert report["totals"]["excluded_pairs"] == 8


def test_a_schema_failure_costs_no_judge_call(tmp_path):
    both_fail = run(
        tmp_path,
        adapter_factory=lambda p, f: FakeAdapter(failures=[FakeFailure("schema_violation")]),
    )
    report = both_fail.report
    assert report["win_rates"][MODELS[1]]["excluded_by_reason"]["both_schema_fail"] == 8
    # 20 pairs, 8 excluded before judging -> 12 judged pairs, 24 passes
    assert report["totals"]["judge_passes"] == 24


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------


def test_labels_for_models_not_in_the_run_are_ignored_and_counted(tmp_path):
    result = run(tmp_path, models=["fake/a", "fake/b"], judge_model="fake/judge")
    cal = result.report["calibration"]
    assert cal["n_labels"] == 0
    assert cal["n_labels_ignored"] == 12
    assert cal["status"] == "no_labels"
    assert result.report["headline_eligible"] is False
    assert all("not in this run" in item["why"] for item in cal["ignored_labels"])


def test_labels_that_match_the_run_are_used(tmp_path):
    result = run(tmp_path)
    cal = result.report["calibration"]
    assert cal["n_labels"] == 12
    assert cal["n_labels_ignored"] == 0
    assert cal["kappa"] is not None
    assert cal["kappa_band"] in ("poor", "slight", "fair", "moderate", "substantial")


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------


def test_a_second_run_is_a_complete_cache_hit(tmp_path):
    first = run(tmp_path)
    second = run(tmp_path, now=datetime(2026, 8, 23, 10, 0, 0, tzinfo=timezone.utc))
    assert first.report["totals"]["cache_hit_rate"] == 0.0
    assert second.report["totals"]["cache_hit_rate"] == 1.0
    assert second.report["totals"]["cost_usd"] == 0.0
    assert second.report["totals"]["cost_if_uncached_usd"] > 0
    assert second.report["per_model"][MODELS[0]]["latency"]["live_fraction"] == 0.0
    # latency is stored with the answer, so a cached rerun reproduces it
    assert (
        second.report["per_model"][MODELS[0]]["latency"]["p50_ms"]
        == first.report["per_model"][MODELS[0]]["latency"]["p50_ms"]
    )


def test_no_cache_ignores_entries_but_still_writes(tmp_path):
    run(tmp_path)
    again = run(tmp_path, no_cache=True, now=datetime(2026, 8, 23, 11, 0, tzinfo=timezone.utc))
    assert again.report["totals"]["cache_hit_rate"] == 0.0
    third = run(tmp_path, now=datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc))
    assert third.report["totals"]["cache_hit_rate"] == 1.0


# --------------------------------------------------------------------------
# warnings
# --------------------------------------------------------------------------


def test_a_judge_under_test_is_warned_about(tmp_path):
    result = run(tmp_path, judge_model=MODELS[0])
    assert result.report["judge"]["under_test"] is True
    assert any("self-preference" in w for w in result.report["warnings"])


def test_an_unverified_price_table_is_warned_about(tmp_path):
    # The shipped table is verified (S6.3); build an unverified one here so
    # the warning path itself stays covered.
    prices_path = tmp_path / "prices-2026-08-23.yaml"
    prices_path.write_text(
        "pinned: 2026-08-23\n"
        "verified: false\n"
        "models:\n"
        "  - {model: anthropic/claude-haiku-4-5, input_per_mtok: 1, output_per_mtok: 5}\n"
        "  - {model: google/gemini-2.5-flash, input_per_mtok: 0.3, output_per_mtok: 2.5}\n"
        "  - {model: anthropic/claude-sonnet-4-6, input_per_mtok: 3, output_per_mtok: 15}\n",
        encoding="utf-8",
    )
    result = run(tmp_path, prices_path=prices_path, prices_dir=None)
    assert result.report["prices_verified"] is False
    assert any("verified: false" in w for w in result.report["warnings"])


def test_the_shipped_price_table_is_verified_and_warns_about_nothing(tmp_path):
    result = run(tmp_path)
    assert result.report["prices_verified"] is True
    assert not any("verified: false" in w for w in result.report["warnings"])


def test_repeats_multiply_the_pairs(tmp_path):
    result = run(tmp_path, repeats=2, max_cost=5.0)
    assert result.report["totals"]["answers"] == 80
    assert result.report["totals"]["pairs"] == 40


# --------------------------------------------------------------------------
# S6.6 execution checks, end to end on the fake adapter
# --------------------------------------------------------------------------


def _checked_suite(tmp_path):
    example = yaml.safe_load(EXAMPLE_SUITE.read_text(encoding="utf-8"))
    doc = {
        "suite": "checked",
        "version": 1,
        "judge": example["judge"],
        "tasks": [
            {
                "id": "code",
                "kind": "code",
                "prompt": "Write {{what}}.",
                "check": {"timeout_s": 5},
                "cases": [
                    {
                        "id": "passes",
                        "vars": {"what": "a"},
                        "check": {"run": 'test -s "$ANSWER" && echo ran-ok'},
                    },
                    {
                        "id": "fails",
                        "vars": {"what": "b"},
                        "check": {"run": "echo broken >&2; exit 7"},
                    },
                    {"id": "unchecked", "vars": {"what": "c"}},
                ],
            }
        ],
    }
    path = tmp_path / "checked.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def test_execution_checks_reach_the_answers_the_report_and_the_html(tmp_path):
    result = run(tmp_path, suite_path=_checked_suite(tmp_path))
    report = result.report

    answers = {}
    for line in (result.report_path.parent / "answers.jsonl").read_text().splitlines():
        record = json.loads(line)
        answers[(record["case"], record["model"])] = record
    for model in MODELS:
        passed = answers[("passes", model)]
        assert passed["check_status"] == "pass"
        assert passed["check_exit"] == 0
        assert passed["check_output"] == "ran-ok"
        failed = answers[("fails", model)]
        assert failed["check_status"] == "fail"
        assert failed["check_exit"] == 7
        assert failed["check_output"] == "[stderr]\nbroken"
        unchecked = answers[("unchecked", model)]
        assert unchecked["check_status"] == "not_applicable"
        assert unchecked["check_exit"] is None

    for model in MODELS:
        assert report["per_model"][model]["check"] == {
            "n": 2, "pass": 1, "fail": 1, "error": 0, "rate": 0.5, "multi_block": 0,
        }
        assert answers[("passes", model)]["check_blocks"][0]["status"] == "pass"
        assert answers[("unchecked", model)]["check_blocks"] == []
        assert report["per_task"][0]["models"][model]["check"]["rate"] == 0.5
    assert report["per_task"][0]["has_check"] is True
    assert {(f["case"], f["model"]) for f in report["check_failures"]} == {
        ("fails", m) for m in MODELS
    }
    assert report["check_failures"][0]["exit_code"] == 7

    md = result.report_md_path.read_text(encoding="utf-8")
    assert "exec pass" in md
    assert "Execution check failures" in md
    assert "50.0% (n=2)" in md

    html = result.report_html_path.read_text(encoding="utf-8")
    assert "exec: PASS (exit 0)" in html
    assert "exec: FAIL (exit 7)" in html
    assert "Execution output" in html
    by_case = {p["case"]: p for p in result.pair_view}
    assert by_case["passes"]["a_check"]["status"] == "pass"
    assert by_case["unchecked"]["a_check"] is None


def test_a_check_never_excludes_a_pair(tmp_path):
    result = run(tmp_path, suite_path=_checked_suite(tmp_path))
    assert result.report["totals"]["excluded_pairs"] == 0
    assert result.report["totals"]["pairs"] == 3


def test_an_unchecked_suite_reports_no_check_numbers(tmp_path):
    result = run(tmp_path)
    for model in MODELS:
        assert result.report["per_model"][model]["check"]["n"] == 0
        assert result.report["per_model"][model]["check"]["rate"] is None
    assert result.report["check_failures"] == []
    assert "Execution check failures" not in result.report_md_path.read_text(encoding="utf-8")
