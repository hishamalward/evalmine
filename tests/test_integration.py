"""The end-to-end run of examples/everyday-eight.yaml against the fake adapter.

Spec S13, "Integration test". Nothing here contacts a provider: --fake routes
every model string to the deterministic fake adapter.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import EXAMPLE_SUITE, PRICES_DIR

from evalmine.core import run_suite
from evalmine.report import render_markdown

MODELS = ["anthropic/claude-haiku-4-5", "google/gemini-2.5-flash"]
START = datetime(2026, 8, 23, 9, 0, 0, tzinfo=timezone.utc)


def go(tmp_path, minutes: int = 0, suite=EXAMPLE_SUITE, **overrides):
    return run_suite(
        suite,
        list(MODELS),
        fake=True,
        prices_dir=PRICES_DIR,
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "reports",
        now=START + timedelta(minutes=minutes),
        retry_sleep=lambda _: None,
        command="evalmine run examples/everyday-eight.yaml --models "
        + ",".join(MODELS)
        + " --fake",
        **overrides,
    )


@pytest.fixture(scope="module")
def first_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("e2e")
    result = go(tmp_path)
    return result, tmp_path


def test_the_run_produced_the_expected_shape(first_run):
    result, _ = first_run
    totals = result.report["totals"]
    assert totals["answers"] == 40  # 20 cases x 2 models
    assert totals["pairs"] == 20
    assert totals["judge_passes"] == 40  # two presentation orders per pair
    assert totals["excluded_pairs"] == 0
    assert result.exit_code == 0


def test_every_documented_top_level_key_is_present(first_run):
    result, _ = first_run
    for key in (
        "tool_version",
        "generated_at",
        "run_id",
        "suite",
        "models",
        "baseline",
        "candidates",
        "judge",
        "prices",
        "run",
        "aborted_over_budget",
        "cost_incomplete",
        "headline_eligible",
        "calibration",
        "win_rates",
        "per_model",
        "per_task",
        "totals",
        "exclusions",
        "errors",
        "warnings",
        "reproduce",
        "what_changed",
        "decision_log_entry",
    ):
        assert key in result.report, key


def test_cost_is_positive_and_split_between_answers_and_judge(first_run):
    result, _ = first_run
    totals = result.report["totals"]
    assert totals["cost_usd"] > 0
    assert totals["cost_answers_usd"] > 0
    assert totals["cost_judge_usd"] > 0
    assert totals["cost_usd"] == pytest.approx(
        totals["cost_answers_usd"] + totals["cost_judge_usd"]
    )
    assert result.report["cost_incomplete"] is False


def test_calibration_is_populated_from_the_twelve_labels(first_run):
    result, _ = first_run
    cal = result.report["calibration"]
    assert cal["n_labels"] == 12
    assert cal["n_labels_ignored"] == 0
    assert cal["kappa"] is not None
    assert cal["kappa_band"] == cal["kappa_band"]
    assert sum(sum(row.values()) for row in cal["confusion"].values()) == 12
    assert cal["status"] in ("ok", "below_floor", "insufficient_labels")


def test_the_win_rate_carries_its_n_and_its_basis(first_run):
    result, _ = first_run
    win = result.report["win_rates"][MODELS[1]]
    assert win["n"] == 20
    assert 0.0 <= win["win_rate"] <= 1.0
    assert win["basis"] == "schema-passing pairs only"
    assert win["ci"] is not None  # n=20 is above the suppression threshold
    assert win["ci"][0] <= win["win_rate"] <= win["ci"][1]
    assert sum(win["counts"].values()) == 20


def test_schema_tasks_are_scored_and_free_text_tasks_are_not(first_run):
    result, _ = first_run
    for model in MODELS:
        schema = result.report["per_model"][model]["schema"]
        assert schema["n"] == 8  # three schema tasks: 3 + 3 + 2 cases
        assert schema["rate"] == 1.0  # the fake generates schema-valid instances


def test_all_the_files_are_written(first_run):
    result, tmp_path = first_run
    run_dir = result.report_path.parent
    assert (run_dir / "report.json").is_file()
    assert (run_dir / "report.md").is_file()
    answers = (run_dir / "answers.jsonl").read_text(encoding="utf-8").strip().splitlines()
    pairs = (run_dir / "pairs.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(answers) == 40
    assert len(pairs) == 20
    assert json.loads(answers[0])["text"]
    assert len(json.loads(pairs[0])["passes"]) == 2

    latest = json.loads((run_dir.parent / "latest.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == result.run_id


def test_report_html_is_written_beside_the_markdown(first_run):
    result, _ = first_run
    path = result.report_path.with_name("report.html")
    assert path.is_file()
    assert result.report_html_path == path
    html = path.read_text(encoding="utf-8")

    # blind by default: no model name is visible until the reveal toggle
    assert 'id="reveal" aria-pressed="false"' in html
    assert ".reveal-inline{display:none}" in html
    assert "data-reveal', '0'" in html
    assert '<div class="verdict reveal-only">' in html

    # the labelling flow, and the thing it exists to produce
    assert 'id="copy">Copy labels YAML' in html
    assert html.count('data-choice="tie"') == result.report["totals"]["pairs"]
    assert "'labels:'" in html
    assert "prefer: ' + prefer(p, rec.choice)" in html

    # the per-task agreement breakdown
    assert "Per-task agreement" in html

    # self-contained: no network, no framework, no external asset
    assert "<script src=" not in html and "http://" not in html and "https://" not in html
    assert len(html) < 2_000_000


def test_report_html_shows_every_judged_pair_side_by_side(first_run):
    result, _ = first_run
    html = result.report_path.with_name("report.html").read_text(encoding="utf-8")
    assert html.count('<article class="pair"') == result.report["totals"]["pairs"]
    assert html.count("<h4>Answer A") == result.report["totals"]["pairs"]
    assert html.count("<h4>Answer B") == result.report["totals"]["pairs"]
    # every answer that was scored is embedded in the page
    for pair in result.pair_view:
        assert pair["a_text"] and pair["b_text"]


def test_the_per_task_agreement_breakdown_is_in_json_and_markdown(first_run):
    result, _ = first_run
    rows = result.report["calibration"]["per_task_agreement"]
    assert rows, "twelve labels across eight tasks must break out per task"
    assert sum(row["n"] for row in rows) == result.report["calibration"]["n_labels"]
    # worst first, same discipline as the per-task win-rate table
    assert [row["agreement"] for row in rows] == sorted(row["agreement"] for row in rows)
    for row in rows:
        assert set(row) >= {"task", "n", "agree", "agreement", "kappa", "kappa_band", "low_n"}
        # kappa only where n makes it meaningful
        assert (row["kappa"] is None) or not row["low_n"]

    markdown = (result.report_path.parent / "report.md").read_text(encoding="utf-8")
    section = markdown.split("## Calibration")[1].split("## Win-rates")[0]
    assert "Per-task agreement" in section
    for row in rows:
        assert f"`{row['task']}`" in section


def test_report_md_renders_the_sections_in_spec_order(first_run):
    result, _ = first_run
    markdown = (result.report_path.parent / "report.md").read_text(encoding="utf-8")
    sections = [
        "## Calibration",
        "## Win-rates",
        "## Per-model scorecard",
        "## Per-task",
        "## What changed",
        "## Failures and exclusions",
        "## Reproduce",
        "## Decision log entry",
    ]
    positions = [markdown.index(s) for s in sections]
    assert positions == sorted(positions)
    assert "over schema-passing pairs only" in markdown.lower()
    assert render_markdown(result.report) == markdown


def test_the_report_carries_no_adjectives(first_run):
    result, _ = first_run
    markdown = (result.report_path.parent / "report.md").read_text(encoding="utf-8").lower()
    for word in ("impressively", "surprisingly", "we recommend", "clearly better"):
        assert word not in markdown


def test_kappa_is_never_printed_without_its_band(first_run):
    result, _ = first_run
    markdown = (result.report_path.parent / "report.md").read_text(encoding="utf-8")
    cal = result.report["calibration"]
    assert f"{cal['kappa']:.2f} ({cal['kappa_band']})" in markdown


# --------------------------------------------------------------------------
# reruns
# --------------------------------------------------------------------------

RUN_SCOPED = {
    "run_id",
    "generated_at",
    "what_changed",
    "decision_log_entry",
}


def normalise(report: dict) -> dict:
    """Strip everything that is *supposed* to differ between two runs."""
    out = copy.deepcopy(report)
    for key in RUN_SCOPED:
        out.pop(key, None)
    out["run"].pop("no_cache", None)
    # the pre-flight estimate covers only the calls that were not already
    # cached, so it is legitimately smaller on a rerun
    out["run"].pop("preflight_estimate_usd", None)
    out["reproduce"] = {k: v for k, v in out["reproduce"].items() if k not in ("live",)}
    totals = out["totals"]
    for key in (
        "cost_usd",
        "cost_answers_usd",
        "cost_judge_usd",
        "live_calls",
        "cache_hits",
        "cache_hit_rate",
    ):
        totals.pop(key, None)
    out["judge"].pop("cached_calls", None)
    out["judge"]["latency"].pop("live_fraction", None)
    for model in out["per_model"].values():
        model.pop("cache_hits", None)
        model["cost"].pop("this_run_usd", None)
        model["latency"].pop("live_fraction", None)
    return out


def test_a_second_identical_run_is_a_full_cache_hit_and_scores_identically(tmp_path):
    first = go(tmp_path)
    second = go(tmp_path, minutes=5)

    assert second.run_id != first.run_id
    assert second.report["totals"]["cache_hit_rate"] == 1.0
    assert second.report["totals"]["cost_usd"] == 0.0
    assert second.report["totals"]["cost_if_uncached_usd"] > 0
    # everything that is not run-scoped is byte-identical
    assert normalise(second.report) == normalise(first.report)


def test_the_second_run_reports_what_changed_against_the_first(tmp_path):
    go(tmp_path)
    second = go(tmp_path, minutes=5)
    changed = second.report["what_changed"]
    assert changed["comparable"] is True
    assert changed["candidates"][MODELS[1]]["win_rate"]["delta"] == 0.0
    assert changed["movers"] == []
    assert changed["calibration"]["kappa"]["delta"] == 0.0


def test_a_modified_suite_refuses_task_level_deltas_and_names_the_task(tmp_path):
    go(tmp_path)

    modified = tmp_path / "modified.yaml"
    text = EXAMPLE_SUITE.read_text(encoding="utf-8")
    text = text.replace(
        "Explain this error to a developer in their first year.",
        "Explain this error to a developer in their second year.",
    )
    modified.write_text(text, encoding="utf-8")

    third = go(tmp_path, minutes=10, suite=modified)
    changed = third.report["what_changed"]
    assert changed["comparable"] is False
    assert "not comparable" in changed["reason"]
    assert changed["tasks_modified"] == ["stacktrace-explain"]
    assert changed["tasks_added"] == [] and changed["tasks_removed"] == []
    assert "candidates" not in changed  # no metric deltas at all

    markdown = (third.report_path.parent / "report.md").read_text(encoding="utf-8")
    assert "stacktrace-explain" in markdown.split("## What changed")[1]
    assert "No metric deltas are shown" in markdown


def test_the_modified_task_is_the_only_cache_miss(tmp_path):
    go(tmp_path)
    modified = tmp_path / "modified.yaml"
    text = EXAMPLE_SUITE.read_text(encoding="utf-8")
    text = text.replace(
        "Explain this error to a developer in their first year.",
        "Explain this error to a developer in their second year.",
    )
    modified.write_text(text, encoding="utf-8")
    third = go(tmp_path, minutes=10, suite=modified)
    # stacktrace-explain has 2 cases x 2 models = 4 fresh answers out of 40
    assert third.report["totals"]["cache_hits"] == 36
