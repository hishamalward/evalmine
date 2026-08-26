"""The CLI verbs and, above all, the exit codes. Spec S4."""

from __future__ import annotations

import json

import pytest
from conftest import EXAMPLE_SUITE, PRICES_DIR, REPO_ROOT

from evalmine.cli import main

PRICE_FILE = str(PRICES_DIR / "prices-2026-08-23.yaml")
MODELS = "anthropic/claude-haiku-4-5,google/gemini-2.5-flash"
EXAMPLE_EXPERIMENT = REPO_ROOT / "examples" / "agent-model-comparison.yaml"


def run_cli(*args) -> int:
    return main(list(args))


def run_args(tmp_path, *extra) -> list[str]:
    return [
        "run",
        str(EXAMPLE_SUITE),
        "--models",
        MODELS,
        "--fake",
        "--prices",
        PRICE_FILE,
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out",
        str(tmp_path / "reports"),
        *extra,
    ]


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------


def test_run_exits_zero_and_prints_a_summary(tmp_path, capsys):
    assert run_cli(*run_args(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "win-rate" in out
    assert "over schema-passing pairs only" in out
    assert "calibration" in out


def test_run_points_at_the_html_report(tmp_path, capsys):
    assert run_cli(*run_args(tmp_path)) == 0
    assert "report.html" in capsys.readouterr().out
    written = list((tmp_path / "reports" / "everyday-eight").glob("*/report.html"))
    assert len(written) == 1


def test_run_json_prints_the_report(tmp_path, capsys):
    assert run_cli(*run_args(tmp_path, "--json")) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["totals"]["answers"] == 40


def test_a_bad_suite_exits_one(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("suite: x\nversion: 9\n", encoding="utf-8")
    code = run_cli("run", str(bad), "--models", MODELS, "--fake", "--prices", PRICE_FILE)
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_an_unknown_model_exits_one(tmp_path, capsys):
    code = run_cli(*run_args(tmp_path)[:3], "openai/nope,fake/a", "--fake", "--prices", PRICE_FILE)
    assert code == 1
    assert "openai/nope" in capsys.readouterr().err


def test_one_model_exits_one(tmp_path, capsys):
    code = run_cli("run", str(EXAMPLE_SUITE), "--models", "fake/a", "--fake", "--prices",
                   PRICE_FILE)
    assert code == 1


def test_a_missing_key_for_a_real_adapter_exits_two(tmp_path, capsys, monkeypatch):
    """Without --fake and without a key, the real adapter refuses immediately."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    code = run_cli(
        "run",
        str(EXAMPLE_SUITE),
        "--models",
        MODELS,
        "--prices",
        PRICE_FILE,
        "--cache-dir",
        str(tmp_path / "cache"),
        "--out",
        str(tmp_path / "reports"),
    )
    assert code == 2
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_over_the_cap_exits_four_before_spending(tmp_path, capsys):
    code = run_cli(*run_args(tmp_path, "--max-cost", "0.0001"))
    assert code == 4
    err = capsys.readouterr().err
    assert "Nothing was spent" in err
    assert not (tmp_path / "reports").exists()


def test_fail_under_calibration_exits_three(tmp_path, capsys):
    """The example suite's judge is uncalibrated against the fake answers."""
    code = run_cli(*run_args(tmp_path, "--fail-under-calibration"))
    assert code == 3
    out = capsys.readouterr().out
    assert "UNCALIBRATED" in out


def test_without_the_flag_an_uncalibrated_run_still_exits_zero(tmp_path):
    assert run_cli(*run_args(tmp_path)) == 0


def test_on_below_floor_fail_in_the_suite_also_exits_three(tmp_path, capsys):
    suite = tmp_path / "strict.yaml"
    suite.write_text(
        EXAMPLE_SUITE.read_text(encoding="utf-8").replace(
            "on_below_floor: flag", "on_below_floor: fail"
        ),
        encoding="utf-8",
    )
    code = run_cli(
        "run", str(suite), "--models", MODELS, "--fake", "--prices", PRICE_FILE,
        "--cache-dir", str(tmp_path / "cache"), "--out", str(tmp_path / "reports"),
    )
    assert code == 3


# --------------------------------------------------------------------------
# validate / prices
# --------------------------------------------------------------------------


def test_validate_exits_zero_and_makes_no_calls(capsys):
    assert run_cli("validate", str(EXAMPLE_SUITE), "--prices", PRICE_FILE) == 0
    out = capsys.readouterr().out
    assert "8 tasks, 20 cases, 12 labels" in out
    # The shipped table is verified (S6.3), so no placeholder note prints.
    assert "verified: false" not in out


def test_validate_catches_an_unrenderable_prompt(tmp_path, capsys):
    suite = tmp_path / "broken.yaml"
    suite.write_text(
        EXAMPLE_SUITE.read_text(encoding="utf-8").replace("{{commit}}", "{{commmit}}"),
        encoding="utf-8",
    )
    assert run_cli("validate", str(suite), "--prices", PRICE_FILE) == 1
    assert "commmit" in capsys.readouterr().err


def test_validate_catches_a_model_the_price_table_does_not_know(tmp_path, capsys):
    suite = tmp_path / "unpriced.yaml"
    suite.write_text(
        EXAMPLE_SUITE.read_text(encoding="utf-8").replace(
            "model: anthropic/claude-sonnet-4-6", "model: anthropic/claude-imaginary-9"
        ),
        encoding="utf-8",
    )
    assert run_cli("validate", str(suite), "--prices", PRICE_FILE) == 1
    assert "claude-imaginary-9" in capsys.readouterr().err


def test_experiment_validate_is_explicitly_zero_execution(capsys):
    assert run_cli("experiment", "validate", str(EXAMPLE_EXPERIMENT)) == 0
    out = capsys.readouterr().out
    assert "3 arms" in out
    assert "6 planned runs" in out
    assert "no agents launched" in out


def test_experiment_plan_prints_rotated_schedule(capsys):
    assert run_cli("experiment", "plan", str(EXAMPLE_EXPERIMENT)) == 0
    out = capsys.readouterr().out
    assert "opus-5-current" in out
    assert "gpt-5-6-sol-current" in out
    assert "dry run only" in out


def test_experiment_plan_json_is_machine_readable(capsys):
    assert run_cli("experiment", "plan", str(EXAMPLE_EXPERIMENT), "--json") == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["schedule"]["run_count"] == 6
    assert len(plan["runs"]) == 6


def test_bad_experiment_exits_one(tmp_path, capsys):
    manifest = tmp_path / "bad.yaml"
    manifest.write_text("experiment: bad\nversion: 2\n", encoding="utf-8")
    assert run_cli("experiment", "validate", str(manifest)) == 1
    assert "error:" in capsys.readouterr().err


def test_prices_prints_the_table_with_no_unverified_warning(capsys):
    assert run_cli("prices", "--table", PRICE_FILE) == 0
    out = capsys.readouterr().out
    assert "anthropic/claude-haiku-4-5" in out
    # The shipped table is verified (S6.3), so no warning prints.
    assert "WARNING: this table is unverified" not in out


def test_prices_for_a_suite_resolves_every_model_it_could_use(capsys):
    assert run_cli("prices", "--table", PRICE_FILE, "--for", str(EXAMPLE_SUITE)) == 0
    assert "resolve" in capsys.readouterr().out


# --------------------------------------------------------------------------
# report / last / compare
# --------------------------------------------------------------------------


@pytest.fixture
def two_runs(tmp_path):
    run_cli(*run_args(tmp_path))
    reports = sorted((tmp_path / "reports" / "everyday-eight").glob("*/report.json"))
    return tmp_path, reports


def test_last_prints_the_most_recent_run_id(two_runs, capsys):
    tmp_path, reports = two_runs
    assert run_cli("last", str(EXAMPLE_SUITE), "--out", str(tmp_path / "reports")) == 0
    assert capsys.readouterr().out.strip() == reports[0].parent.name


def test_last_without_a_report_exits_one(tmp_path):
    assert run_cli("last", str(EXAMPLE_SUITE), "--out", str(tmp_path / "nothing")) == 1


def test_report_re_renders_markdown_from_json(two_runs, capsys):
    tmp_path, reports = two_runs
    markdown = reports[0].with_name("report.md")
    markdown.write_text("clobbered", encoding="utf-8")
    assert run_cli("report", str(reports[0]), "--out", str(tmp_path / "reports")) == 0
    assert "## Calibration" in markdown.read_text(encoding="utf-8")


def test_report_to_stdout(two_runs, capsys):
    _, reports = two_runs
    assert run_cli("report", str(reports[0]), "--stdout") == 0
    assert "## Win-rates" in capsys.readouterr().out


def test_compare_prints_the_delta(two_runs, capsys):
    _, reports = two_runs
    assert run_cli("compare", str(reports[0]), str(reports[0]), "--json") == 0
    diff = json.loads(capsys.readouterr().out)
    assert diff["comparable"] is True
    assert diff["candidates"]["google/gemini-2.5-flash"]["win_rate"]["delta"] == 0.0


def test_compare_human_readable(two_runs, capsys):
    _, reports = two_runs
    assert run_cli("compare", str(reports[0]), str(reports[0])) == 0
    assert "win-rate" in capsys.readouterr().out


def test_a_missing_report_exits_one(capsys):
    assert run_cli("report", "no-such-run-id") == 1
