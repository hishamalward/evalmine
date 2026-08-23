"""argparse and exit codes. No evaluation logic lives here.

Spec: docs/spec.md S4. The cost guard in particular is enforced in
``core.run_suite`` and not here, so that the CLI and the MCP server cannot
diverge on the one question that costs money.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .adapters.base import UnsupportedProviderError
from .core import (
    CostRefused,
    RunError,
    UsageError,
    compare,
    last_report,
    load_report,
    run_suite,
)
from .metrics import format_kappa
from .prices import PriceTableError, UnknownModelError, load_price_table
from .report import render_markdown
from .suite import SuiteError, load_suite

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_RUNTIME = 2
EXIT_UNCALIBRATED = 3
EXIT_REFUSED_PREFLIGHT = 4
EXIT_ABORTED_OVER_BUDGET = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalmine",
        description=(
            "Score a model change against your own tasks: pairwise LLM-judge win-rates "
            "calibrated to your labels, schema-pass rate, latency and cost."
        ),
    )
    parser.add_argument("--version", action="version", version=f"evalmine {__version__}")
    subparsers = parser.add_subparsers(dest="verb", required=True)

    run = subparsers.add_parser("run", help="run a suite over two or more models")
    run.add_argument("suite", help="path to the suite YAML")
    run.add_argument(
        "--models",
        required=True,
        help="comma-separated, ordered; the first is the baseline unless --baseline is given",
    )
    run.add_argument("--baseline", default=None)
    run.add_argument("--judge", default=None, help="overrides judge.model in the suite")
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--max-cost", type=float, default=None, help="USD; overrides limits")
    run.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore existing cache entries; still writes the fresh results",
    )
    run.add_argument("--cache-dir", default=None)
    run.add_argument("--out", default="reports")
    run.add_argument("--prices", default=None, help="a specific price table file")
    run.add_argument(
        "--fake",
        action="store_true",
        help="route every model to the deterministic fake adapter; contacts nothing",
    )
    run.add_argument("--fail-under-calibration", action="store_true")
    run.add_argument("--json", action="store_true", help="print the report JSON on stdout")
    run.add_argument("-v", "--verbose", action="store_true")

    validate = subparsers.add_parser(
        "validate", help="parse, schema-check, render every prompt, resolve every model"
    )
    validate.add_argument("suite")
    validate.add_argument("--prices", default=None)

    prices = subparsers.add_parser("prices", help="print the pinned price table")
    prices.add_argument("--table", default=None)
    prices.add_argument("--for", dest="for_suite", default=None)

    report = subparsers.add_parser("report", help="re-render report.md from report.json")
    report.add_argument("target", help="a run-id, a run directory, or a report.json")
    report.add_argument("--out", default="reports")
    report.add_argument("--stdout", action="store_true", help="print instead of writing")

    last = subparsers.add_parser("last", help="print the most recent run-id for this suite")
    last.add_argument("suite")
    last.add_argument("--out", default="reports")

    compare_cmd = subparsers.add_parser("compare", help="print the delta between two reports")
    compare_cmd.add_argument("report_a")
    compare_cmd.add_argument("report_b")
    compare_cmd.add_argument("--out", default="reports")
    compare_cmd.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.verb == "run":
            return _cmd_run(args, argv)
        if args.verb == "validate":
            return _cmd_validate(args)
        if args.verb == "prices":
            return _cmd_prices(args)
        if args.verb == "report":
            return _cmd_report(args)
        if args.verb == "last":
            return _cmd_last(args)
        if args.verb == "compare":
            return _cmd_compare(args)
    except (SuiteError, UnknownModelError, PriceTableError, UsageError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except CostRefused as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED_PREFLIGHT
    except (RunError, UnsupportedProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    parser.error(f"unknown verb {args.verb!r}")
    return EXIT_USAGE


# --------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace, argv: list[str]) -> int:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    command = "evalmine " + " ".join(argv)
    result = run_suite(
        args.suite,
        models,
        baseline=args.baseline,
        judge_model=args.judge,
        repeats=args.repeats,
        max_cost=args.max_cost,
        no_cache=args.no_cache,
        cache_dir=args.cache_dir,
        out_dir=args.out,
        fake=args.fake,
        prices_path=args.prices,
        command=command,
    )
    report = result.report

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _print_summary(report, result, verbose=args.verbose)

    if result.exit_code == EXIT_ABORTED_OVER_BUDGET:
        return EXIT_ABORTED_OVER_BUDGET
    if not report["headline_eligible"]:
        if args.fail_under_calibration or report["calibration"]["on_below_floor"] == "fail":
            return EXIT_UNCALIBRATED
    return EXIT_OK


def _print_summary(report, result, verbose: bool = False) -> None:
    cal = report["calibration"]
    print(f"run {report['run_id']}  ({report['suite']['name']})")
    print(f"  report: {result.report_md_path}")
    print(
        f"  calibration: {cal['status']} - kappa {format_kappa(cal.get('kappa'))}"
        f" over {cal['n_labels']} labels"
        f" - headline eligible: {str(report['headline_eligible']).lower()}"
    )
    for candidate, win in report["win_rates"].items():
        rate = "n/a" if win["win_rate"] is None else f"{win['win_rate']:.3f}"
        flag = "" if report["headline_eligible"] else " (UNCALIBRATED)"
        ci = win["ci"]
        ci_text = f" [{ci[0]:.3f}-{ci[1]:.3f}]" if ci else " [CI: n too small]"
        print(
            f"  {candidate} vs {win['vs']}: win-rate {rate}{flag}{ci_text}"
            f" over schema-passing pairs only, n={win['n']}"
            f" - flips {win['flips']} - excluded {win['excluded']}"
        )
    totals = report["totals"]
    print(
        f"  cost: ${totals['cost_usd']:.4f} this run "
        f"(answers ${totals['cost_answers_usd']:.4f}, judge ${totals['cost_judge_usd']:.4f});"
        f" if uncached ${totals['cost_if_uncached_usd']:.4f}"
    )
    if report["aborted_over_budget"]:
        print("  ABORTED OVER BUDGET - this report is partial")
    for warning in report["warnings"] if verbose else []:
        print(f"  warning: {warning}")


def _cmd_validate(args: argparse.Namespace) -> int:
    suite = load_suite(args.suite)
    table = load_price_table(args.prices)
    referenced = _models_a_suite_could_use(suite)
    table.resolve_all(sorted(referenced))
    cases = sum(len(t.cases) for t in suite.tasks)
    print(
        f"ok: {suite.path} - {len(suite.tasks)} tasks, {cases} cases, "
        f"{len(suite.labels)} labels; every prompt rendered; "
        f"{len(referenced)} model strings resolved against {table.filename}"
    )
    if not table.verified:
        print(f"note: {table.filename} declares verified: false - its figures are placeholders")
    return EXIT_OK


def _models_a_suite_could_use(suite) -> set[str]:
    referenced = {suite.judge.model}
    for label in suite.labels:
        referenced.add(label.baseline)
        referenced.add(label.candidate)
    return referenced


def _cmd_prices(args: argparse.Namespace) -> int:
    table = load_price_table(args.table)
    print(f"{table.path}")
    print(f"pinned: {table.pinned}  currency: {table.currency}  verified: {table.verified}")
    if not table.verified:
        print("WARNING: this table is unverified; every figure in it is a placeholder")
    print()
    print(f"{'model':<40} {'in $/Mtok':>10} {'out $/Mtok':>11} {'cached $/Mtok':>14}  source")
    for model in sorted(table.rows):
        row = table.rows[model]
        print(
            f"{model:<40} {row.input_per_mtok:>10.4f} {row.output_per_mtok:>11.4f} "
            f"{row.cached_input_per_mtok:>14.4f}  {row.source or '-'}"
        )
    if args.for_suite:
        suite = load_suite(args.for_suite)
        referenced = _models_a_suite_could_use(suite)
        table.resolve_all(sorted(referenced))
        print()
        print(f"all {len(referenced)} model strings referenced by {suite.name} resolve")
    return EXIT_OK


def _cmd_report(args: argparse.Namespace) -> int:
    report = load_report(args.target, out_dir=args.out)
    markdown = render_markdown(report)
    if args.stdout:
        print(markdown)
        return EXIT_OK
    target = Path(args.out) / report["suite"]["name"] / report["run_id"] / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    print(f"wrote {target}")
    return EXIT_OK


def _cmd_last(args: argparse.Namespace) -> int:
    found = last_report(args.suite, out_dir=args.out)
    if found is None:
        print("no report found for this suite", file=sys.stderr)
        return EXIT_USAGE
    print(found["run_id"])
    return EXIT_OK


def _cmd_compare(args: argparse.Namespace) -> int:
    a = load_report(args.report_a, out_dir=args.out)
    b = load_report(args.report_b, out_dir=args.out)
    diff = compare(a, b)
    if args.json:
        print(json.dumps(diff, indent=2, sort_keys=True, ensure_ascii=False))
        return EXIT_OK
    from .report import _render_what_changed

    print(f"# {a['run_id']} -> {b['run_id']}")
    print()
    print(_render_what_changed(diff))
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
