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
from .decision import (
    DecisionError,
    JudgeRefused,
    generate_decision_report,
    judge_experiment,
    verify_decision,
    verify_judging,
)
from .experiment import ExperimentError, build_plan, load_experiment
from .experiment_report import (
    ExperimentReportError,
    generate_experiment_report,
    verify_experiment_report,
)
from .external import (
    ExternalArtifactError,
    import_external_artifacts,
    is_external_import,
    verify_external_import,
)
from .metrics import format_kappa
from .prices import PriceTableError, UnknownModelError, load_price_table
from .report import render_markdown
from .runner import (
    DEFAULT_TURN_TIMEOUT,
    ExecutionRefused,
    RunnerError,
    execute_experiment,
    preflight_experiment,
    verify_execution,
)
from .suite import SuiteError, load_suite
from .validators import (
    ValidationError,
    ValidationRefused,
    check_experiment,
    verify_validation,
)
from .workflow import (
    WorkflowError,
    WorkflowRefused,
    load_workflow,
    run_workflow,
    verify_workflow,
    workflow_plan,
)
from .workspace import (
    discard_prepared,
    prepare_experiment,
    prepare_failed_run_retry,
    verify_prepared,
)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_RUNTIME = 2
EXIT_UNCALIBRATED = 3
EXIT_REFUSED_PREFLIGHT = 4
EXIT_ABORTED_OVER_BUDGET = 5


def _format_progress_duration(duration_ms: object) -> str:
    if not isinstance(duration_ms, (int, float)):
        return ""
    seconds = max(0, round(duration_ms / 1000))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes}m{seconds:02d}s" if minutes else f"{seconds}s"


def _print_experiment_progress(event: dict[str, object]) -> None:
    kind = event.get("event")
    if kind == "execution_started":
        inherited = event.get("inherited_run_count")
        prefix = "retry execution" if isinstance(inherited, int) else "execution"
        inherited_text = f", {inherited} inherited" if isinstance(inherited, int) else ""
        run_word = "run" if event["run_count"] == 1 else "runs"
        print(
            f"{prefix} started: {event['run_count']} {run_word}{inherited_text}, "
            f"max_parallel={event['max_parallel']}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind in {"run_started", "run_completed", "turn_started", "turn_completed"}:
        prefix = f"[{event['run_position']}/{event['run_count']}]"
        subject = f"{event['model']} ({event['arm']})"
        if kind == "run_started":
            message = f"{prefix} {subject} - run started"
        elif kind == "run_completed":
            elapsed = _format_progress_duration(event.get("duration_ms"))
            message = f"{prefix} {subject} - run {event['status']}"
            if elapsed:
                message += f" - {elapsed}"
        else:
            message = (
                f"{prefix} {subject} - turn {event['turn']}/{event['turn_count']} "
                f"{'started' if kind == 'turn_started' else event['status']}"
            )
            if kind == "turn_completed":
                elapsed = _format_progress_duration(event.get("duration_ms"))
                if elapsed:
                    message += f" - {elapsed}"
        print(message, file=sys.stderr, flush=True)
        return
    if kind == "execution_completed":
        print(
            f"execution {event['status']}: {event['succeeded']}/{event['run_count']} "
            "runs succeeded",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind == "judging_started":
        print(
            f"judging started: {event['call_count']} {event['ranking_style']} call(s) "
            f"with {event['model']}",
            file=sys.stderr,
            flush=True,
        )
        return
    if kind in {"judge_call_started", "judge_call_completed"}:
        prefix = f"[{event['call_position']}/{event['call_count']}]"
        if kind == "judge_call_started":
            message = f"{prefix} {event['model']} - judge call started"
        else:
            message = f"{prefix} {event['model']} - judge call {event['status']}"
            elapsed = _format_progress_duration(event.get("duration_ms"))
            if elapsed:
                message += f" - {elapsed}"
        print(message, file=sys.stderr, flush=True)
        return
    if kind == "judging_completed":
        print(
            f"judging completed: {event['call_count']} calls, "
            f"{event['pair_count']} induced pairs",
            file=sys.stderr,
            flush=True,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evalmine",
        description=(
            "Turn direct model suites, isolated agent episodes, or imported artifacts "
            "into calibrated, provenance-checked evaluation evidence."
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

    experiment = subparsers.add_parser(
        "experiment",
        help="plan, run, check, and report a v2 agent experiment",
    )
    experiment_subparsers = experiment.add_subparsers(
        dest="experiment_verb", required=True
    )
    experiment_validate = experiment_subparsers.add_parser(
        "validate",
        help="validate the manifest and all local references; no calls or workspaces",
    )
    experiment_validate.add_argument("manifest")
    experiment_plan = experiment_subparsers.add_parser(
        "plan",
        help="print the deterministic arm x episode x repeat schedule",
    )
    experiment_plan.add_argument("manifest")
    experiment_plan.add_argument(
        "--json", action="store_true", help="print the complete machine-readable plan"
    )
    experiment_prepare = experiment_subparsers.add_parser(
        "prepare",
        help="materialize isolated workspaces and evidence without launching agents",
    )
    experiment_prepare.add_argument("manifest")
    experiment_prepare.add_argument(
        "--out",
        required=True,
        help="artifact base directory; must be outside the seed repository",
    )
    experiment_prepare.add_argument("--json", action="store_true")
    experiment_import = experiment_subparsers.add_parser(
        "import",
        help="pin completed JSONL artifacts for blind report, judge, and decision",
    )
    experiment_import.add_argument(
        "bundle",
        help="directory containing evalmine-import.yaml and hash-pinned JSONL files",
    )
    experiment_import.add_argument(
        "--out",
        required=True,
        help="exact create-once external evidence directory",
    )
    experiment_import.add_argument("--json", action="store_true")
    experiment_retry = experiment_subparsers.add_parser(
        "retry",
        help="prepare a derived envelope that executes only failed runs; launches nothing",
    )
    experiment_retry.add_argument("prepared", help="completed partial execution to inherit")
    experiment_retry.add_argument(
        "--out",
        required=True,
        help="artifact base directory; must be outside the seed repository",
    )
    experiment_retry.add_argument("--json", action="store_true")
    experiment_verify = experiment_subparsers.add_parser(
        "verify",
        help="verify workspaces and confirm that the baseline has not changed",
    )
    experiment_verify.add_argument("prepared")
    experiment_verify.add_argument("--json", action="store_true")
    experiment_preflight = experiment_subparsers.add_parser(
        "preflight",
        help="probe local agent CLI capabilities without making provider calls",
    )
    experiment_preflight.add_argument("prepared")
    experiment_preflight.add_argument("--json", action="store_true")
    experiment_preflight.add_argument(
        "--executable",
        action="append",
        default=[],
        metavar="RUNNER=PATH",
        help="override a runner executable (repeatable; useful for controlled wrappers)",
    )
    experiment_execute = experiment_subparsers.add_parser(
        "execute",
        help="run prepared agent episodes and capture evidence",
    )
    experiment_execute.add_argument("prepared")
    experiment_execute.add_argument(
        "--allow-provider-calls",
        action="store_true",
        help="required acknowledgement that installed CLIs may contact providers",
    )
    experiment_execute.add_argument(
        "--allow-external-writes",
        action="store_true",
        help="required when the manifest permits writes to exact paths outside workspaces",
    )
    experiment_execute.add_argument(
        "--turn-timeout",
        type=int,
        default=DEFAULT_TURN_TIMEOUT,
        metavar="SECONDS",
    )
    experiment_execute.add_argument("--json", action="store_true")
    experiment_execute.add_argument(
        "--executable",
        action="append",
        default=[],
        metavar="RUNNER=PATH",
        help="override a runner executable (repeatable; useful for controlled wrappers)",
    )
    experiment_check = experiment_subparsers.add_parser(
        "check",
        help="run declared objective validators over completed execution evidence",
    )
    experiment_check.add_argument("prepared")
    experiment_check.add_argument(
        "--allow-validator-commands",
        action="store_true",
        help="required before manifest-declared test/lint processes may run",
    )
    experiment_check.add_argument("--json", action="store_true")
    experiment_report = experiment_subparsers.add_parser(
        "report",
        help="generate a self-contained blind HTML episode report and labeling queue",
    )
    experiment_report.add_argument("prepared")
    experiment_report.add_argument(
        "--ranking-style",
        choices=("pairwise", "n-way"),
        default=None,
        help="explicit derived-report ranking style; recorded as an operator override",
    )
    experiment_report.add_argument(
        "--out",
        default=None,
        help="write a create-once report revision here instead of PREPARED/report",
    )
    experiment_report.add_argument(
        "--prices",
        default=None,
        help="pinned price table for API list-price equivalents (default: newest shipped)",
    )
    experiment_report.add_argument("--json", action="store_true")
    experiment_judge = experiment_subparsers.add_parser(
        "judge",
        help="blindly apply the configured pairwise or N-way ranking protocol",
    )
    experiment_judge.add_argument("prepared")
    experiment_judge.add_argument(
        "--ranking-style",
        choices=("pairwise", "n-way"),
        default=None,
        help="explicit judging protocol override; recorded in immutable judge evidence",
    )
    experiment_judge.add_argument(
        "--runner",
        choices=("claude-code", "codex-cli", "gemini-cli", "api-prompt"),
        default=None,
        help="explicit judge runner override; recorded in immutable judge evidence",
    )
    experiment_judge.add_argument(
        "--model",
        default=None,
        help="explicit judge model override; recorded in immutable judge evidence",
    )
    experiment_judge.add_argument(
        "--out",
        default=None,
        help="write a create-once named judge track here instead of PREPARED/judging",
    )
    experiment_judge.add_argument("--allow-provider-calls", action="store_true")
    experiment_judge.add_argument(
        "--max-cost", type=float, default=None, help="USD; required for API judging"
    )
    experiment_judge.add_argument("--prices", default=None)
    experiment_judge.add_argument(
        "--labels",
        action="append",
        default=None,
        help="independent human label export (required for calibration-subset judging)",
    )
    experiment_judge.add_argument(
        "--fake", action="store_true", help="use the deterministic offline judge"
    )
    experiment_judge.add_argument("--json", action="store_true")
    experiment_judge.add_argument(
        "--executable", action="append", default=[], metavar="RUNNER=PATH"
    )
    experiment_decide = experiment_subparsers.add_parser(
        "decide",
        help="import human labels, calibrate the judge, score arms, and write decision HTML",
    )
    experiment_decide.add_argument("prepared")
    experiment_decide.add_argument(
        "--labels", action="append", required=True, help="exported label JSON (repeatable)"
    )
    experiment_decide.add_argument(
        "--judging",
        action="append",
        default=None,
        help="immutable judge evidence root (repeat for side-by-side judges)",
    )
    experiment_decide.add_argument(
        "--out",
        default=None,
        help="write a create-once decision revision here instead of PREPARED/decision",
    )
    experiment_decide.add_argument("--json", action="store_true")
    experiment_discard = experiment_subparsers.add_parser(
        "discard",
        help="remove one marked preparation and its worktree registrations",
    )
    experiment_discard.add_argument("prepared")
    experiment_discard.add_argument(
        "--yes", action="store_true", help="required confirmation for deletion"
    )

    workflow = subparsers.add_parser(
        "workflow", help="validate and run a controlled enrichment/backoff DAG"
    )
    workflow_subparsers = workflow.add_subparsers(dest="workflow_verb", required=True)
    workflow_validate = workflow_subparsers.add_parser(
        "validate", help="validate the DAG and frozen fixture hashes; launches nothing"
    )
    workflow_validate.add_argument("manifest")
    workflow_validate.add_argument("--json", action="store_true")
    workflow_plan_cmd = workflow_subparsers.add_parser(
        "plan", help="show topological levels, fan-out, gates, and commands"
    )
    workflow_plan_cmd.add_argument("manifest")
    workflow_plan_cmd.add_argument("--json", action="store_true")
    workflow_run = workflow_subparsers.add_parser(
        "run", help="copy the workspace, restore fixtures, and execute the DAG"
    )
    workflow_run.add_argument("manifest")
    workflow_run.add_argument("--out", required=True)
    workflow_run.add_argument("--allow-commands", action="store_true")
    workflow_run.add_argument("--allow-provider-calls", action="store_true")
    workflow_run.add_argument("--json", action="store_true")
    workflow_verify = workflow_subparsers.add_parser(
        "verify", help="verify workflow evidence and the frozen final workspace"
    )
    workflow_verify.add_argument("root")
    workflow_verify.add_argument("--json", action="store_true")

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
        if args.verb == "experiment":
            return _cmd_experiment(args)
        if args.verb == "workflow":
            return _cmd_workflow(args)
    except (ExecutionRefused, ValidationRefused, JudgeRefused, WorkflowRefused) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED_PREFLIGHT
    except (
        RunnerError,
        ValidationError,
        ExperimentReportError,
        ExternalArtifactError,
        DecisionError,
        WorkflowError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_RUNTIME
    except (
        ExperimentError,
        SuiteError,
        UnknownModelError,
        PriceTableError,
        UsageError,
    ) as exc:
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
    if result.report_html_path is not None:
        print(f"  html (blind answer pairs + labelling): {result.report_html_path}")
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


def _cmd_experiment(args: argparse.Namespace) -> int:
    if args.experiment_verb == "import":
        result = import_external_artifacts(args.bundle, args.out)
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(
                f"imported {result.record_count} completed artifacts across "
                f"{result.block_count} comparison blocks and {result.condition_count} "
                f"conditions into {result.root}; provider calls: 0"
            )
        return EXIT_OK

    if args.experiment_verb == "retry":
        result = prepare_failed_run_retry(args.prepared, args.out)
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            retry_count = len(result.retry_run_keys)
            retry_label = "retry" if retry_count == 1 else "retries"
            print(
                f"ok: {result.root} - inherited {len(result.inherited_run_keys)} "
                f"succeeded runs; prepared {retry_count} failed-run {retry_label}; "
                "provider calls: 0"
            )
        return EXIT_OK

    if args.experiment_verb == "verify":
        result = verify_prepared(args.prepared)
        execution = None
        validation = None
        report = None
        if (Path(result["root"]) / "execution").is_dir():
            execution = verify_execution(args.prepared)
            result["execution"] = execution
        if (Path(result["root"]) / "validation").is_dir():
            validation = verify_validation(args.prepared)
            result["validation"] = validation
        if (Path(result["root"]) / "report").is_dir():
            report = verify_experiment_report(args.prepared)
            result["report"] = report
        if (Path(result["root"]) / "judging").is_dir():
            result["judging"] = verify_judging(args.prepared)
        if (Path(result["root"]) / "decision").is_dir():
            result["decision"] = verify_decision(args.prepared)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        elif is_external_import(args.prepared):
            external = verify_external_import(args.prepared)
            suffix = f"; report {report['pair_count']} blind pairs" if report else ""
            print(
                f"ok: {external['root']} - {external['record_count']} pinned external "
                f"artifacts across {external['block_count']} blocks; provider calls: 0{suffix}"
            )
        else:
            suffix = (
                f"; execution {execution['status']} ({execution['succeeded']}/"
                f"{execution['run_count']} succeeded)"
                if execution
                else ""
            )
            if validation:
                suffix += (
                    f"; validation {validation['verdict']} ({validation['passed']}/"
                    f"{validation['run_count']} passed)"
                )
            if report:
                suffix += f"; report {report['pair_count']} blind pairs"
            print(
                f"ok: {result['root']} - {result['run_count']} isolated workspaces; "
                f"baseline unchanged{suffix}"
            )
        return EXIT_OK

    if args.experiment_verb == "preflight":
        if is_external_import(args.prepared):
            raise ExperimentError(
                "external artifacts are already completed; preflight is not applicable"
            )
        overrides = _runner_executable_overrides(args.executable)
        result = preflight_experiment(args.prepared, executable_overrides=overrides)
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        elif result.ok:
            runner_text = ", ".join(
                f"{probe.runner} {probe.version or '(unknown version)'}"
                for probe in result.probes
            )
            print(
                f"ok: {result.root} - {result.run_count} runs; {runner_text}; "
                "provider calls: 0"
            )
        else:
            print(
                f"refused: {result.root} - preflight found {len(result.issues)} issue(s)",
                file=sys.stderr,
            )
            for issue in result.issues:
                print(f"  - {issue}", file=sys.stderr)
        return EXIT_OK if result.ok else EXIT_REFUSED_PREFLIGHT

    if args.experiment_verb == "execute":
        if is_external_import(args.prepared):
            raise ExperimentError(
                "external artifacts are already completed; import never launches generation"
            )
        overrides = _runner_executable_overrides(args.executable)
        result = execute_experiment(
            args.prepared,
            allow_provider_calls=args.allow_provider_calls,
            allow_external_writes=args.allow_external_writes,
            turn_timeout=args.turn_timeout,
            executable_overrides=overrides,
            progress=None if args.json else _print_experiment_progress,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(
                f"{result.status}: {result.root} - {result.succeeded}/{result.run_count} "
                "runs succeeded"
            )
        return EXIT_OK if result.status == "completed" else EXIT_RUNTIME

    if args.experiment_verb == "check":
        if is_external_import(args.prepared):
            raise ExperimentError(
                "external artifacts carry completed outputs; experiment validators are not run"
            )
        result = check_experiment(
            args.prepared,
            allow_validator_commands=args.allow_validator_commands,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(
                f"{result.verdict}: {result.root} - {result.passed}/{result.run_count} "
                "runs passed; provider runners launched: 0"
            )
        return EXIT_OK if result.verdict == "passed" else EXIT_RUNTIME

    if args.experiment_verb == "report":
        result = generate_experiment_report(
            args.prepared,
            ranking_style=args.ranking_style,
            output=args.out,
            prices_path=args.prices,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            review_count = (
                result.pair_count
                if result.ranking_style == "pairwise"
                else result.ranking_count
            )
            review_unit = "pairs" if result.ranking_style == "pairwise" else "rankings"
            print(
                f"wrote {result.html} - "
                f"{review_count} blind {review_unit} across "
                f"{result.run_count} runs; provider runners launched: 0"
            )
        return EXIT_OK

    if args.experiment_verb == "judge":
        overrides = _runner_executable_overrides(args.executable)
        result = judge_experiment(
            args.prepared,
            allow_provider_calls=args.allow_provider_calls,
            max_cost_usd=args.max_cost,
            fake=args.fake,
            prices_path=args.prices,
            executable_overrides=overrides,
            ranking_style=args.ranking_style,
            runner_override=args.runner,
            model_override=args.model,
            calibration_label_paths=args.labels,
            output=args.out,
            progress=None if args.json else _print_experiment_progress,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(
                f"{result.status}: {result.root} - {result.pair_count} induced pairs, "
                f"{result.call_count} judge calls"
            )
        return EXIT_OK

    if args.experiment_verb == "decide":
        result = generate_decision_report(
            args.prepared,
            args.labels,
            judging_paths=args.judging,
            output=args.out,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            eligibility = "headline eligible" if result.headline_eligible else "diagnostic only"
            print(
                f"wrote {result.html} - {result.labelled_pairs} human-labelled pairs, "
                f"{result.judged_pairs} judged pairs; {eligibility}"
            )
        return EXIT_OK

    if args.experiment_verb == "discard":
        if not args.yes:
            raise ExperimentError(
                "discard deletes prepared workspaces; pass --yes to confirm the exact path"
            )
        result = discard_prepared(args.prepared)
        print(
            f"discarded {result['root']} - {result['run_count']} workspaces; "
            "this cannot be recovered"
        )
        return EXIT_OK

    experiment = load_experiment(args.manifest)
    plan = build_plan(experiment)
    if args.experiment_verb == "validate":
        turns = sum(len(episode.turns) for episode in experiment.episodes)
        episode_label = "episode" if len(experiment.episodes) == 1 else "episodes"
        print(
            f"ok: {experiment.path} - {len(experiment.arms)} arms, "
            f"{len(experiment.episodes)} {episode_label}, {turns} episode turns, "
            f"{len(plan.runs)} planned runs; no agents launched"
        )
        for warning in plan.warnings:
            print(f"warning: {warning}")
        return EXIT_OK

    if args.experiment_verb == "plan":
        if args.json:
            print(json.dumps(plan.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
            return EXIT_OK
        print(f"plan {plan.plan_id}  ({experiment.name})")
        print(f"  question: {experiment.question}")
        print(
            f"  seed: {experiment.seed.repo} @ {experiment.seed.ref} "
            f"({experiment.seed.commit[:12]}) "
            f"(dirty={experiment.seed.dirty}, untracked={experiment.seed.untracked})"
        )
        print(
            f"  isolation: workspace={experiment.isolation.workspace}, "
            f"session={experiment.isolation.session}, "
            f"external-writes={experiment.isolation.external_writes}"
        )
        print(
            f"  schedule: {experiment.order}, max-parallel={experiment.max_parallel}, "
            f"runs={len(plan.runs)}"
        )
        print()
        print("  arms:")
        for arm in experiment.arms:
            print(
                f"    {arm.id}: {arm.runner} / {arm.model} / {arm.auth}; "
                f"instructions={arm.configuration.instructions}, "
                f"plugins={arm.configuration.plugins}"
            )
        print()
        print(f"{'#':>3}  {'block':<28} {'arm':<24} {'runner':<13} model")
        for run in plan.runs:
            print(
                f"{run.sequence:>3}  {run.block:<28} {run.arm_id:<24} "
                f"{run.runner:<13} {run.model}"
            )
        for warning in plan.warnings:
            print(f"warning: {warning}")
        print("dry run only: no workspaces created, agents launched, or provider calls made")
        return EXIT_OK

    if args.experiment_verb == "prepare":
        prepared = prepare_experiment(experiment, args.out)
        if args.json:
            print(
                json.dumps(
                    prepared.as_dict(), indent=2, sort_keys=True, ensure_ascii=False
                )
            )
            return EXIT_OK
        print(f"prepared {prepared.plan.plan_id}  ({experiment.name})")
        print(f"  artifacts: {prepared.root}")
        print(f"  workspaces: {len(prepared.runs)} ({experiment.isolation.workspace})")
        print(f"  baseline fingerprint: {prepared.baseline_fingerprint[:12]}")
        print("  agents launched: 0; provider calls: 0")
        return EXIT_OK

    raise ExperimentError(f"unknown experiment verb {args.experiment_verb!r}")


def _runner_executable_overrides(values: list[str]) -> dict[str, str]:
    allowed = {"claude-code", "codex-cli", "gemini-cli"}
    overrides: dict[str, str] = {}
    for value in values:
        runner, separator, path = value.partition("=")
        if not separator or not runner or not path:
            raise ExperimentError("--executable must use RUNNER=PATH")
        if runner not in allowed:
            raise ExperimentError(f"unknown --executable runner {runner!r}")
        if runner in overrides:
            raise ExperimentError(f"duplicate --executable override for {runner}")
        overrides[runner] = path
    return overrides


def _cmd_workflow(args: argparse.Namespace) -> int:
    if args.workflow_verb == "verify":
        result = verify_workflow(args.root)
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(
                f"ok: {result['root']} - {result['workflow']} is {result['status']}; "
                f"{result['instance_count']} instances, {result['artifact_count']} artifacts"
            )
        return EXIT_OK

    workflow = load_workflow(args.manifest)
    plan = workflow_plan(workflow)
    if args.workflow_verb == "validate":
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(
                f"ok: {workflow.path} - {len(workflow.nodes)} DAG nodes, "
                f"{plan['instance_count']} instances, {len(workflow.fixtures)} frozen fixtures; "
                "commands launched: 0"
            )
        return EXIT_OK
    if args.workflow_verb == "plan":
        if args.json:
            print(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(f"workflow {workflow.name}  ({workflow.hash[:12]})")
            print(f"  root: {workflow.root}")
            print(f"  max parallel: {workflow.max_parallel}")
            for index, level in enumerate(plan["levels"], 1):
                print(f"  level {index}: {', '.join(level)}")
            print(
                f"  instances: {plan['instance_count']}; frozen fixtures: "
                f"{len(workflow.fixtures)}; provider nodes: "
                f"{', '.join(plan['provider_nodes']) or 'none'}"
            )
            print("dry run only: no workspace copied and no commands launched")
        return EXIT_OK
    if args.workflow_verb == "run":
        result = run_workflow(
            workflow,
            args.out,
            allow_commands=args.allow_commands,
            allow_provider_calls=args.allow_provider_calls,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2, sort_keys=True, ensure_ascii=False))
        else:
            print(
                f"{result.status}: {result.root} - {result.nodes_succeeded}/"
                f"{result.nodes_total} nodes succeeded, {result.instances} instances; "
                f"report {result.html}"
            )
        return EXIT_OK if result.status == "completed" else EXIT_RUNTIME
    raise WorkflowError(f"unknown workflow verb {args.workflow_verb!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
