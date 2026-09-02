"""The MCP control plane over stdio. Spec: docs/spec.md S11 and S13I.

Calls the same ``core.py`` functions the CLI calls and contains no evaluation
logic of its own. What this module adds beyond a thin wrapper is the S11.4
guard rail: an agent supplied these inputs instead of a person typing them,
so the cost cap is lower by default, a request above the ceiling is refused
outright rather than clamped, and ``suite_path`` may not escape the
configured root. No tool returns a raw provider
response - only the summary and the paths on disk (S11.1).

``mcp`` is an optional extra (``pip install evalmine[mcp]``); importing this
module without it installed raises ``ModuleNotFoundError`` from the ``mcp``
import below, which is why every caller of ``main()`` - the console script,
and any test that exercises it - either has the extra installed or expects
that error.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .core import CostRefused, UsageError, load_report
from .core import compare as core_compare
from .core import last_report as core_last_report
from .core import run_suite as core_run_suite
from .decision import generate_decision_report, judge_experiment, verify_decision, verify_judging
from .experiment import build_plan, load_experiment
from .experiment_report import generate_experiment_report, verify_experiment_report
from .external import import_external_artifacts, is_external_import
from .runner import execute_experiment, preflight_experiment, verify_execution
from .suite import load_suite
from .validators import ValidationRefused, check_experiment, verify_validation
from .workflow import load_workflow, run_workflow, verify_workflow, workflow_plan
from .workspace import prepare_experiment, verify_prepared

#: S11.4 defaults. The MCP cap is lower than the CLI's $2.00 default because
#: the human at the CLI typed the number and the agent did not.
DEFAULT_MAX_COST = 1.00
DEFAULT_MAX_COST_CEILING = 5.00

#: Everything else the library layer can raise (bad suite, unknown model,
#: provider failure, ...) is a usage/runtime error, not a money-safety
#: refusal - a tool call lets those propagate so the MCP layer reports them
#: as tool errors instead of pretending to succeed.


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _suite_root() -> Path:
    raw = os.environ.get("EVALMINE_MCP_SUITE_ROOT")
    return (Path(raw) if raw else Path.cwd()).resolve()


def _resolve_under_root(suite_path: str) -> Path | None:
    """``suite_path`` resolved inside the configured root, or ``None`` if it escapes."""
    root = _suite_root()
    candidate = Path(suite_path)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _path_refusal(path: str, kind: str = "path") -> dict[str, Any]:
    root = _suite_root()
    return {
        "refused": True,
        "reason": f"{kind}_outside_root",
        kind: path,
        "root": str(root),
        "hint": f"{kind} must resolve inside EVALMINE_MCP_SUITE_ROOT ({root})",
    }


def _suite_root_refusal(suite_path: str) -> dict[str, Any]:
    root = _suite_root()
    return {
        "refused": True,
        "reason": "suite_path_outside_root",
        "suite_path": suite_path,
        "suite_root": str(root),
        "hint": f"suite_path must resolve inside EVALMINE_MCP_SUITE_ROOT ({root})",
    }


def _effective_cap(
    agent_max_cost: float | None, suite_cap: float | None
) -> tuple[float | None, dict[str, Any] | None]:
    """The S11.4 cap resolution. A non-``None`` refusal means: spend nothing."""
    ceiling = _env_float("EVALMINE_MCP_MAX_COST_CEILING", DEFAULT_MAX_COST_CEILING)
    if agent_max_cost is not None:
        if agent_max_cost > ceiling:
            return None, {
                "refused": True,
                "reason": "max_cost_exceeds_ceiling",
                "requested_usd": agent_max_cost,
                "ceiling_usd": ceiling,
                "hint": (
                    f"the MCP cost ceiling is ${ceiling:.2f}; ask for at most that, or "
                    "raise EVALMINE_MCP_MAX_COST_CEILING"
                ),
            }
        return agent_max_cost, None
    default_cap = _env_float("EVALMINE_MCP_MAX_COST", DEFAULT_MAX_COST)
    cap = min(suite_cap, default_cap) if suite_cap is not None else default_cap
    return cap, None


def _summary_from_report(report: dict[str, Any]) -> dict[str, Any]:
    """The S11.1 summary shape, built from a ``report.json`` dict."""
    win_rates = report["win_rates"]
    per_model: list[dict[str, Any]] = []
    for model in report["models"]:
        row = report["per_model"][model]
        win = win_rates.get(model)
        if win is not None:
            win_rate = win["win_rate"]
            ci = win["ci"]
            n_pairs = win["n"]
            flip_rate = win["flip_rate"]
        else:
            # The baseline has no win-rate row of its own; report how many
            # judged pairs it appeared in, across every candidate.
            win_rate, ci, flip_rate = None, None, None
            n_pairs = sum(w["n"] for w in win_rates.values())
        per_model.append(
            {
                "model": model,
                "role": row["role"],
                "win_rate": win_rate,
                "ci": ci,
                "n_pairs": n_pairs,
                "flip_rate": flip_rate,
                "schema_pass": row["schema"]["rate"],
                "check_pass": (row.get("check") or {}).get("rate"),
                "p50_ms": row["latency"]["p50_ms"],
                "p95_ms": row["latency"]["p95_ms"],
                "cost_usd": row["cost"]["this_run_usd"],
            }
        )

    cal = report["calibration"]
    totals = report["totals"]
    return {
        "headline_eligible": report["headline_eligible"],
        "calibration": {
            "status": cal["status"],
            "kappa": cal["kappa"],
            "agreement": cal["agreement"],
            "n_labels": cal["n_labels"],
            "reason": cal.get("reason"),
        },
        "per_model": per_model,
        "totals": {
            "cost_usd": totals["cost_usd"],
            "cost_answers_usd": totals["cost_answers_usd"],
            "cost_judge_usd": totals["cost_judge_usd"],
            "live_calls": totals["live_calls"],
            "cache_hits": totals["cache_hits"],
            "excluded_pairs": totals["excluded_pairs"],
        },
        "warnings": list(report["warnings"]),
    }


# --------------------------------------------------------------------------
# suite operations (S11.1-S11.3)
# --------------------------------------------------------------------------


def run_suite_impl(
    suite_path: str,
    models: list[str],
    max_cost: float | None = None,
    baseline: str | None = None,
    no_cache: bool = False,
) -> dict[str, Any]:
    """Run a suite and return the summary and the paths - never a raw answer."""
    resolved = _resolve_under_root(suite_path)
    if resolved is None:
        return _suite_root_refusal(suite_path)
    if len(models) < 2:
        raise UsageError("run_suite needs at least two model strings: a baseline and a candidate")

    suite = load_suite(resolved)
    cap, refusal = _effective_cap(max_cost, suite.max_cost_usd)
    if refusal is not None:
        return refusal

    try:
        result = core_run_suite(
            resolved, models, baseline=baseline, max_cost=cap, no_cache=no_cache
        )
    except CostRefused as exc:
        return {
            "refused": True,
            "reason": "estimate_exceeds_cap",
            "estimate_usd": round(exc.estimate, 4),
            "cap_usd": exc.cap,
            "hint": "raise max_cost, cut --models, or run a subset",
        }

    summary = _summary_from_report(result.report)
    return {
        "run_id": result.run_id,
        "report_path": str(result.report_path),
        "report_md_path": str(result.report_md_path),
        "report_html_path": (
            str(result.report_html_path) if result.report_html_path is not None else None
        ),
        **summary,
    }


def compare_impl(report_a: str, report_b: str) -> dict[str, Any]:
    """The S9.3 delta between two reports, by run-id or path."""
    a = load_report(report_a)
    b = load_report(report_b)
    return core_compare(a, b)


def last_report_impl(suite_path: str) -> dict[str, Any]:
    """The most recent run for this suite, read from disk. Zero spend, always."""
    resolved = _resolve_under_root(suite_path)
    if resolved is None:
        return _suite_root_refusal(suite_path)

    found = core_last_report(resolved)
    if found is None:
        return {"found": False}
    report = found["report"]
    return {
        "found": True,
        "run_id": found["run_id"],
        "report_path": found["path"],
        "generated_at": report.get("generated_at"),
        "summary": _summary_from_report(report),
    }


# --------------------------------------------------------------------------
# v2 episode experiment and workflow tools
# --------------------------------------------------------------------------


def experiment_plan_impl(manifest_path: str) -> dict[str, Any]:
    resolved = _resolve_under_root(manifest_path)
    if resolved is None:
        return _path_refusal(manifest_path, "manifest_path")
    plan = build_plan(load_experiment(resolved))
    value = plan.as_dict()
    return {
        "plan_id": value["plan_id"],
        "experiment": value["experiment"],
        "question": value["question"],
        "run_count": value["schedule"]["run_count"],
        "arms": [
            {
                "id": arm["id"],
                "runner": arm["runner"],
                "model": arm["model"],
                "auth": arm["auth"],
            }
            for arm in value["arms"]
        ],
        "episodes": [
            {"id": episode["id"], "repeats": episode["repeats"]}
            for episode in value["episodes"]
        ],
        "warnings": value["warnings"],
        "provider_calls": False,
        "workspaces_created": False,
    }


def experiment_prepare_impl(manifest_path: str, out_dir: str) -> dict[str, Any]:
    manifest = _resolve_under_root(manifest_path)
    out = _resolve_under_root(out_dir)
    if manifest is None:
        return _path_refusal(manifest_path, "manifest_path")
    if out is None:
        return _path_refusal(out_dir, "out_dir")
    result = prepare_experiment(load_experiment(manifest), out)
    return {
        "plan_id": result.plan.plan_id,
        "root": str(result.root),
        "run_count": len(result.runs),
        "baseline_fingerprint": result.baseline_fingerprint,
        "provider_calls": False,
    }


def external_import_impl(bundle_path: str, out_dir: str) -> dict[str, Any]:
    bundle = _resolve_under_root(bundle_path)
    out = _resolve_under_root(out_dir)
    if bundle is None:
        return _path_refusal(bundle_path, "bundle_path")
    if out is None:
        return _path_refusal(out_dir, "out_dir")
    return import_external_artifacts(bundle, out).as_dict()


def experiment_inspect_impl(prepared_path: str) -> dict[str, Any]:
    prepared = _resolve_under_root(prepared_path)
    if prepared is None:
        return _path_refusal(prepared_path, "prepared_path")
    result = verify_prepared(prepared)
    for name, verifier in (
        ("execution", verify_execution),
        ("validation", verify_validation),
        ("report", verify_experiment_report),
        ("judging", verify_judging),
        ("decision", verify_decision),
    ):
        if (prepared / name).is_dir():
            result[name] = verifier(prepared)
    result["provider_calls"] = False
    return result


def experiment_preflight_impl(prepared_path: str) -> dict[str, Any]:
    prepared = _resolve_under_root(prepared_path)
    if prepared is None:
        return _path_refusal(prepared_path, "prepared_path")
    if is_external_import(prepared):
        return {
            "refused": True,
            "reason": "external_artifacts_already_completed",
            "provider_calls": False,
        }
    return preflight_experiment(prepared).as_dict()


def experiment_execute_impl(
    prepared_path: str,
    *,
    confirm_provider_calls: bool = False,
    confirm_external_writes: bool = False,
) -> dict[str, Any]:
    prepared = _resolve_under_root(prepared_path)
    if prepared is None:
        return _path_refusal(prepared_path, "prepared_path")
    if is_external_import(prepared):
        return {
            "refused": True,
            "reason": "external_artifacts_are_not_generated_by_evalmine",
            "provider_calls": False,
        }
    if not confirm_provider_calls or not _env_enabled("EVALMINE_MCP_ALLOW_PROVIDER_CALLS"):
        return {
            "refused": True,
            "reason": "provider_calls_not_authorized",
            "hint": (
                "set EVALMINE_MCP_ALLOW_PROVIDER_CALLS=1 on the server and pass "
                "confirm_provider_calls=true"
            ),
        }
    allow_external = confirm_external_writes and _env_enabled(
        "EVALMINE_MCP_ALLOW_EXTERNAL_WRITES"
    )
    return execute_experiment(
        prepared,
        allow_provider_calls=True,
        allow_external_writes=allow_external,
    ).as_dict()


def experiment_check_impl(
    prepared_path: str, *, confirm_validator_commands: bool = False
) -> dict[str, Any]:
    prepared = _resolve_under_root(prepared_path)
    if prepared is None:
        return _path_refusal(prepared_path, "prepared_path")
    if is_external_import(prepared):
        return {
            "refused": True,
            "reason": "external_artifacts_do_not_run_experiment_validators",
            "provider_calls": False,
        }
    allow = confirm_validator_commands and _env_enabled(
        "EVALMINE_MCP_ALLOW_VALIDATOR_COMMANDS"
    )
    try:
        return check_experiment(prepared, allow_validator_commands=allow).as_dict()
    except ValidationRefused:
        return {
            "refused": True,
            "reason": "validator_commands_not_authorized",
            "hint": (
                "set EVALMINE_MCP_ALLOW_VALIDATOR_COMMANDS=1 and pass "
                "confirm_validator_commands=true"
            ),
        }


def experiment_report_impl(
    prepared_path: str, *, ranking_style: str | None = None
) -> dict[str, Any]:
    prepared = _resolve_under_root(prepared_path)
    if prepared is None:
        return _path_refusal(prepared_path, "prepared_path")
    return generate_experiment_report(prepared, ranking_style=ranking_style).as_dict()


def experiment_judge_impl(
    prepared_path: str,
    *,
    confirm_provider_calls: bool = False,
    max_cost_usd: float | None = None,
    ranking_style: str | None = None,
    label_paths: list[str] | None = None,
) -> dict[str, Any]:
    prepared = _resolve_under_root(prepared_path)
    if prepared is None:
        return _path_refusal(prepared_path, "prepared_path")
    if not confirm_provider_calls or not _env_enabled("EVALMINE_MCP_ALLOW_PROVIDER_CALLS"):
        return {"refused": True, "reason": "provider_calls_not_authorized"}
    ceiling = _env_float("EVALMINE_MCP_MAX_COST_CEILING", DEFAULT_MAX_COST_CEILING)
    if max_cost_usd is not None and max_cost_usd > ceiling:
        return {
            "refused": True,
            "reason": "max_cost_exceeds_ceiling",
            "requested_usd": max_cost_usd,
            "ceiling_usd": ceiling,
        }
    cap = max_cost_usd if max_cost_usd is not None else _env_float(
        "EVALMINE_MCP_MAX_COST", DEFAULT_MAX_COST
    )
    resolved_labels = []
    for path in label_paths or []:
        resolved = _resolve_under_root(path)
        if resolved is None:
            return _path_refusal(path, "label_path")
        resolved_labels.append(str(resolved))
    return judge_experiment(
        prepared,
        allow_provider_calls=True,
        max_cost_usd=cap,
        ranking_style=ranking_style,
        calibration_label_paths=resolved_labels,
    ).as_dict()


def experiment_decide_impl(prepared_path: str, label_paths: list[str]) -> dict[str, Any]:
    prepared = _resolve_under_root(prepared_path)
    if prepared is None:
        return _path_refusal(prepared_path, "prepared_path")
    resolved_labels = []
    for path in label_paths:
        resolved = _resolve_under_root(path)
        if resolved is None:
            return _path_refusal(path, "label_path")
        resolved_labels.append(str(resolved))
    return generate_decision_report(prepared, resolved_labels).as_dict()


def workflow_plan_impl(manifest_path: str) -> dict[str, Any]:
    manifest = _resolve_under_root(manifest_path)
    if manifest is None:
        return _path_refusal(manifest_path, "manifest_path")
    return workflow_plan(load_workflow(manifest))


def workflow_run_impl(
    manifest_path: str,
    out_dir: str,
    *,
    confirm_commands: bool = False,
    confirm_provider_calls: bool = False,
) -> dict[str, Any]:
    manifest = _resolve_under_root(manifest_path)
    out = _resolve_under_root(out_dir)
    if manifest is None:
        return _path_refusal(manifest_path, "manifest_path")
    if out is None:
        return _path_refusal(out_dir, "out_dir")
    if not confirm_commands or not _env_enabled("EVALMINE_MCP_ALLOW_WORKFLOW_COMMANDS"):
        return {"refused": True, "reason": "workflow_commands_not_authorized"}
    workflow = load_workflow(manifest)
    needs_provider = any(node.provider_calls != "none" for node in workflow.nodes)
    allow_provider = confirm_provider_calls and _env_enabled(
        "EVALMINE_MCP_ALLOW_PROVIDER_CALLS"
    )
    if needs_provider and not allow_provider:
        return {"refused": True, "reason": "provider_calls_not_authorized"}
    return run_workflow(
        workflow,
        out,
        allow_commands=True,
        allow_provider_calls=allow_provider,
    ).as_dict()


def workflow_inspect_impl(workflow_path: str) -> dict[str, Any]:
    resolved = _resolve_under_root(workflow_path)
    if resolved is None:
        return _path_refusal(workflow_path, "workflow_path")
    return verify_workflow(resolved)


# --------------------------------------------------------------------------
# MCP wiring
# --------------------------------------------------------------------------

try:
    from mcp.server.mcpserver import MCPServer

    server: MCPServer | None = MCPServer(
        "evalmine",
        instructions=(
            "Plan, prepare, execute, inspect, label, judge, and report evalmine suites, "
            "episode experiments, and controlled workflows. Spend and process launches "
            "are gated independently. Read-only tools return summaries and paths, never "
            "raw provider responses."
        ),
    )

    @server.tool(structured_output=True)
    def run_suite(
        suite_path: str,
        models: list[str],
        max_cost: float | None = None,
        baseline: str | None = None,
        no_cache: bool = False,
    ) -> dict[str, Any]:
        """Run an evalmine suite over two or more models and return the summary."""
        return run_suite_impl(
            suite_path, models, max_cost=max_cost, baseline=baseline, no_cache=no_cache
        )

    @server.tool(structured_output=True)
    def compare(report_a: str, report_b: str) -> dict[str, Any]:
        """Print the S9.3 delta between two reports (by run-id or path)."""
        return compare_impl(report_a, report_b)

    @server.tool(structured_output=True)
    def last_report(suite_path: str) -> dict[str, Any]:
        """The most recent report for this suite, read from disk. Zero spend."""
        return last_report_impl(suite_path)

    @server.tool(structured_output=True)
    def plan_experiment(manifest_path: str) -> dict[str, Any]:
        """Validate and summarize a v2 episode experiment without creating anything."""
        return experiment_plan_impl(manifest_path)

    @server.tool(structured_output=True)
    def prepare_experiment_tool(manifest_path: str, out_dir: str) -> dict[str, Any]:
        """Create isolated workspaces and immutable inputs; launches no provider."""
        return experiment_prepare_impl(manifest_path, out_dir)

    @server.tool(structured_output=True)
    def import_external_artifacts_tool(bundle_path: str, out_dir: str) -> dict[str, Any]:
        """Pin completed JSONL artifacts for blind evaluation; launches no provider."""
        return external_import_impl(bundle_path, out_dir)

    @server.tool(structured_output=True)
    def inspect_experiment(prepared_path: str) -> dict[str, Any]:
        """Verify all evidence envelopes currently present for an experiment."""
        return experiment_inspect_impl(prepared_path)

    @server.tool(structured_output=True)
    def preflight_experiment_tool(prepared_path: str) -> dict[str, Any]:
        """Probe runner capabilities without authenticating or contacting a provider."""
        return experiment_preflight_impl(prepared_path)

    @server.tool(structured_output=True)
    def execute_experiment_tool(
        prepared_path: str,
        confirm_provider_calls: bool = False,
        confirm_external_writes: bool = False,
    ) -> dict[str, Any]:
        """Execute prepared arms only when client and server gates both authorize calls."""
        return experiment_execute_impl(
            prepared_path,
            confirm_provider_calls=confirm_provider_calls,
            confirm_external_writes=confirm_external_writes,
        )

    @server.tool(structured_output=True)
    def check_experiment_tool(
        prepared_path: str, confirm_validator_commands: bool = False
    ) -> dict[str, Any]:
        """Run objective checks; command validators require a separate two-part gate."""
        return experiment_check_impl(
            prepared_path, confirm_validator_commands=confirm_validator_commands
        )

    @server.tool(structured_output=True)
    def report_experiment(
        prepared_path: str, ranking_style: str | None = None
    ) -> dict[str, Any]:
        """Generate the blind self-contained HTML labeling queue."""
        return experiment_report_impl(prepared_path, ranking_style=ranking_style)

    @server.tool(structured_output=True)
    def judge_experiment_tool(
        prepared_path: str,
        confirm_provider_calls: bool = False,
        max_cost_usd: float | None = None,
        ranking_style: str | None = None,
        label_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """Run the configured ranking protocol behind provider and MCP cost gates."""
        return experiment_judge_impl(
            prepared_path,
            confirm_provider_calls=confirm_provider_calls,
            max_cost_usd=max_cost_usd,
            ranking_style=ranking_style,
            label_paths=label_paths,
        )

    @server.tool(structured_output=True)
    def decide_experiment(prepared_path: str, label_paths: list[str]) -> dict[str, Any]:
        """Import label exports, calibrate, score, and write decision HTML."""
        return experiment_decide_impl(prepared_path, label_paths)

    @server.tool(structured_output=True)
    def plan_workflow(manifest_path: str) -> dict[str, Any]:
        """Validate and summarize a workflow DAG without copying or launching."""
        return workflow_plan_impl(manifest_path)

    @server.tool(structured_output=True)
    def run_workflow_tool(
        manifest_path: str,
        out_dir: str,
        confirm_commands: bool = False,
        confirm_provider_calls: bool = False,
    ) -> dict[str, Any]:
        """Run a contained workflow behind command and optional provider gates."""
        return workflow_run_impl(
            manifest_path,
            out_dir,
            confirm_commands=confirm_commands,
            confirm_provider_calls=confirm_provider_calls,
        )

    @server.tool(structured_output=True)
    def inspect_workflow(workflow_path: str) -> dict[str, Any]:
        """Verify immutable workflow evidence and its final workspace hash."""
        return workflow_inspect_impl(workflow_path)

except ImportError:  # pragma: no cover - exercised only without the [mcp] extra
    server = None


def main() -> None:
    if server is None:  # pragma: no cover - exercised only without the [mcp] extra
        raise SystemExit(
            "the mcp package is not installed; run `pip install evalmine[mcp]` to use "
            "evalmine-mcp"
        )
    server.run("stdio")


if __name__ == "__main__":  # pragma: no cover
    main()
