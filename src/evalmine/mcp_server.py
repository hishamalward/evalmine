"""The MCP surface: three tools over stdio. Spec: docs/spec.md S11.

Calls the same ``core.py`` functions the CLI calls and contains no evaluation
logic of its own. What this module adds beyond a thin wrapper is the S11.4
guard rail: an agent supplied these inputs instead of a person typing them,
so the cost cap is lower by default, a request above the ceiling is refused
outright rather than clamped, and ``suite_path`` may not escape the
configured root. None of these three tools ever returns a raw provider
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
from .suite import load_suite

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
# the three tools (S11.1-S11.3)
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
# MCP wiring
# --------------------------------------------------------------------------

try:
    from mcp.server.mcpserver import MCPServer

    server: MCPServer | None = MCPServer(
        "evalmine",
        instructions=(
            "Run your own evalmine suites and read their results. Every call is capped "
            "in USD (S11.4) and refuses, rather than truncates, a run that would exceed "
            "its cap. run_suite spends money (up to the cap); compare and last_report "
            "only read reports already on disk."
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
