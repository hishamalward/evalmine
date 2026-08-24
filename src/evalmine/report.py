"""report.json (the record) and report.md (the human view, rendered from it).

Spec: docs/spec.md S9. The report contains no adjectives, no recommendation and
no "surprisingly": judgment goes in DECISIONS.md, written by a person. What the
report does do is refuse to let a number be read without the thing that
qualifies it - a win-rate without its n, a kappa without its band, a cost
without the date of the price table it came from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import __version__
from .judge import EXCLUSION_REASONS
from .metrics import (
    FLIP_RATE_WARNING,
    bootstrap_ci,
    calibration_status,
    check_pass_rate,
    cohens_kappa,
    format_kappa,
    judge_category,
    latency_stats,
    median,
    per_task_agreement,
    schema_pass_rate,
    seed_from_suite_hash,
    win_rate,
)

if TYPE_CHECKING:  # pragma: no cover
    from .core import RunData

CATEGORY_KEYS = (
    "consistent_win",
    "consistent_loss",
    "tie",
    "soft_win",
    "soft_loss",
    "flip",
)


def _r(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


# --------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------


def build_report(data: RunData) -> dict[str, Any]:
    suite = data.suite
    answers = data.answers
    pairs = data.pairs

    per_model = _per_model(data)
    calibration = _calibration(data)
    win_rates = _win_rates(data)
    per_task = _per_task(data, win_rates)

    cost_answers = sum(a.cost or 0.0 for a in answers)
    cost_uncached = sum(
        (a.cost_if_uncached if a.cost_if_uncached is not None else 0.0) for a in answers
    )
    missing_usage = sum(1 for a in answers if a.status == "ok" and a.cost is None)
    total_calls = len(answers)
    cache_hits = sum(1 for a in answers if a.cached)

    report: dict[str, Any] = {
        "tool_version": __version__,
        "report_version": 1,
        "generated_at": data.generated_at,
        "run_id": data.run_id,
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "hash": suite.hash,
            "path": str(suite.path).replace("\\", "/"),
            "task_hashes": suite.task_hashes,
            "n_tasks": len(suite.tasks),
            "n_cases": sum(len(t.cases) for t in suite.tasks),
            "n_labels": len(suite.labels),
        },
        "models": list(data.models),
        "baseline": data.baseline,
        "candidates": list(data.candidates),
        "judge": {
            "model": data.judge_model,
            "under_test": data.judge_model in data.models,
            "max_tokens": suite.judge.max_tokens,
            "temperature": suite.judge.temperature,
            "calls": data.judge_calls,
            "cached_calls": data.judge_cached_calls,
            "latency": _round_latency(
                latency_stats(
                    data.judge_latencies,
                    len(data.judge_latencies) - data.judge_cached_calls,
                )
            ),
        },
        "prices": {
            "file": data.prices.filename,
            "pinned": data.prices.pinned,
            "currency": data.prices.currency,
            "verified": data.prices.verified,
        },
        "prices_verified": data.prices.verified,
        "run": {
            "repeats": data.repeats,
            "fake": data.fake,
            "no_cache": data.no_cache,
            "max_cost_usd": data.cap,
            "preflight_estimate_usd": _r(data.estimate),
            "cache_dir": str(data.cache_dir).replace("\\", "/"),
        },
        "aborted_over_budget": data.aborted_over_budget,
        "cost_incomplete": missing_usage > 0,
        "headline_eligible": calibration["headline_eligible"],
        "calibration": calibration,
        "win_rates": win_rates,
        "per_model": per_model,
        "per_task": per_task,
        "totals": {
            "answers": total_calls,
            "pairs": len(pairs),
            "judge_passes": sum(len(p.passes) for p in pairs),
            "cost_usd": _r(cost_answers + data.judge_cost),
            "cost_answers_usd": _r(cost_answers),
            "cost_judge_usd": _r(data.judge_cost),
            "cost_if_uncached_usd": _r(cost_uncached + data.judge_cost_if_uncached),
            "cost_answers_if_uncached_usd": _r(cost_uncached),
            "cost_judge_if_uncached_usd": _r(data.judge_cost_if_uncached),
            "missing_usage_calls": missing_usage,
            "live_calls": data.live_calls,
            "cache_hits": cache_hits,
            "cache_hit_rate": _r(cache_hits / total_calls) if total_calls else None,
            "excluded_pairs": sum(1 for p in pairs if p.excluded),
        },
        "exclusions": [
            {
                "task": p.task_id,
                "case": p.case_id,
                "repeat": p.repeat,
                "candidate": p.candidate,
                "reason": p.reason,
            }
            for p in pairs
            if p.excluded
        ],
        "errors": [
            {"task": a.task_id, "case": a.case_id, "model": a.model, "error": a.error}
            for a in answers
            if a.status == "error"
        ],
        "check_failures": [
            {
                "task": a.task_id,
                "case": a.case_id,
                "model": a.model,
                "status": a.check_status,
                "exit_code": a.check_exit,
                "output_head": (a.check_output or "").strip().splitlines()[:1],
            }
            for a in answers
            if a.check_status in ("fail", "error")
        ],
        "unparseable_judge_passes": [
            {"task": p.task_id, "case": p.case_id, "candidate": p.candidate, "order": q.order}
            for p in pairs
            for q in p.passes
            if q.unparseable
        ],
        "warnings": list(data.warnings),
        "reproduce": {
            "command": data.command,
            "price_file": data.prices.filename,
            "cache_dir": str(data.cache_dir).replace("\\", "/"),
            "live": data.live_calls > 0,
        },
    }

    report["what_changed"] = (
        diff_reports(data.previous_report, report, mover_threshold=suite.mover_threshold)
        if data.previous_report
        else None
    )
    report["decision_log_entry"] = decision_log_entry(report)
    return report


def _round_latency(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (_r(value) if isinstance(value, float) else value) for key, value in stats.items()
    }


def _per_model(data: RunData) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in data.models:
        rows = [a for a in data.answers if a.model == model]
        schema_rows = [a for a in rows if a.schema_status != "not_applicable"]
        latencies = [a.latency_ms for a in rows if a.latency_ms is not None]
        live = sum(1 for a in rows if not a.cached and a.latency_ms is not None)
        cost_this_run = sum(a.cost or 0.0 for a in rows)
        cost_uncached = sum(a.cost_if_uncached or 0.0 for a in rows)
        missing = sum(1 for a in rows if a.status == "ok" and a.cost is None)
        n_calls = len(rows)
        out[model] = {
            "role": "baseline" if model == data.baseline else "candidate",
            "calls": n_calls,
            "cache_hits": sum(1 for a in rows if a.cached),
            "errors": sum(1 for a in rows if a.status == "error"),
            "schema": {
                **schema_pass_rate([a.schema_status for a in schema_rows]),
                "modes": sorted({a.schema_mode for a in schema_rows}) or None,
            },
            "check": {
                **check_pass_rate(
                    [a.check_status for a in rows if a.check_status != "not_applicable"]
                ),
                "multi_block": sum(1 for a in rows if len(a.check_blocks) > 1),
            },
            "latency": _round_latency(latency_stats(latencies, live)),
            "cost": {
                "this_run_usd": _r(cost_this_run),
                "if_uncached_usd": _r(cost_uncached),
                "per_1k_calls_usd": _r((cost_uncached / n_calls) * 1000) if n_calls else None,
                "missing_usage_calls": missing,
                "incomplete": missing > 0,
            },
        }
    return out


def _label_key(task: str, case: str, baseline: str, candidate: str) -> tuple[str, str, str, str]:
    return (task, case, baseline, candidate)


def _calibration(data: RunData) -> dict[str, Any]:
    scored = {
        _label_key(p.task_id, p.case_id, p.baseline, p.candidate): p
        for p in data.pairs
        if p.repeat == 0 and not p.excluded and p.score is not None
    }
    excluded = {
        _label_key(p.task_id, p.case_id, p.baseline, p.candidate): p
        for p in data.pairs
        if p.repeat == 0 and p.excluded
    }

    used: list[tuple[str, str]] = []
    used_by_task: list[tuple[str, str, str]] = []
    ignored: list[dict[str, Any]] = []
    for label in data.suite.labels:
        key = _label_key(label.task, label.case, label.baseline, label.candidate)
        pair = scored.get(key)
        if pair is not None:
            used.append((judge_category(pair.score), label.prefer))
            used_by_task.append((label.task, judge_category(pair.score), label.prefer))
            continue
        if key in excluded:
            ignored.append(
                {
                    "task": label.task,
                    "case": label.case,
                    "baseline": label.baseline,
                    "candidate": label.candidate,
                    "why": f"pair excluded from the win-rate ({excluded[key].reason})",
                }
            )
        else:
            ignored.append(
                {
                    "task": label.task,
                    "case": label.case,
                    "baseline": label.baseline,
                    "candidate": label.candidate,
                    "why": "this baseline/candidate pair was not in this run",
                }
            )

    kappa = cohens_kappa(used)
    result = calibration_status(
        kappa,
        min_kappa=data.suite.judge.calibration.min_kappa,
        min_labels=data.suite.judge.calibration.min_labels,
    )
    result["po"] = _r(result.get("po"))
    result["pe"] = _r(result.get("pe"))
    result["agreement"] = _r(result.get("agreement"))
    result["kappa"] = _r(result.get("kappa"))
    result["n_labels_in_suite"] = len(data.suite.labels)
    result["n_labels_ignored"] = len(ignored)
    result["ignored_labels"] = ignored
    result["on_below_floor"] = data.suite.judge.calibration.on_below_floor
    result["per_task_agreement"] = [
        {**row, "agreement": _r(row["agreement"]), "kappa": _r(row["kappa"])}
        for row in per_task_agreement(used_by_task)
    ]
    return result


def _win_rates(data: RunData) -> dict[str, Any]:
    seed = seed_from_suite_hash(data.suite.hash)
    out: dict[str, Any] = {}
    for candidate in data.candidates:
        pairs = [p for p in data.pairs if p.candidate == candidate]
        included = [p for p in pairs if not p.excluded and p.score is not None]
        scores = [p.score for p in included]
        categories = {key: 0 for key in CATEGORY_KEYS}
        for pair in included:
            if pair.category in categories:
                categories[pair.category] += 1
        excluded_by_reason = {reason: 0 for reason in EXCLUSION_REASONS}
        for pair in pairs:
            if pair.excluded and pair.reason in excluded_by_reason:
                excluded_by_reason[pair.reason] += 1

        ci = bootstrap_ci(scores, seed) if scores else None
        n = len(included)
        flips = categories["flip"]
        per_task: dict[str, Any] = {}
        for task in data.suite.tasks:
            task_scores = [p.score for p in included if p.task_id == task.id]
            per_task[task.id] = {
                "win_rate": _r(win_rate(task_scores)),
                "n": len(task_scores),
                "excluded": sum(
                    1 for p in pairs if p.task_id == task.id and p.excluded
                ),
            }

        out[candidate] = {
            "vs": data.baseline,
            "win_rate": _r(win_rate(scores)),
            "ci": [_r(ci[0]), _r(ci[1])] if ci else None,
            "ci_suppressed": ci is None,
            "n": n,
            "counts": categories,
            "flips": flips,
            "flip_rate": _r(flips / n) if n else None,
            "flip_rate_above_warning": bool(n and (flips / n) > FLIP_RATE_WARNING),
            "excluded": sum(1 for p in pairs if p.excluded),
            "excluded_by_reason": excluded_by_reason,
            "per_task": per_task,
            "basis": "schema-passing pairs only",
        }
    return out


def _per_task(data: RunData, win_rates: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in data.suite.tasks:
        answers = [a for a in data.answers if a.task_id == task.id]
        models: dict[str, Any] = {}
        for model in data.models:
            model_rows = [a for a in answers if a.model == model]
            schema_rows = [a for a in model_rows if a.schema_status != "not_applicable"]
            latencies = [a.latency_ms for a in model_rows if a.latency_ms is not None]
            models[model] = {
                "schema": schema_pass_rate([a.schema_status for a in schema_rows]),
                "check": {
                    **check_pass_rate(
                        [
                            a.check_status
                            for a in model_rows
                            if a.check_status != "not_applicable"
                        ]
                    ),
                    "multi_block": sum(1 for a in model_rows if len(a.check_blocks) > 1),
                },
                "p50_ms": _r(median(latencies)),
                "cost_if_uncached_usd": _r(sum(a.cost_if_uncached or 0.0 for a in model_rows)),
            }
        rows.append(
            {
                "task": task.id,
                "kind": task.kind,
                "judged": task.judge,
                "has_schema": task.schema is not None,
                "has_check": any(c.check is not None for c in task.cases),
                "n_cases": len(task.cases),
                "models": models,
                "candidates": {
                    candidate: win_rates[candidate]["per_task"][task.id]
                    for candidate in data.candidates
                },
            }
        )

    def sort_key(row: dict[str, Any]) -> tuple[float, str]:
        if not data.candidates:
            return (1.0, row["task"])
        first = row["candidates"][data.candidates[0]]["win_rate"]
        # tasks the new model is worst at come first; unjudged tasks last
        return (2.0 if first is None else first, row["task"])

    rows.sort(key=sort_key)
    return rows


# --------------------------------------------------------------------------
# S9.3 "what changed"
# --------------------------------------------------------------------------


def diff_reports(
    old: dict[str, Any], new: dict[str, Any], mover_threshold: float = 0.15
) -> dict[str, Any]:
    """The delta between two reports for the same suite name (S9.3)."""
    same_suite_name = old["suite"]["name"] == new["suite"]["name"]
    comparable = same_suite_name and old["suite"]["hash"] == new["suite"]["hash"]

    diff: dict[str, Any] = {
        "against_run_id": old["run_id"],
        "against_generated_at": old.get("generated_at"),
        "comparable": comparable,
        "reason": None,
        "suite_hash_changed": old["suite"]["hash"] != new["suite"]["hash"],
        "models_added": [m for m in new["models"] if m not in old["models"]],
        "models_removed": [m for m in old["models"] if m not in new["models"]],
        "price_table": {
            "from": old["prices"]["file"],
            "to": new["prices"]["file"],
            "changed": old["prices"]["file"] != new["prices"]["file"],
        },
        "labels": {
            "from": old["suite"].get("n_labels"),
            "to": new["suite"].get("n_labels"),
            "changed": old["suite"].get("n_labels") != new["suite"].get("n_labels"),
        },
        "cache_note": {
            "cache_hit_rate": new["totals"].get("cache_hit_rate"),
            "note": (
                "a delta computed entirely from cached answers means the judge or the "
                "scoring changed, not the models"
            ),
        },
    }

    if not same_suite_name:
        diff["reason"] = (
            f"different suites: {old['suite']['name']!r} and {new['suite']['name']!r}"
        )

    old_tasks = old["suite"].get("task_hashes", {})
    new_tasks = new["suite"].get("task_hashes", {})
    diff["tasks_added"] = sorted(set(new_tasks) - set(old_tasks))
    diff["tasks_removed"] = sorted(set(old_tasks) - set(new_tasks))
    diff["tasks_modified"] = sorted(
        t for t in set(old_tasks) & set(new_tasks) if old_tasks[t] != new_tasks[t]
    )

    if not comparable:
        if diff["reason"] is None:
            diff["reason"] = (
                f"suite changed since {old['run_id']} - task-level deltas are not comparable"
            )
        return diff

    diff["calibration"] = {
        "kappa": _delta(old["calibration"].get("kappa"), new["calibration"].get("kappa")),
        "agreement": _delta(
            old["calibration"].get("agreement"), new["calibration"].get("agreement")
        ),
        "status": {"from": old["calibration"]["status"], "to": new["calibration"]["status"]},
    }

    candidates: dict[str, Any] = {}
    movers: list[dict[str, Any]] = []
    for candidate, current in new["win_rates"].items():
        before = old["win_rates"].get(candidate)
        if before is None:
            continue
        candidates[candidate] = {
            "win_rate": _delta(before.get("win_rate"), current.get("win_rate")),
            "n": {"from": before.get("n"), "to": current.get("n")},
            "flip_rate": _delta(before.get("flip_rate"), current.get("flip_rate")),
        }
        for task, after in current.get("per_task", {}).items():
            was = before.get("per_task", {}).get(task)
            if not was or was.get("win_rate") is None or after.get("win_rate") is None:
                continue
            change = after["win_rate"] - was["win_rate"]
            if abs(change) > mover_threshold:
                movers.append(
                    {
                        "candidate": candidate,
                        "task": task,
                        "from": was["win_rate"],
                        "to": after["win_rate"],
                        "delta": _r(change),
                        "n": after.get("n"),
                    }
                )
    diff["candidates"] = candidates
    diff["movers"] = sorted(movers, key=lambda m: (m["candidate"], m["task"]))
    diff["mover_threshold"] = mover_threshold

    models: dict[str, Any] = {}
    for model, current in new["per_model"].items():
        before = old["per_model"].get(model)
        if before is None:
            continue
        models[model] = {
            "schema_pass": _delta(before["schema"].get("rate"), current["schema"].get("rate")),
            "check_pass": _delta(
                (before.get("check") or {}).get("rate"), (current.get("check") or {}).get("rate")
            ),
            "p50_ms": _delta(before["latency"].get("p50_ms"), current["latency"].get("p50_ms")),
            "p95_ms": _delta(before["latency"].get("p95_ms"), current["latency"].get("p95_ms")),
            "cost_if_uncached_usd": _delta(
                before["cost"].get("if_uncached_usd"), current["cost"].get("if_uncached_usd")
            ),
        }
    diff["models"] = models
    return diff


def _delta(before: float | None, after: float | None) -> dict[str, Any]:
    change = None if before is None or after is None else _r(after - before)
    return {"from": before, "to": after, "delta": change}


# --------------------------------------------------------------------------
# S9.4 decision-log entry
# --------------------------------------------------------------------------


def decision_log_entry(report: dict[str, Any]) -> str:
    """The template, pre-filled with this run's numbers (S9.4).

    The tool prints it; the tool does not write DECISIONS.md. This is the one
    artefact a human must author.
    """
    date = (report.get("generated_at") or "")[:10]
    baseline = report["baseline"]
    candidate = report["candidates"][0] if report["candidates"] else "(none)"
    win = report["win_rates"].get(candidate, {}) if report["candidates"] else {}
    cal = report["calibration"]

    ci = win.get("ci")
    ci_text = f"[{ci[0]:.2f}-{ci[1]:.2f}]" if ci else "[CI: n too small]"
    win_text = "n/a" if win.get("win_rate") is None else f"{win['win_rate']:.3f}"
    per_model = report["per_model"]

    def schema_pct(model: str) -> str:
        rate = per_model.get(model, {}).get("schema", {}).get("rate")
        return "n/a" if rate is None else f"{rate * 100:.0f}%"

    def p95(model: str) -> str:
        value = per_model.get(model, {}).get("latency", {}).get("p95_ms")
        return "n/a" if value is None else f"{value:.0f}"

    def cost(model: str) -> str:
        value = per_model.get(model, {}).get("cost", {}).get("if_uncached_usd")
        return "n/a" if value is None else f"{value:.4f}"

    flag = "" if report["headline_eligible"] else " (UNCALIBRATED - not a headline)"

    return "\n".join(
        [
            f"## {date} - {report['suite']['name']} - {baseline} vs {candidate}",
            "",
            f"- **Run:** {report['run_id']} - report: "
            f"reports/{report['suite']['name']}/{report['run_id']}/report.md",
            "- **Question:** <the decision this run was supposed to inform>",
            f"- **Numbers:** win-rate {win_text}{flag} {ci_text}, n={win.get('n', 0)}, "
            f"flips {win.get('flips', 0)} - "
            f"kappa {format_kappa(cal.get('kappa'))} (agreement "
            f"{cal.get('agreement') if cal.get('agreement') is not None else 'n/a'}, "
            f"{cal.get('n_labels', 0)} labels) - "
            f"schema pass {schema_pct(baseline)} -> {schema_pct(candidate)} - "
            f"p95 {p95(baseline)}ms -> {p95(candidate)}ms - "
            f"cost/run ${cost(baseline)} -> ${cost(candidate)}",
            "- **Decision:** adopt | reject | inconclusive | adopt-for-<subset>",
            "- **Why:** <two sentences, in terms of the numbers above>",
            "- **What would change this:** <the result that would reverse the decision>",
            "- **Not measured:** <what this suite does not cover that the decision assumes>",
        ]
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def _check_cell(check: dict[str, Any] | None) -> str:
    """``n/a`` when no answer in the row had a check, else the pass rate with its n."""
    if not check or not check.get("n"):
        return "n/a"
    cell = f"{_pct(check['rate'])} (n={check['n']}"
    if check.get("multi_block"):
        cell += f", {check['multi_block']} multi-block"
    return cell + ")"


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    suite = report["suite"]
    eligible = report["headline_eligible"]
    dagger = "" if eligible else "&dagger;"

    # 1. header
    add(f"# {suite['name']} - run {report['run_id']}")
    add("")
    if suite.get("description"):
        add(suite["description"].strip())
        add("")
    add(f"- Generated: `{report['generated_at']}` by evalmine {report['tool_version']}")
    add(f"- Suite: `{suite['path']}` (hash `{suite['hash'][:12]}`, "
        f"{suite['n_tasks']} tasks, {suite['n_cases']} cases, {suite['n_labels']} labels)")
    models = ", ".join(
        f"`{m}`" + (" (baseline)" if m == report["baseline"] else "") for m in report["models"]
    )
    add(f"- Models: {models}")
    judge_note = " - **also under test**" if report["judge"]["under_test"] else ""
    add(f"- Judge: `{report['judge']['model']}`{judge_note}")
    price_note = "" if report["prices"]["verified"] else " (UNVERIFIED - figures are placeholders)"
    add(f"- Price table: `{report['prices']['file']}` pinned {report['prices']['pinned']}"
        f"{price_note}")
    totals = report["totals"]
    add(f"- Calls: {totals['answers']} answers, {report['judge']['calls']} judge passes - "
        f"cache hit rate {_pct(totals['cache_hit_rate'])}, {totals['live_calls']} live")
    add(f"- Cost this run: {_usd(totals['cost_usd'])} "
        f"(answers {_usd(totals['cost_answers_usd'])}, judge {_usd(totals['cost_judge_usd'])}) - "
        f"if uncached: {_usd(totals['cost_if_uncached_usd'])}")
    if report["cost_incomplete"]:
        add(f"- **Cost is incomplete**: {totals['missing_usage_calls']} calls returned no token "
            "counts, so the totals above are lower bounds")
    if report["aborted_over_budget"]:
        add(f"- **ABORTED OVER BUDGET** at the ${report['run']['max_cost_usd']:.2f} ceiling - "
            "this report is partial and is not a complete run")
    add("")

    # 2. calibration, deliberately above the win-rates
    cal = report["calibration"]
    add("## Calibration (judge vs your labels)")
    add("")
    if not eligible:
        add(f"> **Not headline-eligible: {cal['status']}.** {cal.get('reason') or ''}")
        add(">")
        add("> Win-rates below are flagged and must not be quoted as an established result.")
        add("")
    add(f"- Status: `{cal['status']}` - headline eligible: **{str(eligible).lower()}**")
    add(f"- Cohen's kappa: **{format_kappa(cal.get('kappa'))}** "
        f"(floor {cal['min_kappa']:.2f}, minimum labels {cal['min_labels']})")
    add(f"- Plain agreement: {_pct(cal.get('agreement'))} over {cal['n_labels']} labelled pairs "
        "(agreement is inflated whenever one category dominates - read it next to kappa, "
        "never instead of it)")
    add(f"- Labels in suite: {cal['n_labels_in_suite']} - used: {cal['n_labels']} - "
        f"ignored: {cal['n_labels_ignored']}")
    if cal["ignored_labels"]:
        add("")
        add("Ignored labels:")
        add("")
        for item in cal["ignored_labels"][:20]:
            add(f"- `{item['task']}/{item['case']}` "
                f"({item['baseline']} vs {item['candidate']}): {item['why']}")
        if len(cal["ignored_labels"]) > 20:
            add(f"- ... and {len(cal['ignored_labels']) - 20} more")
    add("")
    add("Confusion matrix (rows: you, columns: judge):")
    add("")
    add("| you \\ judge | baseline | candidate | tie |")
    add("|---|---|---|---|")
    for human in ("baseline", "candidate", "tie"):
        row = cal["confusion"][human]
        add(f"| **{human}** | {row['baseline']} | {row['candidate']} | {row['tie']} |")
    add("")
    add(_render_per_task_agreement(cal))

    # 3. win-rates
    title = "Win-rates" if eligible else "Win-rates (UNCALIBRATED - not a headline)"
    add(f"## {title}")
    add("")
    add("Over schema-passing pairs only: a pair where either side failed to parse or "
        "failed its schema is excluded, not scored a loss (ruling O-3). Read these "
        "beside the schema-pass rates in the scorecard below.")
    add("")
    add("| candidate | win-rate | 95% CI | n | wins | losses | ties | soft W/L | flips | "
        "flip rate | excluded |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for candidate, win in report["win_rates"].items():
        counts = win["counts"]
        ci = win["ci"]
        ci_text = f"{ci[0]:.3f}-{ci[1]:.3f}" if ci else "n too small"
        rate = "n/a" if win["win_rate"] is None else f"{win['win_rate']:.3f}{dagger}"
        flip_rate = _pct(win["flip_rate"])
        if win["flip_rate_above_warning"]:
            flip_rate = f"**{flip_rate}**"
        add(f"| `{candidate}` | {rate} | {ci_text} | {win['n']} | "
            f"{counts['consistent_win']} | {counts['consistent_loss']} | {counts['tie']} | "
            f"{counts['soft_win']}/{counts['soft_loss']} | {win['flips']} | {flip_rate} | "
            f"{win['excluded']} |")
    add("")
    if any(w["flip_rate_above_warning"] for w in report["win_rates"].values()):
        add(f"**A flip rate above {FLIP_RATE_WARNING:.2f} means the judge changed its mind when "
            "the answers changed places more often than a measurement can survive. The "
            "bolded win-rate above is not a measurement.**")
        add("")
    if not eligible:
        add(f"{dagger} uncalibrated: {cal.get('reason') or cal['status']}")
        add("")

    # 4. per-model scorecard
    add("## Per-model scorecard")
    add("")
    add("| model | role | schema pass | mode | parse fail | schema fail | exec pass | "
        "p50 ms | p95 ms | live | cost this run | cost if uncached | $/1k calls |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for model, row in report["per_model"].items():
        schema = row["schema"]
        latency = row["latency"]
        cost = row["cost"]
        modes = ", ".join(schema["modes"]) if schema["modes"] else "-"
        p95 = f"{_num(latency['p95_ms'])} (n={latency['n']})"
        add(f"| `{model}` | {row['role']} | {_pct(schema['rate'])} | {modes} | "
            f"{schema['parse_fail']} | {schema['schema_fail']} | "
            f"{_check_cell(row.get('check'))} | {_num(latency['p50_ms'])} | "
            f"{p95} | {_pct(latency['live_fraction'])} | {_usd(cost['this_run_usd'])} | "
            f"{_usd(cost['if_uncached_usd'])} | {_usd(cost['per_1k_calls_usd'])} |")
    add("")
    add("p95 is nearest-rank; below n=20 it is the maximum, which is why n is printed with it.")
    add("")

    # 5. per-task table
    add("## Per-task")
    add("")
    add("Sorted by the first candidate's win-rate, ascending: what the new model is worst at "
        "is the first thing on screen.")
    add("")
    header = ["task", "kind", "cases"]
    show_check = any(row.get("has_check") for row in report["per_task"])
    for model in report["models"]:
        header.append(f"schema {model}")
    if show_check:
        for model in report["models"]:
            header.append(f"exec {model}")
    for candidate in report["candidates"]:
        header.append(f"win {candidate}")
    header += ["p50 ms (baseline)", "cost if uncached"]
    add("| " + " | ".join(header) + " |")
    add("|" + "---|" * len(header))
    for row in report["per_task"]:
        cells = [f"`{row['task']}`", row["kind"] or "-", str(row["n_cases"])]
        for model in report["models"]:
            cells.append(_pct(row["models"][model]["schema"]["rate"]))
        if show_check:
            for model in report["models"]:
                cells.append(_check_cell(row["models"][model].get("check")))
        for candidate in report["candidates"]:
            entry = row["candidates"][candidate]
            value = "n/a" if entry["win_rate"] is None else f"{entry['win_rate']:.2f}{dagger}"
            cells.append(f"{value} (n={entry['n']})")
        cells.append(_num(row["models"][report["baseline"]]["p50_ms"]))
        cells.append(
            _usd(sum(row["models"][m]["cost_if_uncached_usd"] or 0.0 for m in report["models"]))
        )
        add("| " + " | ".join(cells) + " |")
    add("")

    # 6. what changed
    changed = report.get("what_changed")
    add("## What changed")
    add("")
    if not changed:
        add("No previous report for this suite.")
        add("")
    else:
        add(_render_what_changed(changed))

    # 7. failures and exclusions
    add("## Failures and exclusions")
    add("")
    excluded = report["exclusions"]
    check_failures = report.get("check_failures") or []
    if (
        not excluded
        and not report["errors"]
        and not report["unparseable_judge_passes"]
        and not check_failures
    ):
        add("None.")
        add("")
    else:
        if excluded:
            add(f"{len(excluded)} excluded pairs:")
            add("")
            for item in excluded:
                add(f"- `{item['task']}/{item['case']}` vs `{item['candidate']}`: "
                    f"{item['reason']}")
            add("")
        if report["errors"]:
            add("Provider errors:")
            add("")
            for item in report["errors"]:
                add(f"- `{item['model']}` on `{item['task']}/{item['case']}`: {item['error']}")
            add("")
        if check_failures:
            add("Execution check failures (S6.6):")
            add("")
            for item in check_failures:
                head = item["output_head"][0] if item["output_head"] else ""
                exit_code = (
                    "no exit code" if item["exit_code"] is None else f"exit {item['exit_code']}"
                )
                add(f"- `{item['model']}` on `{item['task']}/{item['case']}`: "
                    f"{item['status']} ({exit_code}) {head}".rstrip())
            add("")
        if report["unparseable_judge_passes"]:
            add("Unparseable judge passes:")
            add("")
            for item in report["unparseable_judge_passes"]:
                add(f"- `{item['task']}/{item['case']}` vs `{item['candidate']}` "
                    f"(order {item['order']})")
            add("")

    if report["warnings"]:
        add("Warnings:")
        add("")
        for warning in report["warnings"]:
            add(f"- {warning}")
        add("")

    # 8. reproduce
    add("## Reproduce")
    add("")
    command = report["reproduce"]["command"] or "(command not recorded)"
    add(f"```\n{command}\n```")
    add("")
    add(f"- Price table: `{report['reproduce']['price_file']}`")
    add(f"- Cache directory: `{report['reproduce']['cache_dir']}`")
    add(f"- This run was {'live' if report['reproduce']['live'] else 'entirely from cache'}")
    add(f"- Pre-flight estimate was {_usd(report['run']['preflight_estimate_usd'])} against a "
        f"${report['run']['max_cost_usd']:.2f} cap (the estimate uses a chars/4 heuristic and "
        "assumes maximal output)")
    add("")

    # 9. decision log entry
    add("## Decision log entry (copy into DECISIONS.md and fill in)")
    add("")
    add("```markdown")
    add(report["decision_log_entry"])
    add("```")
    add("")
    return "\n".join(lines)


def _render_per_task_agreement(cal: dict[str, Any]) -> str:
    """The S7.4 per-task breakdown (added 2026-08-23), worst first."""
    lines: list[str] = []
    add = lines.append
    rows = cal.get("per_task_agreement") or []
    add("Per-task agreement (judge vs your labels), worst first:")
    add("")
    if not rows:
        add("No labels matched a scored pair in this run, so there is nothing to break out.")
        add("")
        return "\n".join(lines)
    add("| task | labels | agreed | agreement | kappa |")
    add("|---|---|---|---|---|")
    for row in rows:
        if row["low_n"]:
            kappa_text = f"n/a (n<{row['min_n']})"
        else:
            kappa_text = format_kappa(row["kappa"])
        add(f"| `{row['task']}` | {row['n']} | {row['agree']} | "
            f"{_pct(row['agreement'])} | {kappa_text} |")
    add("")
    add("An overall kappa can hide a judge tuned to one task's taste and silently failing "
        "another's; this is where that shows. Kappa is suppressed below "
        f"{rows[0]['min_n']} labels on a task, where it would be noise - those rows carry "
        "plain agreement only, and plain agreement over three labels is not evidence.")
    add("")
    return "\n".join(lines)


def _render_what_changed(changed: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append
    if not changed["comparable"]:
        add(f"**{changed['reason']}**")
        add("")
        add(f"- Tasks added: {changed['tasks_added'] or 'none'}")
        add(f"- Tasks removed: {changed['tasks_removed'] or 'none'}")
        add(f"- Tasks modified: {changed['tasks_modified'] or 'none'}")
        add(f"- Models added: {changed['models_added'] or 'none'}")
        add(f"- Models removed: {changed['models_removed'] or 'none'}")
        add(f"- Price table: {changed['price_table']['from']} -> "
            f"{changed['price_table']['to']}")
        add(f"- Labels: {changed['labels']['from']} -> {changed['labels']['to']}")
        add("")
        add("No metric deltas are shown: they would not be comparable.")
        add("")
        return "\n".join(lines)

    add(f"Compared against run `{changed['against_run_id']}` (same suite hash).")
    add("")
    cal = changed["calibration"]
    add(f"- kappa: {cal['kappa']['from']} -> {cal['kappa']['to']} "
        f"(delta {cal['kappa']['delta']})")
    add(f"- agreement: {cal['agreement']['from']} -> {cal['agreement']['to']}")
    add(f"- calibration status: {cal['status']['from']} -> {cal['status']['to']}")
    for candidate, delta in changed["candidates"].items():
        add(f"- `{candidate}` win-rate: {delta['win_rate']['from']} -> "
            f"{delta['win_rate']['to']} (delta {delta['win_rate']['delta']}, "
            f"n {delta['n']['from']} -> {delta['n']['to']}) - flip rate "
            f"{delta['flip_rate']['from']} -> {delta['flip_rate']['to']}")
    for model, delta in changed["models"].items():
        add(f"- `{model}`: schema pass {delta['schema_pass']['from']} -> "
            f"{delta['schema_pass']['to']}, p50 {delta['p50_ms']['from']} -> "
            f"{delta['p50_ms']['to']}, p95 {delta['p95_ms']['from']} -> "
            f"{delta['p95_ms']['to']}, cost if uncached "
            f"{delta['cost_if_uncached_usd']['from']} -> "
            f"{delta['cost_if_uncached_usd']['to']}")
    add("")
    if changed["movers"]:
        add(f"Movers (per-task win-rate moved by more than {changed['mover_threshold']}):")
        add("")
        for mover in changed["movers"]:
            add(f"- `{mover['candidate']}` on `{mover['task']}`: {mover['from']} -> "
                f"{mover['to']} (delta {mover['delta']}, n={mover['n']})")
        add("")
    else:
        add(f"No per-task win-rate moved by more than {changed['mover_threshold']}.")
        add("")
    add(f"Cache note: cache hit rate {_pct(changed['cache_note']['cache_hit_rate'])} - "
        f"{changed['cache_note']['note']}.")
    add("")
    return "\n".join(lines)
