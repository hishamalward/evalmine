"""Self-contained blind HTML evidence reports for v2 agent experiments."""

# ruff: noqa: E501 -- embedded HTML/CSS/JS is kept readable in its native syntax

from __future__ import annotations

import hashlib
import html
import json
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

from .experiment import ExperimentError
from .runner import RunnerError, verify_execution
from .validators import ValidationError, verify_validation
from .workspace import PreparationError, verify_prepared

REPORT_FORMAT = "evalmine-episode-report-v1"
LABEL_FORMAT = "evalmine-human-labels-v1"
CHOICES = ("A", "tie", "B", "unclear")


class ExperimentReportError(ExperimentError):
    """An episode report cannot be built or verified safely."""


@dataclass(frozen=True)
class ExperimentReportResult:
    root: Path
    html: Path
    pair_count: int
    run_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "html": str(self.html),
            "pair_count": self.pair_count,
            "run_count": self.run_count,
            "provider_runners_launched": False,
        }


def episode_ab_run_keys(pair_id: str, first: str, second: str) -> tuple[str, str]:
    """Return a stable blind A/B order for two distinct run keys."""
    if first == second:
        raise ValueError("an episode pair requires two distinct run keys")
    ordered = tuple(sorted((first, second)))
    if hashlib.sha256(pair_id.encode("utf-8")).digest()[0] & 1:
        return ordered[1], ordered[0]
    return ordered


def preferred_run_by_choice(a_run: str, b_run: str) -> dict[str, str]:
    if a_run == b_run:
        raise ValueError("A and B must refer to distinct runs")
    return {"A": a_run, "B": b_run, "tie": "tie", "unclear": "unclear"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentReportError(f"cannot read report input {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise ExperimentReportError(f"report input {path} is not a JSON object")
    return value


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExperimentReportError(f"cannot read report input {path} ({exc})") from exc


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ExperimentReportError(f"refusing to overwrite report file {path}") from exc
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "report-marker.json"
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json_blob(value: dict[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _load_validator_results(validation_dir: Path | None) -> tuple[dict[str, Any] | None, list]:
    if validation_dir is None or not validation_dir.is_dir():
        return None, []
    summary = _read_json(validation_dir / "run.json")
    results: list[dict[str, Any]] = []
    for validator_id in summary.get("validator_order", []):
        result = _read_json(validation_dir / f"{validator_id}.json")
        if result.get("type") == "repository-diff" and result.get("patch"):
            result["patch_text"] = _read_text(validation_dir / result["patch"])
        if result.get("type") == "command":
            result["stdout_text"] = _read_text(validation_dir / result["stdout"])
            result["stderr_text"] = _read_text(validation_dir / result["stderr"])
        results.append(result)
    return summary, results


def _run_view(root: Path, planned: dict[str, Any], validation_exists: bool) -> dict[str, Any]:
    run_key = planned["run_key"]
    prepared_dir = root / "runs" / run_key
    execution_dir = root / "execution" / "runs" / run_key
    run = _read_json(prepared_dir / "run.json")
    treatment = _read_json(prepared_dir / "treatment.json")
    execution = _read_json(execution_dir / "run.json")
    turns: list[dict[str, Any]] = []
    for index in range(1, int(run["turns"]) + 1):
        summary_path = execution_dir / f"turn-{index:03d}.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        final_path = execution_dir / f"turn-{index:03d}.final.txt"
        summary["final"] = _read_text(final_path) if final_path.is_file() else ""
        turns.append(summary)
    validation_dir = root / "validation" / "runs" / run_key if validation_exists else None
    validation, validators = _load_validator_results(validation_dir)
    final = turns[-1]["final"] if turns else ""
    return {
        "run_key": run_key,
        "sequence": run["sequence"],
        "block": run["block"],
        "arm": run["arm"],
        "episode": run["episode"],
        "repeat": run["repeat"],
        "runner": run["runner"],
        "requested_model": execution.get("requested_model", run["model"]),
        "observed_model": execution.get("observed_model"),
        "observed_model_source": execution.get("observed_model_source"),
        "model_identity": execution.get("model_identity", {}),
        "auth": run["auth"],
        "treatment": treatment,
        "execution_status": execution.get("status"),
        "execution_error": execution.get("error"),
        "duration_ms": execution.get("duration_ms"),
        "turns_completed": execution.get("turns_completed", 0),
        "turns_planned": execution.get("turns_planned", run["turns"]),
        "tool_count": sum(len(turn.get("tools", [])) for turn in turns),
        "usage": execution.get("usage", {}),
        "billing": execution.get(
            "billing",
            {"basis": "unknown", "reported_cost_usd": None, "dollar_cost_status": "unavailable"},
        ),
        "turns": turns,
        "final": final,
        "validation": validation,
        "validators": validators,
        "validation_verdict": validation.get("verdict") if validation else "not-run",
    }


def _pair_views(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_block.setdefault(run["block"], []).append(run)
    pairs: list[dict[str, Any]] = []
    for block, block_runs in by_block.items():
        for first, second in combinations(sorted(block_runs, key=lambda item: item["run_key"]), 2):
            identity = f"{block}|{first['run_key']}|{second['run_key']}"
            pair_id = "pair-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            a_key, b_key = episode_ab_run_keys(pair_id, first["run_key"], second["run_key"])
            by_key = {first["run_key"]: first, second["run_key"]: second}
            pairs.append(
                {
                    "pair_id": pair_id,
                    "block": block,
                    "episode": first["episode"],
                    "repeat": first["repeat"],
                    "a_run_key": a_key,
                    "b_run_key": b_key,
                    "a": by_key[a_key],
                    "b": by_key[b_key],
                    "preferred_run_by_choice": preferred_run_by_choice(a_key, b_key),
                }
            )
    return pairs


def _arm_summaries(plan: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms = {arm["id"]: arm for arm in plan["arms"]}
    summaries: list[dict[str, Any]] = []
    for arm_id, arm in arms.items():
        rows = [run for run in runs if run["arm"] == arm_id]
        durations = sorted(
            int(run["duration_ms"]) for run in rows if isinstance(run.get("duration_ms"), int)
        )
        median_duration = median(durations) if durations else None
        summaries.append(
            {
                "arm": arm_id,
                "runner": arm["runner"],
                "model": arm["model"],
                "auth": arm["auth"],
                "configuration": arm["configuration"],
                "runs": len(rows),
                "execution_succeeded": sum(run["execution_status"] == "succeeded" for run in rows),
                "validation_passed": sum(run["validation_verdict"] == "passed" for run in rows),
                "median_duration_ms": median_duration,
                "billing_basis": sorted(
                    {str(run.get("billing", {}).get("basis", "unknown")) for run in rows}
                ),
                "reported_cost_usd": sum(
                    float(run.get("billing", {}).get("reported_cost_usd") or 0)
                    for run in rows
                ),
            }
        )
    return summaries


def build_experiment_report_data(
    root: str | Path, *, generated_at: str | None = None
) -> dict[str, Any]:
    """Build the serializable report view after verifying its source envelopes."""
    try:
        prepared_verification = verify_prepared(root)
        execution_verification = verify_execution(root)
    except (PreparationError, RunnerError) as exc:
        raise ExperimentReportError(str(exc)) from exc
    resolved = Path(root).resolve()
    validation_exists = (resolved / "validation").is_dir()
    validation_verification = None
    if validation_exists:
        try:
            validation_verification = verify_validation(resolved)
        except ValidationError as exc:
            raise ExperimentReportError(str(exc)) from exc
    plan = _read_json(resolved / "plan.json")
    runs = [_run_view(resolved, planned, validation_exists) for planned in plan["runs"]]
    pairs = _pair_views(runs)
    return {
        "format": REPORT_FORMAT,
        "label_format": LABEL_FORMAT,
        "generated_at": generated_at or _now(),
        "prepared_root": str(resolved),
        "plan_id": plan["plan_id"],
        "experiment": plan["experiment"],
        "question": plan["question"],
        "objectives": plan["evaluation"]["objectives"],
        "blind": plan["evaluation"]["blind"],
        "human": plan["evaluation"]["human"],
        "judge": plan["evaluation"]["judge"],
        "verification": {
            "preparation": prepared_verification,
            "execution": execution_verification,
            "validation": validation_verification,
        },
        "runs": runs,
        "arms": _arm_summaries(plan, runs),
        "pairs": pairs,
        "pair_count": len(pairs),
        "run_count": len(runs),
    }


def _status(value: str) -> str:
    css = "pass" if value in {"succeeded", "passed", "completed"} else "fail"
    if value == "not-run":
        css = "muted"
    return f'<span class="status {css}">{_esc(value)}</span>'


def _validator_html(result: dict[str, Any]) -> str:
    kind = result.get("type")
    body = ""
    if kind == "repository-diff":
        changes = result.get("changes", [])
        filtered = result.get("filtered_changes", [])
        names = "".join(
            f"<li><code>{_esc(item['path'])}</code> · {_esc(item['change'])}</li>"
            for item in changes
        )
        filtered_names = "".join(
            f"<li><code>{_esc(item['path'])}</code> · {_esc(item['change'])}</li>"
            for item in filtered
        )
        body = (
            f"<p>{len(changes)} changed file(s); patch omissions "
            f"{len(result.get('patch_omissions', []))}.</p><ul>{names}</ul>"
            f"<p>{len(filtered)} change(s) filtered by the declared validator scope.</p>"
            f"<ul>{filtered_names}</ul>"
            f"<pre>{_esc(result.get('patch_text', ''))}</pre>"
        )
    elif kind in {"required-files", "required-sections"}:
        body = (
            "<ul>"
            + "".join(
                f"<li>{_esc(item.get('path') or item.get('section'))}: "
                f"{'present' if item.get('present') else 'missing'}</li>"
                for item in result.get("checks", [])
            )
            + "</ul>"
        )
    elif kind == "command":
        body = (
            f"<p><code>{_esc(' '.join(result.get('argv', [])))}</code> · exit "
            f"{_esc(result.get('exit_code'))} · {_esc(result.get('duration_ms'))} ms</p>"
            f"<pre>{_esc(result.get('stdout_text', ''))}</pre>"
            f'<pre class="stderr">{_esc(result.get("stderr_text", ""))}</pre>'
        )
    return (
        f"<details><summary>{_status(str(result.get('status', 'unknown')))} "
        f"{_esc(result.get('id'))} <small>{_esc(kind)}</small></summary>{body}</details>"
    )


def _outcome_html(label: str, run: dict[str, Any]) -> str:
    turns = "".join(
        f"<details><summary>Turn {_esc(turn['turn'])} · {_status(turn['status'])} · "
        f"{_esc(turn.get('duration_ms'))} ms · {len(turn.get('tools', []))} tools</summary>"
        f"<pre>{_esc(turn.get('final', ''))}</pre></details>"
        for turn in run["turns"]
    )
    validators = "".join(_validator_html(result) for result in run["validators"])
    error = (
        f'<div class="error">{_esc(run["execution_error"])}</div>'
        if run.get("execution_error")
        else ""
    )
    identity = (
        f'<div class="identity"><b>{_esc(run["arm"])}</b><br>'
        f"<code>{_esc(run['runner'])}</code> · requested "
        f"<code>{_esc(run['requested_model'])}</code> · observed "
        f"<code>{_esc(run.get('observed_model') or 'unavailable')}</code>"
        f" ({_esc(run.get('model_identity', {}).get('confidence') or 'unavailable')})"
        f"<br>instructions={_esc(run['treatment'].get('instructions'))} · "
        f"plugins={_esc(run['treatment'].get('plugins'))}</div>"
    )
    return f"""
      <article class="outcome">
        <header><div><span class="outcome-letter">Outcome {_esc(label)}</span>{identity}</div>
          <div>{_status(run["execution_status"])} {_status(run["validation_verdict"])}</div></header>
        {error}
        <div class="metrics"><span>{_esc(run["duration_ms"])} ms</span><span>{_esc(run["tool_count"])} tools</span><span>{_esc(run["turns_completed"])}/{_esc(run["turns_planned"])} turns</span><span>{_esc(run["billing"].get("basis"))}: {_esc(run["billing"].get("dollar_cost_status"))}</span></div>
        <h4>Final response</h4><pre class="final">{_esc(run["final"])}</pre>
        <details><summary>Trajectory</summary>{turns or "<p>No completed turn evidence.</p>"}</details>
        <details><summary>Objective checks</summary>{validators or "<p>Not run.</p>"}</details>
      </article>"""


def _pair_html(pair: dict[str, Any], index: int) -> str:
    return f"""
    <section class="pair" data-pair="{_esc(pair["pair_id"])}">
      <div class="pair-head"><div><small>Pair {index} · {_esc(pair["episode"])} · repeat {_esc(pair["repeat"])}</small><h3>Which outcome would you keep?</h3></div><span class="saved" aria-live="polite"></span></div>
      <div class="outcomes">{_outcome_html("A", pair["a"])}{_outcome_html("B", pair["b"])}</div>
      <div class="label-controls" role="group" aria-label="Label this pair">
        <button data-choice="A">Prefer A</button><button data-choice="tie">Tie</button><button data-choice="B">Prefer B</button><button data-choice="unclear">Unclear</button>
        <label>Why? <textarea class="note" rows="2" placeholder="Optional decision note"></textarea></label>
      </div>
    </section>"""


def render_experiment_report_html(report: dict[str, Any]) -> str:
    """Render one self-contained responsive report and blind labeling queue."""
    arms = "".join(
        f'<div class="arm-card"><div class="identity"><b>{_esc(arm["arm"])}</b><br>'
        f"<code>{_esc(arm['runner'])}</code> · <code>{_esc(arm['model'])}</code></div>"
        f"<b>{arm['execution_succeeded']}/{arm['runs']}</b> executions succeeded<br>"
        f"<b>{arm['validation_passed']}/{arm['runs']}</b> validations passed<br>"
        f"<span>{_esc(arm['median_duration_ms'])} ms median</span></div>"
        for arm in report["arms"]
    )
    pairs = "".join(_pair_html(pair, index) for index, pair in enumerate(report["pairs"], 1))
    validation = report["verification"]["validation"]
    validation_text = validation["verdict"] if validation else "not-run"
    data = {
        "format": report["format"],
        "label_format": report["label_format"],
        "plan_id": report["plan_id"],
        "experiment": report["experiment"],
        "pairs": [
            {
                "pair_id": pair["pair_id"],
                "block": pair["block"],
                "episode": pair["episode"],
                "repeat": pair["repeat"],
                "a_run_key": pair["a_run_key"],
                "b_run_key": pair["b_run_key"],
                "preferred_run_by_choice": pair["preferred_run_by_choice"],
            }
            for pair in report["pairs"]
        ],
    }
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(report["experiment"])} · evalmine episode evidence</title><style>{_CSS}</style></head>
<body data-reveal="0"><header class="hero"><div><div class="eyebrow">evalmine · episode evidence</div><h1>{_esc(report["question"])}</h1><p>{_esc(report["experiment"])} · plan <code>{_esc(report["plan_id"])}</code></p></div>
<button id="reveal" aria-pressed="false">Reveal identities</button></header>
<main><section class="summary"><div><small>Runs</small><b>{report["run_count"]}</b></div><div><small>Blind pairs</small><b>{report["pair_count"]}</b></div><div><small>Execution</small><b>{_esc(report["verification"]["execution"]["status"])}</b></div><div><small>Validation</small><b>{_esc(validation_text)}</b></div></section>
<section><div class="section-head"><div><div class="eyebrow">Experiment</div><h2>What is being decided</h2></div><div id="progress">0 labeled of {report["pair_count"]} pairs</div></div><p class="question">{_esc(report["question"])}</p><ul>{"".join(f"<li>{_esc(item)}</li>" for item in report["objectives"])}</ul><div class="arms">{arms}</div></section>
<section><div class="section-head"><div><div class="eyebrow">Blind review queue</div><h2>Compare trajectory evidence, then label</h2></div><div class="actions"><button id="export">Export labels JSON</button><label class="import">Import labels<input id="import" type="file" accept="application/json"></label></div></div>
<div class="notice">Identities are hidden by default. Execution status, objective checks, trajectory, diffs, and final responses remain visible because they are the evidence being judged.</div>{pairs or "<p>No comparable run pairs.</p>"}</section>
</main><footer>Generated {_esc(report["generated_at"])} · self-contained · no external assets</footer>
<script type="application/json" id="evalmine-episode-data">{_json_blob(data)}</script><script>{_JS}</script></body></html>"""


def generate_experiment_report(
    root: str | Path, *, generated_at: str | None = None
) -> ExperimentReportResult:
    """Create one report envelope without launching a provider or modifying workspaces."""
    report = build_experiment_report_data(root, generated_at=generated_at)
    prepared_root = Path(root).resolve()
    report_root = prepared_root / "report"
    if report_root.exists() or report_root.is_symlink():
        raise ExperimentReportError(
            f"report evidence already exists at {report_root}; it is never overwritten"
        )
    html_text = render_experiment_report_html(report)
    report_root.mkdir()
    _write_once(report_root / "data.json", _json_bytes(report))
    _write_once(report_root / "index.html", html_text.encode("utf-8"))
    marker = {
        "format": REPORT_FORMAT,
        "prepared_root": str(prepared_root),
        "plan_id": report["plan_id"],
        "generated_at": report["generated_at"],
        "run_count": report["run_count"],
        "pair_count": report["pair_count"],
        "provider_runners_launched": False,
        "evidence_sha256": _report_hashes(report_root),
    }
    _write_once(report_root / "report-marker.json", _json_bytes(marker))
    return ExperimentReportResult(
        report_root,
        report_root / "index.html",
        report["pair_count"],
        report["run_count"],
    )


def verify_experiment_report(root: str | Path) -> dict[str, Any]:
    prepared_root = Path(root).resolve()
    report_root = prepared_root / "report"
    marker = _read_json(report_root / "report-marker.json")
    if marker.get("format") != REPORT_FORMAT:
        raise ExperimentReportError(f"{report_root} has an unknown report format")
    if marker.get("prepared_root") != str(prepared_root):
        raise ExperimentReportError("report marker points at a different prepared experiment")
    expected = marker.get("evidence_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ExperimentReportError("report marker has no evidence hashes")
    actual = _report_hashes(report_root)
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected))
        if not changed:
            changed = sorted(path for path in actual if actual[path] != expected.get(path))
        names = ", ".join(changed[:5]) or "report files"
        raise ExperimentReportError(f"report evidence changed after creation: {names}")
    return {
        "ok": True,
        "format": REPORT_FORMAT,
        "root": str(report_root),
        "html": str(report_root / "index.html"),
        "run_count": marker.get("run_count"),
        "pair_count": marker.get("pair_count"),
    }


_CSS = r"""
:root{--bg:#f4f1e8;--paper:#fffdf7;--ink:#18211e;--muted:#66716d;--line:#d8d6cc;--green:#2e705b;--green-soft:#e4f0e9;--red:#a04635;--red-soft:#f7e8e3;--orange:#a96532;--shadow:0 14px 40px #24362e18;color-scheme:light dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}button,.import{font:inherit}code,pre{font-family:"SFMono-Regular",Consolas,monospace}.hero{background:#15211d;color:#f8faf8;padding:32px max(24px,calc((100vw - 1320px)/2));display:flex;justify-content:space-between;gap:24px;align-items:end}.hero h1{font-size:clamp(26px,4vw,48px);line-height:1.05;max-width:900px;margin:8px 0 10px}.hero p{color:#a9b9b3}.eyebrow{text-transform:uppercase;letter-spacing:.16em;font:700 11px monospace;color:#d59763}button,.import{border:1px solid #73817c;background:#fff;color:#18211e;border-radius:9px;padding:9px 12px;cursor:pointer}.hero button{background:#263832;color:white;border-color:#4c665d}main{max-width:1320px;margin:auto;padding:28px 24px 100px}section{margin:26px 0}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:-52px;position:relative}.summary>div{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:16px;box-shadow:var(--shadow)}.summary small{display:block;color:var(--muted)}.summary b{font-size:24px}.section-head{display:flex;justify-content:space-between;gap:16px;align-items:end}.section-head h2{margin:4px 0}.question{font-size:18px}.arms{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}.arm-card{background:var(--paper);border:1px solid var(--line);padding:14px;border-radius:12px}.identity{display:none;color:var(--orange);margin:4px 0 8px}body[data-reveal="1"] .identity{display:block}.actions{display:flex;gap:8px;align-items:center}.import input{display:none}.notice{padding:12px 14px;border-left:4px solid var(--orange);background:#fff4e8;margin:14px 0}.pair{background:var(--paper);border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);margin:18px 0}.pair-head{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.pair-head h3{margin:3px 0}.saved{color:var(--green)}.outcomes{display:grid;grid-template-columns:1fr 1fr}.outcome{padding:18px;min-width:0;border-right:1px solid var(--line)}.outcome:last-child{border-right:0}.outcome header{display:flex;justify-content:space-between;gap:8px}.outcome-letter{font-size:19px;font-weight:800}.status{display:inline-block;border-radius:99px;padding:3px 7px;font:700 10px monospace;text-transform:uppercase}.status.pass{color:var(--green);background:var(--green-soft)}.status.fail{color:var(--red);background:var(--red-soft)}.status.muted{color:var(--muted);background:#eee}.metrics{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}.metrics span{background:#f0eee7;padding:4px 7px;border-radius:6px;font:11px monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#18211e;color:#dce7e2;border-radius:9px;padding:13px;max-height:420px;overflow:auto}.final{min-height:150px}.stderr{color:#ffb7a8}.error{background:var(--red-soft);color:var(--red);padding:9px;margin:9px 0;border-radius:7px}details{border-top:1px solid var(--line);padding:9px 0}summary{cursor:pointer;font-weight:700}details small{color:var(--muted)}.label-controls{border-top:1px solid var(--line);padding:14px 18px;background:#f1eee5;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.label-controls button.selected{background:var(--green);color:white;border-color:var(--green)}.label-controls label{margin-left:auto;display:flex;align-items:center;gap:8px}.note{width:min(420px,40vw);border:1px solid var(--line);border-radius:8px;padding:8px;background:white;color:#18211e}footer{text-align:center;color:var(--muted);padding:30px}.muted{color:var(--muted)}
@media(prefers-color-scheme:dark){:root{--bg:#0f1513;--paper:#17201d;--ink:#e4ebe8;--muted:#9aaba5;--line:#34423d;--green:#7ac6a8;--green-soft:#19382e;--red:#ee9b85;--red-soft:#3a211d;--orange:#e3a36f;--shadow:none}.notice{background:#2b241b}.metrics span,.label-controls{background:#202b27}.note,button,.import{background:#202b27;color:#e4ebe8}.status.muted{background:#2b3531}}
@media(max-width:760px){.hero,.section-head{display:block}.hero button{margin-top:12px}.summary{grid-template-columns:1fr 1fr;margin-top:-30px}.outcomes{grid-template-columns:1fr}.outcome{border-right:0;border-bottom:1px solid var(--line)}.label-controls label{margin-left:0;width:100%;display:block}.note{width:100%;margin-top:5px}.actions{margin-top:10px;flex-wrap:wrap}}
"""


_JS = r"""
(()=>{'use strict';
const data=JSON.parse(document.getElementById('evalmine-episode-data').textContent);
const key='evalmine:episode-labels:'+data.plan_id;let labels={};
try{labels=JSON.parse(localStorage.getItem(key)||'{}')||{}}catch(e){labels={}}
const cards=[...document.querySelectorAll('[data-pair]')];
function save(){try{localStorage.setItem(key,JSON.stringify(labels))}catch(e){}render()}
function render(){let done=0;for(const card of cards){const id=card.dataset.pair,row=labels[id]||{};if(row.choice)done++;for(const button of card.querySelectorAll('[data-choice]'))button.classList.toggle('selected',button.dataset.choice===row.choice);const note=card.querySelector('.note');if(document.activeElement!==note)note.value=row.note||'';card.querySelector('.saved').textContent=row.choice?'saved locally':''}document.getElementById('progress').textContent=done+' labeled of '+cards.length+' pairs'}
for(const card of cards){const id=card.dataset.pair;for(const button of card.querySelectorAll('[data-choice]'))button.addEventListener('click',()=>{labels[id]={...(labels[id]||{}),choice:button.dataset.choice,labelled_at:new Date().toISOString()};save()});card.querySelector('.note').addEventListener('input',e=>{labels[id]={...(labels[id]||{}),note:e.target.value};save()})}
const reveal=document.getElementById('reveal');reveal.addEventListener('click',()=>{const shown=document.body.dataset.reveal==='1';document.body.dataset.reveal=shown?'0':'1';reveal.setAttribute('aria-pressed',shown?'false':'true');reveal.textContent=shown?'Reveal identities':'Hide identities'});
document.getElementById('export').addEventListener('click',()=>{const rows=[];for(const pair of data.pairs){const row=labels[pair.pair_id];if(!row||!row.choice)continue;rows.push({pair_id:pair.pair_id,block:pair.block,episode:pair.episode,repeat:pair.repeat,a_run_key:pair.a_run_key,b_run_key:pair.b_run_key,choice:row.choice,preferred_run_key:pair.preferred_run_by_choice[row.choice],note:row.note||null,labelled_at:row.labelled_at||null,identities_revealed:document.body.dataset.reveal==='1'})}const payload={format:data.label_format,plan_id:data.plan_id,experiment:data.experiment,exported_at:new Date().toISOString(),labels:rows};const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=data.experiment+'-'+data.plan_id+'-labels.json';a.click();URL.revokeObjectURL(url)});
document.getElementById('import').addEventListener('change',event=>{const file=event.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{try{const payload=JSON.parse(reader.result);if(payload.format!==data.label_format||payload.plan_id!==data.plan_id)throw new Error('Labels belong to a different plan.');const known=new Set(data.pairs.map(pair=>pair.pair_id));for(const row of payload.labels||[]){if(known.has(row.pair_id)&&['A','B','tie','unclear'].includes(row.choice))labels[row.pair_id]={choice:row.choice,note:row.note||'',labelled_at:row.labelled_at||null}}save()}catch(error){alert('Could not import labels: '+error.message)}};reader.readAsText(file);event.target.value=''});render();
})();
"""
