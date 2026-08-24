"""report.html - the same report, plus the two things markdown cannot do.

Spec: docs/spec.md S9.5 (added 2026-08-23). One self-contained file: inline CSS,
inline JS, no external asset, no framework, no network. It carries every section
of ``report.md`` in the same order and then adds:

* **side-by-side answer pairs**, with the model names hidden by default, so the
  human reads the answers exactly as the judge read them - blind. Reading "this
  one is from the new model" first is how a calibration set ends up recording
  your expectations rather than your judgement.
* **the labelling flow**: three buttons per pair and a "copy labels YAML" that
  emits the suite's own ``labels[]`` entries, ready to paste.

The A/B side each answer lands on is randomised per pair - seeded from the pair
id, so it is stable across reloads - and the mapping back from "I preferred A"
to ``prefer: baseline`` or ``prefer: candidate`` is computed **here, in Python**
and baked into the page as a lookup table. The JS never derives it. A flipped
label poisons calibration silently, so the mapping is a tested pure function
rather than a line of browser code nobody can run in CI.
"""

from __future__ import annotations

import hashlib
import html
import json
from typing import TYPE_CHECKING, Any

from .metrics import FLIP_RATE_WARNING, format_kappa

if TYPE_CHECKING:  # pragma: no cover
    from .core import RunData

#: What a human can click, and what each click means for the suite file.
CHOICES = ("A", "tie", "B")


# --------------------------------------------------------------------------
# the blind A/B mapping - the part that must never be wrong
# --------------------------------------------------------------------------


def pair_id(task: str, case: str, baseline: str, candidate: str) -> str:
    """A stable id for one judged pair. Also the seed for its A/B order."""
    return f"{task}|{case}|{baseline}|{candidate}"


def ab_roles(pid: str) -> tuple[str, str]:
    """Which role is shown as Answer A, and which as Answer B.

    Deterministic in the pair id: the same pair lands the same way on every
    reload and every regeneration, so a half-finished labelling session picks up
    where it left off instead of silently swapping sides underneath you.
    """
    digest = hashlib.sha256(pid.encode("utf-8")).digest()
    if digest[0] & 1:
        return ("candidate", "baseline")
    return ("baseline", "candidate")


def prefer_by_choice(roles: tuple[str, str]) -> dict[str, str]:
    """The click-to-``prefer`` table for one pair, given its A/B roles.

    ``{"A": "baseline"|"candidate", "B": ..., "tie": "tie"}`` - exactly the
    ``prefer`` vocabulary of S5.5. This table is what the page ships; the
    browser only looks a value up in it.
    """
    a_role, b_role = roles
    if {a_role, b_role} != {"baseline", "candidate"}:
        raise ValueError(f"A/B roles must be baseline and candidate, got {roles!r}")
    return {"A": a_role, "B": b_role, "tie": "tie"}


# --------------------------------------------------------------------------
# the pair view
# --------------------------------------------------------------------------


def build_pair_view(data: RunData) -> list[dict[str, Any]]:
    """One record per judged pair: the prompt, both answers, the verdict.

    Only ``repeat == 0`` pairs are shown, because that is the set a label can
    address: a ``labels[]`` entry has no repeat index.
    """
    answers = {(a.task_id, a.case_id, a.model, a.repeat): a for a in data.answers}
    labels = {
        (label.task, label.case, label.baseline, label.candidate): label
        for label in data.suite.labels
    }

    view: list[dict[str, Any]] = []
    for pair in data.pairs:
        if pair.repeat != 0:
            continue
        base = answers.get((pair.task_id, pair.case_id, pair.baseline, 0))
        cand = answers.get((pair.task_id, pair.case_id, pair.candidate, 0))
        pid = pair_id(pair.task_id, pair.case_id, pair.baseline, pair.candidate)
        roles = ab_roles(pid)
        texts = {
            "baseline": (base.text if base else ""),
            "candidate": (cand.text if cand else ""),
        }
        errors = {
            "baseline": (base.error if base else "no answer was recorded"),
            "candidate": (cand.error if cand else "no answer was recorded"),
        }
        schema = {
            "baseline": (base.schema_status if base else "n/a"),
            "candidate": (cand.schema_status if cand else "n/a"),
        }
        checks = {
            "baseline": (base.check_view if base else None),
            "candidate": (cand.check_view if cand else None),
        }
        finishes = {
            "baseline": (base.finish_reason if base else None),
            "candidate": (cand.finish_reason if cand else None),
        }
        label = labels.get((pair.task_id, pair.case_id, pair.baseline, pair.candidate))
        task = data.suite.task(pair.task_id)
        prompt = base.prompt if base else (cand.prompt if cand else "")

        view.append(
            {
                "pair_id": pid,
                "task": pair.task_id,
                "case": pair.case_id,
                "kind": task.kind if task else None,
                "baseline": pair.baseline,
                "candidate": pair.candidate,
                "a_role": roles[0],
                "b_role": roles[1],
                "prefer_by_choice": prefer_by_choice(roles),
                "prompt": prompt,
                "a_text": texts[roles[0]],
                "b_text": texts[roles[1]],
                "a_schema_status": schema[roles[0]],
                "b_schema_status": schema[roles[1]],
                "a_check": checks[roles[0]],
                "b_check": checks[roles[1]],
                "a_finish": finishes[roles[0]],
                "b_finish": finishes[roles[1]],
                "a_error": errors[roles[0]] if not texts[roles[0]] else None,
                "b_error": errors[roles[1]] if not texts[roles[1]] else None,
                "excluded": pair.excluded,
                "reason": pair.reason,
                "score": pair.score,
                "category": pair.category,
                "passes": [
                    {"order": p.order, "verdict": p.verdict, "reason": p.reason}
                    for p in pair.passes
                ],
                "existing_prefer": label.prefer if label else None,
                "existing_note": (label.note if label else None),
            }
        )
    return view


def _page_data(report: dict[str, Any], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """The only thing the JS reads: ids and the click-to-prefer table."""
    return {
        "run_id": report["run_id"],
        "suite": report["suite"]["name"],
        "pairs": [
            {
                "pair_id": p["pair_id"],
                "task": p["task"],
                "case": p["case"],
                "baseline": p["baseline"],
                "candidate": p["candidate"],
                "prefer_by_choice": p["prefer_by_choice"],
                "existing_prefer": p["existing_prefer"],
                "existing_note": p["existing_note"],
            }
            for p in pairs
        ],
    }


# --------------------------------------------------------------------------
# escaping
# --------------------------------------------------------------------------


def esc(value: Any) -> str:
    """Everything on this page goes through here. Answers are untrusted text."""
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _json_blob(payload: dict[str, Any]) -> str:
    """JSON for a ``<script type="application/json">`` block.

    ``<``, ``>`` and ``&`` are escaped so that no string in a model's answer -
    or a task id - can close the tag early.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _sentence(text: str | None) -> str:
    """Escaped, and closed with a full stop so it does not run into the next one."""
    if not text:
        return ""
    escaped = esc(text.strip())
    return escaped if escaped.endswith((".", "!", "?")) else escaped + "."


def _pct(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 0) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _table(headers: list[str], rows: list[list[str]], klass: str = "") -> str:
    """A table in an overflow-x container: wide tables scroll, the page does not."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    css = f" class=\"{klass}\"" if klass else ""
    return (
        f'<div class="scroll"><table{css}><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


def render_html(report: dict[str, Any], pairs: list[dict[str, Any]] | None = None) -> str:
    pairs = pairs or []
    parts: list[str] = []
    add = parts.append

    suite = report["suite"]
    title = f"{suite['name']} - run {report['run_id']}"

    add("<!doctype html>")
    add('<html lang="en"><head><meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, initial-scale=1">')
    add(f"<title>{esc(title)}</title>")
    add(f"<style>{_CSS}</style>")
    add("</head><body>")
    add('<main id="top">')

    add(_header(report))
    add(_calibration(report))
    add(_win_rates(report))
    add(_scorecard(report))
    add(_per_task(report))
    add(_what_changed(report))
    add(_failures(report))
    add(_reproduce(report))
    add(_decision_log(report))
    add(_pairs_section(report, pairs))

    add("</main>")
    add(_footer_bar(pairs))
    add(
        '<script type="application/json" id="evalmine-data">'
        + _json_blob(_page_data(report, pairs))
        + "</script>"
    )
    add(f"<script>{_JS}</script>")
    add("</body></html>")
    return "\n".join(parts) + "\n"


def _header(report: dict[str, Any]) -> str:
    suite = report["suite"]
    totals = report["totals"]
    rows: list[str] = []
    models = ", ".join(
        f"<code>{esc(m)}</code>" + (" (baseline)" if m == report["baseline"] else "")
        for m in report["models"]
    )
    judge_note = " &mdash; <b>also under test</b>" if report["judge"]["under_test"] else ""
    price_note = (
        "" if report["prices"]["verified"] else " (UNVERIFIED &mdash; figures are placeholders)"
    )
    rows.append(
        f"<li>Generated <code>{esc(report['generated_at'])}</code> by evalmine "
        f"{esc(report['tool_version'])}</li>"
    )
    rows.append(
        f"<li>Suite <code>{esc(suite['path'])}</code> (hash <code>{esc(suite['hash'][:12])}</code>,"
        f" {suite['n_tasks']} tasks, {suite['n_cases']} cases, {suite['n_labels']} labels)</li>"
    )
    rows.append(f"<li>Models: {models}</li>")
    rows.append(f"<li>Judge: <code>{esc(report['judge']['model'])}</code>{judge_note}</li>")
    rows.append(
        f"<li>Price table <code>{esc(report['prices']['file'])}</code> pinned "
        f"{esc(report['prices']['pinned'])}{price_note}</li>"
    )
    rows.append(
        f"<li>{totals['answers']} answers, {report['judge']['calls']} judge passes &mdash; "
        f"cache hit rate {_pct(totals['cache_hit_rate'])}, {totals['live_calls']} live</li>"
    )
    rows.append(
        f"<li>Cost this run {_usd(totals['cost_usd'])} (answers "
        f"{_usd(totals['cost_answers_usd'])}, judge {_usd(totals['cost_judge_usd'])}) &mdash; "
        f"if uncached {_usd(totals['cost_if_uncached_usd'])}</li>"
    )
    if report["cost_incomplete"]:
        rows.append(
            f"<li><b>Cost is incomplete</b>: {totals['missing_usage_calls']} calls returned no "
            "token counts, so the totals above are lower bounds</li>"
        )
    if report["aborted_over_budget"]:
        rows.append(
            f"<li><b>ABORTED OVER BUDGET</b> at the ${report['run']['max_cost_usd']:.2f} ceiling "
            "&mdash; this report is partial</li>"
        )

    desc = (
        f"<p class=\"lede\">{esc(report['suite']['description'].strip())}</p>"
        if report["suite"].get("description")
        else ""
    )
    return (
        f"<h1>{esc(report['suite']['name'])} <span class=\"runid\">run "
        f"{esc(report['run_id'])}</span></h1>"
        f"{desc}"
        f'<ul class="meta">{"".join(rows)}</ul>'
        '<nav class="jump">Jump to: <a href="#calibration">calibration</a> '
        '<a href="#win-rates">win-rates</a> <a href="#scorecard">scorecard</a> '
        '<a href="#per-task">per-task</a> <a href="#failures">failures</a> '
        '<a href="#pairs">answer pairs &amp; labelling</a></nav>'
    )


def _calibration(report: dict[str, Any]) -> str:
    cal = report["calibration"]
    eligible = report["headline_eligible"]
    out: list[str] = ['<section id="calibration"><h2>Calibration (judge vs your labels)</h2>']
    if not eligible:
        out.append(
            f'<div class="banner"><b>UNCALIBRATED &mdash; not headline-eligible: '
            f"{esc(cal['status'])}.</b> {_sentence(cal.get('reason'))} Win-rates below are "
            "flagged and must not be quoted as an established result.</div>"
        )
    out.append(
        f"<ul class=\"meta\"><li>Status <code>{esc(cal['status'])}</code> &mdash; headline "
        f"eligible: <b>{str(eligible).lower()}</b></li>"
        f"<li>Cohen's kappa: <b>{esc(format_kappa(cal.get('kappa')))}</b> (floor "
        f"{cal['min_kappa']:.2f}, minimum labels {cal['min_labels']})</li>"
        f"<li>Plain agreement {_pct(cal.get('agreement'))} over {cal['n_labels']} labelled "
        "pairs &mdash; agreement is inflated whenever one category dominates, so read it "
        "next to kappa, never instead of it</li>"
        f"<li>Labels in suite {cal['n_labels_in_suite']} &mdash; used {cal['n_labels']} "
        f"&mdash; ignored {cal['n_labels_ignored']}</li></ul>"
    )
    if cal["ignored_labels"]:
        items = "".join(
            f"<li><code>{esc(i['task'])}/{esc(i['case'])}</code> ({esc(i['baseline'])} vs "
            f"{esc(i['candidate'])}): {esc(i['why'])}</li>"
            for i in cal["ignored_labels"][:20]
        )
        more = (
            f"<li>... and {len(cal['ignored_labels']) - 20} more</li>"
            if len(cal["ignored_labels"]) > 20
            else ""
        )
        out.append(f"<h3>Ignored labels</h3><ul class=\"meta\">{items}{more}</ul>")

    out.append("<h3>Confusion matrix</h3><p class=\"note\">Rows: you. Columns: the judge.</p>")
    rows = [
        [
            f"<b>{human}</b>",
            str(cal["confusion"][human]["baseline"]),
            str(cal["confusion"][human]["candidate"]),
            str(cal["confusion"][human]["tie"]),
        ]
        for human in ("baseline", "candidate", "tie")
    ]
    out.append(_table(["you \\ judge", "baseline", "candidate", "tie"], rows))

    out.append("<h3>Per-task agreement</h3>")
    per_task = cal.get("per_task_agreement") or []
    if not per_task:
        out.append(
            '<p class="note">No labels matched a scored pair in this run, so there is '
            "nothing to break out.</p>"
        )
    else:
        rows = []
        for row in per_task:
            kappa_text = (
                f"n/a <span class=\"flag\">n&lt;{row['min_n']}</span>"
                if row["low_n"]
                else esc(format_kappa(row["kappa"]))
            )
            rows.append(
                [
                    f"<code>{esc(row['task'])}</code>",
                    str(row["n"]),
                    str(row["agree"]),
                    _pct(row["agreement"]),
                    kappa_text,
                ]
            )
        out.append(_table(["task", "labels", "agreed", "agreement", "kappa"], rows))
        out.append(
            '<p class="note">Worst first. An overall kappa can hide a judge tuned to one '
            "task's taste and silently failing another's; this is where that shows. Kappa is "
            f"suppressed below {per_task[0]['min_n']} labels on a task, where it would be "
            "noise &mdash; those rows carry plain agreement only, and plain agreement over "
            "three labels is not evidence.</p>"
        )
    out.append("</section>")
    return "".join(out)


def _win_rates(report: dict[str, Any]) -> str:
    eligible = report["headline_eligible"]
    dagger = "" if eligible else "&dagger;"
    title = "Win-rates" if eligible else "Win-rates (UNCALIBRATED &mdash; not a headline)"
    out = [f'<section id="win-rates"><h2>{title}</h2>']
    out.append(
        '<p class="note">Over schema-passing pairs only: a pair where either side failed to '
        "parse or failed its schema is excluded, not scored a loss (ruling O-3). Read these "
        "beside the schema-pass rates in the scorecard below.</p>"
    )
    rows = []
    for candidate, win in report["win_rates"].items():
        ci = win["ci"]
        ci_text = f"{ci[0]:.3f}&ndash;{ci[1]:.3f}" if ci else "n too small"
        rate = "n/a" if win["win_rate"] is None else f"{win['win_rate']:.3f}{dagger}"
        flip_rate = _pct(win["flip_rate"])
        if win["flip_rate_above_warning"]:
            flip_rate = f'<b class="bad">{flip_rate}</b>'
        counts = win["counts"]
        rows.append(
            [
                f"<code>{esc(candidate)}</code>",
                rate,
                ci_text,
                str(win["n"]),
                str(counts["consistent_win"]),
                str(counts["consistent_loss"]),
                str(counts["tie"]),
                f"{counts['soft_win']}/{counts['soft_loss']}",
                str(win["flips"]),
                flip_rate,
                str(win["excluded"]),
            ]
        )
    out.append(
        _table(
            [
                "candidate", "win-rate", "95% CI", "n", "wins", "losses", "ties",
                "soft W/L", "flips", "flip rate", "excluded",
            ],
            rows,
        )
    )
    if any(w["flip_rate_above_warning"] for w in report["win_rates"].values()):
        out.append(
            f'<div class="banner"><b>A flip rate above {FLIP_RATE_WARNING:.2f} means the judge '
            "changed its mind when the answers changed places more often than a measurement "
            "can survive. The bolded win-rate above is not a measurement.</b></div>"
        )
    if not eligible:
        out.append(
            f'<p class="note">{dagger} uncalibrated: '
            f"{esc(report['calibration'].get('reason') or report['calibration']['status'])}</p>"
        )
    out.append("</section>")
    return "".join(out)


def _scorecard(report: dict[str, Any]) -> str:
    rows = []
    for model, row in report["per_model"].items():
        schema = row["schema"]
        latency = row["latency"]
        cost = row["cost"]
        rows.append(
            [
                f"<code>{esc(model)}</code>",
                esc(row["role"]),
                _pct(schema["rate"]),
                esc(", ".join(schema["modes"]) if schema["modes"] else "-"),
                str(schema["parse_fail"]),
                str(schema["schema_fail"]),
                _num(latency["p50_ms"]),
                f"{_num(latency['p95_ms'])} (n={latency['n']})",
                _pct(latency["live_fraction"]),
                _usd(cost["this_run_usd"]),
                _usd(cost["if_uncached_usd"]),
                _usd(cost["per_1k_calls_usd"]),
            ]
        )
    return (
        '<section id="scorecard"><h2>Per-model scorecard</h2>'
        + _table(
            [
                "model", "role", "schema pass", "mode", "parse fail", "schema fail",
                "p50 ms", "p95 ms", "live", "cost this run", "cost if uncached", "$/1k calls",
            ],
            rows,
        )
        + '<p class="note">p95 is nearest-rank; below n=20 it is the maximum, which is why n '
        "is printed with it.</p></section>"
    )


def _per_task(report: dict[str, Any]) -> str:
    dagger = "" if report["headline_eligible"] else "&dagger;"
    headers = ["task", "kind", "cases"]
    headers += [f"schema {esc(m)}" for m in report["models"]]
    headers += [f"win {esc(c)}" for c in report["candidates"]]
    headers += ["p50 ms (baseline)", "cost if uncached"]
    rows = []
    for row in report["per_task"]:
        cells = [f"<code>{esc(row['task'])}</code>", esc(row["kind"] or "-"), str(row["n_cases"])]
        for model in report["models"]:
            cells.append(_pct(row["models"][model]["schema"]["rate"]))
        for candidate in report["candidates"]:
            entry = row["candidates"][candidate]
            value = "n/a" if entry["win_rate"] is None else f"{entry['win_rate']:.2f}{dagger}"
            cells.append(f"{value} (n={entry['n']})")
        cells.append(_num(row["models"][report["baseline"]]["p50_ms"]))
        cells.append(
            _usd(sum(row["models"][m]["cost_if_uncached_usd"] or 0.0 for m in report["models"]))
        )
        rows.append(cells)
    return (
        '<section id="per-task"><h2>Per-task</h2>'
        '<p class="note">Sorted by the first candidate\'s win-rate, ascending: what the new '
        "model is worst at is the first thing on screen.</p>" + _table(headers, rows) + "</section>"
    )


def _what_changed(report: dict[str, Any]) -> str:
    changed = report.get("what_changed")
    out = ['<section id="what-changed"><h2>What changed</h2>']
    if not changed:
        out.append('<p class="note">No previous report for this suite.</p></section>')
        return "".join(out)
    if not changed["comparable"]:
        items = [
            f"Tasks added: {esc(changed['tasks_added'] or 'none')}",
            f"Tasks removed: {esc(changed['tasks_removed'] or 'none')}",
            f"Tasks modified: {esc(changed['tasks_modified'] or 'none')}",
            f"Models added: {esc(changed['models_added'] or 'none')}",
            f"Models removed: {esc(changed['models_removed'] or 'none')}",
            f"Price table: {esc(changed['price_table']['from'])} &rarr; "
            f"{esc(changed['price_table']['to'])}",
            f"Labels: {esc(changed['labels']['from'])} &rarr; {esc(changed['labels']['to'])}",
        ]
        out.append(f"<div class=\"banner\"><b>{esc(changed['reason'])}</b></div>")
        out.append("<ul class=\"meta\">" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
        out.append('<p class="note">No metric deltas are shown: they would not be comparable.'
                   "</p></section>")
        return "".join(out)

    cal = changed["calibration"]
    items = [
        f"Compared against run <code>{esc(changed['against_run_id'])}</code> (same suite hash)",
        f"kappa {esc(cal['kappa']['from'])} &rarr; {esc(cal['kappa']['to'])} "
        f"(delta {esc(cal['kappa']['delta'])})",
        f"agreement {esc(cal['agreement']['from'])} &rarr; {esc(cal['agreement']['to'])}",
        f"calibration status {esc(cal['status']['from'])} &rarr; {esc(cal['status']['to'])}",
    ]
    for candidate, delta in changed["candidates"].items():
        items.append(
            f"<code>{esc(candidate)}</code> win-rate {esc(delta['win_rate']['from'])} &rarr; "
            f"{esc(delta['win_rate']['to'])} (delta {esc(delta['win_rate']['delta'])}, n "
            f"{esc(delta['n']['from'])} &rarr; {esc(delta['n']['to'])}) &mdash; flip rate "
            f"{esc(delta['flip_rate']['from'])} &rarr; {esc(delta['flip_rate']['to'])}"
        )
    for model, delta in changed["models"].items():
        items.append(
            f"<code>{esc(model)}</code> schema pass {esc(delta['schema_pass']['from'])} &rarr; "
            f"{esc(delta['schema_pass']['to'])}, p50 {esc(delta['p50_ms']['from'])} &rarr; "
            f"{esc(delta['p50_ms']['to'])}, p95 {esc(delta['p95_ms']['from'])} &rarr; "
            f"{esc(delta['p95_ms']['to'])}, cost if uncached "
            f"{esc(delta['cost_if_uncached_usd']['from'])} &rarr; "
            f"{esc(delta['cost_if_uncached_usd']['to'])}"
        )
    out.append("<ul class=\"meta\">" + "".join(f"<li>{i}</li>" for i in items) + "</ul>")
    if changed["movers"]:
        rows = [
            [
                f"<code>{esc(m['candidate'])}</code>",
                f"<code>{esc(m['task'])}</code>",
                f"{m['from']}",
                f"{m['to']}",
                f"{m['delta']}",
                str(m["n"]),
            ]
            for m in changed["movers"]
        ]
        out.append(
            f"<h3>Movers (moved by more than {changed['mover_threshold']})</h3>"
            + _table(["candidate", "task", "from", "to", "delta", "n"], rows)
        )
    else:
        out.append(
            f'<p class="note">No per-task win-rate moved by more than '
            f"{changed['mover_threshold']}.</p>"
        )
    out.append(
        f'<p class="note">Cache note: cache hit rate '
        f"{_pct(changed['cache_note']['cache_hit_rate'])} &mdash; "
        f"{esc(changed['cache_note']['note'])}.</p></section>"
    )
    return "".join(out)


def _failures(report: dict[str, Any]) -> str:
    out = ['<section id="failures"><h2>Failures and exclusions</h2>']
    excluded = report["exclusions"]
    if not excluded and not report["errors"] and not report["unparseable_judge_passes"]:
        out.append('<p class="note">None.</p>')
    else:
        if excluded:
            items = "".join(
                f"<li><code>{esc(i['task'])}/{esc(i['case'])}</code> vs "
                f"<code>{esc(i['candidate'])}</code>: {esc(i['reason'])}</li>"
                for i in excluded
            )
            out.append(f"<h3>{len(excluded)} excluded pairs</h3><ul class=\"meta\">{items}</ul>")
        if report["errors"]:
            items = "".join(
                f"<li><code>{esc(i['model'])}</code> on <code>{esc(i['task'])}/"
                f"{esc(i['case'])}</code>: {esc(i['error'])}</li>"
                for i in report["errors"]
            )
            out.append(f"<h3>Provider errors</h3><ul class=\"meta\">{items}</ul>")
        if report["unparseable_judge_passes"]:
            items = "".join(
                f"<li><code>{esc(i['task'])}/{esc(i['case'])}</code> vs "
                f"<code>{esc(i['candidate'])}</code> (order {i['order']})</li>"
                for i in report["unparseable_judge_passes"]
            )
            out.append(f"<h3>Unparseable judge passes</h3><ul class=\"meta\">{items}</ul>")
    if report["warnings"]:
        items = "".join(f"<li>{esc(w)}</li>" for w in report["warnings"])
        out.append(f"<h3>Warnings</h3><ul class=\"meta\">{items}</ul>")
    out.append("</section>")
    return "".join(out)


def _reproduce(report: dict[str, Any]) -> str:
    command = report["reproduce"]["command"] or "(command not recorded)"
    return (
        '<section id="reproduce"><h2>Reproduce</h2>'
        f"<pre class=\"cmd\">{esc(command)}</pre>"
        f"<ul class=\"meta\"><li>Price table <code>{esc(report['reproduce']['price_file'])}</code>"
        f"</li><li>Cache directory <code>{esc(report['reproduce']['cache_dir'])}</code></li>"
        f"<li>This run was {'live' if report['reproduce']['live'] else 'entirely from cache'}</li>"
        f"<li>Pre-flight estimate {_usd(report['run']['preflight_estimate_usd'])} against a "
        f"${report['run']['max_cost_usd']:.2f} cap (the estimate uses a chars/4 heuristic and "
        "assumes maximal output)</li></ul></section>"
    )


def _decision_log(report: dict[str, Any]) -> str:
    return (
        '<section id="decision-log"><h2>Decision log entry</h2>'
        '<p class="note">Copy into DECISIONS.md and fill in. The tool prints it; the tool does '
        "not write it &mdash; this is the one artefact a human must author.</p>"
        f"<pre class=\"cmd\">{esc(report['decision_log_entry'])}</pre></section>"
    )


# --------------------------------------------------------------------------
# the pairs, blind, and the labelling flow
# --------------------------------------------------------------------------


#: finish_reason values worth a flag on the pane, and how they read.
FINISH_LABELS = {
    "max_tokens": "truncated: hit max_tokens",
    "refusal": "refused by the provider's safety layer",
    "length": "truncated: hit max_tokens",
}


def _pane(letter: str, pair: dict[str, Any]) -> str:
    role = pair[f"{letter.lower()}_role"]
    model = pair["baseline"] if role == "baseline" else pair["candidate"]
    text = pair[f"{letter.lower()}_text"]
    error = pair[f"{letter.lower()}_error"]
    schema = pair[f"{letter.lower()}_schema_status"]
    body = (
        f'<pre class="answer">{esc(text)}</pre>'
        if text
        else f'<p class="note">No answer text. {esc(error or "")}</p>'
    )
    schema_flag = (
        f'<span class="flag bad">schema: {esc(schema)}</span>'
        if schema in ("parse_fail", "schema_fail")
        else ""
    )
    finish = pair.get(f"{letter.lower()}_finish")
    #: A refusal or a truncated answer is not the model's best attempt; the
    #: labeller must see that beside the text, not discover it in answers.jsonl.
    finish_flag = (
        f'<span class="flag bad">{esc(FINISH_LABELS[finish])}</span>'
        if finish in FINISH_LABELS
        else ""
    )
    check = pair.get(f"{letter.lower()}_check")
    check_flag = ""
    check_body = ""
    if check is not None:
        status = str(check.get("status", ""))
        exit_code = check.get("exit_code")
        verdict = status.upper() + (f" (exit {exit_code})" if exit_code is not None else "")
        css = "ok" if status == "pass" else "bad"
        check_flag = f'<span class="flag {css}">exec: {esc(verdict)}</span>'
        output = (check.get("output") or "").strip() or "(no output)"
        check_body = (
            '<details class="prompt exec" open><summary>Execution output</summary>'
            f"<pre>{esc(output)}</pre></details>"
        )
    return (
        f'<section class="pane"><h4>Answer {letter}{finish_flag}{schema_flag}{check_flag}'
        f'<span class="model reveal-inline"><code>{esc(model)}</code> '
        f'<span class="role">({esc(role)})</span></span></h4>{body}{check_body}</section>'
    )


def _pairs_section(report: dict[str, Any], pairs: list[dict[str, Any]]) -> str:
    out = ['<section id="pairs"><h2>Answer pairs &amp; labelling</h2>']
    if not pairs:
        out.append('<p class="note">No judged pairs in this run.</p></section>')
        return "".join(out)
    out.append(
        '<p class="note">Model names are hidden. You are reading these exactly as the judge '
        "read them &mdash; blind, and in a per-pair randomised order, so a label records your "
        "judgement rather than your expectations. Label first, then use <b>reveal models</b> "
        "in the bar below to see who wrote what and what the judge said. Your clicks stay in "
        "this browser; <b>copy labels YAML</b> hands you the "
        "<code>labels[]</code> entries to paste into the suite file.</p>"
    )
    for pair in pairs:
        out.append(_pair_card(pair))
    out.append("</section>")
    return "".join(out)


def _pair_card(pair: dict[str, Any]) -> str:
    flags: list[str] = []
    if pair["excluded"]:
        flags.append('<span class="flag bad">excluded from the win-rate</span>')
    if pair["category"] == "flip":
        flags.append('<span class="flag">position flip</span>')
    if pair["existing_prefer"]:
        flags.append('<span class="flag">already labelled</span>')

    verdict_bits: list[str] = []
    if pair["excluded"]:
        verdict_bits.append(f"Excluded: <code>{esc(pair['reason'])}</code>")
    else:
        verdict_bits.append(
            f"Judge: <b>{esc(pair['category'])}</b> (score {pair['score']}) &mdash; a score "
            "above 0.5 favours the candidate"
        )
        for p in pair["passes"]:
            shown = "baseline first" if p["order"] == 1 else "candidate first"
            verdict_bits.append(
                f"pass {p['order']} ({shown}): <b>{esc(p['verdict'])}</b> &mdash; "
                f"{esc(p['reason'] or 'no reason given')}"
            )
    if pair["existing_prefer"]:
        note = f" &mdash; {esc(pair['existing_note'])}" if pair["existing_note"] else ""
        verdict_bits.append(
            f"Your label in the suite file: <b>{esc(pair['existing_prefer'])}</b>{note}"
        )

    verdict = "".join(f"<li>{bit}</li>" for bit in verdict_bits)
    kind = f" <span class=\"flag\">{esc(pair['kind'])}</span>" if pair["kind"] else ""

    return (
        f'<article class="pair" id="pair-{esc(pair["pair_id"])}">'
        f'<header class="pair-head"><span class="pid"><code>{esc(pair["task"])}</code> / '
        f'<code>{esc(pair["case"])}</code></span>{kind}{"".join(flags)}</header>'
        f'<details class="prompt" open><summary>Task prompt</summary>'
        f'<pre>{esc(pair["prompt"])}</pre></details>'
        f'<div class="cols">{_pane("A", pair)}{_pane("B", pair)}</div>'
        f'<div class="verdict reveal-only"><ul class="meta">{verdict}</ul></div>'
        f'<div class="labelrow" data-pair="{esc(pair["pair_id"])}">'
        '<button type="button" class="choice" data-choice="A">Prefer A</button>'
        '<button type="button" class="choice" data-choice="tie">Tie</button>'
        '<button type="button" class="choice" data-choice="B">Prefer B</button>'
        '<input type="text" class="why" maxlength="240" placeholder="why (optional)">'
        '<span class="saved" aria-live="polite"></span>'
        "</div></article>"
    )


def _footer_bar(pairs: list[dict[str, Any]]) -> str:
    return (
        '<footer id="bar">'
        f'<span id="progress">0 labeled of {len(pairs)} pairs</span>'
        '<span class="spacer"></span>'
        '<button type="button" id="reveal" aria-pressed="false">Reveal models</button>'
        '<button type="button" id="copy">Copy labels YAML</button>'
        '<button type="button" id="reset">Clear</button>'
        '<span id="status" aria-live="polite"></span>'
        "</footer>"
        '<div id="yamlbox" hidden><label for="yamlout">labels YAML &mdash; paste into your '
        'suite file</label><textarea id="yamlout" readonly rows="10"></textarea></div>'
    )


# --------------------------------------------------------------------------
# assets - inline, because the page must open from a file:// path forever
# --------------------------------------------------------------------------

_CSS = """
:root{
  --bg:#fbfbfa;--panel:#ffffff;--ink:#16181d;--muted:#5a616e;--line:#e3e5ea;
  --accent:#2f5fd0;--accent-ink:#ffffff;--chip:#eef1f7;
  --warn-bg:#fff5e6;--warn-line:#d99b2b;--warn-ink:#6a4300;
  --bad:#b02a2a;--ok:#1f7a3a;--code:#f2f3f7;--shadow:0 1px 2px rgba(20,22,30,.07);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#14161a;--panel:#1b1e24;--ink:#e6e8ec;--muted:#9aa3b2;--line:#2b3038;
    --accent:#7aa2f7;--accent-ink:#10131a;--chip:#252a33;
    --warn-bg:#3a2c14;--warn-line:#c08a30;--warn-ink:#f0d9a8;
    --bad:#f07171;--ok:#5fc48a;--code:#1f232a;--shadow:0 1px 2px rgba(0,0,0,.4);
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  padding-bottom:5.5rem}
main{max-width:1180px;margin:0 auto;padding:2rem 1.25rem 1rem}
h1{font-size:1.6rem;margin:0 0 .3rem;line-height:1.25}
h2{font-size:1.18rem;margin:2.2rem 0 .6rem;padding-bottom:.3rem;border-bottom:1px solid var(--line)}
h3{font-size:1rem;margin:1.4rem 0 .4rem}
h4{font-size:.9rem;margin:0 0 .5rem;display:flex;gap:.5rem;align-items:baseline;flex-wrap:wrap}
p{margin:.5rem 0}
a{color:var(--accent)}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
code{background:var(--code);padding:.05em .3em;border-radius:3px;font-size:.87em;
  overflow-wrap:anywhere}
.runid{font-weight:400;font-size:.95rem;color:var(--muted)}
.lede{color:var(--muted);max-width:70ch}
.note{color:var(--muted);font-size:.88rem;max-width:82ch}
ul.meta{margin:.5rem 0;padding-left:1.1rem}
ul.meta li{margin:.15rem 0}
nav.jump{margin:1rem 0 0;font-size:.85rem;color:var(--muted)}
nav.jump a{margin-right:.6rem;white-space:nowrap}
.banner{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-ink);
  border-left-width:4px;padding:.7rem .9rem;border-radius:4px;margin:.8rem 0;max-width:82ch}
.bad{color:var(--bad)}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.6rem 0;
  border:1px solid var(--line);border-radius:6px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:.4rem .6rem;text-align:left;white-space:nowrap;border-bottom:1px solid var(--line)}
th{background:var(--chip);font-weight:600}
tbody tr:last-child td{border-bottom:0}
pre.cmd{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:.7rem .8rem;
  overflow-x:auto;font-size:.82rem;white-space:pre-wrap;overflow-wrap:anywhere}
.flag{font-size:.72rem;background:var(--chip);color:var(--muted);border-radius:999px;
  padding:.1rem .5rem;white-space:nowrap}
.flag.bad{color:var(--bad)}
.flag.ok{color:var(--ok)}
.pair{background:var(--panel);border:1px solid var(--line);border-radius:8px;
  padding:.9rem 1rem;margin:1rem 0;box-shadow:var(--shadow)}
.pair-head{display:flex;gap:.5rem;align-items:center;flex-wrap:wrap;margin-bottom:.5rem}
.pid{font-size:.9rem}
details.prompt{margin:.3rem 0 .7rem;font-size:.85rem}
details.prompt summary{cursor:pointer;color:var(--muted)}
details.prompt pre{background:var(--code);border-radius:6px;padding:.6rem .7rem;margin:.4rem 0 0;
  white-space:pre-wrap;overflow-wrap:anywhere;max-height:22rem;overflow:auto}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:.8rem}
@media (max-width:760px){.cols{grid-template-columns:1fr}}
.pane{border:1px solid var(--line);border-radius:6px;padding:.6rem .7rem;min-width:0}
pre.answer{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font-size:.83rem;
  max-height:26rem;overflow:auto}
.model .role{color:var(--muted)}
.reveal-only{display:none}
.reveal-inline{display:none}
body[data-reveal="1"] .reveal-only{display:block}
body[data-reveal="1"] .reveal-inline{display:inline}
.verdict{margin-top:.7rem;border-top:1px dashed var(--line);padding-top:.4rem;font-size:.85rem}
.labelrow{display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;margin-top:.8rem}
button{font:inherit;font-size:.85rem;padding:.35rem .8rem;border-radius:6px;
  border:1px solid var(--line);background:var(--chip);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
button[aria-pressed="true"]{background:var(--accent);color:var(--accent-ink);
  border-color:var(--accent)}
input.why{font:inherit;font-size:.85rem;padding:.35rem .5rem;border-radius:6px;
  border:1px solid var(--line);background:var(--bg);color:var(--ink);flex:1 1 14rem;min-width:0}
.saved{font-size:.78rem;color:var(--muted)}
#bar{position:fixed;left:0;right:0;bottom:0;z-index:10;display:flex;gap:.5rem;align-items:center;
  flex-wrap:wrap;padding:.6rem 1rem;background:var(--panel);border-top:1px solid var(--line);
  font-size:.85rem}
#bar .spacer{flex:1 1 auto}
#status{color:var(--muted);font-size:.78rem}
#yamlbox{position:fixed;left:0;right:0;bottom:3.2rem;z-index:9;background:var(--panel);
  border-top:1px solid var(--line);padding:.6rem 1rem;font-size:.8rem;color:var(--muted)}
#yamlbox textarea{width:100%;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  font-size:.78rem;background:var(--bg);color:var(--ink);border:1px solid var(--line);
  border-radius:6px;padding:.5rem}
"""

_JS = """
(function () {
  var node = document.getElementById('evalmine-data');
  var DATA = JSON.parse(node.textContent);
  var KEY = 'evalmine:labels:' + DATA.run_id;
  var byId = {};
  DATA.pairs.forEach(function (p) { byId[p.pair_id] = p; });

  function load() {
    try {
      var raw = window.localStorage.getItem(KEY);
      return raw ? (JSON.parse(raw) || {}) : {};
    } catch (e) { return {}; }
  }
  function save(state) {
    try { window.localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {}
  }
  var state = load();

  function labelled() {
    return DATA.pairs.filter(function (p) {
      return state[p.pair_id] && state[p.pair_id].choice;
    });
  }
  function progress() {
    var n = labelled().length;
    document.getElementById('progress').textContent =
      n + ' labeled of ' + DATA.pairs.length + ' pairs';
  }

  // The mapping from a click to a suite `prefer` value is NOT computed here.
  // Python generated prefer_by_choice for each pair and baked it into the page.
  function prefer(pair, choice) { return pair.prefer_by_choice[choice]; }

  function yamlScalar(s) {
    return /^[A-Za-z0-9][A-Za-z0-9_.\\/-]*$/.test(s) ? s : JSON.stringify(s);
  }
  function yaml() {
    var lines = ['labels:'];
    DATA.pairs.forEach(function (p) {
      var rec = state[p.pair_id];
      if (!rec || !rec.choice) { return; }
      var parts = [
        'task: ' + yamlScalar(p.task),
        'case: ' + yamlScalar(p.case),
        'baseline: ' + yamlScalar(p.baseline),
        'candidate: ' + yamlScalar(p.candidate),
        'prefer: ' + prefer(p, rec.choice)
      ];
      if (rec.note) { parts.push('note: ' + JSON.stringify(rec.note)); }
      lines.push('  - { ' + parts.join(', ') + ' }');
    });
    return lines.join('\\n') + '\\n';
  }

  function paint(row) {
    var id = row.getAttribute('data-pair');
    var rec = state[id] || {};
    Array.prototype.forEach.call(row.querySelectorAll('button.choice'), function (b) {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-choice') === rec.choice));
    });
    row.querySelector('input.why').value = rec.note || '';
    row.querySelector('.saved').textContent = rec.choice ? 'saved' : '';
  }

  Array.prototype.forEach.call(document.querySelectorAll('.labelrow'), function (row) {
    var id = row.getAttribute('data-pair');
    paint(row);
    row.addEventListener('click', function (event) {
      var button = event.target.closest ? event.target.closest('button.choice') : null;
      if (!button) { return; }
      var choice = button.getAttribute('data-choice');
      var rec = state[id] || {};
      rec.choice = rec.choice === choice ? null : choice;
      state[id] = rec;
      save(state);
      paint(row);
      progress();
    });
    row.querySelector('input.why').addEventListener('input', function (event) {
      var rec = state[id] || {};
      rec.note = event.target.value;
      state[id] = rec;
      save(state);
    });
  });

  document.getElementById('reveal').addEventListener('click', function (event) {
    var on = document.body.getAttribute('data-reveal') === '1';
    document.body.setAttribute('data-reveal', on ? '0' : '1');
    event.currentTarget.setAttribute('aria-pressed', String(!on));
    event.currentTarget.textContent = on ? 'Reveal models' : 'Hide models';
  });

  document.getElementById('copy').addEventListener('click', function () {
    var text = yaml();
    var box = document.getElementById('yamlbox');
    var out = document.getElementById('yamlout');
    box.hidden = false;
    out.value = text;
    out.select();
    var status = document.getElementById('status');
    var n = labelled().length;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        status.textContent = n + ' labels copied';
      }, function () {
        status.textContent = 'copy blocked - select the text below';
      });
    } else {
      status.textContent = 'select the text below and copy';
    }
  });

  document.getElementById('reset').addEventListener('click', function () {
    if (!window.confirm('Clear every label on this page? The suite file is not touched.')) {
      return;
    }
    state = {};
    save(state);
    Array.prototype.forEach.call(document.querySelectorAll('.labelrow'), paint);
    document.getElementById('yamlbox').hidden = true;
    document.getElementById('status').textContent = 'cleared';
    progress();
  });

  document.body.setAttribute('data-reveal', '0');
  progress();
})();
"""
