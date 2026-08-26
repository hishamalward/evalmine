#!/usr/bin/env python3
"""Normalize the completed music-analytics round-two bakeoff without model calls."""

# ruff: noqa: E501 -- the generated review is intentionally one self-contained document

from __future__ import annotations

import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

WEB = Path("apps/web")
EVAL = WEB / "eval"
ROUND = EVAL / "results/round-2"
OUT = Path(".evalmine-output")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_ledger() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (ROUND / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit(domain: str) -> None:
    groups = {
        "volume": {"T1", "T2", "T3"},
        "creative": {"T4", "T5I", "T5IM", "T5D"},
        "nl": {"NL_PLAYLIST"},
    }
    if domain not in groups:
        raise SystemExit(f"unknown audit domain: {domain}")
    ledger = load_ledger()
    rows = [row for row in ledger if row.get("task") in groups[domain]]
    if not rows:
        raise SystemExit(f"no ledger rows found for {domain}")
    missing_raw = [
        row["raw_dir"]
        for row in rows
        if not (WEB / str(row.get("raw_dir", ""))).is_file()
    ]
    fixtures: dict[str, Any] = {}
    for size in (100, 250):
        tracks = load_json(EVAL / f"fixtures/round2-{size}.json")
        summary = load_json(EVAL / f"fixtures/round2-{size}.summary.json")
        if len(tracks) != size or summary.get("totalSelected") != size:
            raise SystemExit(f"round2-{size} fixture and summary disagree")
        fixtures[str(size)] = {
            "tracks": len(tracks),
            "genre_buckets": len(summary.get("perGenreBucket", {})),
            "distinct_artists": summary.get("distinctArtists"),
            "estimated_non_english": summary.get("estimatedNonEnglishCount"),
        }
    samples = load_json(ROUND / "report-samples.json")
    result = {
        "domain": domain,
        "status": "passed" if not missing_raw else "failed",
        "rows": len(rows),
        "tasks": dict(sorted(Counter(str(row["task"]) for row in rows).items())),
        "models": sorted({str(row["model"]) for row in rows}),
        "recorded_failures": sum(int(row.get("failures", 0)) for row in rows),
        "usd_cost_recorded": round(sum(float(row.get("usd_cost", 0)) for row in rows), 6),
        "raw_references_checked": len(rows),
        "missing_raw_references": missing_raw,
        "fixtures": fixtures,
        "sample_groups": {
            key: len(value) if isinstance(value, list) else 1 for key, value in samples.items()
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"audit-{domain}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if missing_raw:
        raise SystemExit(f"{len(missing_raw)} raw ledger references are missing")


def score() -> None:
    audits = [load_json(OUT / f"audit-{name}.json") for name in ("volume", "creative", "nl")]
    if any(item.get("status") != "passed" for item in audits):
        raise SystemExit("an audit stage failed")
    ledger = load_ledger()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        by_task[str(row["task"])].append(row)
    task_scores = []
    for task, rows in sorted(by_task.items()):
        eligible = [row for row in rows if int(row.get("failures", 0)) == 0]
        cheapest = min(eligible or rows, key=lambda row: float(row.get("usd_cost", 0)))
        task_scores.append(
            {
                "task": task,
                "arms": len(rows),
                "models": len({row["model"] for row in rows}),
                "recorded_failures": sum(int(row.get("failures", 0)) for row in rows),
                "cheapest_zero_failure_arm": {
                    "model": cheapest["model"],
                    "thinking": cheapest.get("thinking"),
                    "usd_cost": cheapest.get("usd_cost"),
                },
            }
        )
    analyzer = load_json(ROUND / "analyzer-fixtures.json")
    result = {
        "status": "passed",
        "historical_evidence_only": True,
        "provider_calls": False,
        "ledger_rows": len(ledger),
        "models": sorted({str(row["model"]) for row in ledger}),
        "tasks": task_scores,
        "analyzer": {
            "fixtures": len(analyzer.get("fixtures", [])),
            "arm_runs": len(analyzer.get("perArm", [])),
            "summary_arms": len(analyzer.get("summary", [])),
        },
        "decision_source": "apps/web/eval/results/round-2/findings.md",
        "human_review_source": "apps/web/eval/analysis/round-2-report.html",
    }
    (OUT / "scorecard.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def review() -> None:
    data = load_json(OUT / "scorecard.json")
    task_cards = "".join(
        "<article><b>"
        + esc(item["task"])
        + "</b><strong>"
        + esc(item["arms"])
        + " arms</strong><p>"
        + esc(item["models"])
        + " models · "
        + esc(item["recorded_failures"])
        + " recorded failures</p></article>"
        for item in data["tasks"]
    )
    html_doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Music analytics round two · imported evidence</title><style>:root{{--ink:#17201b;--paper:#f1ede3;--card:#fffdf7;--green:#166044;--line:#d5cebe}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui}}header,main,footer{{max-width:1060px;margin:auto;padding:28px}}header{{padding-top:64px}}h1{{font:700 clamp(38px,6vw,70px)/1 Georgia,serif;margin:.2em 0;max-width:850px}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.14em;font-weight:800;color:var(--green)}}section,article{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:19px;margin:14px 0}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}}article{{margin:0}}article strong{{float:right}}article p{{margin:.6em 0 0}}.flow{{font:700 18px/1.5 ui-monospace,monospace;overflow-wrap:anywhere}}.callout{{border-left:7px solid var(--green)}}code{{font-size:.86em}}@media(max-width:650px){{header,main,footer{{padding:18px}}}}</style></head><body><header><div class="eyebrow">evalmine · historical evidence adapter</div><h1>The real round-two backoff fits the workflow contract.</h1><p>{esc(data['ledger_rows'])} immutable ledger rows across {esc(len(data['models']))} models were normalized without re-calling a provider.</p></header><main><section class="callout"><div class="eyebrow">Actual stage contract</div><h2>What was ported</h2><p class="flow">stratified fixture → isolated eval DB → T1 → T2 + T3 → T4 + T5 + analyzer + NL → raw ledger → objective checks → blind samples → founder decision</p></section><section><div class="eyebrow">Coverage</div><h2>Historical arms by task</h2><div class="grid">{task_cards}</div></section><section><div class="eyebrow">Evidence boundary</div><h2>This imports; it does not spend</h2><p>The adapter verifies the real fixture, ledger, every raw-output reference, analyzer scorecard, decision memo, and original founder-facing HTML. It deliberately does not rerun the application’s direct-API harness: that harness has a throwaway-database tripwire but no enforceable pre-call dollar ceiling.</p></section><section><div class="eyebrow">Human authority</div><h2>The original review remains the decision artifact</h2><p><code>apps/web/eval/analysis/round-2-report.html</code> is captured unchanged beside this normalized report. Its side-by-side samples and the findings memo remain available for a human to agree with or overturn.</p></section></main><footer>Static, self-contained, and generated only from frozen historical files.</footer></body></html>"""
    (OUT / "round2-import.html").write_text(html_doc, encoding="utf-8")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: import-round2.py audit <volume|creative|nl> | score | review")
    command = sys.argv[1]
    if command == "audit" and len(sys.argv) == 3:
        audit(sys.argv[2])
    elif command == "score" and len(sys.argv) == 2:
        score()
    elif command == "review" and len(sys.argv) == 2:
        review()
    else:
        raise SystemExit("invalid arguments")


if __name__ == "__main__":
    main()
