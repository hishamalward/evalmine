"""Offline fixture commands for examples/music-backoff-workflow.yaml."""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    mode, item = argv[0], argv[1] if len(argv) > 1 else None
    root = Path("workflow-output")
    fixture = json.loads(Path("workflow-input/catalog.json").read_text(encoding="utf-8"))
    if mode == "generate" and item:
        write_json(root / "drafts" / f"{item}.json", {"agent": item, "songs": fixture["songs"]})
    elif mode == "enrich" and item:
        draft = json.loads((root / "drafts" / f"{item}.json").read_text(encoding="utf-8"))
        write_json(root / "enriched" / f"{item}.json", {**draft, "enriched": True, "quality": len(item)})
    elif mode == "judge":
        rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((root / "enriched").glob("*.json"))]
        write_json(root / "judge" / "scores.json", {"scores": [{"agent": row["agent"], "score": row["quality"]} for row in rows]})
    elif mode == "score":
        scores = json.loads((root / "judge" / "scores.json").read_text(encoding="utf-8"))["scores"]
        winner = max(scores, key=lambda row: (row["score"], row["agent"]))
        write_json(root / "decision.json", {"winner": winner, "human_review_required": True})
        cards = "".join(f"<li>{html.escape(row['agent'])}: {row['score']}</li>" for row in scores)
        (root / "eye-test.html").write_text(f"<!doctype html><title>Backoff eye test</title><h1>Blind candidates</h1><ul>{cards}</ul>", encoding="utf-8")
    else:
        raise SystemExit(f"unknown fixture mode: {mode!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
