"""First-class import of completed, externally generated artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalmine.cli import main
from evalmine.decision import (
    EpisodeJudgeCall,
    JudgeRefused,
    generate_decision_report,
    judge_experiment,
)
from evalmine.experiment_report import (
    build_experiment_report_data,
    generate_experiment_report,
)
from evalmine.external import (
    ExternalArtifactError,
    import_external_artifacts,
    verify_external_import,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _bundle(
    tmp_path: Path,
    *,
    conditions: int = 3,
    blocks: int = 2,
    ranking_style: str = "n-way",
) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for item_index in range(1, blocks + 1):
        for condition_index in range(1, conditions + 1):
            rows.append(
                {
                    "lane": "structured-summary",
                    "item_id": f"item-{item_index}",
                    "account_id": "account-redacted",
                    "prompt": f"Summarize item {item_index} using the declared fields.",
                    "condition": {
                        "id": f"condition-{condition_index}",
                        "model": f"provider/model-{condition_index}",
                        "prompt_variant": f"prompt-{condition_index}",
                        "width": 80 + condition_index,
                    },
                    "output": {
                        "summary": f"blind output {item_index}/{condition_index}",
                        "confidence": condition_index,
                    },
                    "cost_receipts": {
                        "estimated": {
                            "usd": 0.01 * condition_index,
                            "source": "pinned token estimate",
                        },
                        "ledger": {
                            "usd": 0.011 * condition_index,
                            "source": "external spend ledger row",
                        },
                        "dashboard_observed": {
                            "usd": 0.012 * condition_index,
                            "source": "provider dashboard delta",
                        },
                    },
                }
            )
    raw = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows
    )
    artifact = bundle / "completed.jsonl"
    artifact.write_bytes(raw)
    manifest = {
        "external_artifacts": "sanitized-external-example",
        "version": 1,
        "question": "Which completed condition produces the strongest artifact?",
        "artifacts": [
            {"path": "completed.jsonl", "sha256": hashlib.sha256(raw).hexdigest()}
        ],
        "evaluation": {
            "objectives": ["Correctness", "Usefulness"],
            "blind": "condition",
            "ranking_style": ranking_style,
            "fields": ["summary", "confidence"],
            "human": {
                "required": True,
                "labels_per_pair": 1,
                "coverage": "calibration-subset",
            },
            "judge": {
                "enabled": True,
                "pairwise": ranking_style == "pairwise",
                "position_swap": ranking_style == "pairwise",
                "calibrate": True,
                "runner": "api-prompt",
                "model": "fake/judge",
                "min_kappa": 0.4,
                "min_labels": 2,
            },
        },
    }
    (bundle / "evalmine-import.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    return bundle


def test_external_import_is_pinned_reportable_and_fully_condition_blind(tmp_path: Path):
    bundle = _bundle(tmp_path)
    result = import_external_artifacts(bundle, tmp_path / "evidence")
    assert result.record_count == 6
    assert result.block_count == 2
    assert result.condition_count == 3
    verification = verify_external_import(result.root)
    assert verification["provider_calls"] is False

    report = build_experiment_report_data(result.root)
    assert report["source_format"] == "evalmine-external-artifacts-v1"
    assert report["run_count"] == 6
    assert report["pair_count"] == 6
    assert report["ranking_count"] == 2
    assert report["blind"] == "full-condition"
    assert report["human"]["coverage"] == "calibration-subset"
    totals = report["pricing"]["receipt_totals"]
    assert totals["estimated"]["usd"] == pytest.approx(0.12)
    assert totals["ledger"]["usd"] == pytest.approx(0.132)
    assert totals["dashboard_observed"]["usd"] == pytest.approx(0.144)
    assert totals["reconciliation"]["ledger_to_estimated"] == pytest.approx(1.1)

    generated = generate_experiment_report(result.root)
    html = generated.html.read_text(encoding="utf-8")
    report_data = (generated.root / "data.json").read_text(encoding="utf-8")
    assert '<button id="reveal"' not in html
    assert "condition-1" not in html
    assert "provider/model-1" not in html
    assert "condition-1" not in report_data
    assert "provider/model-1" not in report_data
    assert "prompt-1" not in report_data
    assert '"condition_mapping": "decision-only"' in report_data
    assert "Full condition identities are omitted" in html
    assert "blind output 1/1" in html
    assert html.count("data-field-best=") == 4
    assert html.count("data-field-wrong=") == 12
    assert "Best and wrong by field" in html


def test_external_import_refuses_hash_mismatch_and_detects_tampering(tmp_path: Path):
    bundle = _bundle(tmp_path)
    manifest_path = bundle / "evalmine-import.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["artifacts"][0]["sha256"] = "0" * 64
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    with pytest.raises(ExternalArtifactError, match="sha256 mismatch"):
        import_external_artifacts(bundle, tmp_path / "bad")

    bundle = _bundle(tmp_path / "second")
    result = import_external_artifacts(bundle, tmp_path / "evidence")
    artifacts = result.root / "artifacts.jsonl"
    artifacts.chmod(0o644)
    artifacts.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ExternalArtifactError, match="changed after import"):
        verify_external_import(result.root)


def test_external_import_cli_reports_zero_provider_calls(tmp_path: Path, capsys):
    bundle = _bundle(tmp_path)
    out = tmp_path / "evidence"
    assert main(["experiment", "import", str(bundle), "--out", str(out)]) == 0
    assert "provider calls: 0" in capsys.readouterr().out
    assert main(["experiment", "verify", str(out)]) == 0
    assert "pinned external artifacts" in capsys.readouterr().out


def test_external_import_preserves_optional_producer_correlation_id(tmp_path: Path):
    bundle = _bundle(tmp_path)
    artifact = bundle / "completed.jsonl"
    rows = [json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()]
    rows[0]["correlation_id"] = "synthetic:summary:item-1:condition-1"
    raw = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    artifact.write_bytes(raw)
    manifest_path = bundle / "evalmine-import.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    result = import_external_artifacts(bundle, tmp_path / "correlated-evidence")
    normalized = [
        json.loads(line)
        for line in (result.root / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert normalized[0]["correlation_id"] == "synthetic:summary:item-1:condition-1"


def test_sanitized_external_example_imports_without_generation(tmp_path: Path):
    result = import_external_artifacts(
        REPO_ROOT / "examples" / "external-artifacts", tmp_path / "example-evidence"
    )
    assert result.record_count == 4
    assert result.block_count == 2
    assert result.condition_count == 2
    normalized = [
        json.loads(line)
        for line in (result.root / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    correlation_ids = [record["correlation_id"] for record in normalized]
    assert all(correlation_ids)
    assert len(set(correlation_ids)) == result.record_count
    assert all(len(value) <= 120 for value in correlation_ids)


def test_partial_human_subset_can_calibrate_judge_extension(tmp_path: Path):
    bundle = _bundle(tmp_path, ranking_style="pairwise")
    result = import_external_artifacts(bundle, tmp_path / "evidence")
    report = build_experiment_report_data(result.root)
    labels = []
    for pair, prefer in zip(report["pairs"][:2], ("candidate", "baseline"), strict=True):
        ordered = sorted((pair["a_run_key"], pair["b_run_key"]))
        preferred = ordered[1] if prefer == "candidate" else ordered[0]
        labels.append(
            {
                "pair_id": pair["pair_id"],
                "choice": "A" if preferred == pair["a_run_key"] else "B",
                "preferred_run_key": preferred,
                "identities_revealed": False,
            }
        )
    label_path = tmp_path / "calibration-labels.json"
    label_path.write_text(
        json.dumps(
            {
                "format": "evalmine-human-labels-v1",
                "plan_id": report["plan_id"],
                "annotator": "human-calibrator",
                "labels": labels,
            }
        ),
        encoding="utf-8",
    )
    scripted: list[tuple[str, str]] = []
    for pair_index in range(len(report["pairs"])):
        if pair_index == 0:
            scripted.extend([("2", "candidate"), ("1", "candidate")])
        elif pair_index == 1:
            scripted.extend([("1", "baseline"), ("2", "baseline")])
        else:
            scripted.extend([("tie", "tie"), ("tie", "tie")])
    calls = iter(scripted)

    def judge_call(_prompt: str) -> EpisodeJudgeCall:
        winner, reason = next(calls)
        return EpisodeJudgeCall(
            text=json.dumps({"winner": winner, "reason": reason}),
            latency_ms=1,
            input_tokens=10,
            output_tokens=3,
            cost_usd=0.0,
            raw="safe fake judge output",
            observed_model="fake/judge",
        )

    judging = judge_experiment(
        result.root,
        allow_provider_calls=False,
        caller=judge_call,
        calibration_label_paths=[label_path],
    )
    assert judging.status == "completed"
    assert judging.pair_count == 6
    judging_evidence = json.loads((judging.root / "pairs.json").read_text())
    calibration_source = judging_evidence["calibration_label_sources"][0]
    assert (judging.root / calibration_source["evidence_copy"]).read_bytes() == (
        label_path.read_bytes()
    )
    decision = generate_decision_report(result.root, [label_path])
    data = json.loads((decision.root / "data.json").read_text())
    assert data["human_complete"] is False
    assert data["labelled_pairs"] == 2
    assert data["judged_pairs"] == 6
    assert data["headline_eligible"] is True
    assert data["evidence_populations"] == {
        "human_calibration_pairs": 2,
        "judge_total_pairs": 6,
        "judge_extended_pairs": 4,
    }
    html = decision.html.read_text(encoding="utf-8")
    assert "Human calibration and judge extension stay separate" in html
    assert "Each artifact, side by side by basis" in html
    assert "Dashboard observed" in html
    assert "external spend ledger row" in html
    assert "Blind labels mapped to full conditions" in html


def test_failed_calibration_stops_judge_before_extension(tmp_path: Path):
    result = import_external_artifacts(
        _bundle(tmp_path, ranking_style="pairwise"), tmp_path / "evidence"
    )
    report = build_experiment_report_data(result.root)
    labels = []
    for pair, prefer in zip(report["pairs"][:2], ("candidate", "baseline"), strict=True):
        ordered = sorted((pair["a_run_key"], pair["b_run_key"]))
        preferred = ordered[1] if prefer == "candidate" else ordered[0]
        labels.append(
            {
                "pair_id": pair["pair_id"],
                "choice": "A" if preferred == pair["a_run_key"] else "B",
                "preferred_run_key": preferred,
                "identities_revealed": False,
            }
        )
    label_path = tmp_path / "calibration-labels.json"
    label_path.write_text(
        json.dumps(
            {
                "format": "evalmine-human-labels-v1",
                "plan_id": report["plan_id"],
                "annotator": "human-calibrator",
                "labels": labels,
            }
        ),
        encoding="utf-8",
    )
    call_count = 0

    def judge_call(_prompt: str) -> EpisodeJudgeCall:
        nonlocal call_count
        call_count += 1
        return EpisodeJudgeCall(
            text=json.dumps({"winner": "tie", "reason": "calibration mismatch"}),
            latency_ms=1,
            cost_usd=0.0,
        )

    judging = judge_experiment(
        result.root,
        allow_provider_calls=False,
        caller=judge_call,
        calibration_label_paths=[label_path],
    )
    assert judging.status == "calibration_failed"
    assert judging.pair_count == 2
    assert judging.call_count == 4
    assert call_count == 4
    evidence = json.loads((judging.root / "pairs.json").read_text())
    assert evidence["extension_started"] is False
    assert evidence["calibration_gate"]["status"] == "below_floor"


def test_insufficient_calibration_labels_refuse_before_judge_calls(tmp_path: Path):
    result = import_external_artifacts(
        _bundle(tmp_path, ranking_style="pairwise"), tmp_path / "evidence"
    )
    report = build_experiment_report_data(result.root)
    pair = report["pairs"][0]
    preferred = sorted((pair["a_run_key"], pair["b_run_key"]))[1]
    label_path = tmp_path / "one-label.json"
    label_path.write_text(
        json.dumps(
            {
                "format": "evalmine-human-labels-v1",
                "plan_id": report["plan_id"],
                "labels": [
                    {
                        "pair_id": pair["pair_id"],
                        "choice": "A" if preferred == pair["a_run_key"] else "B",
                        "preferred_run_key": preferred,
                        "identities_revealed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    def should_not_run(_prompt: str) -> EpisodeJudgeCall:
        raise AssertionError("judge caller must not run")

    with pytest.raises(JudgeRefused, match="fewer than the configured minimum"):
        judge_experiment(
            result.root,
            allow_provider_calls=False,
            caller=should_not_run,
            calibration_label_paths=[label_path],
        )
    assert not (result.root / "judging").exists()


def test_n_way_field_best_and_wrong_mappings_are_validated_and_revealed(tmp_path: Path):
    result = import_external_artifacts(_bundle(tmp_path), tmp_path / "evidence")
    report = build_experiment_report_data(result.root)
    ranking_rows = []
    for ranking in report["rankings"]:
        labels = [outcome["label"] for outcome in ranking["outcomes"]]
        by_label = ranking["run_key_by_label"]
        ranking_rows.append(
            {
                "ranking_id": ranking["ranking_id"],
                "order": labels,
                "ranked_run_keys": [by_label[label] for label in labels],
                "unclear": False,
                "field_labels": [
                    {
                        "field": field,
                        "best_label": labels[0],
                        "best_run_key": by_label[labels[0]],
                        "wrong_labels": [labels[-1]],
                        "wrong_run_keys": [by_label[labels[-1]]],
                    }
                    for field in ranking["fields"]
                ],
                "identities_revealed": False,
            }
        )
    labels_path = tmp_path / "field-labels.json"
    labels_path.write_text(
        json.dumps(
            {
                "format": "evalmine-human-rankings-v1",
                "plan_id": report["plan_id"],
                "annotator": "field-reviewer",
                "rankings": ranking_rows,
            }
        ),
        encoding="utf-8",
    )
    decision = generate_decision_report(result.root, [labels_path])
    data = json.loads((decision.root / "data.json").read_text())
    human_review = data["n_way_review"]["human"]
    assert len(human_review) == 2
    assert {row["field"] for row in human_review[0]["field_labels"]} == {
        "summary",
        "confidence",
    }
    html = decision.html.read_text(encoding="utf-8")
    assert "Field labels" in html
    assert "wrong condition-" in html


def test_n_way_calibration_subset_gates_then_extends(tmp_path: Path):
    result = import_external_artifacts(_bundle(tmp_path), tmp_path / "evidence")
    report = build_experiment_report_data(result.root)
    calibration_ranking = report["rankings"][0]
    label_by_run = {
        outcome["run_key"]: outcome["label"]
        for outcome in calibration_ranking["outcomes"]
    }
    sorted_runs = sorted(label_by_run)
    calibration_run_order = [sorted_runs[1], sorted_runs[2], sorted_runs[0]]
    calibration_order = [label_by_run[run_key] for run_key in calibration_run_order]
    label_path = tmp_path / "n-way-calibration.json"
    label_path.write_text(
        json.dumps(
            {
                "format": "evalmine-human-rankings-v1",
                "plan_id": report["plan_id"],
                "rankings": [
                    {
                        "ranking_id": calibration_ranking["ranking_id"],
                        "order": calibration_order,
                        "ranked_run_keys": calibration_run_order,
                        "unclear": False,
                        "field_labels": [
                            {
                                "field": field,
                                "best_label": calibration_order[0],
                                "best_run_key": calibration_run_order[0],
                                "wrong_labels": [calibration_order[-1]],
                                "wrong_run_keys": [calibration_run_order[-1]],
                            }
                            for field in calibration_ranking["fields"]
                        ],
                        "identities_revealed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    extension_ranking = report["rankings"][1]
    extension_order = [
        outcome["label"] for outcome in extension_ranking["outcomes"]
    ]
    responses = iter((calibration_order, extension_order))

    def judge_call(_prompt: str) -> EpisodeJudgeCall:
        return EpisodeJudgeCall(
            text=json.dumps({"ranking": next(responses), "reason": "strict order"}),
            latency_ms=1,
            cost_usd=0.0,
        )

    judging = judge_experiment(
        result.root,
        allow_provider_calls=False,
        caller=judge_call,
        calibration_label_paths=[label_path],
    )
    assert judging.status == "completed"
    assert judging.pair_count == 6
    assert judging.call_count == 2
    evidence = json.loads((judging.root / "pairs.json").read_text())
    assert evidence["calibration_gate"]["status"] == "ok"
    assert evidence["extension_started"] is True

    decision = generate_decision_report(result.root, [label_path])
    decision_data = json.loads((decision.root / "data.json").read_text())
    assert decision_data["human_complete"] is False
    assert decision_data["headline_eligible"] is True
    assert decision_data["evidence_populations"]["judge_extended_pairs"] == 3
