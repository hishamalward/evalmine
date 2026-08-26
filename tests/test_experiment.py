"""The v2 experiment contract and its side-effect-free planner."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalmine.experiment import ExperimentError, build_plan, load_experiment

REPO_ROOT = Path(__file__).resolve().parents[1]

MINIMAL_EXPERIMENT: dict[str, Any] = {
    "experiment": "tiny-agent-test",
    "version": 2,
    "question": "Which arm does the task better?",
    "seed": {"repo": ".", "ref": "HEAD", "dirty": "reject", "untracked": "deny"},
    "isolation": {
        "workspace": "copy",
        "session": "fresh-per-run",
        "external_writes": "deny",
    },
    "schedule": {"order": "rotate", "max_parallel": 2},
    "validators": {
        "repo-unchanged": {"type": "repository-diff", "expect": "unchanged"}
    },
    "arms": [
        {
            "id": "old",
            "runner": "claude-code",
            "model": "old-model",
            "auth": "subscription",
            "configuration": {"instructions": "inherit", "plugins": "inherit"},
        },
        {
            "id": "new",
            "runner": "claude-code",
            "model": "new-model",
            "auth": "subscription",
            "configuration": {"instructions": "none", "plugins": "none"},
        },
    ],
    "episodes": [
        {
            "id": "review",
            "turns": [{"prompt": "Review the repository."}, {"prompt": "Now reconsider."}],
            "validators": ["repo-unchanged"],
            "repeats": 2,
        }
    ],
    "evaluation": {
        "objectives": ["Correctness"],
        "blind": "arm-identity",
        "human": {"required": True},
        "judge": {
            "enabled": True,
            "pairwise": True,
            "position_swap": True,
            "calibrate": True,
        },
    },
}


@pytest.fixture
def write_experiment(tmp_path: Path):
    def _write(doc: dict[str, Any], name: str = "experiment.yaml") -> Path:
        doc["seed"]["repo"] = str(REPO_ROOT)
        path = tmp_path / name
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.mark.parametrize("filename", ["agent-model-comparison.yaml", "agent-config-ablation.yaml"])
def test_example_manifest_loads(filename):
    root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(root / "examples" / filename)
    assert len(experiment.arms) == 3
    assert len(experiment.episodes[0].turns) == 2
    assert experiment.episodes[0].turns[0].prompt_file is not None


def test_plan_expands_every_arm_episode_and_repeat(write_experiment):
    experiment = load_experiment(write_experiment(copy.deepcopy(MINIMAL_EXPERIMENT)))
    plan = build_plan(experiment)
    assert len(plan.runs) == 4
    assert [run.arm_id for run in plan.runs] == ["old", "new", "new", "old"]
    assert [run.repeat for run in plan.runs] == [1, 1, 2, 2]
    assert len({run.run_key for run in plan.runs}) == 4
    assert len({run.session_key for run in plan.runs}) == 4


def test_plan_is_deterministic(write_experiment):
    path = write_experiment(copy.deepcopy(MINIMAL_EXPERIMENT))
    first = build_plan(load_experiment(path)).as_dict()
    second = build_plan(load_experiment(path)).as_dict()
    assert first == second


def test_machine_plan_exposes_treatments_and_evaluation(write_experiment):
    experiment = load_experiment(write_experiment(copy.deepcopy(MINIMAL_EXPERIMENT)))
    plan = build_plan(experiment).as_dict()
    assert plan["arms"][1]["configuration"]["instructions"] == "none"
    assert plan["arms"][1]["configuration"]["plugins"] == "none"
    assert plan["validators"]["repo-unchanged"]["type"] == "repository-diff"
    assert plan["episodes"][0]["turns"][0]["prompt_sha256"]
    assert plan["evaluation"]["blind"] == "arm-identity"


def test_seed_ref_is_resolved_to_a_commit(write_experiment):
    experiment = load_experiment(write_experiment(copy.deepcopy(MINIMAL_EXPERIMENT)))
    assert len(experiment.seed.commit) == 40
    assert all(character in "0123456789abcdef" for character in experiment.seed.commit)


def test_unknown_seed_ref_is_an_error(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["seed"]["ref"] = "definitely-not-a-real-ref"
    with pytest.raises(ExperimentError, match="does not resolve to a commit"):
        load_experiment(write_experiment(doc))


def test_episode_cannot_reference_an_undeclared_validator(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["episodes"][0]["validators"].append("typo-check")
    with pytest.raises(ExperimentError, match="undeclared validator 'typo-check'"):
        load_experiment(write_experiment(doc))


def test_file_section_validator_requires_a_path(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["validators"]["sections"] = {
        "type": "required-sections",
        "target": "file",
        "sections": ["Results"],
    }
    with pytest.raises(ExperimentError, match="not valid"):
        load_experiment(write_experiment(doc))


def test_duplicate_arm_id_is_an_error(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["arms"][1]["id"] = "old"
    with pytest.raises(ExperimentError, match="duplicate arm id"):
        load_experiment(write_experiment(doc))


def test_unknown_key_is_an_error(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["arms"][0]["modle"] = "typo"
    with pytest.raises(ExperimentError, match="modle"):
        load_experiment(write_experiment(doc))


def test_api_auth_requires_an_explicit_per_arm_cost_ceiling(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["arms"][0]["auth"] = "api"
    with pytest.raises(ExperimentError, match="max_cost_usd"):
        load_experiment(write_experiment(doc))
    doc["arms"][0]["max_cost_usd"] = 0.25
    experiment = load_experiment(write_experiment(doc))
    assert experiment.arms[0].max_cost_usd == 0.25


def test_prompt_requires_exactly_one_source(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["episodes"][0]["turns"][0]["prompt_file"] = "also.md"
    with pytest.raises(ExperimentError):
        load_experiment(write_experiment(doc))


def test_missing_prompt_file_is_an_error(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["episodes"][0]["turns"] = [{"prompt_file": "missing.md"}]
    with pytest.raises(ExperimentError, match="not a readable file"):
        load_experiment(write_experiment(doc))


def test_instruction_files_must_exist(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    config = doc["arms"][0]["configuration"]
    config["instructions"] = "files"
    config["instruction_files"] = ["missing.md"]
    with pytest.raises(ExperimentError, match="instruction file"):
        load_experiment(write_experiment(doc))


def test_unused_instruction_file_setting_is_an_error(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["arms"][0]["configuration"]["instruction_files"] = ["ignored.md"]
    with pytest.raises(ExperimentError):
        load_experiment(write_experiment(doc))


def test_manifest_refuses_embedded_credentials(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["question"] = "Use " + "sk-" + "A" * 24
    with pytest.raises(ExperimentError, match="API key"):
        load_experiment(write_experiment(doc))


def test_bias_prone_settings_are_visible_as_plan_warnings(write_experiment):
    doc = copy.deepcopy(MINIMAL_EXPERIMENT)
    doc["isolation"]["session"] = "reuse-per-arm"
    doc["schedule"]["order"] = "fixed"
    plan = build_plan(load_experiment(write_experiment(doc)))
    assert len(plan.warnings) == 2
    assert "leak history" in plan.warnings[0]
    assert "fixed arm order" in plan.warnings[1]
