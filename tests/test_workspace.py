"""Phase-2 workspace isolation, evidence, verification, and disposal."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalmine.cli import main
from evalmine.experiment import ExperimentError, build_plan, load_experiment
from evalmine.workspace import (
    PreparationError,
    discard_prepared,
    prepare_experiment,
    verify_prepared,
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _worktree_paths(repo: Path) -> set[Path]:
    return {
        Path(line.removeprefix("worktree ")).resolve()
        for line in _git(repo, "worktree", "list", "--porcelain").splitlines()
        if line.startswith("worktree ")
    }


@pytest.fixture
def seed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "seed"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "evalmine tests")
    _git(repo, "config", "user.email", "evalmine@example.invalid")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("baseline instructions\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "value.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed")
    return repo


def _manifest_dict(repo: Path, manifest_dir: Path) -> dict[str, Any]:
    return {
        "experiment": "workspace-test",
        "version": 2,
        "question": "Which isolated arm works better?",
        "seed": {
            "repo": str(Path("..") / repo.name),
            "ref": "HEAD",
            "dirty": "reject",
            "untracked": "deny",
        },
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
                "id": "inherit",
                "runner": "claude-code",
                "model": "model-a",
                "auth": "subscription",
                "configuration": {"instructions": "inherit", "plugins": "inherit"},
            },
            {
                "id": "no-instructions",
                "runner": "claude-code",
                "model": "model-b",
                "auth": "subscription",
                "configuration": {"instructions": "none", "plugins": "none"},
            },
        ],
        "episodes": [
            {
                "id": "change",
                "turns": [{"prompt": "Inspect the repository and propose one change."}],
                "validators": ["repo-unchanged"],
                "repeats": 1,
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
def experiment_factory(tmp_path: Path, seed_repo: Path):
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()

    def _make(mutator=None):
        doc = _manifest_dict(seed_repo, manifest_dir)
        if mutator is not None:
            mutator(doc, manifest_dir)
        manifest = manifest_dir / "experiment.yaml"
        manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return load_experiment(manifest), manifest, tmp_path / "artifacts"

    return _make


def _runs_by_arm(prepared) -> dict[str, Path]:
    return {run.arm_id: run.workspace for run in prepared.runs}


def test_copy_workspaces_are_isolated_and_baseline_stays_unchanged(experiment_factory, seed_repo):
    experiment, _, out = experiment_factory()
    prepared = prepare_experiment(experiment, out)
    try:
        workspaces = _runs_by_arm(prepared)
        inherited = workspaces["inherit"]
        without = workspaces["no-instructions"]
        assert not (inherited / ".git").exists()
        assert (inherited / "CLAUDE.md").read_text(encoding="utf-8") == ("baseline instructions\n")
        assert not (without / "CLAUDE.md").exists()
        assert (prepared.root / "manifest.yaml").read_bytes() == experiment.manifest_bytes
        assert (prepared.root / "plan.json").is_file()
        assert (prepared.root / "environment.json").is_file()
        assert (prepared.root / "inputs" / "index.json").is_file()
        assert all((run.workspace.parent / "run.json").is_file() for run in prepared.runs)

        (inherited / "src" / "value.txt").write_text("changed in A\n", encoding="utf-8")
        assert (without / "src" / "value.txt").read_text(encoding="utf-8") == "one\n"
        assert (seed_repo / "src" / "value.txt").read_text(encoding="utf-8") == "one\n"
        assert verify_prepared(prepared.root)["baseline_unchanged"] is True
    finally:
        discard_prepared(prepared.root)


def test_dirty_reject_refuses_before_creating_artifacts(experiment_factory, seed_repo):
    experiment, _, out = experiment_factory()
    (seed_repo / "README.md").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="dirty=reject"):
        prepare_experiment(experiment, out)
    assert not out.exists()


def test_untracked_deny_refuses_before_creating_artifacts(experiment_factory, seed_repo):
    experiment, _, out = experiment_factory()
    (seed_repo / "new.txt").write_text("untracked\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="untracked=deny"):
        prepare_experiment(experiment, out)
    assert not out.exists()


def test_capture_overlays_tracked_and_untracked_files(experiment_factory, seed_repo):
    def configure(doc, _manifest_dir):
        doc["seed"]["dirty"] = "capture"
        doc["seed"]["untracked"] = "include"

    experiment, _, out = experiment_factory(configure)
    (seed_repo / "README.md").write_text("captured change\n", encoding="utf-8")
    (seed_repo / "new.txt").write_text("captured untracked\n", encoding="utf-8")
    prepared = prepare_experiment(experiment, out)
    try:
        for run in prepared.runs:
            assert (run.workspace / "README.md").read_text(encoding="utf-8") == (
                "captured change\n"
            )
            assert (run.workspace / "new.txt").read_text(encoding="utf-8") == (
                "captured untracked\n"
            )
    finally:
        discard_prepared(prepared.root)


def test_allowlisted_untracked_refuses_unexpected_files(experiment_factory, seed_repo):
    def configure(doc, _manifest_dir):
        doc["seed"]["untracked"] = "allowlisted"
        doc["seed"]["untracked_allowlist"] = ["allowed/**"]

    experiment, _, out = experiment_factory(configure)
    (seed_repo / "allowed").mkdir()
    (seed_repo / "allowed" / "note.txt").write_text("allowed\n", encoding="utf-8")
    (seed_repo / "surprise.txt").write_text("not allowed\n", encoding="utf-8")
    with pytest.raises(PreparationError, match="outside untracked_allowlist"):
        prepare_experiment(experiment, out)


def test_allowlisted_untracked_is_copied_when_every_file_matches(experiment_factory, seed_repo):
    def configure(doc, _manifest_dir):
        doc["seed"]["untracked"] = "allowlisted"
        doc["seed"]["untracked_allowlist"] = ["allowed/**"]

    experiment, _, out = experiment_factory(configure)
    (seed_repo / "allowed").mkdir()
    (seed_repo / "allowed" / "note.txt").write_text("allowed\n", encoding="utf-8")
    prepared = prepare_experiment(experiment, out)
    try:
        assert all((run.workspace / "allowed" / "note.txt").is_file() for run in prepared.runs)
    finally:
        discard_prepared(prepared.root)


def test_instruction_file_treatment_replaces_only_its_arm(experiment_factory):
    def configure(doc, manifest_dir):
        (manifest_dir / "candidate.md").write_text("candidate instructions\n", encoding="utf-8")
        config = doc["arms"][1]["configuration"]
        config["instructions"] = "files"
        config["instruction_files"] = ["candidate.md"]

    experiment, _, out = experiment_factory(configure)
    prepared = prepare_experiment(experiment, out)
    try:
        workspaces = _runs_by_arm(prepared)
        assert "baseline instructions" in (workspaces["inherit"] / "CLAUDE.md").read_text(
            encoding="utf-8"
        )
        candidate = (workspaces["no-instructions"] / "CLAUDE.md").read_text(encoding="utf-8")
        assert "candidate instructions" in candidate
        assert "baseline instructions" not in candidate
        treatment = json.loads(
            next(
                run.workspace.parent / "treatment.json"
                for run in prepared.runs
                if run.arm_id == "no-instructions"
            ).read_text(encoding="utf-8")
        )
        assert treatment["plugins"] == "none"
        assert treatment["plugin_enforcement"] == "runner-preflight"
    finally:
        discard_prepared(prepared.root)


def test_external_prompt_content_changes_plan_identity(experiment_factory):
    def configure(doc, manifest_dir):
        (manifest_dir / "task.md").write_text("first prompt\n", encoding="utf-8")
        doc["episodes"][0]["turns"] = [{"prompt_file": "task.md"}]

    first, manifest, _ = experiment_factory(configure)
    first_plan = build_plan(first)
    (manifest.parent / "task.md").write_text("second prompt\n", encoding="utf-8")
    second = load_experiment(manifest)
    assert first.hash == second.hash
    assert first.input_hash != second.input_hash
    assert first_plan.plan_id != build_plan(second).plan_id


def test_external_prompt_refuses_credentials(experiment_factory):
    def configure(doc, manifest_dir):
        keyish = "sk-" + "A" * 24
        (manifest_dir / "task.md").write_text(f"Use {keyish}\n", encoding="utf-8")
        doc["episodes"][0]["turns"] = [{"prompt_file": "task.md"}]

    with pytest.raises(ExperimentError, match="API key"):
        experiment_factory(configure)


def test_prepare_refuses_to_overwrite_existing_evidence(experiment_factory):
    experiment, _, out = experiment_factory()
    prepared = prepare_experiment(experiment, out)
    try:
        with pytest.raises(PreparationError, match="never overwritten"):
            prepare_experiment(experiment, out)
    finally:
        discard_prepared(prepared.root)


def test_workspace_baseline_blobs_redact_key_shaped_source_text(
    experiment_factory, seed_repo
):
    keyish = "sk-" + "C" * 24
    (seed_repo / "tracked-secret.txt").write_text(f"value={keyish}\n", encoding="utf-8")
    _git(seed_repo, "add", "tracked-secret.txt")
    _git(seed_repo, "commit", "-q", "-m", "secret-shaped fixture")
    experiment, _, out = experiment_factory()
    prepared = prepare_experiment(experiment, out)
    try:
        baseline = json.loads(
            (prepared.root / "runs" / prepared.runs[0].run_key / "baseline.json").read_text(
                encoding="utf-8"
            )
        )
        entry = next(
            item for item in baseline["entries"] if item["path"] == "tracked-secret.txt"
        )
        assert entry["blob_redacted"] is True
        blob = (prepared.root / entry["blob"]).read_text(encoding="utf-8")
        assert keyish not in blob
        assert "[REDACTED_CREDENTIAL]" in blob
    finally:
        discard_prepared(prepared.root)


def test_artifacts_must_live_outside_seed(experiment_factory, seed_repo):
    experiment, _, _ = experiment_factory()
    with pytest.raises(PreparationError, match="overlaps seed repository"):
        prepare_experiment(experiment, seed_repo / "artifacts")


def test_verify_detects_baseline_change(experiment_factory, seed_repo):
    experiment, _, out = experiment_factory()
    prepared = prepare_experiment(experiment, out)
    try:
        (seed_repo / "README.md").write_text("changed later\n", encoding="utf-8")
        with pytest.raises(PreparationError, match="changed after preparation"):
            verify_prepared(prepared.root)
    finally:
        discard_prepared(prepared.root)


def test_verify_detects_evidence_tampering(experiment_factory):
    experiment, _, out = experiment_factory()
    prepared = prepare_experiment(experiment, out)
    try:
        evidence = prepared.root / "plan.json"
        evidence.chmod(0o644)
        evidence.write_text("{}\n", encoding="utf-8")
        with pytest.raises(PreparationError, match="evidence changed"):
            verify_prepared(prepared.root)
    finally:
        discard_prepared(prepared.root)


def test_worktree_mode_registers_and_safely_discards_workspaces(experiment_factory, seed_repo):
    def configure(doc, _manifest_dir):
        doc["isolation"]["workspace"] = "worktree"

    experiment, _, out = experiment_factory(configure)
    prepared = prepare_experiment(experiment, out)
    root = prepared.root
    workspace_strings = [str(run.workspace) for run in prepared.runs]
    registered = _worktree_paths(seed_repo)
    assert all(run.workspace in registered for run in prepared.runs)
    assert all((run.workspace / ".git").is_file() for run in prepared.runs)
    result = discard_prepared(root)
    assert result["run_count"] == 2
    assert not root.exists()
    registered = _worktree_paths(seed_repo)
    assert all(Path(path) not in registered for path in workspace_strings)


def test_failed_worktree_preparation_cleans_registration(experiment_factory, seed_repo):
    def configure(doc, _manifest_dir):
        doc["isolation"]["workspace"] = "worktree"
        doc["arms"][0]["runner"] = "pipeline"
        doc["arms"][0]["configuration"]["instructions"] = "none"

    experiment, _, out = experiment_factory(configure)
    with pytest.raises(PreparationError, match="no project-instruction mapping"):
        prepare_experiment(experiment, out)
    assert all(not str(path).startswith(str(out)) for path in _worktree_paths(seed_repo))
    prepared_parent = out / "workspace-test"
    assert not prepared_parent.exists() or not any(prepared_parent.iterdir())


def test_cli_prepare_verify_and_confirmed_discard(experiment_factory, capsys):
    _, manifest, out = experiment_factory()
    assert main(["experiment", "prepare", str(manifest), "--out", str(out)]) == 0
    prepare_output = capsys.readouterr().out
    assert "agents launched: 0" in prepare_output
    prepared_roots = list((out / "workspace-test").iterdir())
    assert len(prepared_roots) == 1
    root = prepared_roots[0]

    assert main(["experiment", "verify", str(root)]) == 0
    assert "baseline unchanged" in capsys.readouterr().out
    assert main(["experiment", "discard", str(root)]) == 1
    assert "pass --yes" in capsys.readouterr().err
    assert main(["experiment", "discard", str(root), "--yes"]) == 0
    assert "cannot be recovered" in capsys.readouterr().out
    assert not root.exists()
