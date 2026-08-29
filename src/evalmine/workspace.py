"""Prepare, verify, and safely discard isolated v2 experiment workspaces.

Preparation is the mutation boundary before provider execution. It may write only to
the explicit artifact root and, for ``worktree`` mode, git's own worktree registry.
It never launches an agent or reads provider configuration.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from . import __version__
from .experiment import Arm, Experiment, ExperimentError, ExperimentPlan, build_plan
from .suite import SECRET_PATTERNS

PREPARED_FORMAT = "evalmine-prepared-v1"
RETRY_FORMAT = "evalmine-failed-run-retry-v1"
MAX_BASELINE_TEXT_BYTES = 1024 * 1024

_RUNNER_COMMANDS = {
    "claude-code": "claude",
    "codex-cli": "codex",
    "gemini-cli": "gemini",
}

_INSTRUCTION_FILES = {
    "claude-code": ("CLAUDE.md",),
    "codex-cli": ("AGENTS.md", "AGENTS.override.md"),
    "gemini-cli": ("GEMINI.md",),
}

_CANONICAL_INSTRUCTION_FILE = {
    "claude-code": "CLAUDE.md",
    "codex-cli": "AGENTS.md",
    "gemini-cli": "GEMINI.md",
}


class PreparationError(ExperimentError):
    """The declared experiment cannot be materialized without contamination."""


@dataclass(frozen=True)
class SeedState:
    head_commit: str
    tracked_dirty: bool
    changed_tracked: tuple[str, ...]
    untracked: tuple[str, ...]
    included_untracked: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class SeedObservation:
    head_commit: str
    diff: bytes
    changed_tracked: tuple[str, ...]
    untracked: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class PreparedRun:
    run_key: str
    arm_id: str
    workspace: Path
    tree_hash: str


@dataclass(frozen=True)
class PreparedExperiment:
    root: Path
    plan: ExperimentPlan
    runs: tuple[PreparedRun, ...]
    baseline_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": PREPARED_FORMAT,
            "root": str(self.root),
            "plan_id": self.plan.plan_id,
            "experiment": self.plan.experiment.name,
            "baseline_fingerprint": self.baseline_fingerprint,
            "run_count": len(self.runs),
            "runs": [
                {
                    "run_key": run.run_key,
                    "arm": run.arm_id,
                    "workspace": str(run.workspace),
                    "tree_hash": run.tree_hash,
                }
                for run in self.runs
            ],
        }


@dataclass(frozen=True)
class PreparedRetry:
    root: Path
    plan_id: str
    parent_root: Path
    inherited_run_keys: tuple[str, ...]
    retry_run_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": RETRY_FORMAT,
            "root": str(self.root),
            "plan_id": self.plan_id,
            "parent_root": str(self.parent_root),
            "run_count": len(self.inherited_run_keys) + len(self.retry_run_keys),
            "inherited_run_count": len(self.inherited_run_keys),
            "retry_run_count": len(self.retry_run_keys),
            "inherited_run_keys": list(self.inherited_run_keys),
            "retry_run_keys": list(self.retry_run_keys),
            "provider_calls": False,
        }


def _git(
    repo: Path,
    args: list[str],
    *,
    text: bool = False,
    timeout: int = 30,
) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=text,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreparationError(f"git {' '.join(args)} failed in {repo} ({exc})") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip() if text else result.stderr.decode("utf-8", "replace").strip()
        detail = stderr or "exit status " + str(result.returncode)
        raise PreparationError(f"git {' '.join(args)} failed in {repo} ({detail})")
    return result


def _nul_paths(data: bytes) -> tuple[str, ...]:
    try:
        return tuple(part.decode("utf-8") for part in data.split(b"\0") if part)
    except UnicodeDecodeError as exc:
        raise PreparationError(f"git returned a path that is not valid UTF-8 ({exc})") from exc


def _path_content_hash(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(f"mode:{stat.S_IMODE(path.lstat().st_mode):04o}\0".encode("ascii"))
    if path.is_symlink():
        digest.update(b"symlink\0")
        digest.update(os.fsencode(os.readlink(path)))
    elif path.is_file():
        digest.update(b"file\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        raise PreparationError(f"unsupported special or nested-repository path: {path}")
    return digest.hexdigest()


def _matches_allowlist(path: str, patterns: tuple[str, ...]) -> bool:
    candidate = PurePosixPath(path)
    return any(
        fnmatch.fnmatchcase(path, pattern) or candidate.match(pattern) for pattern in patterns
    )


def _observe_seed(repo: Path) -> SeedObservation:
    head = _git(repo, ["rev-parse", "--verify", "HEAD^{commit}"], text=True).stdout.strip()
    diff = _git(repo, ["diff", "--binary", "--no-ext-diff", "HEAD", "--"]).stdout
    changed_tracked = _nul_paths(
        _git(repo, ["diff", "--name-only", "--no-renames", "-z", "HEAD", "--"]).stdout
    )
    untracked = _nul_paths(_git(repo, ["ls-files", "--others", "--exclude-standard", "-z"]).stdout)

    fingerprint = hashlib.sha256()
    fingerprint.update(head.encode("ascii"))
    fingerprint.update(b"\0tracked-diff\0")
    fingerprint.update(diff)
    for relative in untracked:
        fingerprint.update(b"\0untracked\0")
        fingerprint.update(relative.encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(_path_content_hash(repo / relative).encode("ascii"))
    return SeedObservation(
        head_commit=head,
        diff=diff,
        changed_tracked=changed_tracked,
        untracked=untracked,
        fingerprint=fingerprint.hexdigest(),
    )


def inspect_seed(experiment: Experiment) -> SeedState:
    """Inspect and enforce the manifest's tracked and untracked seed policy."""
    repo = experiment.seed.repo
    observed = _observe_seed(repo)
    tracked_dirty = bool(observed.diff)
    if experiment.seed.dirty == "reject" and tracked_dirty:
        names = ", ".join(observed.changed_tracked[:5]) or "tracked files"
        extra = " …" if len(observed.changed_tracked) > 5 else ""
        raise PreparationError(
            f"seed policy dirty=reject, but {repo} has changes in {names}{extra}"
        )
    if experiment.seed.dirty == "capture" and observed.head_commit != experiment.seed.commit:
        raise PreparationError(
            "dirty=capture requires seed.ref to resolve to the source worktree's HEAD; "
            f"manifest resolved {experiment.seed.commit[:12]}, "
            f"HEAD is {observed.head_commit[:12]}"
        )

    if experiment.seed.untracked == "deny" and observed.untracked:
        names = ", ".join(observed.untracked[:5])
        extra = " …" if len(observed.untracked) > 5 else ""
        raise PreparationError(f"seed policy untracked=deny, but {repo} has {names}{extra}")
    if experiment.seed.untracked == "include":
        included_untracked = observed.untracked
    elif experiment.seed.untracked == "allowlisted":
        patterns = experiment.seed.untracked_allowlist
        unexpected = tuple(
            path for path in observed.untracked if not _matches_allowlist(path, patterns)
        )
        if unexpected:
            names = ", ".join(unexpected[:5])
            extra = " …" if len(unexpected) > 5 else ""
            raise PreparationError(
                f"untracked files are outside untracked_allowlist: {names}{extra}"
            )
        included_untracked = observed.untracked
    else:
        included_untracked = ()

    return SeedState(
        head_commit=observed.head_commit,
        tracked_dirty=tracked_dirty,
        changed_tracked=observed.changed_tracked,
        untracked=observed.untracked,
        included_untracked=included_untracked,
        fingerprint=observed.fingerprint,
    )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _artifact_root(experiment: Experiment, plan: ExperimentPlan, out_dir: str | Path) -> Path:
    base = Path(out_dir).resolve()
    root = (base / experiment.name / plan.plan_id).resolve()
    repo = experiment.seed.repo.resolve()
    if root == repo or _is_relative_to(root, repo) or _is_relative_to(repo, root):
        raise PreparationError(
            f"artifact root {root} overlaps seed repository {repo}; choose --out outside it"
        )
    return root


def _make_writable_and_retry(function, path, _exc_info) -> None:
    os.chmod(path, stat.S_IWUSR | stat.S_IRUSR)
    function(path)


def _remove_tree(path: Path) -> None:
    if path.exists() or path.is_symlink():
        shutil.rmtree(path, onerror=_make_writable_and_retry)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise PreparationError(f"refusing to overwrite evidence file {path}") from exc
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _evidence_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "prepared.json":
            continue
        relative = path.relative_to(root)
        if "workspace" in relative.parts or relative.parts[0] in {
            "execution",
            "validation",
            "report",
            "judging",
            "decision",
            "workflow",
        }:
            continue
        hashes[relative.as_posix()] = _path_content_hash(path)
    return hashes


def _execution_evidence_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _path_content_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "execution.json"
    }


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"cannot read {label} {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{label} {path} is not a JSON object")
    return value


def _verify_execution_snapshot(
    execution_root: Path,
    *,
    expected_prepared_root: Path,
) -> dict[str, Any]:
    marker = _read_json_object(
        execution_root / "execution.json", label="execution marker"
    )
    if marker.get("format") != "evalmine-execution-v1":
        raise PreparationError(f"{execution_root} has an unknown execution format")
    if marker.get("prepared_root") != str(expected_prepared_root):
        raise PreparationError("parent execution points at a different prepared experiment")
    expected = marker.get("evidence_sha256")
    if not isinstance(expected, dict) or not expected:
        raise PreparationError("parent execution marker has no evidence hashes")
    actual = _execution_evidence_hashes(execution_root)
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected))
        if not changed:
            changed = sorted(path for path in actual if actual[path] != expected.get(path))
        names = ", ".join(changed[:5]) or "execution evidence files"
        raise PreparationError(f"parent execution evidence changed: {names}")
    return marker


def _retry_metadata(root: Path, marker: dict[str, Any]) -> dict[str, Any] | None:
    declared = marker.get("retry")
    if declared is None:
        return None
    if declared != "retry.json":
        raise PreparationError("prepared retry marker must reference retry.json")
    retry = _read_json_object(root / declared, label="retry provenance")
    if retry.get("format") != RETRY_FORMAT:
        raise PreparationError(f"{root / declared} has an unknown retry format")
    return retry


def _verify_retry_provenance(
    root: Path,
    marker: dict[str, Any],
    run_keys: list[str],
) -> dict[str, Any] | None:
    retry = _retry_metadata(root, marker)
    if retry is None:
        return None
    inherited = retry.get("inherited_run_keys")
    retried = retry.get("retry_run_keys")
    if (
        not isinstance(inherited, list)
        or not inherited
        or not isinstance(retried, list)
        or not retried
        or any(not isinstance(run_key, str) for run_key in inherited + retried)
        or len(set(inherited)) != len(inherited)
        or len(set(retried)) != len(retried)
    ):
        raise PreparationError("retry provenance has invalid inherited or retry run keys")
    if set(inherited) & set(retried) or set(inherited + retried) != set(run_keys):
        raise PreparationError("retry provenance does not partition the prepared run keys")
    parent_root = Path(str(retry.get("parent_root", ""))).resolve()
    snapshot = root / "provenance" / "parent-execution"
    parent_execution = _verify_execution_snapshot(
        snapshot,
        expected_prepared_root=parent_root,
    )
    if parent_execution.get("plan_id") != retry.get("parent_plan_id"):
        raise PreparationError("retry provenance parent plan id does not match its snapshot")
    expected_marker_hash = retry.get("parent_execution_marker_sha256")
    actual_marker_hash = _path_content_hash(snapshot / "execution.json")
    if expected_marker_hash != actual_marker_hash:
        raise PreparationError("retry provenance parent execution marker hash changed")
    statuses = parent_execution.get("run_status")
    if not isinstance(statuses, dict):
        raise PreparationError("retry provenance parent execution has no run statuses")
    if any(statuses.get(run_key) != "succeeded" for run_key in inherited):
        raise PreparationError("retry provenance can inherit only succeeded parent runs")
    if any(statuses.get(run_key) == "succeeded" for run_key in retried):
        raise PreparationError("retry provenance cannot retry a succeeded parent run")
    plan = _read_json_object(root / "plan.json", label="retry plan")
    if plan.get("plan_id") != marker.get("plan_id"):
        raise PreparationError("retry plan id does not match its prepared marker")
    return retry


def _safe_tar_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if target != root and not _is_relative_to(target, root):
        raise PreparationError(f"git archive contains unsafe path {name!r}")
    return target


def _extract_commit(repo: Path, commit: str, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=False)
    try:
        process = subprocess.Popen(
            ["git", "-C", str(repo), "archive", "--format=tar", commit],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PreparationError(f"cannot start git archive in {repo} ({exc})") from exc
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                _safe_tar_target(workspace, member.name)
                if member.islnk():
                    _safe_tar_target(workspace, member.linkname)
                archive.extract(member, workspace)
    except (tarfile.TarError, OSError) as exc:
        process.kill()
        process.wait()
        raise PreparationError(f"cannot extract seed commit {commit[:12]} ({exc})") from exc
    stderr = process.stderr.read().decode("utf-8", "replace").strip()
    if process.wait() != 0:
        raise PreparationError(
            f"git archive failed for seed commit {commit[:12]} ({stderr or 'unknown error'})"
        )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        _remove_tree(path)


def _copy_seed_path(source: Path, destination: Path) -> None:
    _remove_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        destination.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
    elif source.is_file():
        shutil.copy2(source, destination, follow_symlinks=False)
    else:
        raise PreparationError(f"unsupported special or submodule path: {source}")


def _overlay_destination(workspace: Path, relative: str) -> Path:
    destination = workspace / relative
    resolved_parent = destination.parent.resolve()
    if not _is_relative_to(resolved_parent, workspace.resolve()):
        raise PreparationError(f"seed overlay path escapes its workspace: {relative!r}")
    return destination


def _apply_seed_overlay(experiment: Experiment, state: SeedState, workspace: Path) -> None:
    repo = experiment.seed.repo
    if experiment.seed.dirty == "capture":
        for relative in state.changed_tracked:
            source = repo / relative
            destination = _overlay_destination(workspace, relative)
            if source.exists() or source.is_symlink():
                _copy_seed_path(source, destination)
            else:
                _remove_path(destination)
    for relative in state.included_untracked:
        _copy_seed_path(repo / relative, _overlay_destination(workspace, relative))


def _instruction_paths(workspace: Path, runner: str) -> list[Path]:
    names = _INSTRUCTION_FILES.get(runner, ())
    found: list[Path] = []
    for name in names:
        found.extend(path for path in workspace.rglob(name) if ".git" not in path.parts)
    return sorted(set(found))


def _apply_instruction_treatment(workspace: Path, arm: Arm) -> None:
    treatment = arm.configuration.instructions
    if treatment == "inherit":
        return
    canonical = _CANONICAL_INSTRUCTION_FILE.get(arm.runner)
    if canonical is None:
        raise PreparationError(
            f"runner {arm.runner!r} has no project-instruction mapping; "
            f"cannot apply instructions={treatment}"
        )
    for path in _instruction_paths(workspace, arm.runner):
        if path.is_dir() and not path.is_symlink():
            raise PreparationError(f"instruction path is a directory, refusing removal: {path}")
        path.unlink()
    if treatment == "files":
        sections = []
        for source in arm.configuration.instruction_files:
            sections.append(
                f"<!-- evalmine input: {source.declared} sha256:{source.hash} -->\n\n"
                f"{source.content.rstrip()}\n"
            )
        target = workspace / canonical
        target.write_text("\n".join(sections), encoding="utf-8")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        real_dirs: list[str] = []
        for directory in sorted(dirs):
            if directory == ".git":
                continue
            path = current_path / directory
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                digest.update(relative.encode("utf-8"))
                digest.update(b"\0")
                digest.update(_path_content_hash(path).encode("ascii"))
                digest.update(b"\0")
            else:
                real_dirs.append(directory)
        dirs[:] = real_dirs
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if relative == ".git" or relative.startswith(".git/"):
                continue
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(_path_content_hash(path).encode("ascii"))
            digest.update(b"\0")
    return digest.hexdigest()


def _write_baseline_blob(root: Path, digest: str, content: bytes) -> dict[str, Any]:
    if len(content) > MAX_BASELINE_TEXT_BYTES or b"\0" in content:
        return {}
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return {}
    redacted = text
    for _label, pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_CREDENTIAL]", redacted)
    blob_content = redacted.encode("utf-8")
    blob_digest = hashlib.sha256(blob_content).hexdigest()
    relative = Path("baseline") / "blobs" / f"{digest}.bin"
    target = root / relative
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != blob_digest:
            raise PreparationError(f"baseline blob collision at {target}")
        return {
            "blob": relative.as_posix(),
            "blob_sha256": blob_digest,
            "blob_redacted": blob_content != content,
        }
    _write_once(target, blob_content)
    return {
        "blob": relative.as_posix(),
        "blob_sha256": blob_digest,
        "blob_redacted": blob_content != content,
    }


def _capture_workspace_baseline(workspace: Path, root: Path) -> dict[str, Any]:
    """Snapshot the treated pre-agent tree without relying on workspace Git metadata."""
    entries: list[dict[str, Any]] = []
    for current, dirs, files in os.walk(workspace, topdown=True, followlinks=False):
        current_path = Path(current)
        real_dirs: list[str] = []
        for directory in sorted(dirs):
            if directory == ".git":
                continue
            path = current_path / directory
            if path.is_symlink():
                entries.append(
                    {
                        "path": path.relative_to(workspace).as_posix(),
                        "type": "symlink",
                        "mode": stat.S_IMODE(path.lstat().st_mode),
                        "target": os.readlink(path),
                    }
                )
            else:
                real_dirs.append(directory)
        dirs[:] = real_dirs
        for name in sorted(files):
            path = current_path / name
            relative = path.relative_to(workspace).as_posix()
            if relative == ".git" or relative.startswith(".git/"):
                continue
            mode = stat.S_IMODE(path.lstat().st_mode)
            if path.is_symlink():
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "mode": mode,
                        "target": os.readlink(path),
                    }
                )
                continue
            if not path.is_file():
                raise PreparationError(f"unsupported special path in workspace baseline: {path}")
            content = path.read_bytes()
            digest = hashlib.sha256(content).hexdigest()
            entries.append(
                {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "size": len(content),
                    "sha256": digest,
                    **_write_baseline_blob(root, digest, content),
                }
            )
    return {
        "format": "evalmine-workspace-baseline-v1",
        "tree_hash": _tree_hash(workspace),
        "entry_count": len(entries),
        "entries": sorted(entries, key=lambda entry: entry["path"]),
    }


def _snapshot_inputs(experiment: Experiment, root: Path) -> None:
    entries: list[dict[str, Any]] = []
    blobs: dict[str, bytes] = {}
    for episode in experiment.episodes:
        for index, turn in enumerate(episode.turns, start=1):
            content = turn.prompt.encode("utf-8")
            digest = hashlib.sha256(content).hexdigest()
            blobs[digest] = content
            entries.append(
                {
                    "kind": "prompt",
                    "logical": f"episode/{episode.id}/turn/{index}",
                    "source": str(turn.prompt_file) if turn.prompt_file else "manifest:inline",
                    "sha256": digest,
                    "blob": f"blobs/{digest}.txt",
                }
            )
    for arm in experiment.arms:
        for index, source in enumerate(arm.configuration.instruction_files, start=1):
            content = source.content.encode("utf-8")
            blobs[source.hash] = content
            entries.append(
                {
                    "kind": "instruction",
                    "logical": f"arm/{arm.id}/instruction/{index}",
                    "source": str(source.path),
                    "declared": source.declared,
                    "sha256": source.hash,
                    "blob": f"blobs/{source.hash}.txt",
                }
            )
    for digest, content in blobs.items():
        _write_once(root / "inputs" / "blobs" / f"{digest}.txt", content)
    _write_once(
        root / "inputs" / "index.json",
        _json_bytes({"input_hash": experiment.input_hash, "entries": entries}),
    )


def _environment_inventory(experiment: Experiment) -> dict[str, Any]:
    runner_commands = sorted(
        {_RUNNER_COMMANDS[arm.runner] for arm in experiment.arms if arm.runner in _RUNNER_COMMANDS}
    )
    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "evalmine_version": __version__,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "runner_executables": {
            command: {"available": shutil.which(command) is not None, "path": shutil.which(command)}
            for command in runner_commands
        },
        "credentials_captured": False,
        "provider_configuration_read": False,
    }


def _arm_by_id(experiment: Experiment) -> dict[str, Arm]:
    return {arm.id: arm for arm in experiment.arms}


def _treatment_dict(arm: Arm) -> dict[str, Any]:
    return {
        "instructions": arm.configuration.instructions,
        "instruction_inputs": [
            {"declared": source.declared, "sha256": source.hash}
            for source in arm.configuration.instruction_files
        ],
        "plugins": arm.configuration.plugins,
        "plugin_allowlist": list(arm.configuration.plugin_allowlist),
        "plugin_directories": list(arm.configuration.plugin_directories),
        "settings": dict(arm.configuration.settings),
        "arguments": list(arm.configuration.arguments),
        "plugin_enforcement": "runner-preflight",
    }


def _cleanup_failed(root: Path, repo: Path, worktrees: list[Path]) -> None:
    for workspace in reversed(worktrees):
        result = subprocess.run(
            ["git", "-C", str(repo), "worktree", "remove", "--force", str(workspace)],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            _remove_tree(workspace)
    _remove_tree(root)
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "prune"],
        check=False,
        capture_output=True,
    )


def prepare_experiment(experiment: Experiment, out_dir: str | Path) -> PreparedExperiment:
    """Materialize every planned run without launching any provider agent."""
    plan = build_plan(experiment)
    state = inspect_seed(experiment)
    root = _artifact_root(experiment, plan, out_dir)
    if root.exists() or root.is_symlink():
        raise PreparationError(
            f"prepared experiment already exists at {root}; evidence is never overwritten"
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    worktrees: list[Path] = []
    prepared_runs: list[PreparedRun] = []
    arm_by_id = _arm_by_id(experiment)
    events: list[dict[str, Any]] = [
        {
            "event": "preparation_started",
            "at": datetime.now(timezone.utc).isoformat(),
            "plan_id": plan.plan_id,
        }
    ]
    try:
        _write_once(root / "manifest.yaml", experiment.manifest_bytes)
        _write_once(root / "plan.json", _json_bytes(plan.as_dict()))
        _write_once(root / "environment.json", _json_bytes(_environment_inventory(experiment)))
        _snapshot_inputs(experiment, root)

        for planned in plan.runs:
            arm = arm_by_id[planned.arm_id]
            run_dir = root / "runs" / planned.run_key
            run_dir.mkdir(parents=True)
            workspace = run_dir / "workspace"
            if experiment.isolation.workspace == "copy":
                _extract_commit(experiment.seed.repo, experiment.seed.commit, workspace)
            else:
                _git(
                    experiment.seed.repo,
                    ["worktree", "add", "--detach", str(workspace), experiment.seed.commit],
                    text=True,
                    timeout=120,
                )
                worktrees.append(workspace)
            _apply_seed_overlay(experiment, state, workspace)
            _apply_instruction_treatment(workspace, arm)
            tree_hash = _tree_hash(workspace)
            baseline = _capture_workspace_baseline(workspace, root)
            _write_once(run_dir / "treatment.json", _json_bytes(_treatment_dict(arm)))
            _write_once(run_dir / "baseline.json", _json_bytes(baseline))
            _write_once(
                run_dir / "run.json",
                _json_bytes(
                    {
                        **planned.as_dict(),
                        "workspace": str(workspace.resolve()),
                        "initial_tree_hash": tree_hash,
                        "seed_commit": experiment.seed.commit,
                        "input_hash": experiment.input_hash,
                    }
                ),
            )
            prepared_runs.append(
                PreparedRun(
                    run_key=planned.run_key,
                    arm_id=planned.arm_id,
                    workspace=workspace.resolve(),
                    tree_hash=tree_hash,
                )
            )
            events.append(
                {
                    "event": "workspace_prepared",
                    "sequence": planned.sequence,
                    "run_key": planned.run_key,
                    "arm": planned.arm_id,
                    "tree_hash": tree_hash,
                }
            )

        after = inspect_seed(experiment)
        if after.fingerprint != state.fingerprint:
            raise PreparationError(
                "seed repository changed while workspaces were being prepared; "
                "the incomplete preparation was discarded"
            )
        events.append(
            {
                "event": "preparation_completed",
                "at": datetime.now(timezone.utc).isoformat(),
                "run_count": len(prepared_runs),
            }
        )
        _write_once(
            root / "preparation.jsonl",
            b"".join(_json_bytes(event).replace(b"\n", b"") + b"\n" for event in events),
        )
        marker = {
            "format": PREPARED_FORMAT,
            "root": str(root),
            "experiment": experiment.name,
            "plan_id": plan.plan_id,
            "input_hash": experiment.input_hash,
            "seed_repo": str(experiment.seed.repo),
            "seed_commit": experiment.seed.commit,
            "baseline_fingerprint": state.fingerprint,
            "workspace_mode": experiment.isolation.workspace,
            "run_keys": [run.run_key for run in prepared_runs],
            "evidence_sha256": _evidence_hashes(root),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_once(root / "prepared.json", _json_bytes(marker))
    except Exception:
        _cleanup_failed(root, experiment.seed.repo, worktrees)
        raise

    return PreparedExperiment(
        root=root,
        plan=plan,
        runs=tuple(prepared_runs),
        baseline_fingerprint=state.fingerprint,
    )


def _read_marker(root: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(root)
    if path.is_symlink():
        raise PreparationError(f"prepared root must not be a symlink: {path}")
    resolved = path.resolve()
    marker_path = resolved / "prepared.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{resolved} is not a valid prepared experiment ({exc})") from exc
    if marker.get("format") != PREPARED_FORMAT:
        raise PreparationError(f"{marker_path} has an unknown preparation format")
    if marker.get("root") != str(resolved):
        raise PreparationError(
            f"prepared marker path mismatch: marker says {marker.get('root')!r}, got {resolved}"
        )
    seed_repo = Path(marker.get("seed_repo", "")).resolve()
    if resolved == seed_repo or resolved == Path(resolved.anchor):
        raise PreparationError(f"refusing unsafe prepared root {resolved}")
    return resolved, marker


def verify_prepared(root: str | Path) -> dict[str, Any]:
    """Verify preparation containment and that the baseline still matches."""
    # Imported artifacts are a separate, zero-generation evidence envelope. They
    # intentionally have no seed repository or workspaces, but report/judge/decide
    # consume them through the same verification boundary.
    from .external import is_external_import, verify_external_import

    if is_external_import(root):
        result = verify_external_import(root)
        return {
            **result,
            "baseline_unchanged": None,
            "workspace_mode": "external-artifacts",
            "retry": None,
        }
    resolved, marker = _read_marker(root)
    seed_repo = Path(marker["seed_repo"])
    run_keys = marker.get("run_keys")
    if not isinstance(run_keys, list) or not run_keys:
        raise PreparationError("prepared marker has no run keys")
    workspace_paths: list[Path] = []
    runs_root = (resolved / "runs").resolve()
    for run_key in run_keys:
        expected_parent = (runs_root / run_key).resolve()
        workspace = (expected_parent / "workspace").resolve()
        if (
            not _is_relative_to(expected_parent, runs_root)
            or not _is_relative_to(workspace, expected_parent)
            or not workspace.is_dir()
        ):
            raise PreparationError(f"missing or escaped workspace for run {run_key!r}")
        workspace_paths.append(workspace)
    if len(set(workspace_paths)) != len(workspace_paths):
        raise PreparationError("multiple runs resolve to the same workspace")

    expected_hashes = marker.get("evidence_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise PreparationError("prepared marker has no evidence hashes")
    actual_hashes = _evidence_hashes(resolved)
    if actual_hashes != expected_hashes:
        changed = sorted(set(actual_hashes) ^ set(expected_hashes))
        if not changed:
            changed = sorted(
                path for path in actual_hashes if actual_hashes[path] != expected_hashes.get(path)
            )
        names = ", ".join(changed[:5]) or "evidence files"
        raise PreparationError(f"prepared evidence changed after creation: {names}")

    retry = _verify_retry_provenance(resolved, marker, run_keys)

    current = _observe_seed(seed_repo)
    baseline_unchanged = current.fingerprint == marker["baseline_fingerprint"]
    if not baseline_unchanged:
        raise PreparationError(
            f"baseline repository {seed_repo} changed after preparation; "
            "comparative execution must not proceed"
        )
    return {
        "ok": True,
        "format": PREPARED_FORMAT,
        "root": str(resolved),
        "plan_id": marker["plan_id"],
        "run_count": len(run_keys),
        "baseline_unchanged": True,
        "workspace_mode": marker["workspace_mode"],
        "retry": (
            {
                "inherited_run_count": len(retry["inherited_run_keys"]),
                "retry_run_count": len(retry["retry_run_keys"]),
            }
            if retry is not None
            else None
        ),
    }


def prepare_failed_run_retry(
    parent: str | Path,
    out_dir: str | Path,
) -> PreparedRetry:
    """Create a self-contained envelope that executes only failed parent runs."""
    verify_prepared(parent)
    parent_root, parent_marker = _read_marker(parent)
    if parent_marker.get("workspace_mode") != "copy":
        raise PreparationError("failed-run retry currently requires workspace=copy")
    parent_plan = _read_json_object(parent_root / "plan.json", label="parent plan")
    if parent_plan.get("isolation", {}).get("external_writes") != "deny":
        raise PreparationError("failed-run retry currently requires external_writes=deny")
    parent_execution_root = parent_root / "execution"
    parent_execution = _verify_execution_snapshot(
        parent_execution_root,
        expected_prepared_root=parent_root,
    )
    run_keys = parent_marker.get("run_keys", [])
    statuses = parent_execution.get("run_status")
    if not isinstance(run_keys, list) or not isinstance(statuses, dict):
        raise PreparationError("parent execution has invalid run metadata")
    inherited = tuple(run_key for run_key in run_keys if statuses.get(run_key) == "succeeded")
    retried = tuple(run_key for run_key in run_keys if statuses.get(run_key) != "succeeded")
    if not inherited:
        raise PreparationError("parent execution has no succeeded runs to inherit")
    if not retried:
        raise PreparationError("parent execution has no failed runs to retry")

    parent_execution_marker_hash = _path_content_hash(
        parent_execution_root / "execution.json"
    )
    retry_digest = hashlib.sha256(
        json.dumps(
            {
                "parent_plan_id": parent_marker["plan_id"],
                "parent_execution_marker_sha256": parent_execution_marker_hash,
                "retry_run_keys": list(retried),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    experiment_name = str(parent_marker["experiment"])
    plan_id = f"{experiment_name}-retry-{retry_digest}"
    base = Path(out_dir).resolve()
    root = (base / experiment_name / plan_id).resolve()
    seed_repo = Path(parent_marker["seed_repo"]).resolve()
    if root == seed_repo or _is_relative_to(root, seed_repo) or _is_relative_to(seed_repo, root):
        raise PreparationError(
            f"retry artifact root {root} overlaps seed repository {seed_repo}; "
            "choose --out outside it"
        )
    if root.exists() or root.is_symlink():
        raise PreparationError(
            f"prepared retry already exists at {root}; evidence is never overwritten"
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    events: list[dict[str, Any]] = [
        {
            "event": "retry_preparation_started",
            "at": datetime.now(timezone.utc).isoformat(),
            "plan_id": plan_id,
            "parent_plan_id": parent_marker["plan_id"],
        }
    ]
    try:
        for name in ("manifest.yaml", "environment.json"):
            _write_once(root / name, (parent_root / name).read_bytes())
        for name in ("inputs", "baseline"):
            shutil.copytree(parent_root / name, root / name, symlinks=True)

        plan = json.loads(json.dumps(parent_plan))
        plan["plan_id"] = plan_id
        plan["provenance"] = {
            "format": RETRY_FORMAT,
            "parent_plan_id": parent_marker["plan_id"],
            "inherited_run_keys": list(inherited),
            "retry_run_keys": list(retried),
        }
        schedule = plan.setdefault("schedule", {})
        schedule["max_parallel"] = min(
            max(1, int(schedule.get("max_parallel", 1))), len(retried)
        )
        _write_once(root / "plan.json", _json_bytes(plan))

        snapshot = root / "provenance" / "parent-execution"
        shutil.copytree(parent_execution_root, snapshot, symlinks=True)
        _verify_execution_snapshot(snapshot, expected_prepared_root=parent_root)

        for run_key in run_keys:
            parent_run_dir = parent_root / "runs" / run_key
            run = _read_json_object(parent_run_dir / "run.json", label="parent run")
            parent_workspace = (parent_run_dir / "workspace").resolve()
            if Path(str(run.get("workspace", ""))).resolve() != parent_workspace:
                raise PreparationError(f"parent run {run_key} points at another workspace")
            execution_run = _read_json_object(
                parent_execution_root / "runs" / run_key / "run.json",
                label="parent run execution",
            )
            if run_key in inherited:
                expected_tree_hash = execution_run.get("final_tree_hash")
                event_name = "run_inherited"
            else:
                expected_tree_hash = run.get("initial_tree_hash")
                event_name = "retry_workspace_prepared"
            if not isinstance(expected_tree_hash, str) or not expected_tree_hash:
                raise PreparationError(f"parent run {run_key} has no usable workspace hash")
            if _tree_hash(parent_workspace) != expected_tree_hash:
                detail = (
                    "changed after its successful execution"
                    if run_key in inherited
                    else "is not pristine after its failed execution"
                )
                raise PreparationError(f"parent workspace for {run_key} {detail}")

            run_dir = root / "runs" / run_key
            run_dir.mkdir(parents=True)
            workspace = run_dir / "workspace"
            shutil.copytree(
                parent_workspace,
                workspace,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git"),
            )
            if _tree_hash(workspace) != expected_tree_hash:
                raise PreparationError(f"copied workspace for {run_key} changed during retry prep")
            for name in ("baseline.json", "treatment.json"):
                _write_once(run_dir / name, (parent_run_dir / name).read_bytes())
            run["workspace"] = str(workspace.resolve())
            _write_once(run_dir / "run.json", _json_bytes(run))
            events.append(
                {
                    "event": event_name,
                    "run_key": run_key,
                    "tree_hash": expected_tree_hash,
                }
            )

        retry = {
            "format": RETRY_FORMAT,
            "parent_root": str(parent_root),
            "parent_plan_id": parent_marker["plan_id"],
            "parent_execution_status": parent_execution.get("status"),
            "parent_execution_marker_sha256": parent_execution_marker_hash,
            "inherited_run_keys": list(inherited),
            "retry_run_keys": list(retried),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_once(root / "retry.json", _json_bytes(retry))
        events.append(
            {
                "event": "retry_preparation_completed",
                "at": datetime.now(timezone.utc).isoformat(),
                "inherited_run_count": len(inherited),
                "retry_run_count": len(retried),
            }
        )
        _write_once(
            root / "preparation.jsonl",
            b"".join(
                json.dumps(event, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
                for event in events
            ),
        )
        marker = {
            "format": PREPARED_FORMAT,
            "root": str(root),
            "experiment": experiment_name,
            "plan_id": plan_id,
            "input_hash": parent_marker["input_hash"],
            "seed_repo": str(seed_repo),
            "seed_commit": parent_marker["seed_commit"],
            "baseline_fingerprint": parent_marker["baseline_fingerprint"],
            "workspace_mode": "copy",
            "run_keys": run_keys,
            "retry": "retry.json",
            "evidence_sha256": _evidence_hashes(root),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
        }
        _write_once(root / "prepared.json", _json_bytes(marker))
        verify_prepared(root)
    except Exception:
        _remove_tree(root)
        raise

    return PreparedRetry(
        root=root,
        plan_id=plan_id,
        parent_root=parent_root,
        inherited_run_keys=inherited,
        retry_run_keys=retried,
    )


def discard_prepared(root: str | Path) -> dict[str, Any]:
    """Remove one exactly marked preparation, including git worktree registrations."""
    resolved, marker = _read_marker(root)
    seed_repo = Path(marker["seed_repo"])
    run_keys = marker.get("run_keys", [])
    if marker.get("workspace_mode") == "worktree":
        runs_root = (resolved / "runs").resolve()
        for run_key in reversed(run_keys):
            expected_parent = (runs_root / run_key).resolve()
            workspace = (expected_parent / "workspace").resolve()
            if not _is_relative_to(expected_parent, runs_root) or not _is_relative_to(
                workspace, expected_parent
            ):
                raise PreparationError(f"unsafe workspace path for run {run_key!r}")
            _git(
                seed_repo,
                ["worktree", "remove", "--force", str(workspace)],
                text=True,
                timeout=120,
            )
    _remove_tree(resolved)
    return {"discarded": True, "root": str(resolved), "run_count": len(run_keys)}
