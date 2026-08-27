"""Deterministic post-execution validators for v2 agent experiments.

Validation is a separate evidence boundary. Built-in validators inspect an immutable
snapshot of the agent's final workspace. Declared command validators run only after
that snapshot has been captured and require an explicit operator acknowledgement.
"""

from __future__ import annotations

import difflib
import fnmatch
import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Any

from .experiment import ExperimentError
from .runner import ProcessDriver, RunnerError, _redact_secrets, verify_execution
from .workspace import PreparationError, _tree_hash, verify_prepared

VALIDATION_FORMAT = "evalmine-validation-v1"
DEFAULT_COMMAND_TIMEOUT = 300
MAX_DIFF_FILE_BYTES = 1024 * 1024
MAX_PATCH_BYTES = 2 * 1024 * 1024
MAX_COMMAND_OUTPUT_BYTES = 1024 * 1024

_SECRET_ENV_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_write_lock = Lock()


class ValidationError(ExperimentError):
    """Validation evidence cannot be produced or verified safely."""


class ValidationRefused(ValidationError):
    """The requested validation requires an explicit operator acknowledgement."""


@dataclass(frozen=True)
class ValidationResult:
    root: Path
    verdict: str
    run_count: int
    passed: int
    failed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "verdict": self.verdict,
            "run_count": self.run_count,
            "passed": self.passed,
            "failed": self.failed,
            "provider_runners_launched": False,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ValidationError(f"refusing to overwrite validation evidence file {path}") from exc
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _write_evidence(path: Path, content: bytes) -> None:
    with _write_lock:
        _write_once(path, content)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read validation input {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"validation input {path} is not a JSON object")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validation_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "validation.json"
    }


def _load_roots(root: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    resolved = Path(root).resolve()
    prepared = _read_json(resolved / "prepared.json")
    plan = _read_json(resolved / "plan.json")
    if prepared.get("root") != str(resolved):
        raise ValidationError("prepared marker points at a different experiment root")
    return resolved, prepared, plan


def _current_snapshot(workspace: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    for current, dirs, files in os.walk(workspace, topdown=True, followlinks=False):
        current_path = Path(current)
        real_dirs: list[str] = []
        for directory in sorted(dirs):
            if directory == ".git":
                continue
            path = current_path / directory
            relative = path.relative_to(workspace).as_posix()
            if path.is_symlink():
                entries[relative] = {
                    "path": relative,
                    "type": "symlink",
                    "mode": stat.S_IMODE(path.lstat().st_mode),
                    "target": os.readlink(path),
                }
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
                entries[relative] = {
                    "path": relative,
                    "type": "symlink",
                    "mode": mode,
                    "target": os.readlink(path),
                }
            elif path.is_file():
                entries[relative] = {
                    "path": relative,
                    "type": "file",
                    "mode": mode,
                    "size": path.stat().st_size,
                    "sha256": _file_hash(path),
                }
            else:
                raise ValidationError(f"unsupported special path in workspace: {path}")
    return entries


def _load_baseline(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    baseline = _read_json(path)
    if baseline.get("format") != "evalmine-workspace-baseline-v1":
        raise ValidationError(f"{path} has an unknown workspace baseline format")
    entries = baseline.get("entries")
    if not isinstance(entries, list):
        raise ValidationError(f"{path} has no baseline entries")
    by_path: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValidationError(f"{path} contains an invalid baseline entry")
        by_path[entry["path"]] = entry
    if len(by_path) != len(entries):
        raise ValidationError(f"{path} contains duplicate baseline paths")
    return baseline, by_path


def _entry_changed(before: dict[str, Any], after: dict[str, Any]) -> bool:
    keys = {"type", "mode", "target", "sha256"}
    return any(before.get(key) != after.get(key) for key in keys)


def _changed_paths(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    changes: list[dict[str, str]] = []
    for relative in sorted(set(before) | set(after)):
        if relative not in before:
            changes.append({"path": relative, "change": "added"})
        elif relative not in after:
            changes.append({"path": relative, "change": "deleted"})
        elif _entry_changed(before[relative], after[relative]):
            changes.append({"path": relative, "change": "modified"})
    return changes


def _matches(path: str, patterns: list[str]) -> bool:
    candidate = PurePosixPath(path)
    return any(
        fnmatch.fnmatchcase(path, pattern) or candidate.match(pattern) for pattern in patterns
    )


def _filter_changes(changes: list[dict[str, str]], spec: dict[str, Any]) -> list[dict[str, str]]:
    include = list(spec.get("include", []))
    exclude = list(spec.get("exclude", []))
    return [
        change
        for change in changes
        if (not include or _matches(change["path"], include))
        and (not exclude or not _matches(change["path"], exclude))
    ]


def _safe_workspace_path(workspace: Path, declared: str) -> Path:
    pure = PurePosixPath(declared)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValidationError(f"validator path must stay inside the workspace: {declared!r}")
    target = (workspace / Path(*pure.parts)).resolve()
    if not _is_relative_to(target, workspace.resolve()):
        raise ValidationError(f"validator path escapes the workspace: {declared!r}")
    return target


def _read_baseline_content(root: Path, entry: dict[str, Any]) -> bytes | None:
    if entry.get("type") != "file" or not entry.get("blob"):
        return None
    blob = (root / str(entry.get("blob", ""))).resolve()
    if not _is_relative_to(blob, (root / "baseline" / "blobs").resolve()):
        raise ValidationError(f"baseline blob path escapes evidence root: {blob}")
    content = blob.read_bytes()
    if hashlib.sha256(content).hexdigest() != entry.get("blob_sha256"):
        raise ValidationError(f"baseline blob hash mismatch: {blob}")
    return content


def _text(content: bytes) -> str | None:
    if len(content) > MAX_DIFF_FILE_BYTES or b"\0" in content:
        return None
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _render_patch(
    root: Path,
    workspace: Path,
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    changes: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], bool]:
    chunks: list[str] = []
    omitted: list[dict[str, str]] = []
    truncated = False
    for change in changes:
        relative = change["path"]
        before = baseline.get(relative)
        after = current.get(relative)
        before_bytes = _read_baseline_content(root, before) if before else b""
        after_bytes = None
        if after and after.get("type") == "file":
            after_bytes = _safe_workspace_path(workspace, relative).read_bytes()
        elif after is None:
            after_bytes = b""
        before_text = _text(before_bytes) if before_bytes is not None else None
        after_text = _text(after_bytes) if after_bytes is not None else None
        if before_text is None or after_text is None:
            reason = (
                "binary-or-large-file"
                if (before is None or before.get("type") == "file")
                and (after is None or after.get("type") == "file")
                else "non-file"
            )
            omitted.append({"path": relative, "reason": reason})
            continue
        diff = "".join(
            difflib.unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=f"a/{relative}" if before else "/dev/null",
                tofile=f"b/{relative}" if after else "/dev/null",
            )
        )
        if not diff and before and after and before.get("mode") != after.get("mode"):
            diff = f"mode change {relative}: {before.get('mode'):04o} -> {after.get('mode'):04o}\n"
        prospective_size = sum(len(chunk.encode("utf-8")) for chunk in chunks) + len(
            diff.encode("utf-8")
        )
        if prospective_size > MAX_PATCH_BYTES:
            truncated = True
            omitted.append({"path": relative, "reason": "patch-size-limit"})
            continue
        chunks.append(diff)
    return _redact_secrets("".join(chunks)), omitted, truncated


def _repository_diff(
    *,
    validator_id: str,
    spec: dict[str, Any],
    root: Path,
    workspace: Path,
    baseline: dict[str, dict[str, Any]],
    current: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    all_changes = _changed_paths(baseline, current)
    changes = _filter_changes(all_changes, spec)
    included_paths = {change["path"] for change in changes}
    filtered_changes = [
        change for change in all_changes if change["path"] not in included_paths
    ]
    patch, omitted, truncated = _render_patch(root, workspace, baseline, current, changes)
    _write_evidence(output_dir / f"{validator_id}.patch", patch.encode("utf-8"))
    expect = spec["expect"]
    passed = expect == "any" or (expect == "changed") == bool(changes)
    max_changed = spec.get("max_changed_files")
    if max_changed is not None and len(changes) > int(max_changed):
        passed = False
    return {
        "id": validator_id,
        "type": "repository-diff",
        "status": "passed" if passed else "failed",
        "expect": expect,
        "changed_file_count": len(changes),
        "changes": changes,
        "filtered_changed_file_count": len(filtered_changes),
        "filtered_changes": filtered_changes,
        "patch": f"{validator_id}.patch",
        "patch_sha256": hashlib.sha256(patch.encode("utf-8")).hexdigest(),
        "patch_truncated": truncated,
        "patch_omissions": omitted,
        "max_changed_files": max_changed,
    }


def _required_files(
    validator_id: str,
    spec: dict[str, Any],
    current: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    non_empty = bool(spec.get("non_empty", True))
    checks = []
    for declared in spec["paths"]:
        entry = current.get(declared)
        present = entry is not None and entry.get("type") == "file"
        populated = present and (not non_empty or int(entry.get("size", 0)) > 0)
        checks.append({"path": declared, "present": present, "non_empty": populated})
    passed = all(check["present"] and check["non_empty"] for check in checks)
    return {
        "id": validator_id,
        "type": "required-files",
        "status": "passed" if passed else "failed",
        "checks": checks,
    }


def _required_sections(
    *,
    validator_id: str,
    spec: dict[str, Any],
    workspace: Path,
    current: dict[str, dict[str, Any]],
    execution_dir: Path,
    turns_planned: int,
) -> dict[str, Any]:
    source: str
    content = ""
    source_error: str | None = None
    if spec["target"] == "final-response":
        source = f"turn-{turns_planned:03d}.final.txt"
        try:
            content = (execution_dir / source).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            source_error = str(exc)
    else:
        source = spec["path"]
        entry = current.get(source)
        if entry is None or entry.get("type") != "file":
            source_error = "file is missing or not a regular file"
        else:
            try:
                content = _safe_workspace_path(workspace, source).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, ValidationError) as exc:
                source_error = str(exc)
    case_sensitive = bool(spec.get("case_sensitive", False))
    haystack = content if case_sensitive else content.casefold()
    sections = []
    for section in spec["sections"]:
        needle = section if case_sensitive else section.casefold()
        sections.append({"section": section, "present": needle in haystack})
    passed = source_error is None and all(section["present"] for section in sections)
    return {
        "id": validator_id,
        "type": "required-sections",
        "status": "passed" if passed else "failed",
        "target": spec["target"],
        "source": source,
        "source_error": source_error,
        "checks": sections,
    }


def _limited_output(value: str) -> tuple[str, bool]:
    redacted = _redact_secrets(value)
    encoded = redacted.encode("utf-8")
    if len(encoded) <= MAX_COMMAND_OUTPUT_BYTES:
        return redacted, False
    tail = encoded[-MAX_COMMAND_OUTPUT_BYTES:].decode("utf-8", "replace")
    return "[OUTPUT TRUNCATED; TAIL FOLLOWS]\n" + tail, True


def _command_env() -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not any(part in name.upper() for part in _SECRET_ENV_PARTS)
    }
    env.update({"CI": "1", "NO_COLOR": "1"})
    return env


def _command(
    *,
    validator_id: str,
    spec: dict[str, Any],
    workspace: Path,
    output_dir: Path,
    driver: ProcessDriver,
) -> dict[str, Any]:
    before = _current_snapshot(workspace)
    argv = list(spec["argv"])
    timeout = int(spec.get("timeout_s", DEFAULT_COMMAND_TIMEOUT))
    expected_exit = int(spec.get("expected_exit", 0))
    process = driver.run(argv, cwd=workspace, timeout=timeout, env=_command_env())
    stdout, stdout_truncated = _limited_output(process.stdout)
    stderr, stderr_truncated = _limited_output(process.stderr)
    _write_evidence(output_dir / f"{validator_id}.stdout.txt", stdout.encode("utf-8"))
    _write_evidence(output_dir / f"{validator_id}.stderr.txt", stderr.encode("utf-8"))
    after = _current_snapshot(workspace)
    mutations = _changed_paths(before, after)
    passed = not process.timed_out and process.returncode == expected_exit
    return {
        "id": validator_id,
        "type": "command",
        "status": "passed" if passed else "failed",
        "argv": argv,
        "timeout_s": timeout,
        "expected_exit": expected_exit,
        "exit_code": process.returncode,
        "timed_out": process.timed_out,
        "duration_ms": process.duration_ms,
        "stdout": f"{validator_id}.stdout.txt",
        "stderr": f"{validator_id}.stderr.txt",
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "workspace_mutations": mutations,
        "environment_values_captured": False,
    }


def _check_one(
    *,
    root: Path,
    validation_root: Path,
    run_key: str,
    declarations: dict[str, dict[str, Any]],
    driver: ProcessDriver,
) -> dict[str, Any]:
    prepared_dir = root / "runs" / run_key
    execution_dir = root / "execution" / "runs" / run_key
    run = _read_json(prepared_dir / "run.json")
    execution = _read_json(execution_dir / "run.json")
    _, baseline = _load_baseline(prepared_dir / "baseline.json")
    workspace = Path(run["workspace"]).resolve()
    anchored_tree_hash = execution.get("final_tree_hash")
    if not isinstance(anchored_tree_hash, str):
        raise ValidationError(f"execution for {run_key} has no final workspace tree hash")
    current_tree_hash = _tree_hash(workspace)
    if current_tree_hash != anchored_tree_hash:
        raise ValidationError(
            f"workspace for {run_key} changed after agent execution; refusing attribution"
        )
    current = _current_snapshot(workspace)
    output_dir = validation_root / "runs" / run_key
    output_dir.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    # Every built-in sees the same post-agent/pre-command snapshot. Commands run last,
    # preserving the diff even when a test tool creates caches or generated files.
    validator_ids = list(run.get("validators", []))
    for validator_id in validator_ids:
        spec = declarations[validator_id]
        kind = spec["type"]
        if kind == "command":
            continue
        if kind == "repository-diff":
            result = _repository_diff(
                validator_id=validator_id,
                spec=spec,
                root=root,
                workspace=workspace,
                baseline=baseline,
                current=current,
                output_dir=output_dir,
            )
        elif kind == "required-files":
            result = _required_files(validator_id, spec, current)
        elif kind == "required-sections":
            result = _required_sections(
                validator_id=validator_id,
                spec=spec,
                workspace=workspace,
                current=current,
                execution_dir=execution_dir,
                turns_planned=int(run["turns"]),
            )
        else:  # schema validation should make this unreachable
            raise ValidationError(f"unknown validator type {kind!r}")
        _write_evidence(output_dir / f"{validator_id}.json", _json_bytes(result))
        results.append(result)

    for validator_id in validator_ids:
        spec = declarations[validator_id]
        if spec["type"] != "command":
            continue
        result = _command(
            validator_id=validator_id,
            spec=spec,
            workspace=workspace,
            output_dir=output_dir,
            driver=driver,
        )
        _write_evidence(output_dir / f"{validator_id}.json", _json_bytes(result))
        results.append(result)

    execution_ok = execution.get("status") == "succeeded"
    passed = execution_ok and all(result["status"] == "passed" for result in results)
    summary = {
        "format": VALIDATION_FORMAT,
        "run_key": run_key,
        "arm": run["arm"],
        "episode": run["episode"],
        "repeat": run["repeat"],
        "execution_status": execution.get("status"),
        "verdict": "passed" if passed else "failed",
        "validator_count": len(results),
        "validators_passed": sum(result["status"] == "passed" for result in results),
        "validators_failed": sum(result["status"] == "failed" for result in results),
        "validator_order": [result["id"] for result in results],
    }
    _write_evidence(output_dir / "run.json", _json_bytes(summary))
    return summary


def check_experiment(
    root: str | Path,
    *,
    allow_validator_commands: bool = False,
    driver: ProcessDriver | None = None,
) -> ValidationResult:
    """Run declared objective validators over one completed execution envelope."""
    try:
        verify_prepared(root)
        verify_execution(root)
    except (PreparationError, RunnerError) as exc:
        raise ValidationError(str(exc)) from exc
    resolved, prepared, plan = _load_roots(root)
    declarations = plan.get("validators", {})
    if not isinstance(declarations, dict):
        raise ValidationError("plan has invalid validator declarations")
    run_keys = list(prepared.get("run_keys", []))
    used = {
        validator_id
        for run_key in run_keys
        for validator_id in _read_json(resolved / "runs" / run_key / "run.json").get(
            "validators", []
        )
    }
    unknown = sorted(used - set(declarations))
    if unknown:
        raise ValidationError(
            "prepared runs reference undeclared validators: " + ", ".join(unknown)
        )
    command_ids = sorted(
        validator_id for validator_id in used if declarations[validator_id].get("type") == "command"
    )
    if command_ids and not allow_validator_commands:
        raise ValidationRefused(
            "command validators execute manifest-declared local processes; pass "
            "--allow-validator-commands (required by: " + ", ".join(command_ids) + ")"
        )
    validation_root = resolved / "validation"
    if validation_root.exists() or validation_root.is_symlink():
        raise ValidationRefused(
            f"validation evidence already exists at {validation_root}; it is never overwritten"
        )
    validation_root.mkdir()
    driver = driver or ProcessDriver()
    max_parallel = max(1, int(plan.get("schedule", {}).get("max_parallel", 1)))
    ledger: list[dict[str, Any]] = [
        {
            "event": "validation_started",
            "at": _now(),
            "run_count": len(run_keys),
            "max_parallel": max_parallel,
        }
    ]
    summaries: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max_parallel) as pool:
        futures = {
            pool.submit(
                _check_one,
                root=resolved,
                validation_root=validation_root,
                run_key=run_key,
                declarations=declarations,
                driver=driver,
            ): run_key
            for run_key in run_keys
        }
        for future in as_completed(futures):
            run_key = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                summary = {
                    "format": VALIDATION_FORMAT,
                    "run_key": run_key,
                    "verdict": "failed",
                    "error": f"validator worker failed: {exc}",
                }
                output_dir = validation_root / "runs" / run_key
                output_dir.mkdir(parents=True, exist_ok=True)
                if not (output_dir / "run.json").exists():
                    _write_evidence(output_dir / "run.json", _json_bytes(summary))
            summaries[run_key] = summary
            ledger.append(
                {
                    "event": "run_validated",
                    "at": _now(),
                    "run_key": run_key,
                    "verdict": summary["verdict"],
                }
            )
    passed = sum(summary.get("verdict") == "passed" for summary in summaries.values())
    failed = len(run_keys) - passed
    verdict = "passed" if failed == 0 else "failed"
    ledger.append(
        {
            "event": "validation_completed",
            "at": _now(),
            "verdict": verdict,
            "passed": passed,
            "failed": failed,
        }
    )
    _write_evidence(
        validation_root / "validation.jsonl",
        b"".join(
            json.dumps(event, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
            for event in ledger
        ),
    )
    marker = {
        "format": VALIDATION_FORMAT,
        "prepared_root": str(resolved),
        "plan_id": prepared["plan_id"],
        "completed_at": ledger[-1]["at"],
        "verdict": verdict,
        "run_count": len(run_keys),
        "passed": passed,
        "failed": failed,
        "provider_runners_launched": False,
        "command_validators_executed": bool(command_ids),
        "run_verdicts": {run_key: summaries[run_key]["verdict"] for run_key in run_keys},
        "evidence_sha256": _validation_hashes(validation_root),
    }
    _write_evidence(validation_root / "validation.json", _json_bytes(marker))
    return ValidationResult(validation_root, verdict, len(run_keys), passed, failed)


def verify_validation(root: str | Path) -> dict[str, Any]:
    """Verify an optional immutable validation envelope."""
    resolved = Path(root).resolve()
    validation_root = resolved / "validation"
    marker = _read_json(validation_root / "validation.json")
    if marker.get("format") != VALIDATION_FORMAT:
        raise ValidationError(f"{validation_root} has an unknown validation format")
    if marker.get("prepared_root") != str(resolved):
        raise ValidationError("validation marker points at a different prepared experiment")
    expected = marker.get("evidence_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValidationError("validation marker has no evidence hashes")
    actual = _validation_hashes(validation_root)
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected))
        if not changed:
            changed = sorted(path for path in actual if actual[path] != expected.get(path))
        names = ", ".join(changed[:5]) or "validation evidence files"
        raise ValidationError(f"validation evidence changed after creation: {names}")
    return {
        "ok": True,
        "format": VALIDATION_FORMAT,
        "root": str(validation_root),
        "verdict": marker.get("verdict"),
        "run_count": marker.get("run_count"),
        "passed": marker.get("passed"),
        "failed": marker.get("failed"),
    }
