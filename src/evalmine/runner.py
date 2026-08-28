"""Subscription-backed agent runners for prepared v2 experiments.

The runner boundary is intentionally separate from preparation. Preflight only
probes local executable capabilities; execution is the sole provider-call path and
requires an explicit caller confirmation in the CLI.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment import ExperimentError
from .suite import SECRET_PATTERNS
from .workspace import _path_content_hash, _tree_hash, _write_once, verify_prepared

EXECUTION_FORMAT = "evalmine-execution-v1"
DEFAULT_TURN_TIMEOUT = 1800
MAX_CODEX_ROLLOUT_BYTES = 64 * 1024 * 1024
MAX_CODEX_ROLLOUT_LINE_BYTES = 8 * 1024 * 1024

_RUNNER_COMMANDS = {
    "claude-code": "claude",
    "codex-cli": "codex",
    "gemini-cli": "gemini",
}

_HELP_COMMANDS = {
    "claude-code": (("--help",),),
    "codex-cli": (("--help",), ("exec", "--help"), ("exec", "resume", "--help")),
    "gemini-cli": (("--help",),),
}

_REQUIRED_HELP = {
    "claude-code": (
        "--print",
        "--output-format",
        "--model",
        "--resume",
        "--session-id",
        "--settings",
        "--safe-mode",
        "--strict-mcp-config",
        "--mcp-config",
        "--disable-slash-commands",
        "--permission-mode",
        "--no-chrome",
        "--max-budget-usd",
        "--plugin-dir",
    ),
    "codex-cli": (
        "--json",
        "--model",
        "--sandbox",
        "--cd",
        "--skip-git-repo-check",
        "resume",
        "--ask-for-approval",
        "--ignore-user-config",
        "--ephemeral",
        "--config",
        "--add-dir",
    ),
    "gemini-cli": (
        "--prompt",
        "--output-format",
        "--model",
        "--resume",
        "--sandbox",
        "--extensions",
        "--approval-mode",
    ),
}

_RESERVED_ARGUMENT_PREFIXES = (
    "-a",
    "-c",
    "-e",
    "-m",
    "-p",
    "-r",
    "-s",
    "-w",
    "--add-dir",
    "--allow",
    "--approval",
    "--auth",
    "--bare",
    "--cd",
    "--config",
    "--danger",
    "--disallow",
    "--ephemeral",
    "--extensions",
    "--ignore",
    "--input-format",
    "--json",
    "--mcp",
    "--model",
    "--output",
    "--permission",
    "--plugin",
    "--prompt",
    "--resume",
    "--safe-mode",
    "--sandbox",
    "--session",
    "--setting",
    "--tools",
    "--worktree",
    "--yolo",
)

_RESERVED_SETTING_PARTS = {
    "add-dir",
    "add_dir",
    "approval",
    "auth",
    "enabledplugins",
    "external_writes",
    "extensions",
    "mcp",
    "model",
    "permissions",
    "plugins",
    "sandbox",
    "session",
    "tools.sandbox",
    "workspace",
}

_SECRET_ENV_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_write_lock = threading.Lock()
_progress_lock = threading.Lock()
ProgressCallback = Callable[[dict[str, Any]], None]
CodexIdentityProbe = Callable[[str, Path], dict[str, Any]]


class RunnerError(ExperimentError):
    """Agent execution could not be performed or recorded safely."""


class ExecutionRefused(RunnerError):
    """Execution failed its zero-call preflight."""


@dataclass(frozen=True)
class ProcessResult:
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


class ProcessDriver:
    """Small injectable subprocess boundary used by preflight and execution."""

    def run(
        self,
        args: list[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        timeout: int,
        env: dict[str, str] | None = None,
    ) -> ProcessResult:
        started = time.monotonic()
        try:
            result = subprocess.run(
                args,
                cwd=cwd,
                input=input_text,
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return ProcessResult(
                args=tuple(args),
                returncode=124,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
                duration_ms=round((time.monotonic() - started) * 1000),
                timed_out=True,
            )
        except OSError as exc:
            return ProcessResult(
                args=tuple(args),
                returncode=127,
                stdout="",
                stderr=str(exc),
                duration_ms=round((time.monotonic() - started) * 1000),
            )
        return ProcessResult(
            args=tuple(args),
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=round((time.monotonic() - started) * 1000),
        )


@dataclass(frozen=True)
class RunnerProbe:
    runner: str
    executable: str | None
    available: bool
    version: str | None
    capabilities_ok: bool
    missing_flags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "runner": self.runner,
            "executable": self.executable,
            "available": self.available,
            "version": self.version,
            "capabilities_ok": self.capabilities_ok,
            "missing_flags": list(self.missing_flags),
            "authentication": "not-probed",
        }


@dataclass(frozen=True)
class PreflightReport:
    root: Path
    ok: bool
    probes: tuple[RunnerProbe, ...]
    issues: tuple[str, ...]
    run_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "root": str(self.root),
            "provider_calls": False,
            "run_count": self.run_count,
            "runners": [probe.as_dict() for probe in self.probes],
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class ExecutionResult:
    root: Path
    status: str
    run_count: int
    succeeded: int
    failed: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": EXECUTION_FORMAT,
            "root": str(self.root),
            "status": self.status,
            "run_count": self.run_count,
            "succeeded": self.succeeded,
            "failed": self.failed,
        }


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerError(f"cannot read prepared evidence {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"prepared evidence {path} is not a JSON object")
    return value


def _load_prepared(root: str | Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    verification = verify_prepared(root)
    resolved = Path(verification["root"])
    marker = _read_json(resolved / "prepared.json")
    plan = _read_json(resolved / "plan.json")
    return resolved, marker, plan


def _prepared_retry(root: Path, marker: dict[str, Any]) -> dict[str, Any] | None:
    declared = marker.get("retry")
    if declared is None:
        return None
    if declared != "retry.json":
        raise RunnerError("prepared retry marker must reference retry.json")
    retry = _read_json(root / declared)
    inherited = retry.get("inherited_run_keys")
    retried = retry.get("retry_run_keys")
    if not isinstance(inherited, list) or not isinstance(retried, list):
        raise RunnerError("prepared retry has invalid inherited or retry run keys")
    return retry


def _resolve_executable(
    runner: str, overrides: dict[str, str] | None
) -> str | None:
    if overrides and runner in overrides:
        path = Path(overrides[runner]).expanduser()
        if path.is_file():
            return str(path.resolve())
        return None
    command = _RUNNER_COMMANDS[runner]
    return shutil.which(command)


def _probe_runner(
    runner: str,
    *,
    cwd: Path,
    overrides: dict[str, str] | None,
    driver: ProcessDriver,
) -> RunnerProbe:
    executable = _resolve_executable(runner, overrides)
    if executable is None:
        return RunnerProbe(runner, None, False, None, False, _REQUIRED_HELP[runner])
    version_result = driver.run([executable, "--version"], cwd=cwd, timeout=10)
    help_results = [
        driver.run([executable, *arguments], cwd=cwd, timeout=10)
        for arguments in _HELP_COMMANDS[runner]
    ]
    help_text = "\n".join(result.stdout + "\n" + result.stderr for result in help_results)
    missing = tuple(flag for flag in _REQUIRED_HELP[runner] if flag not in help_text)
    version_text = (version_result.stdout or version_result.stderr).strip().splitlines()
    return RunnerProbe(
        runner=runner,
        executable=executable,
        available=version_result.returncode == 0
        and all(result.returncode == 0 for result in help_results),
        version=version_text[0][:200] if version_text else None,
        capabilities_ok=(
            version_result.returncode == 0
            and all(result.returncode == 0 for result in help_results)
            and not missing
        ),
        missing_flags=missing,
    )


def _native_sandbox_issue(
    runner: str, executable_overrides: dict[str, str] | None
) -> str | None:
    # An explicit executable may itself be a controlled wrapper and is also the
    # injection point used by the network-free test suite.
    if executable_overrides and runner in executable_overrides:
        return None
    system = platform.system()
    if runner == "claude-code":
        if system == "Darwin" and shutil.which("sandbox-exec") is None:
            return "macOS Seatbelt executable sandbox-exec is unavailable"
        if system == "Linux" and (
            shutil.which("bwrap") is None or shutil.which("socat") is None
        ):
            return "Claude sandboxing requires both bwrap and socat on Linux"
        if system == "Windows":
            return "Claude native sandboxing does not support native Windows"
    if runner == "gemini-cli":
        if system == "Darwin" and shutil.which("sandbox-exec") is None:
            return "macOS Seatbelt executable sandbox-exec is unavailable"
        if system == "Linux" and not any(
            shutil.which(command) for command in ("docker", "podman", "sandbox-exec")
        ):
            return "Gemini sandboxing requires Docker, Podman, or sandbox-exec on Linux"
    return None


def _unsafe_argument(argument: str) -> str | None:
    if not argument.startswith("--") or any(character.isspace() for character in argument):
        return "arguments must use one self-contained --flag or --flag=value token"
    lowered = argument.lower()
    if lowered.startswith(_RESERVED_ARGUMENT_PREFIXES):
        return "argument is owned by evalmine or can weaken isolation"
    if any(pattern.search(argument) for _label, pattern in SECRET_PATTERNS):
        return "argument looks like a credential"
    return None


def _unsafe_setting(key: str, value: Any) -> str | None:
    lowered = key.lower().replace("-", "_")
    if any(part in lowered for part in _RESERVED_SETTING_PARTS):
        return "setting is owned by evalmine or can weaken isolation"
    text = str(value)
    if any(pattern.search(text) for _label, pattern in SECRET_PATTERNS):
        return "setting value looks like a credential"
    return None


def _treatment_issues(
    run: dict[str, Any], treatment: dict[str, Any], plan: dict[str, Any]
) -> list[str]:
    runner = run.get("runner")
    prefix = f"run {run.get('run_key', '<unknown>')} ({runner})"
    issues: list[str] = []
    if runner not in _RUNNER_COMMANDS:
        return [f"{prefix}: runner is not executable in Phase 3A"]
    if run.get("auth") == "api":
        if runner != "claude-code":
            issues.append(
                f"{prefix}: only Claude Code exposes an enforceable per-run API ceiling"
            )
        if not isinstance(run.get("max_cost_usd"), (int, float)):
            issues.append(f"{prefix}: auth=api requires max_cost_usd")
    elif run.get("auth") not in {"subscription", "inherited", "local"}:
        issues.append(f"{prefix}: unsupported auth mode {run.get('auth')!r}")
    isolation = plan.get("isolation", {})
    if isolation.get("external_writes") == "allowlisted":
        if runner == "gemini-cli":
            issues.append(
                f"{prefix}: Gemini external-write allowlists are not enforceable by this runner"
            )
        for declared in isolation.get("external_write_allowlist", []):
            if "{run_key}" not in str(declared):
                issues.append(
                    f"{prefix}: external write target must contain {{run_key}} so arms "
                    "cannot share mutable state"
                )
                continue
            target = Path(str(declared).replace("{run_key}", str(run.get("run_key"))))
            if not target.is_absolute():
                issues.append(f"{prefix}: external write target must be absolute: {declared!r}")
            elif target == Path(target.anchor) or target == Path.home():
                issues.append(f"{prefix}: external write target is too broad: {declared!r}")
            elif target.exists():
                issues.append(
                    f"{prefix}: per-run external write target already exists: {target}"
                )
            elif target.parent.is_symlink() or not target.parent.is_dir():
                issues.append(
                    f"{prefix}: external write target parent is unavailable: {target.parent}"
                )
    elif isolation.get("external_writes") != "deny":
        issues.append(f"{prefix}: unknown external-write policy")
    plugins = treatment.get("plugins")
    instructions = treatment.get("instructions")
    if runner == "claude-code":
        if plugins == "allowlist":
            issues.append(f"{prefix}: exact installed-plugin allowlists are not supported")
        if plugins == "directories":
            workspace = Path(str(run.get("workspace", ""))).resolve()
            for declared in treatment.get("plugin_directories", []):
                target = (workspace / str(declared)).resolve()
                try:
                    target.relative_to(workspace)
                except ValueError:
                    issues.append(f"{prefix}: plugin directory escapes the workspace")
                    continue
                if not target.is_dir() or target.is_symlink():
                    issues.append(
                        f"{prefix}: plugin directory is absent or is a symlink: {declared!r}"
                    )
        if plugins == "none" and instructions != "none":
            issues.append(
                f"{prefix}: Claude can guarantee plugins=none only with instructions=none"
            )
    elif runner == "codex-cli" and plugins == "allowlist":
        issues.append(f"{prefix}: Codex has no documented per-run plugin allowlist")
    for argument in treatment.get("arguments", []):
        reason = _unsafe_argument(argument)
        if reason:
            issues.append(f"{prefix}: unsafe argument {argument!r}: {reason}")
    for key, value in treatment.get("settings", {}).items():
        reason = _unsafe_setting(key, value)
        if reason:
            issues.append(f"{prefix}: unsafe setting {key!r}: {reason}")
    return issues


def preflight_experiment(
    root: str | Path,
    *,
    executable_overrides: dict[str, str] | None = None,
    driver: ProcessDriver | None = None,
) -> PreflightReport:
    """Probe local runner capabilities without making a provider call or writing evidence."""
    driver = driver or ProcessDriver()
    resolved, marker, plan = _load_prepared(root)
    issues: list[str] = []
    if plan.get("isolation", {}).get("session") != "fresh-per-run":
        issues.append("session=reuse-per-arm is not executable; use fresh-per-run")
    retry = _prepared_retry(resolved, marker)
    run_keys = (
        retry["retry_run_keys"] if retry is not None else marker.get("run_keys", [])
    )
    run_docs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    runners: set[str] = set()
    for run_key in run_keys:
        run_dir = resolved / "runs" / run_key
        run = _read_json(run_dir / "run.json")
        treatment = _read_json(run_dir / "treatment.json")
        run_docs.append((run, treatment))
        if run.get("runner") in _RUNNER_COMMANDS:
            runners.add(run["runner"])
        issues.extend(_treatment_issues(run, treatment, plan))
    probes = tuple(
        _probe_runner(
            runner,
            cwd=resolved,
            overrides=executable_overrides,
            driver=driver,
        )
        for runner in sorted(runners)
    )
    for probe in probes:
        if not probe.available:
            issues.append(f"runner {probe.runner}: executable is absent or did not answer")
        elif not probe.capabilities_ok:
            flags = ", ".join(probe.missing_flags)
            issues.append(f"runner {probe.runner}: required CLI capabilities missing: {flags}")
        sandbox_issue = _native_sandbox_issue(probe.runner, executable_overrides)
        if sandbox_issue:
            issues.append(f"runner {probe.runner}: {sandbox_issue}")
    return PreflightReport(
        root=resolved,
        ok=not issues,
        probes=probes,
        issues=tuple(dict.fromkeys(issues)),
        run_count=len(run_docs),
    )


def _nested_settings(settings: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dotted, value in settings.items():
        target = result
        parts = dotted.split(".")
        if any(not part for part in parts):
            raise RunnerError(f"invalid empty component in setting {dotted!r}")
        for part in parts[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise RunnerError(f"setting {dotted!r} conflicts with another override")
            target = existing
        target[parts[-1]] = value
    return result


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _toml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    raise RunnerError("Codex setting overrides cannot be null")


def _claude_settings(
    settings: dict[str, Any], external_write_allowlist: tuple[str, ...] = ()
) -> str:
    safety = {
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "excludedCommands": [],
            "filesystem": {"allowWrite": list(external_write_allowlist)},
        },
        "permissions": {"defaultMode": "acceptEdits"},
    }
    return json.dumps(
        _deep_merge(_nested_settings(settings), safety),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _build_command(
    *,
    runner: str,
    executable: str,
    workspace: Path,
    model: str,
    auth: str,
    max_cost_usd: float | None,
    treatment: dict[str, Any],
    external_write_allowlist: tuple[str, ...],
    turn_index: int,
    total_turns: int,
    session_id: str | None,
) -> tuple[list[str], dict[str, str]]:
    extra = list(treatment.get("arguments", []))
    settings = dict(treatment.get("settings", {}))
    env_overrides: dict[str, str] = {}
    if runner == "claude-code":
        args = [
            executable,
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            model,
            "--permission-mode",
            "acceptEdits",
            "--settings",
            _claude_settings(settings, external_write_allowlist),
            "--no-chrome",
        ]
        if auth == "api":
            if max_cost_usd is None:
                raise RunnerError("Claude API arm has no max_cost_usd")
            args.extend(["--max-budget-usd", str(max_cost_usd)])
        if treatment.get("plugins") == "none":
            args.extend(
                [
                    "--safe-mode",
                    "--strict-mcp-config",
                    "--mcp-config",
                    '{"mcpServers":{}}',
                    "--disable-slash-commands",
                ]
            )
        elif treatment.get("plugins") == "directories":
            for declared in treatment.get("plugin_directories", []):
                args.extend(["--plugin-dir", str((workspace / declared).resolve())])
        if turn_index == 1:
            if session_id is None:
                raise RunnerError("Claude run has no allocated session id")
            args.extend(["--session-id", session_id])
        else:
            if session_id is None:
                raise RunnerError("Claude follow-up has no session id")
            args.extend(["--resume", session_id])
        args.extend(extra)
        return args, env_overrides

    if runner == "codex-cli":
        config_args: list[str] = []
        for key, value in settings.items():
            config_args.extend(["--config", f"{key}={_toml_scalar(value)}"])
        common = [
            "--json",
            "--model",
            model,
            "--skip-git-repo-check",
            *config_args,
        ]
        if treatment.get("plugins") == "none":
            common.append("--ignore-user-config")
        if turn_index == 1:
            args = [
                executable,
                "--ask-for-approval",
                "never",
                "exec",
                "--sandbox",
                "workspace-write",
                "--cd",
                str(workspace),
                *[
                    value
                    for target in external_write_allowlist
                    for value in ("--add-dir", target)
                ],
                *common,
            ]
            if total_turns == 1:
                args.append("--ephemeral")
            args.extend(extra)
            args.append("-")
            return args, env_overrides
        if session_id is None:
            raise RunnerError("Codex follow-up has no thread id")
        args = [executable, "exec", "resume", *common, *extra, session_id, "-"]
        return args, env_overrides

    if runner == "gemini-cli":
        args = [
            executable,
            "--prompt",
            "",
            "--output-format",
            "stream-json",
            "--model",
            model,
            "--sandbox",
            "--approval-mode",
            "auto_edit",
        ]
        plugins = treatment.get("plugins")
        if plugins == "none":
            args.extend(["--extensions", "none"])
        elif plugins == "allowlist":
            for plugin in treatment.get("plugin_allowlist", []):
                args.extend(["--extensions", plugin])
        if turn_index > 1:
            if session_id is None:
                raise RunnerError("Gemini follow-up has no session id")
            args.extend(["--resume", session_id])
        args.extend(extra)
        if platform.system() == "Darwin":
            env_overrides["SEATBELT_PROFILE"] = "strict-open"
        if settings:
            env_overrides["EVALMINE_GEMINI_SETTINGS_JSON"] = json.dumps(
                _nested_settings(settings), sort_keys=True, ensure_ascii=False
            )
        return args, env_overrides
    raise RunnerError(f"unsupported runner {runner!r}")


def _child_env(
    overrides: dict[str, str],
    settings_path: Path | None = None,
    credential_env_names: tuple[str, ...] = (),
) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if name in credential_env_names
        or not any(part in name.upper() for part in _SECRET_ENV_PARTS)
    }
    settings_json = overrides.pop("EVALMINE_GEMINI_SETTINGS_JSON", None)
    env.update(overrides)
    if settings_json is not None:
        if settings_path is None:
            raise RunnerError("Gemini settings path was not allocated")
        _write_once(settings_path, (settings_json + "\n").encode("utf-8"))
        env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(settings_path)
    env["NO_COLOR"] = "1"
    return env


def _redact_secrets(text: str) -> str:
    redacted = text
    for _label, pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED_CREDENTIAL]", redacted)
    return redacted


def _parse_events(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    malformed: list[str] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed.append(line[:500])
            continue
        if isinstance(event, dict):
            events.append(event)
        else:
            malformed.append(line[:500])
    return events, malformed


def _recursive_values(node: Any, key: str):
    if isinstance(node, dict):
        for current_key, value in node.items():
            if current_key == key:
                yield value
            yield from _recursive_values(value, key)
    elif isinstance(node, list):
        for value in node:
            yield from _recursive_values(value, key)


def _first_string(events: list[dict[str, Any]], *keys: str) -> str | None:
    for event in events:
        for key in keys:
            for value in _recursive_values(event, key):
                if isinstance(value, str) and value:
                    return value
    return None


def _codex_home(environment: dict[str, str]) -> Path:
    declared = environment.get("CODEX_HOME")
    if declared:
        return Path(declared).expanduser()
    return Path.home() / ".codex"


def _codex_rollout_identity(session_id: str, codex_home: Path) -> dict[str, Any]:
    """Extract only non-secret model metadata from one persisted Codex session.

    Codex's documented ``exec --json`` stream does not include model identity, while
    non-ephemeral multi-turn runs persist a rollout for resume. Restrict discovery to
    the validated session UUID below ``$CODEX_HOME/sessions`` and retain only model,
    provider, record type, and integrity metadata. Raw rollout content is never copied.
    """
    try:
        canonical_session_id = str(uuid.UUID(session_id))
    except (ValueError, AttributeError):
        return {
            "status": "unavailable",
            "source": "codex-session-rollout",
            "reason": "session-id-is-not-a-uuid",
        }

    sessions = codex_home / "sessions"
    try:
        if sessions.is_symlink():
            raise OSError("sessions root is a symlink")
        sessions_root = sessions.resolve(strict=True)
        if not sessions_root.is_dir():
            raise OSError("sessions root is not a directory")
        matches = [
            path
            for path in sessions_root.rglob(f"*{canonical_session_id}.jsonl")
            if path.is_file() and not path.is_symlink()
        ]
    except OSError:
        return {
            "status": "unavailable",
            "source": "codex-session-rollout",
            "reason": "sessions-unavailable",
        }
    if not matches:
        return {
            "status": "unavailable",
            "source": "codex-session-rollout",
            "reason": "rollout-not-found",
        }
    if len(matches) != 1:
        return {
            "status": "unavailable",
            "source": "codex-session-rollout",
            "reason": "rollout-ambiguous",
        }

    rollout = matches[0]
    try:
        rollout.relative_to(sessions_root)
        if rollout.stat().st_size > MAX_CODEX_ROLLOUT_BYTES:
            raise OSError("rollout exceeds the metadata-reader limit")
        digest = hashlib.sha256()
        records: list[dict[str, str]] = []
        parse_errors = 0
        with rollout.open("rb") as handle:
            for raw_line in handle:
                digest.update(raw_line)
                if len(raw_line) > MAX_CODEX_ROLLOUT_LINE_BYTES:
                    raise OSError("rollout line exceeds the metadata-reader limit")
                try:
                    event = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    parse_errors += 1
                    continue
                if not isinstance(event, dict) or not isinstance(event.get("payload"), dict):
                    continue
                event_type = event.get("type")
                payload = event["payload"]
                model: Any = None
                provider: Any = None
                record_type: str | None = None
                if event_type == "session_meta":
                    recorded_id = payload.get("id") or payload.get("session_id")
                    if recorded_id != canonical_session_id:
                        continue
                    base_instructions = payload.get("base_instructions", {})
                    provenance = (
                        base_instructions.get("provenance", {})
                        if isinstance(base_instructions, dict)
                        else {}
                    )
                    if isinstance(provenance, dict) and provenance.get("type") == "model":
                        model = provenance.get("model")
                    provider = payload.get("model_provider")
                    record_type = "session_meta"
                elif event_type == "turn_context":
                    model = payload.get("model")
                    record_type = "turn_context"
                elif event_type == "event_msg" and payload.get("type") == (
                    "thread_settings_applied"
                ):
                    settings = payload.get("thread_settings", {})
                    if isinstance(settings, dict):
                        model = settings.get("model")
                        provider = settings.get("model_provider_id")
                    record_type = "thread_settings_applied"
                if record_type is None or not isinstance(model, str) or not model:
                    continue
                record = {"record": record_type, "model": model}
                if isinstance(provider, str) and provider:
                    record["provider"] = provider
                records.append(record)
    except (OSError, ValueError):
        return {
            "status": "unavailable",
            "source": "codex-session-rollout",
            "reason": "rollout-unreadable",
        }

    models = sorted({record["model"] for record in records})
    providers = sorted(
        {record["provider"] for record in records if record.get("provider")}
    )
    result: dict[str, Any] = {
        "status": "recorded" if len(models) == 1 else "conflicting" if models else "unavailable",
        "source": "codex-session-rollout",
        "confidence": "runner-runtime" if len(models) == 1 else "unavailable",
        "records": records,
        "models": models,
        "providers": providers,
        "record_count": len(records),
        "parse_error_count": parse_errors,
        "rollout_sha256": digest.hexdigest(),
    }
    if len(models) == 1:
        result["observed_model"] = models[0]
    elif not models:
        result["reason"] = "model-metadata-not-found"
    else:
        result["reason"] = "multiple-runtime-models-recorded"
    if len(providers) == 1:
        result["observed_provider"] = providers[0]
    return result


def _normalize_events(
    runner: str, events: list[dict[str, Any]]
) -> tuple[str | None, str | None, str | None, list[dict[str, Any]], dict[str, Any]]:
    session_id: str | None = None
    observed_model: str | None = None
    final: str | None = None
    tools: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    model_usage: dict[str, Any] = {}
    assistant_models: list[str] = []
    if runner == "claude-code":
        session_id = _first_string(events, "session_id")
        for event in events:
            message = event.get("message")
            if event.get("type") == "assistant" and isinstance(message, dict):
                message_model = message.get("model")
                if isinstance(message_model, str) and message_model:
                    assistant_models.append(message_model)
            if event.get("type") == "result" and isinstance(event.get("result"), str):
                final = event["result"]
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                if isinstance(event.get("total_cost_usd"), (int, float)):
                    usage["reported_cost_usd"] = float(event["total_cost_usd"])
            for block_type in ("tool_use", "server_tool_use"):
                for block in _recursive_values(event, "content"):
                    if isinstance(block, list):
                        for item in block:
                            if isinstance(item, dict) and item.get("type") == block_type:
                                tools.append(
                                    {
                                        "type": block_type,
                                        "name": item.get("name"),
                                        "id": item.get("id"),
                                    }
                                )
        reported_model_usage = next(
            (value for value in _recursive_values(events, "modelUsage") if isinstance(value, dict)),
            None,
        )
        if reported_model_usage:
            model_usage = reported_model_usage
        if assistant_models:
            # Claude Code's modelUsage may contain auxiliary infrastructure models.
            # The assistant message identifies the model that authored the response.
            observed_model = assistant_models[-1]
        elif model_usage:
            observed_model = next(iter(model_usage), None)
    elif runner == "codex-cli":
        session_id = _first_string(events, "thread_id")
        for event in events:
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "agent_message" and isinstance(item.get("text"), str):
                final = item["text"]
            elif item_type not in {None, "reasoning"}:
                tools.append(
                    {
                        "type": item_type,
                        "name": item.get("name") or item.get("command"),
                        "id": item.get("id"),
                    }
                )
        usage_value = next(
            (value for value in _recursive_values(events, "usage") if isinstance(value, dict)),
            None,
        )
        if usage_value:
            usage = usage_value
    else:
        session_id = _first_string(events, "session_id", "sessionId")
        for event in events:
            event_type = event.get("type")
            if event_type in {"message", "assistant"} and event.get("role") in {
                None,
                "assistant",
            }:
                content = event.get("content") or event.get("text")
                if isinstance(content, str):
                    final = content
            if event_type in {"tool_use", "tool_call", "tool_result"}:
                tools.append(
                    {
                        "type": event_type,
                        "name": event.get("name") or event.get("tool_name"),
                        "id": event.get("id") or event.get("tool_id"),
                    }
                )
            if event_type == "result":
                content = event.get("result") or event.get("response")
                if isinstance(content, str):
                    final = content
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
    observed_model = observed_model or _first_string(events, "model", "model_id")
    event_types = [str(event.get("type", "unknown")) for event in events]
    normalized = [
        {"sequence": index, "provider_event": event_type}
        for index, event_type in enumerate(event_types, start=1)
    ]
    auxiliary_models = [model for model in model_usage if model != observed_model]
    return session_id, observed_model, final, tools, {
        "events": normalized,
        "usage": usage,
        "model_usage": model_usage,
        "assistant_models": list(dict.fromkeys(assistant_models)),
        "auxiliary_models": auxiliary_models,
    }


def _safe_command(args: list[str]) -> list[str]:
    safe: list[str] = []
    redact_next = False
    for argument in args:
        if redact_next:
            safe.append("<CONFIG_OVERRIDE>")
            redact_next = False
            continue
        safe.append(argument)
        if argument in {"--settings", "--config", "--mcp-config"}:
            redact_next = True
    safe.append("<PROMPT_VIA_STDIN>")
    return safe


def _prompt_map(root: Path) -> dict[tuple[str, int], str]:
    index = _read_json(root / "inputs" / "index.json")
    prompts: dict[tuple[str, int], str] = {}
    for entry in index.get("entries", []):
        logical = entry.get("logical", "")
        match = re.fullmatch(r"episode/([^/]+)/turn/(\d+)", logical)
        if entry.get("kind") != "prompt" or not match:
            continue
        blob = (root / "inputs" / str(entry.get("blob", ""))).resolve()
        inputs_root = (root / "inputs").resolve()
        try:
            blob.relative_to(inputs_root)
        except ValueError as exc:
            raise RunnerError(f"prompt blob escapes input evidence: {blob}") from exc
        content = blob.read_text(encoding="utf-8")
        expected = entry.get("sha256")
        actual = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if actual != expected:
            raise RunnerError(f"prompt blob hash mismatch for {logical}")
        prompts[(match.group(1), int(match.group(2)))] = content
    return prompts


def _execution_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "execution.json":
            hashes[path.relative_to(root).as_posix()] = _path_content_hash(path)
    return hashes


def verify_execution(root: str | Path) -> dict[str, Any]:
    """Verify the optional execution envelope associated with a prepared root."""
    resolved, marker, _plan = _load_prepared(root)
    execution_root = resolved / "execution"
    execution = _read_json(execution_root / "execution.json")
    if execution.get("format") != EXECUTION_FORMAT:
        raise RunnerError(f"{execution_root} has an unknown execution format")
    if execution.get("prepared_root") != str(resolved):
        raise RunnerError("execution marker points at a different prepared experiment")
    expected = execution.get("evidence_sha256")
    if not isinstance(expected, dict) or not expected:
        raise RunnerError("execution marker has no evidence hashes")
    actual = _execution_hashes(execution_root)
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected))
        if not changed:
            changed = sorted(path for path in actual if actual[path] != expected.get(path))
        names = ", ".join(changed[:5]) or "execution evidence files"
        raise RunnerError(f"execution evidence changed after creation: {names}")
    return {
        "ok": True,
        "format": EXECUTION_FORMAT,
        "root": str(execution_root),
        "status": execution.get("status"),
        "run_count": execution.get("run_count"),
        "succeeded": execution.get("succeeded"),
        "failed": execution.get("failed"),
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _write_execution(path: Path, content: bytes) -> None:
    with _write_lock:
        _write_once(path, content)


def _emit_progress(
    progress: ProgressCallback | None,
    event: dict[str, Any],
) -> None:
    if progress is None:
        return
    with _progress_lock:
        progress(event)


def _run_one(
    *,
    root: Path,
    execution_root: Path,
    run_key: str,
    executable: str,
    prompts: dict[tuple[str, int], str],
    timeout: int,
    driver: ProcessDriver,
    external_write_allowlist: tuple[str, ...],
    run_position: int,
    run_count: int,
    progress: ProgressCallback | None,
    codex_identity_probe: CodexIdentityProbe,
) -> dict[str, Any]:
    source_dir = root / "runs" / run_key
    run = _read_json(source_dir / "run.json")
    treatment = _read_json(source_dir / "treatment.json")
    workspace = Path(run["workspace"])
    output_dir = execution_root / "runs" / run_key
    output_dir.mkdir(parents=True, exist_ok=False)
    runner = run["runner"]
    episode_id = run["episode"]
    total_turns = int(run["turns"])
    session_id = (
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"evalmine:{run['session_key']}"))
        if runner == "claude-code"
        else None
    )
    requested_session = session_id
    started_at = _now()
    started_clock = time.monotonic()
    turn_summaries: list[dict[str, Any]] = []
    status = "succeeded"
    error: str | None = None
    observed_model: str | None = None
    observed_model_source: str | None = None
    codex_home: Path | None = None
    progress_context = {
        "run_position": run_position,
        "run_count": run_count,
        "run_key": run_key,
        "arm": run["arm"],
        "runner": runner,
        "model": run["model"],
        "turn_count": total_turns,
    }
    _emit_progress(
        progress,
        {"event": "run_started", "at": started_at, **progress_context},
    )
    for turn_index in range(1, total_turns + 1):
        prompt = prompts.get((episode_id, turn_index))
        if prompt is None:
            status = "failed"
            error = f"missing prompt for episode {episode_id} turn {turn_index}"
            break
        try:
            command, env_overrides = _build_command(
                runner=runner,
                executable=executable,
                workspace=workspace,
                model=run["model"],
                auth=run["auth"],
                max_cost_usd=(
                    float(run["max_cost_usd"]) / total_turns
                    if run.get("max_cost_usd") is not None
                    else None
                ),
                treatment=treatment,
                external_write_allowlist=external_write_allowlist,
                turn_index=turn_index,
                total_turns=total_turns,
                session_id=session_id,
            )
            settings_path = output_dir / f"turn-{turn_index:03d}.runner-settings.json"
            credential_env_names = (
                ("ANTHROPIC_API_KEY",)
                if runner == "claude-code" and run["auth"] == "api"
                else ()
            )
            env = _child_env(
                env_overrides,
                settings_path=settings_path,
                credential_env_names=credential_env_names,
            )
            if runner == "codex-cli":
                codex_home = _codex_home(env)
            turn_started = _now()
            _emit_progress(
                progress,
                {
                    "event": "turn_started",
                    "at": turn_started,
                    "turn": turn_index,
                    **progress_context,
                },
            )
            process = driver.run(
                command,
                cwd=workspace,
                input_text=prompt,
                timeout=timeout,
                env=env,
            )
        except RunnerError as exc:
            status = "failed"
            error = str(exc)
            break
        stdout = _redact_secrets(process.stdout)
        stderr = _redact_secrets(process.stderr)
        events, malformed = _parse_events(stdout)
        found_session, found_model, final, tools, normalized = _normalize_events(runner, events)
        if runner != "claude-code" and found_session:
            if session_id is None:
                session_id = found_session
            elif session_id != found_session:
                status = "failed"
                error = "runner changed session id within one planned run"
        if observed_model is None and found_model is not None:
            observed_model = found_model
            observed_model_source = "runner-jsonl"
        final = _redact_secrets(final or "")
        turn_ok = (
            process.returncode == 0
            and not process.timed_out
            and not malformed
            and bool(final.strip())
        )
        if turn_ok and turn_index < total_turns and session_id is None:
            turn_ok = False
            error = "runner emitted no session identifier for a follow-up"
        if not turn_ok and error is None:
            if process.timed_out:
                error = f"turn {turn_index} timed out"
            elif process.returncode != 0:
                error = f"turn {turn_index} exited {process.returncode}"
            elif malformed:
                error = f"turn {turn_index} emitted malformed JSONL"
            else:
                error = f"turn {turn_index} emitted no final response"
        stem = output_dir / f"turn-{turn_index:03d}"
        _write_execution(stem.with_suffix(".raw.jsonl"), stdout.encode("utf-8"))
        _write_execution(stem.with_suffix(".stderr.txt"), stderr.encode("utf-8"))
        _write_execution(stem.with_suffix(".final.txt"), final.encode("utf-8"))
        summary = {
            "turn": turn_index,
            "status": "succeeded" if turn_ok else "failed",
            "started_at": turn_started,
            "completed_at": _now(),
            "duration_ms": process.duration_ms,
            "exit_code": process.returncode,
            "timed_out": process.timed_out,
            "command": _safe_command(command),
            "environment_override_names": sorted(env_overrides),
            "event_count": len(events),
            "malformed_event_lines": malformed,
            "events": normalized["events"],
            "tools": tools,
            "usage": normalized["usage"],
            "final_sha256": hashlib.sha256(final.encode("utf-8")).hexdigest(),
        }
        _write_execution(stem.with_suffix(".json"), _json_bytes(summary))
        turn_summaries.append(summary)
        _emit_progress(
            progress,
            {
                "event": "turn_completed",
                "at": summary["completed_at"],
                "turn": turn_index,
                "status": summary["status"],
                "duration_ms": process.duration_ms,
                **progress_context,
            },
        )
        if not turn_ok:
            status = "failed"
            break
    completed_at = _now()
    usage_totals: dict[str, int | float] = {}
    for turn in turn_summaries:
        for key, value in turn.get("usage", {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage_totals[key] = usage_totals.get(key, 0) + value
    metered_cost = usage_totals.get("reported_cost_usd")
    reported_cost = metered_cost if run["auth"] == "api" else None
    billing_basis = {
        "subscription": "subscription",
        "api": "api-metered",
        "local": "local",
        "inherited": "unknown",
    }.get(run["auth"], "unknown")
    model_identity: dict[str, Any] = {
        "status": "recorded" if observed_model is not None else "unavailable",
        "source": observed_model_source,
        "confidence": "runner-reported" if observed_model is not None else "requested-only",
        "requested_model": run["model"],
        "observed_model": observed_model,
    }
    if runner == "codex-cli" and session_id is not None and observed_model is None:
        runtime_identity = codex_identity_probe(
            session_id,
            codex_home or _codex_home(dict(os.environ)),
        )
        runtime_model = runtime_identity.get("observed_model")
        if isinstance(runtime_model, str) and runtime_model:
            observed_model = runtime_model
            observed_model_source = str(runtime_identity.get("source"))
        model_identity = {
            **runtime_identity,
            "requested_model": run["model"],
            "observed_model": observed_model,
        }
    model_identity["matches_requested"] = (
        observed_model == run["model"] if observed_model is not None else None
    )
    run_summary = {
        "format": EXECUTION_FORMAT,
        "run_key": run_key,
        "status": status,
        "error": error,
        "runner": runner,
        "requested_model": run["model"],
        "observed_model": observed_model,
        "observed_model_source": observed_model_source,
        "model_identity": model_identity,
        "auth": run["auth"],
        "session_id": session_id,
        "requested_session_id": requested_session,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_ms": round((time.monotonic() - started_clock) * 1000),
        "turns_completed": len(turn_summaries),
        "turns_planned": total_turns,
        "usage": usage_totals,
        "billing": {
            "basis": billing_basis,
            "reported_cost_usd": reported_cost,
            "meter_equivalent_usd": metered_cost,
            "dollar_cost_status": (
                "reported"
                if reported_cost is not None
                else "not-applicable-subscription"
                if run["auth"] == "subscription"
                else "unavailable"
            ),
            "max_cost_usd": run.get("max_cost_usd"),
        },
        "final_tree_hash": _tree_hash(workspace),
    }
    _write_execution(output_dir / "run.json", _json_bytes(run_summary))
    return run_summary


def execute_experiment(
    root: str | Path,
    *,
    allow_provider_calls: bool,
    allow_external_writes: bool = False,
    turn_timeout: int = DEFAULT_TURN_TIMEOUT,
    executable_overrides: dict[str, str] | None = None,
    driver: ProcessDriver | None = None,
    progress: ProgressCallback | None = None,
    codex_identity_probe: CodexIdentityProbe | None = None,
) -> ExecutionResult:
    """Execute every prepared run after a fresh fail-closed preflight."""
    if not allow_provider_calls:
        raise ExecutionRefused(
            "execution can contact providers and mutate isolated workspaces; "
            "pass --allow-provider-calls"
        )
    if turn_timeout < 1:
        raise ExecutionRefused("--turn-timeout must be at least 1 second")
    driver = driver or ProcessDriver()
    codex_identity_probe = codex_identity_probe or _codex_rollout_identity
    preflight = preflight_experiment(
        root, executable_overrides=executable_overrides, driver=driver
    )
    if not preflight.ok:
        raise ExecutionRefused("preflight failed: " + "; ".join(preflight.issues))
    resolved, marker, plan = _load_prepared(root)
    isolation = plan.get("isolation", {})
    external_write_templates = tuple(isolation.get("external_write_allowlist", []))
    if isolation.get("external_writes") == "allowlisted" and not allow_external_writes:
        raise ExecutionRefused(
            "this experiment authorizes writes outside isolated workspaces; "
            "pass --allow-external-writes to acknowledge the exact manifest allowlist"
        )
    external_by_run = {
        run_key: tuple(
            template.replace("{run_key}", run_key)
            for template in external_write_templates
        )
        for run_key in marker["run_keys"]
    }
    external_targets = [target for targets in external_by_run.values() for target in targets]
    if len(set(external_targets)) != len(external_targets):
        raise ExecutionRefused("external-write templates do not resolve to unique per-run paths")
    for target in external_targets:
        Path(target).mkdir()
    external_before = {target: _tree_hash(Path(target)) for target in external_targets}
    execution_root = resolved / "execution"
    if execution_root.exists() or execution_root.is_symlink():
        raise ExecutionRefused(
            f"execution evidence already exists at {execution_root}; it is never overwritten"
        )
    execution_root.mkdir()
    prompts = _prompt_map(resolved)
    executable_by_runner = {
        probe.runner: probe.executable for probe in preflight.probes if probe.executable
    }
    run_keys = list(marker["run_keys"])
    retry = _prepared_retry(resolved, marker)
    execute_run_keys = list(retry["retry_run_keys"]) if retry is not None else run_keys
    inherited_run_keys = (
        list(retry["inherited_run_keys"]) if retry is not None else []
    )
    max_parallel = min(
        len(execute_run_keys),
        max(1, int(plan.get("schedule", {}).get("max_parallel", 1))),
    )
    started_event = {
        "event": "execution_started",
        "at": _now(),
        "run_count": len(execute_run_keys),
        "max_parallel": max_parallel,
    }
    if retry is not None:
        started_event.update(
            {
                "total_run_count": len(run_keys),
                "inherited_run_count": len(inherited_run_keys),
            }
        )
    ledger: list[dict[str, Any]] = [started_event]
    _emit_progress(progress, started_event)
    summaries: dict[str, dict[str, Any]] = {}
    try:
        if retry is not None:
            inherited_root = resolved / "provenance" / "parent-execution" / "runs"
            for run_key in inherited_run_keys:
                source = inherited_root / run_key
                destination = execution_root / "runs" / run_key
                shutil.copytree(source, destination, symlinks=True)
                summary = _read_json(destination / "run.json")
                if summary.get("status") != "succeeded":
                    raise RunnerError(
                        f"retry provenance inherited non-successful run {run_key}"
                    )
                summaries[run_key] = summary
                ledger.append(
                    {
                        "event": "run_inherited",
                        "at": _now(),
                        "run_key": run_key,
                        "parent_plan_id": retry["parent_plan_id"],
                    }
                )
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            futures = {}
            for run_position, run_key in enumerate(execute_run_keys, start=1):
                run = _read_json(resolved / "runs" / run_key / "run.json")
                runner = run["runner"]
                future = pool.submit(
                    _run_one,
                    root=resolved,
                    execution_root=execution_root,
                    run_key=run_key,
                    executable=executable_by_runner[runner],
                    prompts=prompts,
                    timeout=turn_timeout,
                    driver=driver,
                    external_write_allowlist=external_by_run[run_key],
                    run_position=run_position,
                    run_count=len(execute_run_keys),
                    progress=progress,
                    codex_identity_probe=codex_identity_probe,
                )
                futures[future] = (run_key, run_position, run)
            for future in as_completed(futures):
                run_key, run_position, run = futures[future]
                try:
                    summary = future.result()
                except Exception as exc:  # evidence must survive one broken worker
                    summary = {
                        "format": EXECUTION_FORMAT,
                        "run_key": run_key,
                        "status": "failed",
                        "error": f"runner worker failed: {exc}",
                    }
                    run_output = execution_root / "runs" / run_key
                    run_output.mkdir(parents=True, exist_ok=True)
                    if not (run_output / "run.json").exists():
                        _write_execution(run_output / "run.json", _json_bytes(summary))
                summaries[run_key] = summary
                run_completed = {
                    "event": "run_completed",
                    "at": _now(),
                    "run_position": run_position,
                    "run_count": len(execute_run_keys),
                    "run_key": run_key,
                    "arm": run["arm"],
                    "runner": run["runner"],
                    "model": run["model"],
                    "status": summary["status"],
                    "duration_ms": summary.get("duration_ms"),
                }
                ledger.append(run_completed)
                _emit_progress(progress, run_completed)
        baseline_ok = True
        baseline_error = None
        try:
            verify_prepared(resolved)
        except ExperimentError as exc:
            baseline_ok = False
            baseline_error = str(exc)
        succeeded = sum(summary.get("status") == "succeeded" for summary in summaries.values())
        failed = len(run_keys) - succeeded
        if not baseline_ok:
            failed = max(1, failed)
        status = "completed" if failed == 0 and baseline_ok else "partial"
        external_after = {
            target: _tree_hash(Path(target)) for target in external_targets
        }
        completed_event = {
            "event": "execution_completed",
            "at": _now(),
            "status": status,
            "run_count": len(run_keys),
            "succeeded": succeeded,
            "failed": failed,
            "baseline_unchanged": baseline_ok,
        }
        ledger.append(completed_event)
        _write_execution(
            execution_root / "execution.jsonl",
            b"".join(
                json.dumps(event, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
                for event in ledger
            ),
        )
        execution_marker = {
            "format": EXECUTION_FORMAT,
            "status": status,
            "prepared_root": str(resolved),
            "plan_id": marker["plan_id"],
            "started_at": ledger[0]["at"],
            "completed_at": ledger[-1]["at"],
            "run_count": len(run_keys),
            "succeeded": succeeded,
            "failed": failed,
            "baseline_unchanged": baseline_ok,
            "baseline_error": baseline_error,
            "external_writes": {
                "policy": isolation.get("external_writes"),
                "templates": list(external_write_templates),
                "resolved_targets": external_by_run,
                "before": external_before,
                "after": external_after,
                "changed": sorted(
                    target
                    for target in external_before
                    if external_before[target] != external_after.get(target)
                ),
            },
            "billing": {
                "reported_cost_usd": sum(
                    float(summary.get("billing", {}).get("reported_cost_usd") or 0)
                    for summary in summaries.values()
                ),
                "api_cap_usd": sum(
                    float(summary.get("billing", {}).get("max_cost_usd") or 0)
                    for summary in summaries.values()
                    if summary.get("auth") == "api"
                ),
                "subscription_runs": sum(
                    summary.get("auth") == "subscription" for summary in summaries.values()
                ),
            },
            "runner_inventory": [probe.as_dict() for probe in preflight.probes],
            "run_status": {
                run_key: summaries[run_key].get("status", "failed") for run_key in run_keys
            },
            "credentials_captured": False,
            "environment_values_captured": False,
            "evidence_sha256": _execution_hashes(execution_root),
        }
        if retry is not None:
            execution_marker["retry"] = {
                "format": retry["format"],
                "parent_root": retry["parent_root"],
                "parent_plan_id": retry["parent_plan_id"],
                "parent_execution_marker_sha256": retry[
                    "parent_execution_marker_sha256"
                ],
                "inherited_run_keys": inherited_run_keys,
                "retried_run_keys": execute_run_keys,
            }
        _write_execution(execution_root / "execution.json", _json_bytes(execution_marker))
        _emit_progress(progress, completed_event)
        return ExecutionResult(
            root=execution_root,
            status=status,
            run_count=len(run_keys),
            succeeded=succeeded,
            failed=failed,
        )
    except Exception:
        # Keep partial evidence. Never erase provider output after execution starts.
        raise
