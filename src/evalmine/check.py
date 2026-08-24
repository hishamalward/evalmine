"""Execution checks: run the code an answer contains and record whether it worked.

Spec: docs/spec.md S6.6. A code task judged on prose alone rewards code that
reads well and does not run. A case may declare a ``check``: a shell snippet
that is handed the answer's code and exits 0 if the code satisfies the
contract. The result travels with the answer into the report, into the HTML
pair view (with the actual output, so a human labels on results rather than
on prose), and into the judge prompt, where a failed check cannot beat a
passed one.

Checks run locally, in a fresh temporary directory, under a timeout, with the
secrets stripped from the environment. They are never cached: a check is
cheap, and editing one must re-evaluate every answer without an API call.
What runs is model-written code - keep fixtures synthetic and never point a
check at anything you would mind losing.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass

DEFAULT_CHECK_TIMEOUT_S = 30

#: How much of a check's combined stdout/stderr is kept (the tail).
OUTPUT_TAIL_CHARS = 4000

CHECK_STATUSES = ("pass", "fail", "error", "not_applicable")

#: The first fenced block, whatever its language tag. An answer with no fence
#: is used whole.
_FENCE = re.compile(r"```[ \t]*[\w.+-]*[ \t]*\r?\n(.*?)\r?\n?```", re.S)

#: Environment variables the check's shell never sees.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass(frozen=True)
class CheckSpec:
    """``run`` is required; ``setup`` lays down fixtures first, in the same directory."""

    run: str
    setup: str | None = None
    timeout_s: int = DEFAULT_CHECK_TIMEOUT_S


@dataclass(frozen=True)
class CheckResult:
    status: str  # "pass" | "fail" | "error" | "not_applicable"
    exit_code: int | None = None
    output: str = ""
    code: str = ""

    @property
    def applicable(self) -> bool:
        return self.status != "not_applicable"


NOT_APPLICABLE = CheckResult(status="not_applicable")


def extract_code(text: str) -> str:
    """The first fenced block if there is one, otherwise the whole answer, stripped."""
    match = _FENCE.search(text)
    if match:
        return match.group(1).strip("\r\n")
    return text.strip()


def _looks_secret(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_MARKERS)


def _child_env(answer_path: str, code: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _looks_secret(k)}
    env["ANSWER"] = answer_path
    env["ANSWER_TEXT"] = code
    env["EVALMINE_CHECK"] = "1"
    return env


@dataclass(frozen=True)
class _Outcome:
    exit_code: int | None
    output: str
    timed_out: bool = False


def _sh(script: str, cwd: str, env: dict[str, str], timeout_s: int) -> _Outcome:
    try:
        proc = subprocess.run(
            ["bash", "-c", script],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as exc:
        partial = _combine(_as_text(exc.stdout), _as_text(exc.stderr))
        return _Outcome(None, f"timed out after {timeout_s}s\n{partial}".rstrip(), True)
    except OSError as exc:  # bash missing, permissions
        return _Outcome(None, f"could not start bash: {exc}")
    return _Outcome(proc.returncode, _combine(proc.stdout, proc.stderr))


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _combine(stdout: str, stderr: str) -> str:
    parts = []
    if stdout:
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append("[stderr]\n" + stderr.rstrip("\n"))
    text = "\n".join(parts)
    if len(text) > OUTPUT_TAIL_CHARS:
        text = "...\n" + text[-OUTPUT_TAIL_CHARS:]
    return text


def run_check(spec: CheckSpec, text: str) -> CheckResult:
    """Run one check against one answer. Never raises for anything the code did."""
    code = extract_code(text)
    workdir = tempfile.mkdtemp(prefix="evalmine-check-")
    try:
        answer_path = os.path.join(workdir, "answer")
        with open(answer_path, "w", encoding="utf-8") as fh:
            fh.write(code + "\n")
        env = _child_env(answer_path, code)
        if spec.setup:
            setup = _sh(spec.setup, workdir, env, spec.timeout_s)
            if setup.exit_code != 0:
                return CheckResult(
                    "error", setup.exit_code, "setup failed:\n" + setup.output, code
                )
        outcome = _sh(spec.run, workdir, env, spec.timeout_s)
        if outcome.timed_out or outcome.exit_code is None:
            return CheckResult("fail", None, outcome.output, code)
        status = "pass" if outcome.exit_code == 0 else "fail"
        return CheckResult(status, outcome.exit_code, outcome.output, code)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def summarize(result: CheckResult) -> str:
    """One line: ``PASS (exit 0)`` / ``FAIL (exit 1)`` / ``FAIL (timed out)`` / ``ERROR``."""
    if not result.applicable:
        return "not checked"
    label = result.status.upper()
    if result.exit_code is None:
        return f"{label} (no exit code)"
    return f"{label} (exit {result.exit_code})"
