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

#: Every fenced block, whatever its language tag. An answer with no fence is
#: used whole.
_FENCE = re.compile(r"```[ \t]*[\w.+-]*[ \t]*\r?\n(.*?)\r?\n?```", re.S)

#: An answer with more blocks than this has its earliest ones skipped; the
#: final block is always run.
MAX_BLOCKS = 5

#: Environment variables the check's shell never sees.
_SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


@dataclass(frozen=True)
class CheckSpec:
    """``run`` is required; ``setup`` lays down fixtures first, in the same directory."""

    run: str
    setup: str | None = None
    timeout_s: int = DEFAULT_CHECK_TIMEOUT_S


@dataclass(frozen=True)
class BlockResult:
    """One fenced block, run on its own fixture."""

    index: int  # 1-based position of the block in the answer
    status: str  # "pass" | "fail" | "error"
    exit_code: int | None
    output: str
    code: str


@dataclass(frozen=True)
class CheckResult:
    """The verdict is the final block's; ``blocks`` is every block that ran, in order.

    A model that writes a wrong block, retracts it in prose and writes a second
    one has answered with the second - and the record shows it got there on the
    second try. Both facts travel to the judge and to the report.
    """

    status: str  # "pass" | "fail" | "error" | "not_applicable"
    exit_code: int | None = None
    output: str = ""
    code: str = ""
    blocks: tuple[BlockResult, ...] = ()

    @property
    def applicable(self) -> bool:
        return self.status != "not_applicable"

    @property
    def multi_block(self) -> bool:
        return len(self.blocks) > 1


NOT_APPLICABLE = CheckResult(status="not_applicable")


def extract_blocks(text: str) -> list[str]:
    """Every non-empty fenced block in order; an answer with none is one block, whole."""
    blocks = [m.group(1).strip("\r\n") for m in _FENCE.finditer(text)]
    blocks = [b for b in blocks if b.strip()]
    return blocks or [text.strip()]


def extract_code(text: str) -> str:
    """The answer's final code block: what the check's verdict is about."""
    return extract_blocks(text)[-1]


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


def _run_block(spec: CheckSpec, index: int, code: str) -> BlockResult:
    """One block, one fresh directory, one fixture. Never raises for anything the code did."""
    workdir = tempfile.mkdtemp(prefix="evalmine-check-")
    try:
        answer_path = os.path.join(workdir, "answer")
        with open(answer_path, "w", encoding="utf-8") as fh:
            fh.write(code + "\n")
        env = _child_env(answer_path, code)
        if spec.setup:
            setup = _sh(spec.setup, workdir, env, spec.timeout_s)
            if setup.exit_code != 0:
                return BlockResult(
                    index, "error", setup.exit_code, "setup failed:\n" + setup.output, code
                )
        outcome = _sh(spec.run, workdir, env, spec.timeout_s)
        if outcome.timed_out or outcome.exit_code is None:
            return BlockResult(index, "fail", None, outcome.output, code)
        status = "pass" if outcome.exit_code == 0 else "fail"
        return BlockResult(index, status, outcome.exit_code, outcome.output, code)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def run_check(spec: CheckSpec, text: str) -> CheckResult:
    """Run the check on every code block in the answer; the final block is the verdict."""
    blocks = extract_blocks(text)
    first_index = 1
    if len(blocks) > MAX_BLOCKS:
        first_index = len(blocks) - MAX_BLOCKS + 1
        blocks = blocks[-MAX_BLOCKS:]
    results = tuple(
        _run_block(spec, index, code) for index, code in enumerate(blocks, start=first_index)
    )
    final = results[-1]
    return CheckResult(final.status, final.exit_code, final.output, final.code, results)


def _verdict(status: str, exit_code: int | None) -> str:
    label = status.upper()
    return f"{label} (exit {exit_code})" if exit_code is not None else f"{label} (no exit code)"


def summarize(result: CheckResult) -> str:
    """``PASS (exit 0)``; with several blocks, ``PASS (exit 0) on block 2 of 2: FAIL, PASS``."""
    if not result.applicable:
        return "not checked"
    head = _verdict(result.status, result.exit_code)
    if not result.multi_block:
        return head
    n = len(result.blocks)
    sequence = ", ".join(b.status.upper() for b in result.blocks)
    return f"{head} on block {result.blocks[-1].index} of {n}: {sequence}"
