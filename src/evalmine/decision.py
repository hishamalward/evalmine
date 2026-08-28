"""Episode judging, human-label calibration, and decision reports.

The v2 decision layer consumes immutable execution evidence.  It never changes a
workspace and keeps judge calls in a separate create-once envelope so a report can
be regenerated from inspectable inputs without rewriting provider evidence.
"""

# ruff: noqa: E501 -- self-contained HTML is intentionally kept readable inline

from __future__ import annotations

import hashlib
import html
import json
import os
import stat
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adapters import Request, build_adapter, call_with_retries, split_model
from .adapters.base import Adapter, Response
from .experiment import ExperimentError
from .experiment_report import (
    CHOICES,
    LABEL_FORMAT,
    RANKING_LABEL_FORMAT,
    _api_list_price_equivalent,
    build_experiment_report_data,
)
from .metrics import (
    bootstrap_ci,
    calibration_status,
    call_cost,
    cohens_kappa,
    estimate_tokens,
    judge_category,
    per_task_agreement,
    score_pair,
    seed_from_suite_hash,
    win_rate,
)
from .prices import PriceTable, PriceTableError, load_price_table
from .runner import (
    ProcessDriver,
    RunnerError,
    _child_env,
    _codex_home,
    _codex_rollout_identity,
    _normalize_events,
    _parse_events,
    _redact_secrets,
    _resolve_executable,
)
from .workspace import _path_content_hash, _write_once, verify_prepared

JUDGING_FORMAT = "evalmine-episode-judging-v1"
DECISION_FORMAT = "evalmine-episode-decision-v1"
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["winner", "reason"],
    "properties": {
        "winner": {"type": "string", "enum": ["1", "2", "tie"]},
        "reason": {"type": "string"},
    },
}
NWAY_JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranking", "reason"],
    "properties": {
        "ranking": {
            "type": "array",
            "minItems": 2,
            "items": {"type": "string"},
        },
        "reason": {"type": "string"},
    },
}
JUDGE_SYSTEM = (
    "You are comparing two complete agent trajectories from the same controlled "
    "episode. Identity, provider, model, price, and arm labels are intentionally "
    "hidden. Apply the objectives literally and return JSON only."
)
NWAY_JUDGE_SYSTEM = (
    "You are ranking complete agent trajectories from the same controlled episode. "
    "Identity, provider, model, price, and arm labels are intentionally hidden. Apply "
    "the objectives literally, rank every outcome exactly once from best to worst, and "
    "return JSON only."
)
DEFAULT_JUDGE_MAX_TOKENS = 700
DEFAULT_MIN_KAPPA = 0.40
DEFAULT_MIN_LABELS = 10


class DecisionError(ExperimentError):
    """Judging or decision evidence is invalid, incomplete, or unsafe."""


class JudgeRefused(DecisionError):
    """A judge call was refused before a provider could be contacted."""


@dataclass(frozen=True)
class EpisodeJudgeCall:
    text: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    raw: str = ""
    observed_model: str | None = None
    observed_model_source: str | None = None
    meter_equivalent_usd: float | None = None
    model_usage: dict[str, Any] | None = None
    auxiliary_models: tuple[str, ...] = ()


@dataclass(frozen=True)
class JudgingResult:
    root: Path
    pair_count: int
    call_count: int
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": JUDGING_FORMAT,
            "root": str(self.root),
            "pair_count": self.pair_count,
            "call_count": self.call_count,
            "status": self.status,
        }


@dataclass(frozen=True)
class DecisionResult:
    root: Path
    html: Path
    labelled_pairs: int
    judged_pairs: int
    headline_eligible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": DECISION_FORMAT,
            "root": str(self.root),
            "html": str(self.html),
            "labelled_pairs": self.labelled_pairs,
            "judged_pairs": self.judged_pairs,
            "headline_eligible": self.headline_eligible,
            "provider_calls": False,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DecisionError(f"cannot read {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise DecisionError(f"{path} is not a JSON object")
    return value


def _hashes(root: Path, marker_name: str) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _path_content_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != marker_name
    }


def _write_read_only(path: Path, content: bytes) -> None:
    _write_once(path, content)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _run_summary(run: dict[str, Any]) -> str:
    turns = []
    for turn in run.get("turns", []):
        tool_count = len(turn.get("tools", []))
        turns.append(
            f"TURN {turn.get('turn')} ({turn.get('status')}, {turn.get('duration_ms')} ms)\n"
            f"Prompt:\n{turn.get('prompt', '')}\n"
            f"Tool events: {tool_count}\nFinal response:\n{turn.get('final', '')}"
        )
    validators = []
    for result in run.get("validators", []):
        validators.append(
            f"{result.get('id')} [{result.get('type')}]: {result.get('status')}"
        )
        if result.get("patch_text"):
            validators.append("Patch:\n" + str(result["patch_text"]))
        if result.get("stdout_text") or result.get("stderr_text"):
            validators.append(
                "Command output:\n"
                + str(result.get("stdout_text", ""))
                + "\n"
                + str(result.get("stderr_text", ""))
            )
    return (
        f"Execution: {run.get('execution_status')}\n"
        f"Objective checks: {run.get('validation_verdict')}\n\n"
        + "\n\n".join(turns)
        + "\n\nVALIDATORS\n"
        + ("\n".join(validators) or "not run")
    )


def build_episode_judge_prompt(
    pair: dict[str, Any], objectives: Sequence[str], *, order: int
) -> str:
    """Build one identity-blind prompt; order 2 swaps complete trajectories."""
    if order not in {1, 2}:
        raise ValueError("judge order must be 1 or 2")
    ordered = sorted((pair["a"], pair["b"]), key=lambda item: item["run_key"])
    one, two = ordered if order == 1 else tuple(reversed(ordered))
    objective_text = "\n".join(f"- {item}" for item in objectives)
    return f"""Compare two outcomes from the same episode. Consider the entire trajectory,
workspace changes, objective checks, and final response. A failed objective check is evidence,
not an automatic loss unless the objectives make it decisive. Ties are valid.

=== OBJECTIVES ===
{objective_text}

=== OUTCOME 1 ===
{_run_summary(one)}

=== OUTCOME 2 ===
{_run_summary(two)}

=== VERDICT ===
Return exactly {{"winner":"1"|"2"|"tie","reason":"one concise reason"}}.
"""


def build_episode_ranking_prompt(
    ranking: dict[str, Any], objectives: Sequence[str]
) -> str:
    """Build one identity-blind prompt that ranks every arm in a block together."""
    objective_text = "\n".join(f"- {item}" for item in objectives)
    outcomes = "\n\n".join(
        f"=== OUTCOME {outcome['label']} ===\n{_run_summary(outcome['run'])}"
        for outcome in ranking["outcomes"]
    )
    labels = [str(outcome["label"]) for outcome in ranking["outcomes"]]
    return f"""Rank all outcomes from the same episode together. Consider the complete
trajectory, workspace changes, objective checks, and final response. A failed objective
check is evidence, not an automatic exclusion. Return a strict best-to-worst ordering.

=== OBJECTIVES ===
{objective_text}

{outcomes}

=== RANKING ===
Return exactly one JSON object with "ranking" containing each of {json.dumps(labels)}
exactly once from best to worst, plus "reason" with one concise explanation.
"""


def _parse_ranking_call(call: EpisodeJudgeCall, ranking: dict[str, Any]) -> dict[str, Any]:
    try:
        parsed = json.loads(call.text)
    except json.JSONDecodeError:
        parsed = None
    expected = [str(outcome["label"]) for outcome in ranking["outcomes"]]
    order = parsed.get("ranking") if isinstance(parsed, dict) else None
    valid = (
        isinstance(order, list)
        and all(isinstance(label, str) for label in order)
        and len(order) == len(expected)
        and len(set(order)) == len(order)
        and set(order) == set(expected)
        and isinstance(parsed.get("reason"), str)
    )
    return {
        "ranking_id": ranking["ranking_id"],
        "status": "parsed" if valid else "unparseable",
        "ranking": order if valid else None,
        "ranked_run_keys": (
            [ranking["run_key_by_label"][label] for label in order] if valid else None
        ),
        "reason": parsed["reason"] if valid else None,
        "latency_ms": call.latency_ms,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "cost_usd": call.cost_usd,
        "observed_model": call.observed_model,
        "observed_model_source": call.observed_model_source,
        "meter_equivalent_usd": call.meter_equivalent_usd,
        "model_usage": call.model_usage,
        "auxiliary_models": list(call.auxiliary_models),
    }


def _parse_judge_call(call: EpisodeJudgeCall, order: int) -> dict[str, Any]:
    try:
        parsed = json.loads(call.text)
    except json.JSONDecodeError:
        parsed = None
    valid = (
        isinstance(parsed, dict)
        and parsed.get("winner") in {"1", "2", "tie"}
        and isinstance(parsed.get("reason"), str)
    )
    if not valid:
        return {
            "order": order,
            "status": "unparseable",
            "winner": None,
            "verdict": None,
            "reason": None,
            "latency_ms": call.latency_ms,
            "input_tokens": call.input_tokens,
            "output_tokens": call.output_tokens,
            "cost_usd": call.cost_usd,
            "observed_model": call.observed_model,
            "observed_model_source": call.observed_model_source,
            "meter_equivalent_usd": call.meter_equivalent_usd,
            "model_usage": call.model_usage,
            "auxiliary_models": list(call.auxiliary_models),
        }
    winner = parsed["winner"]
    if winner == "tie":
        verdict = "tie"
    elif order == 1:
        verdict = "baseline" if winner == "1" else "candidate"
    else:
        verdict = "candidate" if winner == "1" else "baseline"
    return {
        "order": order,
        "status": "parsed",
        "winner": winner,
        "verdict": verdict,
        "reason": parsed["reason"],
        "latency_ms": call.latency_ms,
        "input_tokens": call.input_tokens,
        "output_tokens": call.output_tokens,
        "cost_usd": call.cost_usd,
        "observed_model": call.observed_model,
        "observed_model_source": call.observed_model_source,
        "meter_equivalent_usd": call.meter_equivalent_usd,
        "model_usage": call.model_usage,
        "auxiliary_models": list(call.auxiliary_models),
    }


class _ApiJudge:
    def __init__(
        self,
        model: str,
        *,
        adapter: Adapter,
        price_table: PriceTable | None,
        max_tokens: int,
        schema: dict[str, Any] = JUDGE_SCHEMA,
        system: str = JUDGE_SYSTEM,
    ) -> None:
        self.provider, self.model_id = split_model(model)
        self.adapter = adapter
        self.row = price_table.get(model) if price_table is not None else None
        self.max_tokens = max_tokens
        self.schema = schema
        self.system = system

    def __call__(self, prompt: str) -> EpisodeJudgeCall:
        response: Response = call_with_retries(
            self.adapter,
            Request(
                model_id=self.model_id,
                system=self.system,
                prompt=prompt,
                max_tokens=self.max_tokens,
                schema=self.schema,
            ),
        )
        cost = (
            call_cost(
                self.row,
                response.input_tokens,
                response.output_tokens,
                response.cached_input_tokens,
            )
            if self.row is not None
            else 0.0
        )
        return EpisodeJudgeCall(
            text=response.text,
            latency_ms=response.latency_ms,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            cost_usd=cost,
            observed_model=self.model_id,
            observed_model_source="api-response",
            meter_equivalent_usd=cost,
        )


class _SubscriptionJudge:
    def __init__(
        self,
        runner: str,
        model: str,
        cwd: Path,
        schema_path: Path,
        *,
        driver: ProcessDriver,
        executable_overrides: dict[str, str] | None,
        schema: dict[str, Any] = JUDGE_SCHEMA,
        system: str = JUDGE_SYSTEM,
    ) -> None:
        executable = _resolve_executable(runner, executable_overrides)
        if executable is None:
            raise JudgeRefused(f"{runner} executable is unavailable")
        self.runner = runner
        self.model = model
        self.cwd = cwd
        self.schema_path = schema_path
        self.driver = driver
        self.executable = executable
        self.schema = schema
        self.system = system
        self.call_count = 0

    @staticmethod
    def _failure_detail(stdout: str, stderr: str) -> tuple[str, list[str], list[str]]:
        events, malformed = _parse_events(stdout)
        messages: list[str] = []
        for event in events:
            if event.get("type") not in {"error", "turn.failed"}:
                continue
            error = event.get("error")
            if isinstance(error, dict):
                error = error.get("message") or error.get("detail")
            message = event.get("message") or error
            if isinstance(message, str) and message.strip():
                messages.append(message.strip())
        if stderr.strip():
            messages.append(stderr.strip())
        if not messages and malformed:
            messages.append(malformed[-1])
        detail = " | ".join(messages) if messages else "no diagnostic text captured"
        return detail[:1000], [str(event.get("type", "unknown")) for event in events], malformed

    def _command(self) -> list[str]:
        if self.runner == "claude-code":
            return [
                self.executable,
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--model",
                self.model,
                "--permission-mode",
                "plan",
                "--safe-mode",
                "--strict-mcp-config",
                "--mcp-config",
                '{"mcpServers":{}}',
                "--disable-slash-commands",
                "--no-chrome",
                "--no-session-persistence",
                "--json-schema",
                json.dumps(self.schema, separators=(",", ":")),
            ]
        if self.runner == "codex-cli":
            return [
                self.executable,
                "--ask-for-approval",
                "never",
                "exec",
                "--sandbox",
                "read-only",
                "--cd",
                str(self.cwd),
                "--skip-git-repo-check",
                "--json",
                "--model",
                self.model,
                "--ignore-user-config",
                "--ignore-rules",
                "--ephemeral",
                "--output-schema",
                str(self.schema_path),
                "-",
            ]
        if self.runner == "gemini-cli":
            return [
                self.executable,
                "--prompt",
                "",
                "--output-format",
                "stream-json",
                "--model",
                self.model,
                "--sandbox",
                "--approval-mode",
                "plan",
                "--extensions",
                "none",
            ]
        raise JudgeRefused(f"runner {self.runner!r} cannot judge episodes")

    def __call__(self, prompt: str) -> EpisodeJudgeCall:
        self.call_count += 1
        started = time.monotonic()
        result = self.driver.run(
            self._command(),
            cwd=self.cwd,
            input_text=self.system + "\n\n" + prompt,
            timeout=1800,
            env=_child_env({}),
        )
        stdout = _redact_secrets(result.stdout)
        stderr = _redact_secrets(result.stderr)
        if result.returncode != 0 or result.timed_out:
            detail, event_types, malformed = self._failure_detail(stdout, stderr)
            stem = self.schema_path.parent / "calls" / f"{self.call_count:04d}"
            _write_read_only(stem.with_suffix(".raw.txt"), stdout.encode())
            _write_read_only(stem.with_suffix(".stderr.txt"), stderr.encode())
            _write_read_only(
                stem.with_suffix(".failure.json"),
                _json_bytes(
                    {
                        "format": "evalmine-judge-call-failure-v1",
                        "created_at": _now(),
                        "runner": self.runner,
                        "requested_model": self.model,
                        "exit_code": result.returncode,
                        "timed_out": result.timed_out,
                        "duration_ms": result.duration_ms,
                        "detail": detail,
                        "event_types": event_types,
                        "malformed_event_lines": malformed,
                    }
                ),
            )
            raise RunnerError(
                f"episode judge {self.runner} exited {result.returncode}: "
                f"{detail}"
            )
        events, malformed = _parse_events(stdout)
        if malformed:
            raise RunnerError("episode judge emitted malformed JSONL")
        session, observed, final, _tools, normalized = _normalize_events(
            self.runner, events
        )
        observed_source = "runner-event" if observed is not None else None
        if self.runner == "codex-cli" and session is not None and observed is None:
            runtime_identity = _codex_rollout_identity(
                session,
                _codex_home(dict(os.environ)),
            )
            runtime_model = runtime_identity.get("observed_model")
            if isinstance(runtime_model, str) and runtime_model:
                observed = runtime_model
                observed_source = str(runtime_identity.get("source"))
        if not final:
            raise RunnerError("episode judge emitted no final response")
        usage = normalized["usage"]
        return EpisodeJudgeCall(
            text=final,
            raw=stdout,
            latency_ms=result.duration_ms or round((time.monotonic() - started) * 1000),
            input_tokens=usage.get("input_tokens") or usage.get("prompt_tokens"),
            output_tokens=usage.get("output_tokens") or usage.get("completion_tokens"),
            cost_usd=None,
            observed_model=observed,
            observed_model_source=observed_source,
            meter_equivalent_usd=(
                float(usage["reported_cost_usd"])
                if isinstance(usage.get("reported_cost_usd"), (int, float))
                else None
            ),
            model_usage=normalized.get("model_usage") or None,
            auxiliary_models=tuple(normalized.get("auxiliary_models") or ()),
        )


def _estimate_api_judging(
    prompts: Sequence[str],
    model: str,
    table: PriceTable,
    max_tokens: int,
    *,
    system: str = JUDGE_SYSTEM,
) -> float:
    row = table.get(model)
    return sum(
        (estimate_tokens(prompt) + estimate_tokens(system))
        / 1e6
        * row.input_per_mtok
        + max_tokens / 1e6 * row.output_per_mtok
        for prompt in prompts
    )


def judge_experiment(
    root: str | Path,
    *,
    allow_provider_calls: bool,
    max_cost_usd: float | None = None,
    fake: bool = False,
    prices_path: str | Path | None = None,
    caller: Callable[[str], EpisodeJudgeCall] | None = None,
    driver: ProcessDriver | None = None,
    executable_overrides: dict[str, str] | None = None,
    ranking_style: str | None = None,
    runner_override: str | None = None,
    model_override: str | None = None,
    output: str | Path | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> JudgingResult:
    """Run the preregistered pairwise or N-way episode judging protocol."""
    if not allow_provider_calls and caller is None and not fake:
        raise JudgeRefused("judging can contact a provider; pass --allow-provider-calls")
    prepared = Path(verify_prepared(root)["root"])
    report = build_experiment_report_data(prepared, ranking_style=ranking_style)
    config = report["judge"]
    if not config.get("enabled"):
        raise JudgeRefused("this experiment has evaluation.judge.enabled=false")
    ranking_style = str(report.get("ranking_style", "pairwise"))
    style_from_plan = report.get("ranking_style_source") == "plan"
    if ranking_style == "pairwise":
        if style_from_plan and (not config.get("pairwise") or not config.get("position_swap")):
            raise JudgeRefused("pairwise judging requires pairwise=true and position_swap=true")
        schema = JUDGE_SCHEMA
        system = JUDGE_SYSTEM
        prompts = [
            build_episode_judge_prompt(pair, report["objectives"], order=order)
            for pair in report["pairs"]
            for order in (1, 2)
        ]
    elif ranking_style == "n-way":
        if style_from_plan and (config.get("pairwise") or config.get("position_swap")):
            raise JudgeRefused("n-way judging requires pairwise=false and position_swap=false")
        schema = NWAY_JUDGE_SCHEMA
        system = NWAY_JUDGE_SYSTEM
        prompts = [
            build_episode_ranking_prompt(ranking, report["objectives"])
            for ranking in report["rankings"]
        ]
    else:
        raise JudgeRefused(f"unknown ranking style {ranking_style!r}")
    judging_root = Path(output).resolve() if output is not None else prepared / "judging"
    if judging_root.exists() or judging_root.is_symlink():
        raise JudgeRefused(f"judging evidence already exists at {judging_root}")
    runner = str(runner_override or config.get("runner") or "")
    model = str(model_override or config.get("model") or "")
    max_tokens = int(config.get("max_tokens", DEFAULT_JUDGE_MAX_TOKENS))
    price_table: PriceTable | None = None
    estimate: float | None = None
    if caller is None:
        if not runner or not model:
            raise JudgeRefused("enabled judging requires evaluation.judge.runner and model")
        if fake:
            provider, _ = split_model(model) if "/" in model else ("fake", model)
            caller = _ApiJudge(
                model if "/" in model else f"fake/{model}",
                adapter=build_adapter(provider, fake=True),
                price_table=None,
                max_tokens=max_tokens,
                schema=schema,
                system=system,
            )
            runner = "fake"
        elif runner == "api-prompt":
            if max_cost_usd is None:
                max_cost_usd = config.get("max_cost_usd")
            if max_cost_usd is None or float(max_cost_usd) <= 0:
                raise JudgeRefused("API judging requires a positive --max-cost")
            price_table = load_price_table(prices_path)
            estimate = _estimate_api_judging(
                prompts, model, price_table, max_tokens, system=system
            )
            if estimate > float(max_cost_usd):
                raise JudgeRefused(
                    f"judging estimate ${estimate:.4f} exceeds ${float(max_cost_usd):.2f}; "
                    "nothing was spent"
                )
            provider, _model_id = split_model(model)
            caller = _ApiJudge(
                model,
                adapter=build_adapter(provider),
                price_table=price_table,
                max_tokens=max_tokens,
                schema=schema,
                system=system,
            )
        else:
            schema_path = judging_root / "judge-schema.json"
            caller = _SubscriptionJudge(
                runner,
                model,
                prepared,
                schema_path,
                driver=driver or ProcessDriver(),
                executable_overrides=executable_overrides,
                schema=schema,
                system=system,
            )
            judging_root.parent.mkdir(parents=True, exist_ok=True)
            judging_root.mkdir()
            _write_read_only(schema_path, _json_bytes(schema))
    if not judging_root.exists():
        judging_root.parent.mkdir(parents=True, exist_ok=True)
        judging_root.mkdir()
    rows: list[dict[str, Any]] = []
    ranking_rows: list[dict[str, Any]] = []
    total_cost = 0.0
    cost_complete = True
    call_index = 0
    if progress is not None:
        progress(
            {
                "event": "judging_started",
                "ranking_style": ranking_style,
                "call_count": len(prompts),
                "model": model,
            }
        )
    if ranking_style == "pairwise":
        for pair in report["pairs"]:
            ordered_keys = sorted((pair["a_run_key"], pair["b_run_key"]))
            passes = []
            for order in (1, 2):
                call_index += 1
                prompt = build_episode_judge_prompt(pair, report["objectives"], order=order)
                if progress is not None:
                    progress(
                        {
                            "event": "judge_call_started",
                            "call_position": call_index,
                            "call_count": len(prompts),
                            "ranking_style": ranking_style,
                            "model": model,
                        }
                    )
                call = caller(prompt)
                parsed = _parse_judge_call(call, order)
                raw_path = f"calls/{call_index:04d}.raw.txt"
                _write_read_only(
                    judging_root / raw_path,
                    _redact_secrets(call.raw or call.text).encode(),
                )
                parsed["raw"] = raw_path
                parsed["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                passes.append(parsed)
                if progress is not None:
                    progress(
                        {
                            "event": "judge_call_completed",
                            "call_position": call_index,
                            "call_count": len(prompts),
                            "ranking_style": ranking_style,
                            "model": model,
                            "status": parsed["status"],
                            "duration_ms": call.latency_ms,
                        }
                    )
                if call.cost_usd is None:
                    cost_complete = False
                else:
                    total_cost += call.cost_usd
            if all(item["status"] == "parsed" for item in passes):
                score, category = score_pair(passes[0]["verdict"], passes[1]["verdict"])
                status = "scored"
            else:
                score, category, status = None, None, "excluded"
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "block": pair["block"],
                    "episode": pair["episode"],
                    "repeat": pair["repeat"],
                    "baseline_run_key": ordered_keys[0],
                    "candidate_run_key": ordered_keys[1],
                    "status": status,
                    "score": score,
                    "category": category,
                    "passes": passes,
                }
            )
    else:
        for ranking in report["rankings"]:
            call_index += 1
            prompt = build_episode_ranking_prompt(ranking, report["objectives"])
            if progress is not None:
                progress(
                    {
                        "event": "judge_call_started",
                        "call_position": call_index,
                        "call_count": len(prompts),
                        "ranking_style": ranking_style,
                        "model": model,
                    }
                )
            call = caller(prompt)
            parsed = _parse_ranking_call(call, ranking)
            raw_path = f"calls/{call_index:04d}.raw.txt"
            _write_read_only(
                judging_root / raw_path,
                _redact_secrets(call.raw or call.text).encode(),
            )
            parsed["raw"] = raw_path
            parsed["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
            ranking_rows.append(parsed)
            if progress is not None:
                progress(
                    {
                        "event": "judge_call_completed",
                        "call_position": call_index,
                        "call_count": len(prompts),
                        "ranking_style": ranking_style,
                        "model": model,
                        "status": parsed["status"],
                        "duration_ms": call.latency_ms,
                    }
                )
            if call.cost_usd is None:
                cost_complete = False
            else:
                total_cost += call.cost_usd
            positions = {
                run_key: index
                for index, run_key in enumerate(parsed.get("ranked_run_keys") or [])
            }
            for pair in (item for item in report["pairs"] if item["block"] == ranking["block"]):
                ordered_keys = sorted((pair["a_run_key"], pair["b_run_key"]))
                if parsed["status"] == "parsed":
                    score = float(positions[ordered_keys[1]] < positions[ordered_keys[0]])
                    category = judge_category(score)
                    status = "scored"
                else:
                    score, category, status = None, None, "excluded"
                rows.append(
                    {
                        "pair_id": pair["pair_id"],
                        "block": pair["block"],
                        "episode": pair["episode"],
                        "repeat": pair["repeat"],
                        "baseline_run_key": ordered_keys[0],
                        "candidate_run_key": ordered_keys[1],
                        "status": status,
                        "score": score,
                        "category": category,
                        "passes": [],
                        "source_ranking_id": ranking["ranking_id"],
                    }
                )
    payload = {
        "format": JUDGING_FORMAT,
        "created_at": _now(),
        "prepared_root": str(prepared),
        "plan_id": report["plan_id"],
        "runner": runner,
        "model": model,
        "runner_source": "operator-override" if runner_override else "plan",
        "model_source": "operator-override" if model_override else "plan",
        "ranking_style": ranking_style,
        "planned_ranking_style": report["planned_ranking_style"],
        "ranking_style_source": report["ranking_style_source"],
        "artifact_location": "operator-output" if output is not None else "prepared-default",
        "billing_basis": "api" if runner == "api-prompt" else "subscription-or-local",
        "estimated_cost_usd": estimate,
        "cost_usd": total_cost if cost_complete else None,
        "cost_complete": cost_complete,
        "pair_count": len(rows),
        "call_count": call_index,
        "pairs": rows,
        "ranking_count": len(ranking_rows),
        "rankings": ranking_rows,
    }
    _write_read_only(judging_root / "pairs.json", _json_bytes(payload))
    marker = {
        **{key: payload[key] for key in ("format", "created_at", "prepared_root", "plan_id")},
        "status": "completed",
        "pair_count": len(rows),
        "ranking_count": len(ranking_rows),
        "ranking_style": ranking_style,
        "planned_ranking_style": report["planned_ranking_style"],
        "ranking_style_source": report["ranking_style_source"],
        "call_count": call_index,
        "evidence_sha256": _hashes(judging_root, "judging.json"),
    }
    _write_read_only(judging_root / "judging.json", _json_bytes(marker))
    if progress is not None:
        progress(
            {
                "event": "judging_completed",
                "ranking_style": ranking_style,
                "call_count": call_index,
                "pair_count": len(rows),
                "status": "completed",
                "model": model,
            }
        )
    return JudgingResult(judging_root, len(rows), call_index, "completed")


def verify_judging(
    root: str | Path, judging_path: str | Path | None = None
) -> dict[str, Any]:
    prepared = Path(verify_prepared(root)["root"])
    judging_root = (
        Path(judging_path).resolve() if judging_path is not None else prepared / "judging"
    )
    marker = _read_json(judging_root / "judging.json")
    if marker.get("format") != JUDGING_FORMAT or marker.get("prepared_root") != str(prepared):
        raise DecisionError("judging marker does not match this prepared experiment")
    if marker.get("evidence_sha256") != _hashes(judging_root, "judging.json"):
        raise DecisionError("judging evidence changed after creation")
    return {
        "ok": True,
        "format": JUDGING_FORMAT,
        "root": str(judging_root),
        "status": marker.get("status"),
        "pair_count": marker.get("pair_count"),
        "call_count": marker.get("call_count"),
    }


def _load_human_labels(
    label_paths: Sequence[str | Path], report: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = {item["pair_id"]: item for item in report["pairs"]}
    rankings = {item["ranking_id"]: item for item in report.get("rankings", [])}
    labels: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for label_path in label_paths:
        path = Path(label_path).resolve()
        payload = _read_json(path)
        label_format = payload.get("format")
        if label_format not in {LABEL_FORMAT, RANKING_LABEL_FORMAT}:
            raise DecisionError(f"{path}: unknown label format")
        if payload.get("plan_id") != report["plan_id"]:
            raise DecisionError(f"{path}: labels belong to a different plan")
        annotator = str(payload.get("annotator") or path.stem)
        sources.append(
            {
                "path": str(path),
                "sha256": _path_content_hash(path),
                "annotator": annotator,
            }
        )
        if label_format == RANKING_LABEL_FORMAT:
            rows = payload.get("rankings")
            if not isinstance(rows, list):
                raise DecisionError(f"{path}: rankings must be a list")
            for row in rows:
                if not isinstance(row, dict):
                    raise DecisionError(f"{path}: malformed ranking row")
                ranking_id = str(row.get("ranking_id"))
                ranking = rankings.get(ranking_id)
                if ranking is None:
                    raise DecisionError(f"{path}: unknown ranking {ranking_id!r}")
                unclear = bool(row.get("unclear"))
                order = row.get("order")
                expected_labels = set(ranking["run_key_by_label"])
                if not isinstance(order, list) or any(
                    not isinstance(label, str) for label in order
                ):
                    raise DecisionError(f"{path}: malformed order for {ranking_id}")
                if unclear:
                    if order:
                        raise DecisionError(f"{path}: unclear ranking {ranking_id} has an order")
                    positions: dict[str, int] = {}
                else:
                    if len(order) != len(expected_labels) or set(order) != expected_labels:
                        raise DecisionError(
                            f"{path}: ranking {ranking_id} must contain every outcome exactly once"
                        )
                    ranked_run_keys = [ranking["run_key_by_label"][label] for label in order]
                    provided_run_keys = row.get("ranked_run_keys")
                    if provided_run_keys is not None and provided_run_keys != ranked_run_keys:
                        raise DecisionError(
                            f"{path}: ranked-run mapping is inconsistent for {ranking_id}"
                        )
                    positions = {
                        run_key: index for index, run_key in enumerate(ranked_run_keys)
                    }
                for pair in (
                    item for item in report["pairs"] if item["block"] == ranking["block"]
                ):
                    pair_id = pair["pair_id"]
                    key = (annotator, pair_id)
                    if key in seen:
                        raise DecisionError(
                            f"{path}: duplicate label for {pair_id} by {annotator}"
                        )
                    seen.add(key)
                    baseline, candidate = sorted((pair["a_run_key"], pair["b_run_key"]))
                    category = (
                        "unclear"
                        if unclear
                        else "baseline"
                        if positions[baseline] < positions[candidate]
                        else "candidate"
                    )
                    labels.append(
                        {
                            "pair_id": pair_id,
                            "ranking_id": ranking_id,
                            "episode": pair["episode"],
                            "annotator": annotator,
                            "choice": "ranking",
                            "category": category,
                            "note": row.get("note"),
                            "identities_revealed": bool(row.get("identities_revealed")),
                        }
                    )
            continue
        rows = payload.get("labels")
        if not isinstance(rows, list):
            raise DecisionError(f"{path}: labels must be a list")
        for row in rows:
            if not isinstance(row, dict) or row.get("choice") not in CHOICES:
                raise DecisionError(f"{path}: malformed label row")
            pair_id = str(row.get("pair_id"))
            pair = pairs.get(pair_id)
            if pair is None:
                raise DecisionError(f"{path}: unknown pair {pair_id!r}")
            key = (annotator, pair_id)
            if key in seen:
                raise DecisionError(f"{path}: duplicate label for {pair_id} by {annotator}")
            seen.add(key)
            expected = pair["preferred_run_by_choice"][row["choice"]]
            if row.get("preferred_run_key") not in {None, expected}:
                raise DecisionError(f"{path}: preferred-run mapping is inconsistent for {pair_id}")
            ordered = sorted((pair["a_run_key"], pair["b_run_key"]))
            if expected == "tie":
                category = "tie"
            elif expected == "unclear":
                category = "unclear"
            else:
                category = "baseline" if expected == ordered[0] else "candidate"
            labels.append(
                {
                    "pair_id": pair_id,
                    "episode": pair["episode"],
                    "annotator": annotator,
                    "choice": row["choice"],
                    "category": category,
                    "note": row.get("note"),
                    "identities_revealed": bool(row.get("identities_revealed")),
                }
            )
    return labels, sources


def _consensus(labels: Sequence[dict[str, Any]]) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for label in labels:
        if label["category"] == "unclear":
            continue
        bucket = counts.setdefault(label["pair_id"], {})
        bucket[label["category"]] = bucket.get(label["category"], 0) + 1
    result: dict[str, str] = {}
    for pair_id, bucket in counts.items():
        high = max(bucket.values())
        winners = sorted(category for category, count in bucket.items() if count == high)
        result[pair_id] = winners[0] if len(winners) == 1 else "unclear"
    return result


def _comparison_rows(
    report: dict[str, Any], judge_pairs: dict[str, dict[str, Any]], consensus: dict[str, str]
) -> list[dict[str, Any]]:
    by_run = {run["run_key"]: run for run in report["runs"]}
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    for pair in report["pairs"]:
        ordered_runs = sorted((pair["a_run_key"], pair["b_run_key"]))
        baseline_arm = by_run[ordered_runs[0]]["arm"]
        candidate_arm = by_run[ordered_runs[1]]["arm"]
        # Run-key ordering is deterministic but arm ordering can differ. Normalize
        # every score so each aggregate is candidate_arm vs baseline_arm alphabetically.
        arms = sorted((baseline_arm, candidate_arm))
        key = (arms[0], arms[1])
        bucket = buckets.setdefault(key, {"judge": [], "human": []})
        invert = candidate_arm != arms[1]
        judged = judge_pairs.get(pair["pair_id"])
        if judged and judged.get("score") is not None:
            value = float(judged["score"])
            bucket["judge"].append(1.0 - value if invert else value)
        human = consensus.get(pair["pair_id"])
        if human in {"baseline", "candidate", "tie"}:
            value = {"baseline": 0.0, "tie": 0.5, "candidate": 1.0}[human]
            bucket["human"].append(1.0 - value if invert else value)
    seed = seed_from_suite_hash(hashlib.sha256(report["plan_id"].encode()).hexdigest())
    rows = []
    for (baseline, candidate), bucket in sorted(buckets.items()):
        rows.append(
            {
                "baseline": baseline,
                "candidate": candidate,
                "judge": {
                    "n": len(bucket["judge"]),
                    "win_rate": win_rate(bucket["judge"]),
                    "ci": bootstrap_ci(bucket["judge"], seed),
                },
                "human": {
                    "n": len(bucket["human"]),
                    "win_rate": win_rate(bucket["human"]),
                    "ci": bootstrap_ci(bucket["human"], seed),
                },
            }
        )
    return rows


def _arm_rankings(
    report: dict[str, Any], judge_pairs: dict[str, dict[str, Any]], consensus: dict[str, str]
) -> list[dict[str, Any]]:
    by_run = {run["run_key"]: run for run in report["runs"]}
    scores: dict[str, dict[str, list[float]]] = {
        arm["arm"]: {"human": [], "judge": []} for arm in report["arms"]
    }
    for pair in report["pairs"]:
        baseline_key, candidate_key = sorted((pair["a_run_key"], pair["b_run_key"]))
        baseline_arm = by_run[baseline_key]["arm"]
        candidate_arm = by_run[candidate_key]["arm"]
        judged = judge_pairs.get(pair["pair_id"])
        if judged and judged.get("score") is not None:
            value = float(judged["score"])
            scores[candidate_arm]["judge"].append(value)
            scores[baseline_arm]["judge"].append(1.0 - value)
        human = consensus.get(pair["pair_id"])
        if human in {"baseline", "candidate", "tie"}:
            value = {"baseline": 0.0, "tie": 0.5, "candidate": 1.0}[human]
            scores[candidate_arm]["human"].append(value)
            scores[baseline_arm]["human"].append(1.0 - value)
    seed = seed_from_suite_hash(hashlib.sha256(report["plan_id"].encode()).hexdigest())
    rows = []
    for arm, lanes in scores.items():
        rows.append(
            {
                "arm": arm,
                "human": {
                    "n": len(lanes["human"]),
                    "score": win_rate(lanes["human"]),
                    "ci": bootstrap_ci(lanes["human"], seed),
                },
                "judge": {
                    "n": len(lanes["judge"]),
                    "score": win_rate(lanes["judge"]),
                    "ci": bootstrap_ci(lanes["judge"], seed),
                },
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -(row["human"]["score"] if row["human"]["score"] is not None else -1),
            row["arm"],
        ),
    )


def _judging_call_rows(judging: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in judging.get("rankings", []) if isinstance(row, dict)]
    for pair in judging.get("pairs", []):
        if isinstance(pair, dict):
            rows.extend(row for row in pair.get("passes", []) if isinstance(row, dict))
    return rows


def _judge_runtime_summary(
    prepared: Path,
    judging: dict[str, Any] | None,
    *,
    judging_root: Path | None = None,
) -> dict[str, Any]:
    if judging is None:
        return {"status": "not-run", "calls": []}
    evidence_root = judging_root or prepared / "judging"
    calls: list[dict[str, Any]] = []
    for row in _judging_call_rows(judging):
        observed = row.get("observed_model")
        model_usage = row.get("model_usage") if isinstance(row.get("model_usage"), dict) else {}
        auxiliary = list(row.get("auxiliary_models") or [])
        meter_equivalent = row.get("meter_equivalent_usd")
        meter_equivalent_source = (
            "judge-artifact" if isinstance(meter_equivalent, (int, float)) else None
        )
        usage: dict[str, Any] = {}
        raw_relative = row.get("raw")
        if isinstance(raw_relative, str):
            raw_path = evidence_root / raw_relative
            try:
                raw_text = raw_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                raw_text = ""
            events, malformed = _parse_events(raw_text)
            if events and not malformed:
                _session, runtime_model, _final, _tools, normalized = _normalize_events(
                    str(judging.get("runner", "")), events
                )
                observed = runtime_model or observed
                model_usage = normalized.get("model_usage") or model_usage
                auxiliary = list(normalized.get("auxiliary_models") or auxiliary)
                usage = normalized.get("usage", {})
                if isinstance(usage.get("reported_cost_usd"), (int, float)):
                    meter_equivalent = float(usage["reported_cost_usd"])
                    meter_equivalent_source = "runner-reported"
        if meter_equivalent is None and usage:
            try:
                estimate = _api_list_price_equivalent(
                    {
                        "runner": judging.get("runner"),
                        "requested_model": judging.get("model"),
                        "billing": {"basis": "subscription"},
                        "turns": [{"usage": usage}],
                    },
                    load_price_table(),
                )
            except PriceTableError:
                estimate = {"status": "unavailable"}
            if estimate.get("status") == "estimated":
                meter_equivalent = float(estimate["usd"])
                meter_equivalent_source = "reported-tokens-times-pinned-api-prices"
        calls.append(
            {
                "observed_model": observed,
                "auxiliary_models": auxiliary,
                "model_usage": model_usage,
                "meter_equivalent_usd": meter_equivalent,
                "meter_equivalent_source": meter_equivalent_source,
                "latency_ms": row.get("latency_ms"),
                "raw": raw_relative,
            }
        )
    observed_models = list(
        dict.fromkeys(
            str(call["observed_model"])
            for call in calls
            if isinstance(call.get("observed_model"), str)
        )
    )
    auxiliary_models = list(
        dict.fromkeys(
            str(model)
            for call in calls
            for model in call.get("auxiliary_models", [])
            if isinstance(model, str)
        )
    )
    meter_values = [
        float(call["meter_equivalent_usd"])
        for call in calls
        if isinstance(call.get("meter_equivalent_usd"), (int, float))
    ]
    requested = judging.get("model")
    primary_matches_requested = (
        all(model == requested for model in observed_models) if observed_models else None
    )
    return {
        "status": "recorded" if calls else "unavailable",
        "runner": judging.get("runner"),
        "requested_model": requested,
        "observed_models": observed_models,
        "primary_matches_requested": primary_matches_requested,
        "identity_confidence": "runner-reported" if observed_models else "requested-only",
        "auxiliary_models": auxiliary_models,
        "runner_meter_equivalent_usd": sum(meter_values) if len(meter_values) == len(calls) else None,
        "billing_basis": judging.get("billing_basis"),
        "calls": calls,
    }


def _n_way_review(
    report: dict[str, Any],
    label_paths: Sequence[str | Path],
    judging: dict[str, Any] | None,
) -> dict[str, Any]:
    run_to_arm = {
        run["run_key"]: f"{run['arm']} · {run['requested_model']}"
        for run in report["runs"]
    }
    human: list[dict[str, Any]] = []
    for label_path in label_paths:
        payload = _read_json(Path(label_path).resolve())
        if payload.get("format") != RANKING_LABEL_FORMAT:
            continue
        for row in payload.get("rankings", []):
            if not isinstance(row, dict) or row.get("unclear"):
                continue
            run_keys = row.get("ranked_run_keys") or []
            human.append(
                {
                    "ranking_id": row.get("ranking_id"),
                    "arms": [run_to_arm.get(str(run_key), str(run_key)) for run_key in run_keys],
                    "note": row.get("note"),
                    "identities_revealed": bool(row.get("identities_revealed")),
                    "identity_reveal_attestation": row.get("identity_reveal_attestation"),
                }
            )
    judge: list[dict[str, Any]] = []
    if judging is not None:
        for row in judging.get("rankings", []):
            if not isinstance(row, dict) or row.get("status") != "parsed":
                continue
            judge.append(
                {
                    "ranking_id": row.get("ranking_id"),
                    "arms": [
                        run_to_arm.get(str(run_key), str(run_key))
                        for run_key in row.get("ranked_run_keys", [])
                    ],
                    "reason": row.get("reason"),
                }
            )
    return {"human": human, "judge": judge}


def _analyze_judging(
    prepared: Path,
    report: dict[str, Any],
    consensus: dict[str, str],
    judging: dict[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    judge_pairs = {row["pair_id"]: row for row in judging["pairs"]}
    calibration_rows: list[tuple[str, str]] = []
    task_rows: list[tuple[str, str, str]] = []
    disagreements: list[dict[str, Any]] = []
    for pair_id, human in consensus.items():
        judged = judge_pairs.get(pair_id)
        if human == "unclear" or not judged or judged.get("score") is None:
            continue
        judge = judge_category(float(judged["score"]))
        calibration_rows.append((judge, human))
        episode = next(pair["episode"] for pair in report["pairs"] if pair["pair_id"] == pair_id)
        task_rows.append((episode, judge, human))
        if judge != human:
            disagreements.append(
                {"pair_id": pair_id, "episode": episode, "judge": judge, "human": human}
            )
    judge_config = report["judge"]
    calibration = calibration_status(
        cohens_kappa(calibration_rows),
        float(judge_config.get("min_kappa", DEFAULT_MIN_KAPPA)),
        int(judge_config.get("min_labels", DEFAULT_MIN_LABELS)),
    )
    return {
        "id": f"{judging.get('runner', 'judge')}:{judging.get('model', 'unknown')}",
        "runner": judging.get("runner"),
        "model": judging.get("model"),
        "artifact_root": str(artifact_root),
        "judging": judging,
        "runtime": _judge_runtime_summary(prepared, judging, judging_root=artifact_root),
        "n_way_rankings": _n_way_review(report, [], judging)["judge"],
        "judged_pairs": sum(row.get("score") is not None for row in judge_pairs.values()),
        "calibration": calibration,
        "per_episode_agreement": per_task_agreement(task_rows),
        "disagreements": disagreements,
        "comparisons": _comparison_rows(report, judge_pairs, consensus),
        "arm_rankings": _arm_rankings(report, judge_pairs, consensus),
    }


def build_decision_data(
    root: str | Path,
    label_paths: Sequence[str | Path],
    *,
    generated_at: str | None = None,
    judging_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    prepared = Path(verify_prepared(root)["root"])
    report = build_experiment_report_data(prepared)
    labels, sources = _load_human_labels(label_paths, report)
    consensus = _consensus(labels)
    resolved_judging_paths = (
        [Path(path).resolve() for path in judging_paths]
        if judging_paths is not None
        else ([prepared / "judging"] if (prepared / "judging").is_dir() else [])
    )
    judge_lanes: list[dict[str, Any]] = []
    for judging_root in resolved_judging_paths:
        verify_judging(prepared, judging_root)
        judging_doc = _read_json(judging_root / "pairs.json")
        judge_lanes.append(
            _analyze_judging(prepared, report, consensus, judging_doc, judging_root)
        )
    primary_lane = judge_lanes[0] if judge_lanes else None
    judging = primary_lane["judging"] if primary_lane else None
    judge_runtime = primary_lane["runtime"] if primary_lane else {"status": "not-run", "calls": []}
    n_way_review = _n_way_review(report, label_paths, judging)
    calibration = (
        primary_lane["calibration"]
        if primary_lane
        else calibration_status(
            cohens_kappa([]),
            float(report["judge"].get("min_kappa", DEFAULT_MIN_KAPPA)),
            int(report["judge"].get("min_labels", DEFAULT_MIN_LABELS)),
        )
    )
    disagreements = primary_lane["disagreements"] if primary_lane else []
    comparisons = primary_lane["comparisons"] if primary_lane else []
    arm_rankings = primary_lane["arm_rankings"] if primary_lane else []
    labelled_pairs = len({row["pair_id"] for row in labels if row["category"] != "unclear"})
    revealed = sum(bool(row["identities_revealed"]) for row in labels)
    labels_per_pair = int(report["human"].get("labels_per_pair", 1))
    usable_counts: dict[str, int] = {}
    for label in labels:
        if label["category"] != "unclear":
            usable_counts[label["pair_id"]] = usable_counts.get(label["pair_id"], 0) + 1
    under_labelled_pairs = sorted(
        pair["pair_id"]
        for pair in report["pairs"]
        if usable_counts.get(pair["pair_id"], 0) < labels_per_pair
    )
    human_complete = not under_labelled_pairs
    headline_eligible = bool(
        judge_lanes
        and all(lane["calibration"]["headline_eligible"] for lane in judge_lanes)
        and human_complete
    )
    human_scores = [
        row for row in arm_rankings if row["human"]["score"] is not None
    ]
    top_arm = human_scores[0]["arm"] if human_complete and human_scores else None
    tied_top = (
        len(human_scores) > 1
        and human_scores[0]["human"]["score"] == human_scores[1]["human"]["score"]
    )
    recommendation = {
        "status": (
            "insufficient-human-labels"
            if not human_complete
            else "tie"
            if tied_top
            else "human-preferred"
            if top_arm
            else "no-comparable-labels"
        ),
        "arm": None if tied_top else top_arm,
        "basis": "blind human aggregate; judge headline requires calibration",
    }
    ranking_style = str(
        judging.get("ranking_style") if judging else report.get("ranking_style", "pairwise")
    )
    return {
        "format": DECISION_FORMAT,
        "generated_at": generated_at or _now(),
        "prepared_root": str(prepared),
        "plan_id": report["plan_id"],
        "experiment": report["experiment"],
        "question": report["question"],
        "objectives": report["objectives"],
        "ranking_style": ranking_style,
        "arms": report["arms"],
        "pair_count": report["pair_count"],
        "labelled_pairs": labelled_pairs,
        "human_label_count": len(labels),
        "identities_revealed_labels": revealed,
        "label_sources": sources,
        "labels": labels,
        "consensus": consensus,
        "judging": judging,
        "judges": judge_lanes,
        "judge_count": len(judge_lanes),
        "judge_runtime": judge_runtime,
        "n_way_review": n_way_review,
        "judged_pairs": primary_lane["judged_pairs"] if primary_lane else 0,
        "calibration": calibration,
        "per_episode_agreement": (
            primary_lane["per_episode_agreement"] if primary_lane else []
        ),
        "disagreements": disagreements,
        "comparisons": comparisons,
        "arm_rankings": arm_rankings,
        "under_labelled_pairs": under_labelled_pairs,
        "human_complete": human_complete,
        "recommendation": recommendation,
        "headline_eligible": headline_eligible,
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _pct_ci(row: dict[str, Any], key: str) -> str:
    lane = row[key]
    ci = lane.get("ci")
    suffix = "" if ci is None else f" [{_pct(ci[0])}–{_pct(ci[1])}]"
    value = lane.get("win_rate", lane.get("score"))
    return f"{_pct(value)}{suffix}"


def render_decision_html(data: dict[str, Any]) -> str:
    judge_lanes = data.get("judges", [])
    recommendation = data["recommendation"]
    n_way = data.get("ranking_style") == "n-way"
    evidence_label = "Induced pair evidence" if n_way else "Pairwise evidence"
    evidence_note = (
        "Scores are induced from one full N-way ordering per episode/repeat."
        if n_way
        else "Scores use position-swapped judging."
    )
    pair_label = "Induced pairs" if n_way else "Pairs"
    recommendation_text = (
        f"Prefer {recommendation['arm']} on the blind human aggregate."
        if recommendation.get("arm")
        else f"No single arm recommendation: {recommendation['status']}."
    )
    n_way_review = data.get("n_way_review", {})
    human_rankings = "".join(
        "<article><h3>Human ranking</h3><ol>"
        + "".join(f"<li>{_esc(arm)}</li>" for arm in row.get("arms", []))
        + f"</ol><p>{_esc(row.get('note'))}</p>"
        + (
            "<p><small>Recorded blind: identities were revealed after labeling "
            "per operator attestation.</small></p>"
            if (row.get("identity_reveal_attestation") or {}).get("timing")
            == "after-labeling"
            else ""
        )
        + "</article>"
        for row in n_way_review.get("human", [])
    )
    judge_rankings = "".join(
        f"<article><h3>{_esc(lane.get('model'))} ranking</h3><ol>"
        + "".join(f"<li>{_esc(arm)}</li>" for arm in row.get("arms", []))
        + f"</ol><p><b>Rationale:</b> {_esc(row.get('reason'))}</p></article>"
        for lane in judge_lanes
        for row in lane.get("n_way_rankings", [])
    )
    n_way_html = (
        '<section><div class="eyebrow">Full N-way judgment</div>'
        f'<h2>Human and judge orderings</h2><div class="judge-grid">'
        f"{human_rankings}{judge_rankings}</div></section>"
        if n_way and (human_rankings or judge_rankings)
        else ""
    )
    runtime_cards = ""
    for lane in judge_lanes:
        runtime = lane.get("runtime", {})
        runtime_calls = "".join(
            "<ul>"
            + "".join(
                f"<li><code>{_esc(model)}</code> · "
                f"{'primary response' if model == call.get('observed_model') else 'auxiliary usage'}"
                + (
                    f" · ${float(usage.get('costUSD')):.4f} API list-price equivalent"
                    if isinstance(usage, dict)
                    and isinstance(usage.get("costUSD"), (int, float))
                    else ""
                )
                + "</li>"
                for model, usage in call.get("model_usage", {}).items()
            )
            + "</ul>"
            for call in runtime.get("calls", [])
            if call.get("model_usage")
        )
        runtime_total = runtime.get("runner_meter_equivalent_usd")
        runtime_cards += (
            "<article>"
            f"<h3>{_esc(lane.get('model'))}</h3><p>Requested: "
            f"<code>{_esc(runtime.get('requested_model') or 'unavailable')}</code></p>"
            "<p>Primary observed: "
            f"<code>{_esc(', '.join(runtime.get('observed_models', [])) or 'unavailable')}</code>"
            f"</p><p>Identity confidence: <code>{_esc(runtime.get('identity_confidence'))}</code></p>"
            + (
                f"<p>API list-price equivalent: <b>${float(runtime_total):.4f}</b>. "
                "Not the subscription charge.</p>"
                if isinstance(runtime_total, (int, float))
                else "<p>API list-price equivalent unavailable.</p>"
            )
            + runtime_calls
            + "</article>"
        )
    runtime_html = (
        '<section><div class="eyebrow">Judge execution identity</div>'
        f'<h2>Requested and observed models</h2><div class="judge-grid">'
        f"{runtime_cards}</div></section>"
        if runtime_cards
        else ""
    )
    calibration_cards = "".join(
        f"<article class=\"{'ok' if lane['calibration']['headline_eligible'] else 'warn'}\">"
        f"<h3>{_esc(lane.get('model'))}: "
        f"{'eligible' if lane['calibration']['headline_eligible'] else 'diagnostic only'}</h3>"
        f"<p>{_esc(lane['calibration'].get('reason') or 'Calibration passed.')}</p>"
        f"<p>Kappa: <b>{_esc(lane['calibration'].get('kappa'))}</b>; agreement "
        f"{_pct(lane['calibration'].get('agreement'))}; "
        f"n={lane['calibration'].get('n_labels')}.</p></article>"
        for lane in judge_lanes
    )
    calibration_html = (
        '<section><div class="eyebrow">Calibration gates</div>'
        f'<h2>Each judge versus the same human ranking</h2><div class="judge-grid">'
        f"{calibration_cards}</div></section>"
    )
    evidence_cards = ""
    audit_cards = ""
    for lane in judge_lanes:
        rows = "".join(
            f"<tr><td>{_esc(row['candidate'])}</td><td>{_esc(row['baseline'])}</td>"
            f"<td>{_pct_ci(row, 'human')} <small>n={row['human']['n']}</small></td>"
            f"<td>{_pct_ci(row, 'judge')} <small>n={row['judge']['n']}</small></td></tr>"
            for row in lane["comparisons"]
        )
        evidence_cards += (
            f"<article><h3>{_esc(lane.get('model'))}</h3><table><thead><tr>"
            f"<th>Candidate</th><th>Compared with</th><th>Human</th>"
            f"<th>{_esc(lane.get('model'))}</th></tr></thead><tbody>{rows}</tbody></table></article>"
        )
        disagreement_rows = "".join(
            f"<li><code>{_esc(row['pair_id'])}</code> · {_esc(row['episode'])}: "
            f"human {_esc(row['human'])}, judge {_esc(row['judge'])}</li>"
            for row in lane["disagreements"]
        ) or "<li>None.</li>"
        audit_cards += (
            f"<article><h3>{_esc(lane.get('model'))}</h3><ul>{disagreement_rows}</ul></article>"
        )
    evidence_html = (
        f'<section><div class="eyebrow">{evidence_label}</div>'
        f'<h2>Candidate preference by judge</h2><div class="judge-grid">{evidence_cards}</div>'
        f"<p><small>{evidence_note} Confidence intervals are retained in data.json and "
        "suppressed below eight pairs.</small></p></section>"
    )
    audit_html = (
        '<section><div class="eyebrow">Audit queue</div>'
        f'<h2>Human ↔ judge disagreements</h2><div class="judge-grid">{audit_cards}</div></section>'
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(data['experiment'])} · decision</title><style>
:root{{--ink:#17231d;--muted:#607068;--paper:#f5f1e8;--card:#fffdf7;--green:#145f42;--amber:#b86d13;--line:#d8d2c3}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}header,main,footer{{max-width:1320px;margin:auto;padding:28px}}header{{padding-top:64px}}.eyebrow{{color:var(--green);font-weight:800;text-transform:uppercase;letter-spacing:.12em;font-size:12px}}h1{{font:700 clamp(32px,5vw,62px)/1.04 Georgia,serif;margin:.2em 0}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.judge-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}.judge-grid article{{border:1px solid var(--line);border-radius:12px;padding:16px;overflow:auto}}.card,section{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;margin:16px 0}}.card b{{display:block;font-size:30px}}.warn{{border-left:6px solid var(--amber)}}.ok{{border-left:6px solid var(--green)}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;text-align:left;border-bottom:1px solid var(--line)}}small{{color:var(--muted)}}code{{font-size:.86em}}@media(max-width:700px){{.grid{{grid-template-columns:1fr 1fr}}.judge-grid{{grid-template-columns:1fr}}table{{font-size:13px}}header,main,footer{{padding:18px}}}}
</style></head><body><header><div class="eyebrow">evalmine · decision evidence</div><h1>{_esc(data['question'])}</h1><p>{_esc(data['experiment'])} · plan <code>{_esc(data['plan_id'])}</code></p></header><main>
<div class="grid"><div class="card"><small>{pair_label}</small><b>{data['pair_count']}</b></div><div class="card"><small>Human-labelled</small><b>{data['labelled_pairs']}</b></div><div class="card"><small>Judges</small><b>{data.get('judge_count', 0)}</b></div><div class="card"><small>Judged per lane</small><b>{data['judged_pairs']}</b></div></div>
<section class="{'ok' if recommendation.get('arm') else 'warn'}"><div class="eyebrow">Decision</div><h2>{_esc(recommendation_text)}</h2><p>{_esc(recommendation['basis'])}</p></section>
{calibration_html}
{n_way_html}
{runtime_html}
{evidence_html}
{audit_html}
<section><div class="eyebrow">Interpretation</div><h2>What this report can claim</h2><p>{'The judge passed calibration and its aggregate results may be used as headline evidence.' if data['headline_eligible'] else 'Use the human rows and inspect disagreements. Judge aggregates are shown diagnostically, but the calibration gate prevents treating them as a headline conclusion.'}</p><p>{data['identities_revealed_labels']} imported labels were recorded after identity reveal.</p></section></main><footer>Generated {_esc(data['generated_at'])} · self-contained · raw evidence remains beside this report</footer></body></html>"""


def generate_decision_report(
    root: str | Path,
    label_paths: Sequence[str | Path],
    *,
    generated_at: str | None = None,
    judging_paths: Sequence[str | Path] | None = None,
    output: str | Path | None = None,
) -> DecisionResult:
    prepared = Path(verify_prepared(root)["root"])
    decision_root = Path(output).resolve() if output is not None else prepared / "decision"
    if decision_root.exists() or decision_root.is_symlink():
        raise DecisionError(f"decision evidence already exists at {decision_root}")
    data = build_decision_data(
        prepared,
        label_paths,
        generated_at=generated_at,
        judging_paths=judging_paths,
    )
    decision_root.parent.mkdir(parents=True, exist_ok=True)
    decision_root.mkdir()
    labels_root = decision_root / "labels"
    labels_root.mkdir()
    for index, source in enumerate(data["label_sources"], 1):
        original = Path(source["path"])
        _write_read_only(labels_root / f"{index:03d}-{original.name}", original.read_bytes())
    _write_read_only(decision_root / "data.json", _json_bytes(data))
    _write_read_only(decision_root / "index.html", render_decision_html(data).encode())
    marker = {
        "format": DECISION_FORMAT,
        "created_at": data["generated_at"],
        "prepared_root": str(prepared),
        "plan_id": data["plan_id"],
        "labelled_pairs": data["labelled_pairs"],
        "judged_pairs": data["judged_pairs"],
        "headline_eligible": data["headline_eligible"],
        "evidence_sha256": _hashes(decision_root, "decision.json"),
    }
    _write_read_only(decision_root / "decision.json", _json_bytes(marker))
    return DecisionResult(
        decision_root,
        decision_root / "index.html",
        data["labelled_pairs"],
        data["judged_pairs"],
        data["headline_eligible"],
    )


def verify_decision(
    root: str | Path, decision_path: str | Path | None = None
) -> dict[str, Any]:
    prepared = Path(verify_prepared(root)["root"])
    decision_root = (
        Path(decision_path).resolve() if decision_path is not None else prepared / "decision"
    )
    marker = _read_json(decision_root / "decision.json")
    if marker.get("format") != DECISION_FORMAT or marker.get("prepared_root") != str(prepared):
        raise DecisionError("decision marker does not match this prepared experiment")
    if marker.get("evidence_sha256") != _hashes(decision_root, "decision.json"):
        raise DecisionError("decision evidence changed after creation")
    return {
        "ok": True,
        "format": DECISION_FORMAT,
        "root": str(decision_root),
        "headline_eligible": marker.get("headline_eligible"),
        "labelled_pairs": marker.get("labelled_pairs"),
        "judged_pairs": marker.get("judged_pairs"),
    }
