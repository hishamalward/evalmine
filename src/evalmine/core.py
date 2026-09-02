"""The library API: plan a run, guard the spend, make the calls, write a report.

Spec: docs/spec.md S6.4, S7, S9. The CLI and the MCP server are both thin
callers of ``run_suite``; there is exactly one place in evalmine where money can
be spent, and it is here, behind a cap that cannot be bypassed.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .adapters import Request, build_adapter, call_with_retries, split_model
from .adapters.base import Adapter, AdapterError
from .cache import DEFAULT_CACHE_DIR, Cache, answer_payload, cache_key, judge_payload
from .check import CheckSpec, run_check
from .html_report import build_pair_view, render_html
from .judge import PROVIDER_ERROR as REASON_PROVIDER_ERROR
from .judge import Judge, JudgeCall, PairResult, exclusion_reason
from .metrics import (
    call_cost,
    estimate_answer_cost,
    estimate_judge_cost,
    schema_verdict,
)
from .prices import PriceRow, PriceTable, load_price_table
from .report import build_report, render_markdown
from .suite import Suite, Task, load_suite, models_hash

#: S6.4: if no cap is set anywhere. The MCP server passes a lower one, because
#: the human at the CLI typed the number and the agent did not.
DEFAULT_CLI_MAX_COST = 2.00
DEFAULT_MCP_MAX_COST = 1.00

REPORT_DIR_DEFAULT = "reports"


class UsageError(Exception):
    """A bad argument or an unusable suite. Exit 1."""


class RunError(Exception):
    """A provider or runtime failure that stopped the run. Exit 2."""


@dataclass
class CostRefused(Exception):
    """The pre-flight estimate exceeds the cap. Nothing was spent. Exit 4."""

    estimate: float
    cap: float
    breakdown: dict[str, float]

    def __str__(self) -> str:
        lines = [
            f"refusing to start: the pre-flight estimate is ${self.estimate:.4f}, "
            f"which is over the ${self.cap:.2f} cap. Nothing was spent.",
            "Estimated breakdown (the chars/4 heuristic, with output assumed maximal):",
        ]
        for name, value in sorted(self.breakdown.items()):
            lines.append(f"  {name:<40} ${value:.4f}")
        lines.append("Raise --max-cost, cut --models, or run a smaller suite.")
        return "\n".join(lines)


class _BudgetExceeded(Exception):
    """Internal signal: the live ceiling was crossed. Never leaves this module."""


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------


@dataclass
class AnswerRecord:
    task_id: str
    case_id: str
    model: str
    provider: str
    repeat: int
    prompt: str
    system: str | None
    cache_key: str
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0
    provider_reported_cost_usd: float | None = None
    cost_basis: str | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None
    schema_mode: str = "prompted"
    cached: bool = False
    cost: float | None = 0.0
    cost_if_uncached: float | None = None
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    schema_status: str = "not_applicable"
    schema_error: str | None = None
    #: Execution check (S6.6): "pass" | "fail" | "error" | "not_applicable".
    check_status: str = "not_applicable"
    check_exit: int | None = None
    check_output: str | None = None
    #: Every code block the check ran, in order; the verdict above is the last one's.
    check_blocks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def check_view(self) -> dict[str, Any] | None:
        """What the judge and the pair view are shown; None when there was no check."""
        if self.check_status == "not_applicable":
            return None
        return {
            "status": self.check_status,
            "exit_code": self.check_exit,
            "output": self.check_output or "",
            "blocks": list(self.check_blocks),
        }

    @property
    def usable(self) -> bool:
        return self.status == "ok" and self.schema_status in ("pass", "not_applicable")

    def as_jsonl(self) -> dict[str, Any]:
        return {
            "task": self.task_id,
            "case": self.case_id,
            "model": self.model,
            "repeat": self.repeat,
            "cache_key": self.cache_key,
            "cached": self.cached,
            "status": self.status,
            "error": self.error,
            "schema_status": self.schema_status,
            "schema_error": self.schema_error,
            "schema_mode": self.schema_mode,
            "check_status": self.check_status,
            "check_exit": self.check_exit,
            "check_output": self.check_output,
            "check_blocks": self.check_blocks,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "provider_reported_cost_usd": self.provider_reported_cost_usd,
            "cost_basis": self.cost_basis,
            "cost_usd": self.cost,
            "cost_if_uncached_usd": self.cost_if_uncached,
            "finish_reason": self.finish_reason,
            "text": self.text,
        }


@dataclass
class CallPlan:
    task: Task
    case_id: str
    model: str
    provider: str
    repeat: int
    prompt: str
    system: str | None
    key: str
    provider_options: dict[str, Any] | None = None
    cached: bool = False
    check: CheckSpec | None = None


@dataclass
class RunData:
    suite: Suite
    models: list[str]
    baseline: str
    candidates: list[str]
    judge_model: str
    repeats: int
    fake: bool
    no_cache: bool
    prices: PriceTable
    price_rows: dict[str, PriceRow]
    cap: float
    answers: list[AnswerRecord]
    pairs: list[PairResult]
    judge_latencies: list[int]
    judge_cost: float
    judge_cost_if_uncached: float
    judge_calls: int
    judge_cached_calls: int
    cache_hits: int
    live_calls: int
    run_id: str
    generated_at: str
    out_dir: Path
    cache_dir: Path
    command: str | None
    estimate: float
    aborted_over_budget: bool = False
    warnings: list[str] = field(default_factory=list)
    ignored_labels: list[dict[str, Any]] = field(default_factory=list)
    previous_report: dict[str, Any] | None = None


@dataclass
class RunResult:
    run_id: str
    report: dict[str, Any]
    report_path: Path
    report_md_path: Path
    exit_code: int
    report_html_path: Path | None = None
    #: One record per judged pair - prompt, both answers, verdict, and the
    #: blind A/B mapping report.html was built from.
    pair_view: list[dict[str, Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_run_id(when: datetime, suite_hash: str, models: list[str]) -> str:
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{suite_hash[:8]}_{models_hash(models)[:8]}"


def _adapter_provider(model: str, fake: bool) -> tuple[str, str]:
    provider, model_id = split_model(model)
    return ("fake" if fake else provider), model_id


def _provider_options(suite: Suite, model: str, fake: bool) -> dict[str, Any] | None:
    provider, model_id = split_model(model)
    if fake or provider != "openrouter":
        return None
    pinned = suite.openrouter_provider_pins.get(model_id)
    if pinned is None:
        return None
    return {"order": [pinned], "allow_fallbacks": False}


class _AdapterPool:
    """One adapter per provider, built on first use and never before."""

    def __init__(self, fake: bool, factory: Callable[[str, bool], Adapter]) -> None:
        self.fake = fake
        self.factory = factory
        self._built: dict[str, Adapter] = {}

    def get(self, provider: str) -> Adapter:
        if provider not in self._built:
            self._built[provider] = self.factory(provider, self.fake)
        return self._built[provider]

    @property
    def built(self) -> dict[str, Adapter]:
        return dict(self._built)


# --------------------------------------------------------------------------
# run_suite
# --------------------------------------------------------------------------


def run_suite(
    suite_path: str | Path,
    models: list[str],
    *,
    baseline: str | None = None,
    judge_model: str | None = None,
    repeats: int = 1,
    max_cost: float | None = None,
    no_cache: bool = False,
    cache_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    fake: bool = False,
    prices_path: str | Path | None = None,
    prices_dir: str | Path | None = None,
    default_max_cost: float = DEFAULT_CLI_MAX_COST,
    adapter_factory: Callable[[str, bool], Adapter] = build_adapter,
    command: str | None = None,
    now: datetime | None = None,
    write: bool = True,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    """Run one suite over one model list and write one report.

    Raises :class:`CostRefused` before spending anything if the pre-flight
    estimate is over the cap, and :class:`RunError` if a provider failure
    stopped the run.
    """
    suite = load_suite(suite_path)

    if len(models) < 2:
        raise UsageError("--models needs at least two model strings: a baseline and a candidate")
    if len(set(models)) != len(models):
        raise UsageError(f"--models contains a duplicate: {models}")
    baseline = baseline or models[0]
    if baseline not in models:
        raise UsageError(f"--baseline {baseline!r} is not in --models {models}")
    candidates = [m for m in models if m != baseline]
    judge_model = judge_model or suite.judge.model
    if repeats < 1:
        raise UsageError("--repeats must be at least 1")

    # Prices resolve before any adapter is constructed: an unknown model is a
    # hard failure at planning time, never a $0.00 in a finished report.
    prices = load_price_table(prices_path, prices_dir)
    price_rows = prices.resolve_all([*models, judge_model])

    cap = max_cost
    if cap is None:
        cap = suite.max_cost_usd if suite.max_cost_usd is not None else default_max_cost

    cache = Cache(cache_dir or DEFAULT_CACHE_DIR, read=not no_cache)
    when = now or _utc_now()
    run_id = make_run_id(when, suite.hash, models)

    warnings: list[str] = []
    if judge_model in models:
        warnings.append(
            f"the judge ({judge_model}) is also one of the models under test - a known "
            "self-preference risk; read the win-rate with that in mind"
        )
    if not prices.verified:
        warnings.append(
            f"price table {prices.filename} declares verified: false - every cost figure "
            "in this report is a placeholder until the table is checked against the "
            "providers' pricing pages"
        )

    pool = _AdapterPool(fake, adapter_factory)

    # ---- plan -----------------------------------------------------------
    plans: list[CallPlan] = []
    for task in suite.tasks:
        for case in task.cases:
            for model in models:
                provider, _ = _adapter_provider(model, fake)
                for repeat in range(repeats):
                    plans.append(
                        CallPlan(
                            task=task,
                            case_id=case.id,
                            model=model,
                            provider=provider,
                            repeat=repeat,
                            prompt=case.prompt,
                            system=case.system,
                            key="",
                            provider_options=_provider_options(suite, model, fake),
                            check=case.check,
                        )
                    )

    # The cache key needs the adapter's schema mode and version, which is the
    # only reason the adapter is touched before the cost guard runs. Building
    # one costs nothing and contacts nothing; spending happens in complete().
    for plan in plans:
        adapter = pool.get(plan.provider)
        payload = answer_payload(
            provider=plan.provider,
            model=plan.model,
            system=plan.system,
            prompt=plan.prompt,
            params={
                **plan.task.params.as_cache_params(),
                "provider_options": plan.provider_options,
            },
            schema=plan.task.schema,
            schema_mode=adapter.schema_mode_for(plan.task.schema),
            adapter_version=adapter.version,
            repeat=plan.repeat,
        )
        plan.key = cache_key(payload)
        plan.cached = (not no_cache) and cache.path_for(plan.provider, plan.key).is_file()

    # ---- pre-flight estimate (S6.4) --------------------------------------
    breakdown: dict[str, float] = {}
    for plan in plans:
        if plan.cached:
            continue
        row = price_rows[plan.model]
        cost = estimate_answer_cost(row, plan.prompt, plan.system, plan.task.params.max_tokens)
        breakdown[plan.model] = breakdown.get(plan.model, 0.0) + cost

    judge_row = price_rows[judge_model]
    judge_estimate = 0.0
    for task in suite.tasks:
        if not task.judge:
            continue
        for case in task.cases:
            per_pass = estimate_judge_cost(
                judge_row,
                suite.judge.rubric + (task.rubric or ""),
                case.prompt,
                task.params.max_tokens,
                suite.judge.max_tokens,
            )
            judge_estimate += per_pass * 2 * len(candidates) * repeats
    if judge_estimate:
        breakdown[f"{judge_model} (judge)"] = judge_estimate

    estimate = sum(breakdown.values())
    if estimate > cap:
        raise CostRefused(estimate=estimate, cap=cap, breakdown=breakdown)

    # ---- run the answers -------------------------------------------------
    spend = 0.0
    aborted = False
    answers: list[AnswerRecord] = []
    by_key: dict[tuple[str, str, str, int], AnswerRecord] = {}

    def _charge(cost: float | None) -> None:
        nonlocal spend
        if cost:
            spend += cost
        if spend > cap:
            raise _BudgetExceeded()

    try:
        for plan in plans:
            record = _run_one_answer(
                plan, pool, cache, price_rows[plan.model], retry_sleep=retry_sleep
            )
            answers.append(record)
            by_key[(plan.task.id, plan.case_id, plan.model, plan.repeat)] = record
            if record.status == "error":
                warnings.append(
                    f"{plan.model} failed on {plan.task.id}/{plan.case_id}: {record.error}"
                )
            if not record.cached:
                _charge(record.cost)
    except _BudgetExceeded:
        aborted = True
        warnings.append(
            f"aborted mid-run: spend crossed the ${cap:.2f} ceiling. This report covers "
            "only the calls that completed before that point."
        )

    # ---- judge the pairs -------------------------------------------------
    pairs: list[PairResult] = []
    judge_latencies: list[int] = []
    judge_cost = 0.0
    judge_cost_if_uncached = 0.0
    judge_calls = 0
    judge_cached_calls = 0

    if not aborted:
        judge_provider, _ = _adapter_provider(judge_model, fake)
        judge_adapter = pool.get(judge_provider)
        judge_provider_options = _provider_options(suite, judge_model, fake)

        def _judge_call(prompt: str, system: str | None, bypass_cache: bool = False) -> JudgeCall:
            nonlocal spend
            call = _run_one_judge_call(
                prompt=prompt,
                system=system,
                model=judge_model,
                provider=judge_provider,
                adapter=judge_adapter,
                config=suite.judge,
                cache=cache,
                row=judge_row,
                bypass_cache=bypass_cache or no_cache,
                retry_sleep=retry_sleep,
                provider_options=judge_provider_options,
            )
            if not call.cached:
                _charge(call.cost)
            return call

        judge = Judge(suite.judge, _judge_call)
        try:
            pairs = _judge_all_pairs(
                suite=suite,
                candidates=candidates,
                baseline=baseline,
                repeats=repeats,
                answers=by_key,
                judge=judge,
            )
        except _BudgetExceeded:
            aborted = True
            warnings.append(
                f"aborted mid-judging: spend crossed the ${cap:.2f} ceiling. Pairs after "
                "that point were not judged."
            )

        for pair in pairs:
            judge_cost += pair.judge_cost
            judge_cost_if_uncached += pair.judge_cost_if_uncached
            judge_calls += pair.judge_calls
            judge_cached_calls += pair.judge_cached_calls
            judge_latencies.extend(pair.judge_latencies)

    if any(p.reason == REASON_PROVIDER_ERROR for p in pairs):
        warnings.append(
            "at least one pair was excluded because a provider call failed after retries"
        )

    out_root = Path(out_dir or REPORT_DIR_DEFAULT)
    data = RunData(
        suite=suite,
        models=list(models),
        baseline=baseline,
        candidates=candidates,
        judge_model=judge_model,
        repeats=repeats,
        fake=fake,
        no_cache=no_cache,
        prices=prices,
        price_rows=price_rows,
        cap=cap,
        answers=answers,
        pairs=pairs,
        judge_latencies=judge_latencies,
        judge_cost=judge_cost,
        judge_cost_if_uncached=judge_cost_if_uncached,
        judge_calls=judge_calls,
        judge_cached_calls=judge_cached_calls,
        cache_hits=cache.stats.hits,
        live_calls=sum(1 for a in answers if not a.cached) + (judge_calls - judge_cached_calls),
        run_id=run_id,
        generated_at=when.strftime("%Y-%m-%dT%H:%M:%SZ"),
        out_dir=out_root,
        cache_dir=Path(cache_dir or DEFAULT_CACHE_DIR),
        command=command,
        estimate=estimate,
        aborted_over_budget=aborted,
        warnings=warnings,
    )
    data.previous_report = _find_previous_report(out_root, suite.name, run_id)

    report = build_report(data)
    pair_view = build_pair_view(data)
    report_path = out_root / suite.name / run_id / "report.json"
    md_path = report_path.with_name("report.md")
    html_path = report_path.with_name("report.html")
    if write:
        _write_outputs(data, report, pair_view, report_path, md_path, html_path)

    exit_code = 0
    if aborted:
        exit_code = 5
    return RunResult(
        run_id=run_id,
        report=report,
        report_path=report_path,
        report_md_path=md_path,
        report_html_path=html_path,
        pair_view=pair_view,
        exit_code=exit_code,
    )


def _run_one_answer(
    plan: CallPlan,
    pool: _AdapterPool,
    cache: Cache,
    row: PriceRow,
    retry_sleep: Callable[[float], None] = time.sleep,
) -> AnswerRecord:
    adapter = pool.get(plan.provider)
    record = AnswerRecord(
        task_id=plan.task.id,
        case_id=plan.case_id,
        model=plan.model,
        provider=plan.provider,
        repeat=plan.repeat,
        prompt=plan.prompt,
        system=plan.system,
        cache_key=plan.key,
        schema_mode=adapter.schema_mode_for(plan.task.schema),
    )

    entry = cache.get(plan.provider, plan.key)
    if entry is not None:
        record.cached = True
        record.text = entry.get("text", "")
        record.input_tokens = entry.get("input_tokens")
        record.output_tokens = entry.get("output_tokens")
        record.cached_input_tokens = entry.get("cached_input_tokens") or 0
        record.reasoning_tokens = entry.get("reasoning_tokens") or 0
        record.provider_reported_cost_usd = entry.get("reported_cost_usd")
        record.cost_basis = (
            "provider_reported" if record.provider_reported_cost_usd is not None else "price_table"
        )
        record.latency_ms = entry.get("latency_ms")
        record.finish_reason = entry.get("finish_reason")
        record.schema_mode = entry.get("schema_mode", record.schema_mode)
        # A cache hit costs nothing in this run; what it would have cost is
        # still reported, so a fully cached rerun does not read as free work.
        record.cost = 0.0
        record.cost_if_uncached = (
            record.provider_reported_cost_usd
            if record.provider_reported_cost_usd is not None
            else call_cost(
                row,
                record.input_tokens,
                record.output_tokens,
                record.cached_input_tokens,
            )
        )
    else:
        _, model_id = split_model(plan.model)
        req = Request(
            model_id=model_id,
            prompt=plan.prompt,
            system=plan.system,
            max_tokens=plan.task.params.max_tokens,
            temperature=plan.task.params.temperature,
            top_p=plan.task.params.top_p,
            stop=plan.task.params.stop,
            schema=plan.task.schema,
            provider_options=plan.provider_options,
            timeout_s=plan.task.params.timeout_s,
        )
        try:
            response = call_with_retries(adapter, req, sleep=retry_sleep)
        except AdapterError as exc:
            if not exc.retryable:
                # 401/400: stop now rather than after forty calls.
                raise RunError(
                    f"{plan.model} refused the request and will refuse the rest: {exc}"
                ) from exc
            record.status = "error"
            record.error = str(exc)
            record.cost = 0.0
            return record

        record.text = response.text
        record.input_tokens = response.input_tokens
        record.output_tokens = response.output_tokens
        record.cached_input_tokens = response.cached_input_tokens
        record.reasoning_tokens = response.reasoning_tokens
        record.provider_reported_cost_usd = response.reported_cost_usd
        record.latency_ms = response.latency_ms
        record.finish_reason = response.finish_reason
        record.schema_mode = response.schema_mode
        estimated_cost = call_cost(
            row, response.input_tokens, response.output_tokens, response.cached_input_tokens
        )
        record.cost = (
            response.reported_cost_usd
            if response.reported_cost_usd is not None
            else estimated_cost
        )
        record.cost_basis = (
            "provider_reported" if response.reported_cost_usd is not None else "price_table"
        )
        record.cost_if_uncached = record.cost
        cache.put(
            plan.provider,
            plan.key,
            {
                "model": plan.model,
                "text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cached_input_tokens": response.cached_input_tokens,
                "reasoning_tokens": response.reasoning_tokens,
                "reported_cost_usd": response.reported_cost_usd,
                "latency_ms": response.latency_ms,
                "finish_reason": response.finish_reason,
                "schema_mode": response.schema_mode,
                "adapter_version": adapter.version,
                "tool_version": __version__,
            },
        )

    verdict = schema_verdict(record.text, plan.task.schema)
    record.schema_status = verdict.status
    record.schema_error = verdict.error

    # S6.6: the check runs on cached answers too - it is local, cheap, and
    # deliberately outside the cache so that editing a check re-evaluates.
    if plan.check is not None and record.status == "ok":
        result = run_check(plan.check, record.text)
        record.check_status = result.status
        record.check_exit = result.exit_code
        record.check_output = result.output
        record.check_blocks = [
            {
                "index": b.index,
                "status": b.status,
                "exit_code": b.exit_code,
                "output": b.output,
            }
            for b in result.blocks
        ]
    return record


def _run_one_judge_call(
    *,
    prompt: str,
    system: str | None,
    model: str,
    provider: str,
    adapter: Adapter,
    config,
    cache: Cache,
    row: PriceRow,
    bypass_cache: bool,
    retry_sleep: Callable[[float], None] = time.sleep,
    provider_options: dict[str, Any] | None = None,
) -> JudgeCall:
    params = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "top_p": None,
        "stop": None,
        "seed": None,
        "provider_options": provider_options,
    }
    from .judge import JUDGE_SCHEMA

    payload = judge_payload(
        provider=provider,
        model=model,
        system=system,
        prompt=prompt,
        params=params,
        schema=JUDGE_SCHEMA,
        schema_mode=adapter.schema_mode_for(JUDGE_SCHEMA),
        adapter_version=adapter.version,
        repeat=0,
    )
    key = cache_key(payload)

    if not bypass_cache:
        entry = cache.get(provider, key)
        if entry is not None:
            reported_cost = entry.get("reported_cost_usd")
            return JudgeCall(
                text=entry.get("text", ""),
                cost=0.0,
                cost_if_uncached=(
                    reported_cost
                    if reported_cost is not None
                    else call_cost(
                        row,
                        entry.get("input_tokens"),
                        entry.get("output_tokens"),
                        entry.get("cached_input_tokens") or 0,
                    )
                ),
                cached=True,
                latency_ms=entry.get("latency_ms") or 0,
                input_tokens=entry.get("input_tokens"),
                output_tokens=entry.get("output_tokens"),
            )

    _, model_id = split_model(model)
    req = Request(
        model_id=model_id,
        prompt=prompt,
        system=system,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        schema=JUDGE_SCHEMA,
        provider_options=provider_options,
    )
    try:
        response = call_with_retries(adapter, req, sleep=retry_sleep)
    except AdapterError as exc:
        if not exc.retryable:
            raise RunError(f"the judge ({model}) refused the request: {exc}") from exc
        return JudgeCall(text="", cost=0.0, cached=False, latency_ms=0)

    cache.put(
        provider,
        key,
        {
            "model": model,
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cached_input_tokens": response.cached_input_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "reported_cost_usd": response.reported_cost_usd,
            "latency_ms": response.latency_ms,
            "finish_reason": response.finish_reason,
            "schema_mode": response.schema_mode,
            "adapter_version": adapter.version,
            "tool_version": __version__,
        },
    )
    live_cost = (
        response.reported_cost_usd
        if response.reported_cost_usd is not None
        else call_cost(
            row,
            response.input_tokens,
            response.output_tokens,
            response.cached_input_tokens,
        )
    )
    return JudgeCall(
        text=response.text,
        cost=live_cost,
        cost_if_uncached=live_cost,
        cached=False,
        latency_ms=response.latency_ms,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


def _judge_all_pairs(
    *,
    suite: Suite,
    candidates: list[str],
    baseline: str,
    repeats: int,
    answers: dict[tuple[str, str, str, int], AnswerRecord],
    judge: Judge,
) -> list[PairResult]:
    pairs: list[PairResult] = []
    for task in suite.tasks:
        if not task.judge:
            continue
        for case in task.cases:
            for candidate in candidates:
                for repeat in range(repeats):
                    base = answers.get((task.id, case.id, baseline, repeat))
                    cand = answers.get((task.id, case.id, candidate, repeat))
                    if base is None or cand is None:
                        continue

                    if base.status == "error" or cand.status == "error":
                        pairs.append(
                            PairResult(
                                task_id=task.id,
                                case_id=case.id,
                                repeat=repeat,
                                baseline=baseline,
                                candidate=candidate,
                                excluded=True,
                                reason=REASON_PROVIDER_ERROR,
                            )
                        )
                        continue

                    reason = exclusion_reason(
                        base.schema_status in ("parse_fail", "schema_fail"),
                        cand.schema_status in ("parse_fail", "schema_fail"),
                    )
                    if reason is not None:
                        # Ruling O-3: excluded, not an automatic loss, and no
                        # judge call is made at all.
                        pairs.append(
                            PairResult(
                                task_id=task.id,
                                case_id=case.id,
                                repeat=repeat,
                                baseline=baseline,
                                candidate=candidate,
                                excluded=True,
                                reason=reason,
                            )
                        )
                        continue

                    pairs.append(
                        judge.judge_pair(
                            task=task,
                            case_id=case.id,
                            repeat=repeat,
                            baseline=baseline,
                            candidate=candidate,
                            baseline_prompt=case.prompt,
                            baseline_text=base.text,
                            candidate_text=cand.text,
                            baseline_check=base.check_view,
                            candidate_check=cand.check_view,
                        )
                    )
    return pairs


# --------------------------------------------------------------------------
# report I/O
# --------------------------------------------------------------------------


def _write_outputs(
    data: RunData,
    report: dict[str, Any],
    pair_view: list[dict[str, Any]],
    report_path: Path,
    md_path: Path,
    html_path: Path | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    if html_path is not None:
        # report.html carries the answers themselves, which report.json does
        # not: it is built from the run, not re-rendered from the record.
        html_path.write_text(render_html(report, pair_view), encoding="utf-8")

    answers_path = report_path.with_name("answers.jsonl")
    with answers_path.open("w", encoding="utf-8") as handle:
        for answer in data.answers:
            handle.write(json.dumps(answer.as_jsonl(), ensure_ascii=False) + "\n")

    pairs_path = report_path.with_name("pairs.jsonl")
    with pairs_path.open("w", encoding="utf-8") as handle:
        for pair in data.pairs:
            handle.write(json.dumps(_pair_as_jsonl(pair), ensure_ascii=False) + "\n")

    # A pointer file, not a symlink, because CI runs on Windows.
    latest = report_path.parent.parent / "latest.json"
    latest.write_text(
        json.dumps(
            {"run_id": data.run_id, "path": str(report_path).replace("\\", "/")},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _pair_as_jsonl(pair: PairResult) -> dict[str, Any]:
    return {
        "task": pair.task_id,
        "case": pair.case_id,
        "repeat": pair.repeat,
        "baseline": pair.baseline,
        "candidate": pair.candidate,
        "score": pair.score,
        "category": pair.category,
        "excluded": pair.excluded,
        "reason": pair.reason,
        "judge_calls": pair.judge_calls,
        "judge_cost_usd": pair.judge_cost,
        "passes": [
            {
                "order": p.order,
                "winner": p.winner,
                "verdict": p.verdict,
                "reason": p.reason,
                "unparseable": p.unparseable,
                "retried": p.retried,
                "cached": p.cached,
                "latency_ms": p.latency_ms,
            }
            for p in pair.passes
        ],
    }


def _find_previous_report(out_root: Path, suite_name: str, current_run_id: str):
    directory = out_root / suite_name
    if not directory.is_dir():
        return None
    reports: list[dict[str, Any]] = []
    for path in directory.glob("*/report.json"):
        if path.parent.name == current_run_id:
            continue
        try:
            reports.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    if not reports:
        return None
    reports.sort(key=lambda r: (r.get("generated_at", ""), r.get("run_id", "")))
    return reports[-1]


def load_report(path_or_run_id: str | Path, out_dir: str | Path = REPORT_DIR_DEFAULT):
    """Load a report by path, by directory, or by run id under ``out_dir``."""
    candidate = Path(path_or_run_id)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    if candidate.is_dir() and (candidate / "report.json").is_file():
        return json.loads((candidate / "report.json").read_text(encoding="utf-8"))
    for path in Path(out_dir).glob(f"*/{path_or_run_id}/report.json"):
        return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"no report found at {path_or_run_id!r}")


def last_report(suite_path: str | Path, out_dir: str | Path = REPORT_DIR_DEFAULT):
    """The most recent run for this suite, read from disk. Zero spend, always."""
    suite = load_suite(suite_path)
    pointer = Path(out_dir) / suite.name / "latest.json"
    if not pointer.is_file():
        return None
    try:
        info = json.loads(pointer.read_text(encoding="utf-8"))
        return {"run_id": info["run_id"], "path": info["path"], "report": load_report(info["path"])}
    except (OSError, ValueError, KeyError, FileNotFoundError):
        return None


def compare(report_a: dict[str, Any], report_b: dict[str, Any]) -> dict[str, Any]:
    """The S9.3 delta between two reports, in the same shape the report uses."""
    from .report import diff_reports

    return diff_reports(report_a, report_b)
