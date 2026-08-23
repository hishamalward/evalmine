"""The judge protocol: two passes, position swapped, five-value pair score.

Spec: docs/spec.md S7.1-S7.2. The judge never sees a model name, a provider, a
price, a latency, or which answer is the baseline - only the task prompt, the
rubric, and the two answers.

This module makes no calls of its own. ``Judge`` is handed a callable by
``core``, which is where the cache, the cost accounting and the budget ceiling
live; that keeps the protocol testable against a double that spends nothing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .metrics import schema_verdict, score_pair
from .suite import JudgeConfig, Task

JUDGE_SYSTEM = (
    "You are grading two answers to the same task. You do not know which model "
    "produced either answer, and it does not matter. Apply the rubric literally. "
    "Reply with JSON only."
)

#: The judge is asked for exactly this. Passing it as a schema lets a provider
#: that supports structured output enforce it, and lets the fake adapter answer
#: in the right shape.
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["winner", "reason"],
    "properties": {
        "winner": {"type": "string", "enum": ["1", "2", "tie"]},
        "reason": {"type": "string"},
    },
}

# Exclusion reasons (spec S7.2, ruling O-3).
BASELINE_SCHEMA_FAIL = "baseline_schema_fail"
CANDIDATE_SCHEMA_FAIL = "candidate_schema_fail"
BOTH_SCHEMA_FAIL = "both_schema_fail"
PROVIDER_ERROR = "provider_error"
JUDGE_UNPARSEABLE = "judge_unparseable"

EXCLUSION_REASONS = (
    BASELINE_SCHEMA_FAIL,
    CANDIDATE_SCHEMA_FAIL,
    BOTH_SCHEMA_FAIL,
    PROVIDER_ERROR,
    JUDGE_UNPARSEABLE,
)


@dataclass
class JudgeCall:
    """What ``core``'s callable hands back for one judge request."""

    text: str
    cost: float | None = None
    cost_if_uncached: float | None = None
    cached: bool = False
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class PassResult:
    order: int  # 1 = baseline shown first, 2 = candidate shown first
    verdict: str | None  # "baseline" | "candidate" | "tie", or None if unparseable
    winner: str | None  # what the judge literally said: "1" | "2" | "tie"
    reason: str | None
    unparseable: bool = False
    retried: bool = False
    cached: bool = False
    cost: float | None = None
    cost_if_uncached: float | None = None
    latency_ms: int = 0


@dataclass
class PairResult:
    task_id: str
    case_id: str
    repeat: int
    baseline: str
    candidate: str
    score: float | None = None
    category: str | None = None
    excluded: bool = False
    reason: str | None = None
    passes: list[PassResult] = field(default_factory=list)
    judge_calls: int = 0
    judge_cost: float = 0.0
    judge_cost_if_uncached: float = 0.0
    judge_cached_calls: int = 0
    judge_latencies: list[int] = field(default_factory=list)

    @property
    def is_flip(self) -> bool:
        return self.category == "flip"


def compose_rubric(config: JudgeConfig, task: Task) -> str:
    if task.rubric:
        return f"{config.rubric.rstrip()}\n\n{task.rubric.rstrip()}"
    return config.rubric.rstrip()


def build_prompt(task_prompt: str, rubric: str, answer_one: str, answer_two: str) -> str:
    """The judge prompt. Nothing in here identifies either model."""
    return (
        "Two answers were given to the task below. Decide which one is better "
        "according to the rubric, or say they are equally good.\n\n"
        "=== TASK ===\n"
        f"{task_prompt.rstrip()}\n\n"
        "=== RUBRIC ===\n"
        f"{rubric.rstrip()}\n\n"
        "=== ANSWER 1 ===\n"
        f"{answer_one.rstrip()}\n\n"
        "=== ANSWER 2 ===\n"
        f"{answer_two.rstrip()}\n\n"
        "=== YOUR VERDICT ===\n"
        'Reply with JSON only, in exactly this shape: {"winner": "1" | "2" | "tie", '
        '"reason": "<one sentence>"}. Ties are expected and are not a failure.'
    )


def verdict_from_winner(winner: str, order: int) -> str:
    """Map what the judge said back to baseline/candidate for this presentation.

    Order 1 shows the baseline as Answer 1; order 2 shows the candidate as
    Answer 1. The judge is never told which is which.
    """
    if winner == "tie":
        return "tie"
    if order == 1:
        return "baseline" if winner == "1" else "candidate"
    return "candidate" if winner == "1" else "baseline"


def exclusion_reason(baseline_failed: bool, candidate_failed: bool) -> str | None:
    """Ruling O-3: a pair with an unusable answer on either side is excluded."""
    if baseline_failed and candidate_failed:
        return BOTH_SCHEMA_FAIL
    if baseline_failed:
        return BASELINE_SCHEMA_FAIL
    if candidate_failed:
        return CANDIDATE_SCHEMA_FAIL
    return None


class Judge:
    """Runs the two-pass protocol for one pair.

    ``call(prompt, system, bypass_cache) -> JudgeCall`` is injected. The one
    retry of an unparseable response bypasses the cache: the request is
    identical, so reading the cache again would only return the same
    unparseable text.
    """

    def __init__(self, config: JudgeConfig, call: Callable[..., JudgeCall]) -> None:
        self.config = config
        self.call = call

    def _one_pass(self, prompt: str, order: int) -> PassResult:
        result = PassResult(order=order, verdict=None, winner=None, reason=None)
        retried = False
        for attempt in (0, 1):
            call = self.call(prompt=prompt, system=JUDGE_SYSTEM, bypass_cache=attempt == 1)
            result.cost = (result.cost or 0.0) + (call.cost or 0.0)
            result.cost_if_uncached = (result.cost_if_uncached or 0.0) + (
                call.cost_if_uncached if call.cost_if_uncached is not None else (call.cost or 0.0)
            )
            result.cached = call.cached and attempt == 0
            result.latency_ms += call.latency_ms
            verdict = schema_verdict(call.text, JUDGE_SCHEMA)
            if verdict.status == "pass":
                winner = verdict.parsed["winner"]
                result.winner = winner
                result.verdict = verdict_from_winner(winner, order)
                result.reason = verdict.parsed.get("reason")
                result.retried = retried
                return result
            retried = True
        result.unparseable = True
        result.retried = True
        return result

    def judge_pair(
        self,
        *,
        task: Task,
        case_id: str,
        repeat: int,
        baseline: str,
        candidate: str,
        baseline_prompt: str,
        baseline_text: str,
        candidate_text: str,
    ) -> PairResult:
        rubric = compose_rubric(self.config, task)
        pair = PairResult(
            task_id=task.id,
            case_id=case_id,
            repeat=repeat,
            baseline=baseline,
            candidate=candidate,
        )

        prompts = [
            build_prompt(baseline_prompt, rubric, baseline_text, candidate_text),  # order 1
            build_prompt(baseline_prompt, rubric, candidate_text, baseline_text),  # order 2
        ]
        for order, prompt in enumerate(prompts, start=1):
            result = self._one_pass(prompt, order)
            pair.passes.append(result)
            # a retried pass cost two calls, whether or not the retry parsed
            pair.judge_calls += 2 if result.retried else 1
            pair.judge_cost += result.cost or 0.0
            pair.judge_cost_if_uncached += result.cost_if_uncached or 0.0
            pair.judge_cached_calls += int(result.cached)
            pair.judge_latencies.append(result.latency_ms)

        if any(p.unparseable for p in pair.passes):
            pair.excluded = True
            pair.reason = JUDGE_UNPARSEABLE
            return pair

        pair.score, pair.category = score_pair(
            pair.passes[0].verdict, pair.passes[1].verdict  # type: ignore[arg-type]
        )
        return pair
