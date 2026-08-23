"""Every number evalmine reports, as a pure function.

Spec: docs/spec.md S6 and S7. No network, no filesystem, no clock in this
module - which is what makes the numbers testable, and the tests are the point.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import jsonschema

from .prices import PriceRow

#: The three categories both raters are collapsed to (spec S7.4).
CATEGORIES: tuple[str, str, str] = ("baseline", "candidate", "tie")

#: Above this, a win-rate is an artefact of presentation order, not a
#: measurement (spec S7.2).
FLIP_RATE_WARNING = 0.30

#: Below this many included pairs the bootstrap interval is suppressed (S7.3).
MIN_PAIRS_FOR_CI = 8

BOOTSTRAP_RESAMPLES = 10_000

_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)


# --------------------------------------------------------------------------
# S6.1 schema verdicts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemaVerdict:
    status: str  # "pass" | "schema_fail" | "parse_fail" | "not_applicable"
    error: str | None = None
    parsed: Any = None

    @property
    def failed(self) -> bool:
        return self.status in ("schema_fail", "parse_fail")


def extract_json(text: str) -> tuple[bool, Any]:
    """Whole-response JSON, else the first fenced block. Nothing else (S6.1).

    No brace matching, no repair, no retry: a model that cannot emit JSON on
    request is telling you something the harness should not hide.
    """
    if text is None:
        return False, None
    try:
        return True, json.loads(text)
    except (ValueError, TypeError):
        pass
    match = _FENCE_RE.search(text)
    if match:
        try:
            return True, json.loads(match.group(1))
        except ValueError:
            return False, None
    return False, None


def schema_verdict(text: str, schema: dict[str, Any] | None) -> SchemaVerdict:
    if schema is None:
        return SchemaVerdict("not_applicable")
    ok, parsed = extract_json(text)
    if not ok:
        return SchemaVerdict("parse_fail", "response is not JSON and contains no JSON fence")
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda e: list(e.absolute_path))
    if errors:
        first = errors[0]
        where = "/".join(str(p) for p in first.absolute_path) or "(root)"
        return SchemaVerdict("schema_fail", f"at {where}: {first.message}", parsed)
    return SchemaVerdict("pass", None, parsed)


def schema_pass_rate(verdicts: Iterable[SchemaVerdict | str]) -> dict[str, Any]:
    counts = {"pass": 0, "schema_fail": 0, "parse_fail": 0}
    for verdict in verdicts:
        status = verdict if isinstance(verdict, str) else verdict.status
        if status in counts:
            counts[status] += 1
    total = sum(counts.values())
    return {
        "n": total,
        "pass": counts["pass"],
        "schema_fail": counts["schema_fail"],
        "parse_fail": counts["parse_fail"],
        "rate": (counts["pass"] / total) if total else None,
    }


# --------------------------------------------------------------------------
# S6.2 latency
# --------------------------------------------------------------------------


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def p95_nearest_rank(values: Sequence[float]) -> float | None:
    """Nearest rank: the value at 1-indexed position ceil(0.95 * n) (S6.2).

    For n < 20 this is the maximum, which is why the report always prints the n
    beside it - a max is not a percentile.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = math.ceil(0.95 * len(ordered))
    return float(ordered[max(1, index) - 1])


def latency_stats(values: Sequence[float], live_count: int) -> dict[str, Any]:
    n = len(values)
    return {
        "n": n,
        "p50_ms": median(values),
        "p95_ms": p95_nearest_rank(values),
        "min_ms": float(min(values)) if values else None,
        "max_ms": float(max(values)) if values else None,
        "live_fraction": (live_count / n) if n else None,
    }


# --------------------------------------------------------------------------
# S6.3 / S6.4 cost
# --------------------------------------------------------------------------


def call_cost(
    row: PriceRow,
    input_tokens: int | None,
    output_tokens: int | None,
    cached_input_tokens: int | None = 0,
) -> float | None:
    """Cost of one call, or ``None`` when the provider gave us no usage (S6.3).

    Missing usage is not zero. The caller must carry the ``None`` through to
    ``cost_incomplete`` rather than adding a convenient 0.0.
    """
    if input_tokens is None or output_tokens is None:
        return None
    cached = cached_input_tokens or 0
    return (
        input_tokens / 1e6 * row.input_per_mtok
        + output_tokens / 1e6 * row.output_per_mtok
        + cached / 1e6 * row.cached_input_per_mtok
    )


def estimate_tokens(text: str | None) -> int:
    """The documented chars/4 heuristic (S6.4). It over-estimates on purpose."""
    if not text:
        return 0
    return math.ceil(len(text) / 4)


def estimate_answer_cost(row: PriceRow, prompt: str, system: str | None, max_tokens: int) -> float:
    est_input = estimate_tokens(prompt) + estimate_tokens(system)
    return est_input / 1e6 * row.input_per_mtok + max_tokens / 1e6 * row.output_per_mtok


def estimate_judge_cost(
    row: PriceRow, rubric: str, prompt: str, answer_max_tokens: int, judge_max_tokens: int
) -> float:
    """S6.4: the judge sees the rubric, the prompt twice, and two answers."""
    est_input = (
        estimate_tokens(rubric) + 2 * estimate_tokens(prompt) + 2 * answer_max_tokens
    )
    return est_input / 1e6 * row.input_per_mtok + judge_max_tokens / 1e6 * row.output_per_mtok


# --------------------------------------------------------------------------
# S7.2 pair scoring
# --------------------------------------------------------------------------

_PAIR_TABLE: dict[tuple[str, str], tuple[float, str]] = {
    ("candidate", "candidate"): (1.0, "consistent_win"),
    ("baseline", "baseline"): (0.0, "consistent_loss"),
    ("tie", "tie"): (0.5, "tie"),
    ("candidate", "tie"): (0.75, "soft_win"),
    ("baseline", "tie"): (0.25, "soft_loss"),
    ("baseline", "candidate"): (0.5, "flip"),
}


def score_pair(pass1: str, pass2: str) -> tuple[float, str]:
    """Combine the two presentation orders into one score in [0, 1] (S7.2).

    A flip - the judge changing its mind when the answers changed places -
    cancels to 0.5 rather than being thrown away, and is counted separately so
    the report can say how often the judge was reading position, not quality.
    """
    for verdict in (pass1, pass2):
        if verdict not in CATEGORIES:
            raise ValueError(f"verdict must be one of {CATEGORIES}, got {verdict!r}")
    key = tuple(sorted((pass1, pass2)))
    return _PAIR_TABLE[key]  # type: ignore[index]


# --------------------------------------------------------------------------
# S7.3 win-rate
# --------------------------------------------------------------------------


def win_rate(scores: Sequence[float]) -> float | None:
    if not scores:
        return None
    return sum(scores) / len(scores)


def bootstrap_ci(
    scores: Sequence[float],
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
    min_n: int = MIN_PAIRS_FOR_CI,
) -> tuple[float, float] | None:
    """Percentile bootstrap over the pair scores, seeded from the suite hash.

    Bootstrap rather than a binomial interval because a pair score is not
    Bernoulli - it takes five values. ``None`` when n < ``min_n``: an interval
    over six pairs is decoration.
    """
    n = len(scores)
    if n < min_n:
        return None
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += scores[int(rng.random() * n)]
        means.append(total / n)
    means.sort()
    lo = means[max(1, math.ceil(0.025 * resamples)) - 1]
    hi = means[max(1, math.ceil(0.975 * resamples)) - 1]
    return (lo, hi)


def seed_from_suite_hash(suite_hash: str) -> int:
    """S7.3: the RNG seed is the first 8 hex characters of the suite hash."""
    return int(suite_hash[:8], 16)


# --------------------------------------------------------------------------
# S7.4 Cohen's kappa
# --------------------------------------------------------------------------


def judge_category(score: float) -> str:
    if score > 0.5:
        return "candidate"
    if score < 0.5:
        return "baseline"
    return "tie"


def kappa_band(kappa: float | None) -> str:
    """Landis-Koch band name, upper-inclusive at each boundary (S7.4, ruling O-2)."""
    if kappa is None:
        return "undefined"
    if kappa < 0.0:
        return "poor"
    if kappa <= 0.20:
        return "slight"
    if kappa <= 0.40:
        return "fair"
    if kappa <= 0.60:
        return "moderate"
    if kappa <= 0.80:
        return "substantial"
    return "almost perfect"


def format_kappa(kappa: float | None, digits: int = 2) -> str:
    if kappa is None:
        return "null (undefined)"
    return f"{kappa:.{digits}f} ({kappa_band(kappa)})"


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """Cohen's kappa over (judge_category, human_category) pairs (S7.4).

    Edge cases are reported, never smoothed: no labels, an undefined kappa when
    both raters used a single category throughout, and a negative kappa when the
    judge did worse than chance.
    """
    confusion = {h: {j: 0 for j in CATEGORIES} for h in CATEGORIES}
    judge_counts = {c: 0 for c in CATEGORIES}
    human_counts = {c: 0 for c in CATEGORIES}
    agree = 0
    n = 0
    for judge, human in pairs:
        if judge not in CATEGORIES or human not in CATEGORIES:
            raise ValueError(f"categories must be in {CATEGORIES}, got {(judge, human)!r}")
        confusion[human][judge] += 1
        judge_counts[judge] += 1
        human_counts[human] += 1
        agree += int(judge == human)
        n += 1

    base = {
        "n": n,
        "agree": agree,
        "confusion": confusion,
        "judge_counts": judge_counts,
        "human_counts": human_counts,
    }
    if n == 0:
        return {
            **base,
            "po": None,
            "pe": None,
            "kappa": None,
            "kappa_band": kappa_band(None),
            "status": "no_labels",
        }

    po = agree / n
    pe = sum((judge_counts[c] / n) * (human_counts[c] / n) for c in CATEGORIES)
    if abs(1.0 - pe) < 1e-12:
        return {
            **base,
            "po": po,
            "pe": pe,
            "kappa": None,
            "kappa_band": kappa_band(None),
            "status": "undefined_pe_1",
        }
    kappa = (po - pe) / (1.0 - pe)
    return {
        **base,
        "po": po,
        "pe": pe,
        "kappa": kappa,
        "kappa_band": kappa_band(kappa),
        "status": "computed",
    }


def calibration_status(
    kappa_result: dict[str, Any], min_kappa: float, min_labels: int
) -> dict[str, Any]:
    """Turn a kappa result into the S8 gate.

    ``headline_eligible`` is true only when the status is ``ok``. The order of
    the checks is deliberate: "we have no labels" and "kappa is undefined" are
    more fundamental findings than "kappa is a bit low", so they are reported
    first even when the label count is also short.
    """
    status = kappa_result["status"]
    n = kappa_result["n"]
    kappa = kappa_result["kappa"]

    if status == "no_labels":
        reason = "no human labels matched the pairs in this run"
        out_status = "no_labels"
    elif status == "undefined_pe_1":
        reason = (
            "kappa is undefined: both raters used a single category throughout, which "
            "demonstrates nothing about agreement"
        )
        out_status = "undefined_pe_1"
    elif n < min_labels:
        reason = f"{n} labelled pairs is fewer than the configured minimum of {min_labels}"
        out_status = "insufficient_labels"
    elif kappa is not None and kappa < min_kappa:
        reason = (
            f"kappa {kappa:.2f} ({kappa_band(kappa)}) is below the configured floor "
            f"of {min_kappa:.2f}"
        )
        out_status = "below_floor"
    else:
        reason = None
        out_status = "ok"

    return {
        **kappa_result,
        "status": out_status,
        "reason": reason,
        "min_kappa": min_kappa,
        "min_labels": min_labels,
        "headline_eligible": out_status == "ok",
        "agreement": kappa_result.get("po"),
        "n_labels": n,
    }
