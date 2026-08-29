"""Self-contained blind HTML evidence reports for v2 agent experiments."""

# ruff: noqa: E501 -- embedded HTML/CSS/JS is kept readable in its native syntax

from __future__ import annotations

import hashlib
import html
import json
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

from .experiment import ExperimentError
from .external import (
    COST_BASES,
    ExternalArtifactError,
    is_external_import,
    load_external_index,
    load_external_records,
    verify_external_import,
)
from .prices import PriceRow, PriceTable, PriceTableError, UnknownModelError, load_price_table
from .runner import RunnerError, verify_execution
from .validators import ValidationError, verify_validation
from .workspace import PreparationError, verify_prepared

REPORT_FORMAT = "evalmine-episode-report-v1"
LABEL_FORMAT = "evalmine-human-labels-v1"
RANKING_LABEL_FORMAT = "evalmine-human-rankings-v1"
CHOICES = ("A", "tie", "B", "unclear")


class ExperimentReportError(ExperimentError):
    """An episode report cannot be built or verified safely."""


@dataclass(frozen=True)
class ExperimentReportResult:
    root: Path
    html: Path
    pair_count: int
    run_count: int
    ranking_style: str
    ranking_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "html": str(self.html),
            "pair_count": self.pair_count,
            "ranking_style": self.ranking_style,
            "ranking_count": self.ranking_count,
            "review_count": (
                self.pair_count if self.ranking_style == "pairwise" else self.ranking_count
            ),
            "run_count": self.run_count,
            "provider_runners_launched": False,
        }


def episode_ab_run_keys(pair_id: str, first: str, second: str) -> tuple[str, str]:
    """Return a stable blind A/B order for two distinct run keys."""
    if first == second:
        raise ValueError("an episode pair requires two distinct run keys")
    ordered = tuple(sorted((first, second)))
    if hashlib.sha256(pair_id.encode("utf-8")).digest()[0] & 1:
        return ordered[1], ordered[0]
    return ordered


def preferred_run_by_choice(a_run: str, b_run: str) -> dict[str, str]:
    if a_run == b_run:
        raise ValueError("A and B must refer to distinct runs")
    return {"A": a_run, "B": b_run, "tie": "tie", "unclear": "unclear"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentReportError(f"cannot read report input {path} ({exc})") from exc
    if not isinstance(value, dict):
        raise ExperimentReportError(f"report input {path} is not a JSON object")
    return value


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExperimentReportError(f"cannot read report input {path} ({exc})") from exc


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ExperimentReportError(f"refusing to overwrite report file {path}") from exc
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "report-marker.json"
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _json_blob(value: dict[str, Any]) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _duration_text(duration_ms: Any) -> str:
    """Human display for a raw millisecond measurement retained in report data."""
    value = _number(duration_ms)
    if value is None:
        return "duration unavailable"
    milliseconds = max(0, round(value))
    if milliseconds < 1_000:
        return f"{milliseconds} ms"
    seconds = round(milliseconds / 1_000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def _price_model_id(runner: str, model: str, table: PriceTable) -> str:
    if model in table.rows:
        return model
    provider = {
        "claude-code": "anthropic",
        "codex-cli": "openai",
        "gemini-cli": "google",
    }.get(runner)
    return f"{provider}/{model}" if provider else model


def _cost_component(tokens: float, rate: float, multiplier: float = 1.0) -> float:
    return tokens / 1_000_000 * rate * multiplier


def _estimate_anthropic_api_equivalent(
    run: dict[str, Any], row: PriceRow
) -> tuple[dict[str, float], list[str]] | None:
    if row.cache_write_5m_per_mtok is None or row.cache_write_1h_per_mtok is None:
        return None
    components = {
        "uncached_input": 0.0,
        "cache_write_5m": 0.0,
        "cache_write_1h": 0.0,
        "cache_read": 0.0,
        "output": 0.0,
    }
    saw_usage = False
    inference_geographies: set[str] = set()
    for turn in run.get("turns", []):
        usage = turn.get("usage", {})
        input_tokens = _number(usage.get("input_tokens"))
        output_tokens = _number(usage.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            continue
        saw_usage = True
        cache_read = _number(usage.get("cache_read_input_tokens")) or 0.0
        cache_created = _number(usage.get("cache_creation_input_tokens")) or 0.0
        cache_creation = usage.get("cache_creation", {})
        if not isinstance(cache_creation, dict):
            return None
        cache_5m = _number(cache_creation.get("ephemeral_5m_input_tokens")) or 0.0
        cache_1h = _number(cache_creation.get("ephemeral_1h_input_tokens")) or 0.0
        if abs(cache_created - cache_5m - cache_1h) > 0.5:
            return None
        geography = usage.get("inference_geo")
        if isinstance(geography, str):
            inference_geographies.add(geography)
        components["uncached_input"] += _cost_component(input_tokens, row.input_per_mtok)
        components["cache_write_5m"] += _cost_component(
            cache_5m, row.cache_write_5m_per_mtok
        )
        components["cache_write_1h"] += _cost_component(
            cache_1h, row.cache_write_1h_per_mtok
        )
        components["cache_read"] += _cost_component(cache_read, row.cached_input_per_mtok)
        components["output"] += _cost_component(output_tokens, row.output_per_mtok)
    if not saw_usage:
        return None
    assumptions = [
        "Standard/global first-party API rates; subscription billing is not usage-metered.",
        "Output token counts already include any reported thinking tokens.",
    ]
    if inference_geographies:
        assumptions.append(
            "Runner inference geography: " + ", ".join(sorted(inference_geographies)) + "."
        )
    return components, assumptions


def _estimate_openai_api_equivalent(
    run: dict[str, Any], row: PriceRow
) -> tuple[dict[str, float], list[str]] | None:
    components = {
        "uncached_input": 0.0,
        "cached_input": 0.0,
        "cache_write": 0.0,
        "output": 0.0,
    }
    saw_usage = False
    long_context_turns = 0
    reasoning_reported = False
    for turn in run.get("turns", []):
        usage = turn.get("usage", {})
        input_tokens = _number(usage.get("input_tokens"))
        output_tokens = _number(usage.get("output_tokens"))
        if input_tokens is None or output_tokens is None:
            continue
        saw_usage = True
        cached = _number(usage.get("cached_input_tokens")) or 0.0
        cache_write = _number(usage.get("cache_write_input_tokens")) or 0.0
        if cached > input_tokens or cache_write < 0:
            return None
        if cache_write and row.cache_write_5m_per_mtok is None:
            return None
        uncached = input_tokens - cached
        input_multiplier = 1.0
        output_multiplier = 1.0
        if (
            row.long_context_threshold_tokens is not None
            and input_tokens > row.long_context_threshold_tokens
        ):
            long_context_turns += 1
            input_multiplier = row.long_context_input_multiplier
            output_multiplier = row.long_context_output_multiplier
        components["uncached_input"] += _cost_component(
            uncached, row.input_per_mtok, input_multiplier
        )
        components["cached_input"] += _cost_component(
            cached, row.cached_input_per_mtok, input_multiplier
        )
        components["cache_write"] += _cost_component(
            cache_write, row.cache_write_5m_per_mtok or 0.0, input_multiplier
        )
        components["output"] += _cost_component(
            output_tokens, row.output_per_mtok, output_multiplier
        )
        reasoning_reported = reasoning_reported or bool(
            _number(usage.get("reasoning_output_tokens"))
        )
    if not saw_usage:
        return None
    assumptions = [
        "Reported input tokens include cached input; uncached input is their difference.",
        "Subscription billing is not usage-metered.",
    ]
    if long_context_turns:
        assumptions.append(
            f"Published long-context multipliers applied to {long_context_turns} runner "
            "turn(s), including both cached and uncached input."
        )
    if reasoning_reported:
        assumptions.append("Reasoning output tokens are a subset of output, not added twice.")
    return components, assumptions


def _api_list_price_equivalent(
    run: dict[str, Any], table: PriceTable | None
) -> dict[str, Any]:
    if run.get("billing", {}).get("basis") != "subscription":
        return {"status": "not-applicable"}
    if table is None:
        return {"status": "unavailable", "reason": "price-table-unavailable"}
    price_model = _price_model_id(
        str(run.get("runner", "")), str(run.get("requested_model", "")), table
    )
    try:
        row = table.get(price_model)
    except UnknownModelError:
        return {
            "status": "unavailable",
            "reason": "unknown-price-row",
            "model": price_model,
            "price_table": table.filename,
        }
    if price_model.startswith("anthropic/"):
        estimate = _estimate_anthropic_api_equivalent(run, row)
    elif price_model.startswith("openai/"):
        estimate = _estimate_openai_api_equivalent(run, row)
    else:
        estimate = None
    if estimate is None:
        return {
            "status": "unavailable",
            "reason": "usage-breakdown-insufficient",
            "model": price_model,
            "price_table": table.filename,
        }
    components, assumptions = estimate
    usd = sum(components.values())
    meter_equivalent = _number(run.get("billing", {}).get("meter_equivalent_usd"))
    return {
        "status": "estimated",
        "usd": usd,
        "currency": table.currency,
        "basis": "reported token usage multiplied by pinned API list prices; not a subscription charge",
        "model": price_model,
        "price_table": table.filename,
        "price_table_pinned": table.pinned,
        "source": row.source,
        "read_on": row.read_on,
        "price_notes": row.notes,
        "components_usd": components,
        "rates_per_mtok": {
            "input": row.input_per_mtok,
            "cached_input": row.cached_input_per_mtok,
            "cache_write_5m": row.cache_write_5m_per_mtok,
            "cache_write_1h": row.cache_write_1h_per_mtok,
            "output": row.output_per_mtok,
        },
        "runner_meter_equivalent_usd": meter_equivalent,
        "runner_meter_equivalent_delta_usd": (
            usd - meter_equivalent if meter_equivalent is not None else None
        ),
        "assumptions": assumptions + ([row.notes] if row.notes else []),
    }


def _load_validator_results(validation_dir: Path | None) -> tuple[dict[str, Any] | None, list]:
    if validation_dir is None or not validation_dir.is_dir():
        return None, []
    summary = _read_json(validation_dir / "run.json")
    results: list[dict[str, Any]] = []
    for validator_id in summary.get("validator_order", []):
        result = _read_json(validation_dir / f"{validator_id}.json")
        if result.get("type") == "repository-diff" and result.get("patch"):
            result["patch_text"] = _read_text(validation_dir / result["patch"])
        if result.get("type") == "command":
            result["stdout_text"] = _read_text(validation_dir / result["stdout"])
            result["stderr_text"] = _read_text(validation_dir / result["stderr"])
        results.append(result)
    return summary, results


def _episode_prompts(root: Path, plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    index = _read_json(root / "inputs" / "index.json")
    blobs = {
        str(entry.get("logical")): str(entry.get("blob"))
        for entry in index.get("entries", [])
        if entry.get("kind") == "prompt"
    }
    episodes: list[dict[str, Any]] = []
    by_episode: dict[str, list[str]] = {}
    for episode in plan["episodes"]:
        prompts: list[dict[str, Any]] = []
        texts: list[str] = []
        for turn_number, turn in enumerate(episode["turns"], 1):
            logical = f"episode/{episode['id']}/turn/{turn_number}"
            relative = blobs.get(logical)
            if not relative:
                raise ExperimentReportError(f"report input index is missing {logical}")
            prompt = _read_text(root / "inputs" / relative)
            texts.append(prompt)
            prompts.append(
                {
                    "turn": turn_number,
                    "prompt": prompt,
                    "prompt_sha256": turn["prompt_sha256"],
                }
            )
        by_episode[episode["id"]] = texts
        episodes.append(
            {
                "id": episode["id"],
                "title": episode.get("title"),
                "prompts": prompts,
            }
        )
    return episodes, by_episode


def _run_view(
    root: Path,
    planned: dict[str, Any],
    validation_exists: bool,
    prompts_by_episode: dict[str, list[str]],
    price_table: PriceTable | None,
) -> dict[str, Any]:
    run_key = planned["run_key"]
    prepared_dir = root / "runs" / run_key
    execution_dir = root / "execution" / "runs" / run_key
    run = _read_json(prepared_dir / "run.json")
    treatment = _read_json(prepared_dir / "treatment.json")
    execution = _read_json(execution_dir / "run.json")
    turns: list[dict[str, Any]] = []
    for index in range(1, int(run["turns"]) + 1):
        summary_path = execution_dir / f"turn-{index:03d}.json"
        if not summary_path.is_file():
            continue
        summary = _read_json(summary_path)
        final_path = execution_dir / f"turn-{index:03d}.final.txt"
        summary["prompt"] = prompts_by_episode[run["episode"]][index - 1]
        summary["final"] = _read_text(final_path) if final_path.is_file() else ""
        turns.append(summary)
    validation_dir = root / "validation" / "runs" / run_key if validation_exists else None
    validation, validators = _load_validator_results(validation_dir)
    final = turns[-1]["final"] if turns else ""
    view = {
        "run_key": run_key,
        "sequence": run["sequence"],
        "block": run["block"],
        "arm": run["arm"],
        "episode": run["episode"],
        "repeat": run["repeat"],
        "runner": run["runner"],
        "requested_model": execution.get("requested_model", run["model"]),
        "observed_model": execution.get("observed_model"),
        "observed_model_source": execution.get("observed_model_source"),
        "model_identity": execution.get("model_identity", {}),
        "auth": run["auth"],
        "treatment": treatment,
        "execution_status": execution.get("status"),
        "execution_error": execution.get("error"),
        "duration_ms": execution.get("duration_ms"),
        "turns_completed": execution.get("turns_completed", 0),
        "turns_planned": execution.get("turns_planned", run["turns"]),
        "tool_count": sum(len(turn.get("tools", [])) for turn in turns),
        "usage": execution.get("usage", {}),
        "billing": execution.get(
            "billing",
            {"basis": "unknown", "reported_cost_usd": None, "dollar_cost_status": "unavailable"},
        ),
        "turns": turns,
        "final": final,
        "validation": validation,
        "validators": validators,
        "validation_verdict": validation.get("verdict") if validation else "not-run",
    }
    view["billing"] = dict(view["billing"])
    view["billing"]["api_list_price_equivalent"] = _api_list_price_equivalent(
        view, price_table
    )
    return view


def _pair_views(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_block.setdefault(run["block"], []).append(run)
    pairs: list[dict[str, Any]] = []
    for block, block_runs in by_block.items():
        for first, second in combinations(sorted(block_runs, key=lambda item: item["run_key"]), 2):
            identity = f"{block}|{first['run_key']}|{second['run_key']}"
            pair_id = "pair-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            a_key, b_key = episode_ab_run_keys(pair_id, first["run_key"], second["run_key"])
            by_key = {first["run_key"]: first, second["run_key"]: second}
            pairs.append(
                {
                    "pair_id": pair_id,
                    "block": block,
                    "episode": first["episode"],
                    "repeat": first["repeat"],
                    "a_run_key": a_key,
                    "b_run_key": b_key,
                    "a": by_key[a_key],
                    "b": by_key[b_key],
                    "preferred_run_by_choice": preferred_run_by_choice(a_key, b_key),
                }
            )
    return pairs


def _outcome_label(index: int) -> str:
    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


def _ranking_views(
    runs: list[dict[str, Any]], fields: list[str] | None = None
) -> list[dict[str, Any]]:
    by_block: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        by_block.setdefault(run["block"], []).append(run)
    rankings: list[dict[str, Any]] = []
    for block, block_runs in sorted(by_block.items()):
        run_keys = sorted(run["run_key"] for run in block_runs)
        identity = f"{block}|{'|'.join(run_keys)}"
        ranking_id = "ranking-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        ordered = sorted(
            block_runs,
            key=lambda run: hashlib.sha256(
                f"{ranking_id}|{run['run_key']}".encode()
            ).hexdigest(),
        )
        outcomes = [
            {
                "label": _outcome_label(index),
                "run_key": run["run_key"],
                "run": run,
            }
            for index, run in enumerate(ordered)
        ]
        first = ordered[0]
        rankings.append(
            {
                "ranking_id": ranking_id,
                "block": block,
                "episode": first["episode"],
                "repeat": first["repeat"],
                "outcomes": outcomes,
                "run_key_by_label": {
                    outcome["label"]: outcome["run_key"] for outcome in outcomes
                },
                "fields": list(fields or []),
            }
        )
    return rankings


def _receipt_totals(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for basis in COST_BASES:
        values = [
            run.get("billing", {}).get("receipts", {}).get(basis)
            for run in runs
        ]
        observed = [value for value in values if isinstance(value, dict)]
        totals[basis] = {
            "usd": sum(float(value["usd"]) for value in observed),
            "receipt_count": len(observed),
            "artifact_count": len(runs),
            "complete": len(observed) == len(runs),
        }
    estimated = totals["estimated"]["usd"]
    ledger = totals["ledger"]["usd"]
    dashboard = totals["dashboard_observed"]["usd"]
    return {
        **totals,
        "reconciliation": {
            "ledger_to_estimated": ledger / estimated if estimated > 0 else None,
            "dashboard_to_estimated": dashboard / estimated if estimated > 0 else None,
            "dashboard_to_ledger": dashboard / ledger if ledger > 0 else None,
        },
    }


def _external_report_data(
    root: Path,
    *,
    generated_at: str | None,
    ranking_style: str | None,
) -> dict[str, Any]:
    """Translate immutable external records into the common blind-review view."""
    verification = verify_external_import(root)
    index = load_external_index(root, verify=False)
    records = load_external_records(root, verify=False)
    evaluation = index["evaluation"]
    planned_style = str(evaluation.get("ranking_style", "pairwise"))
    if ranking_style is not None and ranking_style not in {"pairwise", "n-way"}:
        raise ExperimentReportError(f"unknown ranking style {ranking_style!r}")
    effective_style = ranking_style or planned_style
    fields = [str(item) for item in evaluation.get("fields", [])]
    runs: list[dict[str, Any]] = []
    episodes_by_block: dict[str, dict[str, Any]] = {}
    for sequence, record in enumerate(records, 1):
        block_identity = json.dumps(
            [record["lane"], record["item_id"], record["account_id"]],
            separators=(",", ":"),
            ensure_ascii=False,
        )
        block_hash = hashlib.sha256(block_identity.encode("utf-8")).hexdigest()[:20]
        block = f"external-{block_hash}"
        episode = f"item-{block_hash}"
        run_identity = f"{index['input_hash']}|{block_identity}|{record['condition']['id']}"
        run_key = "external-" + hashlib.sha256(run_identity.encode("utf-8")).hexdigest()[:24]
        condition = dict(record["condition"])
        field_values = record.get("fields")
        if field_values is None and isinstance(record.get("output"), dict):
            field_values = record["output"]
        receipts = dict(record.get("cost_receipts", {}))
        reconciliation = dict(record.get("cost_reconciliation", {}))
        usage = dict(record.get("usage", {}))
        duration = record.get("duration_ms")
        turn = {
            "turn": 1,
            "status": "succeeded",
            "duration_ms": duration,
            "tools": [],
            "usage": usage,
            "prompt": record["prompt"],
            "final": record["output_text"],
            "external_provenance": record["provenance"],
        }
        runs.append(
            {
                "run_key": run_key,
                "sequence": sequence,
                "block": block,
                "arm": condition["id"],
                "episode": episode,
                "repeat": 1,
                "runner": "external-artifact",
                "requested_model": condition["model"],
                "observed_model": condition["model"],
                "observed_model_source": "external-record",
                "model_identity": {
                    "status": "recorded",
                    "source": "external-record",
                    "confidence": "receipt-declared",
                    "requested_model": condition["model"],
                    "observed_model": condition["model"],
                    "matches_requested": True,
                },
                "auth": "external",
                "treatment": {
                    "instructions": condition["prompt_variant"],
                    "plugins": "not-applicable",
                    "settings": {
                        "width": condition["width"],
                        **dict(condition.get("metadata", {})),
                    },
                },
                "condition": condition,
                "identity_reveal_allowed": False,
                "external": {
                    "lane": record["lane"],
                    "item_id": record["item_id"],
                    "account_id": record["account_id"],
                    "metadata": dict(record.get("metadata", {})),
                    "provenance": record["provenance"],
                },
                "field_values": dict(field_values or {}),
                "execution_status": "succeeded",
                "execution_error": None,
                "duration_ms": duration,
                "turns_completed": 1,
                "turns_planned": 1,
                "tool_count": 0,
                "usage": usage,
                "billing": {
                    "basis": "external-receipts",
                    "reported_cost_usd": None,
                    "dollar_cost_status": "basis-separated",
                    "receipts": receipts,
                    "reconciliation": reconciliation,
                    "api_list_price_equivalent": {
                        "status": "unavailable",
                        "reason": "external-receipts-are-reported-separately",
                    },
                },
                "turns": [turn],
                "final": record["output_text"],
                "validation": None,
                "validators": [],
                "validation_verdict": "not-run",
            }
        )
        episodes_by_block.setdefault(
            block,
            {
                "id": episode,
                "title": f"{record['lane']} · {record['item_id']} · account {record['account_id']}",
                "prompts": [
                    {
                        "turn": 1,
                        "prompt": record["prompt"],
                        "prompt_sha256": hashlib.sha256(
                            record["prompt"].encode("utf-8")
                        ).hexdigest(),
                    }
                ],
                "external": {
                    "lane": record["lane"],
                    "item_id": record["item_id"],
                    "account_id": record["account_id"],
                },
            },
        )
    plan = {
        "arms": [
            {
                "id": condition_id,
                "runner": "external-artifact",
                "model": condition["model"],
                "auth": "external",
                "configuration": {
                    "instructions": condition["prompt_variant"],
                    "plugins": "not-applicable",
                    "settings": {
                        "width": condition["width"],
                        **dict(condition.get("metadata", {})),
                    },
                },
            }
            for condition_id, condition in sorted(index["conditions"].items())
        ]
    }
    arms = _arm_summaries(plan, runs)
    for arm in arms:
        arm_runs = [run for run in runs if run["arm"] == arm["arm"]]
        arm["external_cost_receipts"] = _receipt_totals(arm_runs)
        arm["condition"] = index["conditions"][arm["arm"]]
    cost_summary = _receipt_totals(runs)
    pairs = _pair_views(runs)
    rankings = _ranking_views(runs, fields=fields)
    return {
        "format": REPORT_FORMAT,
        "source_format": index["format"],
        "label_format": LABEL_FORMAT if effective_style == "pairwise" else RANKING_LABEL_FORMAT,
        "generated_at": generated_at or _now(),
        "prepared_root": str(root),
        "plan_id": index["plan_id"],
        "experiment": index["experiment"],
        "question": index["question"],
        "objectives": list(evaluation["objectives"]),
        "blind": "full-condition",
        "human": {
            **dict(evaluation["human"]),
            "coverage": evaluation["human"].get("coverage", "calibration-subset"),
        },
        "judge": dict(evaluation["judge"]),
        "ranking_style": effective_style,
        "planned_ranking_style": planned_style,
        "ranking_style_source": "operator-override" if ranking_style else "plan",
        "fields": fields,
        "episodes": list(episodes_by_block.values()),
        "pricing": {
            "status": "external-receipts",
            "basis_policy": "separate-never-blend",
            "receipt_totals": cost_summary,
            "run_count": len(runs),
        },
        "verification": {
            "external_import": verification,
            "preparation": verification,
            "execution": {"status": "external-completed", "provider_calls": False},
            "validation": None,
        },
        "runs": runs,
        "arms": arms,
        "pairs": pairs,
        "rankings": rankings,
        "pair_count": len(pairs),
        "ranking_count": len(rankings),
        "run_count": len(runs),
    }


def _arm_summaries(plan: dict[str, Any], runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms = {arm["id"]: arm for arm in plan["arms"]}
    summaries: list[dict[str, Any]] = []
    for arm_id, arm in arms.items():
        rows = [run for run in runs if run["arm"] == arm_id]
        durations = sorted(
            int(run["duration_ms"]) for run in rows if isinstance(run.get("duration_ms"), int)
        )
        median_duration = median(durations) if durations else None
        summaries.append(
            {
                "arm": arm_id,
                "runner": arm["runner"],
                "model": arm["model"],
                "auth": arm["auth"],
                "configuration": arm["configuration"],
                "runs": len(rows),
                "execution_succeeded": sum(run["execution_status"] == "succeeded" for run in rows),
                "validation_passed": sum(run["validation_verdict"] == "passed" for run in rows),
                "median_duration_ms": median_duration,
                "billing_basis": sorted(
                    {str(run.get("billing", {}).get("basis", "unknown")) for run in rows}
                ),
                "reported_cost_usd": sum(
                    float(run.get("billing", {}).get("reported_cost_usd") or 0)
                    for run in rows
                ),
            }
        )
    return summaries


def build_experiment_report_data(
    root: str | Path,
    *,
    generated_at: str | None = None,
    ranking_style: str | None = None,
    prices_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the serializable report view after verifying its source envelopes."""
    resolved_input = Path(root).resolve()
    if is_external_import(resolved_input):
        try:
            return _external_report_data(
                resolved_input,
                generated_at=generated_at,
                ranking_style=ranking_style,
            )
        except ExternalArtifactError as exc:
            raise ExperimentReportError(str(exc)) from exc
    try:
        prepared_verification = verify_prepared(root)
        execution_verification = verify_execution(root)
    except (PreparationError, RunnerError) as exc:
        raise ExperimentReportError(str(exc)) from exc
    resolved = Path(root).resolve()
    validation_exists = (resolved / "validation").is_dir()
    validation_verification = None
    if validation_exists:
        try:
            validation_verification = verify_validation(resolved)
        except ValidationError as exc:
            raise ExperimentReportError(str(exc)) from exc
    plan = _read_json(resolved / "plan.json")
    price_table: PriceTable | None = None
    pricing: dict[str, Any]
    try:
        price_table = load_price_table(prices_path)
        pricing = {
            "status": "loaded",
            "price_table": price_table.filename,
            "pinned": price_table.pinned,
            "currency": price_table.currency,
            "verified": price_table.verified,
            "sha256": _file_hash(price_table.path),
        }
    except PriceTableError as exc:
        pricing = {"status": "unavailable", "reason": str(exc)}
    episodes, prompts_by_episode = _episode_prompts(resolved, plan)
    runs = [
        _run_view(resolved, planned, validation_exists, prompts_by_episode, price_table)
        for planned in plan["runs"]
    ]
    estimated_prices = [
        run["billing"]["api_list_price_equivalent"]
        for run in runs
        if run["billing"]["api_list_price_equivalent"].get("status") == "estimated"
    ]
    pricing["estimated_run_count"] = len(estimated_prices)
    pricing["run_count"] = len(runs)
    pricing["api_list_price_equivalent_total_usd"] = sum(
        float(estimate["usd"]) for estimate in estimated_prices
    )
    pairs = _pair_views(runs)
    rankings = _ranking_views(runs)
    planned_ranking_style = str(plan["evaluation"].get("ranking_style", "pairwise"))
    if ranking_style is not None and ranking_style not in {"pairwise", "n-way"}:
        raise ExperimentReportError(f"unknown ranking style {ranking_style!r}")
    effective_ranking_style = ranking_style or planned_ranking_style
    return {
        "format": REPORT_FORMAT,
        "label_format": (
            LABEL_FORMAT if effective_ranking_style == "pairwise" else RANKING_LABEL_FORMAT
        ),
        "generated_at": generated_at or _now(),
        "prepared_root": str(resolved),
        "plan_id": plan["plan_id"],
        "experiment": plan["experiment"],
        "question": plan["question"],
        "objectives": plan["evaluation"]["objectives"],
        "blind": plan["evaluation"]["blind"],
        "human": plan["evaluation"]["human"],
        "judge": plan["evaluation"]["judge"],
        "ranking_style": effective_ranking_style,
        "planned_ranking_style": planned_ranking_style,
        "ranking_style_source": "operator-override" if ranking_style else "plan",
        "episodes": episodes,
        "pricing": pricing,
        "verification": {
            "preparation": prepared_verification,
            "execution": execution_verification,
            "validation": validation_verification,
        },
        "runs": runs,
        "arms": _arm_summaries(plan, runs),
        "pairs": pairs,
        "rankings": rankings,
        "pair_count": len(pairs),
        "ranking_count": len(rankings),
        "run_count": len(runs),
    }


def _status(value: str) -> str:
    css = "pass" if value in {"succeeded", "passed", "completed"} else "fail"
    if value == "not-run":
        css = "muted"
    return f'<span class="status {css}">{_esc(value)}</span>'


def _objective_check_status(run: dict[str, Any]) -> str:
    results = run.get("validators", [])
    if not results:
        return '<span class="status muted">checks not run</span>'
    passed = sum(result.get("status") == "passed" for result in results)
    css = "pass" if passed == len(results) else "warn"
    return f'<span class="status {css}">checks {passed}/{len(results)}</span>'


def _billing_text(run: dict[str, Any]) -> str:
    billing = run.get("billing", {})
    basis = billing.get("basis")
    cost = billing.get("reported_cost_usd")
    if basis == "subscription":
        estimate = billing.get("api_list_price_equivalent", {})
        if estimate.get("status") == "estimated":
            return "Subscription"
        return "Subscription · per-run dollar cost unavailable"
    if basis == "api-metered" and isinstance(cost, (int, float)):
        return f"API metered · ${cost:.4f} reported"
    if basis == "local":
        return "Local execution · no provider charge"
    if basis == "external-receipts":
        return "External receipts · values hidden until identities are revealed"
    return "Per-run dollar cost unavailable"


def _external_receipts_html(run: dict[str, Any]) -> str:
    billing = run.get("billing", {})
    if billing.get("basis") != "external-receipts":
        return ""
    if run.get("identity_reveal_allowed") is False:
        return ""
    receipts = billing.get("receipts", {})
    cards = []
    for basis in COST_BASES:
        receipt = receipts.get(basis)
        label = basis.replace("_", " ")
        if isinstance(receipt, dict):
            cards.append(
                f"<li><b>{_esc(label)}</b>: ${float(receipt['usd']):.6f} USD"
                f" · {_esc(receipt.get('source'))}</li>"
            )
        else:
            cards.append(f"<li><b>{_esc(label)}</b>: unavailable</li>")
    ratios = billing.get("reconciliation", {})
    ratio_rows = "".join(
        f"<li>{_esc(name.replace('_', ' '))}: "
        + (f"{float(value):.4f}×" if isinstance(value, (int, float)) else "unavailable")
        + "</li>"
        for name, value in ratios.items()
    )
    return (
        '<details class="pricing identity"><summary>External cost receipts by basis</summary>'
        "<p>These bases are shown separately and are never added or averaged together.</p>"
        f"<ul>{''.join(cards)}</ul><p><b>Reconciliation ratios</b></p><ul>{ratio_rows}</ul>"
        "</details>"
    )


def _pricing_details_html(run: dict[str, Any]) -> str:
    estimate = run.get("billing", {}).get("api_list_price_equivalent", {})
    if estimate.get("status") != "estimated":
        return ""
    components = "".join(
        f"<li>{_esc(name.replace('_', ' '))}: ${float(value):.4f}</li>"
        for name, value in estimate.get("components_usd", {}).items()
        if float(value) != 0.0
    )
    assumptions = "".join(
        f"<li>{_esc(item)}</li>" for item in estimate.get("assumptions", [])
    )
    meter = estimate.get("runner_meter_equivalent_usd")
    meter_note = (
        f"<p>Runner meter-equivalent: ${float(meter):.4f}.</p>"
        if isinstance(meter, (int, float))
        else ""
    )
    return (
        '<details class="pricing identity"><summary>'
        f"API list-price equivalent ≈ ${float(estimate['usd']):.2f}</summary>"
        f"<p>This is not the subscription charge. It is reported token usage multiplied "
        f"by {_esc(estimate.get('price_table'))}, pinned {_esc(estimate.get('price_table_pinned'))}."
        f"</p><ul>{components}</ul>{meter_note}<ul>{assumptions}</ul></details>"
    )


def _pricing_summary_html(report: dict[str, Any]) -> str:
    pricing = report.get("pricing", {})
    if pricing.get("status") == "external-receipts":
        totals = pricing.get("receipt_totals", {})
        rows = "".join(
            f"<li>{_esc(basis.replace('_', ' '))}: ${float(totals[basis]['usd']):.6f} "
            f"across {_esc(totals[basis]['receipt_count'])}/{_esc(totals[basis]['artifact_count'])} "
            "artifacts</li>"
            for basis in COST_BASES
        )
        ratios = totals.get("reconciliation", {})
        ratio_rows = "".join(
            f"<li>{_esc(name.replace('_', ' '))}: "
            + (f"{float(value):.4f}×" if isinstance(value, (int, float)) else "unavailable")
            + "</li>"
            for name, value in ratios.items()
        )
        return (
            '<div class="notice identity"><b>External cost receipts (identity-revealing):</b>'
            f"<ul>{rows}</ul><p>Bases remain separate; no blended total is reported.</p>"
            f"<ul>{ratio_rows}</ul></div>"
        )
    estimated = pricing.get("estimated_run_count")
    total = pricing.get("api_list_price_equivalent_total_usd")
    if not isinstance(estimated, int) or not isinstance(total, (int, float)) or not estimated:
        return ""
    return (
        '<div class="notice identity"><b>Pricing estimate:</b> '
        f"API list-price equivalent ≈ ${float(total):.2f} across {estimated}/"
        f"{_esc(pricing.get('run_count'))} runs. This is not the subscription charge; "
        f"per-run token and cache breakdowns appear with each revealed outcome.</div>"
    )


def _validator_html(result: dict[str, Any]) -> str:
    kind = result.get("type")
    body = ""
    if kind == "repository-diff":
        changes = result.get("changes", [])
        filtered = result.get("filtered_changes", [])
        names = "".join(
            f"<li><code>{_esc(item['path'])}</code> · {_esc(item['change'])}</li>"
            for item in changes
        )
        filtered_names = "".join(
            f"<li><code>{_esc(item['path'])}</code> · {_esc(item['change'])}</li>"
            for item in filtered
        )
        body = (
            f"<p>{len(changes)} changed file(s); patch omissions "
            f"{len(result.get('patch_omissions', []))}.</p><ul>{names}</ul>"
            f"<p>{len(filtered)} change(s) filtered by the declared validator scope.</p>"
            f"<ul>{filtered_names}</ul>"
            f"<pre>{_esc(result.get('patch_text', ''))}</pre>"
        )
    elif kind in {"required-files", "required-sections"}:
        body = (
            "<ul>"
            + "".join(
                f"<li>{_esc(item.get('path') or item.get('section'))}: "
                f"{'present' if item.get('present') else 'missing'}</li>"
                for item in result.get("checks", [])
            )
            + "</ul>"
        )
    elif kind == "command":
        body = (
            f"<p><code>{_esc(' '.join(result.get('argv', [])))}</code> · exit "
            f"{_esc(result.get('exit_code'))} · {_esc(_duration_text(result.get('duration_ms')))}</p>"
            f"<pre>{_esc(result.get('stdout_text', ''))}</pre>"
            f'<pre class="stderr">{_esc(result.get("stderr_text", ""))}</pre>'
        )
    return (
        f"<details><summary>{_status(str(result.get('status', 'unknown')))} "
        f"{_esc(result.get('id'))} <small>{_esc(kind)}</small></summary>{body}</details>"
    )


def _outcome_html(label: str, run: dict[str, Any]) -> str:
    turns = "".join(
        f"<details><summary>Turn {_esc(turn['turn'])} · {_status(turn['status'])} · "
        f"{_esc(_duration_text(turn.get('duration_ms')))} · {len(turn.get('tools', []))} tools</summary>"
        f"<h5>Prompt</h5><pre class=\"prompt\">{_esc(turn.get('prompt', ''))}</pre>"
        f"<h5>Response</h5><pre>{_esc(turn.get('final', ''))}</pre></details>"
        for turn in run["turns"]
    )
    validators = "".join(_validator_html(result) for result in run["validators"])
    error = (
        f'<div class="error">{_esc(run["execution_error"])}</div>'
        if run.get("execution_error")
        else ""
    )
    condition = run.get("condition")
    if run.get("identity_reveal_allowed") is False:
        identity = ""
    elif isinstance(condition, dict):
        identity = (
            f'<div class="identity"><b>{_esc(condition.get("id"))}</b><br>'
            f"model=<code>{_esc(condition.get('model'))}</code> · "
            f"prompt variant=<code>{_esc(condition.get('prompt_variant'))}</code> · "
            f"width=<code>{_esc(condition.get('width'))}</code></div>"
        )
    else:
        identity = (
            f'<div class="identity"><b>{_esc(run["arm"])}</b><br>'
            f"<code>{_esc(run['runner'])}</code> · requested "
            f"<code>{_esc(run['requested_model'])}</code> · observed "
            f"<code>{_esc(run.get('observed_model') or 'unavailable')}</code>"
            f" ({_esc(run.get('model_identity', {}).get('confidence') or 'unavailable')})"
            f"<br>instructions={_esc(run['treatment'].get('instructions'))} · "
            f"plugins={_esc(run['treatment'].get('plugins'))}</div>"
        )
    return f"""
      <article class="outcome">
        <header><div><span class="outcome-letter">Outcome {_esc(label)}</span>{identity}</div>
          <div>{_status(run["execution_status"])} {_objective_check_status(run)}</div></header>
        {error}
        <div class="metrics"><span>{_esc(_duration_text(run["duration_ms"]))}</span><span>{_esc(run["tool_count"])} tools</span><span>{_esc(run["turns_completed"])}/{_esc(run["turns_planned"])} turns</span><span>{_esc(_billing_text(run))}</span></div>
        {_pricing_details_html(run)}
        {_external_receipts_html(run)}
        <h4>Final response</h4><pre class="final">{_esc(run["final"])}</pre>
        <details><summary>Trajectory</summary>{turns or "<p>No completed turn evidence.</p>"}</details>
        <details><summary>Objective checks</summary>{validators or "<p>Not run.</p>"}</details>
      </article>"""


def _pair_html(pair: dict[str, Any], index: int) -> str:
    return f"""
    <section class="pair" data-pair="{_esc(pair["pair_id"])}">
      <div class="pair-head"><div><small>Pair {index} · {_esc(pair["episode"])} · repeat {_esc(pair["repeat"])}</small><h3>Which outcome would you keep?</h3></div><span class="saved" aria-live="polite"></span></div>
      <div class="outcomes">{_outcome_html("A", pair["a"])}{_outcome_html("B", pair["b"])}</div>
      <div class="label-controls" role="group" aria-label="Label this pair">
        <button data-choice="A">Prefer A</button><button data-choice="tie">Tie</button><button data-choice="B">Prefer B</button><button data-choice="unclear">Unclear</button>
        <label>Why? <textarea class="note" rows="2" placeholder="Optional decision note"></textarea></label>
      </div>
    </section>"""


def _ranking_html(ranking: dict[str, Any], index: int) -> str:
    outcomes = "".join(
        _outcome_html(outcome["label"], outcome["run"])
        for outcome in ranking["outcomes"]
    )
    rank_options = "".join(
        f'<option value="{rank}">{rank}</option>'
        for rank in range(1, len(ranking["outcomes"]) + 1)
    )
    controls = "".join(
        f'<label>Outcome {_esc(outcome["label"])} <select data-rank-label="{_esc(outcome["label"])}">'
        f'<option value="">Choose rank</option>{rank_options}</select></label>'
        for outcome in ranking["outcomes"]
    )
    field_rows = ""
    for field in ranking.get("fields", []):
        value_cells = "".join(
            f"<td><pre>{_esc(json.dumps(outcome['run'].get('field_values', {}).get(field), ensure_ascii=False, sort_keys=True))}</pre></td>"
            for outcome in ranking["outcomes"]
        )
        best_options = "".join(
            f'<option value="{_esc(outcome["label"])}">Outcome {_esc(outcome["label"])}</option>'
            for outcome in ranking["outcomes"]
        )
        wrong_controls = "".join(
            f'<label><input type="checkbox" data-field-wrong="{_esc(field)}" '
            f'data-field-wrong-label="{_esc(outcome["label"])}"> '
            f'{_esc(outcome["label"])} wrong</label>'
            for outcome in ranking["outcomes"]
        )
        field_rows += (
            f'<tr data-field-row="{_esc(field)}"><th>{_esc(field)}</th>{value_cells}'
            f'<td><select data-field-best="{_esc(field)}"><option value="">Choose best</option>'
            f'{best_options}<option value="tie">Tie / no single best</option>'
            f'<option value="unclear">Unclear</option></select></td>'
            f'<td class="wrong-controls">{wrong_controls}</td></tr>'
        )
    field_table = ""
    if field_rows:
        outcome_headers = "".join(
            f"<th>Outcome {_esc(outcome['label'])}</th>" for outcome in ranking["outcomes"]
        )
        field_table = (
            '<div class="field-review"><h4>Best and wrong by field</h4><table><thead><tr>'
            f"<th>Field</th>{outcome_headers}<th>Best</th><th>Flag wrong</th>"
            f"</tr></thead><tbody>{field_rows}</tbody></table></div>"
        )
    return f"""
    <section class="pair ranking" data-ranking="{_esc(ranking["ranking_id"])}">
      <div class="pair-head"><div><small>Ranking {index} · {_esc(ranking["episode"])} · repeat {_esc(ranking["repeat"])}</small><h3>Rank every outcome from best to worst</h3></div><span class="saved" aria-live="polite"></span></div>
      <div class="outcomes n-way">{outcomes}</div>
      {field_table}
      <div class="label-controls ranking-controls" role="group" aria-label="Rank these outcomes">
        {controls}<button data-unclear>Mark unclear</button>
        <label>Why? <textarea class="note" rows="2" placeholder="Optional decision note"></textarea></label>
      </div>
    </section>"""


def _prompts_html(episodes: list[dict[str, Any]]) -> str:
    return "".join(
        f'<article class="task"><h3>{_esc(episode.get("title") or episode["id"])}</h3>'
        + "".join(
            f'<details open><summary>Turn {_esc(turn["turn"])} prompt</summary>'
            f'<pre class="prompt">{_esc(turn["prompt"])}</pre></details>'
            for turn in episode["prompts"]
        )
        + "</article>"
        for episode in episodes
    )


def render_experiment_report_html(report: dict[str, Any]) -> str:
    """Render one self-contained responsive report and blind labeling queue."""
    external_blind = report.get("source_format") == "evalmine-external-artifacts-v1"
    arms = "".join(
        '<div class="arm-card">'
        + (
            ""
            if external_blind
            else f'<div class="identity"><b>{_esc(arm["arm"])}</b><br>'
            f"<code>{_esc(arm['runner'])}</code> · <code>{_esc(arm['model'])}</code></div>"
        )
        + f"<b>{arm['execution_succeeded']}/{arm['runs']}</b> executions succeeded<br>"
        f"<b>{arm['validation_passed']}/{arm['runs']}</b> passed all objective checks<br>"
        f"<span>{_esc(_duration_text(arm['median_duration_ms']))} median</span></div>"
        for arm in report["arms"]
    )
    pairs = "".join(_pair_html(pair, index) for index, pair in enumerate(report["pairs"], 1))
    rankings = "".join(
        _ranking_html(ranking, index) for index, ranking in enumerate(report["rankings"], 1)
    )
    ranking_style = report.get("ranking_style", "pairwise")
    review_count = report["pair_count"] if ranking_style == "pairwise" else report["ranking_count"]
    review_label = "Blind pairs" if ranking_style == "pairwise" else "Blind rankings"
    review_unit = "pairs" if ranking_style == "pairwise" else "rankings"
    style_note = (
        f"operator override of planned {_esc(report['planned_ranking_style'])} style"
        if report.get("ranking_style_source") == "operator-override"
        else "declared in the experiment plan"
    )
    execution_passed = sum(run["execution_status"] == "succeeded" for run in report["runs"])
    validation_passed = sum(run["validation_verdict"] == "passed" for run in report["runs"])
    prompt_cards = _prompts_html(report["episodes"])
    data = {
        "format": report["format"],
        "label_format": report["label_format"],
        "plan_id": report["plan_id"],
        "experiment": report["experiment"],
        "ranking_style": ranking_style,
        "pairs": [
            {
                "pair_id": pair["pair_id"],
                "block": pair["block"],
                "episode": pair["episode"],
                "repeat": pair["repeat"],
                "a_run_key": pair["a_run_key"],
                "b_run_key": pair["b_run_key"],
                "preferred_run_by_choice": pair["preferred_run_by_choice"],
            }
            for pair in report["pairs"]
        ],
        "rankings": [
            {
                "ranking_id": ranking["ranking_id"],
                "block": ranking["block"],
                "episode": ranking["episode"],
                "repeat": ranking["repeat"],
                "outcomes": [
                    {"label": outcome["label"], "run_key": outcome["run_key"]}
                    for outcome in ranking["outcomes"]
                ],
                "fields": list(ranking.get("fields", [])),
            }
            for ranking in report["rankings"]
        ],
    }
    reveal_button = (
        ""
        if external_blind
        else '<button id="reveal" aria-pressed="false">Reveal identities</button>'
    )
    blindness_note = (
        "Full condition identities are omitted from this deck and are revealed only in "
        "the decision report."
        if external_blind
        else "Identities are hidden by default."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(report["experiment"])} · evalmine episode evidence</title><style>{_CSS}</style></head>
<body data-reveal="0"><header class="hero"><div><div class="eyebrow">evalmine · episode evidence</div><h1>{_esc(report["question"])}</h1><p>{_esc(report["experiment"])} · plan <code>{_esc(report["plan_id"])}</code></p></div>
{reveal_button}</header>
<main><section class="summary"><div><small>Runs</small><b>{report["run_count"]}</b></div><div><small>{review_label}</small><b>{review_count}</b></div><div><small>Execution</small><b>{execution_passed}/{report["run_count"]} complete</b></div><div><small>Objective checks</small><b>{validation_passed}/{report["run_count"]} passed all</b></div></section>
<section><div class="section-head"><div><div class="eyebrow">Experiment</div><h2>What is being decided</h2></div><div id="progress">0 labeled of {review_count} {review_unit}</div></div><p class="question">{_esc(report["question"])}</p><ul>{"".join(f"<li>{_esc(item)}</li>" for item in report["objectives"])}</ul><div class="arms">{arms}</div>{_pricing_summary_html(report)}</section>
<section><div class="section-head"><div><div class="eyebrow">Shared task</div><h2>Prompts given identically to every model</h2></div></div>{prompt_cards}</section>
<section><div class="section-head"><div><div class="eyebrow">Blind review queue</div><h2>{'Rank all outcomes together' if ranking_style == 'n-way' else 'Compare trajectory evidence, then label'}</h2></div><div class="actions"><button id="export">Export labels JSON</button><label class="import">Import labels<input id="import" type="file" accept="application/json"></label></div></div>
<div class="notice">Ranking style: <b>{_esc(ranking_style)}</b> ({style_note}). {blindness_note} Mechanical objective checks are evidence annotations, not automatic exclusions. Cost evidence remains hidden while labeling.</div>
<div class="notice"><b>Judge workflow:</b> judge verdicts are intentionally absent from this blind labeling report. Export human labels before viewing judge results; the final decision report compares the independent human and judge rankings and surfaces disagreements. With at most {_esc(report['pair_count'])} induced pair labels here versus a calibration minimum of {_esc(report['judge'].get('min_labels', 'unspecified'))}, judge evidence may remain diagnostic only.</div>{rankings if ranking_style == "n-way" else (pairs or "<p>No comparable run pairs.</p>")}</section>
</main><footer>Generated {_esc(report["generated_at"])} · self-contained · no external assets</footer>
<script type="application/json" id="evalmine-episode-data">{_json_blob(data)}</script><script>{_JS}</script></body></html>"""


def _blind_external_report_artifact(report: dict[str, Any]) -> dict[str, Any]:
    """Remove the condition-to-output map from persisted blind-review evidence."""

    def blind_run(run: dict[str, Any]) -> dict[str, Any]:
        hidden = {
            key: value
            for key, value in run.items()
            if key
            not in {
                "arm",
                "requested_model",
                "observed_model",
                "observed_model_source",
                "model_identity",
                "treatment",
                "condition",
            }
        }
        external = run.get("external", {})
        hidden["external"] = {
            key: external[key]
            for key in ("lane", "item_id", "account_id", "provenance")
            if key in external
        }
        billing = run.get("billing", {})
        hidden["billing"] = {
            "basis": billing.get("basis"),
            "dollar_cost_status": "hidden-until-decision",
        }
        return hidden

    blind = dict(report)
    blind["condition_mapping"] = "decision-only"
    blind["runs"] = [blind_run(run) for run in report["runs"]]
    blind["arms"] = [
        {
            key: value
            for key, value in arm.items()
            if key
            in {
                "runs",
                "execution_succeeded",
                "validation_passed",
                "median_duration_ms",
            }
        }
        for arm in report["arms"]
    ]
    blind["pairs"] = [
        {
            **{
                key: value
                for key, value in pair.items()
                if key not in {"a", "b"}
            },
            "a": blind_run(pair["a"]),
            "b": blind_run(pair["b"]),
        }
        for pair in report["pairs"]
    ]
    blind["rankings"] = [
        {
            **{
                key: value
                for key, value in ranking.items()
                if key != "outcomes"
            },
            "outcomes": [
                {
                    **{
                        key: value
                        for key, value in outcome.items()
                        if key != "run"
                    },
                    "run": blind_run(outcome["run"]),
                }
                for outcome in ranking["outcomes"]
            ],
        }
        for ranking in report["rankings"]
    ]
    return blind


def generate_experiment_report(
    root: str | Path,
    *,
    generated_at: str | None = None,
    ranking_style: str | None = None,
    output: str | Path | None = None,
    prices_path: str | Path | None = None,
) -> ExperimentReportResult:
    """Create one report envelope without launching a provider or modifying workspaces."""
    report = build_experiment_report_data(
        root,
        generated_at=generated_at,
        ranking_style=ranking_style,
        prices_path=prices_path,
    )
    prepared_root = Path(root).resolve()
    report_root = Path(output).resolve() if output is not None else prepared_root / "report"
    if report_root.exists() or report_root.is_symlink():
        raise ExperimentReportError(
            f"report evidence already exists at {report_root}; it is never overwritten"
        )
    html_text = render_experiment_report_html(report)
    persisted_report = (
        _blind_external_report_artifact(report)
        if report.get("source_format") == "evalmine-external-artifacts-v1"
        else report
    )
    report_root.mkdir(parents=True)
    _write_once(report_root / "data.json", _json_bytes(persisted_report))
    _write_once(report_root / "index.html", html_text.encode("utf-8"))
    marker = {
        "format": REPORT_FORMAT,
        "prepared_root": str(prepared_root),
        "plan_id": report["plan_id"],
        "generated_at": report["generated_at"],
        "run_count": report["run_count"],
        "pair_count": report["pair_count"],
        "ranking_count": report["ranking_count"],
        "ranking_style": report["ranking_style"],
        "planned_ranking_style": report["planned_ranking_style"],
        "ranking_style_source": report["ranking_style_source"],
        "provider_runners_launched": False,
        "evidence_sha256": _report_hashes(report_root),
    }
    _write_once(report_root / "report-marker.json", _json_bytes(marker))
    return ExperimentReportResult(
        report_root,
        report_root / "index.html",
        report["pair_count"],
        report["run_count"],
        report["ranking_style"],
        report["ranking_count"],
    )


def verify_experiment_report(root: str | Path) -> dict[str, Any]:
    prepared_root = Path(root).resolve()
    report_root = prepared_root / "report"
    marker = _read_json(report_root / "report-marker.json")
    if marker.get("format") != REPORT_FORMAT:
        raise ExperimentReportError(f"{report_root} has an unknown report format")
    if marker.get("prepared_root") != str(prepared_root):
        raise ExperimentReportError("report marker points at a different prepared experiment")
    expected = marker.get("evidence_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ExperimentReportError("report marker has no evidence hashes")
    actual = _report_hashes(report_root)
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected))
        if not changed:
            changed = sorted(path for path in actual if actual[path] != expected.get(path))
        names = ", ".join(changed[:5]) or "report files"
        raise ExperimentReportError(f"report evidence changed after creation: {names}")
    return {
        "ok": True,
        "format": REPORT_FORMAT,
        "root": str(report_root),
        "html": str(report_root / "index.html"),
        "run_count": marker.get("run_count"),
        "pair_count": marker.get("pair_count"),
    }


_CSS = r"""
.status.warn{color:var(--orange);background:#fff0dc}.task{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:12px 0}.task h3{margin-top:0}.prompt{background:#25332e}.outcomes.n-way{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}.ranking-controls select,.field-review select{font:inherit;border:1px solid var(--line);border-radius:7px;padding:7px;background:var(--paper);color:var(--ink)}.ranking-controls>label:not(:last-child){margin-left:0}.field-review{padding:18px;overflow:auto;border-top:1px solid var(--line)}.field-review table{width:100%;border-collapse:collapse}.field-review th,.field-review td{padding:9px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}.field-review pre{margin:0;min-width:150px;max-height:160px}.wrong-controls label{display:block;white-space:nowrap}
:root{--bg:#f4f1e8;--paper:#fffdf7;--ink:#18211e;--muted:#66716d;--line:#d8d6cc;--green:#2e705b;--green-soft:#e4f0e9;--red:#a04635;--red-soft:#f7e8e3;--orange:#a96532;--shadow:0 14px 40px #24362e18;color-scheme:light dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}button,.import{font:inherit}code,pre{font-family:"SFMono-Regular",Consolas,monospace}.hero{background:#15211d;color:#f8faf8;padding:32px max(24px,calc((100vw - 1320px)/2));display:flex;justify-content:space-between;gap:24px;align-items:end}.hero h1{font-size:clamp(26px,4vw,48px);line-height:1.05;max-width:900px;margin:8px 0 10px}.hero p{color:#a9b9b3}.eyebrow{text-transform:uppercase;letter-spacing:.16em;font:700 11px monospace;color:#d59763}button,.import{border:1px solid #73817c;background:#fff;color:#18211e;border-radius:9px;padding:9px 12px;cursor:pointer}.hero button{background:#263832;color:white;border-color:#4c665d}main{max-width:1320px;margin:auto;padding:28px 24px 100px}section{margin:26px 0}.summary{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:-52px;position:relative}.summary>div{background:var(--paper);border:1px solid var(--line);border-radius:13px;padding:16px;box-shadow:var(--shadow)}.summary small{display:block;color:var(--muted)}.summary b{font-size:24px}.section-head{display:flex;justify-content:space-between;gap:16px;align-items:end}.section-head h2{margin:4px 0}.question{font-size:18px}.arms{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}.arm-card{background:var(--paper);border:1px solid var(--line);padding:14px;border-radius:12px}.identity{display:none;color:var(--orange);margin:4px 0 8px}body[data-reveal="1"] .identity{display:block}.actions{display:flex;gap:8px;align-items:center}.import input{display:none}.notice{padding:12px 14px;border-left:4px solid var(--orange);background:#fff4e8;margin:14px 0}.pair{background:var(--paper);border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:var(--shadow);margin:18px 0}.pair-head{padding:15px 18px;border-bottom:1px solid var(--line);display:flex;justify-content:space-between}.pair-head h3{margin:3px 0}.saved{color:var(--green)}.outcomes{display:grid;grid-template-columns:1fr 1fr}.outcome{padding:18px;min-width:0;border-right:1px solid var(--line)}.outcome:last-child{border-right:0}.outcome header{display:flex;justify-content:space-between;gap:8px}.outcome-letter{font-size:19px;font-weight:800}.status{display:inline-block;border-radius:99px;padding:3px 7px;font:700 10px monospace;text-transform:uppercase}.status.pass{color:var(--green);background:var(--green-soft)}.status.fail{color:var(--red);background:var(--red-soft)}.status.muted{color:var(--muted);background:#eee}.metrics{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0}.metrics span{background:#f0eee7;padding:4px 7px;border-radius:6px;font:11px monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#18211e;color:#dce7e2;border-radius:9px;padding:13px;max-height:420px;overflow:auto}.final{min-height:150px}.stderr{color:#ffb7a8}.error{background:var(--red-soft);color:var(--red);padding:9px;margin:9px 0;border-radius:7px}details{border-top:1px solid var(--line);padding:9px 0}summary{cursor:pointer;font-weight:700}details small{color:var(--muted)}.label-controls{border-top:1px solid var(--line);padding:14px 18px;background:#f1eee5;display:flex;gap:8px;align-items:center;flex-wrap:wrap}.label-controls button.selected{background:var(--green);color:white;border-color:var(--green)}.label-controls label{margin-left:auto;display:flex;align-items:center;gap:8px}.note{width:min(420px,40vw);border:1px solid var(--line);border-radius:8px;padding:8px;background:white;color:#18211e}footer{text-align:center;color:var(--muted);padding:30px}.muted{color:var(--muted)}
@media(prefers-color-scheme:dark){:root{--bg:#0f1513;--paper:#17201d;--ink:#e4ebe8;--muted:#9aaba5;--line:#34423d;--green:#7ac6a8;--green-soft:#19382e;--red:#ee9b85;--red-soft:#3a211d;--orange:#e3a36f;--shadow:none}.notice{background:#2b241b}.metrics span,.label-controls{background:#202b27}.note,button,.import{background:#202b27;color:#e4ebe8}.status.muted{background:#2b3531}}
@media(max-width:760px){.hero,.section-head{display:block}.hero button{margin-top:12px}.summary{grid-template-columns:1fr 1fr;margin-top:-30px}.outcomes{grid-template-columns:1fr}.outcome{border-right:0;border-bottom:1px solid var(--line)}.label-controls label{margin-left:0;width:100%;display:block}.note{width:100%;margin-top:5px}.actions{margin-top:10px;flex-wrap:wrap}}
"""


_JS = r"""
(()=>{'use strict';
const data=JSON.parse(document.getElementById('evalmine-episode-data').textContent);
const style=data.ranking_style||'pairwise',key='evalmine:episode-labels:'+style+':'+data.plan_id;let labels={};
try{labels=JSON.parse(localStorage.getItem(key)||'{}')||{}}catch(e){labels={}}
const pairCards=[...document.querySelectorAll('[data-pair]')],rankingCards=[...document.querySelectorAll('[data-ranking]')];
function fieldComplete(row,fields){return (fields||[]).every(field=>Boolean(((row.field_labels||{})[field]||{}).best))}
function rankingComplete(row,outcomes,fields){if(row.unclear)return true;const values=Object.values(row.ranks||{}).map(Number),rankingDone=values.length===outcomes.length&&new Set(values).size===outcomes.length&&values.every(value=>value>=1&&value<=outcomes.length);return rankingDone&&fieldComplete(row,fields)}
function save(){try{localStorage.setItem(key,JSON.stringify(labels))}catch(e){}render()}
function render(){let done=0;for(const card of pairCards){const id=card.dataset.pair,row=labels[id]||{};if(row.choice)done++;for(const button of card.querySelectorAll('[data-choice]'))button.classList.toggle('selected',button.dataset.choice===row.choice);const note=card.querySelector('.note');if(document.activeElement!==note)note.value=row.note||'';card.querySelector('.saved').textContent=row.choice?'saved locally':''}for(const card of rankingCards){const id=card.dataset.ranking,row=labels[id]||{},ranking=data.rankings.find(item=>item.ranking_id===id);if(rankingComplete(row,ranking.outcomes,ranking.fields))done++;for(const select of card.querySelectorAll('[data-rank-label]'))if(document.activeElement!==select)select.value=String((row.ranks||{})[select.dataset.rankLabel]||'');for(const select of card.querySelectorAll('[data-field-best]'))if(document.activeElement!==select)select.value=String((((row.field_labels||{})[select.dataset.fieldBest]||{}).best)||'');for(const input of card.querySelectorAll('[data-field-wrong]')){const field=((row.field_labels||{})[input.dataset.fieldWrong]||{});input.checked=(field.wrong||[]).includes(input.dataset.fieldWrongLabel)}const unclear=card.querySelector('[data-unclear]');unclear.classList.toggle('selected',Boolean(row.unclear));unclear.textContent=row.unclear?'Unclear selected':'Mark unclear';const note=card.querySelector('.note');if(document.activeElement!==note)note.value=row.note||'';card.querySelector('.saved').textContent=rankingComplete(row,ranking.outcomes,ranking.fields)?'saved locally':''}const total=style==='n-way'?rankingCards.length:pairCards.length;document.getElementById('progress').textContent=done+' labeled of '+total+' '+(style==='n-way'?'rankings':'pairs')}
for(const card of pairCards){const id=card.dataset.pair;for(const button of card.querySelectorAll('[data-choice]'))button.addEventListener('click',()=>{labels[id]={...(labels[id]||{}),choice:button.dataset.choice,labelled_at:new Date().toISOString()};save()});card.querySelector('.note').addEventListener('input',e=>{labels[id]={...(labels[id]||{}),note:e.target.value};save()})}
for(const card of rankingCards){const id=card.dataset.ranking;for(const select of card.querySelectorAll('[data-rank-label]'))select.addEventListener('change',()=>{const row=labels[id]||{},ranks={...(row.ranks||{})},value=Number(select.value),label=select.dataset.rankLabel;if(value){for(const [other,rank] of Object.entries(ranks))if(other!==label&&Number(rank)===value)delete ranks[other];ranks[label]=value}else delete ranks[label];labels[id]={...row,ranks,unclear:false,labelled_at:new Date().toISOString()};save()});for(const select of card.querySelectorAll('[data-field-best]'))select.addEventListener('change',()=>{const row=labels[id]||{},fieldLabels={...(row.field_labels||{})},field=select.dataset.fieldBest,current={...(fieldLabels[field]||{})};current.best=select.value||null;fieldLabels[field]=current;labels[id]={...row,field_labels:fieldLabels,unclear:false,labelled_at:new Date().toISOString()};save()});for(const input of card.querySelectorAll('[data-field-wrong]'))input.addEventListener('change',()=>{const row=labels[id]||{},fieldLabels={...(row.field_labels||{})},field=input.dataset.fieldWrong,label=input.dataset.fieldWrongLabel,current={...(fieldLabels[field]||{})},wrong=new Set(current.wrong||[]);if(input.checked)wrong.add(label);else wrong.delete(label);current.wrong=[...wrong].sort();fieldLabels[field]=current;labels[id]={...row,field_labels:fieldLabels,unclear:false,labelled_at:new Date().toISOString()};save()});card.querySelector('[data-unclear]').addEventListener('click',()=>{const row=labels[id]||{};labels[id]={...row,ranks:{},unclear:!row.unclear,labelled_at:new Date().toISOString()};save()});card.querySelector('.note').addEventListener('input',e=>{labels[id]={...(labels[id]||{}),note:e.target.value};save()})}
const reveal=document.getElementById('reveal');if(reveal)reveal.addEventListener('click',()=>{const shown=document.body.dataset.reveal==='1';document.body.dataset.reveal=shown?'0':'1';reveal.setAttribute('aria-pressed',shown?'false':'true');reveal.textContent=shown?'Reveal identities':'Hide identities'});
document.getElementById('export').addEventListener('click',()=>{let payload;if(style==='n-way'){const rows=[];for(const ranking of data.rankings){const row=labels[ranking.ranking_id]||{};if(!rankingComplete(row,ranking.outcomes,ranking.fields))continue;const order=row.unclear?[]:[...ranking.outcomes].sort((a,b)=>Number(row.ranks[a.label])-Number(row.ranks[b.label])).map(item=>item.label);const byLabel=Object.fromEntries(ranking.outcomes.map(item=>[item.label,item.run_key])),fieldLabels=row.unclear?[]:(ranking.fields||[]).map(field=>{const value=(row.field_labels||{})[field]||{},best=value.best||null,wrong=[...(value.wrong||[])];return {field,best_label:best,best_run_key:byLabel[best]||null,wrong_labels:wrong,wrong_run_keys:wrong.map(label=>byLabel[label])}});rows.push({ranking_id:ranking.ranking_id,block:ranking.block,episode:ranking.episode,repeat:ranking.repeat,order,ranked_run_keys:order.map(label=>byLabel[label]),field_labels:fieldLabels,unclear:Boolean(row.unclear),note:row.note||null,labelled_at:row.labelled_at||null,identities_revealed:document.body.dataset.reveal==='1'})}payload={format:data.label_format,ranking_style:style,plan_id:data.plan_id,experiment:data.experiment,exported_at:new Date().toISOString(),rankings:rows}}else{const rows=[];for(const pair of data.pairs){const row=labels[pair.pair_id];if(!row||!row.choice)continue;rows.push({pair_id:pair.pair_id,block:pair.block,episode:pair.episode,repeat:pair.repeat,a_run_key:pair.a_run_key,b_run_key:pair.b_run_key,choice:row.choice,preferred_run_key:pair.preferred_run_by_choice[row.choice],note:row.note||null,labelled_at:row.labelled_at||null,identities_revealed:document.body.dataset.reveal==='1'})}payload={format:data.label_format,ranking_style:style,plan_id:data.plan_id,experiment:data.experiment,exported_at:new Date().toISOString(),labels:rows}}const blob=new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=data.experiment+'-'+data.plan_id+'-'+(style==='n-way'?'rankings':'labels')+'.json';a.click();URL.revokeObjectURL(url)});
document.getElementById('import').addEventListener('change',event=>{const file=event.target.files[0];if(!file)return;const reader=new FileReader();reader.onload=()=>{try{const payload=JSON.parse(reader.result);if(payload.format!==data.label_format||payload.plan_id!==data.plan_id)throw new Error('Labels belong to a different plan or ranking style.');if(style==='n-way'){const known=new Map(data.rankings.map(item=>[item.ranking_id,item]));for(const row of payload.rankings||[]){const ranking=known.get(row.ranking_id);if(!ranking)continue;const expected=ranking.outcomes.map(item=>item.label).sort(),order=(row.order||[]).map(String);if(!row.unclear&&(order.length!==expected.length||order.slice().sort().join('|')!==expected.join('|')))throw new Error('A ranking must contain every outcome exactly once.');const fieldLabels={};for(const item of row.field_labels||[]){if(!(ranking.fields||[]).includes(item.field))continue;fieldLabels[item.field]={best:item.best_label||null,wrong:[...(item.wrong_labels||[])]}}labels[row.ranking_id]={unclear:Boolean(row.unclear),ranks:Object.fromEntries(order.map((label,index)=>[label,index+1])),field_labels:fieldLabels,note:row.note||'',labelled_at:row.labelled_at||null}}}else{const known=new Set(data.pairs.map(pair=>pair.pair_id));for(const row of payload.labels||[]){if(known.has(row.pair_id)&&['A','B','tie','unclear'].includes(row.choice))labels[row.pair_id]={choice:row.choice,note:row.note||'',labelled_at:row.labelled_at||null}}}save()}catch(error){alert('Could not import labels: '+error.message)}};reader.readAsText(file);event.target.value=''});render();
})();
"""
