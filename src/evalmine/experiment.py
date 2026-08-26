"""Version-2 agent experiment manifests and zero-side-effect run planning.

This module deliberately stops before execution. Loading validates the experiment
contract and local file references; planning deterministically expands all work that a
future executor would perform. Neither operation creates a workspace or calls a model.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from .suite import SECRET_PATTERNS

EXPERIMENT_FORMAT_VERSION = 2
DEFAULT_ORDER = "rotate"
DEFAULT_MAX_PARALLEL = 1
DEFAULT_REPEATS = 1

_SCHEMA_PATH = Path(__file__).with_name("experiment.schema.json")


class ExperimentError(Exception):
    """An experiment manifest that cannot be planned safely."""


@dataclass(frozen=True)
class Seed:
    repo: Path
    repo_declared: str
    ref: str
    commit: str
    dirty: str
    untracked: str
    untracked_allowlist: tuple[str, ...] = ()


@dataclass(frozen=True)
class Isolation:
    workspace: str
    session: str
    external_writes: str
    external_write_allowlist: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputFile:
    declared: str
    path: Path
    content: str
    hash: str


@dataclass(frozen=True)
class ArmConfiguration:
    instructions: str
    plugins: str
    instruction_files: tuple[InputFile, ...] = ()
    plugin_allowlist: tuple[str, ...] = ()
    plugin_directories: tuple[str, ...] = ()
    settings: dict[str, Any] = field(default_factory=dict)
    arguments: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "instructions": self.instructions,
            "instruction_files": [
                {
                    "declared": item.declared,
                    "path": str(item.path),
                    "sha256": item.hash,
                }
                for item in self.instruction_files
            ],
            "plugins": self.plugins,
            "plugin_allowlist": list(self.plugin_allowlist),
            "plugin_directories": list(self.plugin_directories),
            "settings": dict(self.settings),
            "arguments": list(self.arguments),
        }


@dataclass(frozen=True)
class Arm:
    id: str
    runner: str
    model: str
    auth: str
    max_cost_usd: float | None
    configuration: ArmConfiguration


@dataclass(frozen=True)
class Turn:
    prompt: str
    prompt_file: Path | None = None


@dataclass(frozen=True)
class Episode:
    id: str
    title: str | None
    turns: tuple[Turn, ...]
    validators: tuple[str, ...]
    repeats: int


@dataclass(frozen=True)
class ValidatorSpec:
    id: str
    type: str
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, **self.config}


@dataclass(frozen=True)
class Evaluation:
    objectives: tuple[str, ...]
    blind: str
    human_required: bool
    labels_per_pair: int
    judge: dict[str, Any]


@dataclass(frozen=True)
class Experiment:
    name: str
    version: int
    question: str
    seed: Seed
    isolation: Isolation
    order: str
    max_parallel: int
    arms: tuple[Arm, ...]
    validators: tuple[ValidatorSpec, ...]
    episodes: tuple[Episode, ...]
    evaluation: Evaluation
    path: Path
    hash: str
    input_hash: str
    manifest_bytes: bytes = field(repr=False)


@dataclass(frozen=True)
class PlannedRun:
    sequence: int
    block: str
    run_key: str
    arm_id: str
    episode_id: str
    repeat: int
    runner: str
    model: str
    auth: str
    max_cost_usd: float | None
    workspace: str
    session_key: str
    turn_count: int
    validators: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "block": self.block,
            "run_key": self.run_key,
            "arm": self.arm_id,
            "episode": self.episode_id,
            "repeat": self.repeat,
            "runner": self.runner,
            "model": self.model,
            "auth": self.auth,
            "max_cost_usd": self.max_cost_usd,
            "workspace": self.workspace,
            "session_key": self.session_key,
            "turns": self.turn_count,
            "validators": list(self.validators),
        }


@dataclass(frozen=True)
class ExperimentPlan:
    experiment: Experiment
    plan_id: str
    runs: tuple[PlannedRun, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        experiment = self.experiment
        return {
            "plan_version": 1,
            "plan_id": self.plan_id,
            "experiment": experiment.name,
            "question": experiment.question,
            "manifest": str(experiment.path),
            "manifest_hash": experiment.hash,
            "input_hash": experiment.input_hash,
            "seed": {
                "repo": str(experiment.seed.repo),
                "ref": experiment.seed.ref,
                "commit": experiment.seed.commit,
                "dirty": experiment.seed.dirty,
                "untracked": experiment.seed.untracked,
                "untracked_allowlist": list(experiment.seed.untracked_allowlist),
            },
            "isolation": {
                "workspace": experiment.isolation.workspace,
                "session": experiment.isolation.session,
                "external_writes": experiment.isolation.external_writes,
                "external_write_allowlist": list(experiment.isolation.external_write_allowlist),
            },
            "arms": [
                {
                    "id": arm.id,
                    "runner": arm.runner,
                    "model": arm.model,
                    "auth": arm.auth,
                    "max_cost_usd": arm.max_cost_usd,
                    "configuration": arm.configuration.as_dict(),
                }
                for arm in experiment.arms
            ],
            "validators": {
                validator.id: dict(validator.config) for validator in experiment.validators
            },
            "episodes": [
                {
                    "id": episode.id,
                    "title": episode.title,
                    "repeats": episode.repeats,
                    "validators": list(episode.validators),
                    "turns": [
                        {
                            "prompt_file": (
                                str(turn.prompt_file) if turn.prompt_file is not None else None
                            ),
                            "prompt_sha256": hashlib.sha256(
                                turn.prompt.encode("utf-8")
                            ).hexdigest(),
                        }
                        for turn in episode.turns
                    ],
                }
                for episode in experiment.episodes
            ],
            "evaluation": {
                "objectives": list(experiment.evaluation.objectives),
                "blind": experiment.evaluation.blind,
                "human": {
                    "required": experiment.evaluation.human_required,
                    "labels_per_pair": experiment.evaluation.labels_per_pair,
                },
                "judge": dict(experiment.evaluation.judge),
            },
            "schedule": {
                "order": experiment.order,
                "max_parallel": experiment.max_parallel,
                "run_count": len(self.runs),
            },
            "warnings": list(self.warnings),
            "runs": [run.as_dict() for run in self.runs],
        }


def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _walk_strings(node: Any, path: str = "$"):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_strings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk_strings(value, f"{path}[{index}]")
    elif isinstance(node, str):
        yield path, node


def _scan_for_secrets(doc: Any, path: Path) -> None:
    for where, value in _walk_strings(doc):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(value):
                raise ExperimentError(
                    f"{path}: what looks like an {label} appears at {where}. "
                    "Experiment manifests declare an auth mode, never credentials; "
                    "remove the key and rotate it."
                )


def _scan_text_for_secrets(text: str, where: str, manifest: Path) -> None:
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            raise ExperimentError(
                f"{manifest}: what looks like an {label} appears in {where}. "
                "Referenced experiment inputs are copied into evidence; remove the key "
                "and rotate it."
            )


def _load_yaml(path: Path) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ExperimentError(f"{path}: cannot read experiment manifest ({exc})") from exc
    try:
        return yaml.safe_load(raw.decode("utf-8")), raw
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ExperimentError(f"{path}: not valid YAML ({exc})") from exc


def _validate(doc: Any, path: Path) -> None:
    if not isinstance(doc, dict):
        raise ExperimentError(f"{path}: the top level of an experiment must be a mapping")
    version = doc.get("version")
    if version is None:
        raise ExperimentError(f"{path}: 'version' is required (the current experiment format is 2)")
    if version != EXPERIMENT_FORMAT_VERSION:
        raise ExperimentError(
            f"{path}: unknown experiment version {version!r}; this evalmine understands "
            f"version {EXPERIMENT_FORMAT_VERSION} only"
        )

    validator = jsonschema.Draft202012Validator(_load_schema())
    errors = sorted(validator.iter_errors(doc), key=lambda error: list(error.absolute_path))
    if errors:
        error = errors[0]
        where = "/".join(str(part) for part in error.absolute_path) or "(top level)"
        raise ExperimentError(f"{path}: at {where}: {error.message}")


def _unique_ids(items: list[dict[str, Any]], kind: str, path: Path) -> None:
    seen: set[str] = set()
    for item in items:
        item_id = item["id"]
        if item_id in seen:
            raise ExperimentError(f"{path}: duplicate {kind} id {item_id!r}")
        seen.add(item_id)


def _validate_validator_references(doc: dict[str, Any], path: Path) -> None:
    declared = set(doc.get("validators", {}))
    for episode in doc["episodes"]:
        for validator_id in episode.get("validators", []):
            if validator_id not in declared:
                raise ExperimentError(
                    f"{path}: episode {episode['id']!r} references undeclared validator "
                    f"{validator_id!r}"
                )


def _existing_file(manifest: Path, declared: str, where: str) -> Path:
    resolved = (manifest.parent / declared).resolve()
    if not resolved.is_file():
        raise ExperimentError(f"{manifest}: {where} {declared!r} is not a readable file")
    return resolved


def _read_input_file(manifest: Path, declared: str, where: str) -> InputFile:
    path = _existing_file(manifest, declared, where)
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExperimentError(f"{manifest}: cannot read {where} {path} ({exc})") from exc
    if not content.strip():
        raise ExperimentError(f"{manifest}: {where} {path} is empty")
    _scan_text_for_secrets(content, where, manifest)
    return InputFile(
        declared=declared,
        path=path,
        content=content,
        hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _resolve_git_ref(manifest: Path, repo: Path, ref: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ExperimentError(
            f"{manifest}: cannot resolve seed ref {ref!r} in {repo} ({exc})"
        ) from exc
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        detail = result.stderr.strip() or "not a commit"
        raise ExperimentError(
            f"{manifest}: seed ref {ref!r} does not resolve to a commit in {repo} ({detail})"
        )
    return commit


def _load_configuration(raw: dict[str, Any], manifest: Path, arm_id: str) -> ArmConfiguration:
    instruction_files = tuple(
        _read_input_file(manifest, item, f"arm {arm_id!r} instruction file")
        for item in raw.get("instruction_files", [])
    )
    return ArmConfiguration(
        instructions=raw["instructions"],
        plugins=raw["plugins"],
        instruction_files=instruction_files,
        plugin_allowlist=tuple(raw.get("plugin_allowlist", [])),
        plugin_directories=tuple(raw.get("plugin_directories", [])),
        settings=dict(raw.get("settings", {})),
        arguments=tuple(raw.get("arguments", [])),
    )


def load_experiment(path: str | Path) -> Experiment:
    """Load and fully validate a v2 experiment manifest without executing it."""
    path = Path(path)
    doc, raw = _load_yaml(path)
    _validate(doc, path)
    _scan_for_secrets(doc, path)
    _unique_ids(doc["arms"], "arm", path)
    _unique_ids(doc["episodes"], "episode", path)
    _validate_validator_references(doc, path)

    repo_declared = doc["seed"]["repo"]
    repo = (path.parent / repo_declared).resolve()
    if not repo.is_dir():
        raise ExperimentError(f"{path}: seed repo {repo_declared!r} is not a directory")
    commit = _resolve_git_ref(path, repo, doc["seed"]["ref"])

    arms = tuple(
        Arm(
            id=raw_arm["id"],
            runner=raw_arm["runner"],
            model=raw_arm["model"],
            auth=raw_arm["auth"],
            max_cost_usd=(
                float(raw_arm["max_cost_usd"])
                if raw_arm.get("max_cost_usd") is not None
                else None
            ),
            configuration=_load_configuration(raw_arm["configuration"], path, raw_arm["id"]),
        )
        for raw_arm in doc["arms"]
    )
    validators = tuple(
        ValidatorSpec(id=validator_id, type=spec["type"], config=dict(spec))
        for validator_id, spec in doc.get("validators", {}).items()
    )

    episodes: list[Episode] = []
    for raw_episode in doc["episodes"]:
        turns: list[Turn] = []
        for index, raw_turn in enumerate(raw_episode["turns"], start=1):
            if "prompt_file" in raw_turn:
                prompt_input = _read_input_file(
                    path,
                    raw_turn["prompt_file"],
                    f"episode {raw_episode['id']!r} turn {index} prompt_file",
                )
                turns.append(Turn(prompt=prompt_input.content, prompt_file=prompt_input.path))
            else:
                turns.append(Turn(prompt=raw_turn["prompt"]))
        episodes.append(
            Episode(
                id=raw_episode["id"],
                title=raw_episode.get("title"),
                turns=tuple(turns),
                validators=tuple(raw_episode.get("validators", [])),
                repeats=int(raw_episode.get("repeats", DEFAULT_REPEATS)),
            )
        )

    raw_seed = doc["seed"]
    raw_isolation = doc["isolation"]
    raw_schedule = doc.get("schedule", {})
    raw_evaluation = doc["evaluation"]
    raw_human = raw_evaluation["human"]
    manifest_hash = hashlib.sha256(raw).hexdigest()
    input_descriptor = {
        "manifest": manifest_hash,
        "commit": commit,
        "instructions": [
            [item.hash for item in arm.configuration.instruction_files] for arm in arms
        ],
        "turns": [
            [hashlib.sha256(turn.prompt.encode("utf-8")).hexdigest() for turn in episode.turns]
            for episode in episodes
        ],
    }
    input_hash = hashlib.sha256(
        json.dumps(
            input_descriptor,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    return Experiment(
        name=doc["experiment"],
        version=int(doc["version"]),
        question=doc["question"],
        seed=Seed(
            repo=repo,
            repo_declared=repo_declared,
            ref=raw_seed["ref"],
            commit=commit,
            dirty=raw_seed["dirty"],
            untracked=raw_seed["untracked"],
            untracked_allowlist=tuple(raw_seed.get("untracked_allowlist", [])),
        ),
        isolation=Isolation(
            workspace=raw_isolation["workspace"],
            session=raw_isolation["session"],
            external_writes=raw_isolation["external_writes"],
            external_write_allowlist=tuple(raw_isolation.get("external_write_allowlist", [])),
        ),
        order=raw_schedule.get("order", DEFAULT_ORDER),
        max_parallel=int(raw_schedule.get("max_parallel", DEFAULT_MAX_PARALLEL)),
        arms=arms,
        validators=validators,
        episodes=tuple(episodes),
        evaluation=Evaluation(
            objectives=tuple(raw_evaluation["objectives"]),
            blind=raw_evaluation["blind"],
            human_required=bool(raw_human["required"]),
            labels_per_pair=int(raw_human.get("labels_per_pair", 1)),
            judge=dict(raw_evaluation["judge"]),
        ),
        path=path.resolve(),
        hash=manifest_hash,
        input_hash=input_hash,
        manifest_bytes=raw,
    )


def build_plan(experiment: Experiment) -> ExperimentPlan:
    """Deterministically expand an experiment into an inspectable run schedule."""
    warnings: list[str] = []
    if experiment.isolation.session == "reuse-per-arm":
        warnings.append(
            "session reuse can leak history between episodes; fresh-per-run is recommended "
            "for comparative claims"
        )
    if experiment.order == "fixed":
        warnings.append(
            "fixed arm order can confound results with warm-up or time effects; rotate is "
            "recommended"
        )

    runs: list[PlannedRun] = []
    sequence = 0
    block_index = 0
    for episode in experiment.episodes:
        for repeat in range(1, episode.repeats + 1):
            ordered_arms = list(experiment.arms)
            if experiment.order == "rotate":
                offset = block_index % len(ordered_arms)
                ordered_arms = ordered_arms[offset:] + ordered_arms[:offset]
            block = f"{episode.id}-r{repeat:02d}"
            for arm in ordered_arms:
                sequence += 1
                identity = f"{experiment.input_hash}:{episode.id}:{repeat}:{arm.id}"
                suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
                run_key = f"{block}-{arm.id}-{suffix}"
                session_key = (
                    run_key
                    if experiment.isolation.session == "fresh-per-run"
                    else f"{experiment.name}-{arm.id}"
                )
                runs.append(
                    PlannedRun(
                        sequence=sequence,
                        block=block,
                        run_key=run_key,
                        arm_id=arm.id,
                        episode_id=episode.id,
                        repeat=repeat,
                        runner=arm.runner,
                        model=arm.model,
                        auth=arm.auth,
                        max_cost_usd=arm.max_cost_usd,
                        workspace=experiment.isolation.workspace,
                        session_key=session_key,
                        turn_count=len(episode.turns),
                        validators=episode.validators,
                    )
                )
            block_index += 1

    return ExperimentPlan(
        experiment=experiment,
        plan_id=f"{experiment.name}-{experiment.input_hash[:12]}",
        runs=tuple(runs),
        warnings=tuple(warnings),
    )
