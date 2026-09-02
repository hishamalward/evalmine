"""Load and validate a suite file, and render every prompt in it.

Spec: docs/spec.md S5. Everything in here is a hard error or a value; there are
no warnings, because a suite that is half understood produces a report that
looks fine and means nothing.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from .check import DEFAULT_CHECK_TIMEOUT_S, CheckSpec

SUITE_FORMAT_VERSION = 1

_SCHEMA_PATH = Path(__file__).with_name("suite.schema.json")

#: ``{{name}}``, whitespace inside the braces allowed. Nothing else: no
#: expressions, no filters, no loops (spec S5.4).
PLACEHOLDER_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")

#: A suite file must never carry a key. These are matched against every string
#: in the parsed document (spec S1.4).
SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Anthropic API key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")),
    ("OpenAI API key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{16,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}")),
)

DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 700
DEFAULT_TIMEOUT_S = 60
DEFAULT_JUDGE_MAX_TOKENS = 400
DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_MIN_KAPPA = 0.40
DEFAULT_MIN_LABELS = 10
DEFAULT_MOVER_THRESHOLD = 0.15

_PARAM_KEYS = ("temperature", "max_tokens", "timeout_s", "top_p", "stop")


class SuiteError(Exception):
    """A suite file that cannot be trusted. Always exit 1."""


@dataclass(frozen=True)
class CallParams:
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS
    timeout_s: int = DEFAULT_TIMEOUT_S
    top_p: float | None = None
    stop: tuple[str, ...] | None = None

    def as_cache_params(self) -> dict[str, Any]:
        """The ``params`` block of the cache key (spec S6.5)."""
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": list(self.stop) if self.stop else None,
            "seed": None,
        }


@dataclass(frozen=True)
class Case:
    id: str
    vars: dict[str, Any]
    prompt: str
    system: str | None
    #: The execution check for this case (spec S6.6), task defaults already merged.
    check: CheckSpec | None = None


@dataclass(frozen=True)
class Task:
    id: str
    prompt_template: str
    kind: str | None
    schema: dict[str, Any] | None
    rubric: str | None
    judge: bool
    params: CallParams
    cases: tuple[Case, ...]
    hash: str


@dataclass(frozen=True)
class Label:
    task: str
    case: str
    baseline: str
    candidate: str
    prefer: str
    note: str | None = None


@dataclass(frozen=True)
class Calibration:
    min_kappa: float = DEFAULT_MIN_KAPPA
    min_labels: int = DEFAULT_MIN_LABELS
    on_below_floor: str = "flag"


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    rubric: str
    max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS
    temperature: float = DEFAULT_JUDGE_TEMPERATURE
    calibration: Calibration = field(default_factory=Calibration)


@dataclass(frozen=True)
class Suite:
    name: str
    version: int
    description: str | None
    judge: JudgeConfig
    tasks: tuple[Task, ...]
    labels: tuple[Label, ...]
    max_cost_usd: float | None
    mover_threshold: float
    openrouter_provider_pins: dict[str, str]
    path: Path
    hash: str

    def task(self, task_id: str) -> Task | None:
        for t in self.tasks:
            if t.id == task_id:
                return t
        return None

    @property
    def task_hashes(self) -> dict[str, str]:
        return {t.id: t.hash for t in self.tasks}

    def pair_count(self, n_candidates: int, repeats: int = 1) -> int:
        judged_cases = sum(len(t.cases) for t in self.tasks if t.judge)
        return judged_cases * n_candidates * repeats


def canonical_bytes(obj: Any) -> bytes:
    """The one canonical JSON encoding used for every hash in evalmine."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _load_schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _walk_strings(node: Any, path: str = "$"):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_strings(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_strings(v, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def _scan_for_secrets(doc: Any, path: Path) -> None:
    for where, text in _walk_strings(doc):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise SuiteError(
                    f"{path}: what looks like an {label} appears at {where}. "
                    "evalmine reads keys from the environment only; remove it from the "
                    "suite file and rotate it."
                )


def render(template: str, variables: dict[str, Any], where: str) -> str:
    """Substitute ``{{name}}`` once. An unmatched placeholder is a hard error."""
    missing: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            missing.append(name)
            return match.group(0)
        value = variables[name]
        return "" if value is None else str(value)

    rendered = PLACEHOLDER_RE.sub(_sub, template)
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise SuiteError(
            f"{where}: prompt uses {{{{{names}}}}} but the case declares no such var. "
            "A silently empty variable is the easiest way to make an eval meaningless, "
            "so this is an error."
        )
    return rendered


def _params_from(mapping: dict[str, Any], base: CallParams) -> CallParams:
    stop = mapping.get("stop")
    return CallParams(
        temperature=float(mapping.get("temperature", base.temperature)),
        max_tokens=int(mapping.get("max_tokens", base.max_tokens)),
        timeout_s=int(mapping.get("timeout_s", base.timeout_s)),
        top_p=mapping.get("top_p", base.top_p),
        stop=tuple(stop) if stop is not None else base.stop,
    )


def _task_hash(raw_task: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(raw_task)).hexdigest()


def _load_yaml(path: Path) -> tuple[Any, bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SuiteError(f"{path}: cannot read suite file ({exc})") from exc
    try:
        return yaml.safe_load(raw.decode("utf-8")), raw
    except (yaml.YAMLError, UnicodeDecodeError) as exc:
        raise SuiteError(f"{path}: not valid YAML ({exc})") from exc


def _validate(doc: Any, path: Path) -> None:
    if not isinstance(doc, dict):
        raise SuiteError(f"{path}: the top level of a suite must be a mapping")

    version = doc.get("version")
    if version is None:
        raise SuiteError(f"{path}: 'version' is required (the current suite format is 1)")
    if version != SUITE_FORMAT_VERSION:
        raise SuiteError(
            f"{path}: unknown suite version {version!r}; this evalmine understands "
            f"version {SUITE_FORMAT_VERSION} only"
        )

    validator = jsonschema.Draft202012Validator(_load_schema())
    errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.absolute_path))
    if errors:
        err = errors[0]
        where = "/".join(str(p) for p in err.absolute_path) or "(top level)"
        raise SuiteError(f"{path}: at {where}: {err.message}")


def _load_labels(doc: dict[str, Any], path: Path) -> list[dict[str, Any]]:
    if "labels_path" in doc:
        labels_file = (path.parent / doc["labels_path"]).resolve()
        try:
            text = labels_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise SuiteError(f"{path}: labels_path {labels_file} cannot be read ({exc})") from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SuiteError(f"{labels_file}: not valid YAML/JSON ({exc})") from exc
        if isinstance(loaded, dict):
            loaded = loaded.get("labels")
        if not isinstance(loaded, list):
            raise SuiteError(
                f"{labels_file}: expected a list of labels, or a mapping with a 'labels' key"
            )
        full = _load_schema()
        validator = jsonschema.Draft202012Validator(
            {"$defs": full["$defs"], "$ref": "#/$defs/label"}
        )
        for i, item in enumerate(loaded):
            errs = sorted(validator.iter_errors(item), key=lambda e: list(e.absolute_path))
            if errs:
                raise SuiteError(f"{labels_file}: label {i}: {errs[0].message}")
        return loaded
    return list(doc.get("labels") or [])


def _check_from(
    task_level: dict[str, Any] | None, case_level: dict[str, Any] | None
) -> CheckSpec | None:
    """Merge a task's ``check`` defaults under a case's ``check``.

    A task-level block usually carries ``setup`` and ``timeout_s`` shared by every
    case; ``run`` is normally per case. No ``run`` after the merge means no check.
    """
    merged: dict[str, Any] = {**(task_level or {}), **(case_level or {})}
    if not merged.get("run"):
        return None
    return CheckSpec(
        run=merged["run"],
        setup=merged.get("setup"),
        timeout_s=int(merged.get("timeout_s", DEFAULT_CHECK_TIMEOUT_S)),
    )


def _load_task_schema(raw_task: dict[str, Any], path: Path) -> dict[str, Any] | None:
    if "schema" in raw_task:
        return raw_task["schema"]
    if "schema_path" in raw_task:
        schema_file = (path.parent / raw_task["schema_path"]).resolve()
        try:
            text = schema_file.read_text(encoding="utf-8")
        except OSError as exc:
            raise SuiteError(
                f"{path}: task {raw_task['id']}: schema_path {schema_file} cannot be read ({exc})"
            ) from exc
        try:
            loaded = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise SuiteError(f"{schema_file}: not valid YAML/JSON ({exc})") from exc
        if not isinstance(loaded, dict):
            raise SuiteError(f"{schema_file}: a JSON Schema must be a mapping")
        return loaded
    return None


def load_suite(path: str | Path) -> Suite:
    """Parse, validate, render, and return a :class:`Suite`.

    Raises :class:`SuiteError` for anything the caller should exit 1 over.
    """
    path = Path(path)
    doc, raw = _load_yaml(path)
    _validate(doc, path)
    _scan_for_secrets(doc, path)

    defaults = doc.get("defaults") or {}
    base_params = _params_from(defaults, CallParams())
    default_system = defaults.get("system")

    tasks: list[Task] = []
    seen_ids: set[str] = set()
    for raw_task in doc["tasks"]:
        tid = raw_task["id"]
        if tid in seen_ids:
            raise SuiteError(f"{path}: duplicate task id {tid!r}")
        seen_ids.add(tid)

        schema = _load_task_schema(raw_task, path)
        if schema is not None:
            try:
                jsonschema.Draft202012Validator.check_schema(schema)
            except jsonschema.exceptions.SchemaError as exc:
                raise SuiteError(
                    f"{path}: task {tid}: 'schema' is not a valid Draft 2020-12 schema "
                    f"({exc.message})"
                ) from exc

        params = _params_from(
            {k: raw_task[k] for k in _PARAM_KEYS if k in raw_task}, base_params
        )
        system_template = raw_task.get("system", default_system)

        cases: list[Case] = []
        seen_cases: set[str] = set()
        for raw_case in raw_task["cases"]:
            cid = raw_case["id"]
            if cid in seen_cases:
                raise SuiteError(f"{path}: task {tid}: duplicate case id {cid!r}")
            seen_cases.add(cid)
            variables = raw_case.get("vars") or {}
            where = f"{path}: task {tid}, case {cid}"
            prompt = render(raw_task["prompt"], variables, where)
            system = (
                render(system_template, variables, where + " (system)")
                if system_template is not None
                else None
            )
            cases.append(
                Case(
                    id=cid,
                    vars=dict(variables),
                    prompt=prompt,
                    system=system,
                    check=_check_from(raw_task.get("check"), raw_case.get("check")),
                )
            )

        tasks.append(
            Task(
                id=tid,
                prompt_template=raw_task["prompt"],
                kind=raw_task.get("kind"),
                schema=schema,
                rubric=raw_task.get("rubric"),
                judge=bool(raw_task.get("judge", True)),
                params=params,
                cases=tuple(cases),
                hash=_task_hash(raw_task),
            )
        )

    raw_judge = doc["judge"]
    raw_cal = raw_judge.get("calibration") or {}
    judge = JudgeConfig(
        model=raw_judge["model"],
        rubric=raw_judge["rubric"],
        max_tokens=int(raw_judge.get("max_tokens", DEFAULT_JUDGE_MAX_TOKENS)),
        temperature=float(raw_judge.get("temperature", DEFAULT_JUDGE_TEMPERATURE)),
        calibration=Calibration(
            min_kappa=float(raw_cal.get("min_kappa", DEFAULT_MIN_KAPPA)),
            min_labels=int(raw_cal.get("min_labels", DEFAULT_MIN_LABELS)),
            on_below_floor=raw_cal.get("on_below_floor", "flag"),
        ),
    )

    by_id = {t.id: t for t in tasks}
    labels: list[Label] = []
    for raw_label in _load_labels(doc, path):
        task = by_id.get(raw_label["task"])
        if task is None:
            raise SuiteError(
                f"{path}: label references task {raw_label['task']!r}, which the suite "
                "does not define"
            )
        if raw_label["case"] not in {c.id for c in task.cases}:
            raise SuiteError(
                f"{path}: label references case {raw_label['case']!r} of task "
                f"{task.id!r}, which the suite does not define"
            )
        labels.append(
            Label(
                task=raw_label["task"],
                case=raw_label["case"],
                baseline=raw_label["baseline"],
                candidate=raw_label["candidate"],
                prefer=raw_label["prefer"],
                note=raw_label.get("note"),
            )
        )

    limits = doc.get("limits") or {}
    report_cfg = doc.get("report") or {}
    openrouter_cfg = doc.get("openrouter") or {}

    return Suite(
        name=doc["suite"],
        version=int(doc["version"]),
        description=(doc.get("description") or None),
        judge=judge,
        tasks=tuple(tasks),
        labels=tuple(labels),
        max_cost_usd=(float(limits["max_cost_usd"]) if "max_cost_usd" in limits else None),
        mover_threshold=float(report_cfg.get("mover_threshold", DEFAULT_MOVER_THRESHOLD)),
        openrouter_provider_pins=dict(openrouter_cfg.get("provider_pins") or {}),
        path=path,
        hash=hashlib.sha256(raw).hexdigest(),
    )


def models_hash(models: list[str]) -> str:
    return hashlib.sha256("\n".join(models).encode("utf-8")).hexdigest()
