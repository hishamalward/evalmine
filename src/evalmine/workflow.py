"""Controlled direct-argv DAG workflows for enrichment and backoff experiments."""

# ruff: noqa: E501 -- the report is a compact self-contained HTML document

from __future__ import annotations

import fnmatch
import hashlib
import html
import json
import mimetypes
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml

from .experiment import ExperimentError
from .suite import SECRET_PATTERNS, canonical_bytes
from .workspace import _tree_hash, _write_once

WORKFLOW_VERSION = 1
WORKFLOW_FORMAT = "evalmine-workflow-v1"
MAX_CAPTURE_BYTES = 10 * 1024 * 1024
_IDENTIFIER = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
_SECRET_ENV_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_WORKSPACE_EXCLUDED_NAMES = {
    ".claude",
    ".codex",
    ".gemini",
    ".git",
    ".git-credentials",
    ".netrc",
    ".next",
    ".npmrc",
    ".pypirc",
    ".evalmine-cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "node_modules",
}
_WORKSPACE_EXCLUDED_SUFFIXES = (".key", ".p12", ".pem", ".pfx")
_write_lock = threading.Lock()


class WorkflowError(ExperimentError):
    """A workflow could not be validated, executed, or verified."""


class WorkflowRefused(WorkflowError):
    """A workflow was stopped before any declared command launched."""


@dataclass(frozen=True)
class FrozenFixture:
    source: Path
    source_declared: str
    base: str
    target: str
    sha256: str


@dataclass(frozen=True)
class WorkflowNode:
    id: str
    argv: tuple[str, ...]
    cwd: str
    env: tuple[tuple[str, str], ...]
    needs: tuple[str, ...]
    fan_out: tuple[str, ...]
    artifacts: tuple[str, ...]
    timeout: int
    provider_calls: str


@dataclass(frozen=True)
class Workflow:
    name: str
    path: Path
    root: Path
    max_parallel: int
    fixtures: tuple[FrozenFixture, ...]
    nodes: tuple[WorkflowNode, ...]
    hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.name,
            "version": WORKFLOW_VERSION,
            "manifest": str(self.path),
            "manifest_hash": self.hash,
            "root": str(self.root),
            "max_parallel": self.max_parallel,
            "fixtures": [
                {
                    "source": item.source_declared,
                    "base": item.base,
                    "target": item.target,
                    "sha256": item.sha256,
                }
                for item in self.fixtures
            ],
            "nodes": [
                {
                    "id": node.id,
                    "argv": list(node.argv),
                    "cwd": node.cwd,
                    "env": dict(node.env),
                    "needs": list(node.needs),
                    "fan_out": list(node.fan_out),
                    "artifacts": list(node.artifacts),
                    "timeout": node.timeout,
                    "provider_calls": node.provider_calls,
                }
                for node in self.nodes
            ],
        }


@dataclass(frozen=True)
class WorkflowResult:
    root: Path
    html: Path
    status: str
    nodes_succeeded: int
    nodes_total: int
    instances: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": WORKFLOW_FORMAT,
            "root": str(self.root),
            "html": str(self.html),
            "status": self.status,
            "nodes_succeeded": self.nodes_succeeded,
            "nodes_total": self.nodes_total,
            "instances": self.instances,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value or any(char not in _IDENTIFIER for char in value):
        raise WorkflowError(f"{where} must use letters, digits, '_' or '-'")
    return value


def _relative(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowError(f"{where} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise WorkflowError(f"{where} must stay under its declared root")
    return path.as_posix()


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WorkflowError(f"{where} must be a mapping")
    return value


def _only(mapping: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise WorkflowError(f"{where} has unknown field(s): {', '.join(unknown)}")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_secrets(value: Any, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _scan_secrets(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_secrets(item, f"{where}[{index}]")
    elif isinstance(value, str):
        for label, pattern in SECRET_PATTERNS:
            if pattern.search(value):
                raise WorkflowError(f"credential-like {label} appears at {where}")


def _node_env(value: Any, where: str) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    mapping = _mapping(value, where)
    normalized = []
    for name, item in mapping.items():
        if (
            not isinstance(name, str)
            or not name
            or name[0].isdigit()
            or not name.replace("_", "").isalnum()
        ):
            raise WorkflowError(f"{where} names must be environment identifiers")
        if any(part in name.upper() for part in _SECRET_ENV_PARTS):
            raise WorkflowError(f"{where}.{name} may not declare credential-bearing variables")
        if not isinstance(item, str):
            raise WorkflowError(f"{where}.{name} must be a string")
        parsed = urlsplit(item)
        if parsed.password is not None:
            raise WorkflowError(f"{where}.{name} may not embed a password in a URL")
        normalized.append((name, item))
    return tuple(sorted(normalized))


def load_workflow(path: str | Path) -> Workflow:
    manifest = Path(path).resolve()
    try:
        raw = manifest.read_bytes()
        doc = yaml.safe_load(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise WorkflowError(f"cannot read workflow {manifest} ({exc})") from exc
    doc = _mapping(doc, "workflow")
    _only(doc, {"workflow", "version", "root", "max_parallel", "fixtures", "nodes"}, "workflow")
    _scan_secrets(doc, "$ workflow")
    if doc.get("version") != WORKFLOW_VERSION:
        raise WorkflowError(f"workflow.version must be {WORKFLOW_VERSION}")
    name = _identifier(doc.get("workflow"), "workflow.workflow")
    root_declared = str(doc.get("root", "."))
    root = (manifest.parent / root_declared).resolve()
    if not root.is_dir():
        raise WorkflowError(f"workflow.root is not a directory: {root}")
    max_parallel = doc.get("max_parallel", 1)
    if not isinstance(max_parallel, int) or isinstance(max_parallel, bool) or not 1 <= max_parallel <= 64:
        raise WorkflowError("workflow.max_parallel must be an integer from 1 to 64")
    fixtures = []
    for index, raw_fixture in enumerate(doc.get("fixtures", [])):
        fixture = _mapping(raw_fixture, f"fixtures[{index}]")
        _only(fixture, {"source", "base", "target", "sha256"}, f"fixtures[{index}]")
        source_declared = _relative(fixture.get("source"), f"fixtures[{index}].source")
        base = fixture.get("base", "manifest")
        if base not in {"manifest", "root"}:
            raise WorkflowError(f"fixtures[{index}].base must be manifest or root")
        target = _relative(fixture.get("target"), f"fixtures[{index}].target")
        source_parent = manifest.parent if base == "manifest" else root
        source = (source_parent / source_declared).resolve()
        try:
            source.relative_to(source_parent)
        except ValueError as exc:
            raise WorkflowError(f"fixture source escapes its {base} base: {source_declared}") from exc
        if not source.is_file() or source.is_symlink():
            raise WorkflowError(f"fixture source must be a regular file: {source}")
        actual = _hash_file(source)
        expected = fixture.get("sha256", actual)
        if expected != actual:
            raise WorkflowError(f"fixture hash mismatch for {source_declared}")
        fixtures.append(FrozenFixture(source, source_declared, base, target, actual))
    nodes = []
    ids = set()
    raw_nodes = doc.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise WorkflowError("workflow.nodes must be a non-empty list")
    for index, raw_node in enumerate(raw_nodes):
        node = _mapping(raw_node, f"nodes[{index}]")
        _only(
            node,
            {
                "id",
                "argv",
                "cwd",
                "env",
                "needs",
                "fan_out",
                "artifacts",
                "timeout",
                "provider_calls",
            },
            f"nodes[{index}]",
        )
        node_id = _identifier(node.get("id"), f"nodes[{index}].id")
        if node_id in ids:
            raise WorkflowError(f"duplicate workflow node {node_id!r}")
        ids.add(node_id)
        argv = node.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(item, str) and item for item in argv):
            raise WorkflowError(f"nodes[{index}].argv must be a non-empty string list")
        cwd = _relative(node.get("cwd", "."), f"nodes[{index}].cwd")
        node_cwd = (root / cwd).resolve()
        try:
            node_cwd.relative_to(root)
        except ValueError as exc:
            raise WorkflowError(f"nodes[{index}].cwd escapes workflow.root") from exc
        if not node_cwd.is_dir() or node_cwd.is_symlink():
            raise WorkflowError(f"nodes[{index}].cwd is not a regular directory under workflow.root")
        env = _node_env(node.get("env"), f"nodes[{index}].env")
        needs = tuple(_identifier(item, f"nodes[{index}].needs") for item in node.get("needs", []))
        fan_out_raw = node.get("fan_out", [])
        if not isinstance(fan_out_raw, list) or not all(isinstance(item, str) and item for item in fan_out_raw):
            raise WorkflowError(f"nodes[{index}].fan_out must be a string list")
        if len(set(fan_out_raw)) != len(fan_out_raw):
            raise WorkflowError(f"nodes[{index}].fan_out contains duplicates")
        artifacts = tuple(
            _relative(item, f"nodes[{index}].artifacts") for item in node.get("artifacts", [])
        )
        timeout = node.get("timeout", 1800)
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 86400:
            raise WorkflowError(f"nodes[{index}].timeout must be from 1 to 86400 seconds")
        provider_calls = node.get("provider_calls", "none")
        if provider_calls not in {"none", "subscription"}:
            raise WorkflowError(
                f"nodes[{index}].provider_calls must be none or subscription; direct API "
                "work belongs in the cost-capped suite lane"
            )
        nodes.append(
            WorkflowNode(
                node_id,
                tuple(argv),
                cwd,
                env,
                needs,
                tuple(fan_out_raw),
                artifacts,
                timeout,
                provider_calls,
            )
        )
    by_id = {node.id: node for node in nodes}
    for node in nodes:
        missing = sorted(set(node.needs) - set(by_id))
        if missing:
            raise WorkflowError(f"node {node.id!r} needs unknown node(s): {', '.join(missing)}")
        if node.id in node.needs:
            raise WorkflowError(f"node {node.id!r} cannot depend on itself")
    _topological_levels(tuple(nodes))
    return Workflow(
        name,
        manifest,
        root,
        max_parallel,
        tuple(fixtures),
        tuple(nodes),
        hashlib.sha256(canonical_bytes(doc)).hexdigest(),
    )


def _topological_levels(nodes: tuple[WorkflowNode, ...]) -> list[list[WorkflowNode]]:
    remaining = {node.id: node for node in nodes}
    completed: set[str] = set()
    levels = []
    while remaining:
        ready = [node for node in nodes if node.id in remaining and set(node.needs) <= completed]
        if not ready:
            raise WorkflowError("workflow dependency graph contains a cycle")
        levels.append(ready)
        for node in ready:
            remaining.pop(node.id)
            completed.add(node.id)
    return levels


def workflow_plan(workflow: Workflow) -> dict[str, Any]:
    levels = _topological_levels(workflow.nodes)
    return {
        **workflow.as_dict(),
        "levels": [[node.id for node in level] for level in levels],
        "instance_count": sum(max(1, len(node.fan_out)) for node in workflow.nodes),
        "provider_nodes": [node.id for node in workflow.nodes if node.provider_calls != "none"],
        "commands_launched": False,
        "provider_calls": False,
    }


def _replace_item(value: str, item: str | None) -> str:
    if "{item}" in value and item is None:
        raise WorkflowError("{item} appears in a node without fan_out")
    return value.replace("{item}", item or "")


def _safe_env(item: str | None, declared: tuple[tuple[str, str], ...]) -> dict[str, str]:
    env = {
        name: value
        for name, value in os.environ.items()
        if not any(part in name.upper() for part in _SECRET_ENV_PARTS)
    }
    env["NO_COLOR"] = "1"
    env.update(declared)
    if item is not None:
        env["EVALMINE_MATRIX_ITEM"] = item
    return env


def _redact(text: str) -> str:
    for _label, pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED_CREDENTIAL]", text)
    return text


def _artifact_matches(workspace: Path, patterns: tuple[str, ...], item: str | None) -> list[Path]:
    resolved_patterns = tuple(_replace_item(pattern, item) for pattern in patterns)
    matches = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(workspace).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in resolved_patterns):
            matches.append(path)
    return matches


def _run_instance(
    node: WorkflowNode,
    item: str | None,
    *,
    workspace: Path,
    evidence: Path,
) -> dict[str, Any]:
    instance = node.id if item is None else f"{node.id}--{item}"
    safe_instance = "".join(char if char in _IDENTIFIER else "_" for char in instance)
    argv = [_replace_item(value, item) for value in node.argv]
    started = _now()
    clock = time.monotonic()
    try:
        process = subprocess.run(
            argv,
            cwd=workspace / node.cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=node.timeout,
            env=_safe_env(item, node.env),
        )
        exit_code = process.returncode
        stdout = _redact(process.stdout)
        stderr = _redact(process.stderr)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        exit_code = 124
        stdout = _redact(exc.stdout.decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""))
        stderr = _redact(exc.stderr.decode("utf-8", "replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""))
        timed_out = True
    except OSError as exc:
        exit_code, stdout, stderr, timed_out = 127, "", str(exc), False
    log_root = evidence / "logs"
    with _write_lock:
        _write_once(log_root / f"{safe_instance}.stdout.txt", stdout[:1_000_000].encode())
        _write_once(log_root / f"{safe_instance}.stderr.txt", stderr[:1_000_000].encode())
    captured = []
    if exit_code == 0:
        for source in _artifact_matches(workspace, node.artifacts, item):
            relative = source.relative_to(workspace).as_posix()
            if source.stat().st_size > MAX_CAPTURE_BYTES:
                captured.append({"path": relative, "status": "too-large", "size": source.stat().st_size})
                continue
            target = evidence / "artifacts" / safe_instance / relative
            with _write_lock:
                _write_once(target, source.read_bytes())
            mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            captured.append(
                {
                    "path": relative,
                    "evidence": target.relative_to(evidence).as_posix(),
                    "sha256": _hash_file(source),
                    "size": source.stat().st_size,
                    "mime": mime,
                    "eye_test": mime.startswith("image/") or mime == "text/html",
                }
            )
    return {
        "instance": instance,
        "node": node.id,
        "item": item,
        "status": "succeeded" if exit_code == 0 else "failed",
        "started_at": started,
        "completed_at": _now(),
        "duration_ms": round((time.monotonic() - clock) * 1000),
        "argv": argv,
        "cwd": node.cwd,
        "env": dict(node.env),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "provider_calls": node.provider_calls,
        "artifacts": captured,
    }


def _copy_workspace(source: Path, target: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            name
            for name in names
            if name in _WORKSPACE_EXCLUDED_NAMES
            or name == ".env"
            or name.startswith(".env.")
            or name.lower().endswith(_WORKSPACE_EXCLUDED_SUFFIXES)
        }

    shutil.copytree(source, target, symlinks=True, ignore=ignore)


def _evidence_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _hash_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "workflow.json" and "workspace" not in path.relative_to(root).parts
    }


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _render_html(data: dict[str, Any]) -> str:
    nodes = "".join(
        f"<article class='{_esc(node['status'])}'><div><b>{_esc(node['id'])}</b><small> needs {_esc(', '.join(node['needs']) or 'nothing')}</small></div><strong>{_esc(node['status'])}</strong><p>{node['succeeded']}/{node['instances']} instances succeeded · {node['artifact_count']} artifacts</p></article>"
        for node in data["nodes"]
    )
    fixtures = "".join(
        f"<li><code>{_esc(item['target'])}</code> ← {_esc(item['source'])} <small>{_esc(item['sha256'][:12])}</small></li>"
        for item in data["fixtures"]
    ) or "<li>None</li>"
    artifacts = "".join(
        f"<li>{'<span class=eye>eye-test</span>' if item.get('eye_test') else ''} <code>{_esc(item['evidence'])}</code> · {_esc(item['mime'])} · {_esc(item['size'])} bytes</li>"
        for instance in data["instances"]
        for item in instance["artifacts"]
        if item.get("evidence")
    ) or "<li>No declared artifacts captured.</li>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{_esc(data['workflow'])} · workflow</title><style>:root{{--ink:#18231e;--paper:#f3efe5;--card:#fffdf8;--green:#166044;--red:#9b332c;--line:#d7d0c0}}*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui}}header,main,footer{{max-width:1050px;margin:auto;padding:28px}}header{{padding-top:60px}}h1{{font:700 clamp(36px,6vw,68px)/1 Georgia,serif;margin:.2em 0}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:.13em;font-weight:800;color:var(--green)}}section,article{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:18px;margin:14px 0}}article{{display:grid;grid-template-columns:1fr auto;gap:4px 16px}}article p{{grid-column:1/-1;margin:0}}article.failed{{border-left:6px solid var(--red)}}article.succeeded{{border-left:6px solid var(--green)}}small{{display:block;color:#657168}}code{{font-size:.86em}}.eye{{background:#e4bd62;padding:2px 7px;border-radius:99px;font-size:12px;font-weight:700}}@media(max-width:650px){{header,main,footer{{padding:18px}}}}</style></head><body><header><div class="eyebrow">evalmine · workflow evidence</div><h1>{_esc(data['workflow'])}</h1><p>Status <b>{_esc(data['status'])}</b> · {data['instances_succeeded']}/{data['instance_count']} instances succeeded</p></header><main><section><div class="eyebrow">Dependency graph</div><h2>Nodes in execution order</h2>{nodes}</section><section><div class="eyebrow">Frozen inputs</div><h2>Fixtures restored before execution</h2><ul>{fixtures}</ul></section><section><div class="eyebrow">Review</div><h2>Captured artifacts</h2><ul>{artifacts}</ul></section></main><footer>Generated {_esc(data['completed_at'])} · self-contained · command logs and artifacts are retained beside this report</footer></body></html>"""


def run_workflow(
    workflow: Workflow,
    out_dir: str | Path,
    *,
    allow_commands: bool,
    allow_provider_calls: bool = False,
) -> WorkflowResult:
    if not allow_commands:
        raise WorkflowRefused("workflow nodes launch declared commands; pass --allow-commands")
    provider_nodes = [node.id for node in workflow.nodes if node.provider_calls != "none"]
    if provider_nodes and not allow_provider_calls:
        raise WorkflowRefused(
            "subscription provider nodes require --allow-provider-calls: " + ", ".join(provider_nodes)
        )
    out_base = Path(out_dir).resolve()
    result_root = out_base / f"{workflow.name}-{workflow.hash[:12]}"
    if result_root.exists() or result_root.is_symlink():
        raise WorkflowRefused(f"workflow evidence already exists at {result_root}")
    try:
        result_root.relative_to(workflow.root)
    except ValueError:
        pass
    else:
        raise WorkflowRefused("workflow --out must be outside workflow.root")
    result_root.mkdir(parents=True)
    workspace = result_root / "workspace"
    _copy_workspace(workflow.root, workspace)
    inputs = result_root / "inputs"
    for index, fixture in enumerate(workflow.fixtures, 1):
        frozen = inputs / f"{index:03d}-{fixture.source.name}"
        _write_once(frozen, fixture.source.read_bytes())
        if _hash_file(frozen) != fixture.sha256:
            raise WorkflowError(f"frozen fixture changed while copying: {fixture.source_declared}")
        target = workspace / fixture.target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(frozen, target)
    started_at = _now()
    instances = []
    node_status: dict[str, str] = {}
    for level in _topological_levels(workflow.nodes):
        runnable = [node for node in level if all(node_status.get(dep) == "succeeded" for dep in node.needs)]
        skipped = [node for node in level if node not in runnable]
        for node in skipped:
            node_status[node.id] = "skipped"
        futures = {}
        with ThreadPoolExecutor(max_workers=workflow.max_parallel) as pool:
            for node in runnable:
                items: tuple[str | None, ...] = node.fan_out or (None,)
                for item in items:
                    future = pool.submit(_run_instance, node, item, workspace=workspace, evidence=result_root)
                    futures[future] = node.id
            level_results = []
            for future in as_completed(futures):
                level_results.append(future.result())
        instances.extend(sorted(level_results, key=lambda item: item["instance"]))
        for node in runnable:
            own = [item for item in level_results if item["node"] == node.id]
            node_status[node.id] = "succeeded" if own and all(item["status"] == "succeeded" for item in own) else "failed"
    completed_at = _now()
    node_rows = []
    for node in workflow.nodes:
        own = [item for item in instances if item["node"] == node.id]
        node_rows.append(
            {
                "id": node.id,
                "needs": list(node.needs),
                "status": node_status[node.id],
                "instances": len(own) or max(1, len(node.fan_out)),
                "succeeded": sum(item["status"] == "succeeded" for item in own),
                "artifact_count": sum(len(item["artifacts"]) for item in own),
            }
        )
    status = "completed" if all(value == "succeeded" for value in node_status.values()) else "partial"
    data = {
        "format": WORKFLOW_FORMAT,
        "workflow": workflow.name,
        "manifest": str(workflow.path),
        "manifest_hash": workflow.hash,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "provider_calls_authorized": allow_provider_calls,
        "fixtures": workflow.as_dict()["fixtures"],
        "nodes": node_rows,
        "instances": instances,
        "instance_count": len(instances),
        "instances_succeeded": sum(item["status"] == "succeeded" for item in instances),
        "final_workspace_tree_hash": _tree_hash(workspace),
    }
    report_root = result_root / "report"
    _write_once(report_root / "data.json", (json.dumps(data, indent=2, sort_keys=True) + "\n").encode())
    _write_once(report_root / "index.html", _render_html(data).encode())
    marker = {
        "format": WORKFLOW_FORMAT,
        "workflow": workflow.name,
        "manifest_hash": workflow.hash,
        "status": status,
        "created_at": completed_at,
        "final_workspace_tree_hash": data["final_workspace_tree_hash"],
        "evidence_sha256": _evidence_hashes(result_root),
    }
    _write_once(result_root / "workflow.json", (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode())
    return WorkflowResult(
        result_root,
        report_root / "index.html",
        status,
        sum(row["status"] == "succeeded" for row in node_rows),
        len(node_rows),
        len(instances),
    )


def verify_workflow(root: str | Path) -> dict[str, Any]:
    resolved = Path(root).resolve()
    marker_path = resolved / "workflow.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError(f"cannot read workflow marker {marker_path} ({exc})") from exc
    if marker.get("format") != WORKFLOW_FORMAT:
        raise WorkflowError("unknown workflow evidence format")
    if marker.get("evidence_sha256") != _evidence_hashes(resolved):
        raise WorkflowError("workflow evidence changed after creation")
    actual_tree = _tree_hash(resolved / "workspace")
    if actual_tree != marker.get("final_workspace_tree_hash"):
        raise WorkflowError("workflow workspace changed after execution")
    data = json.loads((resolved / "report" / "data.json").read_text(encoding="utf-8"))
    return {
        "ok": True,
        "format": WORKFLOW_FORMAT,
        "root": str(resolved),
        "workflow": marker.get("workflow"),
        "status": marker.get("status"),
        "instance_count": data.get("instance_count"),
        "artifact_count": sum(len(item.get("artifacts", [])) for item in data.get("instances", [])),
    }
