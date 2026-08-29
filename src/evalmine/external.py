"""Immutable import of externally generated artifacts.

The external lane deliberately has no generation hook. It validates a directory of
hash-pinned JSONL files, normalizes each record into a create-once evidence envelope,
and exposes that envelope to the existing report, judge, and decision lifecycles.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema
import yaml

EXTERNAL_FORMAT = "evalmine-external-artifacts-v1"
EXTERNAL_MANIFEST = "evalmine-import.yaml"
EXTERNAL_RECORD_FORMAT = "evalmine-external-record-v1"
COST_BASES = ("estimated", "ledger", "dashboard_observed")

_SCHEMA_PATH = Path(__file__).with_name("external.schema.json")
_RECORD_SCHEMA_PATH = Path(__file__).with_name("external-record.schema.json")


class ExternalArtifactError(Exception):
    """An external artifact bundle is unsafe, malformed, or has changed."""


@dataclass(frozen=True)
class ExternalImportResult:
    root: Path
    plan_id: str
    name: str
    record_count: int
    block_count: int
    condition_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "format": EXTERNAL_FORMAT,
            "root": str(self.root),
            "plan_id": self.plan_id,
            "experiment": self.name,
            "record_count": self.record_count,
            "block_count": self.block_count,
            "condition_count": self.condition_count,
            "provider_calls": False,
        }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _canonical_line(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(content)
    except FileExistsError as exc:
        raise ExternalArtifactError(f"refusing to overwrite external evidence {path}") from exc
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _evidence_hashes(root: Path) -> dict[str, str]:
    derived_lifecycles = {"report", "judging", "decision", "validation", "execution"}
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name != "external.json"
        and path.relative_to(root).parts[0] not in derived_lifecycles
    }


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(bundle: Path) -> tuple[dict[str, Any], bytes]:
    manifest = bundle / EXTERNAL_MANIFEST
    try:
        raw = manifest.read_bytes()
        doc = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as exc:
        raise ExternalArtifactError(f"cannot read {manifest} ({exc})") from exc
    if not isinstance(doc, dict):
        raise ExternalArtifactError(f"{manifest}: expected a YAML mapping")
    try:
        jsonschema.Draft202012Validator(_load_schema(_SCHEMA_PATH)).validate(doc)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "$"
        raise ExternalArtifactError(f"{manifest}: {location}: {exc.message}") from exc
    evaluation = doc["evaluation"]
    style = evaluation.get("ranking_style", "pairwise")
    judge = evaluation["judge"]
    if style == "pairwise" and judge.get("enabled") and (
        not judge.get("pairwise") or not judge.get("position_swap")
    ):
        raise ExternalArtifactError(
            f"{manifest}: pairwise ranking requires pairwise and position_swap judging"
        )
    if style == "n-way" and judge.get("enabled") and (
        judge.get("pairwise") or judge.get("position_swap")
    ):
        raise ExternalArtifactError(
            f"{manifest}: n-way ranking requires pairwise=false and position_swap=false"
        )
    return doc, raw


def _safe_source(bundle: Path, declared: str) -> Path:
    pure = PurePosixPath(declared)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ExternalArtifactError(f"unsafe artifact path {declared!r}")
    candidate = bundle.joinpath(*pure.parts)
    if candidate.is_symlink() or not candidate.is_file():
        raise ExternalArtifactError(f"artifact {declared!r} is not a regular file")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(bundle)
    except ValueError as exc:
        raise ExternalArtifactError(f"artifact {declared!r} escapes the bundle") from exc
    return resolved


def _normalize_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def _receipt_reconciliation(receipts: dict[str, Any]) -> dict[str, float | None]:
    def ratio(numerator: str, denominator: str) -> float | None:
        top = receipts.get(numerator)
        bottom = receipts.get(denominator)
        if not isinstance(top, dict) or not isinstance(bottom, dict):
            return None
        denominator_value = float(bottom["usd"])
        return float(top["usd"]) / denominator_value if denominator_value > 0 else None

    return {
        "ledger_to_estimated": ratio("ledger", "estimated"),
        "dashboard_to_estimated": ratio("dashboard_observed", "estimated"),
        "dashboard_to_ledger": ratio("dashboard_observed", "ledger"),
    }


def _read_records(
    bundle: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[tuple[str, bytes, str]]]:
    validator = jsonschema.Draft202012Validator(_load_schema(_RECORD_SCHEMA_PATH))
    records: list[dict[str, Any]] = []
    sources: list[tuple[str, bytes, str]] = []
    for artifact in manifest["artifacts"]:
        declared = str(artifact["path"])
        source = _safe_source(bundle, declared)
        raw = source.read_bytes()
        actual_hash = _sha256(raw)
        if actual_hash != artifact["sha256"]:
            raise ExternalArtifactError(
                f"{declared}: sha256 mismatch (expected {artifact['sha256']}, got {actual_hash})"
            )
        sources.append((declared, raw, actual_hash))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExternalArtifactError(f"{declared}: JSONL is not UTF-8 ({exc})") from exc
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ExternalArtifactError(
                    f"{declared}:{line_number}: invalid JSON ({exc.msg})"
                ) from exc
            try:
                validator.validate(record)
            except jsonschema.ValidationError as exc:
                location = ".".join(str(part) for part in exc.absolute_path) or "$"
                raise ExternalArtifactError(
                    f"{declared}:{line_number}: {location}: {exc.message}"
                ) from exc
            if record.get("output") is None:
                raise ExternalArtifactError(f"{declared}:{line_number}: output cannot be null")
            receipts = dict(record.get("cost_receipts", {}))
            normalized = {
                "format": EXTERNAL_RECORD_FORMAT,
                **record,
                "output_text": _normalize_output(record["output"]),
                "cost_receipts": receipts,
                "cost_reconciliation": _receipt_reconciliation(receipts),
                "provenance": {
                    "source": declared,
                    "source_line": line_number,
                    "source_sha256": actual_hash,
                },
            }
            records.append(normalized)
    if not records:
        raise ExternalArtifactError("the bundle contains no non-empty JSONL records")
    return records, sources


def _validate_grid(records: list[dict[str, Any]], fields: list[str]) -> dict[str, Any]:
    conditions: dict[str, dict[str, Any]] = {}
    blocks: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in records:
        condition = record["condition"]
        condition_id = str(condition["id"])
        previous = conditions.setdefault(condition_id, condition)
        if previous != condition:
            raise ExternalArtifactError(
                f"condition {condition_id!r} has inconsistent model/prompt/width metadata"
            )
        key = (str(record["lane"]), str(record["item_id"]), str(record["account_id"]))
        bucket = blocks.setdefault(key, [])
        if bucket and bucket[0]["prompt"] != record["prompt"]:
            raise ExternalArtifactError(
                "comparison block has inconsistent prompts: " + "/".join(key)
            )
        if any(row["condition"]["id"] == condition_id for row in bucket):
            raise ExternalArtifactError(
                "duplicate condition in block " + "/".join(key) + f": {condition_id}"
            )
        bucket.append(record)
        values = record.get("fields")
        if values is None and isinstance(record.get("output"), dict):
            values = record["output"]
        missing = [field for field in fields if not isinstance(values, dict) or field not in values]
        if missing:
            raise ExternalArtifactError(
                f"{'/'.join(key)}/{condition_id}: missing declared fields {', '.join(missing)}"
            )
    short = ["/".join(key) for key, rows in blocks.items() if len(rows) < 2]
    if short:
        raise ExternalArtifactError(
            "every comparison block needs at least two conditions; incomplete: "
            + ", ".join(short[:5])
        )
    lane_sets: dict[str, set[str]] = {}
    for (lane, _item, _account), rows in blocks.items():
        found = {str(row["condition"]["id"]) for row in rows}
        expected = lane_sets.setdefault(lane, found)
        if found != expected:
            raise ExternalArtifactError(
                f"lane {lane!r} has inconsistent condition coverage across items"
            )
    return {
        "condition_count": len(conditions),
        "block_count": len(blocks),
        "conditions": conditions,
    }


def import_external_artifacts(
    bundle_dir: str | Path,
    out_dir: str | Path,
    *,
    imported_at: str | None = None,
) -> ExternalImportResult:
    """Pin and normalize a completed artifact bundle without launching providers."""
    bundle = Path(bundle_dir)
    if bundle.is_symlink() or not bundle.is_dir():
        raise ExternalArtifactError(f"external artifact bundle is not a directory: {bundle}")
    bundle = bundle.resolve()
    manifest, manifest_bytes = _load_manifest(bundle)
    fields = [str(item) for item in manifest["evaluation"].get("fields", [])]
    records, sources = _read_records(bundle, manifest)
    grid = _validate_grid(records, fields)
    input_descriptor = {
        "manifest_sha256": _sha256(manifest_bytes),
        "sources": [{"path": path, "sha256": digest} for path, _raw, digest in sources],
        "records_sha256": _sha256(b"".join(_canonical_line(row) for row in records)),
    }
    input_hash = _sha256(_canonical_line(input_descriptor))
    name = str(manifest["external_artifacts"])
    plan_id = f"{name}-{input_hash[:12]}"
    root = Path(out_dir).resolve()
    if root.exists() or root.is_symlink():
        raise ExternalArtifactError(
            f"external evidence already exists at {root}; evidence is never overwritten"
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    root.mkdir()
    try:
        _write_once(root / EXTERNAL_MANIFEST, manifest_bytes)
        for index, (declared, raw, digest) in enumerate(sources, 1):
            suffix = Path(declared).name
            target = root / "sources" / f"{index:03d}-{digest[:12]}-{suffix}"
            _write_once(target, raw)
        normalized = b"".join(_canonical_line(row) for row in records)
        _write_once(root / "artifacts.jsonl", normalized)
        index = {
            "format": EXTERNAL_FORMAT,
            "plan_id": plan_id,
            "experiment": name,
            "question": manifest["question"],
            "input_hash": input_hash,
            "record_count": len(records),
            "block_count": grid["block_count"],
            "condition_count": grid["condition_count"],
            "conditions": grid["conditions"],
            "evaluation": manifest["evaluation"],
            "sources": input_descriptor["sources"],
            "provider_calls": False,
        }
        _write_once(root / "index.json", _json_bytes(index))
        marker = {
            **index,
            "root": str(root),
            "imported_at": imported_at or datetime.now(timezone.utc).isoformat(),
            "evidence_sha256": _evidence_hashes(root),
        }
        _write_once(root / "external.json", _json_bytes(marker))
    except Exception:
        # Only files created in the brand-new exact output directory can exist here.
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.chmod(stat.S_IWUSR | stat.S_IRUSR)
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        root.rmdir()
        raise
    return ExternalImportResult(
        root=root,
        plan_id=plan_id,
        name=name,
        record_count=len(records),
        block_count=int(grid["block_count"]),
        condition_count=int(grid["condition_count"]),
    )


def is_external_import(root: str | Path) -> bool:
    return (Path(root) / "external.json").is_file()


def verify_external_import(root: str | Path) -> dict[str, Any]:
    path = Path(root)
    if path.is_symlink():
        raise ExternalArtifactError(f"external evidence root must not be a symlink: {path}")
    resolved = path.resolve()
    marker_path = resolved / "external.json"
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalArtifactError(f"{resolved} is not valid external evidence ({exc})") from exc
    if marker.get("format") != EXTERNAL_FORMAT:
        raise ExternalArtifactError(f"{marker_path} has an unknown external format")
    if marker.get("root") != str(resolved):
        raise ExternalArtifactError("external marker points at a different evidence root")
    expected = marker.get("evidence_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ExternalArtifactError("external marker has no evidence hashes")
    actual = _evidence_hashes(resolved)
    if actual != expected:
        changed = sorted(set(actual) ^ set(expected))
        if not changed:
            changed = sorted(path for path in actual if actual[path] != expected.get(path))
        raise ExternalArtifactError(
            "external evidence changed after import: " + (", ".join(changed[:5]) or "files")
        )
    records = load_external_records(resolved, verify=False)
    if len(records) != marker.get("record_count"):
        raise ExternalArtifactError("external record count differs from its marker")
    return {
        "ok": True,
        "format": EXTERNAL_FORMAT,
        "root": str(resolved),
        "plan_id": marker.get("plan_id"),
        "experiment": marker.get("experiment"),
        "record_count": marker.get("record_count"),
        "run_count": marker.get("record_count"),
        "block_count": marker.get("block_count"),
        "condition_count": marker.get("condition_count"),
        "provider_calls": False,
    }


def load_external_index(root: str | Path, *, verify: bool = True) -> dict[str, Any]:
    resolved = Path(root).resolve()
    if verify:
        verify_external_import(resolved)
    try:
        value = json.loads((resolved / "index.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExternalArtifactError(f"cannot read external index ({exc})") from exc
    if not isinstance(value, dict) or value.get("format") != EXTERNAL_FORMAT:
        raise ExternalArtifactError("external index has an unknown format")
    return value


def load_external_records(root: str | Path, *, verify: bool = True) -> list[dict[str, Any]]:
    resolved = Path(root).resolve()
    if verify:
        verify_external_import(resolved)
    try:
        lines = (resolved / "artifacts.jsonl").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ExternalArtifactError(f"cannot read normalized external artifacts ({exc})") from exc
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExternalArtifactError(
                f"normalized artifacts:{line_number}: invalid JSON ({exc.msg})"
            ) from exc
        if not isinstance(value, dict) or value.get("format") != EXTERNAL_RECORD_FORMAT:
            raise ExternalArtifactError(
                f"normalized artifacts:{line_number}: unknown record format"
            )
        records.append(value)
    return records
