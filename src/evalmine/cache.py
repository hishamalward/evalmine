"""Disk cache keyed by content hash. Spec: docs/spec.md S6.5.

What is deliberately *not* in the key: the date, the run id, the suite hash, the
suite file path, the price table, the API key, the environment. Two runs a month
apart with the same prompt and params hit the same entry - that is the feature,
and it is what makes a report re-derivable without spending again.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .suite import canonical_bytes

CACHE_FORMAT_VERSION = 1
DEFAULT_CACHE_DIR = ".evalmine-cache"


def cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def answer_payload(
    *,
    provider: str,
    model: str,
    system: str | None,
    prompt: str,
    params: dict[str, Any],
    schema: dict[str, Any] | None,
    schema_mode: str,
    adapter_version: int,
    repeat: int = 0,
) -> dict[str, Any]:
    return {
        "v": CACHE_FORMAT_VERSION,
        "kind": "answer",
        "provider": provider,
        "model": model,
        "system": system,
        "prompt": prompt,
        "params": params,
        "schema": schema,
        "schema_mode": schema_mode,
        "adapter_version": adapter_version,
        "repeat": repeat,
    }


def judge_payload(
    *,
    provider: str,
    model: str,
    system: str | None,
    prompt: str,
    params: dict[str, Any],
    schema: dict[str, Any] | None,
    schema_mode: str,
    adapter_version: int,
    repeat: int = 0,
) -> dict[str, Any]:
    """Identical shape with ``kind: "judge"``.

    A swapped pass renders the two answers in the other order, so its prompt
    differs and it gets its own key. Nothing extra is needed to keep the two
    passes of a pair distinct.
    """
    payload = answer_payload(
        provider=provider,
        model=model,
        system=system,
        prompt=prompt,
        params=params,
        schema=schema,
        schema_mode=schema_mode,
        adapter_version=adapter_version,
        repeat=repeat,
    )
    payload["kind"] = "judge"
    return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    corrupt: int = 0


class Cache:
    """Content-addressed JSON entries under ``<root>/<provider>/<key[:2]>/``."""

    def __init__(self, root: str | Path = DEFAULT_CACHE_DIR, read: bool = True) -> None:
        self.root = Path(root)
        #: ``--no-cache`` skips the read and still performs the write: there is
        #: no flag that disables writing, because a run you cannot re-report
        #: from is not worth having.
        self.read_enabled = read
        self.stats = CacheStats()

    def path_for(self, provider: str, key: str) -> Path:
        return self.root / provider / key[:2] / f"{key}.json"

    def get(self, provider: str, key: str) -> dict[str, Any] | None:
        if not self.read_enabled:
            self.stats.misses += 1
            return None
        path = self.path_for(provider, key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.stats.misses += 1
            return None
        except (OSError, ValueError):
            # A corrupt or unreadable entry is a miss, and will be overwritten.
            self.stats.corrupt += 1
            self.stats.misses += 1
            return None
        if not isinstance(entry, dict) or "text" not in entry:
            self.stats.corrupt += 1
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return entry

    def put(self, provider: str, key: str, entry: dict[str, Any]) -> Path:
        path = self.path_for(provider, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"key": key, "created_at": _utc_now(), **entry}
        # Atomic enough that a killed run leaves no half-written entry behind.
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
        self.stats.writes += 1
        return path
