"""Shared fixtures. Nothing here touches the network; nothing here needs a key."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_SUITE = REPO_ROOT / "examples" / "everyday-eight.yaml"
SPEC = REPO_ROOT / "docs" / "spec.md"
PRICES_DIR = REPO_ROOT / "prices"


MINIMAL_SUITE: dict[str, Any] = {
    "suite": "tiny",
    "version": 1,
    "judge": {
        "model": "fake/judge",
        "rubric": "Prefer the answer that does what was asked.",
        "calibration": {"min_kappa": 0.40, "min_labels": 2},
    },
    "tasks": [
        {
            "id": "echo",
            "prompt": "Say something about {{thing}}.",
            "cases": [
                {"id": "one", "vars": {"thing": "sqlite"}},
                {"id": "two", "vars": {"thing": "yaml"}},
            ],
        }
    ],
}


@pytest.fixture
def minimal_suite() -> dict[str, Any]:
    return copy.deepcopy(MINIMAL_SUITE)


@pytest.fixture
def write_suite(tmp_path: Path):
    """Write a suite dict to a temp file and return its path."""

    def _write(doc: dict[str, Any], name: str = "suite.yaml") -> Path:
        path = tmp_path / name
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    return _write
