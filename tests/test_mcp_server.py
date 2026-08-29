"""The MCP surface: three tools over the fake adapter. Spec S11.

Skips cleanly if the ``mcp`` package is not installed (it is an optional
extra); in this environment it is installed, so these tests exercise the
real MCP SDK - tools are registered on a real ``MCPServer`` and called
through ``server.call_tool``, not by calling the implementation functions
directly.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import shutil

import pytest
import yaml
from conftest import MINIMAL_SUITE, PRICES_DIR

pytest.importorskip("mcp")

import evalmine.mcp_server as mcp_server  # noqa: E402

MODELS = ["fake/a", "fake/b"]


def call_tool(name: str, arguments: dict):
    assert mcp_server.server is not None
    return asyncio.run(mcp_server.server.call_tool(name, arguments))


@pytest.fixture
def mcp_env(tmp_path, monkeypatch):
    """cwd -> an empty tmp_path with a real price table, no MCP env overrides."""
    monkeypatch.chdir(tmp_path)
    env_names = (
        "EVALMINE_MCP_SUITE_ROOT",
        "EVALMINE_MCP_MAX_COST",
        "EVALMINE_MCP_MAX_COST_CEILING",
        "EVALMINE_MCP_ALLOW_PROVIDER_CALLS",
        "EVALMINE_MCP_ALLOW_VALIDATOR_COMMANDS",
        "EVALMINE_MCP_ALLOW_EXTERNAL_WRITES",
        "EVALMINE_MCP_ALLOW_WORKFLOW_COMMANDS",
    )
    for name in env_names:
        monkeypatch.delenv(name, raising=False)
    prices_dir = tmp_path / "prices"
    prices_dir.mkdir()
    shutil.copy(PRICES_DIR / "prices-2026-08-23.yaml", prices_dir / "prices-2026-08-23.yaml")
    return tmp_path


def write_suite(tmp_path, doc, name="suite.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def fake_suite_doc():
    doc = copy.deepcopy(MINIMAL_SUITE)
    doc["judge"]["model"] = "fake/judge"
    return doc


def test_import_external_artifacts_tool_is_zero_provider_and_root_scoped(mcp_env):
    bundle = mcp_env / "external-bundle"
    bundle.mkdir()
    rows = [
        {
            "lane": "example",
            "item_id": "one",
            "account_id": "redacted",
            "prompt": "Choose the better completed answer.",
            "condition": {
                "id": condition,
                "model": f"provider/{condition}",
                "prompt_variant": "v1",
                "width": "default",
            },
            "output": f"output from {condition}",
        }
        for condition in ("a", "b")
    ]
    raw = b"".join((json.dumps(row) + "\n").encode() for row in rows)
    (bundle / "completed.jsonl").write_bytes(raw)
    manifest = {
        "external_artifacts": "mcp-external",
        "version": 1,
        "question": "Which completed answer is stronger?",
        "artifacts": [
            {"path": "completed.jsonl", "sha256": hashlib.sha256(raw).hexdigest()}
        ],
        "evaluation": {
            "objectives": ["Correctness"],
            "blind": "condition",
            "human": {"required": True, "coverage": "calibration-subset"},
            "judge": {
                "enabled": False,
                "pairwise": True,
                "position_swap": True,
                "calibrate": True,
            },
        },
    }
    (bundle / "evalmine-import.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )
    result = call_tool(
        "import_external_artifacts_tool",
        {"bundle_path": bundle.name, "out_dir": "external-evidence"},
    )
    assert result.is_error is False
    assert result.structured_content["record_count"] == 2
    assert result.structured_content["provider_calls"] is False


# --------------------------------------------------------------------------
# run_suite
# --------------------------------------------------------------------------


def test_run_suite_tool_against_the_fake_adapter(mcp_env):
    suite_path = write_suite(mcp_env, fake_suite_doc())
    result = call_tool(
        "run_suite",
        {"suite_path": suite_path.name, "models": MODELS, "no_cache": False},
    )
    assert result.is_error is False
    out = result.structured_content
    assert out is not None
    assert out["run_id"]
    assert out["report_path"]
    assert out["report_md_path"]
    assert "headline_eligible" in out
    assert out["calibration"]["status"] in {
        "ok",
        "below_floor",
        "insufficient_labels",
        "no_labels",
        "undefined_pe_1",
    }
    roles = {row["model"]: row["role"] for row in out["per_model"]}
    assert roles == {"fake/a": "baseline", "fake/b": "candidate"}
    assert out["totals"]["cost_usd"] > 0
    # The tool never leaks raw provider text back to the caller.
    assert "text" not in out and "answers" not in out


# --------------------------------------------------------------------------
# last_report
# --------------------------------------------------------------------------


def test_last_report_tool_reads_the_run_just_written(mcp_env):
    suite_path = write_suite(mcp_env, fake_suite_doc())
    first = call_tool("run_suite", {"suite_path": suite_path.name, "models": MODELS})
    assert first.is_error is False

    result = call_tool("last_report", {"suite_path": suite_path.name})
    assert result.is_error is False
    out = result.structured_content
    assert out["found"] is True
    assert out["run_id"] == first.structured_content["run_id"]
    assert out["generated_at"]
    assert out["summary"]["per_model"]


def test_last_report_tool_reports_not_found_for_a_never_run_suite(mcp_env):
    suite_path = write_suite(mcp_env, fake_suite_doc(), name="never-run.yaml")
    result = call_tool("last_report", {"suite_path": suite_path.name})
    assert result.is_error is False
    assert result.structured_content == {"found": False}


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------


def test_compare_tool_diffs_two_reports(mcp_env):
    suite_path = write_suite(mcp_env, fake_suite_doc())
    first = call_tool("run_suite", {"suite_path": suite_path.name, "models": MODELS})
    run_id_a = first.structured_content["run_id"]

    # A second run of an unmodified suite is a full cache hit and gets a
    # different run-id only via the timestamp component - write a trivially
    # modified suite so the second report differs and is worth diffing.
    doc = fake_suite_doc()
    doc["tasks"][0]["cases"].append({"id": "three", "vars": {"thing": "parquet"}})
    write_suite(mcp_env, doc)
    second = call_tool("run_suite", {"suite_path": suite_path.name, "models": MODELS})
    run_id_b = second.structured_content["run_id"]

    result = call_tool("compare", {"report_a": run_id_a, "report_b": run_id_b})
    assert result.is_error is False
    out = result.structured_content
    assert "comparable" in out
    assert out["reason"] is not None  # the suite hash changed
    assert out["tasks_modified"] or out["suite_hash_changed"]


# --------------------------------------------------------------------------
# refusals (S11.4) - money is never at risk
# --------------------------------------------------------------------------


def test_run_suite_refuses_over_cap_and_spends_nothing(mcp_env):
    suite_path = write_suite(mcp_env, fake_suite_doc())
    result = call_tool(
        "run_suite",
        {"suite_path": suite_path.name, "models": MODELS, "max_cost": 0.0000001},
    )
    assert result.is_error is False
    out = result.structured_content
    assert out["refused"] is True
    assert out["reason"] == "estimate_exceeds_cap"
    assert out["estimate_usd"] > out["cap_usd"]
    # Nothing was spent: no cache entry, no report, no adapter call.
    assert not (mcp_env / ".evalmine-cache").exists() or not any(
        (mcp_env / ".evalmine-cache").rglob("*.json")
    )
    assert not (mcp_env / "reports").exists()


def test_run_suite_refuses_a_suite_path_outside_the_root(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setenv("EVALMINE_MCP_SUITE_ROOT", str(root))

    escaped = write_suite(outside, fake_suite_doc())
    result = call_tool(
        "run_suite", {"suite_path": str(escaped), "models": MODELS}
    )
    assert result.is_error is False
    out = result.structured_content
    assert out["refused"] is True
    assert out["reason"] == "suite_path_outside_root"
    assert not (root / "reports").exists()

    # last_report is read-only but takes the same suite_path input and gets
    # the same containment check.
    result = call_tool("last_report", {"suite_path": str(escaped)})
    assert result.structured_content["refused"] is True
    assert result.structured_content["reason"] == "suite_path_outside_root"


def test_run_suite_refuses_a_max_cost_above_the_ceiling(mcp_env, monkeypatch):
    monkeypatch.setenv("EVALMINE_MCP_MAX_COST_CEILING", "5.00")
    suite_path = write_suite(mcp_env, fake_suite_doc())
    result = call_tool(
        "run_suite", {"suite_path": suite_path.name, "models": MODELS, "max_cost": 9.00}
    )
    out = result.structured_content
    assert out["refused"] is True
    assert out["reason"] == "max_cost_exceeds_ceiling"
    assert out["ceiling_usd"] == 5.00
    assert not (mcp_env / "reports").exists()


# --------------------------------------------------------------------------
# v2 episode/workflow control plane
# --------------------------------------------------------------------------


def test_v2_workflow_plan_is_zero_action_and_run_requires_server_and_client_gates(mcp_env):
    seed = mcp_env / "seed"
    seed.mkdir()
    fixture = mcp_env / "fixture.json"
    fixture.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = mcp_env / "workflow.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "workflow": "mcp-dag",
                "version": 1,
                "root": "seed",
                "fixtures": [
                    {"source": "fixture.json", "target": "input/fixture.json"}
                ],
                "nodes": [
                    {
                        "id": "one",
                        "argv": ["does-not-run-during-plan"],
                        "artifacts": [],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    planned = call_tool("plan_workflow", {"manifest_path": manifest.name})
    assert planned.is_error is False
    assert planned.structured_content["commands_launched"] is False
    assert planned.structured_content["instance_count"] == 1

    refused = call_tool(
        "run_workflow_tool",
        {
            "manifest_path": manifest.name,
            "out_dir": "workflow-runs",
            "confirm_commands": True,
        },
    )
    assert refused.is_error is False
    assert refused.structured_content == {
        "refused": True,
        "reason": "workflow_commands_not_authorized",
    }
    assert not (mcp_env / "workflow-runs").exists()


def test_v2_experiment_paths_cannot_escape_the_mcp_root(mcp_env, tmp_path):
    outside = tmp_path.parent / "outside-experiment.yaml"
    result = call_tool("plan_experiment", {"manifest_path": str(outside)})
    assert result.is_error is False
    assert result.structured_content["refused"] is True
    assert result.structured_content["reason"] == "manifest_path_outside_root"
