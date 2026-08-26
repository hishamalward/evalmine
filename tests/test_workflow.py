"""Controlled workflow DAGs, frozen fixtures, and artifact evidence."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

from evalmine.workflow import (
    WorkflowError,
    WorkflowRefused,
    load_workflow,
    run_workflow,
    verify_workflow,
    workflow_plan,
)


@pytest.fixture
def workflow_manifest(tmp_path: Path) -> Path:
    workspace = tmp_path / "seed"
    workspace.mkdir()
    (workspace / "fixture-source.json").write_text('{"value": 7}\n', encoding="utf-8")
    script = workspace / "fixture.py"
    script.write_text(
        """from pathlib import Path
import json, sys
mode = sys.argv[1]
item = sys.argv[2] if len(sys.argv) > 2 else None
out = Path('out')
out.mkdir(exist_ok=True)
if mode == 'fan':
    source = json.loads(Path('input/frozen.json').read_text())
    (out / f'{item}.json').write_text(json.dumps({'item': item, **source}))
elif mode == 'merge':
    rows = sorted(path.name for path in out.glob('*.json'))
    (out / 'eye-test.html').write_text('<h1>' + ','.join(rows) + '</h1>')
""",
        encoding="utf-8",
    )
    doc = {
        "workflow": "tiny-dag",
        "version": 1,
        "root": "seed",
        "max_parallel": 2,
        "fixtures": [
            {"source": "seed/fixture-source.json", "target": "input/frozen.json"}
        ],
        "nodes": [
            {
                "id": "fan",
                "argv": [sys.executable, "fixture.py", "fan", "{item}"],
                "fan_out": ["a", "b"],
                "artifacts": ["out/{item}.json"],
            },
            {
                "id": "merge",
                "needs": ["fan"],
                "argv": [sys.executable, "fixture.py", "merge"],
                "artifacts": ["out/eye-test.html"],
            },
        ],
    }
    manifest = tmp_path / "workflow.yaml"
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return manifest


def test_workflow_plan_resolves_fanout_dependencies_and_frozen_hashes(workflow_manifest):
    workflow = load_workflow(workflow_manifest)
    plan = workflow_plan(workflow)
    assert plan["levels"] == [["fan"], ["merge"]]
    assert plan["instance_count"] == 3
    assert len(plan["fixtures"][0]["sha256"]) == 64
    assert plan["commands_launched"] is False


def test_root_based_fixtures_and_node_cwd_env_are_explicit(workflow_manifest, tmp_path):
    doc = yaml.safe_load(workflow_manifest.read_text())
    seed = workflow_manifest.parent / "seed"
    (seed / "subdir").mkdir()
    doc["fixtures"] = [
        {
            "source": "fixture-source.json",
            "base": "root",
            "target": "input/frozen.json",
        }
    ]
    doc["nodes"] = [
        {
            "id": "inspect",
            "cwd": "subdir",
            "env": {"WORKFLOW_TEST": "isolated"},
            "argv": [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; import json, os; "
                    "Path('../out').mkdir(); "
                    "Path('../out/env.json').write_text(json.dumps("
                    "{'cwd': Path.cwd().name, 'value': os.environ['WORKFLOW_TEST']}))"
                ),
            ],
            "artifacts": ["out/env.json"],
        }
    ]
    workflow_manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")

    workflow = load_workflow(workflow_manifest)
    plan = workflow_plan(workflow)
    assert plan["fixtures"][0]["base"] == "root"
    assert plan["nodes"][0]["cwd"] == "subdir"
    assert plan["nodes"][0]["env"] == {"WORKFLOW_TEST": "isolated"}

    result = run_workflow(workflow, tmp_path / "runs", allow_commands=True)
    captured = json.loads(
        (result.root / "workspace" / "out" / "env.json").read_text(encoding="utf-8")
    )
    assert captured == {"cwd": "subdir", "value": "isolated"}


def test_workspace_copy_excludes_local_credentials_and_dependency_caches(
    workflow_manifest, tmp_path
):
    seed = workflow_manifest.parent / "seed"
    (seed / ".env").write_text("DO_NOT_COPY=1\n", encoding="utf-8")
    (seed / ".env.local").write_text("DO_NOT_COPY=1\n", encoding="utf-8")
    (seed / ".claude").mkdir()
    (seed / ".claude" / "settings.json").write_text("{}\n", encoding="utf-8")
    (seed / "node_modules").mkdir()
    (seed / "node_modules" / "sentinel").write_text("large\n", encoding="utf-8")

    result = run_workflow(
        load_workflow(workflow_manifest), tmp_path / "runs", allow_commands=True
    )
    workspace = result.root / "workspace"
    assert not (workspace / ".env").exists()
    assert not (workspace / ".env.local").exists()
    assert not (workspace / ".claude").exists()
    assert not (workspace / "node_modules").exists()


@pytest.mark.parametrize(
    ("env", "message"),
    [
        ({"API_KEY": "placeholder"}, "credential-bearing"),
        ({"DATABASE_URL": "postgresql://user:password@localhost/db"}, "embed a password"),
    ],
)
def test_workflow_rejects_declared_secret_environment(workflow_manifest, env, message):
    doc = yaml.safe_load(workflow_manifest.read_text())
    doc["nodes"][0]["env"] = env
    workflow_manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(WorkflowError, match=message):
        load_workflow(workflow_manifest)


def test_workflow_requires_command_gate_then_captures_eye_test_artifact(
    workflow_manifest, tmp_path
):
    workflow = load_workflow(workflow_manifest)
    with pytest.raises(WorkflowRefused, match="allow-commands"):
        run_workflow(workflow, tmp_path / "refused", allow_commands=False)
    result = run_workflow(workflow, tmp_path / "runs", allow_commands=True)
    assert result.status == "completed" and result.instances == 3
    verified = verify_workflow(result.root)
    assert verified["artifact_count"] == 3
    data = json.loads((result.root / "report" / "data.json").read_text())
    eye_tests = [
        artifact
        for instance in data["instances"]
        for artifact in instance["artifacts"]
        if artifact.get("eye_test")
    ]
    assert [item["mime"] for item in eye_tests] == ["text/html"]


def test_provider_nodes_have_a_separate_gate(workflow_manifest, tmp_path):
    doc = yaml.safe_load(workflow_manifest.read_text())
    doc["nodes"][0]["provider_calls"] = "subscription"
    workflow_manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    workflow = load_workflow(workflow_manifest)
    with pytest.raises(WorkflowRefused, match="allow-provider-calls"):
        run_workflow(workflow, tmp_path / "runs", allow_commands=True)


def test_cycles_and_direct_api_shell_nodes_fail_validation(workflow_manifest):
    original = yaml.safe_load(workflow_manifest.read_text())
    cyclic = copy.deepcopy(original)
    cyclic["nodes"][0]["needs"] = ["merge"]
    workflow_manifest.write_text(yaml.safe_dump(cyclic, sort_keys=False), encoding="utf-8")
    with pytest.raises(WorkflowError, match="cycle"):
        load_workflow(workflow_manifest)

    api = copy.deepcopy(original)
    api["nodes"][0]["provider_calls"] = "api"
    workflow_manifest.write_text(yaml.safe_dump(api, sort_keys=False), encoding="utf-8")
    with pytest.raises(WorkflowError, match="cost-capped suite lane"):
        load_workflow(workflow_manifest)


def test_verification_detects_later_workspace_edits(workflow_manifest, tmp_path):
    result = run_workflow(
        load_workflow(workflow_manifest), tmp_path / "runs", allow_commands=True
    )
    (result.root / "workspace" / "out" / "a.json").write_text("changed\n")
    with pytest.raises(WorkflowError, match="workspace changed"):
        verify_workflow(result.root)
