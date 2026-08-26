"""Phase-3 runner preflight, command mapping, and evidence tests.

No test invokes a real provider CLI. ``FakeDriver`` behaves like three installed
executables while recording exactly what evalmine would have launched.
"""

from __future__ import annotations

import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalmine.cli import EXIT_REFUSED_PREFLIGHT, _print_experiment_progress, main
from evalmine.decision import (
    DecisionError,
    EpisodeJudgeCall,
    generate_decision_report,
    judge_experiment,
    verify_decision,
    verify_judging,
)
from evalmine.experiment import load_experiment
from evalmine.experiment_report import (
    CHOICES,
    ExperimentReportError,
    build_experiment_report_data,
    episode_ab_run_keys,
    generate_experiment_report,
    preferred_run_by_choice,
    render_experiment_report_html,
    verify_experiment_report,
)
from evalmine.runner import (
    ExecutionRefused,
    ProcessDriver,
    ProcessResult,
    RunnerError,
    execute_experiment,
    preflight_experiment,
    verify_execution,
)
from evalmine.validators import (
    ValidationError,
    ValidationRefused,
    check_experiment,
    verify_validation,
)
from evalmine.workspace import discard_prepared, prepare_experiment, verify_prepared


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.fixture
def runner_experiment(tmp_path: Path):
    repo = tmp_path / "seed"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "evalmine tests")
    _git(repo, "config", "user.email", "evalmine@example.invalid")
    (repo / "README.md").write_text("runner fixture\n", encoding="utf-8")
    (repo / "CLAUDE.md").write_text("claude instructions\n", encoding="utf-8")
    (repo / "AGENTS.md").write_text("codex instructions\n", encoding="utf-8")
    (repo / "GEMINI.md").write_text("gemini instructions\n", encoding="utf-8")
    plugin = repo / "plugins" / "candidate" / ".claude-plugin"
    plugin.mkdir(parents=True)
    (plugin / "plugin.json").write_text(
        '{"name":"candidate","version":"0.1.0"}\n', encoding="utf-8"
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "seed")

    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    doc: dict[str, Any] = {
        "experiment": "runner-test",
        "version": 2,
        "question": "Can every installed runner execute the same episode?",
        "seed": {
            "repo": "../seed",
            "ref": "HEAD",
            "dirty": "reject",
            "untracked": "deny",
        },
        "isolation": {
            "workspace": "copy",
            "session": "fresh-per-run",
            "external_writes": "deny",
        },
        "schedule": {"order": "rotate", "max_parallel": 3},
        "validators": {
            "repo-unchanged": {"type": "repository-diff", "expect": "unchanged"},
            "readme-present": {
                "type": "required-files",
                "paths": ["README.md"],
                "non_empty": True,
            },
            "follow-up-covered": {
                "type": "required-sections",
                "target": "final-response",
                "sections": ["final for Second private fake prompt."],
            },
        },
        "arms": [
            {
                "id": "claude",
                "runner": "claude-code",
                "model": "claude-test",
                "auth": "subscription",
                "configuration": {"instructions": "none", "plugins": "none"},
            },
            {
                "id": "codex",
                "runner": "codex-cli",
                "model": "codex-test",
                "auth": "subscription",
                "configuration": {"instructions": "none", "plugins": "none"},
            },
            {
                "id": "gemini",
                "runner": "gemini-cli",
                "model": "gemini-test",
                "auth": "subscription",
                "configuration": {"instructions": "none", "plugins": "none"},
            },
        ],
        "episodes": [
            {
                "id": "two-turns",
                "turns": [
                    {"prompt": "First private fake prompt."},
                    {"prompt": "Second private fake prompt."},
                ],
                "validators": [
                    "repo-unchanged",
                    "readme-present",
                    "follow-up-covered",
                ],
            }
        ],
        "evaluation": {
            "objectives": ["Correctness"],
            "blind": "arm-identity",
            "human": {"required": True},
            "judge": {
                "enabled": True,
                "pairwise": True,
                "position_swap": True,
                "calibrate": True,
                "runner": "codex-cli",
                "model": "codex-judge-test",
                "min_labels": 3,
            },
        },
    }
    manifest = manifest_dir / "experiment.yaml"
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    experiment = load_experiment(manifest)
    prepared = prepare_experiment(experiment, tmp_path / "artifacts")
    yield prepared, manifest, doc
    if prepared.root.exists():
        discard_prepared(prepared.root)


@pytest.fixture
def fake_executables(tmp_path: Path) -> dict[str, str]:
    result = {}
    for runner in ("claude-code", "codex-cli", "gemini-cli"):
        path = tmp_path / f"fake-{runner}"
        path.write_text("not executed; FakeDriver owns this path\n", encoding="utf-8")
        result[runner] = str(path)
    return result


class FakeDriver(ProcessDriver):
    HELP = " ".join(
        [
            "--print",
            "--output-format",
            "--model",
            "--resume",
            "--session-id",
            "--settings",
            "--safe-mode",
            "--strict-mcp-config",
            "--mcp-config",
            "--disable-slash-commands",
            "--permission-mode",
            "--no-chrome",
            "--json",
            "--sandbox",
            "--cd",
            "--skip-git-repo-check",
            "resume",
            "--ask-for-approval",
            "--ignore-user-config",
            "--ephemeral",
            "--config",
            "--prompt",
            "--extensions",
            "--approval-mode",
            "--max-budget-usd",
            "--add-dir",
            "--plugin-dir",
        ]
    )

    def __init__(
        self,
        *,
        fail_runner: str | None = None,
        emit_secret_runner: str | None = None,
        mutate_runner: str | None = None,
    ):
        self.calls: list[dict[str, Any]] = []
        self.fail_runner = fail_runner
        self.emit_secret_runner = emit_secret_runner
        self.mutate_runner = mutate_runner
        self._lock = threading.Lock()

    def run(self, args, *, cwd, input_text=None, timeout, env=None):
        executable = Path(args[0]).name
        runner = executable.removeprefix("fake-")
        with self._lock:
            self.calls.append(
                {
                    "runner": runner,
                    "args": list(args),
                    "cwd": str(cwd),
                    "input": input_text,
                    "env_names": sorted((env or {}).keys()),
                }
            )
        if args[-1] == "--version":
            return ProcessResult(tuple(args), 0, f"{runner} fake-1.0\n", "", 1)
        if "--help" in args:
            return ProcessResult(tuple(args), 0, self.HELP + "\n", "", 1)
        if self.fail_runner == runner:
            return ProcessResult(tuple(args), 9, "", "fake failure\n", 2)
        if self.mutate_runner == runner:
            (cwd / "README.md").write_text(
                "runner fixture\nagent changed this\n", encoding="utf-8"
            )

        model = args[args.index("--model") + 1]
        run_key = cwd.parent.name
        session_id = f"session-{run_key}"
        final_suffix = input_text
        if self.emit_secret_runner == runner:
            final_suffix = "sk-" + "A" * 24
        if runner == "claude-code":
            if "--session-id" in args:
                session_id = args[args.index("--session-id") + 1]
            else:
                session_id = args[args.index("--resume") + 1]
            events = [
                {"type": "system", "subtype": "init", "session_id": session_id},
                {
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "tool_use", "name": "Read", "id": "tool-1"}]
                    },
                },
                {
                    "type": "result",
                    "session_id": session_id,
                    "result": f"claude final for {final_suffix}",
                    "modelUsage": {model: {"inputTokens": 4, "outputTokens": 3}},
                    "usage": {"input_tokens": 4, "output_tokens": 3},
                    "total_cost_usd": 0.0125,
                },
            ]
        elif runner == "codex-cli":
            events = []
            if "resume" not in args:
                events.append({"type": "thread.started", "thread_id": session_id})
            events.append(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "item-1",
                        "type": "agent_message",
                        "text": f"codex final for {final_suffix}",
                    },
                    "model": model,
                }
            )
        else:
            if "--resume" in args:
                session_id = args[args.index("--resume") + 1]
            events = [
                {"type": "init", "session_id": session_id, "model": model},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": f"gemini final for {final_suffix}",
                },
            ]
        stdout = "".join(json.dumps(event) + "\n" for event in events)
        return ProcessResult(tuple(args), 0, stdout, "", 3)


def _execution_calls(driver: FakeDriver) -> list[dict[str, Any]]:
    return [
        call
        for call in driver.calls
        if "--help" not in call["args"] and call["args"][-1] != "--version"
    ]


def test_preflight_is_read_only_and_probes_all_runners(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    before = sorted(path.relative_to(prepared.root) for path in prepared.root.rglob("*"))
    driver = FakeDriver()
    report = preflight_experiment(
        prepared.root, executable_overrides=fake_executables, driver=driver
    )
    after = sorted(path.relative_to(prepared.root) for path in prepared.root.rglob("*"))
    assert report.ok is True
    assert {probe.runner for probe in report.probes} == set(fake_executables)
    assert before == after
    assert _execution_calls(driver) == []


def test_execute_maps_three_clis_reuses_only_run_local_sessions_and_writes_evidence(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    driver = FakeDriver()
    result = execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=driver,
    )
    assert result.status == "completed"
    assert result.succeeded == 3
    assert verify_prepared(prepared.root)["ok"] is True
    assert verify_execution(prepared.root)["status"] == "completed"

    calls = _execution_calls(driver)
    assert len(calls) == 6
    expected_prompts = {"First private fake prompt.", "Second private fake prompt."}
    assert all(call["input"] in expected_prompts for call in calls)
    assert all("private fake prompt" not in " ".join(call["args"]) for call in calls)
    assert all(
        not any(
            marker in name.upper()
            for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
        )
        for call in calls
        for name in call["env_names"]
    )

    claude = [call for call in calls if call["runner"] == "claude-code"]
    assert "--safe-mode" in claude[0]["args"]
    assert "--session-id" in claude[0]["args"]
    assert "--resume" in claude[1]["args"]
    assert claude[0]["args"][claude[0]["args"].index("--session-id") + 1] == (
        claude[1]["args"][claude[1]["args"].index("--resume") + 1]
    )
    codex = [call for call in calls if call["runner"] == "codex-cli"]
    assert "workspace-write" in codex[0]["args"]
    assert all("--skip-git-repo-check" in call["args"] for call in codex)
    assert "--ignore-user-config" in codex[0]["args"]
    assert "resume" in codex[1]["args"]
    gemini = [call for call in calls if call["runner"] == "gemini-cli"]
    assert ["--extensions", "none"] == gemini[0]["args"][-2:]
    assert "--resume" in gemini[1]["args"]

    marker = json.loads((result.root / "execution.json").read_text(encoding="utf-8"))
    assert marker["status"] == "completed"
    assert marker["credentials_captured"] is False
    assert marker["evidence_sha256"]
    for run in prepared.runs:
        run_dir = result.root / "runs" / run.run_key
        assert (run_dir / "run.json").is_file()
        assert (run_dir / "turn-001.raw.jsonl").is_file()
        assert (run_dir / "turn-002.final.txt").read_text(encoding="utf-8")
        turn = json.loads((run_dir / "turn-001.json").read_text(encoding="utf-8"))
        assert turn["command"][-1] == "<PROMPT_VIA_STDIN>"


def test_partial_runner_failure_keeps_evidence_and_returns_partial(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    result = execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(fail_runner="codex-cli"),
    )
    assert result.status == "partial"
    assert result.succeeded == 2
    codex_run = next(run for run in prepared.runs if run.arm_id == "codex")
    run_evidence = json.loads(
        (result.root / "runs" / codex_run.run_key / "run.json").read_text(encoding="utf-8")
    )
    assert run_evidence["status"] == "failed"
    assert run_evidence["error"] == "turn 1 exited 9"
    assert (result.root / "runs" / codex_run.run_key / "turn-001.stderr.txt").is_file()


def test_execution_reports_safe_progress_without_prompts(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    events: list[dict[str, Any]] = []
    result = execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
        progress=events.append,
    )

    assert result.status == "completed"
    assert events[0] == {
        "event": "execution_started",
        "at": events[0]["at"],
        "run_count": 3,
        "max_parallel": 3,
    }
    assert events[-1]["event"] == "execution_completed"
    assert events[-1]["succeeded"] == 3
    turn_starts = [event for event in events if event["event"] == "turn_started"]
    turn_ends = [event for event in events if event["event"] == "turn_completed"]
    assert len(turn_starts) == len(turn_ends) == 6
    assert {event["run_position"] for event in turn_starts} == {1, 2, 3}
    assert all("prompt" not in event for event in events)


def test_cli_progress_is_immediate_and_human_readable(capsys):
    _print_experiment_progress(
        {"event": "execution_started", "run_count": 3, "max_parallel": 3}
    )
    _print_experiment_progress(
        {
            "event": "turn_completed",
            "run_position": 2,
            "run_count": 3,
            "model": "gpt-test",
            "arm": "codex",
            "turn": 1,
            "turn_count": 2,
            "status": "succeeded",
            "duration_ms": 369_000,
        }
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.splitlines() == [
        "execution started: 3 runs, max_parallel=3",
        "[2/3] gpt-test (codex) - turn 1/2 succeeded - 6m09s",
    ]


def test_execution_verification_detects_tampering(runner_experiment, fake_executables):
    prepared, _, _ = runner_experiment
    result = execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    target = next(result.root.glob("runs/*/turn-001.final.txt"))
    target.chmod(0o644)
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RunnerError, match="execution evidence changed"):
        verify_execution(prepared.root)


def test_execution_requires_explicit_confirmation(runner_experiment, fake_executables):
    prepared, _, _ = runner_experiment
    with pytest.raises(ExecutionRefused, match="allow-provider-calls"):
        execute_experiment(
            prepared.root,
            allow_provider_calls=False,
            executable_overrides=fake_executables,
            driver=FakeDriver(),
        )
    assert not (prepared.root / "execution").exists()


def test_execution_redacts_key_shaped_runner_output_before_writing(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    result = execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(emit_secret_runner="claude-code"),
    )
    keyish = "sk-" + "A" * 24
    for path in result.root.rglob("*"):
        if path.is_file():
            assert keyish not in path.read_text(encoding="utf-8")
    claude_run = next(run for run in prepared.runs if run.arm_id == "claude")
    final = (
        result.root / "runs" / claude_run.run_key / "turn-001.final.txt"
    ).read_text(encoding="utf-8")
    assert "[REDACTED_CREDENTIAL]" in final


def test_objective_validators_pass_and_write_verifiable_evidence(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    result = check_experiment(prepared.root)
    assert result.verdict == "passed"
    assert result.passed == 3
    assert verify_validation(prepared.root)["verdict"] == "passed"
    for run in prepared.runs:
        run_dir = result.root / "runs" / run.run_key
        assert (run_dir / "repo-unchanged.patch").read_text(encoding="utf-8") == ""
        summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        assert summary["validators_passed"] == 3
        assert summary["validator_order"] == [
            "repo-unchanged",
            "readme-present",
            "follow-up-covered",
        ]


def test_repository_diff_uses_the_treated_baseline_and_failed_checks_remain_evidence(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    changed_run = prepared.runs[0]
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(mutate_runner="claude-code"),
    )
    result = check_experiment(prepared.root)
    assert result.verdict == "failed"
    assert result.passed == 2
    diff = json.loads(
        (result.root / "runs" / changed_run.run_key / "repo-unchanged.json").read_text(
            encoding="utf-8"
        )
    )
    assert diff["status"] == "failed"
    assert diff["changes"] == [{"change": "modified", "path": "README.md"}]
    patch = (result.root / "runs" / changed_run.run_key / "repo-unchanged.patch").read_text(
        encoding="utf-8"
    )
    assert " runner fixture" in patch
    assert "+agent changed this" in patch


def test_check_refuses_to_attribute_workspace_edits_made_after_execution(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    changed_run = prepared.runs[0]
    (changed_run.workspace / "README.md").write_text("edited later\n", encoding="utf-8")
    result = check_experiment(prepared.root)
    assert result.verdict == "failed"
    summary = json.loads(
        (result.root / "runs" / changed_run.run_key / "run.json").read_text(
            encoding="utf-8"
        )
    )
    assert "changed after agent execution" in summary["error"]


class ValidatorDriver(ProcessDriver):
    def __init__(self):
        self.calls: list[tuple[list[str], Path]] = []

    def run(self, args, *, cwd, input_text=None, timeout, env=None):
        self.calls.append((list(args), cwd))
        secret = "sk-" + "B" * 24
        return ProcessResult(tuple(args), 0, f"ok {secret}\n", "", 7)


def test_command_validators_have_a_separate_gate_and_redact_output(
    runner_experiment, fake_executables
):
    original, manifest, doc = runner_experiment
    doc["experiment"] = "runner-command-validator-test"
    doc["validators"]["unit-tests"] = {
        "type": "command",
        "argv": ["fake-test", "--quiet"],
        "timeout_s": 20,
    }
    doc["episodes"][0]["validators"].append("unit-tests")
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    prepared = prepare_experiment(
        load_experiment(manifest), original.root.parent.parent / "command-artifacts"
    )
    try:
        execute_experiment(
            prepared.root,
            allow_provider_calls=True,
            executable_overrides=fake_executables,
            driver=FakeDriver(),
        )
        with pytest.raises(ValidationRefused, match="allow-validator-commands"):
            check_experiment(prepared.root)
        assert not (prepared.root / "validation").exists()

        driver = ValidatorDriver()
        result = check_experiment(
            prepared.root, allow_validator_commands=True, driver=driver
        )
        assert result.verdict == "passed"
        assert len(driver.calls) == 3
        assert all(call[0] == ["fake-test", "--quiet"] for call in driver.calls)
        for stdout in result.root.glob("runs/*/unit-tests.stdout.txt"):
            content = stdout.read_text(encoding="utf-8")
            assert "sk-" + "B" * 24 not in content
            assert "[REDACTED_CREDENTIAL]" in content
    finally:
        discard_prepared(prepared.root)


def test_validation_verification_detects_tampering(runner_experiment, fake_executables):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    result = check_experiment(prepared.root)
    target = next(result.root.glob("runs/*/follow-up-covered.json"))
    target.chmod(0o644)
    target.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="validation evidence changed"):
        verify_validation(prepared.root)


def test_cli_check_and_verify_include_validation_evidence(
    runner_experiment, fake_executables, capsys
):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    assert main(["experiment", "check", str(prepared.root)]) == 0
    assert "provider runners launched: 0" in capsys.readouterr().out
    assert main(["experiment", "report", str(prepared.root)]) == 0
    assert "blind pairs" in capsys.readouterr().out
    assert main(["experiment", "verify", str(prepared.root), "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["validation"]["verdict"] == "passed"
    assert verified["report"]["pair_count"] == 3


def test_episode_pair_mapping_is_stable_complete_and_never_self_pairs():
    observed = set()
    for index in range(100):
        pair_id = f"pair-{index}"
        a_run, b_run = episode_ab_run_keys(pair_id, "run-z", "run-a")
        assert {a_run, b_run} == {"run-a", "run-z"}
        assert episode_ab_run_keys(pair_id, "run-z", "run-a") == (a_run, b_run)
        mapping = preferred_run_by_choice(a_run, b_run)
        assert set(mapping) == set(CHOICES)
        assert {mapping["A"], mapping["B"]} == {"run-a", "run-z"}
        observed.add(a_run)
    assert observed == {"run-a", "run-z"}
    with pytest.raises(ValueError, match="distinct"):
        episode_ab_run_keys("pair", "same", "same")


def test_episode_report_is_self_contained_blind_and_maps_every_label_to_the_visible_run(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    check_experiment(prepared.root)
    result = generate_experiment_report(
        prepared.root, generated_at="2026-08-25T12:00:00+00:00"
    )
    assert result.pair_count == 3
    assert verify_experiment_report(prepared.root)["pair_count"] == 3
    html = result.html.read_text(encoding="utf-8")
    assert "<script src=" not in html and "<link rel=" not in html
    assert "http://" not in html and "https://" not in html
    assert 'body data-reveal="0"' in html
    assert ".identity{display:none" in html
    assert html.count('data-choice="A"') == 3
    assert html.count('data-choice="B"') == 3
    assert html.count('data-choice="tie"') == 3
    assert html.count('data-choice="unclear"') == 3
    assert "evalmine:episode-labels:" in html
    assert "Export labels JSON" in html and "Import labels" in html
    assert "pair.preferred_run_by_choice[row.choice]" in html
    assert "repo-unchanged" in html and "follow-up-covered" in html

    blob = html.split('<script type="application/json" id="evalmine-episode-data">')[1]
    blob = blob.split("</script>")[0]
    page = json.loads(blob)
    run_keys = {run.run_key for run in prepared.runs}
    for pair in page["pairs"]:
        assert pair["a_run_key"] in run_keys and pair["b_run_key"] in run_keys
        assert pair["a_run_key"] != pair["b_run_key"]
        mapping = pair["preferred_run_by_choice"]
        assert mapping["A"] == pair["a_run_key"]
        assert mapping["B"] == pair["b_run_key"]


def test_episode_report_escapes_model_output_and_embedded_json(runner_experiment, fake_executables):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    report = build_experiment_report_data(
        prepared.root, generated_at="2026-08-25T12:00:00+00:00"
    )
    hostile = '</pre><script>alert("owned")</script> & <b>bold</b>'
    report["runs"][0]["final"] = hostile
    for pair in report["pairs"]:
        for side in ("a", "b"):
            if pair[side]["run_key"] == report["runs"][0]["run_key"]:
                pair[side]["final"] = hostile
    html = render_experiment_report_html(report)
    assert '<script>alert("owned")' not in html
    assert "&lt;/pre&gt;&lt;script&gt;alert" in html
    blob = html.split('<script type="application/json" id="evalmine-episode-data">')[1]
    blob = blob.split("</script>")[0]
    assert "<" not in blob and ">" not in blob
    json.loads(blob)


def test_episode_report_is_create_once_and_tamper_evident(
    runner_experiment, fake_executables
):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    result = generate_experiment_report(prepared.root)
    with pytest.raises(ExperimentReportError, match="never overwritten"):
        generate_experiment_report(prepared.root)
    result.html.chmod(0o644)
    result.html.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ExperimentReportError, match="report evidence changed"):
        verify_experiment_report(prepared.root)


def test_episode_judging_labels_calibration_and_decision_report_are_end_to_end(
    runner_experiment, fake_executables, tmp_path
):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    check_experiment(prepared.root)
    report = build_experiment_report_data(prepared.root)
    scripted = iter(
        [
            ("2", "candidate"),
            ("1", "candidate"),
            ("1", "baseline"),
            ("2", "baseline"),
            ("2", "candidate"),
            ("1", "candidate"),
        ]
    )

    def judge_call(_prompt: str) -> EpisodeJudgeCall:
        winner, reason = next(scripted)
        return EpisodeJudgeCall(
            text=json.dumps({"winner": winner, "reason": reason}),
            latency_ms=7,
            input_tokens=10,
            output_tokens=3,
            cost_usd=0.001,
            raw="safe raw judge evidence",
            observed_model="fake-judge",
        )

    result = judge_experiment(
        prepared.root,
        allow_provider_calls=False,
        caller=judge_call,
    )
    assert result.call_count == 6
    assert verify_judging(prepared.root)["pair_count"] == 3

    judged = json.loads((prepared.root / "judging" / "pairs.json").read_text())
    category_by_pair = {row["pair_id"]: row["category"] for row in judged["pairs"]}
    labels = []
    for pair in report["pairs"]:
        category = category_by_pair[pair["pair_id"]]
        ordered = sorted((pair["a_run_key"], pair["b_run_key"]))
        preferred = ordered[1] if category == "consistent_win" else ordered[0]
        choice = "A" if preferred == pair["a_run_key"] else "B"
        labels.append(
            {
                "pair_id": pair["pair_id"],
                "choice": choice,
                "preferred_run_key": preferred,
                "identities_revealed": False,
            }
        )
    label_path = tmp_path / "labels.json"
    label_path.write_text(
        json.dumps(
            {
                "format": "evalmine-human-labels-v1",
                "plan_id": report["plan_id"],
                "annotator": "human-one",
                "labels": labels,
            }
        ),
        encoding="utf-8",
    )
    decision = generate_decision_report(prepared.root, [label_path])
    assert decision.headline_eligible is True
    assert verify_decision(prepared.root)["labelled_pairs"] == 3
    html = decision.html.read_text(encoding="utf-8")
    assert "Pairwise evidence" in html and "Human ↔ judge disagreements" in html
    with pytest.raises(DecisionError, match="already exists"):
        generate_decision_report(prepared.root, [label_path])


def test_cli_fake_episode_judge_is_offline_and_needs_no_provider_confirmation(
    runner_experiment, fake_executables, capsys
):
    prepared, _, _ = runner_experiment
    execute_experiment(
        prepared.root,
        allow_provider_calls=True,
        executable_overrides=fake_executables,
        driver=FakeDriver(),
    )
    assert main(["experiment", "judge", str(prepared.root), "--fake"]) == 0
    assert "position-swapped calls" in capsys.readouterr().out
    assert verify_judging(prepared.root)["call_count"] == 6


def test_preflight_rejects_api_auth_and_unsafe_runner_overrides(
    runner_experiment, fake_executables
):
    prepared, manifest, doc = runner_experiment
    doc["experiment"] = "runner-refusal-test"
    doc["arms"][0]["auth"] = "api"
    doc["arms"][0]["max_cost_usd"] = 0.25
    doc["arms"][0]["runner"] = "codex-cli"
    doc["arms"][0]["configuration"] = {
        "instructions": "files",
        "instruction_files": ["candidate.md"],
        "plugins": "allowlist",
        "plugin_allowlist": ["candidate-plugin"],
        "settings": {"sandbox.enabled": False},
        "arguments": ["--dangerously-skip-permissions"],
    }
    (manifest.parent / "candidate.md").write_text("candidate\n", encoding="utf-8")
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    refusal = prepare_experiment(
        load_experiment(manifest), prepared.root.parent.parent / "refusal-artifacts"
    )
    try:
        report = preflight_experiment(
            refusal.root, executable_overrides=fake_executables, driver=FakeDriver()
        )
        assert report.ok is False
        issue_text = "\n".join(report.issues)
        assert "only Claude Code" in issue_text
        assert "plugin allowlist" in issue_text
        assert "unsafe setting" in issue_text
        assert "unsafe argument" in issue_text
    finally:
        discard_prepared(refusal.root)


def test_claude_api_arm_has_a_native_budget_ceiling_and_only_receives_its_api_key(
    runner_experiment, fake_executables, tmp_path, monkeypatch
):
    prepared, manifest, doc = runner_experiment
    doc["experiment"] = "api-ceiling-test"
    doc["arms"][0]["auth"] = "api"
    doc["arms"][0]["max_cost_usd"] = 0.25
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    api_prepared = prepare_experiment(
        load_experiment(manifest), prepared.root.parent.parent / "api-artifacts"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only-not-evidence")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-claude")
    driver = FakeDriver()
    try:
        result = execute_experiment(
            api_prepared.root,
            allow_provider_calls=True,
            executable_overrides=fake_executables,
            driver=driver,
        )
        assert result.status == "completed"
        claude_calls = [
            call for call in _execution_calls(driver) if call["runner"] == "claude-code"
        ]
        assert all("--max-budget-usd" in call["args"] for call in claude_calls)
        assert all(
            call["args"][call["args"].index("--max-budget-usd") + 1] == "0.125"
            for call in claude_calls
        )
        assert all("ANTHROPIC_API_KEY" in call["env_names"] for call in claude_calls)
        run_key = next(run.run_key for run in api_prepared.runs if run.arm_id == "claude")
        summary = json.loads(
            (api_prepared.root / "execution" / "runs" / run_key / "run.json").read_text()
        )
        assert summary["billing"]["basis"] == "api-metered"
        assert summary["billing"]["reported_cost_usd"] == pytest.approx(0.025)
    finally:
        discard_prepared(api_prepared.root)


def test_claude_plugin_directory_is_a_pinned_workspace_local_session_addition(
    runner_experiment, fake_executables
):
    prepared, manifest, doc = runner_experiment
    doc["experiment"] = "plugin-directory-test"
    doc["arms"][0]["configuration"] = {
        "instructions": "inherit",
        "plugins": "directories",
        "plugin_directories": ["plugins/candidate"],
    }
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    plugin_prepared = prepare_experiment(
        load_experiment(manifest), prepared.root.parent.parent / "plugin-artifacts"
    )
    driver = FakeDriver()
    try:
        result = execute_experiment(
            plugin_prepared.root,
            allow_provider_calls=True,
            executable_overrides=fake_executables,
            driver=driver,
        )
        assert result.status == "completed"
        claude = next(
            call
            for call in _execution_calls(driver)
            if call["runner"] == "claude-code"
        )
        plugin_path = Path(claude["args"][claude["args"].index("--plugin-dir") + 1])
        assert plugin_path.is_dir()
        assert plugin_path.relative_to(Path(claude["cwd"])) == Path("plugins/candidate")
    finally:
        discard_prepared(plugin_prepared.root)


def test_external_write_allowlist_needs_a_second_gate_and_maps_to_native_sandboxes(
    runner_experiment, fake_executables, tmp_path
):
    prepared, manifest, doc = runner_experiment
    external = tmp_path / "explicit-external-target"
    external.mkdir()
    doc["experiment"] = "external-allowlist-test"
    doc["isolation"] = {
        "workspace": "copy",
        "session": "fresh-per-run",
        "external_writes": "allowlisted",
        "external_write_allowlist": [str(external / "{run_key}")],
    }
    doc["arms"] = doc["arms"][:2]
    manifest.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    external_prepared = prepare_experiment(
        load_experiment(manifest), prepared.root.parent.parent / "external-artifacts"
    )
    try:
        with pytest.raises(ExecutionRefused, match="allow-external-writes"):
            execute_experiment(
                external_prepared.root,
                allow_provider_calls=True,
                executable_overrides=fake_executables,
                driver=FakeDriver(),
            )
        driver = FakeDriver()
        result = execute_experiment(
            external_prepared.root,
            allow_provider_calls=True,
            allow_external_writes=True,
            executable_overrides=fake_executables,
            driver=driver,
        )
        assert result.status == "completed"
        calls = _execution_calls(driver)
        claude = next(call for call in calls if call["runner"] == "claude-code")
        settings = json.loads(claude["args"][claude["args"].index("--settings") + 1])
        claude_target = settings["sandbox"]["filesystem"]["allowWrite"][0]
        assert claude_target.startswith(str(external))
        codex = next(call for call in calls if call["runner"] == "codex-cli")
        codex_target = codex["args"][codex["args"].index("--add-dir") + 1]
        assert codex_target.startswith(str(external))
        assert claude_target != codex_target
        marker = json.loads(
            (external_prepared.root / "execution" / "execution.json").read_text()
        )
        assert marker["external_writes"]["templates"] == [str(external / "{run_key}")]
        assert len(marker["external_writes"]["resolved_targets"]) == 2
        assert marker["external_writes"]["changed"] == []
    finally:
        discard_prepared(external_prepared.root)


def test_cli_execute_without_confirmation_returns_preflight_exit(
    runner_experiment, capsys
):
    prepared, _, _ = runner_experiment
    assert main(["experiment", "execute", str(prepared.root)]) == EXIT_REFUSED_PREFLIGHT
    assert "allow-provider-calls" in capsys.readouterr().err
