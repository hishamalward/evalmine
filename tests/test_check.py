"""Execution checks (spec S6.6). Every check here runs real bash on throwaway
temp dirs: nothing contacts a provider, nothing needs a key."""

from __future__ import annotations

import os

from evalmine.check import CheckResult, CheckSpec, extract_code, run_check, summarize


def test_extract_code_prefers_the_first_fence():
    text = "Here you go:\n```bash\ngit log --oneline\n```\nand also\n```\nsecond\n```"
    assert extract_code(text) == "git log --oneline"


def test_extract_code_without_a_fence_is_the_whole_answer_stripped():
    assert extract_code("  jq '.[] | .title'  \n") == "jq '.[] | .title'"


def test_extract_code_handles_language_tags_and_crlf():
    crlf = "```python\r\ndef f():\r\n    return 1\r\n```"
    assert extract_code(crlf) == "def f():\r\n    return 1"
    assert extract_code("``` sh \nls\n```") == "ls"


def test_a_passing_check_records_exit_zero_and_the_output():
    spec = CheckSpec(run='grep -q hello "$ANSWER" && echo ran-ok')
    result = run_check(spec, "```bash\necho hello\n```")
    assert result.status == "pass"
    assert result.exit_code == 0
    assert result.output == "ran-ok"
    assert result.code == "echo hello"
    assert summarize(result) == "PASS (exit 0)"


def test_a_failing_check_records_the_exit_code_and_stderr():
    spec = CheckSpec(run="echo nope; echo err >&2; exit 3")
    result = run_check(spec, "anything")
    assert result.status == "fail"
    assert result.exit_code == 3
    assert result.output == "nope\n[stderr]\nerr"
    assert summarize(result) == "FAIL (exit 3)"


def test_a_setup_failure_is_an_error_not_a_verdict_on_the_answer():
    spec = CheckSpec(setup="echo fixture-broke >&2; exit 2", run="true")
    result = run_check(spec, "anything")
    assert result.status == "error"
    assert result.exit_code == 2
    assert result.output.startswith("setup failed:")
    assert "fixture-broke" in result.output


def test_a_timeout_is_a_fail_with_no_exit_code():
    spec = CheckSpec(run="echo started; sleep 5", timeout_s=1)
    result = run_check(spec, "anything")
    assert result.status == "fail"
    assert result.exit_code is None
    assert result.output.startswith("timed out after 1s")
    assert summarize(result) == "FAIL (no exit code)"


def test_checks_run_in_a_fresh_directory_seeded_by_setup():
    here = os.getcwd()
    spec = CheckSpec(
        setup="echo data > fixture.txt",
        run=f'test "$(cat fixture.txt)" = data && test "$(pwd)" != "{here}"',
    )
    assert run_check(spec, "x").status == "pass"


def test_the_temp_directory_is_removed_afterwards():
    spec = CheckSpec(run="pwd")
    result = run_check(spec, "x")
    assert result.status == "pass"
    assert not os.path.exists(result.output.strip())


def test_secrets_never_reach_the_check(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-not-for-you")
    monkeypatch.setenv("SOME_TOKEN", "t")
    monkeypatch.setenv("HARMLESS_FLAG", "1")
    spec = CheckSpec(
        run='test -z "$ANTHROPIC_API_KEY" && test -z "$SOME_TOKEN" && test "$HARMLESS_FLAG" = 1'
    )
    assert run_check(spec, "x").status == "pass"


def test_the_answer_reaches_the_check_as_a_file_and_as_text():
    spec = CheckSpec(run='test "$(cat "$ANSWER")" = "echo hi" && test "$ANSWER_TEXT" = "echo hi"')
    assert run_check(spec, "echo hi").status == "pass"


def test_long_output_keeps_the_tail():
    spec = CheckSpec(run="seq 1 5000")
    result = run_check(spec, "x")
    assert result.status == "pass"
    assert result.output.startswith("...")
    assert result.output.endswith("5000")


def test_not_applicable_summary():
    assert summarize(CheckResult(status="not_applicable")) == "not checked"
