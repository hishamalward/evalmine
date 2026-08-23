"""Suite loading and validation. Spec S5, success criterion S13.11."""

from __future__ import annotations

import pytest
from conftest import EXAMPLE_SUITE, SPEC

from evalmine.suite import CallParams, SuiteError, load_suite, render


def test_example_suite_loads():
    suite = load_suite(EXAMPLE_SUITE)
    assert suite.name == "everyday-eight"
    assert suite.version == 1
    assert len(suite.tasks) == 8
    assert sum(len(t.cases) for t in suite.tasks) == 20
    assert len([t for t in suite.tasks if t.schema]) == 3
    assert len(suite.labels) == 12
    assert suite.max_cost_usd == 1.50
    assert suite.judge.calibration.min_kappa == 0.40
    assert suite.judge.calibration.min_labels == 10
    # every prompt rendered: no placeholder survives
    for task in suite.tasks:
        for case in task.cases:
            assert "{{" not in case.prompt


def test_spec_and_example_suite_are_byte_identical():
    """Spec S5.6: the fenced block in the spec is the example file, verbatim."""
    spec_text = SPEC.read_text(encoding="utf-8")
    start = spec_text.index("<!-- BEGIN examples/everyday-eight.yaml -->")
    end = spec_text.index("<!-- END examples/everyday-eight.yaml -->")
    block = spec_text[start:end]
    block = block.split("```yaml\n", 1)[1].rsplit("```\n", 1)[0]
    assert block == EXAMPLE_SUITE.read_text(encoding="utf-8")


def test_unknown_top_level_key_is_an_error(minimal_suite, write_suite):
    minimal_suite["rubrik"] = "a typo that must not be silently ignored"
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "rubrik" in str(exc.value)


def test_unknown_task_key_is_an_error(minimal_suite, write_suite):
    minimal_suite["tasks"][0]["promptt"] = "typo"
    with pytest.raises(SuiteError):
        load_suite(write_suite(minimal_suite))


def test_unknown_version_is_an_error(minimal_suite, write_suite):
    minimal_suite["version"] = 2
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "version" in str(exc.value)


def test_missing_version_is_an_error(minimal_suite, write_suite):
    del minimal_suite["version"]
    with pytest.raises(SuiteError):
        load_suite(write_suite(minimal_suite))


def test_placeholder_with_no_var_names_task_case_and_variable(minimal_suite, write_suite):
    minimal_suite["tasks"][0]["prompt"] = "Say something about {{thing}} and {{other}}."
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    message = str(exc.value)
    assert "echo" in message and "one" in message and "other" in message


def test_placeholder_in_system_is_also_checked(minimal_suite, write_suite):
    minimal_suite["tasks"][0]["system"] = "You are reviewing {{absent}}."
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "absent" in str(exc.value)


def test_label_for_unknown_case_is_an_error(minimal_suite, write_suite):
    minimal_suite["labels"] = [
        {
            "task": "echo",
            "case": "three",
            "baseline": "fake/a",
            "candidate": "fake/b",
            "prefer": "tie",
        }
    ]
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "three" in str(exc.value)


def test_label_for_unknown_task_is_an_error(minimal_suite, write_suite):
    minimal_suite["labels"] = [
        {
            "task": "nope",
            "case": "one",
            "baseline": "fake/a",
            "candidate": "fake/b",
            "prefer": "tie",
        }
    ]
    with pytest.raises(SuiteError):
        load_suite(write_suite(minimal_suite))


def test_labels_and_labels_path_are_mutually_exclusive(minimal_suite, write_suite, tmp_path):
    minimal_suite["labels"] = []
    minimal_suite["labels_path"] = "labels.yaml"
    with pytest.raises(SuiteError):
        load_suite(write_suite(minimal_suite))


def test_labels_path_is_loaded(minimal_suite, write_suite, tmp_path):
    (tmp_path / "labels.yaml").write_text(
        "labels:\n"
        "  - {task: echo, case: one, baseline: fake/a, candidate: fake/b, prefer: candidate}\n",
        encoding="utf-8",
    )
    minimal_suite["labels_path"] = "labels.yaml"
    suite = load_suite(write_suite(minimal_suite))
    assert len(suite.labels) == 1
    assert suite.labels[0].prefer == "candidate"


def test_schema_and_schema_path_are_mutually_exclusive(minimal_suite, write_suite):
    minimal_suite["tasks"][0]["schema"] = {"type": "object"}
    minimal_suite["tasks"][0]["schema_path"] = "s.json"
    with pytest.raises(SuiteError):
        load_suite(write_suite(minimal_suite))


def test_invalid_task_schema_is_an_error(minimal_suite, write_suite):
    minimal_suite["tasks"][0]["schema"] = {"type": "not-a-type"}
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "Draft 2020-12" in str(exc.value)


def test_duplicate_task_id_is_an_error(minimal_suite, write_suite):
    minimal_suite["tasks"].append(dict(minimal_suite["tasks"][0]))
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "duplicate task id" in str(exc.value)


def test_duplicate_case_id_is_an_error(minimal_suite, write_suite):
    minimal_suite["tasks"][0]["cases"].append({"id": "one", "vars": {"thing": "again"}})
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "duplicate case id" in str(exc.value)


def test_a_key_shaped_string_refuses_to_load(minimal_suite, write_suite):
    # assembled at runtime so that no key-shaped literal exists in this file
    # for a secret scanner to trip over
    keyish = "sk-" + "ant-" + "api03-" + "A" * 20
    minimal_suite["tasks"][0]["prompt"] = f"Use {keyish} {{{{thing}}}}"
    with pytest.raises(SuiteError) as exc:
        load_suite(write_suite(minimal_suite))
    assert "Anthropic API key" in str(exc.value)


def test_task_params_override_defaults(minimal_suite, write_suite):
    minimal_suite["defaults"] = {"temperature": 0.2, "max_tokens": 100, "timeout_s": 5}
    minimal_suite["tasks"][0]["max_tokens"] = 900
    suite = load_suite(write_suite(minimal_suite))
    params = suite.tasks[0].params
    assert params.temperature == 0.2
    assert params.max_tokens == 900
    assert params.timeout_s == 5


def test_defaults_are_the_spec_defaults(minimal_suite, write_suite):
    suite = load_suite(write_suite(minimal_suite))
    assert suite.tasks[0].params == CallParams()
    assert suite.tasks[0].params.max_tokens == 700
    assert suite.tasks[0].params.timeout_s == 60
    assert suite.judge.max_tokens == 400


def test_suite_hash_follows_the_bytes(minimal_suite, write_suite):
    first = load_suite(write_suite(minimal_suite))
    minimal_suite["description"] = "now with a description"
    second = load_suite(write_suite(minimal_suite, name="suite2.yaml"))
    assert first.hash != second.hash


def test_task_hash_is_per_task(minimal_suite, write_suite):
    first = load_suite(write_suite(minimal_suite))
    minimal_suite["tasks"][0]["prompt"] = "Say two things about {{thing}}."
    second = load_suite(write_suite(minimal_suite, name="suite2.yaml"))
    assert first.tasks[0].hash != second.tasks[0].hash


def test_render_allows_whitespace_in_braces_and_substitutes_once():
    assert render("a {{ x }} b", {"x": "{{y}}"}, "test") == "a {{y}} b"


def test_bad_yaml_is_a_suite_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("suite: [unclosed\n", encoding="utf-8")
    with pytest.raises(SuiteError):
        load_suite(path)
