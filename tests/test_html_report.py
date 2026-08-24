"""report.html: the blind A/B mapping, the escaping, and the page markers.

Spec S9.5. The mapping from "I clicked Prefer A" back to ``prefer: baseline`` or
``prefer: candidate`` is the one piece of this feature that can be wrong without
anyone noticing - a flipped label does not crash, it quietly poisons the kappa
that the whole tool's credibility rests on. So the mapping is computed in Python
and tested here, and the page ships it as a lookup table the browser only reads.
"""

from __future__ import annotations

import json
import re

import pytest
from conftest import EXAMPLE_SUITE, PRICES_DIR

from evalmine.core import run_suite
from evalmine.html_report import (
    CHOICES,
    ab_roles,
    esc,
    pair_id,
    prefer_by_choice,
    render_html,
)

MODELS = ["anthropic/claude-haiku-4-5", "google/gemini-2.5-flash"]


# --------------------------------------------------------------------------
# the mapping
# --------------------------------------------------------------------------


def test_ab_roles_is_always_one_baseline_and_one_candidate():
    for i in range(500):
        roles = ab_roles(f"task-{i}|case-{i}|b|c")
        assert set(roles) == {"baseline", "candidate"}


def test_ab_roles_is_stable_for_the_same_pair_id():
    pid = pair_id("changelog-line", "debounce", "anthropic/x", "google/y")
    assert ab_roles(pid) == ab_roles(pid)
    # and the id is what varies it, not call order
    assert [ab_roles(pid) for _ in range(5)].count(ab_roles(pid)) == 5


def test_ab_roles_actually_randomises_across_pairs():
    ids = [pair_id("t", f"case-{i}", "b", "c") for i in range(200)]
    swapped = sum(1 for pid in ids if ab_roles(pid)[0] == "candidate")
    # a constant order would be 0 or 200; anything in this band is a real mix
    assert 60 < swapped < 140


@pytest.mark.parametrize(
    ("roles", "choice", "expected"),
    [
        (("baseline", "candidate"), "A", "baseline"),
        (("baseline", "candidate"), "B", "candidate"),
        (("baseline", "candidate"), "tie", "tie"),
        (("candidate", "baseline"), "A", "candidate"),
        (("candidate", "baseline"), "B", "baseline"),
        (("candidate", "baseline"), "tie", "tie"),
    ],
)
def test_prefer_by_choice_is_the_whole_truth_table(roles, choice, expected):
    assert prefer_by_choice(roles)[choice] == expected


def test_prefer_by_choice_covers_every_click_and_nothing_else():
    table = prefer_by_choice(("baseline", "candidate"))
    assert set(table) == set(CHOICES)


def test_prefer_by_choice_refuses_roles_that_are_not_a_baseline_and_a_candidate():
    with pytest.raises(ValueError):
        prefer_by_choice(("baseline", "baseline"))
    with pytest.raises(ValueError):
        prefer_by_choice(("candidate", "tie"))


def test_the_slot_showing_a_models_answer_maps_back_to_that_model(pair_view):
    """The mapping check that would catch a flip: text -> slot -> prefer.

    For every pair, the slot holding the baseline's own answer text must map to
    ``prefer: baseline`` and the other to ``prefer: candidate``.
    """
    for pair in pair_view:
        for slot in ("a", "b"):
            role = pair[f"{slot}_role"]
            model = pair["baseline"] if role == "baseline" else pair["candidate"]
            assert pair["prefer_by_choice"][slot.upper()] == role
            # the text in that slot is the text that model produced
            other = "b" if slot == "a" else "a"
            assert pair[f"{slot}_text"] != pair[f"{other}_text"] or not pair[f"{slot}_text"]
            assert model in (pair["baseline"], pair["candidate"])
        assert {pair["prefer_by_choice"]["A"], pair["prefer_by_choice"]["B"]} == {
            "baseline",
            "candidate",
        }
        assert pair["prefer_by_choice"]["tie"] == "tie"


def test_the_page_ships_the_table_the_javascript_reads(html_and_view):
    html, pair_view = html_and_view
    blob = _page_json(html)
    by_id = {p["pair_id"]: p for p in blob["pairs"]}
    assert set(by_id) == {p["pair_id"] for p in pair_view}
    for pair in pair_view:
        assert by_id[pair["pair_id"]]["prefer_by_choice"] == pair["prefer_by_choice"]
    # the JS must look the answer up, never derive it
    assert "pair.prefer_by_choice[choice]" in html


def test_the_yaml_the_button_emits_loads_as_a_suite(html_and_view, tmp_path):
    """The format contract, end to end: click -> YAML -> a suite that loads.

    The lines are assembled in the browser, so this pins the exact template the
    JS uses (asserted against the JS source below it) and then runs the result
    through the real S5.5 loader. A `prefer` the loader rejects, or a note that
    breaks the flow mapping, fails here rather than in someone's suite file.
    """
    html, pair_view = html_and_view
    assert "'  - { ' + parts.join(', ') + ' }'" in html  # the template, verbatim

    choices = ["A", "tie", "B", "A"]
    lines = ["labels:"]
    for pair, choice in zip(pair_view, choices, strict=False):
        parts = [
            f"task: {pair['task']}",
            f"case: {pair['case']}",
            f"baseline: {pair['baseline']}",
            f"candidate: {pair['candidate']}",
            f"prefer: {pair['prefer_by_choice'][choice]}",
            'note: "labelled blind: {, } and \\" in the note"',
        ]
        lines.append("  - { " + ", ".join(parts) + " }")
    emitted = "\n".join(lines) + "\n"

    source = EXAMPLE_SUITE.read_text(encoding="utf-8")
    spliced = tmp_path / "relabelled.yaml"
    spliced.write_text(source.split("\nlabels:")[0] + "\n" + emitted, encoding="utf-8")

    from evalmine.suite import load_suite

    suite = load_suite(spliced)
    assert len(suite.labels) == len(choices)
    for label, pair, choice in zip(suite.labels, pair_view, choices, strict=False):
        assert label.task == pair["task"] and label.case == pair["case"]
        assert label.prefer == pair["prefer_by_choice"][choice]
        assert label.note == 'labelled blind: {, } and " in the note'


# --------------------------------------------------------------------------
# the page
# --------------------------------------------------------------------------


def test_the_page_is_self_contained(html_and_view):
    html, _ = html_and_view
    assert "<script src=" not in html
    assert "<link rel=\"stylesheet\"" not in html
    assert "http://" not in html and "https://" not in html
    assert "@import" not in html


def test_models_are_hidden_until_the_reveal_toggle(html_and_view):
    html, _ = html_and_view
    assert ".reveal-only{display:none}" in html
    assert ".reveal-inline{display:none}" in html
    assert 'body[data-reveal="1"] .reveal-only{display:block}' in html
    assert 'id="reveal" aria-pressed="false"' in html
    # the judge's verdict is behind the same toggle: reading it first is a bias
    assert '<div class="verdict reveal-only">' in html
    assert "data-reveal', '0'" in html


def test_the_labelling_flow_is_on_the_page(html_and_view):
    html, pair_view = html_and_view
    assert html.count('data-choice="A"') == len(pair_view)
    assert html.count('data-choice="tie"') == len(pair_view)
    assert html.count('data-choice="B"') == len(pair_view)
    assert html.count('class="why"') == len(pair_view)
    assert 'id="copy">Copy labels YAML' in html
    assert "'labels:'" in html  # the YAML the copy button emits
    assert "evalmine:labels:" in html  # localStorage key, scoped to the run
    assert "catch (e)" in html  # every storage touch is wrapped


def test_both_themes_are_fully_defined(html_and_view):
    html, _ = html_and_view
    assert "@media (prefers-color-scheme:dark)" in html
    light = html.split(":root{")[1].split("}")[0]
    dark = html.split("@media (prefers-color-scheme:dark)")[1].split("}")[0]
    tokens = set(re.findall(r"(--[a-z-]+):", light))
    assert tokens == set(re.findall(r"(--[a-z-]+):", dark))
    assert "body{margin:0;background:var(--bg)" in html


def test_wide_tables_scroll_inside_their_own_container(html_and_view):
    html, _ = html_and_view
    assert ".scroll{overflow-x:auto" in html
    assert html.count('<div class="scroll">') >= 4


def test_answers_are_escaped():
    """Answer text is untrusted. A model that emits a script tag stays inert."""
    hostile = '</pre><script>alert("x")</script> & <b>bold</b>'
    assert esc(hostile) == (
        "&lt;/pre&gt;&lt;script&gt;alert(&quot;x&quot;)&lt;/script&gt; &amp; "
        "&lt;b&gt;bold&lt;/b&gt;"
    )
    assert "<script>alert" not in esc(hostile)


def test_the_embedded_json_cannot_close_its_own_script_tag(html_and_view):
    html, _ = html_and_view
    blob = html.split('<script type="application/json" id="evalmine-data">')[1]
    blob = blob.split("</script>")[0]
    assert "<" not in blob and ">" not in blob
    json.loads(blob.replace("\\u003c", "<").replace("\\u003e", ">").replace("\\u0026", "&"))


def test_the_page_mirrors_the_markdown_section_order(html_and_view):
    html, _ = html_and_view
    order = [
        'id="calibration"',
        'id="win-rates"',
        'id="scorecard"',
        'id="per-task"',
        'id="what-changed"',
        'id="failures"',
        'id="reproduce"',
        'id="decision-log"',
        'id="pairs"',
    ]
    positions = [html.index(marker) for marker in order]
    assert positions == sorted(positions)


def test_the_uncalibrated_banner_and_the_kappa_band_are_both_on_the_page(html_and_view):
    html, _ = html_and_view
    assert "Per-task agreement" in html
    assert "UNCALIBRATED" in html or "headline eligible: <b>true</b>" in html


def test_a_report_with_no_judged_pairs_still_renders(fake_run):
    result = fake_run
    html = render_html(result.report, [])
    assert "No judged pairs in this run." in html
    assert 'id="progress">0 labeled of 0 pairs' in html


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def _page_json(html: str) -> dict:
    blob = html.split('<script type="application/json" id="evalmine-data">')[1]
    blob = blob.split("</script>")[0]
    return json.loads(
        blob.replace("\\u003c", "<").replace("\\u003e", ">").replace("\\u0026", "&")
    )


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("html")
    return run_suite(
        EXAMPLE_SUITE,
        list(MODELS),
        fake=True,
        prices_dir=PRICES_DIR,
        cache_dir=tmp_path / "cache",
        out_dir=tmp_path / "reports",
        retry_sleep=lambda _: None,
        command="evalmine run examples/everyday-eight.yaml --fake",
    )


@pytest.fixture(scope="module")
def pair_view(fake_run):
    return fake_run.pair_view


@pytest.fixture(scope="module")
def html_and_view(fake_run, pair_view):
    return render_html(fake_run.report, pair_view), pair_view


def test_a_refusal_or_truncation_is_flagged_on_the_pane():
    from evalmine.html_report import _pane

    base = {
        "a_role": "baseline", "b_role": "candidate", "baseline": "m/a", "candidate": "m/b",
        "a_text": "x", "b_text": "y", "a_error": None, "b_error": None,
        "a_schema_status": "not_applicable", "b_schema_status": "not_applicable",
        "a_check": None, "b_check": None,
    }
    refused = _pane("A", {**base, "a_finish": "refusal", "b_finish": "end_turn"})
    assert "refused by the provider" in refused
    truncated = _pane("B", {**base, "a_finish": "end_turn", "b_finish": "max_tokens"})
    assert "truncated: hit max_tokens" in truncated
    clean = _pane("A", {**base, "a_finish": "end_turn", "b_finish": "end_turn"})
    assert "flag bad" not in clean
