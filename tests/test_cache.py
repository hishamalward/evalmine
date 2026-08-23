"""The content-hash cache key and the store. Spec S6.5, success criterion S13.9."""

from __future__ import annotations

import copy
import json
import os

from evalmine.cache import Cache, answer_payload, cache_key, judge_payload
from evalmine.suite import canonical_bytes

BASE = dict(
    provider="fake",
    model="fake/a",
    system="be terse",
    prompt="Say something about sqlite.",
    params={"temperature": 0.0, "max_tokens": 700, "top_p": None, "stop": None, "seed": None},
    schema={"type": "object"},
    schema_mode="prompted",
    adapter_version=1,
    repeat=0,
)


def key_with(**overrides) -> str:
    payload = answer_payload(**{**BASE, **overrides})
    return cache_key(payload)


def test_identical_inputs_give_an_identical_key():
    assert key_with() == key_with()
    assert len(key_with()) == 64


def test_every_part_of_the_payload_changes_the_key():
    baseline = key_with()
    assert key_with(prompt="Say something else.") != baseline
    assert key_with(system=None) != baseline
    assert key_with(model="fake/b") != baseline
    assert key_with(provider="anthropic") != baseline
    assert key_with(schema=None) != baseline
    assert key_with(schema_mode="native") != baseline
    assert key_with(adapter_version=2) != baseline
    assert key_with(repeat=1) != baseline
    for param, value in [
        ("temperature", 0.7),
        ("max_tokens", 100),
        ("top_p", 0.9),
        ("stop", ["\n"]),
        ("seed", 7),
    ]:
        params = dict(BASE["params"])
        params[param] = value
        assert key_with(params=params) != baseline, param


def test_the_kind_separates_answers_from_judge_passes():
    answer = cache_key(answer_payload(**BASE))
    judge = cache_key(judge_payload(**BASE))
    assert answer != judge


def test_a_swapped_judge_pass_gets_its_own_key_by_its_prompt_alone():
    first = cache_key(judge_payload(**{**BASE, "prompt": "A then B"}))
    second = cache_key(judge_payload(**{**BASE, "prompt": "B then A"}))
    assert first != second


def test_the_key_ignores_the_date_run_id_suite_and_environment(monkeypatch):
    before = key_with()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-" + "ant-" + "not-a-real-key")
    monkeypatch.setenv("TZ", "Pacific/Auckland")
    after = key_with()
    assert before == after
    # nothing run-scoped is even representable in the payload
    payload = answer_payload(**BASE)
    assert set(payload) == {
        "v",
        "kind",
        "provider",
        "model",
        "system",
        "prompt",
        "params",
        "schema",
        "schema_mode",
        "adapter_version",
        "repeat",
    }


def test_canonical_encoding_is_stable_and_sorted():
    encoded = canonical_bytes({"b": 1, "a": {"d": 2, "c": [3, "é"]}})
    assert encoded == '{"a":{"c":[3,"é"],"d":2},"b":1}'.encode()
    # key ordering in the input does not matter
    assert cache_key(answer_payload(**BASE)) == cache_key(
        dict(reversed(list(answer_payload(**BASE).items())))
    )


def test_round_trip_through_the_store(tmp_path):
    cache = Cache(tmp_path)
    key = key_with()
    assert cache.get("fake", key) is None
    cache.put("fake", key, {"text": "hello", "input_tokens": 3, "output_tokens": 1})
    entry = cache.get("fake", key)
    assert entry["text"] == "hello"
    assert entry["key"] == key
    assert entry["created_at"].endswith("Z")
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1
    assert cache.stats.writes == 1


def test_the_path_shards_by_provider_and_key_prefix(tmp_path):
    cache = Cache(tmp_path)
    key = key_with()
    path = cache.path_for("fake", key)
    assert path.parent.name == key[:2]
    assert path.parent.parent.name == "fake"
    assert path.name == f"{key}.json"


def test_a_corrupt_entry_is_a_miss_and_is_overwritten(tmp_path):
    cache = Cache(tmp_path)
    key = key_with()
    path = cache.path_for("fake", key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("fake", key) is None
    assert cache.stats.corrupt == 1
    cache.put("fake", key, {"text": "recovered"})
    assert cache.get("fake", key)["text"] == "recovered"


def test_no_cache_skips_the_read_but_still_writes(tmp_path):
    writer = Cache(tmp_path)
    key = key_with()
    writer.put("fake", key, {"text": "cached"})

    no_read = Cache(tmp_path, read=False)
    assert no_read.get("fake", key) is None
    no_read.put("fake", key, {"text": "fresh"})
    assert Cache(tmp_path).get("fake", key)["text"] == "fresh"


def test_put_leaves_no_temporary_files_behind(tmp_path):
    cache = Cache(tmp_path)
    key = key_with()
    path = cache.put("fake", key, {"text": "x"})
    assert not [p for p in os.listdir(path.parent) if p.endswith(".tmp")]
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == "x"


def test_payload_is_not_mutated_by_key_computation():
    payload = answer_payload(**BASE)
    snapshot = copy.deepcopy(payload)
    cache_key(payload)
    assert payload == snapshot
