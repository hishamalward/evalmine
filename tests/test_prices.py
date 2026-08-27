"""Price table lookup and cost arithmetic. Spec S6.3, success criterion S13.7."""

from __future__ import annotations

import pytest
from conftest import PRICES_DIR

from evalmine.experiment_report import _api_list_price_equivalent, _duration_text
from evalmine.metrics import call_cost
from evalmine.prices import (
    PriceRow,
    PriceTableError,
    UnknownModelError,
    load_price_table,
    newest_table_path,
)


def test_shipped_table_loads_and_declares_itself_verified():
    table = load_price_table(prices_dir=PRICES_DIR)
    assert table.pinned == "2026-08-27"
    assert table.currency == "USD"
    # Re-verified 2026-08-27 against each provider's own pricing page.
    assert table.verified is True
    assert "UNVERIFIED" not in table.describe()
    for model in (
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4-6",
        "openai/gpt-5.6-sol",
        "google/gemini-2.5-flash",
        "fake/a",
        "fake/b",
    ):
        assert table.get(model).model == model
    opus = table.get("anthropic/claude-opus-5")
    assert opus.cache_write_1h_per_mtok == 10.0
    sol = table.get("openai/gpt-5.6-sol")
    assert sol.long_context_threshold_tokens == 272_000
    assert sol.long_context_input_multiplier == 2.0


def test_newest_table_is_chosen_by_pinned_date(tmp_path):
    (tmp_path / "prices-2026-01-01.yaml").write_text(
        "pinned: 2026-01-01\nmodels:\n  - {model: fake/a, input_per_mtok: 9, output_per_mtok: 9}\n",
        encoding="utf-8",
    )
    (tmp_path / "prices-2026-08-23.yaml").write_text(
        "pinned: 2026-08-23\nmodels:\n  - {model: fake/a, input_per_mtok: 1, output_per_mtok: 2}\n",
        encoding="utf-8",
    )
    assert newest_table_path(tmp_path).name == "prices-2026-08-23.yaml"
    table = load_price_table(prices_dir=tmp_path)
    assert table.get("fake/a").input_per_mtok == 1.0


def test_unknown_model_raises_naming_the_string_and_the_file():
    table = load_price_table(prices_dir=PRICES_DIR)
    with pytest.raises(UnknownModelError) as exc:
        table.get("anthropic/claude-does-not-exist")
    message = str(exc.value)
    assert "anthropic/claude-does-not-exist" in message
    assert "prices-2026-08-27.yaml" in message


def test_resolve_all_raises_on_the_first_unknown_model():
    table = load_price_table(prices_dir=PRICES_DIR)
    assert set(table.resolve_all(["fake/a", "fake/b", "fake/a"])) == {"fake/a", "fake/b"}
    with pytest.raises(UnknownModelError):
        table.resolve_all(["fake/a", "openai/nope"])


def test_cost_is_hand_checked_including_cached_input():
    row = PriceRow(
        model="x/y", input_per_mtok=3.0, output_per_mtok=15.0, cached_input_per_mtok=0.30
    )
    # 1_000_000 in @ $3, 200_000 out @ $15, 500_000 cached @ $0.30
    cost = call_cost(row, 1_000_000, 200_000, 500_000)
    assert cost == pytest.approx(3.0 + 3.0 + 0.15)


def test_missing_usage_is_null_not_zero():
    row = PriceRow(model="x/y", input_per_mtok=3.0, output_per_mtok=15.0)
    assert call_cost(row, None, 100) is None
    assert call_cost(row, 100, None) is None
    assert call_cost(row, 0, 0) == 0.0


def test_subscription_claude_usage_has_an_api_list_price_equivalent():
    table = load_price_table(prices_dir=PRICES_DIR)
    run = {
        "runner": "claude-code",
        "requested_model": "claude-opus-5",
        "billing": {"basis": "subscription", "meter_equivalent_usd": 3.9666955},
        "turns": [
            {
                "usage": {
                    "input_tokens": 44,
                    "output_tokens": 23_777,
                    "cache_read_input_tokens": 2_471_685,
                    "cache_creation_input_tokens": 177_676,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 177_676,
                    },
                    "inference_geo": "not_available",
                }
            },
            {
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 5_824,
                    "cache_read_input_tokens": 177_676,
                    "cache_creation_input_tokens": 12_500,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 0,
                        "ephemeral_1h_input_tokens": 12_500,
                    },
                    "inference_geo": "not_available",
                }
            },
        ],
    }
    estimate = _api_list_price_equivalent(run, table)
    assert estimate["status"] == "estimated"
    assert estimate["usd"] == pytest.approx(3.9666955)
    assert estimate["runner_meter_equivalent_delta_usd"] == pytest.approx(0.0)
    assert estimate["components_usd"]["cache_write_1h"] == pytest.approx(1.90176)


def test_subscription_sol_usage_applies_long_context_pricing_per_turn():
    table = load_price_table(prices_dir=PRICES_DIR)
    run = {
        "runner": "codex-cli",
        "requested_model": "gpt-5.6-sol",
        "billing": {"basis": "subscription", "meter_equivalent_usd": None},
        "turns": [
            {
                "usage": {
                    "input_tokens": 2_261_469,
                    "cached_input_tokens": 2_074_368,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 16_077,
                    "reasoning_output_tokens": 7_852,
                }
            },
            {
                "usage": {
                    "input_tokens": 187_761,
                    "cached_input_tokens": 179_968,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 1_098,
                    "reasoning_output_tokens": 626,
                }
            },
        ],
    }
    estimate = _api_list_price_equivalent(run, table)
    assert estimate["status"] == "estimated"
    assert estimate["usd"] == pytest.approx(3.7637316)
    assert estimate["components_usd"]["cached_input"] == pytest.approx(1.7314816)
    assert any("1 runner turn" in item for item in estimate["assumptions"])
    assert any("cached and uncached" in item for item in estimate["assumptions"])


def test_report_durations_are_human_readable_without_losing_raw_milliseconds():
    assert _duration_text(459_464) == "7m 39s"
    assert _duration_text(385_763) == "6m 26s"
    assert _duration_text(181_805) == "3m 02s"
    assert _duration_text(725) == "725 ms"


def test_missing_table_is_an_error(tmp_path):
    with pytest.raises(PriceTableError):
        newest_table_path(tmp_path / "nothing-here")


def test_malformed_row_is_an_error(tmp_path):
    (tmp_path / "prices-2026-08-23.yaml").write_text(
        "models:\n  - {model: fake/a, input_per_mtok: not-a-number, output_per_mtok: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(PriceTableError):
        load_price_table(prices_dir=tmp_path)


def test_duplicate_row_is_an_error(tmp_path):
    (tmp_path / "prices-2026-08-23.yaml").write_text(
        "models:\n"
        "  - {model: fake/a, input_per_mtok: 1, output_per_mtok: 1}\n"
        "  - {model: fake/a, input_per_mtok: 2, output_per_mtok: 2}\n",
        encoding="utf-8",
    )
    with pytest.raises(PriceTableError):
        load_price_table(prices_dir=tmp_path)
