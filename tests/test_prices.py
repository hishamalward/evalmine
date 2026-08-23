"""Price table lookup and cost arithmetic. Spec S6.3, success criterion S13.7."""

from __future__ import annotations

import pytest
from conftest import PRICES_DIR

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
    assert table.pinned == "2026-08-23"
    assert table.currency == "USD"
    # Re-verified 2026-08-23 against each provider's own pricing page.
    assert table.verified is True
    assert "UNVERIFIED" not in table.describe()
    for model in ("anthropic/claude-sonnet-4-6", "google/gemini-2.5-flash", "fake/a", "fake/b"):
        assert table.get(model).model == model


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
    assert "prices-2026-08-23.yaml" in message


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
