"""The pinned price table and its loader. Spec: docs/spec.md S6.3.

Two rules the rest of the code depends on:

* an unknown model string is a hard failure, raised during planning and before
  any adapter is constructed - there is no fallback price and no ``$0.00``;
* a table that has not been verified against the providers' pricing pages says
  so, and every report generated from it repeats that.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_TABLE_GLOB = "prices-*.yaml"
_DATE_RE = re.compile(r"prices-(\d{4}-\d{2}-\d{2})\.ya?ml$")

#: Shipped inside the wheel by pyproject's force-include (as ``price_tables``,
#: so the data directory cannot shadow this module), and present at the repo
#: root when running from a checkout.
_PACKAGED_PRICES = Path(__file__).with_name("price_tables")
_REPO_PRICES = Path(__file__).resolve().parents[2] / "prices"


class UnknownModelError(Exception):
    """A model string with no row in the price table. Always exit 1."""


class PriceTableError(Exception):
    """The table itself is missing or malformed. Always exit 1."""


@dataclass(frozen=True)
class PriceRow:
    model: str
    input_per_mtok: float
    output_per_mtok: float
    cached_input_per_mtok: float = 0.0
    cache_write_5m_per_mtok: float | None = None
    cache_write_1h_per_mtok: float | None = None
    long_context_threshold_tokens: int | None = None
    long_context_input_multiplier: float = 1.0
    long_context_output_multiplier: float = 1.0
    source: str | None = None
    read_on: str | None = None


@dataclass(frozen=True)
class PriceTable:
    path: Path
    pinned: str
    currency: str
    verified: bool
    notes: str | None
    rows: dict[str, PriceRow]

    def get(self, model: str) -> PriceRow:
        try:
            return self.rows[model]
        except KeyError:
            raise UnknownModelError(
                f"model {model!r} has no row in the price table {self.path}. "
                "evalmine has no fallback price: add a row (with its source URL and the "
                "date you read it) or correct the model string. A silent $0.00 is how a "
                "cost comparison becomes a lie."
            ) from None

    def resolve_all(self, models: list[str]) -> dict[str, PriceRow]:
        """Resolve every model string, or raise on the first that does not."""
        return {model: self.get(model) for model in dict.fromkeys(models)}

    @property
    def filename(self) -> str:
        return self.path.name

    def describe(self) -> str:
        suffix = "" if self.verified else " (UNVERIFIED - figures are placeholders)"
        return f"{self.filename}{suffix}"


def _search_dirs(prices_dir: str | Path | None) -> list[Path]:
    if prices_dir is not None:
        return [Path(prices_dir)]
    return [Path.cwd() / "prices", _PACKAGED_PRICES, _REPO_PRICES]


def newest_table_path(prices_dir: str | Path | None = None) -> Path:
    """The newest ``prices-YYYY-MM-DD.yaml`` in the first directory that has one."""
    tried: list[Path] = []
    for directory in _search_dirs(prices_dir):
        tried.append(directory)
        if not directory.is_dir():
            continue
        candidates = sorted(p for p in directory.glob(_TABLE_GLOB) if _DATE_RE.search(p.name))
        if candidates:
            return candidates[-1]
    where = ", ".join(str(p) for p in tried)
    raise PriceTableError(
        f"no price table found (looked for {_TABLE_GLOB} in: {where}). "
        "Pass --prices <file>, or run from a directory that has a prices/ directory."
    )


def _row_from(raw: dict[str, Any], path: Path) -> PriceRow:
    try:
        model = raw["model"]
        return PriceRow(
            model=model,
            input_per_mtok=float(raw["input_per_mtok"]),
            output_per_mtok=float(raw["output_per_mtok"]),
            cached_input_per_mtok=float(raw.get("cached_input_per_mtok") or 0.0),
            cache_write_5m_per_mtok=(
                float(raw["cache_write_5m_per_mtok"])
                if raw.get("cache_write_5m_per_mtok") is not None
                else None
            ),
            cache_write_1h_per_mtok=(
                float(raw["cache_write_1h_per_mtok"])
                if raw.get("cache_write_1h_per_mtok") is not None
                else None
            ),
            long_context_threshold_tokens=(
                int(raw["long_context_threshold_tokens"])
                if raw.get("long_context_threshold_tokens") is not None
                else None
            ),
            long_context_input_multiplier=float(
                raw.get("long_context_input_multiplier") or 1.0
            ),
            long_context_output_multiplier=float(
                raw.get("long_context_output_multiplier") or 1.0
            ),
            source=raw.get("source"),
            read_on=(str(raw["read_on"]) if raw.get("read_on") is not None else None),
        )
    except KeyError as exc:
        raise PriceTableError(f"{path}: a price row is missing {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise PriceTableError(f"{path}: a price row has a non-numeric price ({exc})") from exc


def load_price_table(
    path: str | Path | None = None, prices_dir: str | Path | None = None
) -> PriceTable:
    table_path = Path(path) if path is not None else newest_table_path(prices_dir)
    if not table_path.is_file():
        raise PriceTableError(f"price table {table_path} does not exist")
    try:
        doc = yaml.safe_load(table_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise PriceTableError(f"{table_path}: cannot be read as YAML ({exc})") from exc
    if not isinstance(doc, dict) or not isinstance(doc.get("models"), list):
        raise PriceTableError(f"{table_path}: expected a mapping with a 'models' list")

    rows: dict[str, PriceRow] = {}
    for raw in doc["models"]:
        if not isinstance(raw, dict):
            raise PriceTableError(f"{table_path}: every entry under 'models' must be a mapping")
        row = _row_from(raw, table_path)
        if row.model in rows:
            raise PriceTableError(f"{table_path}: duplicate price row for {row.model!r}")
        rows[row.model] = row

    pinned = doc.get("pinned")
    if pinned is None:
        match = _DATE_RE.search(table_path.name)
        pinned = match.group(1) if match else "unknown"

    return PriceTable(
        path=table_path,
        pinned=str(pinned),
        currency=str(doc.get("currency", "USD")),
        verified=bool(doc.get("verified", False)),
        notes=doc.get("notes"),
        rows=rows,
    )
