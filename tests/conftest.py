from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from src.config import Rules, SchemaMapping, load_mapping, load_rules

HEADER = ["SKU Code", "Product Description", "Brand", "Category", "Pack Size", "UoM", "Units/Case",
          "Store Code", "Store Name", "Channel", "Region", "Promo ID", "Promo Type", "Promo Start",
          "Promo End", "Week", "Sales Qty", "Sales Value (EGP)"]


def raw_row(**kw: str) -> dict[str, str]:
    """A clean raw row (raw header names) with overrides by standard-ish short names."""
    base = {
        "SKU Code": "SKU1", "Product Description": "Milk 1L", "Brand": "Nilegold", "Category": "Dairy",
        "Pack Size": "1L", "UoM": "EA", "Units/Case": "12", "Store Code": "ST1", "Store Name": "MegaMart Maadi",
        "Channel": "Supermarket", "Region": "Cairo", "Promo ID": "", "Promo Type": "", "Promo Start": "",
        "Promo End": "", "Week": "2025-W03", "Sales Qty": "10", "Sales Value (EGP)": "100.00",
    }
    alias = {"sku": "SKU Code", "brand": "Brand", "uom": "UoM", "upc": "Units/Case", "store": "Store Code",
             "store_name": "Store Name", "channel": "Channel", "promo": "Promo ID", "promo_type": "Promo Type",
             "promo_start": "Promo Start", "promo_end": "Promo End", "week": "Week", "units": "Sales Qty",
             "value": "Sales Value (EGP)", "name": "Product Description", "region": "Region"}
    for k, v in kw.items():
        base[alias.get(k, k)] = v
    return base


def write_csv(path: Path, rows: list[dict[str, str]], delimiter: str = ",", encoding: str = "utf-8") -> Path:
    df = pl.DataFrame(rows, schema={h: pl.String for h in HEADER})
    path.write_bytes(df.write_csv(separator=delimiter).encode(encoding))
    return path


@pytest.fixture(scope="session")
def rules() -> Rules:
    return load_rules()


@pytest.fixture(scope="session")
def mapping() -> SchemaMapping:
    return load_mapping()


@pytest.fixture
def make_csv(tmp_path: Path):
    def _make(rows: list[dict[str, str]], name: str = "raw.csv", **kw: str) -> Path:
        return write_csv(tmp_path / name, rows, **kw)

    return _make


@pytest.fixture
def prep(make_csv, rules, mapping):
    """rows -> pipeline.Prepared (full ingest/stage/fix/split on a tiny CSV)."""
    from src import pipeline

    def _prep(rows: list[dict[str, str]]):
        return pipeline.prepare(make_csv(rows), rules, mapping)

    return _prep


def weeks(n: int, **kw: str) -> list[dict[str, str]]:
    """n consecutive clean weeks for one product x market, starting 2025-W01."""
    return [raw_row(week=f"2025-W{i + 1:02d}", **kw) for i in range(n)]
