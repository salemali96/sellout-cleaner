from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src import fixes
from src.ingest import scan_raw
from src.split import MappingError, apply_mapping, split_tables, stage
from tests.conftest import raw_row


def staged(make_csv, rules, mapping, rows) -> pl.DataFrame:
    lf, _ = scan_raw(make_csv(rows), rules.ingest)
    df = stage(apply_mapping(lf, mapping), rules).collect()
    df = fixes.convert_case_pack(df.lazy(), rules).collect()
    return fixes.collect_fix_log(df)[0]


def test_apply_mapping_renames_and_drops_unmapped(make_csv, rules, mapping) -> None:
    lf, _ = scan_raw(make_csv([raw_row()]), rules.ingest)
    out = apply_mapping(lf.with_columns(pl.lit("x").alias("Junk")), mapping).collect()
    assert "product_id" in out.columns and "Junk" not in out.columns and "SKU Code" not in out.columns


def test_apply_mapping_missing_column_raises(make_csv, rules, mapping) -> None:
    lf, _ = scan_raw(make_csv([raw_row()]), rules.ingest)
    with pytest.raises(MappingError):
        apply_mapping(lf.drop("Week"), mapping)


@pytest.mark.parametrize("raw", ["2025-W03", "2025W03", "2025-w3", "W03-2025", "202503", "2025-01-13",
                                 "15/01/2025", "19/01/2025", " 2025-W03 "])
def test_week_formats_normalize_to_iso_monday(make_csv, rules, mapping, raw: str) -> None:
    df = staged(make_csv, rules, mapping, [raw_row(week=raw)])
    assert df["week_start_date"][0] == dt.date(2025, 1, 13)
    assert (df["year"][0], df["week"][0]) == (2025, 3)


def test_week_one_belongs_to_previous_calendar_year(make_csv, rules, mapping) -> None:
    df = staged(make_csv, rules, mapping, [raw_row(week="2025-W01"), raw_row(week="31/12/2024")])
    assert df["week_start_date"].to_list() == [dt.date(2024, 12, 30)] * 2
    assert df["year"].to_list() == [2025, 2025]


@pytest.mark.parametrize("raw", ["2025-W54", "2025-W53", "garbage", "202599"])
def test_invalid_weeks_are_null(make_csv, rules, mapping, raw: str) -> None:
    df = staged(make_csv, rules, mapping, [raw_row(week=raw)])
    assert df["week_start_date"][0] is None


def test_placeholders_become_null(make_csv, rules, mapping) -> None:
    df = staged(make_csv, rules, mapping, [raw_row(brand="N/A", channel=" - ", units="null")])
    assert df["brand"][0] is None and df["channel"][0] is None and df["units"][0] is None
    assert df["units_raw"][0] == "null"


def test_case_pack_converted_to_units(make_csv, rules, mapping) -> None:
    df = staged(make_csv, rules, mapping, [raw_row(uom="CS", upc="12", units="3", value="360")])
    assert df["units"][0] == 36 and df["uom"][0] == "EA"


def test_split_builds_four_tables(make_csv, rules, mapping) -> None:
    rows = [
        raw_row(week="2025-W03"),
        raw_row(week="2025-W04", store_name="ميجا مارت المعادي", promo="PR1", promo_type="TPR",
                promo_start="2025-01-20", promo_end="26/01/2025"),
        raw_row(week="2025-W05", brand="N/A"),
        raw_row(sku="SKU2", week="2025-W03", brand="Sunvale"),
    ]
    tables = split_tables(staged(make_csv, rules, mapping, rows), mapping, rules)
    assert set(tables) == {"product", "market", "promo", "sellout"}
    prod = tables["product"]
    assert prod.height == 2 and prod.filter(pl.col("product_id") == "SKU1")["brand"][0] == "Nilegold"
    mkt = tables["market"]
    assert mkt.height == 1
    assert mkt["market_name"][0] == "MegaMart Maadi" and mkt["market_name_ar"][0] == "ميجا مارت المعادي"
    promo = tables["promo"]
    assert promo.to_dicts() == [{"promo_id": "PR1", "promo_type": "TPR", "promo_start_date": dt.date(2025, 1, 20),
                                 "promo_end_date": dt.date(2025, 1, 26)}]
    so = tables["sellout"]
    assert so.height == 4 and so["avg_price"][0] == 10.0
    assert {"product_id", "market_id", "week_start_date", "units", "value", "row_nr"} <= set(so.columns)


def test_majority_vote_picks_most_frequent(make_csv, rules, mapping) -> None:
    rows = [raw_row(week=f"2025-W0{i}") for i in range(1, 4)] + [raw_row(week="2025-W05", brand="Wrong")]
    prod = split_tables(staged(make_csv, rules, mapping, rows), mapping, rules)["product"]
    assert prod["brand"][0] == "Nilegold"


def test_promo_without_definition_is_not_a_dimension_row(make_csv, rules, mapping) -> None:
    rows = [raw_row(promo="PR-ORPHAN")]
    assert split_tables(staged(make_csv, rules, mapping, rows), mapping, rules)["promo"].height == 0
