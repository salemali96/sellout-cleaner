"""Wide raw file -> product, market, promo dimensions + sellout fact.

Dimension attributes are resolved by majority vote (most frequent non-null value per key, ties
broken alphabetically). Conflicts are not resolved silently: the dimension checks report them.
"""

from __future__ import annotations

import polars as pl

from src import fixes
from src.config import Rules, SchemaMapping
from src.ingest import ROW_NR

PROMO_DATE_COLUMNS = ["promo_start_date", "promo_end_date"]
SELLOUT_COLUMNS = ["product_id", "market_id", "week_start_date", "year", "week", "units", "value",
                   "avg_price", "promo_id", "uom", ROW_NR]


class MappingError(ValueError):
    pass


def apply_mapping(lf: pl.LazyFrame, mapping: SchemaMapping) -> pl.LazyFrame:
    """Rename raw columns to standard names. Unmapped raw columns are dropped; missing mapped ones raise."""
    present = set(lf.collect_schema().names())
    missing = [raw for raw in mapping.columns if raw not in present]
    if missing:
        raise MappingError(f"Mapped raw columns not found in file: {missing}")
    return lf.select([pl.col(ROW_NR)] + [pl.col(raw).alias(std) for raw, std in mapping.columns.items()])


def stage(lf: pl.LazyFrame, rules: Rules) -> pl.LazyFrame:
    """Non-semantic safe fixes, all lazy: trim, null placeholders, week, numbers, promo dates."""
    lf = fixes.keep_raw(lf, ["units", "value"])
    lf = fixes.trim_whitespace(lf)
    lf = fixes.normalize_nulls(lf, rules)
    lf = fixes.normalize_week(lf, rules)
    lf = fixes.parse_numbers(lf, ["units", "value"])
    present = set(lf.collect_schema().names())
    return fixes.parse_dates(lf, [c for c in PROMO_DATE_COLUMNS if c in present], rules)


def majority(df: pl.DataFrame, key: str, attrs: list[str]) -> pl.DataFrame:
    """One row per key; each attribute = its most frequent non-null value (ties -> smallest)."""
    out = df.select(key).filter(pl.col(key).is_not_null()).unique().sort(key)
    for a in attrs:
        best = (
            df.filter(pl.col(key).is_not_null() & pl.col(a).is_not_null())
            .group_by([key, a]).len()
            .sort([key, "len", a], descending=[False, True, False])
            .unique(key, keep="first", maintain_order=True)
            .drop("len")
        )
        out = out.join(best, on=key, how="left")
    return out


def with_market_scripts(df: pl.DataFrame, rules: Rules) -> pl.DataFrame:
    """Split the mixed-language market_name into market_name (Latin) and market_name_ar (Arabic)."""
    is_ar = pl.col("market_name").str.contains(rules.market.arabic_regex).fill_null(False)
    return df.with_columns(
        pl.when(is_ar).then(pl.col("market_name")).alias("market_name_ar"),
        pl.when(~is_ar).then(pl.col("market_name")).alias("market_name"),
    )


def build_product(df: pl.DataFrame, mapping: SchemaMapping) -> pl.DataFrame:
    spec = mapping.tables["product"]
    out = majority(df, spec.key[0], spec.attributes)
    if "units_per_case" in out.columns:
        out = out.with_columns(pl.col("units_per_case").cast(pl.Int64, strict=False))
    return out


def build_market(df: pl.DataFrame, mapping: SchemaMapping, rules: Rules) -> pl.DataFrame:
    spec = mapping.tables["market"]
    attrs = spec.attributes + (["market_name_ar"] if "market_name" in spec.attributes else [])
    return majority(with_market_scripts(df, rules), spec.key[0], attrs)


def build_promo(df: pl.DataFrame, mapping: SchemaMapping) -> pl.DataFrame:
    """Promo events that are actually defined (at least one attribute present)."""
    spec = mapping.tables["promo"]
    out = majority(df, spec.key[0], spec.attributes)
    return out.filter(pl.any_horizontal([pl.col(a).is_not_null() for a in spec.attributes]))


def build_sellout(df: pl.DataFrame) -> pl.DataFrame:
    avg = pl.when(pl.col("units") != 0).then(pl.col("value") / pl.col("units")).alias("avg_price")
    return df.with_columns(avg).select([c for c in SELLOUT_COLUMNS if c in df.columns or c == "avg_price"])


def split_tables(df: pl.DataFrame, mapping: SchemaMapping, rules: Rules) -> dict[str, pl.DataFrame]:
    """Staged (and fixed) wide frame -> the 4 output tables (before quarantine)."""
    return {
        "product": build_product(df, mapping),
        "market": build_market(df, mapping, rules),
        "promo": build_promo(df, mapping),
        "sellout": build_sellout(df),
    }
