"""Product dimension checks (run on the staged wide frame, before majority vote)."""

from __future__ import annotations

import polars as pl

from src.checks.base import concat_issues, conflicting_values, issues, lazy
from src.config import Rules

Frame = pl.DataFrame | pl.LazyFrame
ATTRS = ["product_name", "brand", "category", "pack_size", "units_per_case"]


def check_conflicting_attributes(df: Frame, rules: Rules, attrs: list[str] | None = None) -> pl.DataFrame:
    names = set(lazy(df).collect_schema().names())
    return concat_issues([
        conflicting_values(df, "product_id", a, rule_id="product.conflicting_attribute", table="product", rules=rules)
        for a in (attrs or ATTRS) if a in names
    ])


def check_invalid_units_per_case(df: Frame, rules: Rules) -> pl.DataFrame:
    """Products whose units_per_case is missing, non-numeric or <= 0 (case-pack rows cannot be converted)."""
    upc = pl.col("units_per_case").cast(pl.Float64, strict=False)
    bad = (
        lazy(df)
        .filter(pl.col("product_id").is_not_null() & (upc.is_null() | (upc <= 0)))
        .group_by("product_id")
        .agg(pl.col("units_per_case").cast(pl.String).unique().sort().str.join(",").alias("_v"), pl.len().alias("_n"))
        .sort("product_id")
    )
    return issues(bad, rule_id="product.invalid_units_per_case", table="product", rules=rules,
                  row_key=pl.col("product_id"), column="units_per_case", value=pl.col("_v"),
                  reason=pl.format("units_per_case missing or <= 0 on {} rows", pl.col("_n")))
