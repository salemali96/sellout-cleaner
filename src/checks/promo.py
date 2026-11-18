"""Promo dimension checks."""

from __future__ import annotations

import polars as pl

from src.checks.base import concat_issues, conflicting_values, issues, lazy
from src.config import Rules

Frame = pl.DataFrame | pl.LazyFrame
ATTRS = ["promo_type", "promo_start_date", "promo_end_date"]


def check_invalid_date_range(promo: Frame, rules: Rules) -> pl.DataFrame:
    """Promo dimension rows whose end date is before the start date."""
    bad = lazy(promo).filter(pl.col("promo_end_date") < pl.col("promo_start_date"))
    return issues(bad, rule_id="promo.invalid_date_range", table="promo", rules=rules, row_key=pl.col("promo_id"),
                  column="promo_start_date,promo_end_date",
                  value=pl.format("{} -> {}", pl.col("promo_start_date"), pl.col("promo_end_date")),
                  reason="promo ends before it starts (start/end probably swapped)")


def check_conflicting_attributes(df: Frame, rules: Rules) -> pl.DataFrame:
    """Same promo_id with different type/dates across rows (staged wide frame)."""
    return concat_issues([
        conflicting_values(df, "promo_id", a, rule_id="promo.conflicting_attribute", table="promo", rules=rules)
        for a in ATTRS
    ])
