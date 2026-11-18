"""Market dimension checks. Arabic and Latin names are separate attributes, so an Arabic alias is not a conflict."""

from __future__ import annotations

import polars as pl

from src.checks.base import concat_issues, conflicting_values
from src.config import Rules
from src.split import with_market_scripts


def check_conflicting_names(df: pl.DataFrame, rules: Rules) -> pl.DataFrame:
    """Markets with more than one distinct name in the same script (case/whitespace-insensitive)."""
    scripts = with_market_scripts(df.select("market_id", "market_name"), rules)
    return concat_issues([
        conflicting_values(scripts, "market_id", col, rule_id="market.conflicting_name", table="market", rules=rules)
        for col in ["market_name", "market_name_ar"]
    ])
