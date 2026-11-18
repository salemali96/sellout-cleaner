"""Generic column checks: required values and null placeholders."""

from __future__ import annotations

import polars as pl

from src.checks.base import concat_issues, is_placeholder, issues, lazy
from src.config import Rules

Frame = pl.DataFrame | pl.LazyFrame


def check_missing_required(df: Frame, rules: Rules) -> pl.DataFrame:
    """Required columns that are null after normalization (staged frame). Measures use their raw text."""
    lf = lazy(df)
    names = set(lf.collect_schema().names())
    out = []
    for c in rules.required_columns:
        if c not in names:
            continue
        raw = f"{c}_raw" if f"{c}_raw" in names else c
        missing = pl.col(raw).is_null() | is_placeholder(pl.col(raw), rules)
        out.append(issues(lf.filter(missing), rule_id="generic.missing_required", table="raw", rules=rules,
                          row_key=pl.col("row_nr"), column=c, value=pl.col(raw),
                          reason=f"required column {c} is empty"))
    return concat_issues(out)


def check_null_placeholders(raw: Frame, rules: Rules) -> pl.DataFrame:
    """Placeholder text ('N/A', '-', 'NULL', ...) in non-optional columns. Run on the mapped RAW frame."""
    lf = lazy(raw)
    schema = lf.collect_schema()
    cols = [c for c, t in schema.items() if t == pl.String and c not in rules.optional_columns]
    out = [
        issues(lf.filter(is_placeholder(pl.col(c), rules) & (pl.col(c).str.strip_chars() != "")),
               rule_id="generic.null_placeholder", table="raw", rules=rules, row_key=pl.col("row_nr"),
               column=c, value=pl.col(c), reason="placeholder text treated as null")
        for c in cols
    ]
    return concat_issues(out)
