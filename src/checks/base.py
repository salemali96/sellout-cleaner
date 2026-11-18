"""Shared issue schema and helpers. Every check returns a DataFrame with ISSUE_SCHEMA columns.

row_key convention:
  - row-level checks (tables raw/sellout/cross_table): the raw `row_nr` as a string -> eligible for quarantine
  - group-level checks: a readable composite key, e.g. "SKU10001" or "SKU10001|ST0001|2025-03-10"
"""

from __future__ import annotations

import polars as pl

from src.config import Rules

ISSUE_SCHEMA: dict[str, pl.DataType] = {
    "rule_id": pl.String(), "table": pl.String(), "severity": pl.String(), "row_key": pl.String(),
    "column": pl.String(), "value": pl.String(), "reason": pl.String(),
}
ISSUE_COLUMNS = list(ISSUE_SCHEMA)


def empty_issues() -> pl.DataFrame:
    return pl.DataFrame(schema=ISSUE_SCHEMA)


def lazy(frame: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    # rechunk: polars 2.1 panics in pl.format over multi-chunk string views (no-op if already 1 chunk)
    return frame.rechunk().lazy() if isinstance(frame, pl.DataFrame) else frame


def as_str(e: pl.Expr) -> pl.Expr:
    return e.cast(pl.String)


def issues(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    rule_id: str,
    table: str,
    rules: Rules,
    row_key: pl.Expr,
    column: str,
    value: pl.Expr,
    reason: pl.Expr | str,
) -> pl.DataFrame:
    """Project already-filtered offending rows into the standard issue shape."""
    reason_e = pl.lit(reason) if isinstance(reason, str) else reason
    return (
        lazy(frame)
        .select(
            pl.lit(rule_id).alias("rule_id"),
            pl.lit(table).alias("table"),
            pl.lit(rules.severity_of(rule_id)).alias("severity"),
            as_str(row_key).alias("row_key"),
            pl.lit(column).alias("column"),
            as_str(value).alias("value"),
            as_str(reason_e).alias("reason"),
        )
        .collect()
    )


def concat_issues(frames: list[pl.DataFrame]) -> pl.DataFrame:
    frames = [f for f in frames if f.height]
    return pl.concat(frames, how="vertical") if frames else empty_issues()


def is_placeholder(e: pl.Expr, rules: Rules) -> pl.Expr:
    return e.str.strip_chars().str.to_lowercase().is_in(rules.placeholder_set).fill_null(False)


def normalized_text(e: pl.Expr) -> pl.Expr:
    """Case/whitespace-insensitive form used to compare attribute values."""
    return e.str.to_lowercase().str.strip_chars().str.replace_all(r"\s+", " ")


def conflicting_values(
    frame: pl.DataFrame | pl.LazyFrame, key: str, attr: str, *, rule_id: str, table: str, rules: Rules
) -> pl.DataFrame:
    """Keys whose attribute has more than one distinct (normalized) non-null value."""
    counts = (
        lazy(frame)
        .filter(pl.col(key).is_not_null() & pl.col(attr).is_not_null())
        .with_columns(normalized_text(pl.col(attr).cast(pl.String)).alias("_norm"))
        .group_by([key, "_norm"])
        .agg(pl.col(attr).cast(pl.String).mode().first().alias("_shown"), pl.len().alias("_n"))
        .sort([key, "_n", "_norm"], descending=[False, True, False])
        .group_by(key, maintain_order=True)
        .agg(
            pl.len().alias("_distinct"),
            pl.format("{} ({})", pl.col("_shown"), pl.col("_n")).str.join(" | ").alias("_values"),
        )
        .filter(pl.col("_distinct") > 1)
    )
    return issues(
        counts, rule_id=rule_id, table=table, rules=rules, row_key=pl.col(key), column=attr,
        value=pl.col("_values"),
        reason=pl.format("{} distinct values for {} (value (rows)); majority value kept", pl.col("_distinct"), pl.lit(attr)),
    )
