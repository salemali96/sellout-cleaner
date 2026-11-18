"""Sellout fact checks: weeks, numbers, duplicates, negatives, price outliers, gaps, case-pack rows.

Input is the staged wide frame (standard column names, `row_nr`, parsed `week_start_date`/units/value).
"""

from __future__ import annotations

import polars as pl

from src.checks.base import concat_issues, is_placeholder, issues, lazy
from src.config import Rules

KEY = ["product_id", "market_id", "week_start_date"]
VALUES = ["units", "value", "promo_id"]
T = "sellout"
Frame = pl.DataFrame | pl.LazyFrame


def _row_key() -> pl.Expr:
    return pl.col("row_nr")


def check_unparseable_week(df: Frame, rules: Rules) -> pl.DataFrame:
    bad = lazy(df).filter(pl.col("week_raw").is_not_null() & pl.col("week_start_date").is_null())
    return issues(bad, rule_id="sellout.unparseable_week", table=T, rules=rules, row_key=_row_key(),
                  column="week_raw", value=pl.col("week_raw"),
                  reason="week not recognised as ISO week, yyyyww or a configured date format")


def check_unparseable_number(df: Frame, rules: Rules) -> pl.DataFrame:
    out = []
    for c in ["units", "value"]:
        raw = pl.col(f"{c}_raw")
        bad = lazy(df).filter(raw.is_not_null() & ~is_placeholder(raw, rules) & pl.col(c).is_null())
        out.append(issues(bad, rule_id="sellout.unparseable_number", table=T, rules=rules, row_key=_row_key(),
                          column=c, value=raw, reason=f"{c} is not a number"))
    return concat_issues(out)


def _with_identity_rank(df: Frame) -> pl.LazyFrame:
    """Rank rows within groups of identical key+values (1 = first occurrence by row_nr)."""
    return (
        lazy(df)
        .filter(pl.all_horizontal([pl.col(k).is_not_null() for k in KEY]))
        .with_columns(
            pl.col("row_nr").rank("ordinal").over(KEY + VALUES, order_by="row_nr").alias("_dup_rank"),
            pl.col("row_nr").min().over(KEY + VALUES).alias("_first_row"),
        )
    )


def check_exact_duplicates(df: Frame, rules: Rules) -> pl.DataFrame:
    """Same key and identical values (after normalization). First copy is kept, the rest flagged."""
    dups = _with_identity_rank(df).filter(pl.col("_dup_rank") > 1)
    return issues(dups, rule_id="sellout.exact_duplicate", table=T, rules=rules, row_key=_row_key(),
                  column="product_id,market_id,week_start_date", value=pl.col("_first_row"),
                  reason=pl.format("identical duplicate of row {}; dropped", pl.col("_first_row")))


def check_duplicate_key_conflict(df: Frame, rules: Rules) -> pl.DataFrame:
    """Same key, different values. Which rows are flagged depends on dedup.conflict_strategy."""
    strategy = rules.dedup.conflict_strategy
    firsts = _with_identity_rank(df).filter(pl.col("_dup_rank") == 1)
    grp = firsts.with_columns(
        pl.len().over(KEY).alias("_n"),
        pl.col("row_nr").min().over(KEY).alias("_min"),
        pl.col("row_nr").max().over(KEY).alias("_max"),
        pl.col("row_nr").sort().cast(pl.String).str.join(",").over(KEY).alias("_rows"),
    ).filter(pl.col("_n") > 1)
    if strategy == "keep_first":
        grp = grp.filter(pl.col("row_nr") != pl.col("_min"))
    elif strategy == "keep_last":
        grp = grp.filter(pl.col("row_nr") != pl.col("_max"))
    return issues(grp, rule_id="sellout.duplicate_key_conflict", table=T, rules=rules, row_key=_row_key(),
                  column="product_id,market_id,week_start_date", value=pl.col("_rows"),
                  reason=pl.format("same key with different values in rows {} (strategy: {})",
                                   pl.col("_rows"), pl.lit(strategy)))


def check_negative_values(df: Frame, rules: Rules) -> pl.DataFrame:
    neg = lazy(df).filter((pl.col("units") < 0) | (pl.col("value") < 0))
    col = pl.when(pl.col("units") < 0).then(pl.lit("units")).otherwise(pl.lit("value"))
    return issues(neg.with_columns(col.alias("_col")), rule_id="sellout.negative_value", table=T, rules=rules,
                  row_key=_row_key(), column="units,value",
                  value=pl.format("units={} value={}", pl.col("units"), pl.col("value")),
                  reason=pl.format("negative {} (possible return); kept", pl.col("_col")))


def check_price_outliers(df: Frame, rules: Rules) -> pl.DataFrame:
    """avg_price = value/units outside median*(1 +/- tolerance) of the product's median price."""
    tol = rules.price.outlier_tolerance
    priced = (
        lazy(df)
        .filter((pl.col("units") > 0) & (pl.col("value") > 0) & pl.col("product_id").is_not_null())
        .with_columns((pl.col("value") / pl.col("units")).alias("_price"))
        .with_columns(
            pl.col("_price").median().over("product_id").alias("_median"),
            pl.len().over("product_id").alias("_n"),
        )
    )
    out = priced.filter(
        (pl.col("_n") >= rules.price.min_rows_for_median)
        & ((pl.col("_price") < pl.col("_median") * (1 - tol)) | (pl.col("_price") > pl.col("_median") * (1 + tol)))
    )
    return issues(out, rule_id="sellout.price_outlier", table=T, rules=rules, row_key=_row_key(),
                  column="avg_price", value=pl.col("_price").round(2),
                  reason=pl.format("avg_price {} outside +/-{}% of product median {}", pl.col("_price").round(2),
                                   pl.lit(int(tol * 100)), pl.col("_median").round(2)))


def check_missing_weeks(df: Frame, rules: Rules) -> pl.DataFrame:
    """Weeks missing between the first and last week of each product x market series."""
    if not rules.gaps.enabled:
        return issues(lazy(df).head(0), rule_id="sellout.missing_week", table=T, rules=rules,
                      row_key=pl.lit(""), column="week_start_date", value=pl.lit(""), reason="")
    present = lazy(df).filter(pl.all_horizontal([pl.col(k).is_not_null() for k in KEY])).select(KEY).unique()
    expected = (
        present.group_by(KEY[:2])
        .agg(pl.date_ranges(pl.col("week_start_date").min(), pl.col("week_start_date").max(), "1w").first()
             .alias("week_start_date"))
        .explode("week_start_date")
    )
    missing = expected.join(present, on=KEY, how="anti").sort(KEY)
    return issues(missing, rule_id="sellout.missing_week", table=T, rules=rules,
                  row_key=pl.format("{}|{}|{}", *[pl.col(k) for k in KEY]), column="week_start_date",
                  value=pl.col("week_start_date"), reason="no row for this week inside the series")


def check_case_pack_rows(df: Frame, rules: Rules) -> pl.DataFrame:
    """Rows reported in cases (run on the frame BEFORE convert_case_pack)."""
    cases = [u.upper() for u in rules.case_pack.case_uom_values]
    rows = lazy(df).filter(pl.col("uom").str.to_uppercase().is_in(cases).fill_null(False))
    return issues(rows, rule_id="sellout.case_pack_row", table=T, rules=rules, row_key=_row_key(), column="uom",
                  value=pl.format("{} x {}", pl.col("units"), pl.col("units_per_case")),
                  reason="units reported in cases; converted to units with units_per_case")
