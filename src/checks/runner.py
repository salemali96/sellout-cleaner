"""Registry of all checks keyed by rule_id, run against a CheckContext."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import polars as pl

from src.checks import cross_table, generic, market, product, promo, sellout
from src.checks.base import concat_issues
from src.config import Rules


@dataclass
class CheckContext:
    raw: pl.LazyFrame          # mapped to standard names, values untouched (for placeholder detection)
    staged: pl.DataFrame       # normalized (nulls, whitespace, weeks, numbers), before case-pack conversion
    fixed: pl.DataFrame        # after all safe fixes
    tables: dict[str, pl.DataFrame]


CheckFn = Callable[[CheckContext, Rules], pl.DataFrame]

CHECKS: dict[str, CheckFn] = {
    "generic.missing_required": lambda c, r: generic.check_missing_required(c.staged, r),
    "generic.null_placeholder": lambda c, r: generic.check_null_placeholders(c.raw, r),
    "sellout.unparseable_week": lambda c, r: sellout.check_unparseable_week(c.staged, r),
    "sellout.unparseable_number": lambda c, r: sellout.check_unparseable_number(c.staged, r),
    "sellout.case_pack_row": lambda c, r: sellout.check_case_pack_rows(c.staged, r),
    "sellout.exact_duplicate": lambda c, r: sellout.check_exact_duplicates(c.fixed, r),
    "sellout.duplicate_key_conflict": lambda c, r: sellout.check_duplicate_key_conflict(c.fixed, r),
    "sellout.negative_value": lambda c, r: sellout.check_negative_values(c.fixed, r),
    "sellout.price_outlier": lambda c, r: sellout.check_price_outliers(c.fixed, r),
    "sellout.missing_week": lambda c, r: sellout.check_missing_weeks(c.fixed, r),
    "product.conflicting_attribute": lambda c, r: product.check_conflicting_attributes(c.fixed, r),
    "product.invalid_units_per_case": lambda c, r: product.check_invalid_units_per_case(c.fixed, r),
    "market.conflicting_name": lambda c, r: market.check_conflicting_names(c.fixed, r),
    "promo.invalid_date_range": lambda c, r: promo.check_invalid_date_range(c.tables["promo"], r),
    "promo.conflicting_attribute": lambda c, r: promo.check_conflicting_attributes(c.fixed, r),
    "cross_table.orphan_promo": lambda c, r: cross_table.check_orphan_promo(c.fixed, c.tables["promo"], r),
    "cross_table.orphan_product": lambda c, r: cross_table.check_orphan_product(c.fixed, c.tables["product"], r),
    "cross_table.orphan_market": lambda c, r: cross_table.check_orphan_market(c.fixed, c.tables["market"], r),
}


@dataclass
class CheckRun:
    issues: pl.DataFrame
    timings: dict[str, float] = field(default_factory=dict)


def run_checks(ctx: CheckContext, rules: Rules, only: list[str] | None = None) -> CheckRun:
    """Run enabled checks (optionally a subset). Checks never mutate the context."""
    selected = [r for r in CHECKS if rules.is_enabled(r) and (only is None or r in only)]
    frames, timings = [], {}
    for rule_id in selected:
        t0 = time.perf_counter()
        frames.append(CHECKS[rule_id](ctx, rules))
        timings[rule_id] = round(time.perf_counter() - t0, 3)
    return CheckRun(concat_issues(frames), timings)
