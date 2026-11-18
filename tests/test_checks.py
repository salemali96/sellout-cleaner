"""One test per check: a small clean fixture with exactly one issue injected."""

from __future__ import annotations

import polars as pl
import pytest

from src.checks import cross_table, generic, market, product, promo, sellout
from src.checks.base import ISSUE_COLUMNS
from src.checks.runner import CHECKS, run_checks
from src.config import Rules
from tests.conftest import raw_row, weeks


def keys(df: pl.DataFrame) -> list[str]:
    return sorted(df["row_key"].to_list())


def assert_shape(df: pl.DataFrame, rule_id: str, severity: str) -> None:
    assert df.columns == ISSUE_COLUMNS
    assert set(df["rule_id"]) == {rule_id}
    assert set(df["severity"]) == {severity}


# ---- generic --------------------------------------------------------------------------------

def test_missing_required(prep, rules) -> None:
    p = prep(weeks(3) + [raw_row(week="2025-W04", units="N/A"), raw_row(week="2025-W05", sku="")])
    out = generic.check_missing_required(p.ctx.staged, rules)
    assert_shape(out, "generic.missing_required", "error")
    assert sorted(zip(out["column"], out["row_key"])) == [("product_id", "4"), ("units", "3")]


def test_null_placeholders(prep, rules) -> None:
    p = prep(weeks(3) + [raw_row(week="2025-W04", brand="n/a", channel=" - ", promo="")])
    out = generic.check_null_placeholders(p.ctx.raw, rules)
    assert_shape(out, "generic.null_placeholder", "warning")
    assert sorted(out["column"]) == ["brand", "channel"]  # empty optional promo is fine
    assert set(out["row_key"]) == {"3"}


# ---- sellout --------------------------------------------------------------------------------

def test_unparseable_week(prep, rules) -> None:
    p = prep(weeks(2) + [raw_row(week="week three")])
    out = sellout.check_unparseable_week(p.ctx.staged, rules)
    assert_shape(out, "sellout.unparseable_week", "error")
    assert keys(out) == ["2"]


def test_unparseable_number(prep, rules) -> None:
    p = prep(weeks(2) + [raw_row(week="2025-W03", value="12O.5"), raw_row(week="2025-W04", value="1,200.50")])
    out = sellout.check_unparseable_number(p.ctx.staged, rules)
    assert_shape(out, "sellout.unparseable_number", "error")
    assert keys(out) == ["2"]  # thousand separator is fine, letter O is not


def test_exact_duplicates_identical_after_week_normalization(prep, rules) -> None:
    p = prep(weeks(3) + [raw_row(week="202502")])  # same as row 1 (2025-W02) once normalized
    out = sellout.check_exact_duplicates(p.ctx.fixed, rules)
    assert_shape(out, "sellout.exact_duplicate", "error")
    assert keys(out) == ["3"] and out["value"][0] == "1"


@pytest.mark.parametrize(("strategy", "expected"), [("quarantine_both", ["1", "3"]), ("keep_first", ["3"]),
                                                    ("keep_last", ["1"])])
def test_duplicate_key_conflict(prep, rules, strategy: str, expected: list[str]) -> None:
    p = prep(weeks(3) + [raw_row(week="2025-W02", units="99", value="990")])
    r = rules.model_copy(update={"dedup": rules.dedup.model_copy(update={"conflict_strategy": strategy})})
    out = sellout.check_duplicate_key_conflict(p.ctx.fixed, r)
    assert_shape(out, "sellout.duplicate_key_conflict", "error")
    assert keys(out) == expected


def test_conflict_ignores_exact_copies(prep, rules) -> None:
    p = prep(weeks(3) + [raw_row(week="2025-W02")])
    assert sellout.check_duplicate_key_conflict(p.ctx.fixed, rules).height == 0


def test_negative_values(prep, rules) -> None:
    p = prep(weeks(3) + [raw_row(week="2025-W04", units="-2", value="-20")])
    out = sellout.check_negative_values(p.ctx.fixed, rules)
    assert_shape(out, "sellout.negative_value", "warning")
    assert keys(out) == ["3"]


def test_price_outliers(prep, rules) -> None:
    rows = weeks(6)  # price 10.0
    rows[2] = raw_row(week="2025-W03", units="10", value="1000")    # 100.0 -> outlier
    rows[4] = raw_row(week="2025-W05", units="10", value="140")     # 14.0 -> within +50%
    out = sellout.check_price_outliers(prep(rows).ctx.fixed, rules)
    assert_shape(out, "sellout.price_outlier", "warning")
    assert keys(out) == ["2"]


def test_price_check_runs_after_case_pack_conversion(prep, rules) -> None:
    rows = weeks(5) + [raw_row(week="2025-W06", uom="CS", upc="12", units="2", value="240")]
    assert sellout.check_price_outliers(prep(rows).ctx.fixed, rules).height == 0


def test_missing_weeks(prep, rules) -> None:
    rows = [r for i, r in enumerate(weeks(6)) if i not in (2, 3)]
    out = sellout.check_missing_weeks(prep(rows).ctx.fixed, rules)
    assert_shape(out, "sellout.missing_week", "warning")
    assert keys(out) == ["SKU1|ST1|2025-01-13", "SKU1|ST1|2025-01-20"]


def test_case_pack_rows(prep, rules) -> None:
    p = prep(weeks(2) + [raw_row(week="2025-W03", uom="Case", units="3", value="360")])
    out = sellout.check_case_pack_rows(p.ctx.staged, rules)
    assert_shape(out, "sellout.case_pack_row", "warning")
    assert keys(out) == ["2"]


# ---- product / market / promo ------------------------------------------------------------

def test_product_conflicting_attribute(prep, rules) -> None:
    p = prep(weeks(4) + [raw_row(week="2025-W05", brand="Sunvale"), raw_row(week="2025-W06", brand=" NILEGOLD ")])
    out = product.check_conflicting_attributes(p.ctx.fixed, rules)
    assert_shape(out, "product.conflicting_attribute", "warning")
    assert keys(out) == ["SKU1"] and out["column"][0] == "brand"
    assert out["value"][0].startswith("Nilegold (5)")  # case variant counted with the majority


def test_product_invalid_units_per_case(prep, rules) -> None:
    p = prep(weeks(2) + [raw_row(sku="SKU2", upc="0")])
    out = product.check_invalid_units_per_case(p.ctx.fixed, rules)
    assert_shape(out, "product.invalid_units_per_case", "warning")
    assert keys(out) == ["SKU2"]


def test_market_conflicting_name(prep, rules) -> None:
    rows = weeks(4) + [
        raw_row(week="2025-W05", store_name="ميجا مارت المعادي"),     # Arabic alias: fine
        raw_row(week="2025-W06", store_name="MEGAMART  maadi"),       # case/space variant: fine
        raw_row(store="ST2", store_name="FreshWay Zamalek"),
        raw_row(store="ST2", week="2025-W04", store_name="FreshWay Dokki"),  # real conflict
    ]
    out = market.check_conflicting_names(prep(rows).ctx.fixed, rules)
    assert_shape(out, "market.conflicting_name", "warning")
    assert keys(out) == ["ST2"]


def test_promo_invalid_date_range(prep, rules) -> None:
    rows = weeks(2, promo="PR1", promo_type="TPR", promo_start="2025-01-12", promo_end="2025-01-06")
    out = promo.check_invalid_date_range(prep(rows).ctx.tables["promo"], rules)
    assert_shape(out, "promo.invalid_date_range", "warning")
    assert keys(out) == ["PR1"]


def test_promo_conflicting_attribute(prep, rules) -> None:
    rows = weeks(3, promo="PR1", promo_type="TPR", promo_start="2025-01-06", promo_end="2025-01-26")
    rows.append(raw_row(sku="SKU2", promo="PR1", promo_type="BOGO", promo_start="2025-01-06", promo_end="26/01/2025"))
    out = promo.check_conflicting_attributes(prep(rows).ctx.fixed, rules)
    assert_shape(out, "promo.conflicting_attribute", "warning")
    assert out["column"].to_list() == ["promo_type"]  # same end date in another format is not a conflict


# ---- cross table ----------------------------------------------------------------------------

def test_orphan_promo(prep, rules) -> None:
    rows = weeks(2, promo="PR1", promo_type="TPR", promo_start="2025-01-01", promo_end="2025-01-31")
    rows.append(raw_row(week="2025-W03", promo="PR-GHOST"))
    p = prep(rows)
    out = cross_table.check_orphan_promo(p.ctx.fixed, p.ctx.tables["promo"], rules)
    assert_shape(out, "cross_table.orphan_promo", "warning")
    assert keys(out) == ["2"]


def test_orphan_product(prep, rules) -> None:
    p = prep(weeks(2))
    out = cross_table.check_orphan_product(p.ctx.fixed, p.ctx.tables["product"].filter(pl.lit(False)), rules)
    assert_shape(out, "cross_table.orphan_product", "error")
    assert keys(out) == ["0", "1"]


def test_orphan_market(prep, rules) -> None:
    p = prep(weeks(2))
    out = cross_table.check_orphan_market(p.ctx.fixed, p.ctx.tables["market"].filter(pl.lit(False)), rules)
    assert_shape(out, "cross_table.orphan_market", "error")
    assert keys(out) == ["0", "1"]


# ---- runner ---------------------------------------------------------------------------------

def test_clean_data_has_no_issues(prep, rules) -> None:
    assert run_checks(prep(weeks(8)).ctx, rules).issues.height == 0


def test_every_enabled_rule_is_registered(rules: Rules) -> None:
    assert set(rules.enabled_rules) <= set(CHECKS)
    assert set(CHECKS) <= set(rules.severity)


def test_disabled_rule_does_not_run(prep, rules) -> None:
    p = prep(weeks(3) + [raw_row(week="2025-W04", units="-1", value="-10")])
    r = rules.model_copy(update={"enabled_rules": [x for x in rules.enabled_rules if x != "sellout.negative_value"]})
    assert "sellout.negative_value" not in set(run_checks(p.ctx, r).issues["rule_id"])
