"""Compare detected issues with ground_truth.json (from scripts/make_dirty_data.py) row by row."""

from __future__ import annotations

from typing import Any

import polars as pl

ATTR_COLUMNS = ["product_name", "brand", "category", "pack_size", "channel", "region"]


def _keys(issues: pl.DataFrame, rule_id: str, column: str | None = None) -> set[str]:
    f = issues.filter(pl.col("rule_id") == rule_id)
    if column:
        f = f.filter(pl.col("column") == column)
    return set(f["row_key"].to_list())


def _score(name: str, rule_id: str | None, expected: set[str], detected: set[str]) -> dict[str, Any]:
    tp = len(expected & detected)
    return {
        "issue": name, "rule_id": rule_id or "-", "expected": len(expected), "detected": len(detected),
        "tp": tp, "fp": len(detected - expected), "fn": len(expected - detected),
        "precision": round(tp / len(detected), 4) if detected else (1.0 if not expected else 0.0),
        "recall": round(tp / len(expected), 4) if expected else 1.0,
    }


def _rows(xs: list[int]) -> set[str]:
    return {str(x) for x in xs}


def compare(issues: pl.DataFrame, truth: dict[str, Any], market: pl.DataFrame | None = None) -> pl.DataFrame:
    gt = truth["issues"]
    out = [
        _score("exact_duplicate", "sellout.exact_duplicate",
               {str(max(g)) for g in gt["exact_duplicate"]["groups"]}, _keys(issues, "sellout.exact_duplicate")),
        _score("duplicate_key_conflict", "sellout.duplicate_key_conflict",
               {str(r) for g in gt["duplicate_key_conflict"]["groups"] for r in g},
               _keys(issues, "sellout.duplicate_key_conflict")),
        _score("week_format (unparseable)", "sellout.unparseable_week", set(), _keys(issues, "sellout.unparseable_week")),
        _score("null_placeholder", "generic.null_placeholder",
               {f"{c}:{r}" for c, v in gt["null_placeholder"]["by_column"].items() for r in v["rows"]},
               {f"{c}:{k}" for c in ATTR_COLUMNS for k in _keys(issues, "generic.null_placeholder", c)}),
        _score("null_measure", "generic.missing_required", _rows(gt["null_measure"]["rows"]),
               _keys(issues, "generic.missing_required", "units")),
        _score("negative_units", "sellout.negative_value", _rows(gt["negative_units"]["rows"]),
               _keys(issues, "sellout.negative_value")),
        _score("price_outlier", "sellout.price_outlier", _rows(gt["price_outlier"]["rows"]),
               _keys(issues, "sellout.price_outlier")),
        _score("case_pack", "sellout.case_pack_row", _rows(gt["case_pack"]["rows"]), _keys(issues, "sellout.case_pack_row")),
        _score("missing_week", "sellout.missing_week",
               {f"{k['product_id']}|{k['market_id']}|{k['week_start_date']}" for k in gt["missing_week"]["keys"]},
               _keys(issues, "sellout.missing_week")),
        _score("orphan_promo", "cross_table.orphan_promo", _rows(gt["orphan_promo"]["rows"]),
               _keys(issues, "cross_table.orphan_promo")),
        _score("promo_invalid_date_range", "promo.invalid_date_range",
               set(gt["promo_invalid_date_range"]["promo_ids"]), _keys(issues, "promo.invalid_date_range")),
        _score("market_name_conflict", "market.conflicting_name", set(gt["market_name_conflict"]["market_ids"]),
               _keys(issues, "market.conflicting_name")),
        _score("product_attribute_conflict", "product.conflicting_attribute",
               set(gt["product_attribute_conflict"]["product_ids"]), _keys(issues, "product.conflicting_attribute")),
    ]
    if market is not None:
        with_ar = set(market.filter(pl.col("market_name_ar").is_not_null())["market_id"].to_list())
        out.append(_score("market_name_arabic (in market_name_ar)", None,
                          set(gt["market_name_arabic"]["market_ids"]), with_ar))
    return pl.DataFrame(out)
