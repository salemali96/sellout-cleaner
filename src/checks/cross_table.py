"""Referential checks between the sellout fact and the dimensions."""

from __future__ import annotations

import polars as pl

from src.checks.base import issues, lazy
from src.config import Rules

Frame = pl.DataFrame | pl.LazyFrame


def _orphans(fact: Frame, dim: Frame, key: str, rule_id: str, rules: Rules, what: str) -> pl.DataFrame:
    rows = lazy(fact).filter(pl.col(key).is_not_null()).join(lazy(dim).select(key), on=key, how="anti")
    return issues(rows, rule_id=rule_id, table="sellout", rules=rules, row_key=pl.col("row_nr"), column=key,
                  value=pl.col(key), reason=f"{key} not found in {what} dimension")


def check_orphan_promo(fact: Frame, promo: Frame, rules: Rules) -> pl.DataFrame:
    return _orphans(fact, promo, "promo_id", "cross_table.orphan_promo", rules, "promo")


def check_orphan_product(fact: Frame, product: Frame, rules: Rules) -> pl.DataFrame:
    return _orphans(fact, product, "product_id", "cross_table.orphan_product", rules, "product")


def check_orphan_market(fact: Frame, market: Frame, rules: Rules) -> pl.DataFrame:
    return _orphans(fact, market, "market_id", "cross_table.orphan_market", rules, "market")
