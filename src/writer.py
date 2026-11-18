"""Quarantine selection and Parquet output.

Layout of the output directory:
  product.parquet, market.parquet, promo.parquet
  sellout/year=YYYY/week=WW/*.parquet     (hive partitioned)
  quarantine.parquet                        (one row per quarantined raw row, with rule_id + reason)
  issues.parquet                            (every issue found, errors and warnings)
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.split import build_sellout

ERROR = "error"


@dataclass
class Outputs:
    tables: dict[str, pl.DataFrame]   # final product/market/promo/sellout
    quarantine: pl.DataFrame


def row_level_errors(issues: pl.DataFrame) -> pl.DataFrame:
    """Error issues that point at a raw row (row_key is a row number), aggregated per row."""
    return (
        issues.filter((pl.col("severity") == ERROR) & pl.col("row_key").str.contains(r"^\d+$"))
        .with_columns(pl.col("row_key").cast(pl.UInt32).alias("row_nr"))
        .group_by("row_nr")
        .agg(
            pl.col("rule_id").unique().sort().str.join("|").alias("rule_id"),
            pl.format("{}: {}", pl.col("rule_id"), pl.col("reason")).unique().sort().str.join("; ").alias("reason"),
        )
    )


def select_quarantine(fixed: pl.DataFrame, issues: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split the fixed wide frame into (kept rows, quarantined rows with rule_id/reason)."""
    errs = row_level_errors(issues)
    quarantine = fixed.join(errs, on="row_nr", how="inner").sort("row_nr")
    kept = fixed.join(errs.select("row_nr"), on="row_nr", how="anti")
    return kept, quarantine


def build_outputs(
    fixed: pl.DataFrame,
    tables: dict[str, pl.DataFrame],
    issues: pl.DataFrame,
    extra_quarantine: pl.DataFrame | None = None,
) -> Outputs:
    """extra_quarantine: rows already removed from `fixed` by approved fixes (e.g. delete_rows)."""
    kept, quarantine = select_quarantine(fixed, issues)
    if extra_quarantine is not None and extra_quarantine.height:
        quarantine = pl.concat([quarantine, extra_quarantine.select(quarantine.columns)]).sort("row_nr")
    final = {k: v for k, v in tables.items() if k != "sellout"}
    final["sellout"] = build_sellout(kept).sort(["year", "week", "product_id", "market_id"])
    return Outputs(final, quarantine)


def write_outputs(out: Outputs, issues: pl.DataFrame, out_dir: Path) -> dict[str, str]:
    """Write all Parquet files. The sellout partition folder is replaced on every run."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for name in ["product", "market", "promo"]:
        p = out_dir / f"{name}.parquet"
        out.tables[name].write_parquet(p)
        paths[name] = str(p)
    sellout_dir = out_dir / "sellout"
    if sellout_dir.exists():
        shutil.rmtree(sellout_dir)
    out.tables["sellout"].write_parquet(sellout_dir, partition_by=["year", "week"], mkdir=True)
    paths["sellout"] = str(sellout_dir)
    for name, df in [("quarantine", out.quarantine), ("issues", issues)]:
        p = out_dir / f"{name}.parquet"
        df.write_parquet(p)
        paths[name] = str(p)
    return paths


def scan_sellout(out_dir: Path) -> pl.LazyFrame:
    return pl.scan_parquet(out_dir / "sellout" / "**" / "*.parquet", hive_partitioning=True)
