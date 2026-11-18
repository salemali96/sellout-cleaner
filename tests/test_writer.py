from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from src import pipeline, reconcile
from src.writer import scan_sellout
from tests.conftest import raw_row, weeks


def run(make_csv, rules, mapping, rows, tmp_path: Path) -> pipeline.RunResult:
    return pipeline.run(make_csv(rows), tmp_path / "out", rules, mapping)


def dirty_rows() -> list[dict[str, str]]:
    return weeks(6) + [
        raw_row(week="2025-W02"),                                   # exact duplicate of row 1 -> quarantined copy
        raw_row(week="2025-W03", units="99", value="990"),          # conflict with row 2 -> both quarantined
        raw_row(sku="SKU2", units="N/A"),                           # missing measure -> quarantined
        raw_row(sku="SKU2", week="2025-W04", units="-3", value="-30"),  # negative -> kept
        raw_row(sku="SKU3", uom="CS", upc="12", units="2", value="240"),  # case pack -> 24 units
    ]


def test_outputs_written_and_partitioned(make_csv, rules, mapping, tmp_path: Path) -> None:
    res = run(make_csv, rules, mapping, dirty_rows(), tmp_path)
    out = res.out_dir
    for f in ["product.parquet", "market.parquet", "promo.parquet", "quarantine.parquet", "issues.parquet",
              "run_log.json", "dq_report.html"]:
        assert (out / f).exists(), f
    assert (out / "sellout" / "year=2025" / "week=1").is_dir()
    so = scan_sellout(out).collect()
    assert {"year", "week", "product_id", "market_id", "week_start_date", "units", "value"} <= set(so.columns)
    assert pl.read_parquet(out / "product.parquet")["product_id"].to_list() == ["SKU1", "SKU2", "SKU3"]


def test_quarantine_rows_and_reasons(make_csv, rules, mapping, tmp_path: Path) -> None:
    res = run(make_csv, rules, mapping, dirty_rows(), tmp_path)
    q = pl.read_parquet(res.out_dir / "quarantine.parquet")
    assert dict(zip(q["row_nr"].to_list(), q["rule_id"].to_list())) == {
        2: "sellout.duplicate_key_conflict", 6: "sellout.exact_duplicate",
        7: "sellout.duplicate_key_conflict", 8: "generic.missing_required",
    }
    assert q["reason"].str.len_chars().min() > 0
    so = scan_sellout(res.out_dir).collect()
    assert -3.0 in so["units"].to_list()                       # warnings are kept
    assert so.filter(pl.col("product_id") == "SKU3")["units"].to_list() == [24.0]


def test_reconciliation_passes(make_csv, rules, mapping, tmp_path: Path) -> None:
    res = run(make_csv, rules, mapping, dirty_rows(), tmp_path)
    rec = res.log["reconciliation"]
    assert rec["passed"] and rec["raw"]["rows"] == 11
    assert rec["units_added_by_fixes"] == 22.0  # 2 cases x 12 - 2
    log = json.loads((res.out_dir / "run_log.json").read_text())
    assert log["row_counts"]["quarantine"] == 4 and log["row_counts"]["sellout"] == 7


def test_reconciliation_detects_lost_rows(make_csv, rules, mapping, tmp_path: Path) -> None:
    res = run(make_csv, rules, mapping, dirty_rows(), tmp_path)
    q = res.out_dir / "quarantine.parquet"
    pl.read_parquet(q).head(1).write_parquet(q)  # simulate rows silently dropped
    rec = reconcile.reconcile(11, res.prepared.ctx.staged, res.out_dir, 22.0)
    assert not rec.passed and rec.differences["rows"] == 3


def test_keep_first_strategy_keeps_one_conflict_row(make_csv, rules, mapping, tmp_path: Path) -> None:
    r = rules.model_copy(update={"dedup": rules.dedup.model_copy(update={"conflict_strategy": "keep_first"})})
    res = pipeline.run(make_csv(dirty_rows()), tmp_path / "out", r, mapping)
    q = pl.read_parquet(res.out_dir / "quarantine.parquet")
    assert 2 not in q["row_nr"].to_list() and 7 in q["row_nr"].to_list()
    assert res.log["reconciliation"]["passed"]


def test_cli_run(make_csv, tmp_path: Path) -> None:
    from src.cli import main

    assert main(["run", "--file", str(make_csv(dirty_rows())), "--out", str(tmp_path / "cli")]) == 0
    assert (tmp_path / "cli" / "dq_report.html").exists()
