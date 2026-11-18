"""Phase 1 pipeline: ingest -> stage -> safe fixes -> split -> checks -> quarantine -> write -> reconcile."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from src import fixes, ingest
from src.checks.runner import CheckContext, CheckRun, run_checks
from src.config import Rules, SchemaMapping
from src.ingest import FileFormat
from src.split import apply_mapping, split_tables, stage


@dataclass
class Prepared:
    ctx: CheckContext
    file_format: FileFormat
    fix_log: list[fixes.FixRecord]
    timings: dict[str, float] = field(default_factory=dict)


class Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}

    def lap(self, name: str, t0: float) -> float:
        now = time.perf_counter()
        self.timings[name] = round(now - t0, 3)
        return now


def prepare(path: Path, rules: Rules, mapping: SchemaMapping, work_dir: Path | None = None) -> Prepared:
    """Read lazily, normalize, apply safe fixes and split. One collect of the staged frame."""
    timer = Timer()
    t = time.perf_counter()
    lf, fmt = ingest.scan_raw(path, rules.ingest, work_dir)
    raw = apply_mapping(lf, mapping)
    staged_full = stage(raw, rules).collect().rechunk()
    t = timer.lap("ingest_stage", t)
    staged, log = fixes.collect_fix_log(staged_full)
    fixed_full = fixes.convert_case_pack(staged.lazy(), rules).collect().rechunk()
    fixed, case_log = fixes.collect_fix_log(fixed_full, set(rules.fixes.risky))
    t = timer.lap("safe_fixes", t)
    tables = split_tables(fixed, mapping, rules)
    timer.lap("split", t)
    ctx = CheckContext(raw=raw, staged=staged, fixed=fixed.drop("_units_added"), tables=tables)
    return Prepared(ctx, fmt, log + case_log, timer.timings)


def check(prepared: Prepared, rules: Rules, only: list[str] | None = None) -> CheckRun:
    return run_checks(prepared.ctx, rules, only)


@dataclass
class RunResult:
    log: dict
    issues: pl.DataFrame
    prepared: Prepared
    out_dir: Path


def finalize(
    path: Path,
    out_dir: Path,
    rules: Rules,
    prepared: Prepared,
    checks: CheckRun,
    ctx: CheckContext | None = None,
    extra_quarantine: pl.DataFrame | None = None,
    extra_fixes: list[fixes.FixRecord] | None = None,
    extra_log: dict | None = None,
    timer: Timer | None = None,
    t_start: float | None = None,
) -> RunResult:
    """Quarantine, write Parquet, reconcile against the raw file, write run_log.json + dq_report.html."""
    import datetime as dt
    import json

    from src import reconcile, report, writer

    timer = timer or Timer()
    ctx = ctx or prepared.ctx
    t = time.perf_counter()
    outputs = writer.build_outputs(ctx.fixed, ctx.tables, checks.issues, extra_quarantine)
    t = timer.lap("quarantine", t)
    paths = writer.write_outputs(outputs, checks.issues, out_dir)
    t = timer.lap("write_parquet", t)
    raw_rows = int(ingest.scan_raw(path, rules.ingest, out_dir / ".cache")[0].select(pl.len()).collect().item())
    all_fixes = prepared.fix_log + (extra_fixes or [])
    units_added = sum(f.detail.get("units_added", 0.0) for f in all_fixes)
    recon = reconcile.reconcile(raw_rows, prepared.ctx.staged, out_dir, units_added)
    timer.lap("reconcile", t)
    if t_start is not None:
        timer.timings["total"] = round(time.perf_counter() - t_start, 3)

    log = {
        "started_at": dt.datetime.now().isoformat(timespec="seconds"),
        "input": {"file": str(path), "encoding": prepared.file_format.encoding,
                  "delimiter": prepared.file_format.delimiter},
        "row_counts": {"raw": raw_rows, **{k: v.height for k, v in outputs.tables.items()},
                       "quarantine": outputs.quarantine.height},
        "issue_counts": {r["rule_id"]: r["count"] for r in report.issue_summary(checks.issues).to_dicts()},
        "fixes": [f.to_dict() for f in all_fixes],
        "reconciliation": recon.to_dict(),
        "timings": timer.timings,
        "check_timings": checks.timings,
        "outputs": paths,
        "config": {"conflict_strategy": rules.dedup.conflict_strategy,
                   "price_tolerance": rules.price.outlier_tolerance},
        **(extra_log or {}),
    }
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    report.write_report(log, checks.issues, out_dir / "dq_report.html")
    return RunResult(log, checks.issues, prepared, out_dir)


def run(path: Path, out_dir: Path, rules: Rules, mapping: SchemaMapping) -> RunResult:
    """Full Phase 1 run (no LLM). Writes Parquet outputs, run_log.json and dq_report.html into out_dir."""
    t_all = time.perf_counter()
    prepared = prepare(path, rules, mapping, work_dir=out_dir / ".cache")
    timer = Timer()
    timer.timings.update(prepared.timings)
    t = time.perf_counter()
    checks = check(prepared, rules)
    timer.lap("checks", t)
    return finalize(path, out_dir, rules, prepared, checks, timer=timer, t_start=t_all)
