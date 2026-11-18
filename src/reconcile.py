"""Mandatory reconciliation: raw totals (rows, units, value) = output + quarantine.

Totals are read back from the written Parquet files, so this verifies what is on disk.
Units: the case-pack fix legitimately adds units (cases -> units); that delta comes from the fix log:
    raw_units + units_added_by_fixes == output_units + quarantine_units
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import polars as pl

from src.writer import scan_sellout

TOLERANCE = 1e-6


@dataclass
class Totals:
    rows: int
    units: float
    value: float


@dataclass
class Reconciliation:
    raw: Totals
    output: Totals
    quarantine: Totals
    units_added_by_fixes: float
    passed: bool
    differences: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


def totals(lf: pl.LazyFrame) -> Totals:
    row = lf.select(pl.len().alias("rows"), pl.col("units").sum().alias("units"),
                    pl.col("value").sum().alias("value")).collect().row(0, named=True)
    return Totals(int(row["rows"]), float(row["units"] or 0), float(row["value"] or 0))


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE * max(1.0, abs(a), abs(b))


def reconcile(raw_rows: int, staged: pl.DataFrame, out_dir: Path, units_added: float) -> Reconciliation:
    """raw_rows comes from an independent count of the raw file; staged holds parsed pre-fix measures."""
    raw_t = totals(staged.lazy())
    raw = Totals(raw_rows, raw_t.units, raw_t.value)
    output = totals(scan_sellout(out_dir))
    quarantine = totals(pl.scan_parquet(out_dir / "quarantine.parquet"))
    diffs = {
        "rows": raw.rows - (output.rows + quarantine.rows),
        "units": raw.units + units_added - (output.units + quarantine.units),
        "value": raw.value - (output.value + quarantine.value),
    }
    passed = (
        diffs["rows"] == 0
        and _close(raw.units + units_added, output.units + quarantine.units)
        and _close(raw.value, output.value + quarantine.value)
    )
    return Reconciliation(raw, output, quarantine, units_added, passed, {k: round(v, 6) for k, v in diffs.items()})
