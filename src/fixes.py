"""Fix functions. Checks never mutate data; fixes do, and every fix is logged.

Safe fixes run lazily and tag the rows they change with a boolean `_fix_<name>` marker column.
`collect_fix_log` turns those markers into a fix log after the single collect.
Risky fixes (merges, deletions) refuse to run without explicit human approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import polars as pl

from src.config import Rules
from src.weeks import is_canonical_week, week_start_expr

MARKER_PREFIX = "_fix_"
TEXT_EXCLUDE = {"row_nr"}


class ApprovalRequired(PermissionError):
    """Raised when a risky fix is called without human approval."""


@dataclass
class FixRecord:
    name: str
    risky: bool
    rows_affected: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _marker(name: str) -> str:
    return f"{MARKER_PREFIX}{name}"


def _text_columns(lf: pl.LazyFrame) -> list[str]:
    schema = lf.collect_schema()
    return [c for c, t in schema.items() if t == pl.String and c not in TEXT_EXCLUDE and not c.endswith("_raw")]


def trim_whitespace(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Strip and collapse internal whitespace in all text columns."""
    cols = _text_columns(lf)
    cleaned = {c: pl.col(c).str.strip_chars().str.replace_all(r"\s+", " ") for c in cols}
    changed = pl.any_horizontal([(cleaned[c] != pl.col(c)).fill_null(False) for c in cols])
    return lf.with_columns(changed.alias(_marker("trim_whitespace"))).with_columns(
        [e.alias(c) for c, e in cleaned.items()]
    )


def normalize_nulls(lf: pl.LazyFrame, rules: Rules) -> pl.LazyFrame:
    """Turn placeholders like 'N/A', '-', 'NULL' (case-insensitive) and empty strings into real nulls."""
    cols = _text_columns(lf)
    ph = rules.placeholder_set

    def is_ph(c: str) -> pl.Expr:
        return pl.col(c).str.strip_chars().str.to_lowercase().is_in(ph).fill_null(False)

    visible = pl.any_horizontal([is_ph(c) & (pl.col(c).str.strip_chars() != "") for c in cols])
    return lf.with_columns(visible.alias(_marker("normalize_nulls"))).with_columns(
        [pl.when(is_ph(c)).then(None).otherwise(pl.col(c)).alias(c) for c in cols]
    )


def normalize_week(lf: pl.LazyFrame, rules: Rules) -> pl.LazyFrame:
    """Parse `week_raw` in any accepted format into week_start_date (ISO Monday), iso year and week."""
    ws = week_start_expr("week_raw", rules.week.date_formats)
    return lf.with_columns(ws).with_columns(
        pl.col("week_start_date").dt.iso_year().alias("year"),
        pl.col("week_start_date").dt.week().cast(pl.Int32).alias("week"),
        (pl.col("week_start_date").is_not_null() & ~is_canonical_week("week_raw")).alias(_marker("normalize_week")),
    )


def keep_raw(lf: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    """Snapshot original text as `<col>_raw` (untouched by later fixes) for checks and quarantine."""
    return lf.with_columns([pl.col(c).alias(f"{c}_raw") for c in columns])


def parse_numbers(lf: pl.LazyFrame, columns: list[str]) -> pl.LazyFrame:
    """Cast measure columns to Float64 (thousand separators and spaces removed). Bad text -> null."""
    return lf.with_columns(
        [pl.col(c).str.replace_all(r"[,\s]", "").cast(pl.Float64, strict=False).alias(c) for c in columns]
    )


def parse_dates(lf: pl.LazyFrame, columns: list[str], rules: Rules) -> pl.LazyFrame:
    """Parse free-format date columns (e.g. promo start/end) using rules.week.date_formats."""
    return lf.with_columns(
        [
            pl.coalesce([pl.col(c).str.strptime(pl.Date, f, strict=False) for f in rules.week.date_formats]).alias(c)
            for c in columns
        ]
    )


def convert_case_pack(lf: pl.LazyFrame, rules: Rules) -> pl.LazyFrame:
    """Rows reported in cases: units *= units_per_case and uom -> 'EA'. Rows without a valid pack size are untouched."""
    upc = pl.col("units_per_case").cast(pl.Float64, strict=False)
    is_case = pl.col("uom").str.to_uppercase().is_in([u.upper() for u in rules.case_pack.case_uom_values])
    convert = (is_case & (upc > 0)).fill_null(False)
    return lf.with_columns(
        convert.alias(_marker("convert_case_pack")),
        pl.when(convert).then(pl.col("units") * upc).otherwise(pl.col("units")).alias("units"),
        pl.when(convert).then(pl.lit("EA")).otherwise(pl.col("uom")).alias("uom"),
        pl.when(convert).then(pl.col("units") * upc - pl.col("units")).otherwise(0.0).alias("_units_added"),
    )


def collect_fix_log(df: pl.DataFrame, risky: set[str] | None = None) -> tuple[pl.DataFrame, list[FixRecord]]:
    """Summarize `_fix_*` markers into FixRecords and drop them from the frame."""
    markers = [c for c in df.columns if c.startswith(MARKER_PREFIX)]
    log = []
    for m in markers:
        name = m.removeprefix(MARKER_PREFIX)
        detail: dict[str, Any] = {}
        if name == "convert_case_pack" and "_units_added" in df.columns:
            detail["units_added"] = float(df["_units_added"].sum())
        log.append(FixRecord(name, name in (risky or set()), int(df[m].sum()), detail))
    return df.drop(markers), log


# ---- risky fixes: require approval -------------------------------------------------------


def _require(approved: bool, name: str) -> None:
    if not approved:
        raise ApprovalRequired(f"'{name}' is a risky fix and needs human approval")


def merge_ids(df: pl.DataFrame, column: str, mapping: dict[str, str], approved: bool = False) -> tuple[pl.DataFrame, FixRecord]:
    """Replace ids in `column` (e.g. merge duplicate products/markets). RISKY."""
    _require(approved, f"merge_{column}")
    affected = int(df[column].is_in(list(mapping)).sum())
    out = df.with_columns(pl.col(column).replace(mapping))
    return out, FixRecord(f"merge_{column}", True, affected, {"mapping": mapping})


def delete_rows(df: pl.DataFrame, row_nrs: list[int], reason: str, approved: bool = False) -> tuple[pl.DataFrame, pl.DataFrame, FixRecord]:
    """Move rows to quarantine (never silently dropped). RISKY. Returns (kept, removed, record)."""
    _require(approved, "delete_rows")
    mask = pl.col("row_nr").is_in(row_nrs)
    removed = df.filter(mask)
    return df.filter(~mask), removed, FixRecord("delete_rows", True, removed.height, {"reason": reason})


# ---- dimension fixes (safe) --------------------------------------------------------------


def swap_promo_dates(promo: pl.DataFrame, promo_ids: list[str]) -> tuple[pl.DataFrame, FixRecord]:
    """Swap start/end for the given promos where end < start."""
    hit = pl.col("promo_id").is_in(promo_ids) & (pl.col("promo_end_date") < pl.col("promo_start_date"))
    n = int(promo.select(hit.sum()).item())
    out = promo.with_columns(
        pl.when(hit).then(pl.col("promo_end_date")).otherwise(pl.col("promo_start_date")).alias("promo_start_date"),
        pl.when(hit).then(pl.col("promo_start_date")).otherwise(pl.col("promo_end_date")).alias("promo_end_date"),
    )
    return out, FixRecord("swap_promo_dates", False, n, {"promo_ids": promo_ids})


def add_promo_stubs(promo: pl.DataFrame, promo_ids: list[str], label: str = "UNDEFINED") -> tuple[pl.DataFrame, FixRecord]:
    """Add placeholder promo rows for ids referenced by sellout but never defined (keeps referential integrity)."""
    new = sorted(set(promo_ids) - set(promo["promo_id"].to_list()))
    stubs = pl.DataFrame({"promo_id": new}, schema={"promo_id": pl.String}).with_columns(pl.lit(label).alias("promo_type"))
    out = pl.concat([promo, stubs], how="diagonal_relaxed").sort("promo_id")
    return out, FixRecord("add_promo_stubs", False, len(new), {"promo_ids": new, "promo_type": label})


# ---- fix plan: what the agent builds, replayed deterministically -------------------------


@dataclass
class FixOp:
    """One fix the agent chose. Params are plain JSON. Risky ops need approved=True to run."""
    name: str
    params: dict[str, Any]
    risky: bool
    rationale: str = ""
    approved: bool = False
    resolves: str | None = None          # rule_id expected to disappear for `targets`
    targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlanResult:
    fixed: pl.DataFrame
    tables: dict[str, pl.DataFrame]
    rules: Rules
    manual_quarantine: pl.DataFrame
    records: list[FixRecord]


def apply_plan(fixed: pl.DataFrame, plan: list[FixOp], rules: Rules, mapping: Any) -> PlanResult:
    """Replay a fix plan over the fixed wide frame: rule ops, then fact ops, re-split, then dimension ops."""
    from src.split import split_tables

    records: list[FixRecord] = []
    removed: list[pl.DataFrame] = []
    for op in [o for o in plan if o.name == "set_conflict_strategy"]:
        _require(op.approved or not op.risky, op.name)
        rules = rules.model_copy(update={"dedup": rules.dedup.model_copy(update={"conflict_strategy": op.params["strategy"]})})
        records.append(FixRecord(op.name, op.risky, 0, op.params))
    for op in plan:
        if op.name in ("merge_products", "merge_markets"):
            col = "product_id" if op.name == "merge_products" else "market_id"
            fixed, rec = merge_ids(fixed, col, {m["from_id"]: m["to_id"] for m in op.params["merges"]}, op.approved)
            records.append(rec)
        elif op.name == "delete_rows":
            fixed, gone, rec = delete_rows(fixed, op.params["row_nrs"], op.params.get("reason", ""), op.approved)
            removed.append(gone.with_columns(pl.lit("manual.delete_rows").alias("rule_id"),
                                             pl.lit(f"manual.delete_rows: {op.rationale or op.params.get('reason', '')}").alias("reason")))
            records.append(rec)
    tables = split_tables(fixed, mapping, rules)
    for op in plan:
        if op.name == "swap_promo_dates":
            tables["promo"], rec = swap_promo_dates(tables["promo"], op.params["promo_ids"])
            records.append(rec)
        elif op.name == "add_promo_stubs":
            tables["promo"], rec = add_promo_stubs(tables["promo"], op.params["promo_ids"])
            records.append(rec)
    manual = pl.concat(removed, how="diagonal_relaxed") if removed else pl.DataFrame()
    return PlanResult(fixed, tables, rules, manual, records)
