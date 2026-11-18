"""Deterministic tools the agent calls. Tools read data or add FixOps to the plan; none of them edits data.

Data changes only happen in `src.fixes.apply_plan`, which refuses risky ops that a human has not approved.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from src.checks.runner import CheckContext, CheckRun, run_checks
from src.config import Rules, SchemaMapping
from src.fixes import FixOp, PlanResult, apply_plan
from src.ingest import ROW_NR, raw_columns, scan_raw
from src.pipeline import Prepared

# ---- standard schema (what the profiler maps onto) ----------------------------------------

STANDARD_COLUMNS: dict[str, str] = {
    "product_id": "product / SKU / EAN code (key of product table)",
    "product_name": "product description",
    "brand": "brand name",
    "category": "product category",
    "pack_size": "pack size text like 500ml, 6x200ml",
    "uom": "unit of measure of the quantity: EA/PC (units) or CS/CASE (cases)",
    "units_per_case": "number of units in one case",
    "market_id": "store / outlet / market code (key of market table)",
    "market_name": "store / market name (any language)",
    "channel": "trade channel (hypermarket, pharmacy, ...)",
    "region": "geographic region / governorate",
    "promo_id": "promotion id (empty when no promo)",
    "promo_type": "promotion mechanic (TPR, BOGO, ...)",
    "promo_start_date": "promotion start date",
    "promo_end_date": "promotion end date",
    "week_raw": "period of the sale: ISO week, yyyyww, or a date",
    "units": "quantity sold",
    "value": "sales value / revenue in money",
}
REQUIRED_STANDARD = ["product_id", "market_id", "week_raw", "units", "value"]
TABLE_TEMPLATE: dict[str, dict[str, list[str]]] = {
    "product": {"key": ["product_id"], "attributes": ["product_name", "brand", "category", "pack_size", "units_per_case"]},
    "market": {"key": ["market_id"], "attributes": ["market_name", "channel", "region"]},
    "promo": {"key": ["promo_id"], "attributes": ["promo_type", "promo_start_date", "promo_end_date"]},
    "sellout": {"key": ["product_id", "market_id", "week_start_date"], "measures": ["units", "value"],
                "foreign_keys": ["promo_id"]},
}


def profile_columns(path: Path, rules: Rules, n_rows: int = 5000, n_samples: int = 8) -> list[dict[str, Any]]:
    """Header + sample values + fill/numeric/date-like rates for each raw column (first n_rows rows)."""
    lf, _ = scan_raw(path, rules.ingest)
    df = lf.head(n_rows).collect()
    ph = rules.placeholder_set
    out = []
    for c in raw_columns(lf):
        s = df[c].str.strip_chars()
        filled = s.filter(~s.str.to_lowercase().is_in(ph))
        out.append({
            "raw_column": c,
            "fill_rate": round(filled.len() / max(df.height, 1), 3),
            "distinct": filled.n_unique(),
            "numeric_rate": round(filled.str.replace_all(",", "").cast(pl.Float64, strict=False).is_not_null().mean() or 0, 3),
            "samples": filled.unique(maintain_order=True).head(n_samples).to_list(),
        })
    return out


@dataclass
class MappingProposal:
    columns: dict[str, str]                 # raw -> standard (ignored columns removed)
    low_confidence: list[str] = field(default_factory=list)
    notes: str = ""


def validate_mapping(columns: dict[str, str], raw_cols: list[str]) -> list[str]:
    errors = [f"raw column {r!r} does not exist" for r in columns if r not in raw_cols]
    errors += [f"{s!r} is not a standard column" for s in columns.values() if s not in STANDARD_COLUMNS]
    used = list(columns.values())
    errors += [f"standard column {s!r} mapped more than once" for s in sorted(set(used)) if used.count(s) > 1]
    errors += [f"required standard column {s!r} is not mapped" for s in REQUIRED_STANDARD if s not in used]
    return errors


def mapping_from_columns(columns: dict[str, str]) -> SchemaMapping:
    """Table split: the template, restricted to the standard columns that were actually mapped."""
    mapped = set(columns.values()) | {"week_start_date"}
    tables = {}
    for name, spec in TABLE_TEMPLATE.items():
        tables[name] = {k: [c for c in v if c in mapped] if k != "key" else v for k, v in spec.items()}
    return SchemaMapping.model_validate({"columns": columns, "tables": tables})


def write_mapping(mapping: SchemaMapping, path: Path, header: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.with_suffix(path.suffix + ".bak").write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    body = yaml.safe_dump(mapping.model_dump(exclude_defaults=True), sort_keys=False, allow_unicode=True)
    path.write_text(header + body, encoding="utf-8")
    return path


# ---- workspace: the state the cleaner tools read ------------------------------------------


@dataclass
class Workspace:
    path: Path
    rules: Rules
    mapping: SchemaMapping
    prepared: Prepared
    plan: list[dict[str, Any]] = field(default_factory=list)   # {"op": FixOp dict, "status": ...}
    current: PlanResult | None = None
    checks: CheckRun | None = None

    def active_ops(self) -> list[FixOp]:
        return [FixOp(**p["op"]) for p in self.plan if p["status"] in ("applied", "approved")]

    def materialize(self) -> tuple[CheckContext, CheckRun]:
        """Replay the plan from the base data and re-run every check."""
        base = self.prepared.ctx
        self.current = apply_plan(base.fixed, self.active_ops(), self.rules, self.mapping)
        ctx = CheckContext(raw=base.raw, staged=base.staged, fixed=self.current.fixed, tables=self.current.tables)
        self.checks = run_checks(ctx, self.current.rules)
        return ctx, self.checks

    @property
    def issues(self) -> pl.DataFrame:
        assert self.checks is not None
        return self.checks.issues


def _clip(v: Any, n: int = 120) -> Any:
    return v[:n] + "..." if isinstance(v, str) and len(v) > n else v


def issue_summary(issues: pl.DataFrame, samples: int = 3) -> list[dict[str, Any]]:
    summary = issues.group_by(["rule_id", "table", "severity"]).len().sort("len", descending=True)
    out = []
    for r in summary.to_dicts():
        ex = issues.filter(pl.col("rule_id") == r["rule_id"]).head(samples).select("row_key", "column", "value", "reason")
        out.append({**r, "count": r.pop("len"), "examples": [{k: _clip(v) for k, v in e.items()} for e in ex.to_dicts()]})
    return out


ID_COLUMNS = ["promo_id", "product_id", "market_id"]


def _is_row_key(key: str) -> bool:
    return key.isdigit()


def evidence(ws: Workspace, max_ids: int = 30) -> dict[str, Any]:
    """Everything the cleaner usually looks up first, computed by code so it needs no tool calls.

    Per rule: how many issues, which entity ids are affected (dimension-level rules) or the most common values
    (row-level rules); plus product/market pairs that look like duplicates under two ids.
    """
    per_rule: list[dict[str, Any]] = []
    for r in issue_summary(ws.issues, samples=1):
        sub = ws.issues.filter(pl.col("rule_id") == r["rule_id"])
        keys = sub["row_key"].unique().sort().to_list()
        entry = {"rule_id": r["rule_id"], "table": r["table"], "severity": r["severity"], "count": r["count"],
                 "example": r["examples"][0] if r["examples"] else None}
        id_cols = sub.filter(pl.col("column").is_in(ID_COLUMNS))
        if keys and not _is_row_key(str(keys[0])):
            entry["affected_ids"] = [_clip(k, 60) for k in keys[:max_ids]]
            entry["affected_ids_total"] = len(keys)
        elif id_cols.height == sub.height and sub.height:  # row-level, but about ids (e.g. orphan promo ids)
            ids = id_cols["value"].drop_nulls().unique().sort().to_list()
            entry["affected_ids"] = [_clip(v, 60) for v in ids[:max_ids]]
            entry["affected_ids_total"] = len(ids)
        else:
            top = sub.group_by("value").len().sort("len", descending=True).head(5)
            entry["top_values"] = [{"value": _clip(v, 60), "n": n} for v, n in top.iter_rows()]
        per_rule.append(entry)
    dups = {}
    for table in ("product", "market"):
        groups = t_find_duplicate_dimensions(ws, table)["groups"][:10]
        if groups:
            dups[table] = [[{k: _clip(v, 60) for k, v in row.items()} for row in g] for g in groups]
    return {"issues": per_rule, "possible_duplicate_ids": dups or "none found (no need to call find_duplicate_dimensions)"}


def issues_fingerprint(issues: pl.DataFrame) -> str:
    """Stable id of an issue set: identical file + mapping + rules give the same fingerprint."""
    import hashlib

    cols = issues.select("rule_id", "row_key", "column", "value").sort("rule_id", "row_key", "column", "value")
    digest = hashlib.sha256(cols.write_csv().encode("utf-8")).hexdigest()
    return f"{digest[:32]}:{issues.height}"


# ---- cleaner tools --------------------------------------------------------------------------

def _schema(props: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": required or list(props), "additionalProperties": False}


_STR_LIST = {"type": "array", "items": {"type": "string"}}
_RATIONALE = {"type": "string", "description": "why this fix is right, in one or two sentences"}
_ALL_FLAGGED = {"type": "boolean", "description": "true = every id flagged by the matching rule (then pass promo_ids "
                                                  "as []); cheaper than listing many ids"}
_MERGES = {"type": "array", "items": _schema({"from_id": {"type": "string"}, "to_id": {"type": "string"}})}

TOOL_SPECS: list[dict[str, Any]] = [
    {"name": "get_issue_samples", "description": "Return up to `limit` issues (max 20) for one rule_id.",
     "input_schema": _schema({"rule_id": {"type": "string"}, "limit": {"type": "integer"}})},
    {"name": "lookup_dimension", "description": "Return dimension rows (product, market or promo) for the given ids.",
     "input_schema": _schema({"table": {"type": "string", "enum": ["product", "market", "promo"]}, "ids": _STR_LIST})},
    {"name": "get_rows", "description": "Return staged sellout rows by row_nr (max 20), with raw week/units/value text.",
     "input_schema": _schema({"row_nrs": {"type": "array", "items": {"type": "integer"}}})},
    {"name": "find_duplicate_dimensions",
     "description": "Return groups of products or markets whose ids differ but whose names are the same after "
                    "ignoring case, spaces and punctuation. These are merge CANDIDATES only: compare the other "
                    "attributes before proposing a merge.",
     "input_schema": _schema({"table": {"type": "string", "enum": ["product", "market"]}})},
    {"name": "withdraw_fix", "description": "Remove a fix you added earlier (by plan_index) from the plan.",
     "input_schema": _schema({"plan_index": {"type": "integer"}, "rationale": _RATIONALE})},
    {"name": "fix_swap_promo_dates", "description": "SAFE. Swap start/end dates of promos whose end is before start.",
     "input_schema": _schema({"promo_ids": _STR_LIST, "all_flagged": _ALL_FLAGGED, "rationale": _RATIONALE})},
    {"name": "fix_add_promo_stubs",
     "description": "SAFE. Add placeholder promo rows (promo_type UNDEFINED) for promo ids used in sellout but never defined.",
     "input_schema": _schema({"promo_ids": _STR_LIST, "all_flagged": _ALL_FLAGGED, "rationale": _RATIONALE})},
    {"name": "fix_set_conflict_strategy",
     "description": "Choose how duplicate keys with different values are handled. quarantine_both is SAFE; "
                    "keep_first / keep_last publish one of the conflicting values and are RISKY (need approval).",
     "input_schema": _schema({"strategy": {"type": "string", "enum": ["quarantine_both", "keep_first", "keep_last"]},
                              "rationale": _RATIONALE})},
    {"name": "fix_merge_products", "description": "RISKY (needs approval). Re-point product ids onto another product id.",
     "input_schema": _schema({"merges": _MERGES, "rationale": _RATIONALE})},
    {"name": "fix_merge_markets", "description": "RISKY (needs approval). Re-point market ids onto another market id.",
     "input_schema": _schema({"merges": _MERGES, "rationale": _RATIONALE})},
    {"name": "fix_delete_rows", "description": "RISKY (needs approval). Move rows to quarantine (never truly deleted).",
     "input_schema": _schema({"row_nrs": {"type": "array", "items": {"type": "integer"}}, "reason": {"type": "string"},
                              "rationale": _RATIONALE})},
]
for _t in TOOL_SPECS:
    _t["strict"] = True


class ToolError(ValueError):
    pass


def _ids(ws: Workspace, table: str) -> set[str]:
    assert ws.current is not None
    return set(ws.current.tables[table][f"{table}_id"].to_list())


def _issue_keys(ws: Workspace, rule_id: str, value_in: list[str] | None = None) -> list[str]:
    f = ws.issues.filter(pl.col("rule_id") == rule_id)
    if value_in is not None:
        f = f.filter(pl.col("value").is_in(value_in) | pl.col("row_key").is_in(value_in))
    return f["row_key"].to_list()


def _add(ws: Workspace, op: FixOp) -> dict[str, Any]:
    status = "pending" if op.risky else "applied"
    ws.plan.append({"op": op.to_dict(), "status": status})
    return {"status": status, "plan_index": len(ws.plan) - 1,
            "note": "queued for human approval" if op.risky else "will be applied and verified by the validator"}


def t_get_issue_samples(ws: Workspace, rule_id: str, limit: int) -> Any:
    rows = ws.issues.filter(pl.col("rule_id") == rule_id).head(max(1, min(limit, 20)))
    return [{k: _clip(v) for k, v in r.items()} for r in rows.to_dicts()]


def t_lookup_dimension(ws: Workspace, table: str, ids: list[str]) -> Any:
    assert ws.current is not None
    return ws.current.tables[table].filter(pl.col(f"{table}_id").is_in(ids[:30])).to_dicts()


def t_get_rows(ws: Workspace, row_nrs: list[int]) -> Any:
    cols = ["row_nr", "product_id", "market_id", "week_raw", "week_start_date", "units_raw", "units", "value_raw",
            "value", "uom", "promo_id"]
    df = ws.prepared.ctx.staged.filter(pl.col(ROW_NR).is_in(row_nrs[:20]))
    return df.select([c for c in cols if c in df.columns]).to_dicts()


def t_find_duplicate_dimensions(ws: Workspace, table: str) -> Any:
    assert ws.current is not None
    df, key, name = ws.current.tables[table], f"{table}_id", f"{table}_name"
    norm = pl.col(name).str.to_lowercase().str.replace_all(r"[^\p{L}\p{N}]+", "")
    keyed = df.filter(pl.col(name).is_not_null()).with_columns(norm.alias("_norm"))
    dup = keyed.filter(pl.col("_norm").len().over("_norm") > 1).sort("_norm", key)
    groups = [g.drop("_norm").head(6).to_dicts() for g in dup.partition_by("_norm", maintain_order=True)[:20]]
    return {"groups": groups, "note": "same name is not proof of the same entity"}


def _flagged_values(ws: Workspace, rule_id: str) -> list[str]:
    sub = ws.issues.filter(pl.col("rule_id") == rule_id)
    col = "value" if sub.filter(pl.col("column").is_in(ID_COLUMNS)).height == sub.height else "row_key"
    return sorted(sub[col].drop_nulls().unique().to_list())


def t_swap_promo_dates(ws: Workspace, promo_ids: list[str], all_flagged: bool, rationale: str) -> Any:
    if all_flagged:
        promo_ids = _flagged_values(ws, "promo.invalid_date_range")
    if not promo_ids:
        raise ToolError("no promo ids given (pass ids, or all_flagged=true)")
    flagged = set(_issue_keys(ws, "promo.invalid_date_range"))
    bad = [p for p in promo_ids if p not in flagged]
    if bad:
        raise ToolError(f"promo ids without an invalid_date_range issue: {bad}")
    return _add(ws, FixOp("swap_promo_dates", {"promo_ids": promo_ids}, False, rationale,
                          resolves="promo.invalid_date_range", targets=promo_ids))


def t_add_promo_stubs(ws: Workspace, promo_ids: list[str], all_flagged: bool, rationale: str) -> Any:
    if all_flagged:
        promo_ids = _flagged_values(ws, "cross_table.orphan_promo")
    if not promo_ids:
        raise ToolError("no promo ids given (pass ids, or all_flagged=true)")
    known = _ids(ws, "promo")
    bad = [p for p in promo_ids if p in known]
    if bad:
        raise ToolError(f"promo ids already defined: {bad}")
    targets = _issue_keys(ws, "cross_table.orphan_promo", promo_ids)
    return _add(ws, FixOp("add_promo_stubs", {"promo_ids": promo_ids}, False, rationale,
                          resolves="cross_table.orphan_promo", targets=targets))


def t_set_conflict_strategy(ws: Workspace, strategy: str, rationale: str) -> Any:
    return _add(ws, FixOp("set_conflict_strategy", {"strategy": strategy}, strategy != "quarantine_both", rationale))


def _merge(ws: Workspace, table: str, merges: list[dict[str, str]], rationale: str) -> Any:
    ids = _ids(ws, table)
    bad = [m for m in merges if m["from_id"] not in ids or m["to_id"] not in ids or m["from_id"] == m["to_id"]]
    if bad or not merges:
        raise ToolError(f"invalid merges (ids must exist and differ): {bad or 'empty'}")
    return _add(ws, FixOp(f"merge_{table}s", {"merges": merges}, True, rationale))


def t_delete_rows(ws: Workspace, row_nrs: list[int], reason: str, rationale: str) -> Any:
    existing = set(ws.prepared.ctx.fixed.filter(pl.col(ROW_NR).is_in(row_nrs))[ROW_NR].to_list())
    missing = sorted(set(row_nrs) - existing)
    if missing or not row_nrs:
        raise ToolError(f"unknown row_nrs: {missing[:20] or 'empty'}")
    return _add(ws, FixOp("delete_rows", {"row_nrs": sorted(set(row_nrs)), "reason": reason}, True, rationale))


def t_withdraw_fix(ws: Workspace, plan_index: int, rationale: str) -> Any:
    if not 0 <= plan_index < len(ws.plan) or ws.plan[plan_index]["status"] == "withdrawn":
        raise ToolError(f"no active fix at plan_index {plan_index}")
    ws.plan[plan_index]["status"] = "withdrawn"
    ws.plan[plan_index]["withdraw_rationale"] = rationale
    return {"status": "withdrawn", "plan_index": plan_index}


HANDLERS: dict[str, Callable[..., Any]] = {
    "withdraw_fix": t_withdraw_fix,
    "get_issue_samples": t_get_issue_samples,
    "lookup_dimension": t_lookup_dimension,
    "get_rows": t_get_rows,
    "find_duplicate_dimensions": t_find_duplicate_dimensions,
    "fix_swap_promo_dates": t_swap_promo_dates,
    "fix_add_promo_stubs": t_add_promo_stubs,
    "fix_set_conflict_strategy": t_set_conflict_strategy,
    "fix_merge_products": lambda ws, merges, rationale: _merge(ws, "product", merges, rationale),
    "fix_merge_markets": lambda ws, merges, rationale: _merge(ws, "market", merges, rationale),
    "fix_delete_rows": t_delete_rows,
}


def call_tool(ws: Workspace, name: str, args: dict[str, Any]) -> tuple[str, bool]:
    """Run a tool. Returns (JSON result text, is_error)."""
    if name not in HANDLERS:
        return f"unknown tool {name!r}", True
    try:
        return json.dumps(HANDLERS[name](ws, **args), default=str, ensure_ascii=False), False
    except (ToolError, TypeError, KeyError) as e:
        return f"Error: {e}", True
