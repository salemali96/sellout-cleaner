"""Generate a hard sellout CSV for testing the AI agents, plus an answer key.

On top of the normal dirt from make_dirty_data (duplicates, conflicts, blanks, negative sales, price outliers,
case packs, missing weeks, orphan promos, swapped promo dates) it adds judgment calls:

  - unfamiliar column names and extra columns           -> the Profiler must re-map
  - one product sold under two codes                    -> the Cleaner should propose a merge (needs approval)
  - one store re-coded half way                         -> the Cleaner should propose a merge (needs approval)
  - a different store that only shares a name (trap)    -> must NOT be merged
  - Excel junk: total/subtotal rows, a repeated header  -> must be quarantined, never counted as sales
  - semicolon separator and a UTF-8 BOM (Excel export)  -> ingest must detect them

Usage:
    uv run python scripts/make_challenge_data.py [--rows 30000] [--seed 7] [--out data/raw/sellout_challenge.csv]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_dirty_data import Rates, make  # noqa: E402

RENAME = {
    "SKU Code": "Item #", "Product Description": "Item Desc", "Brand": "Mfr Brand", "Category": "Cat.",
    "Pack Size": "Size", "UoM": "Sell Unit", "Units/Case": "Pcs/Ctn", "Store Code": "Outlet ID",
    "Store Name": "Outlet", "Channel": "Trade Channel", "Region": "Area", "Promo ID": "Promo Code",
    "Promo Type": "Mechanic", "Promo Start": "Promo From", "Promo End": "Promo To", "Week": "Period",
    "Sales Qty": "Qty Sold", "Sales Value (EGP)": "Net Sales EGP",
}
ORDER = ["Period", "Outlet ID", "Outlet", "Area", "Trade Channel", "Item #", "Item Desc", "Mfr Brand", "Cat.", "Size",
         "Sell Unit", "Pcs/Ctn", "Qty Sold", "Net Sales EGP", "Promo Code", "Mechanic", "Promo From", "Promo To",
         "Remarks", "Extract Date"]
HARDER = Rates(exact_duplicate=0.01, duplicate_key_conflict=0.006, null_placeholder=0.03, null_measure=0.003,
               negative_units=0.005, price_outlier=0.004, case_pack=0.02, orphan_promo=0.003, missing_week_pairs=0.03)


def busiest(df: pl.DataFrame, col: str, other: str, skip: set[str]) -> list[str]:
    """Ids of `col` sold with the most distinct `other` values (so half of them can be moved)."""
    counts = df.group_by(col).agg(pl.col(other).n_unique().alias("n")).filter(~pl.col(col).is_in(list(skip)))
    return counts.sort(["n", col], descending=[True, False])[col].to_list()


def move_half(df: pl.DataFrame, col: str, old: str, other: str, set_to: dict[str, str]) -> tuple[pl.DataFrame, list[str]]:
    """Give `old` new values (`set_to`) on whole series: every row for half of its `other` partners."""
    partners = sorted(df.filter(pl.col(col) == old)[other].unique().to_list())
    moved = partners[::2]
    hit = (pl.col(col) == old) & pl.col(other).is_in(moved)
    df = df.with_columns([pl.when(hit).then(pl.lit(v)).otherwise(pl.col(c)).alias(c) for c, v in set_to.items()])
    return df, moved


def name_of(df: pl.DataFrame, id_col: str, name_col: str, id_: str) -> str:
    names = df.filter(pl.col(id_col) == id_)[name_col].drop_nulls()
    names = names.filter(~names.is_in(["N/A", "-", "NULL", "n/a", "#N/A"]))
    return names.mode().sort()[0]


def junk_rows(df: pl.DataFrame) -> pl.DataFrame:
    def num(c: str) -> pl.Expr:
        return pl.col(c).str.replace_all(",", "").cast(pl.Float64, strict=False)

    by_area = (df.group_by("Area").agg(num("Qty Sold").sum().alias("q"), num("Net Sales EGP").sum().alias("v"))
               .drop_nulls("Area").filter(~pl.col("Area").is_in(["N/A", "-", "NULL", "n/a", "#N/A"])).sort("Area"))
    rows = [{"Period": "Subtotal", "Area": a, "Qty Sold": f"{q:.0f}", "Net Sales EGP": f"{v:.2f}",
             "Remarks": f"Subtotal {a}", "_tag": "subtotal"} for a, q, v in by_area.head(3).iter_rows()]
    rows.append({"Period": "Grand Total", "Item #": "TOTAL", "Qty Sold": f"{df.select(num('Qty Sold').sum()).item():.0f}",
                 "Net Sales EGP": f"{df.select(num('Net Sales EGP').sum()).item():.2f}", "_tag": "grand_total"})
    junk = pl.DataFrame(rows, schema={c: pl.String for c in [*ORDER, "_tag"]})
    header = pl.DataFrame([{**{c: c for c in ORDER}, "_tag": "repeated_header"}], schema=junk.schema)
    return pl.concat([junk, header])


def build(rows: int, seed: int) -> tuple[pl.DataFrame, dict]:
    df, truth = make(rows, seed, HARDER)
    df = df.rename(RENAME).with_columns(
        pl.lit(None, pl.String).alias("Remarks"), pl.lit("2026-01-04").alias("Extract Date"),
        pl.lit(None, pl.String).alias("_tag")).select([*ORDER, "_tag"])
    df = df.with_columns(pl.when(pl.col("Qty Sold").str.starts_with("-")).then(pl.lit("returned stock"))
                         .otherwise(pl.col("Remarks")).alias("Remarks"))
    taken: set[str] = set()

    # 1-2. two products, each partly sold under a second code with the same (upper-cased) name
    product_aliases = []
    for pid in busiest(df, "Item #", "Outlet ID", taken)[:2]:
        alias, name = pid.replace("SKU", "SKU-"), name_of(df, "Item #", "Item Desc", pid)
        df, outlets = move_half(df, "Item #", pid, "Outlet ID", {"Item #": alias, "Item Desc": name.upper()})
        taken.add(pid)
        product_aliases.append({"original": pid, "alias": alias, "name": name, "outlets_using_alias": len(outlets)})

    # 3. one store re-coded (system migration): same store, new code, lower-cased name
    sid = busiest(df, "Outlet ID", "Item #", taken)[0]
    s_name = name_of(df, "Outlet ID", "Outlet", sid)
    df, items = move_half(df, "Outlet ID", sid, "Item #", {"Outlet ID": f"EG-{sid}", "Outlet": s_name.lower()})
    taken.add(sid)
    market_alias = {"original": sid, "alias": f"EG-{sid}", "name": s_name, "items_using_alias": len(items)}

    # 4. trap: a different store (other area and channel) that happens to have the same name
    other = busiest(df, "Outlet ID", "Item #", taken)[0]
    df, items = move_half(df, "Outlet ID", other, "Item #",
                          {"Outlet ID": "ST9001", "Outlet": s_name, "Area": "Upper Egypt", "Trade Channel": "Pharmacy"})
    trap = {"id": "ST9001", "same_name_as": sid, "name": s_name, "area": "Upper Egypt", "channel": "Pharmacy",
            "items": len(items)}

    # 5. Excel junk: subtotal rows, a grand total and a repeated header row, spread through the file
    junk = junk_rows(df)
    cut = [df.height // 4, df.height // 2, (3 * df.height) // 4]
    df = pl.concat([df[: cut[0]], junk[4:5], df[cut[0]: cut[1]], junk[0:2], df[cut[1]: cut[2]], junk[2:3],
                    df[cut[2]:], junk[3:4]])  # header copy early, subtotals in the middle, grand total last
    tags = df.with_row_index("row_nr").filter(pl.col("_tag").is_not_null()).select("row_nr", "_tag").to_dicts()

    original = truth["issues"]
    answers = {
        "how_to_use": "Upload the CSV, run the AI agents with 'Re-map columns with AI' ticked, then compare with this key.",
        "file": {"rows_including_junk": df.height, "separator": ";", "encoding": "utf-8-sig (Excel BOM)"},
        "profiler_should": {
            "map": {v: k for k, v in RENAME.items()},
            "ignore": ["Remarks", "Extract Date"],
        },
        "cleaner_should_propose_with_approval": {
            "merge_products": product_aliases,
            "merge_markets": [market_alias],
        },
        "cleaner_must_not": {"merge_markets": trap,
                             "why": "same name, but a different area and channel: a different store"},
        "must_end_in_quarantine": {"junk_rows": tags,
                                   "why": "subtotals, the grand total and a repeated header are not sales"},
        "safe_fixes_expected": {"swap_promo_dates": original["promo_invalid_date_range"]["promo_ids"],
                                "add_promo_stubs": original["orphan_promo"]["promo_ids"]},
        "rules_should_catch_counts": {k: original[k].get("count") for k in
                                      ["exact_duplicate", "duplicate_key_conflict", "null_measure", "negative_units",
                                       "price_outlier", "case_pack", "missing_week", "orphan_promo"]},
        "note": "row numbers of the base issues shift because junk rows were inserted; counts still hold",
    }
    return df.drop("_tag"), answers


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=30_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("data/raw/sellout_challenge.csv"))
    args = ap.parse_args()
    df, answers = build(args.rows, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(df.write_csv(separator=";").encode("utf-8-sig"))
    key = args.out.with_name(f"{args.out.stem}_answers.json")
    key.write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {df.height:,} rows -> {args.out}\nanswer key -> {key}")


if __name__ == "__main__":
    main()
