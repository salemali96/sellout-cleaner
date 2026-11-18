"""Generate a realistic dirty sellout CSV plus ground_truth.json describing every injected issue.

Row numbers in ground_truth.json are 0-based data-row indexes in the CSV (header excluded),
which matches the `row_nr` column added by `src.ingest`.

Usage:
    uv run python scripts/make_dirty_data.py --rows 500000
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

YEAR = 2025
N_WEEKS = 52  # ISO 2025 has 52 weeks
WEEK1_MONDAY = np.datetime64("2024-12-30")

BRANDS = ["Nilegold", "Delta Farms", "Sunvale", "Karnak", "Oasis Fresh", "Pharos", "Desert Rose", "Lotus Valley"]
CATEGORIES = ["Dairy", "Snacks", "Beverages", "Biscuits", "Juice", "Confectionery"]
SIZES = ["200ml", "500ml", "1L", "75g", "150g", "300g", "6x200ml", "12x75g"]
CHAINS = [("MegaMart", "ميجا مارت"), ("FreshWay", "فريش واي"), ("CityMarket", "سيتي ماركت"),
          ("Al Baraka Market", "سوق البركة"), ("Nile Kiosk", "كشك النيل"), ("HealthPlus Pharmacy", "صيدلية هيلث بلس")]
DISTRICTS = [("Maadi", "المعادي"), ("Zamalek", "الزمالك"), ("Heliopolis", "مصر الجديدة"), ("Nasr City", "مدينة نصر"),
             ("Dokki", "الدقي"), ("Giza", "الجيزة"), ("Alexandria", "الإسكندرية"), ("6th October", "السادس من أكتوبر"),
             ("New Cairo", "القاهرة الجديدة"), ("Mohandessin", "المهندسين"), ("Tanta", "طنطا"), ("Mansoura", "المنصورة")]
REGION_OF = {"Alexandria": "Alexandria", "Giza": "Giza", "Dokki": "Giza", "6th October": "Giza", "Mohandessin": "Giza",
             "Tanta": "Delta", "Mansoura": "Delta"}
CHANNELS = ["Hypermarket", "Supermarket", "Traditional Trade", "Pharmacy", "E-commerce"]
PROMO_TYPES = ["TPR", "BOGO", "Display", "Feature", "Bundle"]
PLACEHOLDERS = ["N/A", "-", "NULL", "n/a", "#N/A"]
NULLABLE_ATTRS = ["product_name", "brand", "category", "pack_size", "channel", "region"]
WEEK_FORMATS = {"iso_week": 0.40, "yyyyww": 0.25, "iso_date_monday": 0.15, "dmy_any_weekday": 0.10, "compact_week": 0.10}

RAW_HEADER = {
    "product_id": "SKU Code", "product_name": "Product Description", "brand": "Brand", "category": "Category",
    "pack_size": "Pack Size", "uom": "UoM", "units_per_case": "Units/Case", "market_id": "Store Code",
    "market_name": "Store Name", "channel": "Channel", "region": "Region", "promo_id": "Promo ID",
    "promo_type": "Promo Type", "promo_start": "Promo Start", "promo_end": "Promo End", "week": "Week",
    "units": "Sales Qty", "value": "Sales Value (EGP)",
}


@dataclass
class Rates:
    exact_duplicate: float = 0.004
    duplicate_key_conflict: float = 0.003
    null_placeholder: float = 0.02
    null_measure: float = 0.001
    negative_units: float = 0.003
    price_outlier: float = 0.002
    case_pack: float = 0.01
    orphan_promo: float = 0.001
    missing_week_pairs: float = 0.015


@dataclass
class Truth:
    data: dict = field(default_factory=dict)


def n_of(rate: float, total: int, minimum: int = 2) -> int:
    return max(minimum, int(round(rate * total)))


def build_products(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    brand = rng.choice(BRANDS, n)
    cat = rng.choice(CATEGORIES, n)
    size = rng.choice(SIZES, n)
    variant = rng.integers(1, 99, n)
    return {
        "product_id": np.array([f"SKU{10001 + i}" for i in range(n)], dtype=object),
        "product_name": np.array([f"{b} {c} V{v} {s}" for b, c, v, s in zip(brand, cat, variant, size)], dtype=object),
        "brand": brand.astype(object), "category": cat.astype(object), "pack_size": size.astype(object),
        "units_per_case": rng.choice([6, 12, 24], n),
        "base_price": np.round(rng.uniform(5, 200, n), 2),
    }


def build_markets(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    chain_i = rng.integers(0, len(CHAINS), n)
    dist_i = rng.integers(0, len(DISTRICTS), n)
    en, ar, region = [], [], []
    seen: dict[str, int] = {}
    for c, d in zip(chain_i, dist_i):
        base_en = f"{CHAINS[c][0]} {DISTRICTS[d][0]}"
        k = seen.get(base_en, 0) + 1
        seen[base_en] = k
        suffix = f" {k}" if k > 1 else ""
        en.append(base_en + suffix)
        ar.append(f"{CHAINS[c][1]} {DISTRICTS[d][1]}{suffix}")
        region.append(REGION_OF.get(DISTRICTS[d][0], "Cairo"))
    return {
        "market_id": np.array([f"ST{i + 1:04d}" for i in range(n)], dtype=object),
        "market_name": np.array(en, dtype=object), "market_name_ar": np.array(ar, dtype=object),
        "channel": rng.choice(CHANNELS, n).astype(object), "region": np.array(region, dtype=object),
    }


def build_promos(rng: np.random.Generator, n: int, n_products: int) -> list[dict]:
    promos = []
    for j in range(n):
        start = int(rng.integers(0, N_WEEKS - 4))
        length = int(rng.integers(1, 5))
        prods = rng.choice(n_products, max(2, n_products // 20), replace=False)
        promos.append({"promo_id": f"PR{YEAR}-{j + 1:03d}", "type": str(rng.choice(PROMO_TYPES)),
                       "start_w": start, "end_w": start + length - 1, "products": prods,
                       "date_fmt": "%Y-%m-%d" if rng.random() < 0.7 else "%d/%m/%Y", "inverted": False})
    return promos


def fmt_date(d: np.datetime64, fmt: str) -> str:
    return d.astype("datetime64[D]").item().strftime(fmt)


def render_weeks(w: np.ndarray, fmt_codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Render ISO week index (0-based) as strings in the requested formats."""
    ww = np.char.zfill((w + 1).astype(str), 2).astype(object)
    monday = WEEK1_MONDAY + (w * 7).astype("timedelta64[D]")
    any_day = monday + rng.integers(0, 7, len(w)).astype("timedelta64[D]")
    iso_date = pl.Series(monday).cast(pl.Date).dt.strftime("%Y-%m-%d").to_numpy().astype(object)
    dmy = pl.Series(any_day).cast(pl.Date).dt.strftime("%d/%m/%Y").to_numpy().astype(object)
    out = np.empty(len(w), dtype=object)
    names = list(WEEK_FORMATS)
    choices = {
        "iso_week": f"{YEAR}-W" + ww, "yyyyww": f"{YEAR}" + ww, "iso_date_monday": iso_date,
        "dmy_any_weekday": dmy, "compact_week": f"{YEAR}W" + ww,
    }
    for i, name in enumerate(names):
        mask = fmt_codes == i
        out[mask] = choices[name][mask]
    return out


def make(rows: int, seed: int, rates: Rates) -> tuple[pl.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    n_pairs = math.ceil(rows / N_WEEKS)
    n_markets = max(10, min(120, n_pairs // 10))
    n_products = max(20, min(400, math.ceil(n_pairs * 3 / n_markets)))
    prod = build_products(rng, n_products)
    mkt = build_markets(rng, n_markets)
    promos = build_promos(rng, max(4, min(40, n_products // 5)), n_products)

    pair_ids = rng.choice(n_products * n_markets, n_pairs, replace=False)
    p = np.repeat(pair_ids // n_markets, N_WEEKS)
    m = np.repeat(pair_ids % n_markets, N_WEEKS)
    w = np.tile(np.arange(N_WEEKS), n_pairs)
    pair_of_row = np.repeat(np.arange(n_pairs), N_WEEKS)

    # --- missing weeks: drop 1-3 interior weeks from some series -------------------------
    keep = np.ones(len(p), dtype=bool)
    missing: list[dict] = []
    gap_pairs = rng.choice(n_pairs, n_of(rates.missing_week_pairs, n_pairs), replace=False)
    for gp in gap_pairs:
        weeks = rng.choice(np.arange(2, N_WEEKS - 2), int(rng.integers(1, 4)), replace=False)
        for wk in sorted(weeks):
            keep[gp * N_WEEKS + wk] = False
            monday = WEEK1_MONDAY + np.timedelta64(int(wk) * 7, "D")
            missing.append({"product_id": prod["product_id"][pair_ids[gp] // n_markets],
                            "market_id": mkt["market_id"][pair_ids[gp] % n_markets],
                            "week_start_date": str(monday)})
    p, m, w, pair_of_row = p[keep], m[keep], w[keep], pair_of_row[keep]
    n = len(p)

    # --- promos ---------------------------------------------------------------------------
    promo_idx = np.full(n, -1)
    for j, pr in enumerate(promos):
        mask = np.isin(p, pr["products"]) & (w >= pr["start_w"]) & (w <= pr["end_w"]) & (promo_idx == -1)
        promo_idx[mask] = j
    inverted = rng.choice(len(promos), max(1, len(promos) // 20), replace=False)
    for j in inverted:
        promos[j]["inverted"] = True

    # --- clean measures -------------------------------------------------------------------
    units = rng.integers(1, 300, n).astype(np.int64)
    price = prod["base_price"][p] * np.clip(1 + rng.normal(0, 0.04, n), 0.88, 1.12)
    price = np.where(promo_idx >= 0, price * 0.8, price)
    value = np.round(units * price, 2)
    uom = np.full(n, "EA", dtype=object)
    units_str = units.astype(str).astype(object)

    # --- string columns -------------------------------------------------------------------
    cols: dict[str, np.ndarray] = {k: prod[k][p].copy() for k in ["product_id", "product_name", "brand", "category", "pack_size"]}
    cols["units_per_case"] = prod["units_per_case"][p].astype(str).astype(object)
    for k in ["market_id", "market_name", "channel", "region"]:
        cols[k] = mkt[k][m].copy()

    # --- disjoint row-level injections ----------------------------------------------------
    perm = rng.permutation(n)
    cursor = 0

    def take(k: int, pool: np.ndarray | None = None) -> np.ndarray:
        nonlocal cursor
        if pool is None:
            sel = perm[cursor:cursor + k]
            cursor += k
            return np.sort(sel)
        cand = perm[cursor:]
        cand = cand[np.isin(cand, pool)][:k]
        perm_rest = perm[cursor:]
        perm[cursor:] = np.concatenate([cand, perm_rest[~np.isin(perm_rest, cand)]])
        cursor += len(cand)
        return np.sort(cand)

    exact_src = take(n_of(rates.exact_duplicate, n))
    conflict_src = take(n_of(rates.duplicate_key_conflict, n))
    null_measure = take(n_of(rates.null_measure, n))
    negative = take(n_of(rates.negative_units, n))
    outlier = take(n_of(rates.price_outlier, n))
    case_pack = take(n_of(rates.case_pack, n))
    orphan = take(n_of(rates.orphan_promo, n), pool=np.flatnonzero(promo_idx == -1))
    null_attr = take(n_of(rates.null_placeholder, n))

    # negatives (returns): price stays consistent
    units[negative] = -rng.integers(1, 20, len(negative))
    value[negative] = np.round(units[negative] * price[negative], 2)
    # price outliers: value x10 or x0.1
    factor = rng.choice([10.0, 0.1], len(outlier))
    value[outlier] = np.round(value[outlier] * factor, 2)
    # case-pack rows: quantity reported in cases, value unchanged in money
    upc = prod["units_per_case"][p[case_pack]]
    cases = rng.integers(1, 6, len(case_pack))
    units[case_pack] = cases
    value[case_pack] = np.round(cases * upc * price[case_pack], 2)
    uom[case_pack] = rng.choice(["CS", "CS", "Case"], len(case_pack))
    units_str = units.astype(str).astype(object)
    # null measures
    units_str[null_measure] = rng.choice(PLACEHOLDERS, len(null_measure))
    # null attribute placeholders
    attr_col = rng.choice(NULLABLE_ATTRS, len(null_attr))
    attr_ph = rng.choice(PLACEHOLDERS, len(null_attr))
    placeholder_truth: dict[str, list[int]] = {c: [] for c in NULLABLE_ATTRS}
    for r, c, ph in zip(null_attr, attr_col, attr_ph):
        cols[c][r] = ph
        placeholder_truth[c].append(int(r))

    # --- market name variants (whole-file, not disjoint) ----------------------------------
    mk_ids = np.arange(n_markets)
    ar_markets = rng.choice(mk_ids, max(1, int(n_markets * 0.4)), replace=False)
    rest = np.setdiff1d(mk_ids, ar_markets)
    variant_markets = rng.choice(rest, max(1, int(n_markets * 0.1)), replace=False)
    rest = np.setdiff1d(rest, variant_markets)
    conflict_markets = rng.choice(rest, max(1, n_markets // 40), replace=False)
    r_ar = np.flatnonzero(np.isin(m, ar_markets) & (rng.random(n) < 0.3))
    cols["market_name"][r_ar] = mkt["market_name_ar"][m[r_ar]]
    r_var = np.flatnonzero(np.isin(m, variant_markets) & (rng.random(n) < 0.2))
    for r in r_var:
        nm = mkt["market_name"][m[r]]
        cols["market_name"][r] = rng.choice([nm.upper(), nm.lower() + " ", "  " + nm.replace(" ", "  ")])
    conflict_market_rows: list[int] = []
    for cm in conflict_markets:
        rows_cm = np.flatnonzero(m == cm)
        bad = rng.choice(rows_cm, min(5, max(1, len(rows_cm) // 10)), replace=False)
        other = mkt["market_name"][(cm + 1) % n_markets]
        cols["market_name"][bad] = other
        conflict_market_rows += [int(x) for x in bad]

    # --- product attribute conflicts -------------------------------------------------------
    conflict_products = rng.choice(n_products, max(1, n_products // 100), replace=False)
    conflict_product_rows: list[int] = []
    taken = set(placeholder_truth["brand"])
    for cp in conflict_products:
        rows_cp = np.array([r for r in np.flatnonzero(p == cp) if r not in taken])
        if len(rows_cp) == 0:
            continue
        bad = rng.choice(rows_cp, min(5, max(1, len(rows_cp) // 10)), replace=False)
        wrong = next(b for b in BRANDS if b != prod["brand"][cp])
        cols["brand"][bad] = wrong
        conflict_product_rows += [int(x) for x in bad]

    # --- promo columns ---------------------------------------------------------------------
    promo_id = np.full(n, "", dtype=object)
    promo_type = np.full(n, "", dtype=object)
    promo_start = np.full(n, "", dtype=object)
    promo_end = np.full(n, "", dtype=object)
    for j, pr in enumerate(promos):
        rows_j = promo_idx == j
        s = WEEK1_MONDAY + np.timedelta64(pr["start_w"] * 7, "D")
        e = WEEK1_MONDAY + np.timedelta64(pr["end_w"] * 7 + 6, "D")
        if pr["inverted"]:
            s, e = e, s
        promo_id[rows_j] = pr["promo_id"]
        promo_type[rows_j] = pr["type"]
        promo_start[rows_j] = fmt_date(s, pr["date_fmt"])
        promo_end[rows_j] = fmt_date(e, pr["date_fmt"])
    orphan_ids = [f"PR{YEAR}-{900 + k:03d}" for k in range(max(1, len(orphan) // 3))]
    promo_id[orphan] = rng.choice(orphan_ids, len(orphan))
    used_orphans = sorted(set(promo_id[orphan]))

    # --- week formats ----------------------------------------------------------------------
    fmt_codes = rng.choice(len(WEEK_FORMATS), n, p=list(WEEK_FORMATS.values()))
    week = render_weeks(w, fmt_codes, rng)

    base = pl.DataFrame({
        "_gid": np.arange(n), "product_id": cols["product_id"], "product_name": cols["product_name"],
        "brand": cols["brand"], "category": cols["category"], "pack_size": cols["pack_size"], "uom": uom,
        "units_per_case": cols["units_per_case"], "market_id": cols["market_id"], "market_name": cols["market_name"],
        "channel": cols["channel"], "region": cols["region"], "promo_id": promo_id, "promo_type": promo_type,
        "promo_start": promo_start, "promo_end": promo_end, "week": week,
        "units": units_str, "value": np.char.mod("%.2f", value).astype(object),
    }, schema_overrides={k: pl.String for k in RAW_HEADER})

    # --- duplicates ------------------------------------------------------------------------
    exact = base[exact_src.tolist()].with_columns(pl.int_range(n, n + len(exact_src)).alias("_gid"))
    conf = base[conflict_src.tolist()]
    bump = rng.integers(1, 50, len(conflict_src))
    new_units = units[conflict_src] + bump
    new_value = np.round(new_units * price[conflict_src], 2)
    new_fmt = (fmt_codes[conflict_src] + rng.integers(1, len(WEEK_FORMATS), len(conflict_src))) % len(WEEK_FORMATS)
    conf = conf.with_columns(
        pl.int_range(n + len(exact_src), n + len(exact_src) + len(conflict_src)).alias("_gid"),
        pl.Series("units", new_units.astype(str)),
        pl.Series("value", np.char.mod("%.2f", new_value)),
        pl.Series("week", render_weeks(w[conflict_src], new_fmt, rng), dtype=pl.String),
    )
    full = pl.concat([base, exact, conf])

    # --- shuffle and map generation ids -> final row numbers --------------------------------
    order = rng.permutation(full.height)
    full = full[order.tolist()]
    pos = np.empty(full.height, dtype=np.int64)
    pos[full["_gid"].to_numpy()] = np.arange(full.height)

    def rows_of(gids: np.ndarray | list[int]) -> list[int]:
        return sorted(int(pos[g]) for g in gids)

    exact_gids = np.arange(n, n + len(exact_src))
    conflict_gids = np.arange(n + len(exact_src), n + len(exact_src) + len(conflict_src))
    fmt_names = list(WEEK_FORMATS)
    all_fmt = np.concatenate([fmt_codes, fmt_codes[exact_src], new_fmt])
    inverted_ids = sorted(promos[j]["promo_id"] for j in inverted if np.any(promo_idx == j))

    truth = {
        "generator": {"rows_requested": rows, "seed": seed, "n_products": n_products, "n_markets": n_markets,
                      "n_promos": len(promos), "year": YEAR, "rates": rates.__dict__},
        "row_numbering": "0-based data row index in the CSV (header excluded) == ingest row_nr",
        "n_rows": full.height,
        "issues": {
            "exact_duplicate": {"rule_id": "sellout.exact_duplicate", "count": len(exact_src),
                                "groups": [sorted([int(pos[s]), int(pos[g])]) for s, g in zip(exact_src, exact_gids)],
                                "note": "each group: [original, copy]; one copy should be dropped"},
            "duplicate_key_conflict": {"rule_id": "sellout.duplicate_key_conflict", "count": len(conflict_src),
                                       "groups": [sorted([int(pos[s]), int(pos[g])]) for s, g in zip(conflict_src, conflict_gids)],
                                       "note": "same product/market/week, different units/value, week often in a different format"},
            "week_format": {"rule_id": None, "counts": {fmt_names[i]: int((all_fmt == i).sum()) for i in range(len(fmt_names))},
                            "note": "all formats are valid and must normalize to the ISO Monday"},
            "null_placeholder": {"rule_id": "generic.null_placeholder",
                                 "by_column": {c: {"count": len(v), "rows": rows_of(v)} for c, v in placeholder_truth.items()}},
            "null_measure": {"rule_id": "generic.missing_required", "column": "units",
                             "count": len(null_measure), "rows": rows_of(null_measure)},
            "negative_units": {"rule_id": "sellout.negative_value", "count": len(negative), "rows": rows_of(negative)},
            "price_outlier": {"rule_id": "sellout.price_outlier", "count": len(outlier), "rows": rows_of(outlier)},
            "case_pack": {"rule_id": "sellout.case_pack_row", "count": len(case_pack), "rows": rows_of(case_pack),
                          "units_added_by_conversion": int((cases * upc - cases).sum())},
            "missing_week": {"rule_id": "sellout.missing_week", "count": len(missing), "keys": missing},
            "orphan_promo": {"rule_id": "cross_table.orphan_promo", "count": len(orphan), "promo_ids": used_orphans,
                             "rows": rows_of(orphan)},
            "promo_invalid_date_range": {"rule_id": "promo.invalid_date_range", "promo_ids": inverted_ids},
            "market_name_arabic": {"rule_id": None, "market_ids": sorted(mkt["market_id"][ar_markets].tolist()),
                                   "note": "legit Arabic aliases; expected to land in market_name_ar, not an error"},
            "market_name_variant": {"rule_id": None, "market_ids": sorted(mkt["market_id"][variant_markets].tolist()),
                                    "note": "case/whitespace variants; must normalize to one name, not an error"},
            "market_name_conflict": {"rule_id": "market.conflicting_name",
                                     "market_ids": sorted(mkt["market_id"][conflict_markets].tolist()),
                                     "rows": sorted(int(pos[r]) for r in conflict_market_rows)},
            "product_attribute_conflict": {"rule_id": "product.conflicting_attribute", "column": "brand",
                                           "product_ids": sorted({prod["product_id"][p[r]] for r in conflict_product_rows}),
                                           "rows": sorted(int(pos[r]) for r in conflict_product_rows)},
        },
    }
    blank_to_null = [pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
                     for c in ["promo_id", "promo_type", "promo_start", "promo_end"]]
    return full.drop("_gid").with_columns(blank_to_null).rename(RAW_HEADER), truth


def _summary_count(issue: dict) -> str:
    if "count" in issue:
        return str(issue["count"])
    if "counts" in issue:
        return ", ".join(f"{k}={v}" for k, v in issue["counts"].items())
    if "by_column" in issue:
        return str(sum(c["count"] for c in issue["by_column"].values()))
    ids = issue.get("market_ids") or issue.get("product_ids") or issue.get("promo_ids") or []
    return f"{len(ids)} ids"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/raw/sellout.csv"))
    ap.add_argument("--truth", type=Path, default=None, help="default: ground_truth.json next to --out")
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("--encoding", default="utf-8", help="utf-8 | utf-8-sig | cp1256")
    args = ap.parse_args()

    t0 = time.perf_counter()
    df, truth = make(args.rows, args.seed, Rates())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = df.write_csv(separator=args.delimiter)
    args.out.write_bytes(text.encode(args.encoding))
    truth["file"] = {"path": str(args.out), "delimiter": args.delimiter, "encoding": args.encoding}
    truth_path = args.truth or args.out.with_name("ground_truth.json")
    truth_path.write_text(json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {k: _summary_count(v) for k, v in truth["issues"].items()}
    print(f"wrote {df.height:,} rows -> {args.out} in {time.perf_counter() - t0:.1f}s")
    print(f"ground truth -> {truth_path}")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")


if __name__ == "__main__":
    main()
