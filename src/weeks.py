"""ISO-week normalization as Polars expressions.

Accepted inputs (after trimming):
  2025-W03, 2025W03, 2025-w3   -> ISO year/week
  W03-2025, W3 2025            -> ISO year/week
  202503                       -> yyyyww (ISO year 2025, week 03)
  any date in rules.week.date_formats (e.g. 2025-01-15, 15/01/2025) -> Monday of that ISO week
"""

from __future__ import annotations

import polars as pl

_YEAR_WEEK = r"^(\d{4})-?[Ww](\d{1,2})$"
_WEEK_YEAR = r"^[Ww](\d{1,2})[-\s/]?(\d{4})$"
_YYYYWW = r"^(\d{4})(\d{2})$"


def iso_monday(year: pl.Expr, week: pl.Expr) -> pl.Expr:
    """Monday of ISO (year, week). Null if the week does not exist in that ISO year."""
    jan4 = pl.date(year, 1, 4)
    week1_monday = jan4 - pl.duration(days=jan4.dt.weekday() - 1)
    monday = week1_monday + pl.duration(weeks=week - 1)
    valid = (week >= 1) & (week <= 53) & (monday.dt.iso_year() == year)
    return pl.when(valid).then(monday)


def to_monday(date: pl.Expr) -> pl.Expr:
    return date - pl.duration(days=date.dt.weekday() - 1)


def week_start_expr(col: str, date_formats: list[str]) -> pl.Expr:
    s = pl.col(col).str.strip_chars()

    def grp(pattern: str, i: int) -> pl.Expr:
        return s.str.extract(pattern, i).cast(pl.Int32, strict=False)

    dates = [to_monday(s.str.strptime(pl.Date, fmt, strict=False)) for fmt in date_formats]
    return pl.coalesce(
        iso_monday(grp(_YEAR_WEEK, 1), grp(_YEAR_WEEK, 2)),
        iso_monday(grp(_WEEK_YEAR, 2), grp(_WEEK_YEAR, 1)),
        iso_monday(grp(_YYYYWW, 1), grp(_YYYYWW, 2)),
        *dates,
    ).alias("week_start_date")


def is_canonical_week(col: str) -> pl.Expr:
    """True if the raw value already uses the canonical 'YYYY-Www' form."""
    return pl.col(col).str.contains(r"^\d{4}-W\d{2}$").fill_null(False)
