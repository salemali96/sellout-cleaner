from __future__ import annotations

import codecs

import polars as pl
import pytest

from src.ingest import ROW_NR, detect_delimiter, detect_encoding, scan_raw
from tests.conftest import raw_row

ENCODINGS = ["utf-8", "utf-8-sig", "cp1256", "latin-1"]


def test_detect_encoding_utf8() -> None:
    assert detect_encoding("كارفور,Maadi".encode("utf-8"), ENCODINGS) == "utf-8"


def test_detect_encoding_bom() -> None:
    assert detect_encoding(codecs.BOM_UTF8 + b"a,b\n1,2\n", ENCODINGS) == "utf-8-sig"


def test_detect_encoding_cp1256() -> None:
    assert detect_encoding("ميجا مارت المعادي,1\n".encode("cp1256"), ENCODINGS) == "cp1256"


def test_detect_encoding_tolerates_cut_multibyte_char() -> None:
    sample = "ميجا".encode("utf-8")[:-1]  # sample boundary in the middle of a 2-byte char
    assert detect_encoding(sample, ENCODINGS) == "utf-8"


@pytest.mark.parametrize("delim", [",", ";", "\t", "|"])
def test_detect_delimiter(delim: str) -> None:
    text = "\n".join(delim.join(["a", "b", "c", "d"]) for _ in range(5))
    assert detect_delimiter(text, [",", ";", "\t", "|"]) == delim


def test_detect_delimiter_ignores_commas_inside_quotes() -> None:
    text = 'id;name;value\n1;"Milk, 1L";10,5\n2;"Juice, 2L";11,0\n'
    assert detect_delimiter(text, [",", ";"]) == ";"


@pytest.mark.parametrize(("delim", "enc"), [(",", "utf-8"), (";", "cp1256"), ("\t", "utf-8-sig"), ("|", "utf-8")])
def test_scan_raw_roundtrip(make_csv, rules, delim: str, enc: str) -> None:
    rows = [raw_row(store_name="ميجا مارت المعادي"), raw_row(sku="SKU2", units="N/A")]
    path = make_csv(rows, delimiter=delim, encoding=enc)
    lf, fmt = scan_raw(path, rules.ingest, work_dir=path.parent / "cache")
    df = lf.collect()
    assert isinstance(lf, pl.LazyFrame)
    assert (fmt.delimiter, fmt.encoding) == (delim, enc)
    assert df.columns[0] == ROW_NR and "SKU Code" in df.columns  # BOM not glued to first header
    assert df[ROW_NR].to_list() == [0, 1]
    assert df["Store Name"][0] == "ميجا مارت المعادي"
    assert all(t == pl.String for c, t in df.schema.items() if c != ROW_NR)
    assert df["Promo ID"][0] == ""  # empty kept as "" for placeholder checks
