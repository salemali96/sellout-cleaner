"""Encoding/delimiter detection and lazy reading of the raw CSV.

Everything is read as String (no type inference) so that dirty values are preserved for checks.
Each row gets `row_nr` = 0-based data-row index, used as the lineage key everywhere downstream.
"""

from __future__ import annotations

import codecs
import csv
import io
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.config import IngestRules

ROW_NR = "row_nr"
_CHUNK = 8 * 1024 * 1024


@dataclass(frozen=True)
class FileFormat:
    encoding: str
    delimiter: str


def read_sample(path: Path, n_bytes: int) -> bytes:
    with path.open("rb") as fh:
        return fh.read(n_bytes)


def detect_encoding(sample: bytes, candidates: list[str]) -> str:
    """First candidate that decodes the sample cleanly. A UTF-8 BOM always wins."""
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    for enc in candidates:
        try:
            decoder = codecs.getincrementaldecoder(enc)()
            decoder.decode(sample, final=False)  # tolerate a multibyte char cut at the sample edge
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "latin-1"


def detect_delimiter(text: str, candidates: list[str], max_lines: int = 50) -> str:
    """Pick the candidate giving the most columns with a constant column count across lines."""
    lines = [ln for ln in text.splitlines()[:max_lines] if ln.strip()]
    if len(lines) > 1:
        lines = lines[:-1]  # last line may be cut by the sample boundary
    best, best_score = candidates[0], (-1, -1)
    for delim in candidates:
        widths = [len(row) for row in csv.reader(io.StringIO("\n".join(lines)), delimiter=delim)]
        if not widths or widths[0] < 2:
            continue
        consistent = sum(w == widths[0] for w in widths)
        score = (consistent, widths[0])
        if score > best_score:
            best, best_score = delim, score
    return best


def detect_format(path: Path, rules: IngestRules) -> FileFormat:
    sample = read_sample(path, rules.sample_bytes)
    encoding = detect_encoding(sample, rules.candidate_encodings)
    text = codecs.getincrementaldecoder(encoding)(errors="replace").decode(sample, final=False)
    return FileFormat(encoding=encoding, delimiter=detect_delimiter(text, rules.candidate_delimiters))


def to_utf8(path: Path, encoding: str, work_dir: Path) -> Path:
    """Polars only reads UTF-8: stream-transcode other encodings in chunks (never fully in memory)."""
    if encoding in ("utf-8", "utf8"):
        return path
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{path.stem}.utf8{path.suffix}"
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    with path.open("rb") as src, out.open("w", encoding="utf-8", newline="") as dst:
        while chunk := src.read(_CHUNK):
            dst.write(decoder.decode(chunk))
        dst.write(decoder.decode(b"", final=True))
    return out


def scan_raw(path: Path, rules: IngestRules, work_dir: Path | None = None) -> tuple[pl.LazyFrame, FileFormat]:
    """Lazily scan the raw CSV as all-String columns with a `row_nr` lineage column."""
    fmt = detect_format(path, rules)
    utf8_path = to_utf8(path, fmt.encoding, work_dir or path.parent / ".cache")
    lf = pl.scan_csv(
        utf8_path,
        separator=fmt.delimiter,
        infer_schema=False,
        empty_string_is_null=False,  # keep "" so placeholder checks see it; nulls normalized later
        row_index_name=ROW_NR,
    )
    return lf, fmt


def raw_columns(lf: pl.LazyFrame) -> list[str]:
    return [c for c in lf.collect_schema().names() if c != ROW_NR]
