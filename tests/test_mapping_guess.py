"""Header matching by known names: typical files are mapped by code, with no AI call."""

from __future__ import annotations

from pathlib import Path

from scripts.make_challenge_data import ORDER, RENAME
from scripts.make_dirty_data import RAW_HEADER
from src.agent import tools
from src.config import load_mapping

SAVED = load_mapping(Path(__file__).resolve().parents[1] / "config" / "schema_mapping.yaml").columns  # the hand-checked mapping for the standard export


def test_standard_and_renamed_headers_are_matched_without_ai() -> None:
    assert tools.guess_mapping(list(RAW_HEADER.values())) == SAVED
    challenge = tools.guess_mapping(ORDER)  # 'Item #', 'Cat.', 'Net Sales EGP', ... plus Remarks / Extract Date
    assert challenge is not None and "Remarks" not in challenge and "Extract Date" not in challenge
    assert challenge == {RENAME[raw]: std for raw, std in SAVED.items()}


def test_unknown_or_ambiguous_headers_are_left_to_the_profiler() -> None:
    assert tools.guess_mapping([*RAW_HEADER.values(), "Fld_07"]) is None             # unknown column: ask the AI
    assert tools.guess_mapping([*RAW_HEADER.values(), "Quantity"]) is None           # units claimed twice
    assert tools.guess_mapping(["Item", "Store Code", "Week", "Qty"]) is None         # sales value missing
