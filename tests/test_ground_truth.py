"""End-to-end: generate dirty data with known issues and require every one to be detected exactly."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.make_dirty_data import Rates, make
from src import evaluate, pipeline


@pytest.mark.parametrize("seed", [1, 7])
def test_detections_match_ground_truth(tmp_path: Path, rules, mapping, seed: int) -> None:
    df, truth = make(20_000, seed, Rates())
    path = tmp_path / "sellout.csv"
    df.write_csv(path)
    prepared = pipeline.prepare(path, rules, mapping)
    issues = pipeline.check(prepared, rules).issues
    scores = evaluate.compare(issues, truth, prepared.ctx.tables["market"])
    bad = scores.filter((scores["precision"] < 1) | (scores["recall"] < 1))
    assert bad.height == 0, bad
    assert scores["expected"].sum() > 0
