"""The Streamlit app in hosted mode (e.g. a Hugging Face Space): visitors never share files or AI settings."""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import env
from src.agent import providers

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def hosted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    for p in providers.PROVIDERS.values():
        for var in (p.key_var, p.model_var, p.url_var):
            if var:
                monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv(providers.PROVIDER_VAR, raising=False)
    monkeypatch.setenv("SELLOUT_HOSTED", "1")
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-host-shared-key-0001")
    monkeypatch.setattr(env.load_env, "__defaults__", (tmp_path / "no.env", False))  # never read the real .env
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(providers, "check", lambda p, key=None, base_url=None: ("ok", "ok", ["model-a"]))
    yield tmp_path
    providers.use_session(None)


def visit() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def test_visitors_get_private_workspaces_and_the_shared_key(hosted: Path) -> None:
    a, b = visit(), visit()
    ws_a, ws_b = Path(a.session_state["workspace"]), Path(b.session_state["workspace"])
    assert ws_a != ws_b and ws_a.parent == ws_b.parent == hosted / "sellout-workspaces"
    assert (ws_a / "schema_mapping.yaml").exists() and (ws_a / "raw").is_dir()

    picker = a.sidebar.selectbox(key="side_prov")
    assert "ollama" not in picker.options and "custom" not in picker.options  # no server-side local/URL calls
    assert any("shared key from the host" in m.value for m in a.sidebar.markdown)
    assert a.session_state["ai_settings"] == {}  # nothing of the host's key copied into the visitor's session


def test_old_workspaces_are_removed(hosted: Path) -> None:
    import os
    import time

    old = hosted / "sellout-workspaces" / "old"
    old.mkdir(parents=True)
    os.utime(old, (time.time() - 3 * 86400,) * 2)
    visit()
    assert not old.exists()


def test_unfamiliar_columns_get_a_friendly_box_and_ai_mapping(hosted: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(APP).parent))
    from scripts.make_challenge_data import RENAME, build
    from src.agent import graph, tools

    df, _ = build(600, 3)
    df = df.rename({"Item #": "Fld_07"})  # one header no name rule knows: the Profiler (AI) is needed
    csv = hosted / "challenge.csv"
    csv.write_bytes(df.write_csv(separator=";").encode("utf-8-sig"))
    standard = {"SKU Code": "product_id", "Product Description": "product_name", "Brand": "brand", "Category": "category",
                "Pack Size": "pack_size", "UoM": "uom", "Units/Case": "units_per_case", "Store Code": "market_id",
                "Store Name": "market_name", "Channel": "channel", "Region": "region", "Promo ID": "promo_id",
                "Promo Type": "promo_type", "Promo Start": "promo_start_date", "Promo End": "promo_end_date",
                "Week": "week_raw", "Sales Qty": "units", "Sales Value (EGP)": "value"}
    answer = {RENAME[old]: std for old, std in standard.items()}
    answer["Fld_07"] = answer.pop("Item #")
    monkeypatch.setattr(providers, "make_llm", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(graph, "propose_mapping", lambda s, profile, cols: tools.MappingProposal(answer, [], ""))

    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["file"] = str(csv)
    at.run()
    assert not at.exception, at.exception
    assert any("don't match the saved mapping" in m.value for m in at.markdown)
    assert any("First tell the tool" in i.value for i in at.info)  # clean tab explains instead of failing
    next(b for b in at.button if b.label == "🧭 Map columns with AI").click().run()
    assert not at.exception, at.exception
    assert not any("don't match the saved mapping" in m.value for m in at.markdown)
    assert any(b.label == "Run cleaning" for b in at.button)


def test_known_header_names_are_mapped_without_ai(hosted: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    sys.path.insert(0, str(Path(APP).parent))
    from scripts.make_challenge_data import build
    from src.agent import graph

    def no_ai(*a: object, **k: object) -> None:
        raise AssertionError("AI must not be called for known header names")

    monkeypatch.setattr(graph, "propose_mapping", no_ai)
    df, _ = build(600, 3)
    csv = hosted / "challenge.csv"
    csv.write_bytes(df.write_csv(separator=";").encode("utf-8-sig"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["file"] = str(csv)
    at.run()
    assert not at.exception, at.exception
    assert not any("don't match the saved mapping" in m.value for m in at.markdown)
    assert any(b.label == "Run cleaning" for b in at.button)
    next(b for b in at.button if b.label == "Run cleaning").click().run()  # results view in a hosted workspace
    assert not at.exception, at.exception
    assert any("Full files are in the Download tab" in c.value for c in at.caption)
