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
