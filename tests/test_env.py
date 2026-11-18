from pathlib import Path

import pytest

from src import env


def test_parse_env_handles_comments_quotes_and_export() -> None:
    text = '# c\n\nANTHROPIC_API_KEY="sk-1"\nexport OTHER = x \nBROKEN\nEMPTY=\n'
    assert env.parse_env(text) == {"ANTHROPIC_API_KEY": "sk-1", "OTHER": "x", "EMPTY": ""}


def test_load_env_sets_missing_and_keeps_shell_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / ".env"
    p.write_text("A_KEY=from_file\nB_KEY=from_file\nC_KEY=\n")
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.setenv("B_KEY", "from_shell")
    monkeypatch.delenv("C_KEY", raising=False)
    env.load_env(p)
    import os
    assert os.environ["A_KEY"] == "from_file" and os.environ["B_KEY"] == "from_shell" and "C_KEY" not in os.environ
    env.load_env(p, override=True)
    assert os.environ["B_KEY"] == "from_file"
