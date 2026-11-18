"""Load settings such as ANTHROPIC_API_KEY from the project's .env file (never hard-coded)."""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
KEY = "ANTHROPIC_API_KEY"


def parse_env(text: str) -> dict[str, str]:
    """KEY=VALUE lines; blank lines and # comments ignored; optional surrounding quotes stripped."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[name.strip().removeprefix("export ").strip()] = value
    return out


def load_env(path: Path = ENV_FILE, override: bool = False) -> dict[str, str]:
    """Copy non-empty values from .env into os.environ. A key already set in the shell wins unless override."""
    if not path.exists():
        return {}
    values = {k: v for k, v in parse_env(path.read_text(encoding="utf-8")).items() if v}
    for k, v in values.items():
        if override or not os.environ.get(k):
            os.environ[k] = v
    return values


def api_key_status() -> str:
    """'env_file', 'shell' or 'missing': where the key comes from, without ever exposing it."""
    if load_env_values().get(KEY):
        return "env_file"
    return "shell" if os.environ.get(KEY) else "missing"


def load_env_values(path: Path = ENV_FILE) -> dict[str, str]:
    return {k: v for k, v in parse_env(path.read_text(encoding="utf-8")).items() if v} if path.exists() else {}


def validate_key(key: str) -> str:
    key = key.strip()
    if not key or not key.isprintable() or any(c.isspace() for c in key):
        raise ValueError("That does not look like an API key (empty, or contains spaces or line breaks).")
    return key


def use_key(key: str) -> None:
    """Make the key active for this process only (nothing is written to disk)."""
    os.environ[KEY] = validate_key(key)


def _name(line: str) -> str:
    return line.strip().removeprefix("export ").split("=")[0].strip()


def set_values(values: dict[str, str], persist: bool = True, path: Path = ENV_FILE) -> None:
    """Make settings active now; with persist, also write them to .env (owner-only), keeping other lines."""
    for k, v in values.items():
        os.environ[k] = v
    if not persist:
        return
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    kept = [ln for ln in lines if _name(ln) not in values]
    path.write_text("\n".join([*kept, *(f"{k}={v}" for k, v in values.items())]) + "\n", encoding="utf-8")
    path.chmod(0o600)


def clear_value(name: str, path: Path = ENV_FILE) -> None:
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([*(ln for ln in lines if _name(ln) != name), f"{name}="]) + "\n", encoding="utf-8")
    os.environ.pop(name, None)


def save_key(key: str, path: Path = ENV_FILE) -> None:
    """Write the Anthropic key to .env (owner-only permissions) and make it active now."""
    set_values({KEY: validate_key(key)}, path=path)


def remove_key(path: Path = ENV_FILE) -> None:
    clear_value(KEY, path)


def masked_key() -> str:
    """Enough to recognise which key is active, never the key itself."""
    k = os.environ.get(KEY, "")
    return f"{k[:7]}...{k[-4:]}" if len(k) >= 20 else "(set)"


def check_key() -> tuple[str, str]:
    """Ask the API whether the active key is accepted: ("ok" | "rejected" | "unreachable" | "error", message).

    Free: it lists models and generates nothing.
    """
    import anthropic

    try:
        anthropic.Anthropic().models.list(limit=1)
    except anthropic.AuthenticationError:
        return "rejected", "The API rejected this key. Check that you copied all of it."
    except anthropic.APIConnectionError:
        return "unreachable", "Could not reach the Anthropic API. Check your internet connection or proxy."
    except Exception as e:  # noqa: BLE001 - report anything else as-is
        return "error", f"{type(e).__name__}: {e}"
    return "ok", "Key works."
