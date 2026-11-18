"""Persistent memory of human decisions on risky fixes, so the same question is not asked twice."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


def signature(name: str, params: dict[str, Any]) -> str:
    return f"{name}:{json.dumps(params, sort_keys=True, ensure_ascii=False)}"


class Memory:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.data: dict[str, Any] = {"decisions": {}, "plans": {}}
        if path and path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            self.data.setdefault("decisions", {})
            self.data.setdefault("plans", {})

    def decision(self, name: str, params: dict[str, Any]) -> bool | None:
        hit = self.data["decisions"].get(signature(name, params))
        return None if hit is None else bool(hit["approved"])

    def remember(self, name: str, params: dict[str, Any], approved: bool, by: str) -> None:
        self.data["decisions"][signature(name, params)] = {
            "approved": approved, "by": by, "at": dt.datetime.now().isoformat(timespec="seconds"),
        }
        self.save()

    def plan(self, fingerprint: str) -> list[dict[str, Any]] | None:
        hit = self.data["plans"].get(fingerprint)
        return None if hit is None else list(hit["ops"])

    def remember_plan(self, fingerprint: str, ops: list[dict[str, Any]], keep: int = 20) -> None:
        plans = self.data["plans"]
        plans[fingerprint] = {"ops": ops, "at": dt.datetime.now().isoformat(timespec="seconds")}
        for old in sorted(plans, key=lambda k: plans[k]["at"])[:-keep]:
            del plans[old]
        self.save()

    def save(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
