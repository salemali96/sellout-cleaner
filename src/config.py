"""Pydantic models for config/rules.yaml and config/schema_mapping.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


class IngestRules(BaseModel):
    sample_bytes: int = 1_048_576
    candidate_delimiters: list[str] = [",", ";", "\t", "|"]
    candidate_encodings: list[str] = ["utf-8", "utf-8-sig", "cp1256", "latin-1"]


class WeekRules(BaseModel):
    start_day: Literal["monday"] = "monday"
    six_digit_format: Literal["yyyyww"] = "yyyyww"
    date_formats: list[str] = ["%Y-%m-%d", "%d/%m/%Y"]


class DedupRules(BaseModel):
    conflict_strategy: Literal["quarantine_both", "keep_first", "keep_last"] = "quarantine_both"


class PriceRules(BaseModel):
    outlier_tolerance: float = 0.5
    min_rows_for_median: int = 3


class GapRules(BaseModel):
    enabled: bool = True


class CasePackRules(BaseModel):
    case_uom_values: list[str] = ["CS", "CASE", "CTN"]
    unit_uom_values: list[str] = ["EA", "PC", "UNIT", "PCS"]


class MarketRules(BaseModel):
    arabic_regex: str = "[؀-ۿ]"


class FixRules(BaseModel):
    safe: list[str] = []
    risky: list[str] = []


class Rules(BaseModel):
    ingest: IngestRules = IngestRules()
    null_placeholders: list[str] = ["", "N/A", "NA", "-", "NULL"]
    optional_columns: list[str] = []
    required_columns: list[str] = ["product_id", "market_id", "week_raw", "units", "value"]
    week: WeekRules = WeekRules()
    dedup: DedupRules = DedupRules()
    price: PriceRules = PriceRules()
    gaps: GapRules = GapRules()
    case_pack: CasePackRules = CasePackRules()
    market: MarketRules = MarketRules()
    severity: dict[str, Literal["error", "warning"]] = Field(default_factory=dict)
    enabled_rules: list[str] = Field(default_factory=list)
    fixes: FixRules = FixRules()

    def severity_of(self, rule_id: str) -> str:
        return self.severity.get(rule_id, "warning")

    def is_enabled(self, rule_id: str) -> bool:
        return rule_id in self.enabled_rules

    @property
    def placeholder_set(self) -> list[str]:
        return sorted({p.strip().lower() for p in self.null_placeholders})


class TableSpec(BaseModel):
    key: list[str]
    attributes: list[str] = []
    measures: list[str] = []
    foreign_keys: list[str] = []


class SchemaMapping(BaseModel):
    columns: dict[str, str]
    tables: dict[str, TableSpec]

    @property
    def standard_columns(self) -> list[str]:
        return list(dict.fromkeys(self.columns.values()))


def load_rules(path: Path | None = None) -> Rules:
    return Rules.model_validate(yaml.safe_load((path or CONFIG_DIR / "rules.yaml").read_text(encoding="utf-8")))


def load_mapping(path: Path | None = None) -> SchemaMapping:
    return SchemaMapping.model_validate(
        yaml.safe_load((path or CONFIG_DIR / "schema_mapping.yaml").read_text(encoding="utf-8"))
    )
