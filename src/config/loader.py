"""
Loads raw rent roll rows from any supported source format, remaps field names,
coerces types, and validates through the LeaseRecord Pydantic model before
data reaches the calculation layer.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .schemas import LeaseRecord

_CONFIG_PATH = Path(__file__).parent / "mapping_config.json"


def load_mapping_config(path: Path = _CONFIG_PATH) -> dict:
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _map_fields(raw: dict[str, Any], field_map: dict[str, str]) -> dict[str, Any]:
    """Remap source-specific column names to core schema names.

    Fields whose source column is absent from the row are silently skipped;
    Pydantic will apply the schema default for any optional field, or raise a
    validation error if the field is truly required.
    """
    mapped: dict[str, Any] = {}
    for core_field, source_field in field_map.items():
        if source_field not in raw:
            continue   # optional field missing from this source — let Pydantic use default
        mapped[core_field] = raw[source_field]
    return mapped


def _coerce_bool(value: Any, transforms: dict) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    s = str(value).strip()
    if s in transforms.get("true_values", []):
        return True
    if s in transforms.get("false_values", []):
        return False
    raise ValueError(
        f"Cannot coerce '{value}' to bool. "
        f"Expected one of {transforms.get('true_values')} or {transforms.get('false_values')}."
    )


def _coerce_date(value: Any, date_format: str):
    if hasattr(value, "year"):  # already a date/datetime
        return value if not hasattr(value, "hour") else value.date()
    return datetime.strptime(str(value).strip(), date_format).date()


def _coerce_types(
    row: dict[str, Any],
    date_format: str,
    value_transforms: dict[str, dict],
) -> dict[str, Any]:
    date_fields = {"lease_start", "lease_end"}
    coerced = dict(row)

    for field in date_fields:
        if field in coerced:
            coerced[field] = _coerce_date(coerced[field], date_format)

    if "is_delinquent" in coerced:
        transforms = value_transforms.get("is_delinquent", {})
        coerced["is_delinquent"] = _coerce_bool(coerced["is_delinquent"], transforms)

    for numeric_field in ("square_footage",):
        if numeric_field in coerced:
            coerced[numeric_field] = int(float(str(coerced[numeric_field]).replace(",", "")))

    for float_field in ("base_rent_psf",):
        if float_field in coerced:
            coerced[float_field] = float(str(coerced[float_field]).replace(",", "").replace("$", ""))

    return coerced


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_lease_record(
    source_name: str,
    raw_row: dict[str, Any],
    config: dict | None = None,
) -> LeaseRecord:
    """
    Full pipeline for a single row: field remapping → type coercion → Pydantic validation.
    Raises KeyError on missing fields, ValueError on bad values, ValidationError on
    constraint violations.
    """
    if config is None:
        config = load_mapping_config()

    sources = config.get("sources", {})
    if source_name not in sources:
        raise ValueError(
            f"Unknown source '{source_name}'. Registered sources: {list(sources.keys())}"
        )

    source_cfg   = sources[source_name]
    field_map    = source_cfg["field_map"]
    date_format  = source_cfg.get("date_format", "%Y-%m-%d")
    transforms   = source_cfg.get("value_transforms", {})

    mapped  = _map_fields(raw_row, field_map)
    coerced = _coerce_types(mapped, date_format, transforms)
    return LeaseRecord(**coerced)


@dataclass
class BatchResult:
    valid:   list[LeaseRecord]
    errors:  list[dict[str, Any]]   # {"row_index": int, "raw": dict, "error": str}

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def success_rate(self) -> float:
        total = len(self.valid) + len(self.errors)
        return len(self.valid) / total if total else 0.0


def validate_batch(
    source_name: str,
    rows: list[dict[str, Any]],
    config: dict | None = None,
) -> BatchResult:
    """
    Validate a list of raw rows. Returns a BatchResult with all valid LeaseRecords
    and a structured list of per-row errors — never raises.
    """
    if config is None:
        config = load_mapping_config()

    valid:  list[LeaseRecord]     = []
    errors: list[dict[str, Any]]  = []

    for idx, row in enumerate(rows):
        try:
            valid.append(parse_lease_record(source_name, row, config))
        except (KeyError, ValueError, ValidationError) as exc:
            errors.append({"row_index": idx, "raw": row, "error": str(exc)})

    return BatchResult(valid=valid, errors=errors)
