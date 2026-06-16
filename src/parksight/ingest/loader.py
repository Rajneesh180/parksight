from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from parksight import config

_SOURCE_COLUMNS = [
    "id",
    "latitude",
    "longitude",
    "location",
    "vehicle_number",
    "vehicle_type",
    "violation_type",
    "created_datetime",
    "junction_name",
    "police_station",
    "device_id",
    "created_by_id",
    "validation_status",
]


def _parse_labels(raw: object) -> list[str]:
    if not isinstance(raw, str):
        return []
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def _severity(labels: list[str]) -> float:
    return max((config.SEVERITY.get(label, 0.0) for label in labels), default=0.0)


def _tokenize(plates: pd.Series, salt: str) -> pd.Series:
    def digest(value: object) -> str:
        return hashlib.sha256(f"{salt}:{value}".encode()).hexdigest()[:16]

    return plates.map(digest)


def load_violations(
    path: Path | str | None = None, *, anonymize: bool = True
) -> pd.DataFrame:
    source = Path(path) if path is not None else config.RAW_DATA_PATH
    frame = pd.read_csv(source, usecols=_SOURCE_COLUMNS, dtype="string")

    latitude = pd.to_numeric(frame["latitude"], errors="coerce")
    longitude = pd.to_numeric(frame["longitude"], errors="coerce")
    inside = latitude.between(*config.LAT_BOUNDS) & longitude.between(*config.LON_BOUNDS)

    frame = frame.loc[inside].copy()
    frame["latitude"] = latitude[inside]
    frame["longitude"] = longitude[inside]

    logged_at = pd.to_datetime(frame["created_datetime"], utc=True, errors="coerce")
    local = logged_at.dt.tz_convert(config.IST)
    frame["logged_at"] = logged_at
    frame["date"] = local.dt.date.astype("string")
    frame["hour"] = local.dt.hour

    frame["labels"] = frame["violation_type"].map(_parse_labels)
    frame["severity"] = frame["labels"].map(_severity)
    frame["is_parking"] = frame["labels"].map(
        lambda labels: any("PARKING" in label for label in labels)
    )

    frame["cell"] = (
        frame["latitude"].round(config.CELL_DECIMALS).astype("string")
        + ","
        + frame["longitude"].round(config.CELL_DECIMALS).astype("string")
    )

    if anonymize:
        frame["vehicle_token"] = _tokenize(frame["vehicle_number"], config.SALT)
        frame = frame.drop(columns=["vehicle_number"])

    return frame.drop(columns=["violation_type"]).reset_index(drop=True)
