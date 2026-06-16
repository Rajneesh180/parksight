from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from parksight.features.disruption import flow_disruption


@dataclass(frozen=True)
class PriorityWeights:
    volume: float = 0.4
    severity: float = 0.3
    disruption: float = 0.2
    persistence: float = 0.1


def _unit_scale(values: pd.Series) -> pd.Series:
    low, high = values.min(), values.max()
    if high == low:
        return pd.Series(0.0, index=values.index)
    return (values - low) / (high - low)


def _dominant_station(stations: pd.Series) -> object:
    modes = stations.mode()
    return modes.iat[0] if not modes.empty else pd.NA


def blend_priority(cells: pd.DataFrame, weights: PriorityWeights | None = None) -> pd.Series:
    weights = weights or PriorityWeights()
    return (
        weights.volume * _unit_scale(np.log1p(cells["volume"]))
        + weights.severity * cells["severity"]
        + weights.disruption * cells["disruption"]
        + weights.persistence * cells["persistence"]
    )


def rank_hotspots(frame: pd.DataFrame, weights: PriorityWeights | None = None) -> pd.DataFrame:
    enriched = frame.assign(disruption=flow_disruption(frame["location"]))
    span_days = enriched["date"].nunique()

    grouped = enriched.groupby("cell")
    cells = pd.DataFrame(
        {
            "volume": grouped.size(),
            "severity": grouped["severity"].mean(),
            "disruption": grouped["disruption"].mean(),
            "active_days": grouped["date"].nunique(),
            "latitude": grouped["latitude"].mean(),
            "longitude": grouped["longitude"].mean(),
            "station": grouped["police_station"].agg(_dominant_station),
        }
    )
    cells["persistence"] = cells["active_days"] / span_days
    cells["priority"] = blend_priority(cells, weights)
    return cells.sort_values("priority", ascending=False)
