from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from parksight import config

RELOCATION_THRESHOLD = 0.5


def _mode(values: pd.Series) -> object:
    modes = values.mode()
    return modes.iat[0] if not modes.empty else pd.NA


def _continuity(devices: Iterable[object], elsewhere: set[object]) -> float:
    present = [device for device in devices if pd.notna(device)]
    if not present:
        return float("nan")
    return sum(device in elsewhere for device in present) / len(present)


def _classify(
    frame: pd.DataFrame, split: str, appear: int, vanish: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    half = np.where((frame["date"] < split).fillna(False), "h1", "h2")
    tagged = frame.assign(half=half)

    counts = tagged.groupby(["cell", "half"]).size().unstack(fill_value=0)
    h1 = counts["h1"] if "h1" in counts else pd.Series(0, index=counts.index)
    h2 = counts["h2"] if "h2" in counts else pd.Series(0, index=counts.index)

    status = pd.Series("other", index=counts.index, dtype="object")
    status[(h1 >= 5) & (h2 >= 5)] = "persistent"
    status[(h1 <= vanish) & (h2 >= appear)] = "emerging"
    status[(h1 >= appear) & (h2 <= vanish)] = "declining"
    return tagged, status, h1, h2


def transitions(
    frame: pd.DataFrame,
    split: str = config.CHURN_SPLIT,
    appear: int = 15,
    vanish: int = 2,
) -> pd.DataFrame:
    tagged, status, h1, h2 = _classify(frame, split, appear, vanish)

    geo = tagged.groupby("cell").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        station=("police_station", _mode),
    )

    cells = pd.DataFrame({"h1": h1, "h2": h2}).join(geo)
    cells["delta"] = cells["h2"] - cells["h1"]
    cells["status"] = status

    active = {key: set(part["device_id"].dropna()) for key, part in tagged.groupby("half")}
    devices_in = {
        key: part.groupby("cell")["device_id"].agg(lambda s: set(s.dropna()))
        for key, part in tagged.groupby("half")
    }

    cells["enforcement_continuity"] = np.nan
    for cell in cells.index[status == "emerging"]:
        seen = devices_in.get("h2", {}).get(cell, set())
        cells.loc[cell, "enforcement_continuity"] = _continuity(seen, active.get("h1", set()))
    for cell in cells.index[status == "declining"]:
        seen = devices_in.get("h1", {}).get(cell, set())
        cells.loc[cell, "enforcement_continuity"] = _continuity(seen, active.get("h2", set()))

    return cells.sort_values("delta", key=lambda s: s.abs(), ascending=False)


def relocation_labels(continuity: pd.Series) -> pd.Series:
    labels = pd.Series("genuine signal", index=continuity.index, dtype="object")
    labels[continuity >= RELOCATION_THRESHOLD] = "likely relocation"
    labels[continuity.isna()] = "unclassified"
    return labels


def _empty_relocation() -> dict:
    nan = float("nan")
    return {
        "emerging_cells": 0,
        "device_base_rate": nan,
        "observed_continuity": nan,
        "null_continuity": nan,
        "null_ci95": [nan, nan],
        "lift": nan,
        "p_value": nan,
        "genuine_emerging": 0,
    }


def relocation_report(
    frame: pd.DataFrame,
    split: str = config.CHURN_SPLIT,
    appear: int = 15,
    vanish: int = 2,
    iterations: int = 2000,
    seed: int = 0,
    identity: str = "device_id",
) -> dict:
    tagged, status, _, _ = _classify(frame, split, appear, vanish)
    active_h1 = set(tagged.loc[tagged["half"] == "h1", identity].dropna())
    pool = np.array(sorted(set(tagged.loc[tagged["half"] == "h2", identity].dropna())))
    if pool.size == 0 or not (status == "emerging").any():
        return _empty_relocation()

    veteran = np.array([device in active_h1 for device in pool])
    devices_h2 = tagged[tagged["half"] == "h2"].groupby("cell")[identity].agg(
        lambda series: set(series.dropna())
    )

    observed, sizes = [], []
    for cell in status.index[status == "emerging"]:
        present = [device for device in devices_h2.get(cell, set()) if pd.notna(device)]
        if present:
            observed.append(np.mean([device in active_h1 for device in present]))
            sizes.append(len(present))
    observed = np.array(observed)
    sizes = np.array(sizes)

    generator = np.random.default_rng(seed)
    null = np.array(
        [
            np.mean([veteran[generator.choice(pool.size, size=k, replace=False)].mean() for k in sizes])
            for _ in range(iterations)
        ]
    )
    low, high = np.percentile(null, [2.5, 97.5])
    return {
        "emerging_cells": int(observed.size),
        "device_base_rate": float(veteran.mean()),
        "observed_continuity": float(observed.mean()),
        "null_continuity": float(null.mean()),
        "null_ci95": [float(low), float(high)],
        "lift": float(observed.mean() - null.mean()),
        "p_value": float((null >= observed.mean()).mean()),
        "genuine_emerging": int((observed < RELOCATION_THRESHOLD).sum()),
    }
