from __future__ import annotations

import numpy as np
import pandas as pd

LIGHT_VEHICLES = {"SCOOTER", "MOTOR CYCLE", "MOPED", "BICYCLE", "CYCLE"}
EXPOSURES = ("device_days", "active_days", "devices")
INVALID_STATUS = {"rejected", "duplicate"}


def _week_index(dates: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(dates, errors="coerce")
    origin = parsed.min()
    return ((parsed - origin).dt.days // 7).astype("Int64")


def junction_events(frame: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    junction = frame["junction_name"].fillna("")
    events = frame.loc[junction.str.startswith("BTP")].copy()
    if clean and "validation_status" in events.columns:
        events = events[~events["validation_status"].isin(INVALID_STATUS)]
    events["junction"] = events["junction_name"].astype(str)
    events["week"] = _week_index(events["date"])
    events = events[events["week"].notna() & events["device_id"].notna()].copy()
    events["week"] = events["week"].astype(int)
    upper = events["vehicle_type"].fillna("").str.upper()
    events["heavy"] = (~upper.isin(LIGHT_VEHICLES)).astype(float)
    return events


def junction_panel(frame: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    events = junction_events(frame, clean=clean)
    key = ["junction", "week"]
    count = events.groupby(key).size().rename("count")
    device_days = events.drop_duplicates(key + ["device_id", "date"]).groupby(key).size().rename("device_days")
    active_days = events.drop_duplicates(key + ["date"]).groupby(key).size().rename("active_days")
    devices = events.drop_duplicates(key + ["device_id"]).groupby(key).size().rename("devices")
    agg = events.groupby(key).agg(
        heavy_share=("heavy", "mean"),
        severity=("severity", "mean"),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
    )
    panel = pd.concat([count, device_days, active_days, devices, agg], axis=1).reset_index()
    panel = panel[panel["device_days"] > 0]
    return panel.reset_index(drop=True)


def _collapse(times: np.ndarray, min_gap: float) -> np.ndarray:
    if len(times) == 0:
        return times
    kept = [times[0]]
    for value in times[1:]:
        if value - kept[-1] >= min_gap:
            kept.append(value)
    return np.array(kept)


def junction_streams(frame: pd.DataFrame, top_k: int = 30, min_gap_minutes: float = 30.0, clean: bool = True):
    from parksight.impact.hawkes import daily_profile

    events = junction_events(frame, clean=clean)
    origin = events["logged_at"].min()
    events = events.assign(t=(events["logged_at"] - origin).dt.total_seconds() / 86400.0)
    horizon = float(events["t"].max()) + 1.0
    profile = daily_profile(events["t"].to_numpy(dtype=float))
    busiest = events.groupby("junction").size().sort_values(ascending=False).head(top_k).index
    min_gap = min_gap_minutes / (60.0 * 24.0)
    streams = []
    for junction in busiest:
        times = np.sort(events.loc[events["junction"] == junction, "t"].to_numpy(dtype=float))
        episodes = _collapse(times, min_gap)
        if len(episodes) >= 2:
            streams.append(episodes)
    return streams, horizon, profile
