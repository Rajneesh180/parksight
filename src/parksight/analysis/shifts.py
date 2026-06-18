from __future__ import annotations

import numpy as np
import pandas as pd

from parksight.optimize.coverage import covered_fraction, greedy_indices

SHIFTS: dict[str, tuple[int, int]] = {"night_00_06": (0, 6), "day_07_13": (7, 13)}
MIN_BUSY = 200


def assign_shift(hours: pd.Series) -> pd.Series:
    labels = pd.Series("other", index=hours.index, dtype="object")
    for name, (low, high) in SHIFTS.items():
        labels[(hours >= low) & (hours <= high)] = name
    return labels


def _cell_grid(frame: pd.DataFrame) -> tuple[list[str], dict[str, int], np.ndarray]:
    cells = sorted(frame["cell"].dropna().unique())
    index = {cell: i for i, cell in enumerate(cells)}
    coords = np.array([[float(part) for part in cell.split(",")] for cell in cells])
    return cells, index, coords


def _volume(frame: pd.DataFrame, index: dict[str, int]) -> np.ndarray:
    vector = np.zeros(len(index))
    for cell, count in frame.groupby("cell").size().items():
        if cell in index:
            vector[index[cell]] = count
    return vector


def shift_report(
    frame: pd.DataFrame, teams: int = 12, reach: float = 0.003, top: int = 12
) -> dict:
    sub = frame[frame["hour"].notna()].copy()
    sub["hour"] = sub["hour"].astype(int)
    sub["shift"] = assign_shift(sub["hour"])
    enforced = sub[sub["shift"] != "other"]
    after_14 = float((sub["hour"] >= 14).mean()) if len(sub) else float("nan")
    if enforced.empty:
        return {
            "teams": teams,
            "reach": reach,
            "total_enforced": 0,
            "unenforced_after_14h_share": after_14,
            "global_peak_shift": None,
            "busy_cells": 0,
            "peak_shift_differs_share": float("nan"),
            "static_coverage": 0.0,
            "shift_aware_coverage": 0.0,
            "uplift": 0.0,
            "shifts": {},
        }

    _, index, coords = _cell_grid(enforced)
    allday = _volume(enforced, index)
    grand = allday.sum()
    static_idx, _, _ = greedy_indices(coords, allday, teams, reach)

    busy = enforced.groupby("cell").size()
    busy = busy[busy >= MIN_BUSY].index
    global_peak = enforced.groupby("shift").size().idxmax()
    peak = enforced[enforced["cell"].isin(busy)].groupby("cell")["shift"].agg(
        lambda series: series.value_counts().idxmax()
    )

    static_total = aware_total = 0.0
    shifts_out: dict[str, dict] = {}
    for name, (low, high) in SHIFTS.items():
        part = enforced[enforced["shift"] == name]
        vector = _volume(part, index)
        aware_idx, _, _ = greedy_indices(coords, vector, teams, reach)
        static_cov = covered_fraction(coords, coords[static_idx], vector, reach)
        aware_cov = covered_fraction(coords, coords[aware_idx], vector, reach)
        static_total += static_cov * vector.sum()
        aware_total += aware_cov * vector.sum()
        ranked = part.groupby("cell").size().sort_values(ascending=False).head(top)
        shifts_out[name] = {
            "records": int(vector.sum()),
            "hours": f"{low:02d}:00-{high:02d}:59",
            "static_coverage": float(static_cov),
            "shift_aware_coverage": float(aware_cov),
            "top_cells": [
                {
                    "cell": cell,
                    "latitude": float(cell.split(",")[0]),
                    "longitude": float(cell.split(",")[1]),
                    "volume": int(count),
                }
                for cell, count in ranked.items()
            ],
        }

    return {
        "teams": teams,
        "reach": reach,
        "total_enforced": int(grand),
        "unenforced_after_14h_share": after_14,
        "global_peak_shift": global_peak,
        "busy_cells": int(len(busy)),
        "peak_shift_differs_share": float((peak != global_peak).mean()) if len(busy) else float("nan"),
        "static_coverage": static_total / grand,
        "shift_aware_coverage": aware_total / grand,
        "uplift": (aware_total - static_total) / grand,
        "shifts": shifts_out,
    }


def shift_forecast(
    frame: pd.DataFrame, forecast: dict[str, float], teams: int = 12, reach: float = 0.003, top: int = 12
) -> dict:
    sub = frame[frame["hour"].notna()].copy()
    sub["hour"] = sub["hour"].astype(int)
    sub["shift"] = assign_shift(sub["hour"])
    enforced = sub[sub["shift"] != "other"]

    cells = list(forecast)
    coords = np.array([[float(part) for part in cell.split(",")] for cell in cells])
    values = np.array([forecast[cell] for cell in cells], dtype=float)
    index = {cell: i for i, cell in enumerate(cells)}

    inside = enforced[enforced["cell"].isin(index)]
    counts = inside.groupby(["cell", "shift"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)

    out: dict[str, dict] = {}
    for name in SHIFTS:
        share = np.zeros(len(cells))
        if name in counts.columns:
            cell_share = (counts[name] / totals).fillna(0.0)
            for cell, value in cell_share.items():
                share[index[cell]] = value
        intensity = values * share
        plan_idx, _, _ = greedy_indices(coords, intensity, teams, reach)
        ranked = np.argsort(-intensity)[:top]
        out[name] = {
            "plan": [cells[i] for i in plan_idx],
            "top_cells": [
                {
                    "cell": cells[i],
                    "latitude": float(coords[i, 0]),
                    "longitude": float(coords[i, 1]),
                    "forecast_intensity": float(intensity[i]),
                }
                for i in ranked
                if intensity[i] > 0
            ],
        }
    return out
