from __future__ import annotations

import numpy as np
import pandas as pd

from parksight.impact import intensity
from parksight.optimize.coverage import greedy_indices

REACH = 0.003
TEAM_GRID = (6, 8, 12, 16, 20)


def corrected_table(panel: pd.DataFrame) -> pd.DataFrame:
    grouped = panel.groupby("junction")
    count = grouped["count"].sum()
    effort = grouped["device_days"].sum()
    shrunk, _ = intensity.eb_shrunk_rate(count.to_numpy(float), effort.to_numpy(float))
    table = pd.DataFrame(
        {
            "corrected_intensity": shrunk,
            "raw_count": count.to_numpy(float),
            "effort": effort.to_numpy(float),
            "raw_per_effort": count.to_numpy(float) / effort.to_numpy(float),
            "latitude": grouped["latitude"].mean().to_numpy(float),
            "longitude": grouped["longitude"].mean().to_numpy(float),
        },
        index=count.index,
    )
    return table.sort_values("corrected_intensity", ascending=False)


def select(table: pd.DataFrame, value: str, teams: int, reach: float = REACH) -> list:
    coords = table[["latitude", "longitude"]].to_numpy(float)
    idx, _, _ = greedy_indices(coords, table[value].to_numpy(float), teams, reach)
    return list(table.index[idx])


def select_topn(table: pd.DataFrame, value: str, teams: int) -> list:
    return list(table.sort_values(value, ascending=False).head(teams).index)


def _pct(corrected: float, raw: float) -> float:
    if not raw or np.isnan(raw) or np.isnan(corrected):
        return float("nan")
    return float((corrected - raw) / raw * 100.0)


def backtest(panel: pd.DataFrame, team_grid=TEAM_GRID, reach: float = REACH, train_frac: float = 0.7, min_test_effort: int = 3) -> dict:
    weeks = np.sort(panel["week"].unique())
    cut = weeks[int(len(weeks) * train_frac)]
    train = panel[panel["week"] < cut]
    test = panel[panel["week"] >= cut]

    table = corrected_table(train)
    grouped = test.groupby("junction")
    test_yield = grouped["count"].sum() / grouped["device_days"].sum()
    test_effort = grouped["device_days"].sum()
    eligible = test_yield[test_effort >= min_test_effort]

    def mean_yield(junctions: list) -> float:
        values = eligible.reindex(junctions).dropna()
        return float(values.mean()) if len(values) else float("nan")

    curve = []
    for teams in team_grid:
        cov_c = mean_yield(select(table, "corrected_intensity", teams, reach))
        cov_r = mean_yield(select(table, "raw_count", teams, reach))
        top_c = mean_yield(select_topn(table, "corrected_intensity", teams))
        top_r = mean_yield(select_topn(table, "raw_count", teams))
        curve.append(
            {
                "teams": int(teams),
                "coverage": {"corrected": cov_c, "raw": cov_r, "uplift_pct": _pct(cov_c, cov_r)},
                "topn": {"corrected": top_c, "raw": top_r, "uplift_pct": _pct(top_c, top_r)},
            }
        )
    return {"reach": reach, "city_mean_yield": float(eligible.mean()), "curve": curve}
