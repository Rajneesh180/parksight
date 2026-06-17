from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import BallTree

_REACH_DEGREES = 0.003


def _coverage_matrix(coords: np.ndarray, reach: float) -> tuple[csr_matrix, np.ndarray]:
    neighbours = BallTree(coords).query_radius(coords, r=reach)
    rows = np.repeat(np.arange(len(coords)), [len(group) for group in neighbours])
    cols = np.concatenate(neighbours) if len(coords) else np.empty(0, dtype=int)
    matrix = csr_matrix((np.ones(len(cols)), (rows, cols)), shape=(len(coords), len(coords)))
    return matrix, neighbours


def greedy_indices(
    coords: np.ndarray, value: np.ndarray, teams: int, reach: float = _REACH_DEGREES
) -> tuple[list[int], list[float], list[float]]:
    if len(coords) == 0 or teams < 1:
        return [], [], []
    coverage, neighbours = _coverage_matrix(coords, reach)
    total = value.sum()
    if total <= 0:
        return [], [], []

    covered = np.zeros(len(coords), dtype=bool)
    chosen: list[int] = []
    gains: list[float] = []
    shares: list[float] = []
    running = 0.0

    for _ in range(min(teams, len(coords))):
        marginal = coverage.dot(value * ~covered)
        if chosen:
            marginal[chosen] = -1.0
        best = int(np.argmax(marginal))
        if marginal[best] <= 0:
            break
        covered[neighbours[best]] = True
        chosen.append(best)
        running += float(marginal[best])
        gains.append(float(marginal[best]))
        shares.append(running / total)
    return chosen, gains, shares


def covered_fraction(
    coords: np.ndarray, centres: np.ndarray, value: np.ndarray, reach: float = _REACH_DEGREES
) -> float:
    if len(centres) == 0 or value.sum() == 0:
        return 0.0
    reached = np.unique(np.concatenate(BallTree(coords).query_radius(centres, r=reach)))
    return float(value[reached].sum() / value.sum())


def maximal_coverage(
    cells: pd.DataFrame, teams: int, value: str = "priority", reach: float = _REACH_DEGREES
) -> pd.DataFrame:
    if teams < 1 or cells.empty:
        return cells.iloc[:0].assign(
            marginal_gain=pd.Series(dtype=float), cumulative_share=pd.Series(dtype=float)
        )

    coords = cells[["latitude", "longitude"]].to_numpy()
    chosen, gains, shares = greedy_indices(coords, cells[value].to_numpy(), teams, reach)
    plan = cells.iloc[chosen].copy()
    plan["marginal_gain"] = gains
    plan["cumulative_share"] = shares
    return plan


def top_k(cells: pd.DataFrame, teams: int, value: str = "priority") -> pd.DataFrame:
    return cells.sort_values(value, ascending=False).head(teams).copy()


def covered_value(
    cells: pd.DataFrame, plan: pd.DataFrame, value: str = "volume", reach: float = _REACH_DEGREES
) -> float:
    if plan.empty or cells.empty:
        return 0.0
    return covered_fraction(
        cells[["latitude", "longitude"]].to_numpy(),
        plan[["latitude", "longitude"]].to_numpy(),
        cells[value].to_numpy(),
        reach,
    )
