from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree


def select_cells(frame: pd.DataFrame, max_nodes: int = 1500) -> list[str]:
    counts = frame.groupby("cell").size().sort_values(ascending=False)
    return counts.head(max_nodes).index.tolist()


def knn_edges(coords: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray]:
    tree = BallTree(coords)
    distance, index = tree.query(coords, k=k + 1)
    rows: list[int] = []
    cols: list[int] = []
    weights: list[float] = []
    for i in range(len(coords)):
        for j in range(1, k + 1):
            neighbour = int(index[i, j])
            weight = 1.0 / (distance[i, j] + 1e-6)
            rows += [i, neighbour]
            cols += [neighbour, i]
            weights += [weight, weight]
    edge_index = np.array([rows, cols], dtype=np.int64)
    edge_weight = np.array(weights, dtype=np.float32)
    edge_weight /= edge_weight.max()
    return edge_index, edge_weight


def daily_tensor(
    frame: pd.DataFrame, cells: list[str]
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    sub = frame[frame["cell"].isin(cells)]
    sub = sub[sub["date"].notna()]
    dates = sorted(sub["date"].unique().tolist())
    date_index = {date: i for i, date in enumerate(dates)}
    cell_index = {cell: i for i, cell in enumerate(cells)}

    days = sub["date"].map(date_index).to_numpy().astype(np.int64)
    nodes = sub["cell"].map(cell_index).to_numpy().astype(np.int64)
    severities = sub["severity"].to_numpy().astype(np.float32)

    counts = np.zeros((len(dates), len(cells)), dtype=np.float32)
    severity = np.zeros_like(counts)
    np.add.at(counts, (days, nodes), 1.0)
    np.add.at(severity, (days, nodes), severities)

    coords = np.array([[float(x) for x in cell.split(",")] for cell in cells])
    return counts, severity, dates, coords


def feature_tensor(counts: np.ndarray, severity: np.ndarray, dates: list[str]) -> np.ndarray:
    steps, nodes = counts.shape
    weekday = np.array([pd.Timestamp(date).weekday() for date in dates], dtype=np.float32)
    dow_sin = np.broadcast_to(np.sin(2 * np.pi * weekday / 7)[:, None], (steps, nodes))
    dow_cos = np.broadcast_to(np.cos(2 * np.pi * weekday / 7)[:, None], (steps, nodes))
    weekend = np.broadcast_to((weekday >= 5).astype(np.float32)[:, None], (steps, nodes))
    stacked = np.stack(
        [np.log1p(counts), np.log1p(severity), dow_sin, dow_cos, weekend], axis=-1
    )
    return stacked.astype(np.float32)
