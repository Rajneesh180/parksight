from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from scipy.sparse import csr_matrix, diags
from sklearn.neighbors import BallTree

FEATURE_NAMES = [
    "prev_7",
    "prev_14",
    "prev_window",
    "mean_daily",
    "last_day",
    "week_trend",
    "severity",
    "neighbour_prev_7",
    "weekday",
]


def _neighbour_mean(coords: np.ndarray, k: int) -> csr_matrix:
    neighbours = BallTree(coords).query(coords, k=k + 1)[1][:, 1:]
    nodes = len(coords)
    rows = np.repeat(np.arange(nodes), k)
    cols = neighbours.reshape(-1)
    matrix = csr_matrix((np.ones(rows.size), (rows, cols)), shape=(nodes, nodes))
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    degree[degree == 0] = 1.0
    return diags(1.0 / degree) @ matrix


def _window_features(
    counts: np.ndarray,
    severity: np.ndarray,
    neighbour: csr_matrix,
    start: int,
    lookback: int,
    weekday: int,
) -> np.ndarray:
    prev_7 = counts[start - 7 : start].sum(0)
    prior_7 = counts[start - 14 : start - 7].sum(0)
    prev_window = counts[start - lookback : start].sum(0)
    return np.column_stack(
        [
            prev_7,
            counts[start - 14 : start].sum(0),
            prev_window,
            prev_window / lookback,
            counts[start - 1],
            prev_7 - prior_7,
            severity[start - lookback : start].mean(0),
            neighbour.dot(prev_7),
            np.full(counts.shape[1], float(weekday)),
        ]
    )


def _table(counts, severity, neighbour, dates, starts, lookback, horizon):
    features = []
    targets = []
    for start in starts:
        weekday = pd.Timestamp(dates[start]).weekday()
        features.append(_window_features(counts, severity, neighbour, start, lookback, weekday))
        targets.append(counts[start : start + horizon].sum(0))
    return np.vstack(features), np.concatenate(targets)


def train(counts, severity, dates, coords, train_starts, val_starts, lookback, horizon, k=8):
    neighbour = _neighbour_mean(coords, k)
    x_train, y_train = _table(counts, severity, neighbour, dates, train_starts, lookback, horizon)
    x_val, y_val = _table(counts, severity, neighbour, dates, val_starts, lookback, horizon)

    model = LGBMRegressor(
        objective="poisson",
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=0,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric="l1",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(0)],
    )
    return model, neighbour


def predict(model, neighbour, counts, severity, dates, starts, lookback):
    rows = []
    for start in starts:
        weekday = pd.Timestamp(dates[start]).weekday()
        feature = _window_features(counts, severity, neighbour, start, lookback, weekday)
        rows.append(np.clip(model.predict(feature), 0, None))
    return np.stack(rows)


def forecast_next(model, neighbour, counts, severity, dates, lookback):
    weekday = pd.Timestamp(dates[-1]).weekday()
    feature = _window_features(counts, severity, neighbour, len(dates), lookback, weekday)
    return np.clip(model.predict(feature), 0, None)
