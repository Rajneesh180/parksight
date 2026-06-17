from __future__ import annotations

import numpy as np


def persistence(counts: np.ndarray, starts: list[int], horizon: int) -> np.ndarray:
    return np.stack([counts[start - horizon : start].sum(axis=0) for start in starts])


def historical_mean(train_targets: np.ndarray, windows: int) -> np.ndarray:
    return np.tile(train_targets.mean(axis=0), (windows, 1))
