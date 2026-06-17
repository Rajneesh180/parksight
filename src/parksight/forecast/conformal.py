from __future__ import annotations

import numpy as np

LEVELS: tuple[float, ...] = (0.5, 0.8, 0.9)


def _scale(pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.asarray(pred, dtype=float) + 1.0)


def _quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
    return float(np.sort(scores)[k - 1])


def quantiles(actual: np.ndarray, pred: np.ndarray, levels: tuple[float, ...] = LEVELS) -> dict[str, float]:
    scores = (np.abs(actual - pred) / _scale(pred)).ravel()
    return {str(int(level * 100)): _quantile(scores, 1 - level) for level in levels}


def calibration(actual: np.ndarray, pred: np.ndarray, levels: tuple[float, ...] = LEVELS) -> list[dict]:
    windows = actual.shape[0]
    scores = np.abs(actual - pred) / _scale(pred)
    report = []
    for level in levels:
        covered, widths = [], []
        for w in range(windows):
            calib = np.concatenate([scores[j] for j in range(windows) if j != w])
            q = _quantile(calib, 1 - level)
            half = q * _scale(pred[w])
            low = np.clip(pred[w] - half, 0, None)
            high = pred[w] + half
            covered.append(float(np.mean((actual[w] >= low) & (actual[w] <= high))))
            widths.append(float(np.median(high - low)))
        report.append(
            {
                "target": level,
                "empirical_coverage": float(np.mean(covered)),
                "median_width": float(np.mean(widths)),
            }
        )
    return report


def interval(value: float, q: float) -> tuple[float, float]:
    half = q * float(np.sqrt(value + 1.0))
    return max(0.0, value - half), value + half
