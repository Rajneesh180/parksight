from __future__ import annotations

import numpy as np


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def precision_at_k(actual: np.ndarray, predicted: np.ndarray, k: int) -> float:
    scores = []
    for i in range(len(actual)):
        top_pred = set(np.argsort(-predicted[i])[:k])
        top_actual = set(np.argsort(-actual[i])[:k])
        scores.append(len(top_pred & top_actual) / k)
    return float(np.mean(scores))


def volatile_mask(actual: np.ndarray, quantile: float = 0.75) -> np.ndarray:
    volatility = actual.std(axis=0)
    return volatility >= np.quantile(volatility, quantile)


def per_window_mae(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.abs(actual - predicted).mean(axis=1)


def paired_bootstrap(baseline: np.ndarray, model: np.ndarray, iterations: int = 5000, seed: int = 0):
    difference = baseline - model
    generator = np.random.default_rng(seed)
    resamples = [
        generator.choice(difference, size=difference.size, replace=True).mean()
        for _ in range(iterations)
    ]
    low, high = np.percentile(resamples, [2.5, 97.5])
    return float(difference.mean()), float(low), float(high)
