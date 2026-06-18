from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import kstest

BETA_MIN = 0.1
BETA_MAX = 48.0


def _beta(param: float) -> float:
    return BETA_MIN + (BETA_MAX - BETA_MIN) / (1.0 + np.exp(-param))


def _phase_hour(times: np.ndarray) -> np.ndarray:
    return (np.floor(times * 24.0) % 24).astype(int)


def daily_profile(times: np.ndarray) -> np.ndarray:
    counts = np.bincount(_phase_hour(times), minlength=24).astype(float) + 0.5
    return counts / counts.mean()


def _rate(times: np.ndarray, profile: np.ndarray | None) -> np.ndarray:
    if profile is None:
        return np.ones(len(times))
    return profile[_phase_hour(times)]


def _integral(times: np.ndarray, profile: np.ndarray | None) -> np.ndarray:
    if profile is None:
        return times
    cycle = profile.sum()
    hours = times * 24.0
    full = np.floor(hours).astype(int)
    frac = hours - full
    prefix = np.concatenate([[0.0], np.cumsum(profile)])
    out = np.empty(len(times))
    for i in range(len(times)):
        m = full[i]
        out[i] = ((m // 24) * cycle + prefix[m % 24] + frac[i] * profile[m % 24]) / 24.0
    return out


def _excitation(times: np.ndarray, beta: float) -> np.ndarray:
    a = np.zeros(len(times))
    for i in range(1, len(times)):
        a[i] = np.exp(-beta * (times[i] - times[i - 1])) * (1.0 + a[i - 1])
    return a


def stream_loglik(times, mu, alpha, beta, horizon, profile=None) -> float:
    if len(times) == 0:
        return -mu * horizon
    a = _excitation(times, beta)
    intensity = mu * _rate(times, profile) + alpha * beta * a
    compensator = mu * horizon + alpha * np.sum(1.0 - np.exp(-beta * (horizon - times)))
    return float(np.sum(np.log(intensity)) - compensator)


def _background(n: int, alpha: float, horizon: float) -> float:
    return (1.0 - alpha) * n / horizon


def fit_pooled(streams: list[np.ndarray], horizon: float, profile: np.ndarray | None = None) -> dict:
    sizes = np.array([len(s) for s in streams])

    def negative(params: np.ndarray) -> float:
        alpha = 1.0 / (1.0 + np.exp(-params[0]))
        beta = _beta(params[1])
        total = 0.0
        for times, n in zip(streams, sizes):
            total += stream_loglik(times, _background(n, alpha, horizon), alpha, beta, horizon, profile)
        return -total

    result = minimize(negative, x0=np.array([0.0, -3.0]), method="Nelder-Mead")
    alpha = float(1.0 / (1.0 + np.exp(-result.x[0])))
    beta = float(_beta(result.x[1]))
    poisson_ll = float(
        sum(stream_loglik(times, n / horizon, 0.0, 1.0, horizon, profile) for times, n in zip(streams, sizes))
    )
    return {
        "branching_ratio": alpha,
        "decay_per_day": beta,
        "mean_lifetime_days": 1.0 / beta,
        "loglik": -float(result.fun),
        "poisson_loglik": poisson_ll,
        "loglik_gain": -float(result.fun) - poisson_ll,
        "n_streams": int(len(streams)),
        "n_events": int(sizes.sum()),
    }


def goodness_of_fit(streams, alpha, beta, horizon, profile=None) -> dict:
    gaps: list[np.ndarray] = []
    for times in streams:
        n = len(times)
        if n < 3:
            continue
        a = _excitation(times, beta)
        prior = np.arange(n)
        compensator = _background(n, alpha, horizon) * _integral(times, profile) + alpha * (prior - a)
        gaps.append(np.diff(compensator))
    pooled = np.concatenate(gaps)
    pooled = pooled[pooled >= 0]
    statistic, pvalue = kstest(pooled, "expon")
    return {"ks_statistic": float(statistic), "ks_pvalue": float(pvalue), "n_gaps": int(len(pooled))}
