from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import gammaln
from sklearn.cluster import KMeans


@dataclass(frozen=True)
class Scale:
    mean: np.ndarray
    std: np.ndarray

    def apply(self, coords: np.ndarray) -> np.ndarray:
        return (coords - self.mean) / self.std


def _scale(coords: np.ndarray) -> Scale:
    return Scale(coords.mean(axis=0), coords.std(axis=0) + 1e-9)


def thin_plate_basis(coords: np.ndarray, knots: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - knots[None, :, :]
    r = np.sqrt((diff ** 2).sum(-1))
    radial = np.where(r > 1e-12, r ** 2 * np.log(np.where(r > 1e-12, r, 1.0)), 0.0)
    null = np.column_stack([np.ones(len(coords)), coords])
    return np.column_stack([null, radial])


def _knots(scaled: np.ndarray, n_knots: int, seed: int) -> np.ndarray:
    unique = np.unique(scaled, axis=0)
    k = min(n_knots, len(unique))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(scaled).cluster_centers_


def _temporal(weeks: np.ndarray, ref: np.ndarray) -> np.ndarray:
    centre = ref.mean()
    span = ref.std() + 1e-9
    w = (weeks - centre) / span
    return np.column_stack([w, w ** 2])


def _design(panel: pd.DataFrame, knots: np.ndarray, scale: Scale, ref_weeks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = scale.apply(panel[["latitude", "longitude"]].to_numpy(dtype=float))
    spatial = thin_plate_basis(coords, knots)
    temporal = _temporal(panel["week"].to_numpy(float), ref_weeks)
    heavy = panel[["heavy_share"]].to_numpy(float)
    design = np.column_stack([spatial, temporal, heavy])
    offset = np.log(panel["device_days"].to_numpy(float))
    return design, offset


def eb_shrunk_rate(count: np.ndarray, effort: np.ndarray) -> tuple[np.ndarray, dict]:
    from scipy.optimize import minimize

    count = np.asarray(count, dtype=float)
    effort = np.asarray(effort, dtype=float)
    global_rate = count.sum() / effort.sum()

    def negative(params: np.ndarray) -> float:
        shape = np.exp(params[0])
        mean = np.exp(params[1])
        m = effort * mean
        ll = gammaln(count + shape) - gammaln(shape) - gammaln(count + 1)
        ll = ll + shape * np.log(shape / (shape + m)) + count * np.log(m / (shape + m) + 1e-12)
        return -float(ll.sum())

    result = minimize(negative, np.array([0.0, np.log(global_rate)]), method="Nelder-Mead")
    shape = float(np.exp(result.x[0]))
    mean = float(np.exp(result.x[1]))
    posterior = (count + shape) / (effort + shape / mean)
    return posterior, {"shape": shape, "prior_rate": mean}


def nb_alpha(y: np.ndarray, mu: np.ndarray) -> float:
    w = ((y - mu) ** 2 - y) / mu
    return float(max((mu * w).sum() / (mu ** 2).sum(), 1e-6))


def _poisson_ll(y: np.ndarray, mu: np.ndarray) -> float:
    return float((y * np.log(mu) - mu - gammaln(y + 1)).sum())


def _nb_ll(y: np.ndarray, mu: np.ndarray, alpha: float) -> float:
    r = 1.0 / alpha
    return float(
        (gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))).sum()
    )


def fit_intensity(panel: pd.DataFrame, exposure: str = "device_days", n_knots: int = 30, seed: int = 0):
    work = panel.assign(device_days=panel[exposure].astype(float)).reset_index(drop=True)
    coords = work[["latitude", "longitude"]].to_numpy(dtype=float)
    scale = _scale(coords)
    knots = _knots(scale.apply(coords), n_knots, seed)
    ref_weeks = work["week"].to_numpy(float)
    design, offset = _design(work, knots, scale, ref_weeks)
    y = work["count"].to_numpy(float)

    poisson = sm.GLM(y, design, family=sm.families.Poisson(), offset=offset).fit()
    alpha = nb_alpha(y, np.asarray(poisson.fittedvalues))
    model = sm.GLM(y, design, family=sm.families.NegativeBinomial(alpha=alpha), offset=offset).fit()

    rate = np.exp(design @ np.asarray(model.params))
    scored = work.assign(adjusted=rate)
    impact = (
        scored.groupby("junction")
        .agg(
            adjusted_intensity=("adjusted", "mean"),
            raw_count=("count", "sum"),
            exposure=(exposure, "sum"),
            heavy_share=("heavy_share", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .sort_values("adjusted_intensity", ascending=False)
    )
    impact["raw_per_effort"] = impact["raw_count"] / impact["exposure"]
    return impact, {"alpha": alpha, "n_obs": int(len(work))}


def heldout_loglik(panel: pd.DataFrame, train_frac: float = 0.7, n_knots: int = 30, seed: int = 0) -> dict:
    work = panel.assign(device_days=panel["device_days"].astype(float)).reset_index(drop=True)
    coords = work[["latitude", "longitude"]].to_numpy(dtype=float)
    scale = _scale(coords)
    knots = _knots(scale.apply(coords), n_knots, seed)
    ref_weeks = work["week"].to_numpy(float)
    weeks = np.sort(work["week"].unique())
    cut = weeks[int(len(weeks) * train_frac)]
    tr = (work["week"] < cut).to_numpy()
    te = ~tr

    design, offset = _design(work, knots, scale, ref_weeks)
    y = work["count"].to_numpy(float)

    pois_tr = sm.GLM(y[tr], design[tr], family=sm.families.Poisson(), offset=offset[tr]).fit()
    alpha = nb_alpha(y[tr], np.asarray(pois_tr.fittedvalues))
    nb_tr = sm.GLM(y[tr], design[tr], family=sm.families.NegativeBinomial(alpha=alpha), offset=offset[tr]).fit()
    null = sm.GLM(y[tr], np.ones((tr.sum(), 1)), family=sm.families.NegativeBinomial(alpha=alpha), offset=offset[tr]).fit()

    mu_nb = np.asarray(nb_tr.predict(design[te], offset=offset[te]))
    mu_pois = np.asarray(pois_tr.predict(design[te], offset=offset[te]))
    mu_null = np.asarray(null.predict(np.ones((te.sum(), 1)), offset=offset[te]))
    n = int(te.sum())
    return {
        "test_obs": n,
        "alpha": float(alpha),
        "nb_ll_per_obs": _nb_ll(y[te], mu_nb, alpha) / n,
        "poisson_ll_per_obs": _poisson_ll(y[te], mu_pois) / n,
        "null_ll_per_obs": _nb_ll(y[te], mu_null, alpha) / n,
    }
