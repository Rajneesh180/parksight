from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from parksight.impact import deploy, exposure, hawkes, intensity


def _productivity(panel: pd.DataFrame, train_frac: float = 0.7, min_test_effort: int = 3, top: int = 20) -> dict:
    weeks = np.sort(panel["week"].unique())
    cut = weeks[int(len(weeks) * train_frac)]
    train = panel[panel["week"] < cut]
    test = panel[panel["week"] >= cut]

    adjusted, _ = intensity.fit_intensity(train)
    train_grp = train.groupby("junction")
    raw_count = train_grp["count"].sum()
    train_effort = train_grp["device_days"].sum()
    raw_yield = raw_count / train_effort
    shrunk_values, _ = intensity.eb_shrunk_rate(raw_count.to_numpy(), train_effort.to_numpy())
    shrunk = pd.Series(shrunk_values, index=raw_count.index)

    test_grp = test.groupby("junction")
    test_yield = (test_grp["count"].sum() / test_grp["device_days"].sum()).rename("test_yield")
    test_effort = test_grp["device_days"].sum()
    eligible = test_yield[test_effort >= min_test_effort]

    frame = pd.DataFrame({
        "adjusted": adjusted["adjusted_intensity"],
        "raw_count": raw_count,
        "raw_yield": raw_yield,
        "eb_shrunk": shrunk,
    }).join(eligible, how="inner").dropna()

    def rank_corr(column: str) -> float:
        return float(spearmanr(frame[column], frame["test_yield"]).statistic)

    def top_yield(column: str) -> float:
        chosen = frame.sort_values(column, ascending=False).head(top)
        return float(chosen["test_yield"].mean())

    return {
        "n_junctions": int(len(frame)),
        "spearman_vs_next_period_yield": {
            "eb_shrunk_yield": rank_corr("eb_shrunk"),
            "raw_yield_per_effort": rank_corr("raw_yield"),
            "adjusted_intensity": rank_corr("adjusted"),
            "raw_count": rank_corr("raw_count"),
        },
        "top20_next_period_yield": {
            "eb_shrunk_yield": top_yield("eb_shrunk"),
            "raw_count": top_yield("raw_count"),
            "all_eligible_mean": float(frame["test_yield"].mean()),
        },
    }


def _bias(panel: pd.DataFrame) -> float:
    log_y = np.log1p(panel["count"].to_numpy(float))
    log_e = np.log1p(panel["device_days"].to_numpy(float))
    return float(np.corrcoef(log_y, log_e)[0, 1])


def _sensitivity(panel: pd.DataFrame, primary: pd.DataFrame) -> dict:
    base = primary["adjusted_intensity"]
    out = {}
    for proxy in exposure.EXPOSURES:
        if proxy == "device_days":
            continue
        other, _ = intensity.fit_intensity(panel, exposure=proxy)
        joined = base.to_frame("a").join(other["adjusted_intensity"].rename("b"), how="inner")
        tau, _ = kendalltau(joined["a"], joined["b"])
        out[proxy] = float(tau)
    return out


def _validation_robustness(frame: pd.DataFrame) -> dict:
    panel_clean = exposure.junction_panel(frame, clean=True)
    panel_full = exposure.junction_panel(frame, clean=False)
    clean_rank = deploy.corrected_table(panel_clean)["corrected_intensity"]
    full_rank = deploy.corrected_table(panel_full)["corrected_intensity"]
    joined = clean_rank.to_frame("clean").join(full_rank.rename("full"), how="inner")
    tau, _ = kendalltau(joined["clean"], joined["full"])
    dropped = 1.0 - panel_clean["count"].sum() / panel_full["count"].sum()
    return {
        "rejected_duplicate_share": float(dropped),
        "kendall_tau_clean_vs_full": float(tau),
        "deployment_uplift_clean_pct": float(deploy.backtest(panel_clean)["curve"][2]["topn"]["uplift_pct"]),
    }


def impact_report(frame: pd.DataFrame, top_k: int = 30, top_n: int = 25) -> dict:
    panel = exposure.junction_panel(frame)
    impact, meta = intensity.fit_intensity(panel)
    heldout = intensity.heldout_loglik(panel)

    streams, horizon, profile = exposure.junction_streams(frame, top_k=top_k)
    self_exciting = hawkes.fit_pooled(streams, horizon, profile)
    fit = hawkes.goodness_of_fit(streams, self_exciting["branching_ratio"], self_exciting["decay_per_day"], horizon, profile)

    corrected = deploy.corrected_table(panel).join(impact[["adjusted_intensity", "heavy_share"]], how="left")
    junctions = [
        {
            "junction": name,
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "raw_count": int(row["raw_count"]),
            "exposure": int(row["effort"]),
            "raw_per_effort": float(row["raw_per_effort"]),
            "corrected_intensity": float(row["corrected_intensity"]),
            "adjusted_intensity": float(row["adjusted_intensity"]) if pd.notna(row["adjusted_intensity"]) else None,
            "heavy_share": float(row["heavy_share"]) if pd.notna(row["heavy_share"]) else None,
        }
        for name, row in corrected.iterrows()
    ]

    return {
        "n_junctions": int(impact.shape[0]),
        "n_junction_weeks": int(panel.shape[0]),
        "exposure_bias_corr": _bias(panel),
        "model": {
            "likelihood": "negative_binomial",
            "dispersion_alpha": float(meta["alpha"]),
            "offset": "log enforcement effort (distinct device-days)",
            "spatial": "low-rank thin-plate spline",
            "heldout": heldout,
        },
        "self_excitation": {**self_exciting, **fit},
        "exposure_sensitivity_kendall_tau": _sensitivity(panel, impact),
        "productivity_validation": _productivity(panel),
        "deployment": deploy.backtest(panel),
        "validation_robustness": _validation_robustness(frame),
        "junctions": junctions,
    }
