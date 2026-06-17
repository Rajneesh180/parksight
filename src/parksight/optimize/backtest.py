from __future__ import annotations

import numpy as np

from parksight.optimize.coverage import covered_fraction, greedy_indices


def _arm_coverage(coords, ranking, target, teams, reach):
    chosen, _, _ = greedy_indices(coords, ranking, teams, reach)
    return covered_fraction(coords, coords[chosen], target, reach)


def deployment_backtest(
    coords: np.ndarray,
    forecast: np.ndarray,
    status_quo: np.ndarray,
    actual: np.ndarray,
    teams_grid: list[int],
    reach: float = 0.003,
    volatile: np.ndarray | None = None,
) -> dict[str, list[dict]]:
    windows = len(actual)
    targets = {"overall": actual}
    if volatile is not None:
        targets["volatile"] = actual * volatile

    report: dict[str, list[dict]] = {name: [] for name in targets}
    for teams in teams_grid:
        forecast_sets = [greedy_indices(coords, forecast[w], teams, reach)[0] for w in range(windows)]
        status_sets = [greedy_indices(coords, status_quo[w], teams, reach)[0] for w in range(windows)]
        for name, target in targets.items():
            forecast_cov = [
                covered_fraction(coords, coords[forecast_sets[w]], target[w], reach)
                for w in range(windows)
            ]
            status_cov = [
                covered_fraction(coords, coords[status_sets[w]], target[w], reach)
                for w in range(windows)
            ]
            oracle_cov = [_arm_coverage(coords, target[w], target[w], teams, reach) for w in range(windows)]
            report[name].append(
                {
                    "teams": int(teams),
                    "forecast": float(np.mean(forecast_cov)),
                    "status_quo": float(np.mean(status_cov)),
                    "oracle": float(np.mean(oracle_cov)),
                    "forecast_uplift": float(np.mean(forecast_cov) - np.mean(status_cov)),
                    "headroom_captured": _headroom_share(
                        np.mean(forecast_cov), np.mean(status_cov), np.mean(oracle_cov)
                    ),
                }
            )
    return report


def _headroom_share(forecast: float, status_quo: float, oracle: float) -> float:
    headroom = oracle - status_quo
    if headroom <= 0:
        return 0.0
    return float((forecast - status_quo) / headroom)
