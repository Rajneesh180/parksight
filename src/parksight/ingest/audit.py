from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from parksight import config


def gini(counts: np.ndarray) -> float:
    values = np.sort(np.asarray(counts, dtype=float))
    n = values.size
    total = values.sum()
    if n == 0 or total == 0:
        return 0.0
    rank = np.arange(1, n + 1)
    return float((2 * (rank * values).sum()) / (n * total) - (n + 1) / n)


@dataclass(frozen=True)
class Concentration:
    cells: int
    share_top_1pct: float
    share_top_5pct: float
    share_top_10pct: float
    gini: float


def spatial_concentration(frame: pd.DataFrame) -> Concentration:
    per_cell = frame.groupby("cell").size().sort_values(ascending=False).to_numpy()
    total = per_cell.sum()

    def head_share(fraction: float) -> float:
        k = max(1, int(len(per_cell) * fraction))
        return float(per_cell[:k].sum() / total)

    return Concentration(
        cells=len(per_cell),
        share_top_1pct=head_share(0.01),
        share_top_5pct=head_share(0.05),
        share_top_10pct=head_share(0.10),
        gini=gini(per_cell),
    )


@dataclass(frozen=True)
class Recidivism:
    offenders: int
    repeat_offenders: int
    repeat_share: float
    max_citations: int


def recidivism(frame: pd.DataFrame) -> Recidivism:
    per_vehicle = frame.groupby("vehicle_token").size()
    repeat = per_vehicle[per_vehicle > 1]
    return Recidivism(
        offenders=int(per_vehicle.size),
        repeat_offenders=int(repeat.size),
        repeat_share=float(repeat.sum() / per_vehicle.sum()),
        max_citations=int(per_vehicle.max()),
    )


@dataclass(frozen=True)
class Churn:
    emerging: int
    declining: int
    persistent: int


def spatial_churn(
    frame: pd.DataFrame,
    split: str = config.CHURN_SPLIT,
    appear: int = 15,
    vanish: int = 2,
) -> Churn:
    half = np.where((frame["date"] < split).fillna(False), "h1", "h2")
    counts = frame.assign(half=half).groupby(["cell", "half"]).size().unstack(fill_value=0)
    h1 = counts["h1"] if "h1" in counts else pd.Series(0, index=counts.index)
    h2 = counts["h2"] if "h2" in counts else pd.Series(0, index=counts.index)
    return Churn(
        emerging=int(((h1 <= vanish) & (h2 >= appear)).sum()),
        declining=int(((h1 >= appear) & (h2 <= vanish)).sum()),
        persistent=int(((h1 >= 5) & (h2 >= 5)).sum()),
    )


def hour_profile(frame: pd.DataFrame) -> pd.Series:
    return frame["hour"].value_counts().sort_index()


def audit_report(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "parking_share": float(frame["is_parking"].mean()),
        "date_range": [frame["date"].min(), frame["date"].max()],
        "concentration": asdict(spatial_concentration(frame)),
        "recidivism": asdict(recidivism(frame)),
        "churn": asdict(spatial_churn(frame)),
    }
