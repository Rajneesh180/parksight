from __future__ import annotations

import numpy as np
import pandas as pd

_ROAD_WEIGHTS = {
    "ring road": 1.0,
    "highway": 1.0,
    "flyover": 0.95,
    "main road": 0.8,
    "road": 0.5,
}
_RESIDENTIAL_FLOOR = 0.4
_INTERSECTION = r"circle|junction|cross"
_GENERATOR = r"market|mall|hospital|school|metro|railway|bus"
_INTERSECTION_BOOST = 0.25
_GENERATOR_BOOST = 0.15


def flow_disruption(location: pd.Series) -> pd.Series:
    text = location.fillna("").str.lower()

    weight = np.full(len(text), _RESIDENTIAL_FLOOR)
    for token, value in _ROAD_WEIGHTS.items():
        present = text.str.contains(token, regex=False).to_numpy()
        weight = np.maximum(weight, np.where(present, value, 0.0))

    intersection = np.where(text.str.contains(_INTERSECTION), _INTERSECTION_BOOST, 0.0)
    generator = np.where(text.str.contains(_GENERATOR), _GENERATOR_BOOST, 0.0)

    disruption = np.minimum(weight * (1 + intersection + generator), 1.0)
    return pd.Series(disruption, index=location.index, name="disruption")
