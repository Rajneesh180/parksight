from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from parksight import config
from parksight.analysis.emergence import relocation_report, transitions
from parksight.ingest import loader
from parksight.optimize.coverage import covered_value, maximal_coverage, top_k
from parksight.scoring.priority import PriorityWeights, blend_priority, rank_hotspots


@dataclass(frozen=True)
class Ranking:
    hotspots: pd.DataFrame
    plan: pd.DataFrame
    coverage: float


@dataclass(frozen=True)
class Intelligence:
    hotspots: pd.DataFrame
    transitions: pd.DataFrame

    def plan(self, teams: int) -> pd.DataFrame:
        return maximal_coverage(self.hotspots, teams)

    def volume_coverage(self, plan: pd.DataFrame) -> float:
        return covered_value(self.hotspots, plan, "volume")

    def reprioritize(self, weights: PriorityWeights, teams: int) -> Ranking:
        scored = self.hotspots.assign(
            priority=blend_priority(self.hotspots, weights)
        ).sort_values("priority", ascending=False)
        placements = maximal_coverage(scored, teams)
        return Ranking(scored, placements, covered_value(scored, placements, "volume"))


@lru_cache(maxsize=4)
def build(source: str = "sample") -> Intelligence:
    frame = loader.load_violations(config.source_path(source))
    return Intelligence(hotspots=rank_hotspots(frame), transitions=transitions(frame))


@lru_cache(maxsize=1)
def load_full() -> Intelligence:
    hotspots = pd.read_parquet(config.HOTSPOTS_PATH).set_index("cell")
    moving = pd.read_parquet(config.TRANSITIONS_PATH).set_index("cell")
    return Intelligence(hotspots=hotspots, transitions=moving)


def intelligence() -> Intelligence:
    if config.HOTSPOTS_PATH.exists() and config.TRANSITIONS_PATH.exists():
        return load_full()
    return build("sample")


@lru_cache(maxsize=1)
def relocation() -> dict:
    if config.RELOCATION_PATH.exists():
        return json.loads(config.RELOCATION_PATH.read_text())
    return relocation_report(loader.load_violations(config.SAMPLE_DATA_PATH))


@lru_cache(maxsize=1)
def shifts() -> dict | None:
    if config.SHIFTS_PATH.exists():
        return json.loads(config.SHIFTS_PATH.read_text())
    return None


@lru_cache(maxsize=1)
def conformal() -> dict | None:
    if config.CONFORMAL_PATH.exists():
        return json.loads(config.CONFORMAL_PATH.read_text())
    return None


@lru_cache(maxsize=1)
def impact() -> dict | None:
    if config.IMPACT_PATH.exists():
        return json.loads(config.IMPACT_PATH.read_text())
    return None


@lru_cache(maxsize=1)
def forecast_cells() -> pd.DataFrame:
    payload = json.loads(config.FORECAST_PATH.read_text())
    rows = [
        {
            "cell": cell,
            "latitude": float(cell.split(",")[0]),
            "longitude": float(cell.split(",")[1]),
            "forecast": value,
        }
        for cell, value in payload["forecast"].items()
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Deployment:
    plan: pd.DataFrame
    greedy_coverage: float
    naive_coverage: float


@lru_cache(maxsize=1)
def deployment_backtest() -> dict | None:
    if not config.BACKTEST_PATH.exists():
        return None
    return json.loads(config.BACKTEST_PATH.read_text())


def forecast_deployment(teams: int) -> Deployment:
    cells = forecast_cells()
    greedy = maximal_coverage(cells, teams, value="forecast")
    naive = top_k(cells, teams, value="forecast")
    return Deployment(
        plan=greedy,
        greedy_coverage=covered_value(cells, greedy, "forecast"),
        naive_coverage=covered_value(cells, naive, "forecast"),
    )
