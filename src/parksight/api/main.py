from __future__ import annotations

import json
import math

import pandas as pd
from fastapi import FastAPI, Query
from pydantic import BaseModel

from parksight import config, service

app = FastAPI(title="ParkSight", version="0.1.0")


class Hotspot(BaseModel):
    cell: str
    station: str | None
    volume: int
    priority: float
    latitude: float
    longitude: float


class Patrol(BaseModel):
    cell: str
    station: str | None
    latitude: float
    longitude: float
    marginal_gain: float


class Plan(BaseModel):
    teams: int
    volume_coverage: float
    patrols: list[Patrol]


class Transition(BaseModel):
    cell: str
    station: str | None
    status: str
    h1: int
    h2: int
    enforcement_continuity: float | None
    latitude: float
    longitude: float


class Forecast(BaseModel):
    cell: str
    latitude: float
    longitude: float
    next_week: float


class Placement(BaseModel):
    cell: str
    latitude: float
    longitude: float
    forecast: float
    marginal_gain: float


class Deployment(BaseModel):
    teams: int
    greedy_coverage: float
    naive_coverage: float
    placements: list[Placement]


class Relocation(BaseModel):
    emerging_cells: int
    device_base_rate: float
    observed_continuity: float
    null_continuity: float
    null_ci95: list[float]
    lift: float
    p_value: float
    genuine_emerging: int


class BacktestPoint(BaseModel):
    teams: int
    forecast: float
    status_quo: float
    oracle: float
    forecast_uplift: float
    headroom_captured: float


class Backtest(BaseModel):
    n_windows: int
    n_cells: int
    horizon_days: int
    overall: list[BacktestPoint]
    volatile: list[BacktestPoint]


def _station(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


def _optional(value: object) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/hotspots", response_model=list[Hotspot])
def hotspots(limit: int = Query(50, ge=1, le=500)) -> list[Hotspot]:
    top = service.intelligence().hotspots.head(limit)
    return [
        Hotspot(
            cell=cell,
            station=_station(row.station),
            volume=int(row.volume),
            priority=float(row.priority),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
        )
        for cell, row in top.iterrows()
    ]


@app.get("/plan", response_model=Plan)
def plan(teams: int = Query(12, ge=1, le=200)) -> Plan:
    intel = service.intelligence()
    chosen = intel.plan(teams)
    patrols = [
        Patrol(
            cell=cell,
            station=_station(row.station),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            marginal_gain=float(row.marginal_gain),
        )
        for cell, row in chosen.iterrows()
    ]
    return Plan(teams=teams, volume_coverage=intel.volume_coverage(chosen), patrols=patrols)


@app.get("/transitions", response_model=list[Transition])
def transitions(
    status: str = "emerging",
    limit: int = Query(50, ge=1, le=500),
) -> list[Transition]:
    table = service.intelligence().transitions
    subset = table[table["status"] == status].head(limit)
    return [
        Transition(
            cell=cell,
            station=_station(row.station),
            status=row.status,
            h1=int(row.h1),
            h2=int(row.h2),
            enforcement_continuity=_optional(row.enforcement_continuity),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
        )
        for cell, row in subset.iterrows()
    ]


@app.get("/forecast", response_model=list[Forecast])
def forecast(limit: int = Query(50, ge=1, le=500)) -> list[Forecast]:
    if not config.FORECAST_PATH.exists():
        return []
    payload = json.loads(config.FORECAST_PATH.read_text())
    ranked = sorted(payload["forecast"].items(), key=lambda item: item[1], reverse=True)
    result = []
    for cell, value in ranked[:limit]:
        latitude, longitude = (float(part) for part in cell.split(","))
        result.append(Forecast(cell=cell, latitude=latitude, longitude=longitude, next_week=float(value)))
    return result


@app.get("/relocation", response_model=Relocation)
def relocation() -> Relocation:
    return Relocation(**service.relocation())


@app.get("/shifts")
def shifts() -> dict | None:
    return service.shifts()


@app.get("/conformal")
def conformal() -> dict | None:
    return service.conformal()


@app.get("/impact")
def impact() -> dict | None:
    return service.impact()


@app.get("/backtest", response_model=Backtest | None)
def backtest() -> Backtest | None:
    report = service.deployment_backtest()
    return Backtest(**report) if report else None


@app.get("/deployment", response_model=Deployment)
def deployment(teams: int = Query(12, ge=1, le=200)) -> Deployment:
    result = service.forecast_deployment(teams)
    placements = [
        Placement(
            cell=str(row.cell),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
            forecast=float(row.forecast),
            marginal_gain=float(row.marginal_gain),
        )
        for _, row in result.plan.iterrows()
    ]
    return Deployment(
        teams=teams,
        greedy_coverage=result.greedy_coverage,
        naive_coverage=result.naive_coverage,
        placements=placements,
    )
