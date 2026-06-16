from __future__ import annotations

import os
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _PACKAGE_DIR.parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

_DEFAULT_RAW = WORKSPACE_ROOT / "Round 2" / "jan to may police violation_anonymized791b166.csv"

RAW_DATA_PATH = Path(os.environ.get("PARKSIGHT_DATA", _DEFAULT_RAW))
SAMPLE_DATA_PATH = REPO_ROOT / "data" / "sample" / "violations_sample.csv"
FINDINGS_PATH = REPO_ROOT / "data" / "findings_full.json"
RELOCATION_PATH = REPO_ROOT / "data" / "relocation_full.json"
SHIFTS_PATH = REPO_ROOT / "data" / "shifts_full.json"
FORECAST_PATH = REPO_ROOT / "data" / "forecast_full.json"
FORECAST_EVAL_PATH = REPO_ROOT / "data" / "forecast_eval.npz"
SEED_ROBUSTNESS_PATH = REPO_ROOT / "data" / "seed_robustness.json"
CONFORMAL_PATH = REPO_ROOT / "data" / "conformal_full.json"
BACKTEST_PATH = REPO_ROOT / "data" / "deployment_backtest.json"
HOTSPOTS_PATH = REPO_ROOT / "data" / "hotspots_full.parquet"
TRANSITIONS_PATH = REPO_ROOT / "data" / "transitions_full.parquet"
IMPACT_PATH = REPO_ROOT / "data" / "impact_full.json"
REPORTS_DIR = REPO_ROOT / "reports"


def source_path(source: str) -> Path:
    return {"full": RAW_DATA_PATH, "sample": SAMPLE_DATA_PATH}.get(source, Path(source))


IST = "Asia/Kolkata"
CHURN_SPLIT = "2024-01-23"
CELL_DECIMALS = 3
LAT_BOUNDS = (12.6, 13.4)
LON_BOUNDS = (77.3, 77.9)
SALT = os.environ.get("PARKSIGHT_SALT")
if SALT is None:
    raise RuntimeError("PARKSIGHT_SALT must be set; it keys the plate token and is never committed")

SEVERITY: dict[str, float] = {
    "DOUBLE PARKING": 1.0,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 1.0,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 0.9,
    "PARKING NEAR ROAD CROSSING": 0.9,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.9,
    "PARKING IN A MAIN ROAD": 0.8,
    "NO PARKING": 0.5,
    "WRONG PARKING": 0.5,
    "PARKING ON FOOTPATH": 0.3,
    "PARKING OTHER THAN BUS STOP": 0.3,
}
