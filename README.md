# ParkSight

Parking enforcement intelligence for Bengaluru. Built for Flipkart Gridlock 2.0 (Round 2, Theme 1).

## What's this

BTP logs ~300k parking citations across 168 junctions but a raw hotspot map just shows where patrols already go. 83% of the variance in raw ranking comes from patrol presence (r=0.91 with effort). ParkSight corrects for that and ranks junctions by violations per patrol-day instead.

On top of that:
- Next-week forecast per cell (GRU + conformal 80% intervals) that feeds a greedy team planner
- Relocation filter to flag whether a new hotspot is real or just a moved patrol
- Shift breakdown, night vs day, catches the after-2pm coverage gap

## Numbers (out of sample)

Corrected ranking gets 9.3 violations/team vs 8.0 on raw counts (+17%), Spearman 0.69 vs 0.58. Holds after removing rejected/duplicate records (tau 0.90). Forecast beats persistence by 13% (p<0.001). Only 6 of 102 emerging hotspots are actually new.

## Setup

Needs `PARKSIGHT_SALT` env var for plate anonymization.

```
pip install -e ".[dashboard,service]"
export PARKSIGHT_SALT=any-non-empty-string
PYTHONPATH=src streamlit run src/parksight/dashboard/app.py
```

API: `PYTHONPATH=src uvicorn parksight.api.main:app`

Docker: `docker compose up --build`

## Code

All source is under `src/parksight/`:

- `ingest/` - data loading, cleaning, plate tokenization
- `impact/` - NB intensity model, EB shrinkage, Hawkes process, deployment ranking
- `forecast/` - GRU, conformal intervals, GBM cross-check
- `optimize/` - coverage planner, backtest
- `analysis/` - relocation detection, shift analysis
- `dashboard/`, `api/`, `service.py` - serving layer
