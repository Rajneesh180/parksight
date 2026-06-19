.PHONY: setup sample audit relocation shifts impact forecast seed-robustness conformal backtest sensitivity serve dashboard test lint format typecheck

export PARKSIGHT_SALT ?= parksight-local

setup:
	pip install -e ".[dev]"

sample:
	PYTHONPATH=src python scripts/make_sample.py

audit:
	PYTHONPATH=src python scripts/run_audit.py --source full

relocation:
	PYTHONPATH=src python scripts/run_relocation.py --source full

shifts:
	PYTHONPATH=src python scripts/run_shifts.py --source full

impact:
	PYTHONPATH=src python scripts/run_impact.py --source full

forecast:
	PYTHONPATH=src python scripts/run_forecast.py --source full

seed-robustness:
	PYTHONPATH=src python scripts/run_seed_robustness.py --source full

conformal:
	PYTHONPATH=src python scripts/run_conformal.py

backtest:
	PYTHONPATH=src python scripts/run_backtest.py

sensitivity:
	PYTHONPATH=src python scripts/run_sensitivity.py

serve:
	PYTHONPATH=src uvicorn parksight.api.main:app --reload

dashboard:
	PYTHONPATH=src streamlit run src/parksight/dashboard/app.py

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy src
