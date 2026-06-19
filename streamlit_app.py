import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

salt = "streamlit-demo-salt"
try:
    salt = st.secrets["PARKSIGHT_SALT"]
except Exception:
    pass
os.environ.setdefault("PARKSIGHT_SALT", salt)

runpy.run_module("parksight.dashboard.app", run_name="__main__")
