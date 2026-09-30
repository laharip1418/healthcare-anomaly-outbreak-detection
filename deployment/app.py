"""Hosted preview of the dashboard (entry point for Streamlit Community Cloud).

It shows a fixed, precomputed snapshot of SYNTHETIC results that is bundled next to
this file. The preview only reads the two files below: it does not run Spark,
Prophet, model training or the data generator, and it never writes to the database.
Clone the repository to run the complete pipeline locally (see README.md).

Run locally:  python -m streamlit run deployment/app.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO_DATABASE = HERE / "demo_healthcare.db"
DEMO_EVALUATION = HERE / "demo_evaluation.json"

# The dashboard code lives in the project folder one level up.
sys.path.insert(0, str(HERE.parent))
import dashboard  # noqa: E402

dashboard.render(DEMO_DATABASE, DEMO_EVALUATION, preview=True)
