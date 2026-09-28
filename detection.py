"""Detection functions shared by train_models.py, spark_pipeline.py and the tests.

All data in this project is SYNTHETIC and intended only for academic demonstration.

1. Isolation Forest   - flags unusual combinations of patient vital signs.
2. DBSCAN             - finds tight geographic groups of distress events.
3. Risk score         - a simple academic heuristic combining both results.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

EARTH_RADIUS_KM = 6371.0

# Ground-truth columns written by generate_data.py. They must never be used as
# model features, so the functions below refuse any DataFrame that contains them.
LABEL_COLUMNS = {"is_anomaly", "anomaly_type", "is_corrupt", "corruption_type", "is_outbreak_event", "outbreak_id"}


def check_no_label_columns(events: pd.DataFrame) -> None:
    """Raise an error if ground-truth label columns were passed in by mistake."""
    found = sorted(LABEL_COLUMNS & set(events.columns))
    if found:
        raise ValueError(f"Ground-truth label columns must not be given to the model: {found}")


# ---------------------------------------------------------------------------
# 1. Isolation Forest (patient vital signs)
# ---------------------------------------------------------------------------
def train_isolation_forest(events: pd.DataFrame, seed: int = config.MODEL_RANDOM_SEED) -> Pipeline:
    """Train a StandardScaler + IsolationForest pipeline on vital signs.

    Isolation Forest is UNSUPERVISED: it only sees the six vital-sign columns and
    never the ground-truth labels. It learns what typical readings look like and
    isolates readings that are easy to separate from the rest.
    """
    check_no_label_columns(events)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("forest", IsolationForest(
            n_estimators=config.ISOLATION_FOREST_TREES,
            contamination=config.ISOLATION_FOREST_CONTAMINATION,
            random_state=seed,
        )),
    ])
    model.fit(events[config.VITAL_FEATURES])
    return model


def score_vitals(model: Pipeline, events: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the events with two new columns.

    anomaly_score      higher = more unusual; above 0 means the model flags it
    predicted_anomaly  True if the model flags the reading as anomalous
    """
    check_no_label_columns(events)
    features = events[config.VITAL_FEATURES]
    result = events.copy()
    # decision_function is negative for anomalies, so flip the sign to make
    # "higher = more anomalous", which is easier to read on the dashboard.
    result["anomaly_score"] = -model.decision_function(features)
    result["predicted_anomaly"] = model.predict(features) == -1
    return result


def save_model(model: Pipeline, path: Path = config.ISOLATION_FOREST_PATH) -> None:
    """Save the trained pipeline with joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path = config.ISOLATION_FOREST_PATH) -> Pipeline:
    """Load a trained pipeline, with a helpful message if it does not exist yet."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Run: python train_models.py")
    return joblib.load(path)


# ---------------------------------------------------------------------------
# 2. DBSCAN (geographic clusters)
# ---------------------------------------------------------------------------
def find_clusters(
    events: pd.DataFrame,
    eps_km: float = config.DBSCAN_EPS_KM,
    min_samples: int = config.DBSCAN_MIN_SAMPLES,
) -> pd.DataFrame:
    """Cluster geographic events by latitude and longitude with DBSCAN.

    Returns a copy of the events with two new columns:
        cluster_id   cluster number, or -1 for noise (not part of any cluster)
        in_cluster   True if the event belongs to a detected cluster
    Cluster numbers are only meaningful within one call.
    """
    check_no_label_columns(events)
    result = events.copy()
    if result.empty:
        result["cluster_id"] = pd.Series(dtype="int64")
        result["in_cluster"] = pd.Series(dtype="bool")
        return result

    # The haversine metric expects [latitude, longitude] in radians, and eps as
    # an angle: distance in km divided by the Earth's radius in km.
    coordinates = np.radians(result[["latitude", "longitude"]].to_numpy())
    dbscan = DBSCAN(
        eps=eps_km / EARTH_RADIUS_KM,
        min_samples=min_samples,
        metric="haversine",
        algorithm="ball_tree",
    )
    result["cluster_id"] = dbscan.fit_predict(coordinates)
    result["in_cluster"] = result["cluster_id"] != -1   # DBSCAN uses -1 for noise
    return result


# ---------------------------------------------------------------------------
# 3. Risk score (simple academic heuristic)
# ---------------------------------------------------------------------------
def risk_level(risk_score: float) -> str:
    """Low below 30, Medium from 30 up to (not including) 60, High at 60 or above."""
    if risk_score >= config.RISK_HIGH_THRESHOLD:
        return "High"
    if risk_score >= config.RISK_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def calculate_risk(anomaly_rate: float, clustered_cases: float) -> dict:
    """Combine a region's anomaly rate and clustered cases into a 0-100 score.

    This is a simple ACADEMIC HEURISTIC for demonstration. It is NOT a
    probability, NOT a medical prediction and NOT clinically validated.

        anomaly_part = min(anomaly_rate / 0.10, 1)
        cluster_part = min(clustered_cases / 20, 1)
        risk_score   = 100 * (0.5 * anomaly_part + 0.5 * cluster_part)
    """
    anomaly_part = min(anomaly_rate / config.RISK_ANOMALY_RATE_FOR_MAX, 1.0)
    cluster_part = min(clustered_cases / config.RISK_CLUSTER_CASES_FOR_MAX, 1.0)
    risk_score = round(100 * (0.5 * anomaly_part + 0.5 * cluster_part), 2)
    return {
        "anomaly_part": round(anomaly_part, 4),
        "cluster_part": round(cluster_part, 4),
        "risk_score": risk_score,
        "risk_level": risk_level(risk_score),
    }
