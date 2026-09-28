"""Tests for the detection functions (Isolation Forest, DBSCAN, risk score).

Run from the project folder with:  .\\.venv\\Scripts\\python.exe -m pytest
"""

import numpy as np
import pandas as pd
import pytest

import config
import detection
import generate_data


@pytest.fixture(scope="module")
def vitals():
    """Small synthetic vitals dataset (features only) for fast tests."""
    events, _labels = generate_data.generate_vitals_dataset(seed=1, num_rows=800, id_prefix="T")
    return events


def geo_points(coordinates):
    return pd.DataFrame(
        {"event_id": [f"G-{i}" for i in range(len(coordinates))],
         "latitude": [lat for lat, _ in coordinates],
         "longitude": [lon for _, lon in coordinates]}
    )


# ---------------------------------------------------------------- Isolation Forest
def test_isolation_forest_is_deterministic_with_same_seed(vitals):
    first = detection.score_vitals(detection.train_isolation_forest(vitals, seed=7), vitals)
    second = detection.score_vitals(detection.train_isolation_forest(vitals, seed=7), vitals)
    assert np.array_equal(first["anomaly_score"], second["anomaly_score"])
    assert first["predicted_anomaly"].equals(second["predicted_anomaly"])


def test_saved_model_gives_same_predictions(vitals, tmp_path):
    model = detection.train_isolation_forest(vitals)
    path = tmp_path / "model.joblib"
    detection.save_model(model, path)
    loaded = detection.load_model(path)
    before = detection.score_vitals(model, vitals)
    after = detection.score_vitals(loaded, vitals)
    assert np.array_equal(before["anomaly_score"], after["anomaly_score"])
    assert before["predicted_anomaly"].equals(after["predicted_anomaly"])


def test_extreme_vitals_score_as_more_anomalous(vitals):
    model = detection.train_isolation_forest(vitals)
    normal = vitals.head(200)
    extreme = pd.DataFrame([{
        "heart_rate": 190, "systolic_bp": 220, "diastolic_bp": 140,
        "oxygen_saturation": 72.0, "body_temperature": 41.5, "respiratory_rate": 45,
    }])
    normal_scores = detection.score_vitals(model, normal)["anomaly_score"]
    extreme_result = detection.score_vitals(model, extreme)
    assert extreme_result["anomaly_score"].iloc[0] > normal_scores.max()
    assert bool(extreme_result["predicted_anomaly"].iloc[0])


def test_label_columns_are_rejected(vitals):
    with_labels = vitals.assign(is_anomaly=False)
    with pytest.raises(ValueError, match="label"):
        detection.train_isolation_forest(with_labels)
    model = detection.train_isolation_forest(vitals)
    with pytest.raises(ValueError, match="label"):
        detection.score_vitals(model, with_labels)
    with pytest.raises(ValueError, match="label"):
        detection.find_clusters(geo_points([(12.97, 77.59)]).assign(outbreak_id="OB-001"))


def test_missing_model_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_models.py"):
        detection.load_model(tmp_path / "missing.joblib")


# ---------------------------------------------------------------- DBSCAN
def test_dbscan_finds_tight_group():
    # 8 points within about 100 m of each other, plus 3 far-away points.
    rng = np.random.default_rng(0)
    tight = [(12.970 + d1, 77.590 + d2) for d1, d2 in rng.uniform(-0.0005, 0.0005, size=(8, 2))]
    far = [(13.10, 77.40), (12.80, 77.80), (13.20, 77.90)]
    result = detection.find_clusters(geo_points(tight + far))
    tight_rows, far_rows = result.iloc[:8], result.iloc[8:]
    assert tight_rows["in_cluster"].all()
    assert tight_rows["cluster_id"].nunique() == 1 and tight_rows["cluster_id"].iloc[0] != -1
    assert not far_rows["in_cluster"].any()


def test_dbscan_labels_scattered_points_as_noise():
    # 10 points each at least ~5 km apart: none of them has enough neighbours.
    scattered = [(12.90 + 0.05 * i, 77.50 + 0.05 * (i % 3)) for i in range(10)]
    result = detection.find_clusters(geo_points(scattered))
    assert (result["cluster_id"] == -1).all()
    assert not result["in_cluster"].any()


def test_dbscan_handles_empty_input():
    result = detection.find_clusters(geo_points([]))
    assert result.empty
    assert {"cluster_id", "in_cluster"} <= set(result.columns)


# ---------------------------------------------------------------- Risk score
def test_risk_level_boundaries():
    assert detection.risk_level(0) == "Low"
    assert detection.risk_level(29.99) == "Low"
    assert detection.risk_level(30) == "Medium"
    assert detection.risk_level(59.99) == "Medium"
    assert detection.risk_level(60) == "High"
    assert detection.risk_level(100) == "High"


def test_risk_score_formula_and_components():
    assert detection.calculate_risk(0.0, 0) == {
        "anomaly_part": 0.0, "cluster_part": 0.0, "risk_score": 0.0, "risk_level": "Low"}
    # 5% anomalies = half the anomaly part; no clusters -> 25
    assert detection.calculate_risk(0.05, 0)["risk_score"] == 25.0
    # full anomaly part alone -> 50 (Medium)
    assert detection.calculate_risk(0.10, 0)["risk_level"] == "Medium"
    # 50 + 0.5 * 100 * (4 / 20) = 60 (High)
    result = detection.calculate_risk(0.10, 4)
    assert result["risk_score"] == 60.0 and result["risk_level"] == "High"
    assert result["anomaly_part"] == 1.0 and result["cluster_part"] == 0.2
    # both parts are capped at 1, so the score never exceeds 100
    assert detection.calculate_risk(0.50, 500)["risk_score"] == 100.0


def test_config_thresholds_match_documented_heuristic():
    assert config.RISK_MEDIUM_THRESHOLD == 30 and config.RISK_HIGH_THRESHOLD == 60
    assert config.RISK_ANOMALY_RATE_FOR_MAX == 0.10 and config.RISK_CLUSTER_CASES_FOR_MAX == 20
