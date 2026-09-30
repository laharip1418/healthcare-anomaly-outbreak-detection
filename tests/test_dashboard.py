"""Tests for the dashboard's data loading and a headless run of dashboard.py.

Run from the project folder with:  .\\.venv\\Scripts\\python.exe -m pytest
"""

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
import dashboard
import database

DASHBOARD = str(Path(__file__).resolve().parent.parent / "dashboard.py")


def fill_database(db_path):
    """A few stored results, as the pipeline and forecast.py would write them."""
    database.create_tables(db_path)
    times = pd.date_range("2026-01-01", periods=48, freq="5min", tz="UTC")
    database.insert_vital_results(pd.DataFrame([{
        "event_id": f"V-{i}", "patient_id": f"P-{i % 5}", "region": "R1" if i % 2 else "R2",
        "event_time": t, "created_at": t, "processed_at": t + timedelta(seconds=2),
        "heart_rate": 150 if i == 3 else 75, "systolic_bp": 120, "diastolic_bp": 80, "oxygen_saturation": 97.0,
        "body_temperature": 36.8, "respiratory_rate": 16, "latitude": 12.97, "longitude": 77.59,
        "anomaly_score": 0.2 if i == 3 else -0.1, "predicted_anomaly": i == 3, "latency_seconds": 2.0,
    } for i, t in enumerate(times)]), db_path)
    database.insert_geo_events(pd.DataFrame([{
        "event_id": f"G-{i}", "region": "R3", "event_time": t, "created_at": t,
        "processed_at": t + timedelta(seconds=1), "latitude": 12.97, "longitude": 77.66,
        "symptom_type": "fever", "severity": 2, "case_count": 2, "latency_seconds": 1.0,
    } for i, t in enumerate(times)]), db_path)
    # G-30 was clustered by the older window below; the latest window found no clusters
    database.update_cluster_assignments(pd.DataFrame([
        {"event_id": "G-30", "cluster_id": "20260101T0300-0", "in_cluster": True}]), db_path)
    database.save_region_risk(pd.DataFrame([
        {"region": "R1", "window_end": times[-1], "calculated_at": times[-1], "anomaly_rate": 0.0,
         "clustered_cases": 0, "anomaly_part": 0.0, "cluster_part": 0.0, "risk_score": 0.0, "risk_level": "Low"},
        {"region": "R3", "window_end": times[-1], "calculated_at": times[-1], "anomaly_rate": 0.1,
         "clustered_cases": 8, "anomaly_part": 1.0, "cluster_part": 0.4, "risk_score": 70.0, "risk_level": "High"},
    ]), db_path)
    # a cluster from an OLDER window: the latest window (times[-1]) found no clusters
    database.save_clusters(pd.DataFrame([{
        "cluster_id": "20260101T0300-0", "window_end": times[36], "calculated_at": times[36], "region": "R3",
        "center_latitude": 12.97, "center_longitude": 77.66, "event_count": 6, "total_cases": 12,
    }]), db_path)
    database.replace_forecast(pd.DataFrame({
        "forecast_time": pd.date_range("2026-01-01 04:00", periods=8, freq="15min"),
        "predicted_cases": 6.0, "lower_cases": 4.0, "upper_cases": 8.0,
        "generated_at": pd.Timestamp.now(tz="UTC"), "history_points": 16,
    }), db_path)


def test_missing_database_is_reported_and_not_created(tmp_path):
    missing = tmp_path / "missing.db"
    assert dashboard.database_status(missing) == "missing"
    assert not missing.exists()


def test_data_loading_handles_an_empty_database(tmp_path):
    db_path = tmp_path / "test.db"
    database.create_tables(db_path)
    data = dashboard.load_data(db_path)
    assert all(frame.empty for frame in data.values())
    summary = dashboard.summarize(data)
    assert summary["patient_events"] == 0 and summary["top_risk_score"] is None
    assert dashboard.latency_summary(data["vitals_results"]) is None


def test_data_loading_returns_the_stored_values(tmp_path):
    db_path = tmp_path / "test.db"
    fill_database(db_path)
    data = dashboard.load_data(db_path)
    summary = dashboard.summarize(data)
    assert summary["patient_events"] == 48 and summary["patient_anomalies"] == 1
    assert summary["geo_events"] == 48
    assert summary["top_risk_score"] == 70.0 and summary["top_risk_level"] == "High" and summary["top_risk_region"] == "R3"
    # the only cluster belongs to an older window, so the latest window has none ...
    assert summary["clusters_latest_window"] == 0 and summary["cluster_records"] == 1
    # ... while the event it clustered is still shown as clustered on the map
    assert list(data["geo_events"].loc[data["geo_events"]["in_cluster"] == 1, "event_id"]) == ["G-30"]
    latency = dashboard.latency_summary(data["vitals_results"])
    assert latency["median"] == 2.0 and latency["max"] == 2.0
    assert latency["throughput"] == pytest.approx(48 / (47 * 300))
    flagged = dashboard.filter_vitals(data["vitals_results"], data["vitals_results"]["time"].min(),
                                      data["vitals_results"]["time"].max(), ["R1", "R2"], flagged_only=True)
    assert list(flagged["event_id"]) == ["V-3"]


def page_text(app) -> str:
    """All visible text of a rendered AppTest page, to check what it shows."""
    parts = [element.value for kind in ("title", "subheader", "caption", "markdown", "info", "warning")
             for element in getattr(app, kind)]
    parts += [f"{metric.label} {metric.value}" for metric in app.metric]
    parts += [frame.value.to_string() for frame in app.dataframe]
    return "\n".join(str(part) for part in parts)


def test_percentages_are_display_only():
    assert [dashboard.percent(v) for v in (0.7042, 0.8475, 0.7692)] == ["70.4%", "84.8%", "76.9%"]


def test_dashboard_runs_without_a_database(monkeypatch, tmp_path):
    # AppTest runs dashboard.py fresh in this process, so it sees the patched config paths.
    monkeypatch.setattr(config, "DATABASE_PATH", tmp_path / "missing.db")
    monkeypatch.setattr(config, "EVALUATION_RESULTS_PATH", tmp_path / "missing.json")
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not app.exception
    assert not app.warning
    assert any("spark_pipeline.py" in info.value for info in app.info)
    assert str(tmp_path) not in page_text(app)
    assert not (tmp_path / "missing.db").exists()


def test_dashboard_renders_all_sections_from_stored_results(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    fill_database(db_path)
    results_path = tmp_path / "evaluation.json"
    results_path.write_text(json.dumps({
        "created_at": "2026-09-28T20:14:02+00:00", "seeds": {"train": 42, "test": 43},
        "isolation_forest": {"test_rows": 1200, "precision": 0.7042, "recall": 0.8475, "f1": 0.7692,
                             "false_positive_rate": 0.0184,
                             "true_positives": 50, "false_positives": 21, "false_negatives": 9, "true_negatives": 1120},
        "dbscan": {"test_events": 10, "outbreaks": 1, "outbreaks_detected": 1, "precision": 0.5, "recall": 0.5,
                   "f1": 0.5, "true_positives": 1, "false_positives": 1, "false_negatives": 1, "true_negatives": 7},
    }), encoding="utf-8")
    monkeypatch.setattr(config, "DATABASE_PATH", db_path)
    monkeypatch.setattr(config, "EVALUATION_RESULTS_PATH", results_path)
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()

    assert not app.exception
    assert not app.warning                                  # no warning banner
    assert [header.value for header in app.subheader] == [
        "1. System summary", "2. Recent patient readings", "3. Geographic activity",
        "4. Case history and forecast", "5. Model evaluation", "6. Processing performance"]
    captions = [caption.value for caption in app.main.caption]
    assert captions[0] == "Synthetic healthcare monitoring dashboard"
    assert captions[-1] == "Academic project using simulated data."
    assert "Held-out synthetic test data" in captions
    assert str(tmp_path) not in page_text(app) and str(config.PROJECT_ROOT) not in page_text(app)

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Processed patient readings"] == "48"
    assert metrics["Active clusters — latest window"] == "0"   # newest window only
    assert metrics["Highest regional risk score"] == "70.0 / 100"
    assert metrics["Risk level — East District"] == "High"

    # processing performance: eight separate metrics, patient readings first
    performance = [(m.label, m.value) for m in app.metric][6:]
    assert performance == [
        ("Median latency", "2.00 s"), ("95th percentile", "2.00 s"), ("Maximum latency", "2.00 s"),
        ("Throughput (events/s)", f"{48 / (47 * 300):.1f}"),
        ("Median latency", "1.00 s"), ("95th percentile", "1.00 s"), ("Maximum latency", "1.00 s"),
        ("Throughput (events/s)", f"{48 / (47 * 300):.1f}"),
    ]

    # tables: flagged readings, regional risk, evaluation (no active clusters, so that shows a message)
    assert len(app.dataframe) == 3
    evaluation = app.dataframe[2].value.set_index("Detector")
    forest = evaluation.loc["Isolation Forest (patient vital signs)"]
    assert (forest["Precision"], forest["Recall"], forest["F1"]) == ("70.4%", "84.8%", "76.9%")
    assert forest["False-positive rate"] == "1.8%"
    # a results file from an older train_models.py has no false-positive rate: shown as a dash
    assert evaluation.loc["DBSCAN (outbreak events)", "False-positive rate"] == "—"
    assert (forest["TP"], forest["FP"], forest["FN"], forest["TN"]) == (50, 21, 9, 1120)
    assert any("No active clusters" in info.value for info in app.info)
    assert app.expander[0].label == "About the metrics"

    # the map legend separates historical cluster membership from the active count
    map_traces = {trace.get("name") for trace in json.loads(app.get("plotly_chart")[0].proto.spec)["data"]}
    assert {"Clustered when processed", "Not clustered"} <= map_traces


def test_refresh_button_loads_new_results(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    fill_database(db_path)
    monkeypatch.setattr(config, "DATABASE_PATH", db_path)
    monkeypatch.setattr(config, "EVALUATION_RESULTS_PATH", tmp_path / "missing.json")
    app = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert {m.label: m.value for m in app.metric}["Processed patient readings"] == "48"

    # new results arrive while the page is open (as when the pipeline runs again)
    later = pd.Timestamp("2026-01-01 04:00", tz="UTC")
    database.insert_vital_results(pd.DataFrame([{
        "event_id": "V-new", "patient_id": "P-1", "region": "R1", "event_time": later, "created_at": later,
        "processed_at": later + timedelta(seconds=2), "heart_rate": 75, "systolic_bp": 120, "diastolic_bp": 80,
        "oxygen_saturation": 97.0, "body_temperature": 36.8, "respiratory_rate": 16, "latitude": 12.97,
        "longitude": 77.59, "anomaly_score": -0.1, "predicted_anomaly": False, "latency_seconds": 2.0,
    }]), db_path)
    app.sidebar.button[0].click().run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric}["Processed patient readings"] == "49"
