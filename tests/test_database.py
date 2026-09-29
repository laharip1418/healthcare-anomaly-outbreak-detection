"""Tests for the SQLite storage functions. Every test uses a temporary database.

Run from the project folder with:  .\\.venv\\Scripts\\python.exe -m pytest
"""

import sqlite3

import pandas as pd
import pytest

import config
import database


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "test.db"
    database.create_tables(path)
    return path


def vital_result(event_id="V-00000001"):
    return pd.DataFrame([{
        "event_id": event_id, "patient_id": "P-0001", "region": "R1",
        "event_time": pd.Timestamp("2026-01-01 00:05:00"),               # naive = UTC, as Spark gives it
        "created_at": "2026-09-28T19:37:53.517+00:00",
        "processed_at": pd.Timestamp("2026-09-28T19:37:55.017", tz="UTC"),
        "heart_rate": 72, "systolic_bp": 120, "diastolic_bp": 80, "oxygen_saturation": 97.5,
        "body_temperature": 36.8, "respiratory_rate": 16, "latitude": 12.97, "longitude": 77.59,
        "anomaly_score": -0.12, "predicted_anomaly": False, "latency_seconds": 1.5,
    }])


def geo_event(event_id="G-00000001"):
    return pd.DataFrame([{
        "event_id": event_id, "region": "R2", "event_time": "2026-01-01T00:05:00+00:00",
        "created_at": "2026-09-28T19:37:53.517+00:00", "processed_at": "2026-09-28T19:37:54.517+00:00",
        "latitude": 12.91, "longitude": 77.59, "symptom_type": "fever", "severity": 2, "case_count": 3,
        "latency_seconds": 1.0,
    }])


def test_database_file_and_tables_are_created(db_path):
    assert db_path.exists()
    connection = sqlite3.connect(db_path)
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    connection.close()
    assert {"vitals_results", "geo_events", "clusters", "region_risk"} <= tables


def test_wal_mode_is_enabled(db_path):
    connection = sqlite3.connect(db_path)
    mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
    connection.close()
    assert mode.lower() == "wal"


def test_test_database_is_temporary(db_path):
    assert db_path != config.DATABASE_PATH
    assert config.DATA_DIR not in db_path.parents


def test_vital_result_can_be_inserted_and_read(db_path):
    assert database.insert_vital_results(vital_result(), db_path) == 1
    rows = database.read_vitals(db_path=db_path)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["event_id"] == "V-00000001" and row["heart_rate"] == 72
    assert row["predicted_anomaly"] == 0
    # all timestamps are stored as the same UTC text format
    assert row["event_time"] == "2026-01-01T00:05:00.000+00:00"
    assert row["created_at"] == "2026-09-28T19:37:53.517+00:00"
    assert row["processed_at"] == "2026-09-28T19:37:55.017+00:00"


def test_geo_event_can_be_inserted_and_read(db_path):
    assert database.insert_geo_events(geo_event(), db_path) == 1
    rows = database.read_geo_events(db_path=db_path)
    assert len(rows) == 1
    assert rows.iloc[0]["symptom_type"] == "fever"
    assert rows.iloc[0]["in_cluster"] == 0 and rows.iloc[0]["cluster_id"] is None
    assert database.latest_geo_event_time(db_path) == "2026-01-01T00:05:00.000+00:00"


def test_reinserting_same_event_id_does_not_add_rows(db_path):
    database.insert_vital_results(vital_result(), db_path)
    database.insert_geo_events(geo_event(), db_path)
    # a replay after a Spark restart sends the same events again
    assert database.insert_vital_results(vital_result(), db_path) == 0
    assert database.insert_geo_events(geo_event(), db_path) == 0
    assert database.count_rows("vitals_results", db_path) == 1
    assert database.count_rows("geo_events", db_path) == 1


def test_cluster_assignment_and_summary_can_be_saved(db_path):
    database.insert_geo_events(geo_event(), db_path)
    assignment = pd.DataFrame([{"event_id": "G-00000001", "cluster_id": "20260101T0005-0", "in_cluster": True}])
    assert database.update_cluster_assignments(assignment, db_path) == 1
    assert database.read_geo_events(db_path=db_path).iloc[0]["in_cluster"] == 1

    summary = pd.DataFrame([{
        "cluster_id": "20260101T0005-0", "window_end": "2026-01-01T00:05:00+00:00",
        "calculated_at": "2026-09-28T19:37:55+00:00", "region": "R2",
        "center_latitude": 12.91, "center_longitude": 77.59, "event_count": 6, "total_cases": 14,
    }])
    database.save_clusters(summary, db_path)
    database.save_clusters(summary, db_path)          # saving the same window again replaces it
    clusters = database.read_clusters(db_path)
    assert len(clusters) == 1 and clusters.iloc[0]["total_cases"] == 14


def test_region_risk_can_be_saved_and_read(db_path):
    risk = pd.DataFrame([{
        "region": "R1", "window_end": "2026-01-01T01:00:00+00:00", "calculated_at": "2026-09-28T19:40:00+00:00",
        "anomaly_rate": 0.1, "clustered_cases": 4, "anomaly_part": 1.0, "cluster_part": 0.2,
        "risk_score": 60.0, "risk_level": "High",
    }])
    database.save_region_risk(risk, db_path)
    database.save_region_risk(risk, db_path)
    rows = database.read_region_risk(db_path)
    assert len(rows) == 1
    assert rows.iloc[0]["risk_level"] == "High" and rows.iloc[0]["cluster_part"] == 0.2


def test_read_vitals_uses_time_window(db_path):
    early, late = vital_result("V-1"), vital_result("V-2")
    late["event_time"] = pd.Timestamp("2026-01-01 02:00:00")
    database.insert_vital_results(early, db_path)
    database.insert_vital_results(late, db_path)
    window = database.read_vitals("2026-01-01T01:00:00.000+00:00", "2026-01-01T02:00:00.000+00:00", db_path)
    assert list(window["event_id"]) == ["V-2"]
