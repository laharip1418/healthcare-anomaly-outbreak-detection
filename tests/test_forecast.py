"""Tests for forecast.py. Every test uses a temporary database.

Run from the project folder with:  .\\.venv\\Scripts\\python.exe -m pytest
"""

import pandas as pd
import pytest

import config
import database
import forecast


def geo_rows(times_and_cases):
    """Minimal processed geo events for the given (UTC text time, case_count) pairs."""
    return pd.DataFrame([{
        "event_id": f"G-{i}", "region": "R1", "event_time": time, "created_at": time, "processed_at": time,
        "latitude": 12.97, "longitude": 77.59, "symptom_type": "fever", "severity": 2, "case_count": cases,
        "latency_seconds": 1.0,
    } for i, (time, cases) in enumerate(times_and_cases)])


def test_aggregation_builds_prophet_input():
    events = pd.DataFrame({
        "event_time": ["2026-01-01T00:00:00.000+00:00", "2026-01-01T00:05:00.000+00:00",
                       "2026-01-01T00:10:00.000+00:00", "2026-01-01T00:30:00.000+00:00",
                       "2026-01-01T00:35:00.000+00:00", "2026-01-01T00:40:00.000+00:00",
                       "2026-01-01T00:45:00.000+00:00"],
        "case_count": [2, 1, 3, 4, 1, 1, 5],
    })
    history = forecast.aggregate_case_counts(events, interval_minutes=15)
    assert list(history.columns) == ["ds", "y"]
    # 00:15 has no events (0 cases); 00:45 is left out because its ticks 00:50 and 00:55 have not arrived yet.
    assert list(history["ds"]) == list(pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:15", "2026-01-01 00:30"]))
    assert list(history["y"]) == [6.0, 0.0, 6.0]
    assert history["ds"].dt.tz is None   # Prophet needs times without a time zone


def test_negative_predictions_are_stored_as_zero():
    predictions = pd.DataFrame({
        "ds": pd.to_datetime(["2026-01-01 05:00", "2026-01-01 05:15"]),
        "yhat": [-1.5, 3.25], "yhat_lower": [-4.0, 1.0], "yhat_upper": [0.5, 5.5],
    })
    rows = forecast.prepare_forecast_rows(predictions, history_points=20, generated_at=pd.Timestamp.now(tz="UTC"))
    assert list(rows["predicted_cases"]) == [0.0, 3.25]
    assert list(rows["lower_cases"]) == [0.0, 1.0]
    assert (rows[["predicted_cases", "lower_cases", "upper_cases"]] >= 0).all().all()


def test_saved_forecast_is_replaced(tmp_path):
    db_path = tmp_path / "test.db"
    database.create_tables(db_path)
    now = pd.Timestamp.now(tz="UTC")
    first = pd.DataFrame({"forecast_time": pd.to_datetime(["2026-01-01 05:00", "2026-01-01 05:15", "2026-01-01 05:30"]),
                          "predicted_cases": [1.0, 2.0, 3.0], "lower_cases": [0.0, 1.0, 2.0],
                          "upper_cases": [2.0, 3.0, 4.0], "generated_at": now, "history_points": 20})
    second = first.head(2).assign(predicted_cases=[7.0, 8.0], history_points=24)
    database.replace_forecast(first, db_path)
    database.replace_forecast(second, db_path)
    stored = database.read_forecast(db_path)
    assert list(stored["predicted_cases"]) == [7.0, 8.0]
    assert set(stored["history_points"]) == {24}
    assert stored["forecast_time"].iloc[0] == "2026-01-01T05:00:00.000+00:00"


def test_missing_or_too_little_data_gives_helpful_errors(tmp_path):
    missing = tmp_path / "missing.db"
    with pytest.raises(forecast.ForecastError, match="spark_pipeline.py"):
        forecast.run_forecast(missing)
    assert not missing.exists()          # checking must not create the database

    db_path = tmp_path / "test.db"
    database.create_tables(db_path)
    with pytest.raises(forecast.ForecastError, match="no processed geographic events"):
        forecast.run_forecast(db_path)

    # 3 complete intervals are far fewer than FORECAST_MIN_POINTS
    times = pd.date_range("2026-01-01", periods=9, freq="5min", tz="UTC")
    database.insert_geo_events(geo_rows([(t, 1) for t in times]), db_path)
    with pytest.raises(forecast.ForecastError, match=f"at least {config.FORECAST_MIN_POINTS}"):
        forecast.run_forecast(db_path)


def test_real_prophet_forecast_on_small_history(tmp_path):
    db_path = tmp_path / "test.db"
    database.create_tables(db_path)
    ticks = pd.date_range("2026-01-01", periods=6 * 12, freq="5min", tz="UTC")   # 6 simulated hours
    database.insert_geo_events(geo_rows([(t, 1 + i % 3) for i, t in enumerate(ticks)]), db_path)

    summary = forecast.run_forecast(db_path)
    stored = database.read_forecast(db_path)
    assert summary["history_points"] == 24 and summary["forecast_points"] == config.FORECAST_HORIZON_STEPS
    assert len(stored) == config.FORECAST_HORIZON_STEPS
    assert (stored["predicted_cases"] >= 0).all()
    assert (stored["lower_cases"] <= stored["predicted_cases"]).all()
    assert (stored["predicted_cases"] <= stored["upper_cases"]).all()
    assert stored["forecast_time"].iloc[0] == "2026-01-01T06:00:00.000+00:00"
