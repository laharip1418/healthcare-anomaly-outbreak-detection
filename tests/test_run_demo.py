"""Tests for run_demo.py that do not start Spark (the real run is described in README.md).

Run from the project folder with:  .\\.venv\\Scripts\\python.exe -m pytest
"""

import json
from types import SimpleNamespace

import pytest

import config
import database
import run_demo
from test_dashboard import fill_database


@pytest.fixture
def fake_steps(monkeypatch):
    """Record the scripts run_demo would start instead of running them."""
    calls = []
    monkeypatch.setattr(run_demo, "check_prerequisites", lambda: None)
    monkeypatch.setattr(run_demo, "check_results", lambda: {})
    monkeypatch.setattr(run_demo, "print_summary", lambda *args: None)

    def fake_run(command, cwd):
        calls.append(command[1:])
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(run_demo.subprocess, "run", fake_run)
    return calls


def write_labels(labels_dir, written, corrupt):
    labels_dir.mkdir()
    with open(labels_dir / "vitals_labels.jsonl", "w", encoding="utf-8") as file:
        for number in range(written):
            file.write(json.dumps({"event_id": f"V-{number}", "is_corrupt": number < corrupt}) + "\n")


def test_default_ticks_give_enough_history_for_the_forecast():
    assert run_demo.MIN_TICKS == 36
    assert run_demo.MIN_TICKS <= run_demo.DEFAULT_TICKS <= run_demo.MAX_TICKS


@pytest.mark.parametrize("ticks", ["10", "5000"])
def test_tick_count_outside_the_allowed_range_is_refused(ticks):
    with pytest.raises(SystemExit) as error:
        run_demo.main(["--ticks", ticks, "--yes"])
    assert error.value.code == 2


def test_steps_run_in_order_with_the_selected_seed(fake_steps):
    assert run_demo.main(["--seed", "101", "--ticks", "40", "--yes"]) == 0
    assert fake_steps == [
        ["generate_data.py", "--mode", "stream", "--fresh", "--seed", "101", "--ticks", "40",
         "--seconds-per-tick", "0"],
        ["spark_pipeline.py", "--mode", "once", "--reset"],
        ["forecast.py"],
    ]


def test_default_seed_is_44(fake_steps):
    assert run_demo.main(["--yes"]) == 0
    assert fake_steps[0][fake_steps[0].index("--seed") + 1] == "44" == str(config.STREAM_SEED)


def test_nothing_is_replaced_without_confirmation(fake_steps, monkeypatch):
    monkeypatch.setattr(run_demo.sys.stdin, "isatty", lambda: False)
    assert run_demo.main(["--seed", "101"]) == 1
    assert fake_steps == []


def test_a_failing_step_stops_the_demo_with_a_nonzero_exit(fake_steps, monkeypatch):
    def failing_spark(command, cwd):
        fake_steps.append(command[1:])
        return SimpleNamespace(returncode=1 if command[1] == "spark_pipeline.py" else 0)
    monkeypatch.setattr(run_demo.subprocess, "run", failing_spark)
    assert run_demo.main(["--yes"]) == 1
    assert [call[0] for call in fake_steps] == ["generate_data.py", "spark_pipeline.py"]   # forecast.py never ran


def test_results_are_checked_and_counted(tmp_path):
    db_path, labels_dir = tmp_path / "demo.db", tmp_path / "labels"
    fill_database(db_path)
    write_labels(labels_dir, written=50, corrupt=2)
    results = run_demo.check_results(db_path, labels_dir)
    assert results["integrity"] == "ok"
    assert results["patient_readings"] == 48 and results["rejected_readings"] == 2
    assert results["corrupt_readings_written"] == 2
    assert results["anomalies"] == 1 and results["geo_events"] == 48 and results["risk_rows"] == 2
    assert results["latest_window_clusters"] == 0 and results["cluster_records"] == 1
    assert results["forecast_rows"] == 8


def test_missing_results_make_the_check_fail(tmp_path):
    db_path, labels_dir = tmp_path / "demo.db", tmp_path / "labels"
    database.create_tables(db_path)
    write_labels(labels_dir, written=1, corrupt=0)
    with pytest.raises(run_demo.DemoError, match="no rows in vitals_results"):
        run_demo.check_results(db_path, labels_dir)
    with pytest.raises(run_demo.DemoError, match="No database"):
        run_demo.check_results(tmp_path / "missing.db", labels_dir)


def test_paths_are_shown_relative_to_the_project():
    assert run_demo.relative(config.DATABASE_PATH) == "data/healthcare.db"
