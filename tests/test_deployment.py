"""Tests for the hosted read-only preview in deployment/ (no Spark, Prophet or training).

Run from the project folder with:  .\\.venv\\Scripts\\python.exe -m pytest
"""

import hashlib
import json
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import config
import dashboard
import database

DEPLOYMENT = config.PROJECT_ROOT / "deployment"
APP = str(DEPLOYMENT / "app.py")
DEMO_DATABASE = DEPLOYMENT / "demo_healthcare.db"
DEMO_EVALUATION = DEPLOYMENT / "demo_evaluation.json"
NOT_USED_BY_PREVIEW = ["pyspark", "prophet", "cmdstanpy", "sklearn", "joblib"]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(sql: str):
    return database.read_only_query(sql, db_path=DEMO_DATABASE).iloc[0, 0]


def test_deployment_files_exist():
    for name in ("app.py", "requirements.txt", "demo_healthcare.db", "demo_evaluation.json"):
        assert (DEPLOYMENT / name).is_file(), name


def test_deployment_files_are_not_ignored_by_git():
    # *.db files are ignored in general, so the preview snapshot needs its own exception in .gitignore
    if shutil.which("git") is None or not (config.PROJECT_ROOT / ".git").exists():
        pytest.skip("not a Git checkout")
    for name in ("app.py", "requirements.txt", "demo_healthcare.db", "demo_evaluation.json"):
        result = subprocess.run(["git", "check-ignore", "-q", f"deployment/{name}"], cwd=config.PROJECT_ROOT)
        assert result.returncode == 1, f"deployment/{name} is ignored by .gitignore"


def test_preview_requirements_contain_only_dashboard_packages():
    packages = [re.split(r"[=<>!~]", line)[0].strip().lower()
                for line in (DEPLOYMENT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")]
    assert sorted(packages) == ["numpy", "pandas", "plotly", "streamlit"]


def test_demo_database_is_a_valid_standalone_file():
    assert one("PRAGMA integrity_check") == "ok"
    assert one("PRAGMA journal_mode") == "delete"          # no WAL, so no -wal or -shm file is needed
    assert not list(DEPLOYMENT.glob("demo_healthcare.db-*"))
    for table in dashboard.TABLES:
        assert one(f"SELECT COUNT(*) FROM {table}") > 0, table


def test_demo_database_cannot_be_written_through_the_dashboard_connection():
    connection = database.connect_read_only(DEMO_DATABASE)
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            connection.execute("DELETE FROM forecasts")
    finally:
        connection.close()


def test_demo_evaluation_matches_the_saved_final_results():
    bundled = json.loads(DEMO_EVALUATION.read_text(encoding="utf-8"))
    table = dashboard.evaluation_table(bundled).set_index("Detector")
    forest = bundled["isolation_forest"]
    assert table.loc["Isolation Forest (patient vital signs)", "Precision"] == dashboard.percent(forest["precision"])
    assert int(table.loc["DBSCAN (outbreak events)", "TP"]) == bundled["dbscan"]["true_positives"]
    # the local results are generated (not in Git); when present, the bundled copy must be identical
    if config.EVALUATION_RESULTS_PATH.is_file():
        assert DEMO_EVALUATION.read_bytes() == config.EVALUATION_RESULTS_PATH.read_bytes()


def test_deployment_files_contain_no_paths_or_secrets():
    pattern = re.compile(r"[A-Za-z]:\\|\\Users\\|/Users/|/home/|password|api[_-]?key|token|BEGIN [A-Z ]*KEY", re.I)
    for path in DEPLOYMENT.iterdir():
        if path.is_file() and path.suffix != ".db":
            assert not pattern.search(path.read_text(encoding="utf-8")), path.name
    # the database is binary, so check every text value stored in it
    for table in dashboard.TABLES:
        for column in database.read_only_query(f"SELECT * FROM {table}", db_path=DEMO_DATABASE).select_dtypes("object"):
            values = database.read_only_query(f"SELECT DISTINCT {column} FROM {table}", db_path=DEMO_DATABASE)
            assert not values[column].dropna().astype(str).str.contains(pattern).any(), f"{table}.{column}"


def test_preview_renders_the_bundled_snapshot_without_writing(monkeypatch):
    before = {path.name: file_hash(path) for path in DEPLOYMENT.iterdir() if path.is_file()}

    # the preview must not need the pipeline packages or any write access to the database
    for name in NOT_USED_BY_PREVIEW:
        monkeypatch.setitem(sys.modules, name, None)          # importing it would now fail
    def no_writes(*args, **kwargs):
        raise AssertionError("the preview must not open a writable connection")
    monkeypatch.setattr(database, "connect", no_writes)
    monkeypatch.setattr(database, "create_tables", no_writes)
    # make sure the preview does not fall back to the local project files
    monkeypatch.setattr(config, "DATABASE_PATH", Path("missing-local.db"))
    monkeypatch.setattr(config, "EVALUATION_RESULTS_PATH", Path("missing-local.json"))

    app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception
    assert len(app.subheader) == 6
    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["Processed patient readings"] == f"{one('SELECT COUNT(*) FROM vitals_results'):,}"
    assert metrics["Processed geographic events"] == f"{one('SELECT COUNT(*) FROM geo_events'):,}"
    assert len(app.get("plotly_chart")) == 2                   # map and forecast
    assert "Precision" in app.dataframe[-1].value.columns       # evaluation table
    assert dashboard.PREVIEW_NOTE in [caption.value for caption in app.caption]

    app.sidebar.button[0].click().run()                        # Refresh only re-reads the snapshot
    app.toggle[0].set_value(False).run()                       # offline map
    assert not app.exception

    after = {path.name: file_hash(path) for path in DEPLOYMENT.iterdir() if path.is_file()}
    assert after == before                                     # nothing created, changed or removed


def test_preview_shows_a_short_error_when_the_snapshot_is_missing(tmp_path):
    missing_app = tmp_path / "app.py"
    missing_app.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(config.PROJECT_ROOT)!r})\n"
        "import dashboard\n"
        f"dashboard.render({str(tmp_path / 'missing.db')!r}, {str(tmp_path / 'missing.json')!r}, preview=True)\n",
        encoding="utf-8")
    app = AppTest.from_file(str(missing_app), default_timeout=60).run()
    assert not app.exception
    assert "demo snapshot is missing" in app.error[0].value
    assert not (tmp_path / "missing.db").exists()


def test_preview_shows_a_short_error_when_the_snapshot_is_not_a_database(tmp_path):
    broken = tmp_path / "broken.db"
    broken.write_bytes(b"this is not an SQLite database")
    broken_app = tmp_path / "app.py"
    broken_app.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(config.PROJECT_ROOT)!r})\n"
        "import dashboard\n"
        f"dashboard.render({str(broken)!r}, {str(tmp_path / 'missing.json')!r}, preview=True)\n",
        encoding="utf-8")
    app = AppTest.from_file(str(broken_app), default_timeout=60).run()
    assert not app.exception
    assert "could not be read" in app.error[0].value
