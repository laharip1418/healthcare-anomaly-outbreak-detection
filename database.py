"""SQLite storage for processed results (built-in sqlite3 module only).

spark_pipeline.py writes to the database and the dashboard only reads it.
All data stored here is SYNTHETIC and intended only for academic demonstration.

Tables
------
vitals_results  one row per validated vital-sign event, with its anomaly score
geo_events      one row per validated geographic event, with its latest cluster
clusters        one summary row per DBSCAN cluster found in a recent window
region_risk     one row per region per window with the academic risk score

Timestamps are stored as UTC text such as "2026-01-01T02:05:00.000+00:00", so
they sort and compare correctly as plain strings.
    event_time   simulated time of the event (from the generator)
    created_at   real time the generator wrote the event
    processed_at real time the pipeline stored the processed event
    window_end   simulated time at the end of the recent window used for DBSCAN
    calculated_at real time a cluster summary or risk score was calculated
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

import config

VITAL_COLUMNS = [
    "event_id", "patient_id", "region", "event_time", "created_at", "processed_at",
    "heart_rate", "systolic_bp", "diastolic_bp", "oxygen_saturation", "body_temperature",
    "respiratory_rate", "latitude", "longitude", "anomaly_score", "predicted_anomaly", "latency_seconds",
]
GEO_COLUMNS = [
    "event_id", "region", "event_time", "created_at", "processed_at", "latitude", "longitude",
    "symptom_type", "severity", "case_count", "latency_seconds",
]
CLUSTER_COLUMNS = [
    "cluster_id", "window_end", "calculated_at", "region",
    "center_latitude", "center_longitude", "event_count", "total_cases",
]
RISK_COLUMNS = [
    "region", "window_end", "calculated_at", "anomaly_rate", "clustered_cases",
    "anomaly_part", "cluster_part", "risk_score", "risk_level",
]
TIME_COLUMNS = {"event_time", "created_at", "processed_at", "window_end", "calculated_at"}

CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS vitals_results (
    event_id          TEXT PRIMARY KEY,
    patient_id        TEXT NOT NULL,
    region            TEXT NOT NULL,
    event_time        TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    processed_at      TEXT NOT NULL,
    heart_rate        INTEGER,
    systolic_bp       INTEGER,
    diastolic_bp      INTEGER,
    oxygen_saturation REAL,
    body_temperature  REAL,
    respiratory_rate  INTEGER,
    latitude          REAL,
    longitude         REAL,
    anomaly_score     REAL,
    predicted_anomaly INTEGER,          -- 1 = flagged by Isolation Forest, 0 = normal
    latency_seconds   REAL
);
CREATE INDEX IF NOT EXISTS idx_vitals_time ON vitals_results (event_time);

CREATE TABLE IF NOT EXISTS geo_events (
    event_id        TEXT PRIMARY KEY,
    region          TEXT NOT NULL,
    event_time      TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    processed_at    TEXT NOT NULL,
    latitude        REAL NOT NULL,
    longitude       REAL NOT NULL,
    symptom_type    TEXT,
    severity        INTEGER,
    case_count      INTEGER,
    cluster_id      TEXT,               -- NULL when the event is noise (not in a cluster)
    in_cluster      INTEGER DEFAULT 0,  -- 1 = part of a DBSCAN cluster in its latest window
    latency_seconds REAL
);
CREATE INDEX IF NOT EXISTS idx_geo_time ON geo_events (event_time);

CREATE TABLE IF NOT EXISTS clusters (
    cluster_id       TEXT PRIMARY KEY,  -- window end + DBSCAN number, e.g. "20260101T0200-0"
    window_end       TEXT NOT NULL,
    calculated_at    TEXT NOT NULL,
    region           TEXT,
    center_latitude  REAL,
    center_longitude REAL,
    event_count      INTEGER,
    total_cases      INTEGER
);

CREATE TABLE IF NOT EXISTS region_risk (
    region          TEXT NOT NULL,
    window_end      TEXT NOT NULL,
    calculated_at   TEXT NOT NULL,
    anomaly_rate    REAL,
    clustered_cases INTEGER,
    anomaly_part    REAL,
    cluster_part    REAL,
    risk_score      REAL,
    risk_level      TEXT,
    PRIMARY KEY (region, window_end)
);
"""


def to_utc_text(value) -> str:
    """Convert a datetime, pandas Timestamp or ISO string to UTC text with milliseconds."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")      # Spark gives UTC times without a zone
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat(timespec="milliseconds")


def connect(db_path: Path = config.DATABASE_PATH) -> sqlite3.Connection:
    """Open a connection. The timeout waits up to 30 s if another writer is busy."""
    return sqlite3.connect(db_path, timeout=30)


def create_tables(db_path: Path = config.DATABASE_PATH) -> None:
    """Create the database file and tables, and switch on WAL mode.

    WAL (write-ahead logging) lets the dashboard read while Spark is writing.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    connection = connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(CREATE_TABLES_SQL)
        connection.commit()
    finally:
        connection.close()


def _rows(data: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Turn a DataFrame into plain Python tuples in the given column order."""
    table = data[columns].copy()
    for column in TIME_COLUMNS & set(columns):
        table[column] = table[column].map(to_utc_text)
    table = table.astype(object).where(table.notna(), None)
    return [tuple(row) for row in table.itertuples(index=False, name=None)]


def _write(db_path: Path, sql: str, rows: list[tuple]) -> int:
    """Run one short transaction and return the number of changed rows."""
    if not rows:
        return 0
    connection = connect(db_path)
    try:
        with connection:                    # commits at the end, or rolls back on error
            before = connection.total_changes
            connection.executemany(sql, rows)
            return connection.total_changes - before
    finally:
        connection.close()


def _placeholders(columns: list[str]) -> str:
    return f"({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})"


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def insert_vital_results(results: pd.DataFrame, db_path: Path = config.DATABASE_PATH) -> int:
    """Insert scored vital-sign events. Returns how many new rows were added.

    INSERT OR IGNORE keeps the first stored copy of an event, so events that
    Spark replays after a restart do not create duplicates (and keep their
    original processed_at and latency).
    """
    data = results.copy()
    data["predicted_anomaly"] = data["predicted_anomaly"].astype(int)
    sql = "INSERT OR IGNORE INTO vitals_results " + _placeholders(VITAL_COLUMNS)
    return _write(db_path, sql, _rows(data, VITAL_COLUMNS))


def insert_geo_events(events: pd.DataFrame, db_path: Path = config.DATABASE_PATH) -> int:
    """Insert validated geographic events (cluster columns are filled in later)."""
    sql = "INSERT OR IGNORE INTO geo_events " + _placeholders(GEO_COLUMNS)
    return _write(db_path, sql, _rows(events, GEO_COLUMNS))


def update_cluster_assignments(events: pd.DataFrame, db_path: Path = config.DATABASE_PATH) -> int:
    """Store each event's cluster from the latest DBSCAN run (cluster_id None = noise)."""
    rows = [
        (row.cluster_id, int(row.in_cluster), row.event_id)
        for row in events[["cluster_id", "in_cluster", "event_id"]].itertuples(index=False)
    ]
    sql = "UPDATE geo_events SET cluster_id = ?, in_cluster = ? WHERE event_id = ?"
    return _write(db_path, sql, rows)


def save_clusters(clusters: pd.DataFrame, db_path: Path = config.DATABASE_PATH) -> int:
    """Save cluster summaries. Re-running the same window replaces its rows."""
    sql = "INSERT OR REPLACE INTO clusters " + _placeholders(CLUSTER_COLUMNS)
    return _write(db_path, sql, _rows(clusters, CLUSTER_COLUMNS))


def save_region_risk(risk: pd.DataFrame, db_path: Path = config.DATABASE_PATH) -> int:
    """Save risk rows. Re-running the same window replaces its rows."""
    sql = "INSERT OR REPLACE INTO region_risk " + _placeholders(RISK_COLUMNS)
    return _write(db_path, sql, _rows(risk, RISK_COLUMNS))


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------
def _read(sql: str, params: tuple = (), db_path: Path = config.DATABASE_PATH) -> pd.DataFrame:
    connection = connect(db_path)
    try:
        return pd.read_sql_query(sql, connection, params=params)
    finally:
        connection.close()


def read_vitals(start: str | None = None, end: str | None = None,
                db_path: Path = config.DATABASE_PATH) -> pd.DataFrame:
    """Vital results with start < event_time <= end (UTC text); all rows if no range is given."""
    if start is None or end is None:
        return _read("SELECT * FROM vitals_results ORDER BY event_time", db_path=db_path)
    return _read("SELECT * FROM vitals_results WHERE event_time > ? AND event_time <= ? ORDER BY event_time",
                 (start, end), db_path)


def read_geo_events(start: str | None = None, end: str | None = None,
                    db_path: Path = config.DATABASE_PATH) -> pd.DataFrame:
    """Geographic events with start < event_time <= end (UTC text); all rows if no range is given."""
    if start is None or end is None:
        return _read("SELECT * FROM geo_events ORDER BY event_time", db_path=db_path)
    return _read("SELECT * FROM geo_events WHERE event_time > ? AND event_time <= ? ORDER BY event_time",
                 (start, end), db_path)


def read_clusters(db_path: Path = config.DATABASE_PATH) -> pd.DataFrame:
    return _read("SELECT * FROM clusters ORDER BY window_end, cluster_id", db_path=db_path)


def read_region_risk(db_path: Path = config.DATABASE_PATH) -> pd.DataFrame:
    return _read("SELECT * FROM region_risk ORDER BY window_end, region", db_path=db_path)


def latest_geo_event_time(db_path: Path = config.DATABASE_PATH) -> str | None:
    """The newest simulated event_time stored in geo_events (None if empty)."""
    latest = _read("SELECT MAX(event_time) AS latest FROM geo_events", db_path=db_path)["latest"].iloc[0]
    return None if pd.isna(latest) else latest


def count_rows(table: str, db_path: Path = config.DATABASE_PATH) -> int:
    if table not in {"vitals_results", "geo_events", "clusters", "region_risk"}:
        raise ValueError(f"Unknown table: {table}")
    return int(_read(f"SELECT COUNT(*) AS n FROM {table}", db_path=db_path)["n"].iloc[0])
