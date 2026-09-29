"""Real-time pipeline: JSON event files -> Spark Structured Streaming -> SQLite.

All data is SYNTHETIC and intended only for academic demonstration.

Two streaming queries run in one local Spark process:
  vitals query  validate -> Isolation Forest score -> save to vitals_results
  geo query     validate -> save to geo_events -> DBSCAN on the recent window
                -> save clusters -> calculate regional risk -> save region_risk

Ground-truth label files are never read here.

Usage:
  python spark_pipeline.py --mode once          process the files that exist now, then stop
  python spark_pipeline.py --mode continuous    keep watching for new files (Ctrl+C to stop)
  add --reset to delete the database and checkpoints first (needed after
  "generate_data.py --mode stream --fresh", because the new files reuse old names)

Windows note: PySpark is imported inside functions, only AFTER
prepare_spark_environment() has put the compatibility JAR on the classpath.
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path

import pandas as pd

import config
import database
import detection

VITAL_REQUIRED = ["event_id", "patient_id", "region_id", "event_time", "created_at",
                  "latitude", "longitude", *config.VITAL_FEATURES]
GEO_REQUIRED = ["event_id", "region_id", "event_time", "created_at", "latitude", "longitude",
                "symptom_category", "severity", "case_count"]

isolation_forest = None   # loaded once in main() before the streams start


# ---------------------------------------------------------------------------
# Spark setup
# ---------------------------------------------------------------------------
def prepare_spark_environment() -> None:
    """Must run BEFORE PySpark starts Java."""
    if not config.BARE_LOCAL_FS_JAR.is_file():
        raise SystemExit(f"Missing {config.BARE_LOCAL_FS_JAR}\nRun .\\setup_windows.ps1 first.")

    java_home = os.environ.get("JAVA_HOME")
    java_found = (java_home and shutil.which("java", path=str(Path(java_home) / "bin"))) or shutil.which("java")
    if not java_found:
        raise SystemExit("Java 17 was not found. Install it (see README.md), "
                         "then open a NEW PowerShell window and try again.")

    # PySpark passes PYSPARK_SUBMIT_ARGS to spark-submit when it starts Java, so the
    # JAR is on the classpath before Hadoop loads BareLocalFileSystem. PySpark splits
    # this text like a Unix shell, so the path must use forward slashes.
    os.environ["PYSPARK_SUBMIT_ARGS"] = f'--driver-class-path "{config.BARE_LOCAL_FS_JAR.as_posix()}" pyspark-shell'
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    os.environ["SPARK_LOCAL_IP"] = config.SPARK_BIND_ADDRESS     # listen on this computer only


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((config.SPARK_BIND_ADDRESS, port)) != 0


def create_spark_session():
    from pyspark.sql import SparkSession

    ui_enabled = port_is_free(config.SPARK_UI_PORT)
    if ui_enabled:
        print(f"Spark UI: http://{config.SPARK_BIND_ADDRESS}:{config.SPARK_UI_PORT}")
    else:
        print(f"Port {config.SPARK_UI_PORT} is in use, so the Spark UI is switched off.")

    spark = (
        SparkSession.builder.appName("healthcare-anomaly-outbreak-detection")
        .master(config.SPARK_MASTER)
        .config("spark.sql.shuffle.partitions", config.SPARK_SHUFFLE_PARTITIONS)
        .config("spark.driver.bindAddress", config.SPARK_BIND_ADDRESS)
        .config("spark.driver.host", config.SPARK_BIND_ADDRESS)
        .config("spark.ui.enabled", str(ui_enabled).lower())
        .config("spark.ui.port", config.SPARK_UI_PORT)
        .config("spark.sql.session.timeZone", "UTC")
        # Windows compatibility: pure-Java local files and checkpoints (no winutils.exe)
        .config("spark.hadoop.fs.file.impl", config.SPARK_FILE_SYSTEM_CLASS)
        .config("spark.sql.streaming.checkpointFileManagerClass", config.SPARK_CHECKPOINT_MANAGER_CLASS)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def vitals_schema():
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType, TimestampType
    return StructType([
        StructField("event_id", StringType()),
        StructField("patient_id", StringType()),
        StructField("region_id", StringType()),
        StructField("event_time", TimestampType()),
        StructField("heart_rate", IntegerType()),
        StructField("systolic_bp", IntegerType()),
        StructField("diastolic_bp", IntegerType()),
        StructField("oxygen_saturation", DoubleType()),
        StructField("body_temperature", DoubleType()),
        StructField("respiratory_rate", IntegerType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("created_at", TimestampType()),
    ])


def geo_schema():
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType, TimestampType
    return StructType([
        StructField("event_id", StringType()),
        StructField("event_time", TimestampType()),
        StructField("region_id", StringType()),
        StructField("region_name", StringType()),
        StructField("latitude", DoubleType()),
        StructField("longitude", DoubleType()),
        StructField("symptom_category", StringType()),
        StructField("severity", IntegerType()),
        StructField("case_count", IntegerType()),
        StructField("created_at", TimestampType()),
    ])


# ---------------------------------------------------------------------------
# Validation (in Spark) and conversion to pandas
# ---------------------------------------------------------------------------
def validate_vitals(batch_df):
    """Drop rows with missing fields or physically impossible readings."""
    from pyspark.sql import functions as F
    valid = batch_df.dropna(subset=VITAL_REQUIRED)
    for column, (low, high) in config.VITAL_VALID_RANGES.items():
        valid = valid.filter(F.col(column).between(low, high))
    return valid.filter(F.col("systolic_bp") > F.col("diastolic_bp"))


def validate_geo(batch_df):
    """Drop rows with missing fields, impossible coordinates or invalid case counts."""
    from pyspark.sql import functions as F
    low, high = config.SEVERITY_RANGE
    return (
        batch_df.dropna(subset=GEO_REQUIRED)
        .filter(F.col("latitude").between(-90, 90) & F.col("longitude").between(-180, 180))
        .filter((F.col("case_count") >= 1) & F.col("severity").between(low, high))
    )


def to_pandas_checked(valid_df, stream_name: str) -> pd.DataFrame:
    """Convert a small Spark DataFrame to pandas, refusing unexpectedly large batches."""
    rows = valid_df.count()
    if rows > config.MAX_ROWS_PER_BATCH:
        raise RuntimeError(f"{stream_name} micro-batch has {rows} rows, more than MAX_ROWS_PER_BATCH "
                           f"({config.MAX_ROWS_PER_BATCH}). Lower SPARK_MAX_FILES_PER_TRIGGER in config.py.")
    return valid_df.toPandas()


def add_processing_time(events: pd.DataFrame) -> None:
    """Record processed_at (now, UTC) and latency_seconds = processed_at - created_at."""
    processed_at = pd.Timestamp.now(tz="UTC")
    events["processed_at"] = processed_at
    created_at = pd.to_datetime(events["created_at"], utc=True)
    events["latency_seconds"] = (processed_at - created_at).dt.total_seconds().round(3)


# ---------------------------------------------------------------------------
# Micro-batch handlers
# ---------------------------------------------------------------------------
def process_vitals_batch(batch_df, batch_id: int) -> None:
    total = batch_df.count()
    if total == 0:
        return
    events = to_pandas_checked(validate_vitals(batch_df), "vitals")
    rejected = total - len(events)
    stored = flagged = 0
    if not events.empty:
        events = events.rename(columns={"region_id": "region"})
        scored = detection.score_vitals(isolation_forest, events)
        add_processing_time(scored)                       # immediately before storing
        stored = database.insert_vital_results(scored)
        flagged = int(scored["predicted_anomaly"].sum())
    print(f"[vitals batch {batch_id}] read {total}, rejected {rejected}, "
          f"new rows {stored}, flagged anomalies {flagged}", flush=True)


def process_geo_batch(batch_df, batch_id: int) -> None:
    total = batch_df.count()
    if total == 0:
        return
    events = to_pandas_checked(validate_geo(batch_df), "geo")
    rejected = total - len(events)
    stored = 0
    if not events.empty:
        events = events.rename(columns={"region_id": "region", "symptom_category": "symptom_type"})
        add_processing_time(events)                       # immediately before storing
        stored = database.insert_geo_events(events)
    clusters, window_end = update_clusters_and_risk()
    print(f"[geo batch {batch_id}] read {total}, rejected {rejected}, new rows {stored}, "
          f"clusters {clusters} in window ending {window_end}", flush=True)


def update_clusters_and_risk() -> tuple[int, str | None]:
    """Run DBSCAN on the recent window (across micro-batches) and update the risk scores."""
    latest = database.latest_geo_event_time()
    if latest is None:
        return 0, None

    # 1. Read the recent simulated-time window from SQLite.
    window_end = pd.Timestamp(latest)
    window_start = window_end - timedelta(minutes=config.DBSCAN_WINDOW_MINUTES)
    start_text, end_text = database.to_utc_text(window_start), database.to_utc_text(window_end)
    calculated_at = pd.Timestamp.now(tz="UTC")
    recent = database.read_geo_events(start_text, end_text)

    # 2. Run DBSCAN and give each cluster an ID that is unique across windows.
    clustered = detection.find_clusters(recent)
    prefix = window_end.strftime("%Y%m%dT%H%M")
    clustered["cluster_id"] = [f"{prefix}-{n}" if n != -1 else None for n in clustered["cluster_id"]]
    database.update_cluster_assignments(clustered)

    # 3. Save one summary row per cluster.
    in_cluster = clustered[clustered["in_cluster"]]
    summaries = (
        in_cluster.groupby("cluster_id")
        .agg(region=("region", lambda regions: regions.mode().iloc[0]),
             center_latitude=("latitude", "mean"),
             center_longitude=("longitude", "mean"),
             event_count=("event_id", "count"),
             total_cases=("case_count", "sum"))
        .reset_index()
    )
    summaries["window_end"] = end_text
    summaries["calculated_at"] = calculated_at
    database.save_clusters(summaries)

    # 4. Regional risk (academic heuristic) from the same recent window.
    recent_vitals = database.read_vitals(start_text, end_text)
    risk_rows = []
    for region in config.REGIONS:
        region_id = region["region_id"]
        region_vitals = recent_vitals[recent_vitals["region"] == region_id]
        anomaly_rate = float(region_vitals["predicted_anomaly"].mean()) if len(region_vitals) else 0.0
        clustered_cases = int(in_cluster.loc[in_cluster["region"] == region_id, "case_count"].sum())
        risk_rows.append({
            "region": region_id, "window_end": end_text, "calculated_at": calculated_at,
            "anomaly_rate": round(anomaly_rate, 4), "clustered_cases": clustered_cases,
            **detection.calculate_risk(anomaly_rate, clustered_cases),
        })
    database.save_region_risk(pd.DataFrame(risk_rows))
    return len(summaries), end_text


# ---------------------------------------------------------------------------
# Running the queries
# ---------------------------------------------------------------------------
def start_query(spark, name: str, folder: Path, schema, handler, checkpoint: Path, mode: str):
    stream = (
        spark.readStream.schema(schema)
        .option("pathGlobFilter", "*.json")
        .option("maxFilesPerTrigger", config.SPARK_MAX_FILES_PER_TRIGGER)
        .json(folder.as_uri())
    )
    writer = stream.writeStream.queryName(name).foreachBatch(handler).option("checkpointLocation", checkpoint.as_uri())
    if mode == "once":
        writer = writer.trigger(availableNow=True)      # process what exists now, then stop
    else:
        writer = writer.trigger(processingTime=f"{config.SPARK_TRIGGER_SECONDS} seconds")
    return writer.start()


def reset_outputs() -> None:
    """Delete the SQLite database and the streaming checkpoints (inside data/ only)."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{config.DATABASE_PATH}{suffix}").unlink(missing_ok=True)
    for folder in (config.VITALS_CHECKPOINT_DIR, config.GEO_CHECKPOINT_DIR):
        shutil.rmtree(folder, ignore_errors=True)
    print("Reset: deleted the database and checkpoints.")


def stop_all(queries, spark) -> None:
    for query in queries:
        try:
            if query.isActive:
                query.stop()
        except Exception as error:            # Java may already be shutting down after Ctrl+C
            print(f"Could not stop query cleanly: {error}")
    try:
        spark.stop()
    except Exception as error:
        print(f"Could not stop Spark cleanly: {error}")


def main() -> None:
    global isolation_forest
    parser = argparse.ArgumentParser(description="Spark Structured Streaming pipeline (synthetic data only).")
    parser.add_argument("--mode", choices=["once", "continuous"], required=True)
    parser.add_argument("--reset", action="store_true", help="delete the database and checkpoints first")
    parser.add_argument("--stop-after", type=float, default=None,
                        help="continuous mode: stop automatically after this many seconds")
    args = parser.parse_args()

    print("Processing SYNTHETIC data for academic demonstration only.")
    if args.reset:
        reset_outputs()
    for folder in (config.VITALS_INCOMING_DIR, config.GEO_INCOMING_DIR, config.CHECKPOINT_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    database.create_tables()
    try:
        isolation_forest = detection.load_model()
    except FileNotFoundError as error:
        raise SystemExit(str(error))

    prepare_spark_environment()
    spark = create_spark_session()
    queries = []
    try:
        if args.mode == "once":
            # Vitals first, so the risk scores see all vital results (repeatable output).
            for name, folder, schema, handler, checkpoint in (
                ("vitals", config.VITALS_INCOMING_DIR, vitals_schema(), process_vitals_batch, config.VITALS_CHECKPOINT_DIR),
                ("geo", config.GEO_INCOMING_DIR, geo_schema(), process_geo_batch, config.GEO_CHECKPOINT_DIR),
            ):
                query = start_query(spark, name, folder, schema, handler, checkpoint, "once")
                queries.append(query)
                query.awaitTermination()
        else:
            queries.append(start_query(spark, "vitals", config.VITALS_INCOMING_DIR, vitals_schema(),
                                       process_vitals_batch, config.VITALS_CHECKPOINT_DIR, "continuous"))
            queries.append(start_query(spark, "geo", config.GEO_INCOMING_DIR, geo_schema(),
                                       process_geo_batch, config.GEO_CHECKPOINT_DIR, "continuous"))
            print("Streaming queries started. Waiting for new files (Ctrl+C to stop).", flush=True)
            started = time.monotonic()
            while all(query.isActive for query in queries):
                if args.stop_after is not None and time.monotonic() - started >= args.stop_after:
                    print(f"Stopping after {args.stop_after:g} seconds.")
                    break
                time.sleep(1)
            for query in queries:
                if query.exception() is not None:
                    raise RuntimeError(f"Query '{query.name}' failed: {query.exception()}")
    except KeyboardInterrupt:
        print("\nStopping (Ctrl+C).")
    finally:
        stop_all(queries, spark)

    print("SQLite rows:", {table: database.count_rows(table)
                           for table in ("vitals_results", "geo_events", "clusters", "region_risk")})


if __name__ == "__main__":
    main()
