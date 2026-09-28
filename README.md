# Autonomous Real-Time Healthcare Anomaly and Outbreak Detection System

An entry-level Bachelor of Computer Science academic project. It processes a
simulated stream of healthcare events, flags abnormal patient vital signs,
finds geographic clusters that may indicate an outbreak, forecasts case counts
and shows everything on a Streamlit dashboard.

> **SYNTHETIC DATA NOTICE**
> Every patient, reading, location, outbreak and result in this project is
> **simulated** and intended **only for academic demonstration**. It is not a
> clinical system, has not been clinically validated, and must not be used for
> any medical or public-health decision. Region names are fictional; map
> coordinates are arbitrary positions used only to draw the map.

## Planned data flow

```
generate_data.py  ──►  data/incoming/vitals/*.json, data/incoming/geo/*.json
      │                  (event files with event_time and created_at)
      └──────────────►  data/labels/   (ground truth, never read by the models)

spark_pipeline.py  (PySpark Structured Streaming)
   read JSON files → validate → Isolation Forest on vitals
   → save validated geo events to SQLite
   → read geo events from a recent simulated-time window (spanning many micro-batches)
   → DBSCAN on that window → save clusters → risk score → SQLite

SQLite (data/healthcare.db)  ──►  dashboard.py (Streamlit, read only)
forecast.py (run manually)   ──►  Prophet forecast of hourly case counts, saved to SQLite
evaluate.py                  ──►  precision, recall, F1 and latency against the labels
```

The risk score will be a **simple academic heuristic** that combines the share
of abnormal vital readings in a region with the size of nearby DBSCAN clusters.
It is not a probability and it is not a clinically validated score.

## Project files

| File | Purpose |
|---|---|
| `config.py` | All settings (paths, seeds, simulation sizes, Spark settings) |
| `generate_data.py` | Generates synthetic events and separate ground-truth labels |
| `setup_windows.ps1` | Checks Java 17 and `.venv`, downloads and verifies the Spark Windows compatibility JAR |
| `tests/test_generator.py` | Tests for the generator |
| `spark_pipeline.py`, `train_models.py`, `database.py`, `forecast.py`, `evaluate.py`, `dashboard.py` | Planned for later phases |

## Installation (Windows PowerShell)

Requirements: Python 3.12 and Eclipse Temurin JDK 17. Run these commands
from the project folder.

```powershell
# 1. Install Java 17 (skip if `java -version` already shows 17), then open a NEW PowerShell window
winget install --id EclipseAdoptium.Temurin.17.JDK --exact
java -version

# 2. Create and activate the virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install the packages (inside .venv only)
python -m pip install -r requirements.txt

# 4. Windows setup: checks Java and .venv, downloads the compatibility JAR
.\setup_windows.ps1
```

If PowerShell blocks `Activate.ps1` or `setup_windows.ps1`, either run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` (affects only the
current window) or skip activation and call `.\.venv\Scripts\python.exe`
directly, as the commands below do.

## Spark on Windows

Spark normally needs the native Hadoop files `winutils.exe` and `hadoop.dll`
to read and write local files on Windows. Apache does not publish official
Windows builds of them, so this project **does not use them**. Instead:

1. `setup_windows.ps1` downloads one small (about 9 KB) pure-Java JAR from
   Maven Central, `com.globalmentor:hadoop-bare-naked-local-fs:0.1.0`, into
   `jars/`, and checks its SHA-256 and the SHA-1 published by Maven Central.
   The JAR is not committed to Git.
2. Spark must receive that project-relative JAR through `--driver-class-path`
   **before** PySpark starts (set in the `PYSPARK_SUBMIT_ARGS` environment
   variable), so Java can load it at startup.
3. Spark must use these two settings (both are in `config.py`):
   - `spark.hadoop.fs.file.impl` =
     `com.globalmentor.apache.hadoop.fs.BareLocalFileSystem`
     (pure-Java local file access)
   - `spark.sql.streaming.checkpointFileManagerClass` =
     `org.apache.spark.sql.execution.streaming.checkpointing.FileSystemBasedCheckpointFileManager`
     (a built-in Spark class that makes checkpoints use the same file access)
4. These warnings may still appear and are expected with this setup:
   - `WARN Shell: Did not find winutils.exe ... HADOOP_HOME and hadoop.home.dir are unset`
   - `WARN NativeCodeLoader: Unable to load native-hadoop library for your platform`

This setup was verified with PySpark 4.2.0 (pinned in `requirements.txt`) and
Java 17: batch JSON reading, Structured Streaming, checkpoint creation and
restarting from a checkpoint without reprocessing old files. Spark runs as
`local[2]` with two shuffle partitions and listens only on `127.0.0.1`.

## Usage (Phase 1)

```powershell
# Training and test datasets for the models
.\.venv\Scripts\python.exe generate_data.py --mode train

# Simulated live stream: one vitals file and one geo file per second
.\.venv\Scripts\python.exe generate_data.py --mode stream --minutes 2 --fresh

# Tests
.\.venv\Scripts\python.exe -m pytest
```

Each stream tick advances the simulated clock by 5 minutes, so a 2-minute run
covers 10 simulated hours. Synthetic outbreaks start on a fixed schedule
(simulated tick 20, then every 60 ticks), so every run contains some.

## Current development status

| Phase | Status |
|---|---|
| 1. Environment, project files, data generator, Windows Spark setup | Complete |
| 2. Isolation Forest, DBSCAN, simple evaluation | Not started |
| 3. Spark Structured Streaming pipeline, SQLite, latency | Not started |
| 4. Prophet forecast, Streamlit dashboard, GIS map | Not started |
| 5. Tests, full demonstration, final evaluation | Not started |

## Results

**Pending.** No precision, recall, F1 or latency results have been measured
yet. When they are, they will be produced by `evaluate.py`, reported as
results on synthetic data only, and will not describe real clinical performance.
