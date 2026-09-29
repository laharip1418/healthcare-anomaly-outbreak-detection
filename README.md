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

## Data flow

```
generate_data.py  ──►  data/incoming/vitals/*.json, data/incoming/geo/*.json
      │                  (event files with event_time and created_at)
      └──────────────►  data/labels/   (ground truth, never read by the models)

spark_pipeline.py  (PySpark Structured Streaming, two queries in one local[2] process)
   vitals: read JSON → validate → Isolation Forest score → SQLite
   geo:    read JSON → validate → save to SQLite
           → read geo events from the last 60 simulated minutes (spanning many micro-batches)
           → DBSCAN on that window → save clusters → regional risk score → SQLite

SQLite (data/healthcare.db)  ──►  dashboard.py (Streamlit, read only)        [Phase 4]
forecast.py (run manually)   ──►  Prophet forecast of hourly case counts     [Phase 4]
evaluate.py                  ──►  precision, recall, F1 and latency report   [Phase 5]
```

### Risk score (simple academic heuristic)

For each region, over the last 60 simulated minutes:

```
anomaly_part = min(share of abnormal vital readings / 0.10, 1)
cluster_part = min(cases inside DBSCAN clusters / 20, 1)
risk_score   = 100 × (0.5 × anomaly_part + 0.5 × cluster_part)
risk_level   = Low (< 30), Medium (30–59), High (≥ 60)
```

Both parts are stored next to the score so the result can be explained. This is
a **simple academic heuristic**: it is **not a probability**, not a medical
prediction and **not clinically validated**.

## Project files

| File | Purpose |
|---|---|
| `config.py` | All settings (paths, seeds, simulation sizes, model, Spark and validation settings) |
| `generate_data.py` | Generates synthetic events and separate ground-truth labels |
| `detection.py` | Isolation Forest, DBSCAN (haversine distance) and the risk-score heuristic |
| `train_models.py` | Trains Isolation Forest (no labels) and evaluates both detectors on held-out synthetic data |
| `database.py` | SQLite tables (WAL mode) and small insert/read functions |
| `spark_pipeline.py` | Structured Streaming pipeline: validation, detection, risk score, SQLite output |
| `setup_windows.ps1` | Checks Java 17 and `.venv`, downloads and verifies the Spark Windows compatibility JAR |
| `tests/` | Tests for the generator, detection functions and database |
| `forecast.py`, `dashboard.py`, `evaluate.py` | Planned for Phases 4 and 5 |

SQLite tables: `vitals_results` (scored readings), `geo_events` (events with
their latest cluster), `clusters` (one summary per cluster and window) and
`region_risk` (risk components, score and level per region and window). Every
processed event stores `created_at`, `processed_at` and `latency_seconds`.

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

## Usage

```powershell
# 1. Training/test datasets, then train and evaluate the models
.\.venv\Scripts\python.exe generate_data.py --mode train
.\.venv\Scripts\python.exe train_models.py

# 2a. Finite demo: write stream files, then process them once and stop
.\.venv\Scripts\python.exe generate_data.py --mode stream --ticks 40 --seconds-per-tick 0 --fresh
.\.venv\Scripts\python.exe spark_pipeline.py --mode once --reset

# 2b. Live demo: start the pipeline first (terminal 1), then the generator (terminal 2)
.\.venv\Scripts\python.exe spark_pipeline.py --mode continuous --reset
.\.venv\Scripts\python.exe generate_data.py --mode stream --minutes 2 --fresh

# Tests (they do not start Spark)
.\.venv\Scripts\python.exe -m pytest
```

- Each stream tick advances the simulated clock by 5 minutes, so a 2-minute
  run covers 10 simulated hours. Synthetic outbreaks start on a fixed schedule
  (simulated tick 20, then every 60 ticks), so every run contains some.
- `--reset` deletes the SQLite database and Spark checkpoints. Use it whenever
  the generator was run with `--fresh`, because new files reuse old file names
  and Spark would otherwise treat them as already processed.
- Continuous mode stops with Ctrl+C, or automatically with `--stop-after SECONDS`.
- Restarting the pipeline resumes from its checkpoint; events that Spark
  replays are ignored by SQLite (`INSERT OR IGNORE` on `event_id`).

## Current development status

| Phase | Status |
|---|---|
| 1. Environment, project files, data generator, Windows Spark setup | Complete |
| 2. Isolation Forest, DBSCAN, risk score, simple evaluation | Complete |
| 3. Spark Structured Streaming pipeline, SQLite, latency measurement | Complete |
| 4. Prophet forecast, Streamlit dashboard, GIS map | Not started |
| 5. Final evaluation (`evaluate.py`), full demonstration | Not started |

## Results so far (synthetic data only)

> All numbers below come from **synthetic data** produced by this project's own
> generator. They depend on how the generator is configured and **do not
> demonstrate real clinical or public-health performance**.

### Detection on held-out synthetic data (`train_models.py`, 28 Sep 2026)

Isolation Forest was trained without labels on 5,000 readings (seed 42) and
tested on 2,000 separate readings (seed 43, 2.95% labelled anomalies).
DBSCAN was replayed over 1,214 held-out geographic events (seed 43, 5 synthetic
outbreaks) with a 60-minute sliding window. Settings were chosen before the
test run and not tuned on the test labels.

| Detector | Precision | Recall | F1 | TP / FP / FN / TN |
|---|---|---|---|---|
| Isolation Forest (vital signs) | 0.704 | 0.848 | 0.769 | 50 / 21 / 9 / 1,920 |
| DBSCAN (outbreak events) | 0.989 | 0.810 | 0.891 | 278 / 3 / 65 / 868 |

- Isolation Forest caught every synthetic fever, hypotension, hypertensive-crisis
  and tachycardia-with-low-oxygen reading, but **none** of the bradycardia
  readings (a low heart rate on its own).
- DBSCAN found all 5 synthetic outbreaks; most missed outbreak events are the
  first few of each outbreak, before enough nearby events exist to form a cluster.

### Processing latency in one pipeline verification run (29 Sep 2026)

One short continuous-mode run: 60 ticks written 1 second apart (600 vital
readings, 255 geographic events) on a Dell OptiPlex 5050 (4 logical CPUs,
16 GB RAM), Spark `local[2]`, 2-second trigger, at most 10 files per
micro-batch. Latency is each event's stored `processed_at − created_at`, so it
includes the wait for the next trigger.

| Events | Median | 95th percentile | Maximum | Share within 3 s |
|---|---|---|---|---|
| Vital signs (591 stored) | 1.64 s | 5.62 s | 6.86 s | 89.8% |
| Geographic events (255 stored) | 1.47 s | 3.02 s | 6.67 s | 93.3% |

This is a single short run, not a benchmark. A reproducible latency and
detection report will be produced by `evaluate.py` in Phase 5.

## Known limitations

- The generator occasionally creates a "valid" reading whose systolic and
  diastolic pressure are equal (about 0.5–1% of readings). The pipeline's
  validation rule (systolic must be higher) correctly rejects them.
- The regional risk score is recalculated when geographic events arrive, and
  in continuous mode it uses the vital results already stored at that moment.
- Micro-batches are converted to pandas on the driver, which suits this small
  local project but would not scale to large data volumes.
