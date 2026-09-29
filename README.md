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

forecast.py (run by hand)    ──►  Prophet forecast of case counts per 15 simulated minutes → SQLite
SQLite (data/healthcare.db)  ──►  dashboard.py (Streamlit, read only, http://localhost:8501)
evaluate.py                  ──►  final precision, recall, F1 and latency report   [Phase 5]
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
| `forecast.py` | Prophet forecast of synthetic case counts, run by hand, saved to SQLite |
| `dashboard.py` | Streamlit dashboard that only reads the results |
| `.streamlit/config.toml` | Streamlit settings: listen on 127.0.0.1:8501 only, no usage statistics |
| `setup_windows.ps1` | Checks Java 17 and `.venv`, downloads and verifies the Spark Windows compatibility JAR |
| `tests/` | Tests for the generator, detection, database, forecast and dashboard |
| `evaluate.py` | Planned for Phase 5 |

SQLite tables: `vitals_results` (scored readings), `geo_events` (events with
their latest cluster), `clusters` (one summary per cluster and window),
`region_risk` (risk components, score and level per region and window) and
`forecasts` (the latest Prophet forecast). Every processed event stores
`created_at`, `processed_at` and `latency_seconds`.

### Prophet forecast (`forecast.py`)

`forecast.py` runs only when you start it; nothing runs it automatically. It:

1. reads the processed geographic events (`event_time`, `case_count`) from SQLite;
2. adds up the cases per **15 simulated minutes** (empty intervals count as 0;
   the newest interval is left out while it is still filling up);
3. needs at least **12** complete intervals (3 simulated hours) of history;
4. fits a simple **trend-only** Prophet model (the synthetic history covers
   hours, not days, so daily/weekly/yearly seasonality are switched off);
5. forecasts the next **8** intervals (2 simulated hours) with an 80% range;
6. replaces the previous forecast in the `forecasts` table in one transaction.

Case counts cannot be negative, so a predicted, lower or upper value below 0 is
stored as 0. These settings are in `config.py` (`FORECAST_*`). This is a
forecast of **synthetic** data for demonstration, **not** a clinical or
public-health prediction.

### Streamlit dashboard (`dashboard.py`)

The dashboard only **reads** results through short read-only SQLite
connections. It never writes to the database, trains models, starts Spark or
generates data, and it never shows the ground-truth labels. It has six sections:

1. **System summary** – processed patient readings, readings flagged by
   Isolation Forest, geographic events, *Active clusters — latest window*
   (clusters in the newest DBSCAN window only), and the highest regional risk
   score and level in the latest window.
2. **Recent patient readings** – time, patient, region, vital signs, anomaly
   score and whether the model flagged the reading.
3. **Geographic activity** – a Plotly map (OpenStreetMap background, no API
   key) of events and the regions' latest risk level, plus tables of regional
   risk components and the active clusters. Grey events were not clustered;
   blue events (*Clustered when processed*) belonged to a cluster in the most
   recent DBSCAN window that included them, so blue points can remain after
   that cluster is no longer active. Marker size shows the case count. Without
   internet, switch off *Map background* to get a plain latitude/longitude chart.
4. **Case history and forecast** – historical cases per 15 simulated minutes,
   the Prophet forecast and its 80% range, separated by a "Forecast starts" line.
5. **Model evaluation** – the held-out synthetic results saved by
   `train_models.py` (`data/results/phase2_evaluation.json`); precision, recall
   and F1 are shown as percentages with one decimal place.
6. **Processing performance** – median, 95th-percentile and maximum latency and
   the throughput, shown separately for patient readings and geographic events.

A collapsed **About the metrics** section at the bottom briefly explains the
anomaly score, the latest DBSCAN window, the risk score, the forecast range,
latency, throughput and the evaluation results. The dashboard itself only shows
a short "synthetic" subtitle and footer; the full synthetic-data and
non-clinical limitations are in this README.

Sidebar filters: time range (simulated UTC), region, severity, symptom type,
and flagged-only / all patient readings. The filters apply to the patient
table and the geographic activity section; the summary, forecast, evaluation
and performance sections always use all stored results. Click **Refresh data**
to re-read the database; the page does not refresh by itself.

A distinct count of historical outbreaks is deliberately not shown: DBSCAN
re-runs on an overlapping 60-minute window each time events arrive, so one
synthetic outbreak is recorded under several window-specific cluster IDs.

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

## Running the complete project

Run these commands in this order from Windows PowerShell. They were verified
end to end on 29 Sep 2026 (Phase 4).

```powershell
# 0. Enter the project and activate the environment
cd "C:\healthcare-anomaly-outbreak-detection"
.\.venv\Scripts\Activate.ps1

# 1. Generate the training and held-out synthetic datasets
python generate_data.py --mode train

# 2. Train Isolation Forest and evaluate both detectors (writes data/results/phase2_evaluation.json)
python train_models.py

# 3. Write simulated stream files: 120 ticks, one per second (about 2 minutes, 10 simulated hours)
python generate_data.py --mode stream --fresh

# 4. Process all stream files with Spark once, then stop (about 30 seconds)
python spark_pipeline.py --mode once --reset

# 5. Create the Prophet forecast (a few seconds)
python forecast.py

# 6. Start the dashboard, then open http://localhost:8501 in a browser
python -m streamlit run dashboard.py
```

Stop the dashboard with **Ctrl+C** in its PowerShell window. It only listens on
this computer (`127.0.0.1:8501`, set in `.streamlit/config.toml`).

Useful variations:

```powershell
# Step 3 without waiting: write the 120 ticks immediately
python generate_data.py --mode stream --ticks 120 --seconds-per-tick 0 --fresh

# Live demo instead of steps 3-4: pipeline in terminal 1, generator in terminal 2
python spark_pipeline.py --mode continuous --reset     # stop with Ctrl+C, or add --stop-after 180
python generate_data.py --mode stream --fresh
# afterwards: python forecast.py, then refresh the dashboard

# Tests (they do not start Spark)
python -m pytest
```

- Each stream tick advances the simulated clock by 5 minutes. Synthetic
  outbreaks start on a fixed schedule (simulated tick 20, then every 60 ticks),
  so every run contains some.
- `--fresh` is needed whenever stream files from an earlier run exist.
  `--reset` deletes the SQLite database and Spark checkpoints; use it after
  `--fresh`, because new files reuse old file names and Spark would otherwise
  treat them as already processed.
- `--seconds-per-tick 0` must be combined with `--ticks`.
- Restarting the pipeline resumes from its checkpoint; events that Spark
  replays are ignored by SQLite (`INSERT OR IGNORE` on `event_id`).
- `python <script>.py --help` lists every option of a script.

## Troubleshooting

| Message or symptom | What to do |
|---|---|
| Dashboard: *No results database was found* | Run steps 1–4, then click **Refresh data**. |
| `Model not found at ...models\isolation_forest.joblib` | Run `python train_models.py` (after `python generate_data.py --mode train`). |
| `Missing ...vitals_train.csv` from `train_models.py` | Run `python generate_data.py --mode train` first. |
| Dashboard: *No forecast yet* | Run `python forecast.py`, then click **Refresh data**. |
| `forecast.py`: *Only N complete 15-minute intervals ... at least 12 are needed* | Generate a longer stream (step 3 gives 40 intervals) and run step 4 again. |
| Dashboard: *New events have been processed since this forecast was made* | Run `python forecast.py` again. |
| Dashboard: *No evaluation results found* | Run `python train_models.py`. |
| `Event files from an earlier run already exist` | Add `--fresh` to the generator command, and `--reset` to the next pipeline run. |
| `Missing ...jars\hadoop-bare-naked-local-fs-0.1.0.jar` | Run `.\setup_windows.ps1`. |
| Map background stays blank | The OpenStreetMap tiles need internet: switch off *Map background* in the sidebar. |
| Very large latency after `--mode once` | Expected: the files were written before Spark started. Use the live demo for real-time latency. |

## Current development status

| Phase | Status |
|---|---|
| 1. Environment, project files, data generator, Windows Spark setup | Complete |
| 2. Isolation Forest, DBSCAN, risk score, simple evaluation | Complete |
| 3. Spark Structured Streaming pipeline, SQLite, latency measurement | Complete |
| 4. Prophet forecast, Streamlit dashboard, geographic map | Complete |
| 5. Final evaluation (`evaluate.py`), full demonstration, screenshots | Not started |

## Results so far (synthetic data only)

> All numbers below come from **synthetic data** produced by this project's own
> generator. They depend on how the generator is configured and **do not
> demonstrate real clinical or public-health performance**.

### Detection on held-out synthetic data (`train_models.py`; identical results on 28 and 29 Sep 2026)

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

### Forecast

The dashboard shows whatever forecast `forecast.py` last produced from the
data in the database. It is a trend-only model of synthetic case counts; no
forecast accuracy has been measured yet, so no accuracy figure is claimed.

## Known limitations

- The generator occasionally creates a "valid" reading whose systolic and
  diastolic pressure are equal (about 0.5–1% of readings). The pipeline's
  validation rule (systolic must be higher) correctly rejects them. This will
  be addressed before the final Phase 5 evaluation.
- The regional risk score is recalculated when geographic events arrive, and
  in continuous mode it uses the vital results already stored at that moment.
- Micro-batches are converted to pandas on the driver, which suits this small
  local project but would not scale to large data volumes.
- The Prophet model is trend-only because the synthetic history covers hours,
  not days; it forecasts the total case count, not per-region counts.
- Latency shown by the dashboard is only a real-time measurement after a live
  (continuous-mode) run; after `--mode once` it includes the time the files
  waited before Spark started.
- The map background uses OpenStreetMap tiles and needs internet; a plain
  latitude/longitude chart is available offline.
- Prophet 1.4 triggers a NumPy deprecation warning (visible when running the
  tests); `requirements.txt` keeps NumPy below 2.6 so it cannot become an error.
- The dashboard does not refresh by itself; click **Refresh data**.
