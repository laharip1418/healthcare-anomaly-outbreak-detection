# Autonomous Real-Time Healthcare Anomaly and Outbreak Detection System

A student project that streams synthetic healthcare events through PySpark, detects abnormal vital signs and geographic outbreak clusters, forecasts case counts and shows the results on a Streamlit dashboard.

**Live Demo:** [View Live Dashboard](https://healthcare-anomaly-outbreak-detection-7awsftoteivzdwmohsvlgx.streamlit.app/)

The hosted dashboard is read-only and shows pre-generated synthetic demonstration data. Spark and the data-generation pipeline run locally, not on Streamlit Community Cloud.

## Overview

Hospitals and public-health teams watch two kinds of signals: individual patients whose vital signs suddenly look abnormal, and many similar cases appearing close together, which can be an early sign of an outbreak. This project demonstrates how both can be monitored in (near) real time on one local computer.

All data is **synthetic**. A generator simulates 100 patients in six fictional districts and writes their vital signs (heart rate, blood pressure, oxygen saturation, temperature, respiratory rate) and location-tagged symptom reports as JSON files, together with occasional simulated outbreaks and deliberately invalid records. The ground-truth labels are stored separately and are only used for evaluation.

The completed system produces, in an SQLite database:

- every valid reading with its anomaly score and flag,
- geographic clusters of symptom reports,
- a risk score for each district,
- a short-term forecast of case counts,
- processing timestamps for measuring latency,

and displays them on an interactive dashboard.

## Key Features

- **Synthetic event generator** with fixed, selectable seeds, so every run is reproducible
- **PySpark Structured Streaming** pipeline that validates each micro-batch and rejects invalid records
- **Vital-sign anomaly detection** with an unsupervised Isolation Forest (trained without labels)
- **Geographic clustering** with DBSCAN and haversine distance over a 60-minute sliding window
- **Regional risk score** that combines the share of abnormal readings and the number of clustered cases
- **Prophet forecast** of case counts for the next two simulated hours
- **Streamlit dashboard** with a map, filters, forecast chart, evaluation table and latency figures
- **One-command demo** (`run_demo.py`) that runs the whole pipeline with a chosen seed
- **Automated tests** with pytest (65 tests, none of which need Spark)

## Demo

![Dashboard overview](screenshots/dashboard-overview.png)

[View the live dashboard](https://healthcare-anomaly-outbreak-detection-7awsftoteivzdwmohsvlgx.streamlit.app/), which shows a fixed snapshot of pre-generated synthetic results. To run the dashboard locally (see [Getting Started](#getting-started) for the full pipeline):

```powershell
python -m streamlit run dashboard.py
```

## Architecture

```
Synthetic generator  ─►  JSON files  ─►  PySpark Structured Streaming
                                              │
                          ┌───────────────────┴───────────────────┐
                          ▼                                       ▼
              Isolation Forest (vitals)                 DBSCAN (locations)
                          └───────────────────┬───────────────────┘
                                              ▼
                                SQLite  ◄──  risk score
                                  │
                        Prophet forecast (writes back to SQLite)
                                  │
                                  ▼
                          Streamlit dashboard
```

- **Generator** (`generate_data.py`): writes one file of readings and one file of symptom reports per simulated 5-minute tick.
- **Spark pipeline** (`spark_pipeline.py`): two streaming queries in one local Spark process (`local[2]`). It validates the records, scores readings with Isolation Forest, and re-runs DBSCAN on the last 60 simulated minutes whenever new reports arrive.
- **Risk score** (`detection.py`): `100 × (0.5 × anomaly part + 0.5 × cluster part)`, labelled Low, Medium or High. It is a simple heuristic, not a probability.
- **SQLite** (`database.py`): stores results and the time each event was created and processed.
- **Forecast** (`forecast.py`): a trend-only Prophet model of total cases per 15 simulated minutes.
- **Dashboard** (`dashboard.py`): reads the database through read-only connections; it never changes data.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3.12 |
| Stream processing | PySpark 4.2 (Spark Structured Streaming), Java 17 |
| Machine learning | scikit-learn (Isolation Forest, DBSCAN) |
| Forecasting | Prophet |
| Storage | SQLite |
| Dashboard | Streamlit, Plotly (OpenStreetMap background) |
| Testing | pytest |

## Results on Synthetic Data

These figures come from the project's own generated data and from one short run on a single desktop PC. They show that the pipeline works as designed; they do not demonstrate clinical performance.

**Detection on held-out data** (2,000 readings and 1,214 location reports that were not used for training):

| Detector | Precision | Recall | F1 | False-positive rate |
|---|---|---|---|---|
| Isolation Forest (vital signs) | 74.6% | 84.8% | 79.4% | 0.9% |
| DBSCAN (outbreak reports) | 98.9% | 81.0% | 89.1% | 0.3% |

- DBSCAN found all 5 of the 5 generated outbreaks.
- Isolation Forest detected every generated fever, low-blood-pressure, very-high-blood-pressure and fast-heart-rate-with-low-oxygen reading, but **missed all bradycardia (slow heart rate) cases**, because a low heart rate on its own is not unusual enough for the model.

**Latency in one short live run** (120 ticks written one second apart while Spark was running):

| | Median | 95th percentile |
|---|---|---|
| Patient readings | 1.47 s | 2.48 s |
| Location reports | 1.45 s | 2.35 s |

Latency is the time from an event being written to its result being stored. The slowest events took about 5 seconds.

## Screenshots

**Summary, filters and flagged readings**
![Dashboard overview](screenshots/dashboard-overview.png)

**Map of reports, district risk and active clusters**
![Geographic activity](screenshots/geographic-activity.png)

**Case forecast, model evaluation and processing latency**
![Forecast, evaluation and performance](screenshots/forecast-and-evaluation.png)

## Getting Started

Requirements: Windows 10 or 11, Python 3.12 and Java 17 (for example Eclipse Temurin JDK 17). Run these commands in PowerShell.

```powershell
# 1. Clone the repository
git clone https://github.com/laharip1418/healthcare-anomaly-outbreak-detection.git
cd healthcare-anomaly-outbreak-detection

# 2. Create and activate a virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install the requirements
python -m pip install -r requirements.txt

# 4. Windows setup: checks Java and downloads a small file-system JAR for Spark
.\setup_windows.ps1

# 5. Generate the training and held-out test data
python generate_data.py --mode train

# 6. Train Isolation Forest and evaluate both detectors
python train_models.py

# 7. Run the complete demo
python run_demo.py --seed 101

# 8. Start the dashboard
python -m streamlit run dashboard.py
```

If PowerShell blocks the scripts, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, which affects only the current window.

`setup_windows.ps1` downloads `hadoop-bare-naked-local-fs` (a pure-Java JAR from Maven Central) and verifies its checksums, so Spark can read and write local files on Windows without `winutils.exe`.

## Quick Demo

```powershell
python run_demo.py --seed 101
python -m streamlit run dashboard.py
```

Then open **http://localhost:8501** (stop the dashboard with Ctrl+C).

`run_demo.py` checks that the packages, trained model, Java and the Spark JAR are available and asks before it replaces the previous demo data (stream files, the SQLite database, Spark checkpoints and the forecast). It then generates 120 ticks of synthetic events with the chosen seed, processes them once with Spark, creates the Prophet forecast and checks the database. It takes about a minute. Because all files exist before Spark starts, its latency figures are not real-time measurements; use the live run below for that.

## Manual Pipeline

<details>
<summary>Run each stage yourself</summary>

```powershell
python generate_data.py --mode train                     # training and held-out test data (seeds 42 and 43)
python train_models.py                                   # train and evaluate the models
python generate_data.py --mode stream --fresh --seed 101 --ticks 120 --seconds-per-tick 0
python spark_pipeline.py --mode once --reset             # process the files once, then stop
python forecast.py                                       # forecast case counts
python -m streamlit run dashboard.py                     # dashboard at http://localhost:8501
```

Live (real-time) run, in two PowerShell windows:

```powershell
# window 1: wait until it prints "Streaming queries started"
python spark_pipeline.py --mode continuous --reset --stop-after 180

# window 2: writes one tick per second for about two minutes
python generate_data.py --mode stream --fresh --seed 101
```

Afterwards run `python forecast.py` and click **Refresh data** on the dashboard. `--fresh` deletes old stream files and `--reset` deletes the database and Spark checkpoints; use them together. Every script lists its options with `--help`.

</details>

## Project Structure

```
├── generate_data.py      synthetic events and separate ground-truth labels
├── spark_pipeline.py     Spark Structured Streaming pipeline
├── detection.py          Isolation Forest, DBSCAN and the risk score
├── train_models.py       model training and evaluation on held-out data
├── forecast.py           Prophet case-count forecast
├── database.py           SQLite tables and helpers
├── dashboard.py          Streamlit dashboard
├── run_demo.py           one-command demo
├── config.py             all settings (seeds, sizes, model and Spark settings)
├── setup_windows.ps1     Windows setup for Spark
├── deployment/           read-only preview with a fixed snapshot of synthetic results
├── tests/                pytest tests
└── screenshots/          dashboard images
```

Generated data, the trained model, the database and the downloaded JAR are created locally and are not stored in Git.

## Testing

```powershell
python -m pytest
```

Result: **65 tests passed**. The tests cover the generator, validation rules, detection, database, forecast, dashboard, demo runner and hosted preview; they do not start Spark. The Spark pipeline itself was checked with the one-command demo and the live run described above.

## Limitations

- Synthetic data only; the system has not been validated on real clinical data and must not be used for medical decisions.
- Spark runs locally in a single process (`local[2]`); micro-batches are processed with pandas, which suits small data only.
- The risk score is a fixed heuristic with hand-picked weights and thresholds.
- The forecast is trend-only and covers total cases, not individual districts; its accuracy has not been measured.
- Isolation Forest misses readings whose only abnormal value is a slow heart rate (bradycardia).
- The map background uses OpenStreetMap and needs internet; an offline latitude/longitude view is available with the **Map background** switch.
