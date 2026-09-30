"""Run the whole SYNTHETIC demonstration with one command.

All data is simulated for academic demonstration only.

Steps (each one runs an existing project script with the current Python):
  1. check that the packages, trained model, Spark JAR and scripts exist
  2. python generate_data.py --mode stream --fresh --seed SEED --ticks TICKS --seconds-per-tick 0
  3. python spark_pipeline.py --mode once --reset        (Spark stays at local[2])
  4. python forecast.py
  5. check the SQLite results and print a summary

Because every file is written before Spark starts, the latency of this quick
run is NOT a real-time measurement (see README.md for the live demonstration).

Usage:
  python run_demo.py --seed 101
  python run_demo.py --seed 101 --ticks 120 --yes      (--yes: do not ask before replacing data)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import config
import database

REQUIRED_PACKAGES = ["pyspark", "pandas", "sklearn", "joblib", "prophet", "streamlit", "plotly"]
REQUIRED_SCRIPTS = ["generate_data.py", "spark_pipeline.py", "forecast.py", "dashboard.py"]
# Prophet needs FORECAST_MIN_POINTS complete intervals of FORECAST_INTERVAL_MINUTES each.
MIN_TICKS = config.FORECAST_MIN_POINTS * config.FORECAST_INTERVAL_MINUTES // config.SIMULATED_MINUTES_PER_TICK
MAX_TICKS = 600                  # keeps the demonstration small on a shared computer
DEFAULT_TICKS = 120              # 10 simulated hours: 40 forecast intervals and two synthetic outbreaks


class DemoError(Exception):
    """A step failed; the message says what to do."""


def relative(path: Path) -> str:
    """Show project paths relative to the project folder (never a personal absolute path)."""
    try:
        return Path(path).resolve().relative_to(config.PROJECT_ROOT).as_posix()
    except ValueError:            # outside the project (for example a test folder): show the name only
        return Path(path).name


def check_prerequisites() -> None:
    missing = [name for name in REQUIRED_PACKAGES if importlib.util.find_spec(name) is None]
    if missing:
        raise DemoError(f"Missing Python packages: {', '.join(missing)}.\n"
                        "Activate the virtual environment (.\\.venv\\Scripts\\Activate.ps1) and run: "
                        "pip install -r requirements.txt")
    if not config.ISOLATION_FOREST_PATH.is_file():
        raise DemoError(f"No trained model at {relative(config.ISOLATION_FOREST_PATH)}. Run first:\n"
                        "  python generate_data.py --mode train\n  python train_models.py")
    if not config.BARE_LOCAL_FS_JAR.is_file():
        raise DemoError(f"Missing {relative(config.BARE_LOCAL_FS_JAR)}. Run .\\setup_windows.ps1 first.")
    java_home = os.environ.get("JAVA_HOME")
    if not ((java_home and shutil.which("java", path=str(Path(java_home) / "bin"))) or shutil.which("java")):
        raise DemoError("Java 17 was not found. Install it (see README.md) and open a new PowerShell window.")
    for script in REQUIRED_SCRIPTS:
        if not (config.PROJECT_ROOT / script).is_file():
            raise DemoError(f"Missing project script: {script}")
    config.DATA_DIR.mkdir(exist_ok=True)


def confirm_replacement(assume_yes: bool) -> None:
    print("This demonstration will replace:\n"
          f"  - the generated stream files in {relative(config.DATA_DIR / 'incoming')}/ and their labels\n"
          f"  - the demonstration database {relative(config.DATABASE_PATH)}\n"
          f"  - the Spark checkpoints in {relative(config.CHECKPOINT_DIR)}/\n"
          "  - the previous Prophet forecast")
    if assume_yes:
        return
    if not sys.stdin.isatty():
        raise DemoError("Not running interactively: add --yes to confirm the replacement.")
    if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
        raise DemoError("Cancelled; nothing was changed.")


def run_step(title: str, arguments: list[str]) -> None:
    """Run one project script with the current Python interpreter; stop the demo if it fails."""
    print(f"\n=== {title} ===\n> python {' '.join(arguments)}", flush=True)
    result = subprocess.run([sys.executable, *arguments], cwd=config.PROJECT_ROOT)
    if result.returncode != 0:
        raise DemoError(f"'{title}' failed (exit code {result.returncode}). See the messages above.")


def count_labels(path: Path) -> tuple[int, int]:
    """Readings written by the generator and how many were deliberately corrupted (verification only)."""
    written = corrupt = 0
    with open(path, encoding="utf-8") as file:
        for line in file:
            written += 1
            corrupt += bool(json.loads(line)["is_corrupt"])
    return written, corrupt


def check_results(db_path: Path = config.DATABASE_PATH, labels_dir: Path = config.LABELS_DIR) -> dict:
    """Read-only checks of the demonstration database. Raises DemoError if something is missing."""
    if not Path(db_path).is_file():
        raise DemoError(f"No database was created at {relative(db_path)}.")

    def one(sql: str):
        """The single value returned by a read-only query."""
        return database.read_only_query(sql, db_path=db_path).iloc[0, 0]

    integrity = one("PRAGMA integrity_check")
    counts = {table: int(one(f"SELECT COUNT(*) FROM {table}"))
              for table in ("vitals_results", "geo_events", "region_risk", "clusters", "forecasts")}
    missing_times = int(one("SELECT COUNT(*) FROM vitals_results WHERE processed_at IS NULL OR latency_seconds IS NULL")
                        + one("SELECT COUNT(*) FROM geo_events WHERE processed_at IS NULL OR latency_seconds IS NULL"))
    problems = [f"no rows in {table}" for table in ("vitals_results", "geo_events", "region_risk", "forecasts")
                if counts[table] == 0]
    if missing_times:
        problems.append(f"{missing_times} stored events have no processing time")
    if integrity != "ok":
        problems.append(f"SQLite integrity check returned: {integrity}")
    if problems:
        raise DemoError("The results are incomplete: " + "; ".join(problems))

    written, corrupt = count_labels(Path(labels_dir) / "vitals_labels.jsonl")
    return {
        "integrity": integrity,
        "patient_readings": counts["vitals_results"],
        "readings_written": written,
        "rejected_readings": written - counts["vitals_results"],
        "corrupt_readings_written": corrupt,
        "anomalies": int(one("SELECT COALESCE(SUM(predicted_anomaly), 0) FROM vitals_results")),
        "geo_events": counts["geo_events"],
        "risk_rows": counts["region_risk"],
        "cluster_records": counts["clusters"],
        "latest_window_clusters": int(one(
            "SELECT COUNT(*) FROM clusters WHERE window_end = (SELECT MAX(window_end) FROM region_risk)")),
        "forecast_rows": counts["forecasts"],
    }


def print_summary(seed: int, ticks: int, results: dict) -> None:
    print(f"""
=== Demonstration finished (SYNTHETIC data) ===
  stream seed:                  {seed}
  ticks:                        {ticks} ({ticks * config.SIMULATED_MINUTES_PER_TICK} simulated minutes)
  patient readings stored:      {results['patient_readings']}
  readings rejected:            {results['rejected_readings']} of {results['readings_written']} \
({results['corrupt_readings_written']} were deliberately corrupted by the generator)
  flagged anomalies:            {results['anomalies']}
  geographic events:            {results['geo_events']}
  clusters in latest window:    {results['latest_window_clusters']} ({results['cluster_records']} cluster records in all windows)
  regional risk rows:           {results['risk_rows']}
  forecast rows:                {results['forecast_rows']}
  SQLite integrity check:       {results['integrity']}
  database:                     {relative(config.DATABASE_PATH)}

Latency from this quick run is not a real-time measurement (all files existed before Spark started).
Dashboard: python -m streamlit run dashboard.py   then open http://localhost:{config.STREAMLIT_PORT}
If the dashboard is already open, click "Refresh data".""")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the complete SYNTHETIC demonstration with one command.")
    parser.add_argument("--seed", type=int, default=config.STREAM_SEED,
                        help=f"stream seed (default {config.STREAM_SEED}); the same seed gives the same data")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS,
                        help=f"number of 5-minute simulated ticks, {MIN_TICKS}-{MAX_TICKS} (default {DEFAULT_TICKS})")
    parser.add_argument("--yes", action="store_true", help="replace the previous demonstration data without asking")
    args = parser.parse_args(argv)
    if not MIN_TICKS <= args.ticks <= MAX_TICKS:
        parser.error(f"--ticks must be between {MIN_TICKS} (enough history for the forecast) and {MAX_TICKS}")

    print("SYNTHETIC healthcare demonstration (academic project, not for medical use).")
    try:
        check_prerequisites()
        confirm_replacement(args.yes)
        run_step("1/3 Generate the synthetic stream", ["generate_data.py", "--mode", "stream", "--fresh",
                                                       "--seed", str(args.seed), "--ticks", str(args.ticks),
                                                       "--seconds-per-tick", "0"])
        run_step("2/3 Process it with Spark (once mode)", ["spark_pipeline.py", "--mode", "once", "--reset"])
        run_step("3/3 Forecast case counts with Prophet", ["forecast.py"])
        print_summary(args.seed, args.ticks, check_results())
    except DemoError as error:
        print(f"\nDemo stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
