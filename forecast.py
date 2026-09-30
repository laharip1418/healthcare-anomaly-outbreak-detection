"""Forecast SYNTHETIC case counts with Prophet (run by hand, never automatically).

All data is simulated. This is an academic forecast of the generator's
synthetic case counts, NOT a clinical or public-health prediction.

Steps:
  1. Read processed geographic events (event_time, case_count) from SQLite.
  2. Add up case_count per FORECAST_INTERVAL_MINUTES of simulated time.
     Intervals with no events count as 0. The newest interval is left out if
     it is still filling up (its last tick has not arrived yet).
  3. Fit a simple Prophet model (trend only: the synthetic history covers
     hours, not days, so daily/weekly/yearly seasonality are switched off).
  4. Forecast FORECAST_HORIZON_STEPS intervals ahead.
  5. Replace the previous forecast in the SQLite "forecasts" table.

Case counts cannot be negative, so any predicted, lower or upper value below
zero is stored as 0. Prophet's raw output is not changed in any other way.

Run after spark_pipeline.py has processed some events:
    python forecast.py
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
import database


class ForecastError(Exception):
    """A problem the user can fix, such as missing or too little data."""


def load_geo_history(db_path: Path = config.DATABASE_PATH) -> pd.DataFrame:
    """Read event_time and case_count of the processed geographic events (read-only)."""
    if not Path(db_path).exists():
        raise ForecastError(
            f"No database found at {db_path}.\n"
            "Process some events first:\n"
            "  python generate_data.py --mode stream --fresh\n"
            "  python spark_pipeline.py --mode once --reset"
        )
    if "geo_events" not in database.table_names(db_path):
        raise ForecastError(
            f"The database at {db_path} has no geo_events table.\n"
            "Run: python spark_pipeline.py --mode once --reset"
        )
    events = database.read_only_query("SELECT event_time, case_count FROM geo_events", db_path=db_path)
    if events.empty:
        raise ForecastError(
            "The database contains no processed geographic events yet.\n"
            "Run: python generate_data.py --mode stream --fresh\n"
            "then: python spark_pipeline.py --mode once --reset"
        )
    return events


def aggregate_case_counts(events: pd.DataFrame,
                          interval_minutes: int = config.FORECAST_INTERVAL_MINUTES) -> pd.DataFrame:
    """Sum case_count per interval and return Prophet's input columns: ds (time) and y (cases)."""
    if events.empty:
        return pd.DataFrame({"ds": pd.Series(dtype="datetime64[ns]"), "y": pd.Series(dtype="float64")})

    times = pd.to_datetime(events["event_time"], utc=True)
    cases = pd.Series(events["case_count"].to_numpy(), index=times)
    # resample() also creates the empty intervals between events, with a sum of 0.
    counts = cases.resample(f"{interval_minutes}min").sum()

    # Leave out the newest interval if its last tick has not arrived yet.
    last_interval_start = counts.index[-1]
    last_tick_of_interval = last_interval_start + timedelta(
        minutes=interval_minutes - config.SIMULATED_MINUTES_PER_TICK)
    if times.max() < last_tick_of_interval:
        counts = counts.iloc[:-1]

    # Prophet needs times without a time zone; all times here are UTC.
    return pd.DataFrame({"ds": counts.index.tz_localize(None), "y": counts.to_numpy(dtype=float)})


def check_enough_history(history: pd.DataFrame, minimum: int = config.FORECAST_MIN_POINTS) -> None:
    if len(history) < minimum:
        raise ForecastError(
            f"Only {len(history)} complete {config.FORECAST_INTERVAL_MINUTES}-minute intervals of history "
            f"are available; at least {minimum} are needed.\n"
            "Generate a longer stream and process it again, for example:\n"
            "  python generate_data.py --mode stream --minutes 2 --fresh\n"
            "  python spark_pipeline.py --mode once --reset"
        )


def fit_and_forecast(history: pd.DataFrame,
                     horizon: int = config.FORECAST_HORIZON_STEPS,
                     interval_minutes: int = config.FORECAST_INTERVAL_MINUTES,
                     interval_width: float = config.FORECAST_INTERVAL_WIDTH) -> pd.DataFrame:
    """Fit Prophet on the history and predict the next `horizon` intervals."""
    from cmdstanpy.utils import get_logger
    from prophet import Prophet   # imported here because it is slow to load and only needed now

    # Hide the fitting engine's routine progress messages (warnings are still shown).
    get_logger().setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    np.random.seed(config.MODEL_RANDOM_SEED)   # makes the uncertainty range repeatable

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=False,
        interval_width=interval_width,
    )
    model.fit(history)
    future = model.make_future_dataframe(periods=horizon, freq=f"{interval_minutes}min", include_history=False)
    return model.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def prepare_forecast_rows(predictions: pd.DataFrame, history_points: int, generated_at) -> pd.DataFrame:
    """Turn Prophet's output into database rows. Values below zero are stored as 0 (counts cannot be negative)."""
    return pd.DataFrame({
        "forecast_time": pd.to_datetime(predictions["ds"]),
        "predicted_cases": predictions["yhat"].clip(lower=0).round(2),
        "lower_cases": predictions["yhat_lower"].clip(lower=0).round(2),
        "upper_cases": predictions["yhat_upper"].clip(lower=0).round(2),
        "generated_at": generated_at,
        "history_points": history_points,
    })


def run_forecast(db_path: Path = config.DATABASE_PATH) -> dict:
    """Load history, fit Prophet, save the forecast and return a short summary."""
    events = load_geo_history(db_path)
    history = aggregate_case_counts(events)
    check_enough_history(history)

    predictions = fit_and_forecast(history)
    rows = prepare_forecast_rows(predictions, len(history), pd.Timestamp.now(tz="UTC"))
    database.create_tables(db_path)          # adds the forecasts table to databases created without it
    saved = database.replace_forecast(rows, db_path)
    return {
        "history_points": len(history),
        "history_start": history["ds"].min(),
        "history_end": history["ds"].max(),
        "forecast_points": saved,
        "forecast_start": rows["forecast_time"].min(),
        "forecast_end": rows["forecast_time"].max(),
        "database": Path(db_path),
    }


def main() -> None:
    argparse.ArgumentParser(
        description="Forecast SYNTHETIC case counts with Prophet and save them to SQLite (academic demo only)."
    ).parse_args()

    print("Forecasting SYNTHETIC case counts for academic demonstration only (not a clinical prediction).")
    try:
        summary = run_forecast()
    except ForecastError as error:
        raise SystemExit(f"\nForecast not created:\n{error}")

    fmt = "%Y-%m-%d %H:%M"
    print(f"  interval size:         {config.FORECAST_INTERVAL_MINUTES} simulated minutes")
    print(f"  historical points used: {summary['history_points']} "
          f"({summary['history_start']:{fmt}} to {summary['history_end']:{fmt}} simulated UTC)")
    print(f"  forecast points created: {summary['forecast_points']} "
          f"({summary['forecast_start']:{fmt}} to {summary['forecast_end']:{fmt}} simulated UTC)")
    database_name = summary["database"].resolve()
    if database_name.is_relative_to(config.PROJECT_ROOT):      # show project paths without the personal folder
        database_name = database_name.relative_to(config.PROJECT_ROOT).as_posix()
    print(f"  saved to:              {database_name} (table: forecasts)")
    print("Done.")


if __name__ == "__main__":
    main()
