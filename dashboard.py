"""Streamlit dashboard for the SYNTHETIC healthcare anomaly and outbreak demo.

Run:   python -m streamlit run dashboard.py
Open:  http://localhost:8501          (stop with Ctrl+C in the terminal)

The dashboard only READS results. It never writes to SQLite, trains models,
starts Spark or generates data. Every query uses a short read-only connection
that is closed straight away. Ground-truth labels are never read.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import database
import forecast

REGION_NAMES = {region["region_id"]: region["region_name"] for region in config.REGIONS}
SEVERITY_LABELS = {1: "1 (lowest)", 2: "2", 3: "3", 4: "4 (highest)"}
TABLES = ["vitals_results", "geo_events", "clusters", "region_risk", "forecasts"]

# Colours come from a colour-blind-checked palette. Risk levels are always also
# written as text, so colour is never the only way to read them.
HISTORY_COLOR = "#2a78d6"
FORECAST_COLOR = "#eb6834"
FORECAST_BAND = "rgba(235, 104, 52, 0.18)"
CLUSTER_COLOR = "#2a78d6"
NOISE_COLOR = "#898781"
RISK_COLORS = {"Low": "#0ca30c", "Medium": "#fab219", "High": "#d03b3b"}


# ---------------------------------------------------------------------------
# Data loading (plain functions, also used by the tests)
# ---------------------------------------------------------------------------
def database_status(db_path: Path) -> str:
    """'missing' (no file), 'empty' (no pipeline tables yet) or 'ok'. Never creates the file."""
    if not Path(db_path).exists():
        return "missing"
    if "vitals_results" not in database.table_names(db_path):
        return "empty"
    return "ok"


def add_time_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Add a 'time' column (UTC, without time zone) parsed from a stored UTC text column."""
    frame = frame.copy()
    if column in frame.columns:
        frame["time"] = pd.to_datetime(frame[column], utc=True).dt.tz_localize(None)
    else:
        frame["time"] = pd.Series(dtype="datetime64[ns]")
    return frame


def load_data(db_path: Path) -> dict[str, pd.DataFrame]:
    """Read every results table (empty DataFrame if a table does not exist yet)."""
    existing = database.table_names(db_path)
    data = {}
    for table in TABLES:
        if table in existing:
            data[table] = database.read_only_query(f"SELECT * FROM {table}", db_path=db_path)
        else:
            data[table] = pd.DataFrame()
    data["vitals_results"] = add_time_column(data["vitals_results"], "event_time")
    data["geo_events"] = add_time_column(data["geo_events"], "event_time")
    data["region_risk"] = add_time_column(data["region_risk"], "window_end")
    data["forecasts"] = add_time_column(data["forecasts"], "forecast_time")
    return data


def latest_risk(risk: pd.DataFrame) -> pd.DataFrame:
    """Risk rows of the most recent window."""
    if risk.empty:
        return risk
    return risk[risk["window_end"] == risk["window_end"].max()]


def latest_clusters(clusters: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    """Clusters found in the most recent DBSCAN window.

    Every DBSCAN run saves risk rows for its window, but cluster rows only when it
    found clusters, so the latest window is taken from the risk table.
    """
    if clusters.empty or risk.empty:
        return clusters.iloc[0:0]
    return clusters[clusters["window_end"] == risk["window_end"].max()]


def summarize(data: dict[str, pd.DataFrame]) -> dict:
    """Headline numbers for the summary tiles (whole database, no filters)."""
    vitals, geo, clusters = data["vitals_results"], data["geo_events"], data["clusters"]
    latest = latest_risk(data["region_risk"])
    top = latest.sort_values("risk_score", ascending=False).iloc[0] if not latest.empty else None
    return {
        "patient_events": len(vitals),
        "patient_anomalies": int(vitals["predicted_anomaly"].sum()) if not vitals.empty else 0,
        "geo_events": len(geo),
        "clusters_latest_window": len(latest_clusters(clusters, data["region_risk"])),
        "cluster_records": len(clusters),
        "top_risk_score": float(top["risk_score"]) if top is not None else None,
        "top_risk_level": top["risk_level"] if top is not None else None,
        "top_risk_region": top["region"] if top is not None else None,
        "latest_window_end": top["window_end"] if top is not None else None,
    }


def latency_summary(frame: pd.DataFrame) -> dict | None:
    """Median, 95th percentile and maximum latency, plus throughput when it can be calculated."""
    if frame.empty or "latency_seconds" not in frame.columns:
        return None
    latency = frame["latency_seconds"].dropna()
    if latency.empty:
        return None
    processed = pd.to_datetime(frame["processed_at"], utc=True)
    span = (processed.max() - processed.min()).total_seconds()
    return {
        "events": len(latency),
        "median": float(latency.median()),
        "p95": float(latency.quantile(0.95)),
        "max": float(latency.max()),
        # events stored per second between the first and last stored event
        "throughput": len(frame) / span if len(frame) >= 2 and span >= 1 else None,
        "span_seconds": span,
    }


def load_phase2_results(path: Path = config.PHASE2_RESULTS_PATH) -> dict | None:
    """The held-out evaluation saved by train_models.py (read-only), or None if it is missing."""
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def filter_vitals(vitals: pd.DataFrame, start, end, regions: list[str], flagged_only: bool) -> pd.DataFrame:
    if vitals.empty:
        return vitals
    keep = vitals["time"].between(start, end) & vitals["region"].isin(regions)
    if flagged_only:
        keep &= vitals["predicted_anomaly"] == 1
    return vitals[keep]


def filter_geo(geo: pd.DataFrame, start, end, regions: list[str],
               severities: list[int], symptoms: list[str]) -> pd.DataFrame:
    if geo.empty:
        return geo
    keep = (geo["time"].between(start, end) & geo["region"].isin(regions)
            & geo["severity"].isin(severities) & geo["symptom_type"].isin(symptoms))
    return geo[keep]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def map_figure(geo: pd.DataFrame, latest: pd.DataFrame, use_tiles: bool) -> go.Figure:
    """Events and regional risk on an OpenStreetMap background, or on a plain latitude/longitude chart."""
    figure = go.Figure()

    def add_points(lat, lon, **settings):
        if use_tiles:
            figure.add_trace(go.Scattermap(lat=lat, lon=lon, **settings))
        else:
            figure.add_trace(go.Scatter(x=lon, y=lat, **settings))

    # Regional risk from the latest window: large see-through circles labelled with the level.
    if not latest.empty:
        regions = pd.DataFrame(config.REGIONS).merge(latest, left_on="region_id", right_on="region")
        for level, color in RISK_COLORS.items():
            part = regions[regions["risk_level"] == level]
            if part.empty:
                continue
            add_points(
                part["latitude"], part["longitude"], mode="markers+text", name=f"Region risk: {level}",
                marker=dict(size=48, color=color, opacity=0.35),
                text=[f"{name}<br>{level} ({score:.0f})" for name, score in zip(part["region_name"], part["risk_score"])],
                textposition="middle center",
                hovertext=[f"{name}: {level} risk, score {score:.1f}<br>"
                           f"anomaly part {a:.2f}, cluster part {c:.2f}"
                           for name, score, a, c in zip(part["region_name"], part["risk_score"],
                                                        part["anomaly_part"], part["cluster_part"])],
                hoverinfo="text",
            )

    # Events: grey = not clustered (noise); blue = clustered in the most recent DBSCAN window that
    # included the event, so blue points can remain after their cluster is no longer active.
    if not geo.empty:
        in_cluster = geo["in_cluster"].astype(bool)
        for part, name, color, opacity in ((geo[~in_cluster], "Not clustered", NOISE_COLOR, 0.6),
                                           (geo[in_cluster], "Clustered when processed", CLUSTER_COLOR, 0.9)):
            if part.empty:
                continue
            add_points(
                part["latitude"], part["longitude"], mode="markers", name=name,
                marker=dict(size=8 + 2 * part["case_count"].clip(upper=6), color=color, opacity=opacity),
                hovertext=[f"{e} at {t:%H:%M}<br>{REGION_NAMES.get(r, r)}<br>{s}, severity {v}, {c} case(s)<br>"
                           f"cluster: {k if isinstance(k, str) else 'none (noise)'}"
                           for e, t, r, s, v, c, k in zip(part["event_id"], part["time"], part["region"],
                                                           part["symptom_type"], part["severity"],
                                                           part["case_count"], part["cluster_id"])],
                hoverinfo="text",
            )

    center_lat = sum(region["latitude"] for region in config.REGIONS) / len(config.REGIONS)
    center_lon = sum(region["longitude"] for region in config.REGIONS) / len(config.REGIONS)
    if use_tiles:
        figure.update_layout(map=dict(style="open-street-map", center=dict(lat=center_lat, lon=center_lon), zoom=10.8))
    else:
        figure.update_layout(xaxis_title="Longitude", yaxis_title="Latitude",
                             yaxis=dict(scaleanchor="x", scaleratio=1))
    figure.update_layout(height=540, margin=dict(l=0, r=0, t=30, b=0),
                         legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
    return figure


def forecast_figure(history: pd.DataFrame, predicted: pd.DataFrame) -> go.Figure:
    """Historical case counts (blue) followed by the Prophet forecast (orange, dashed, with its range)."""
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=history["ds"], y=history["y"], mode="lines+markers", name="History",
        line=dict(color=HISTORY_COLOR, width=2), marker=dict(size=5),
    ))
    if not predicted.empty:
        figure.add_trace(go.Scatter(
            x=list(predicted["time"]) + list(predicted["time"][::-1]),
            y=list(predicted["upper_cases"]) + list(predicted["lower_cases"][::-1]),
            fill="toself", fillcolor=FORECAST_BAND, line=dict(width=0),
            name=f"Forecast range ({config.FORECAST_INTERVAL_WIDTH:.0%})", hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=predicted["time"], y=predicted["predicted_cases"], mode="lines+markers", name="Prophet forecast",
            line=dict(color=FORECAST_COLOR, width=2, dash="dash"), marker=dict(size=5),
        ))
        start = predicted["time"].min()
        figure.add_shape(type="line", x0=start, x1=start, y0=0, y1=1, yref="paper",
                         line=dict(color=NOISE_COLOR, dash="dot", width=1))
        figure.add_annotation(x=start, y=1, yref="paper", text="Forecast starts", showarrow=False,
                              xanchor="left", yanchor="bottom")
    figure.update_layout(
        height=380, hovermode="x unified", margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Simulated time (UTC)",
        yaxis=dict(title=f"Cases per {config.FORECAST_INTERVAL_MINUTES} simulated minutes", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return figure


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
def percent(value: float) -> str:
    """0.7042 -> '70.4%' (display only; the stored value is not changed)."""
    return f"{value:.1%}"


def show_missing_database(status: str) -> None:
    what = "No results database was found" if status == "missing" else "The results database has no pipeline tables yet"
    st.info(
        f"{what}. Run these commands from the project folder, then click **Refresh data**:\n\n"
        "```\npython generate_data.py --mode train\npython train_models.py\n"
        "python generate_data.py --mode stream --fresh\npython spark_pipeline.py --mode once --reset\n"
        "python forecast.py\n```"
    )


def show_about_metrics() -> None:
    with st.expander("About the metrics", expanded=False):
        st.markdown(
            "- **Anomaly score** – the Isolation Forest score of a patient reading; higher is more unusual, "
            "and readings above 0 are flagged. Flags are model predictions, not ground truth.\n"
            f"- **Latest DBSCAN window** – DBSCAN clusters the geographic events of the last "
            f"{config.DBSCAN_WINDOW_MINUTES} simulated minutes whenever new events arrive. *Active clusters* "
            "counts only the newest window; blue map points keep the cluster from the last window that "
            "included them.\n"
            "- **Risk score** – a simple academic heuristic from 0 to 100 (50% abnormal-reading share, "
            "50% clustered cases in the latest window). It is not a probability.\n"
            f"- **Forecast range** – Prophet's {config.FORECAST_INTERVAL_WIDTH:.0%} uncertainty interval for "
            f"cases per {config.FORECAST_INTERVAL_MINUTES} simulated minutes; values below 0 are shown as 0.\n"
            "- **Processing latency** – processed_at − created_at for each stored event. After "
            "`--mode once` it includes the time the files waited before Spark started.\n"
            "- **Throughput** – stored events divided by the time between the first and last stored event; "
            "it is not a maximum capacity.\n"
            "- **Evaluation results** – precision, recall, F1 and false-positive rate from `train_models.py` on "
            "held-out synthetic test data. They depend on how the data was generated and do not show clinical "
            "performance."
        )


def render() -> None:
    st.set_page_config(page_title="Healthcare Anomaly and Outbreak Detection", layout="wide")
    st.title("Healthcare Anomaly and Outbreak Detection")
    st.caption("Synthetic healthcare monitoring dashboard")

    with st.sidebar:
        st.header("Controls")
        st.button("Refresh data", help="Read the latest results from the database again")

    status = database_status(config.DATABASE_PATH)
    if status != "ok":
        show_missing_database(status)
        st.stop()

    data = load_data(config.DATABASE_PATH)
    vitals, geo, clusters = data["vitals_results"], data["geo_events"], data["clusters"]
    risk, predicted = data["region_risk"], data["forecasts"]
    if vitals.empty and geo.empty:
        st.info("The database exists but no events have been processed yet. Run:\n\n"
                "```\npython generate_data.py --mode stream --fresh\npython spark_pipeline.py --mode once --reset\n```")
        st.stop()

    # ---------------- Sidebar filters ----------------
    with st.sidebar:
        st.header("Filters")
        all_times = pd.concat([vitals["time"], geo["time"]])
        first, last = all_times.min().to_pydatetime(), all_times.max().to_pydatetime()
        if first < last:
            start, end = st.slider("Time range", min_value=first, max_value=last, value=(first, last),
                                   step=timedelta(minutes=5), format="MM-DD HH:mm", help="Simulated time (UTC)")
        else:
            start, end = first, last
        regions = st.multiselect("Region", options=list(REGION_NAMES), default=list(REGION_NAMES),
                                 format_func=lambda r: f"{REGION_NAMES[r]} ({r})")
        severities = st.multiselect("Severity", options=list(SEVERITY_LABELS),
                                    default=list(SEVERITY_LABELS), format_func=SEVERITY_LABELS.get)
        symptom_options = sorted(geo["symptom_type"].dropna().unique()) if not geo.empty else []
        symptoms = st.multiselect("Symptom type", options=symptom_options, default=symptom_options)
        readings = st.radio("Patient readings", ["Flagged anomalies only", "All readings"])
        use_tiles = st.toggle("Map background", value=True, help="Turn off to use the offline latitude/longitude view.")
        st.caption("Filters apply to readings and geographic activity.")

    # ---------------- 1. Summary ----------------
    st.subheader("1. System summary")
    summary = summarize(data)
    row = st.columns(3)
    row[0].metric("Processed patient readings", f"{summary['patient_events']:,}")
    row[1].metric("Readings flagged by Isolation Forest", f"{summary['patient_anomalies']:,}")
    row[2].metric("Processed geographic events", f"{summary['geo_events']:,}")
    row = st.columns(3)
    row[0].metric("Active clusters — latest window", summary["clusters_latest_window"],
                  help="Counts only clusters active in the newest DBSCAN processing window.")
    if summary["top_risk_score"] is not None:
        region_name = REGION_NAMES.get(summary["top_risk_region"], summary["top_risk_region"])
        window_end = pd.to_datetime(summary["latest_window_end"], utc=True)
        row[1].metric("Highest regional risk score", f"{summary['top_risk_score']:.1f} / 100",
                      help=f"Latest window, ending {window_end:%Y-%m-%d %H:%M} simulated UTC")
        row[2].metric(f"Risk level — {region_name}", summary["top_risk_level"])
    else:
        row[1].metric("Highest regional risk score", "—")
        row[2].metric("Risk level", "—")

    # ---------------- 2. Recent patient readings ----------------
    st.subheader("2. Recent patient readings")
    shown = filter_vitals(vitals, start, end, regions, readings == "Flagged anomalies only")
    if shown.empty:
        st.info("No patient readings match the current filters.")
    else:
        recent = shown.sort_values("event_time", ascending=False).head(50)
        st.dataframe(pd.DataFrame({
            "Simulated time": recent["time"].dt.strftime("%Y-%m-%d %H:%M"),
            "Patient": recent["patient_id"],
            "Region": recent["region"].map(REGION_NAMES),
            "Heart rate": recent["heart_rate"],
            "Blood pressure": recent["systolic_bp"].astype(str) + "/" + recent["diastolic_bp"].astype(str),
            "SpO₂ %": recent["oxygen_saturation"],
            "Temp °C": recent["body_temperature"],
            "Resp. rate": recent["respiratory_rate"],
            "Anomaly score": recent["anomaly_score"].round(3),
            "Flagged by model": recent["predicted_anomaly"].map({1: "Yes", 0: "No"}),
        }), hide_index=True)
        st.caption(f"Newest {len(recent)} of {len(shown):,} matching readings")

    # ---------------- 3. Geographic activity ----------------
    st.subheader("3. Geographic activity")
    geo_shown = filter_geo(geo, start, end, regions, severities, symptoms)
    latest = latest_risk(risk)
    latest_shown = latest[latest["region"].isin(regions)] if not latest.empty else latest
    try:
        figure = map_figure(geo_shown, latest_shown, use_tiles)
    except Exception:   # never let a map problem break the page
        st.caption("Map background unavailable; showing the latitude/longitude view.")
        figure = map_figure(geo_shown, latest_shown, use_tiles=False)
    st.plotly_chart(figure)
    st.caption(f"{len(geo_shown):,} events shown · marker size = case count")

    st.markdown("**Regional risk — latest window**")
    if latest_shown.empty:
        st.info("No regional risk scores yet.")
    else:
        st.dataframe(pd.DataFrame({
            "Region": latest_shown["region"].map(REGION_NAMES),
            "Abnormal-reading share": latest_shown["anomaly_rate"],
            "Clustered cases": latest_shown["clustered_cases"],
            "Anomaly part": latest_shown["anomaly_part"],
            "Cluster part": latest_shown["cluster_part"],
            "Risk score": latest_shown["risk_score"],
            "Risk level": latest_shown["risk_level"],
        }).sort_values("Risk score", ascending=False), hide_index=True)
    st.markdown("**Active clusters — latest window**")
    newest = latest_clusters(clusters, risk)
    if newest.empty:
        st.info("No active clusters in the latest window.")
    else:
        st.dataframe(pd.DataFrame({
            "Cluster": newest["cluster_id"],
            "Region": newest["region"].map(REGION_NAMES),
            "Events": newest["event_count"],
            "Cases": newest["total_cases"],
            "Centre latitude": newest["center_latitude"].round(4),
            "Centre longitude": newest["center_longitude"].round(4),
        }), hide_index=True)

    # ---------------- 4. Case history and forecast ----------------
    st.subheader("4. Case history and forecast")
    history = forecast.aggregate_case_counts(geo[["event_time", "case_count"]]) if not geo.empty else pd.DataFrame()
    if history.empty:
        st.info("No case-count history yet.")
    else:
        st.plotly_chart(forecast_figure(history, predicted))
        if predicted.empty:
            st.info("No forecast yet. Run `python forecast.py`, then click **Refresh data**.")
        else:
            generated = pd.to_datetime(predicted["generated_at"].iloc[0], utc=True)
            trained_on = int(predicted["history_points"].iloc[0])
            st.caption(f"Forecast generated {generated:%Y-%m-%d %H:%M} UTC from {trained_on} history intervals")
            if trained_on != len(history):
                st.info("New events have been processed since this forecast was made. "
                        "Run `python forecast.py` again to update it.")

    # ---------------- 5. Model evaluation ----------------
    st.subheader("5. Model evaluation")
    st.caption("Held-out synthetic test data")
    results = load_phase2_results()
    if results is None:
        st.info("No evaluation results found. Run `python train_models.py`.")
    else:
        rows = []
        for detector, key, items in (("Isolation Forest (patient vital signs)", "isolation_forest", "test_rows"),
                                     ("DBSCAN (outbreak events)", "dbscan", "test_events")):
            result = results[key]
            rows.append({
                "Detector": detector, "Test items": int(result[items]),
                "Precision": percent(result["precision"]), "Recall": percent(result["recall"]),
                "F1": percent(result["f1"]),
                # older results files (before Phase 5) do not have this value
                "False-positive rate": percent(result["false_positive_rate"]) if "false_positive_rate" in result else "—",
                "TP": int(result["true_positives"]), "FP": int(result["false_positives"]),
                "FN": int(result["false_negatives"]), "TN": int(result["true_negatives"]),
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    # ---------------- 6. Processing performance ----------------
    st.subheader("6. Processing performance")
    groups = st.columns(2)
    for group, (label, frame) in zip(groups, (("Patient readings", vitals), ("Geographic events", geo))):
        stats = latency_summary(frame)
        with group.container(border=True):
            st.markdown(f"**{label}**")
            if stats is None:
                st.info("No latency data yet.")
                continue
            top, bottom = st.columns(2), st.columns(2)
            top[0].metric("Median latency", f"{stats['median']:.2f} s")
            top[1].metric("95th percentile", f"{stats['p95']:.2f} s")
            bottom[0].metric("Maximum latency", f"{stats['max']:.2f} s")
            # the unit is in the label so the value fits its tile at normal desktop width
            if stats["throughput"] is None:
                bottom[1].metric("Throughput (events/s)", "—",
                                 help="Needs at least two events processed at least 1 s apart.")
            else:
                bottom[1].metric("Throughput (events/s)", f"{stats['throughput']:.1f}",
                                 help=f"{stats['events']:,} events stored over {stats['span_seconds']:.0f} s")

    show_about_metrics()
    st.divider()
    st.caption("Academic project using simulated data.")


if __name__ == "__main__":
    render()
