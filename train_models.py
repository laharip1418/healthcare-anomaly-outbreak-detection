"""Train the Isolation Forest model and evaluate both detectors on held-out SYNTHETIC data.

All data and results are synthetic and intended only for academic demonstration.
They say nothing about real clinical or public-health performance.

Steps:
  1. Train Isolation Forest on data/training/vitals_train.csv (no labels used).
  2. Save it to models/isolation_forest.joblib.
  3. Score the held-out data/training/vitals_test.csv and compare with the
     separate label file (labels are used ONLY here, for evaluation).
  4. Run DBSCAN over data/training/geo_test.csv with a sliding window and
     compare with the separate outbreak labels.
  5. Save the metrics to data/results/phase2_evaluation.json.

Run after:  python generate_data.py --mode train
Usage:      python train_models.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

import config
import detection

VITALS_TRAIN = config.TRAINING_DIR / "vitals_train.csv"
VITALS_TEST = config.TRAINING_DIR / "vitals_test.csv"
VITALS_TEST_LABELS = config.LABELS_DIR / "vitals_test_labels.csv"
GEO_TEST = config.TRAINING_DIR / "geo_test.csv"
GEO_TEST_LABELS = config.LABELS_DIR / "geo_test_labels.csv"
GEO_TEST_OUTBREAKS = config.LABELS_DIR / "geo_test_outbreaks.json"


def classification_metrics(actual: pd.Series, predicted: pd.Series) -> dict:
    """Precision, recall, F1 and confusion matrix for boolean labels."""
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[False, True]).ravel()
    return {
        "precision": round(float(precision_score(actual, predicted, zero_division=0)), 4),
        "recall": round(float(recall_score(actual, predicted, zero_division=0)), 4),
        "f1": round(float(f1_score(actual, predicted, zero_division=0)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def print_metrics(metrics: dict) -> None:
    print(f"  precision: {metrics['precision']:.3f}")
    print(f"  recall:    {metrics['recall']:.3f}")
    print(f"  F1 score:  {metrics['f1']:.3f}")
    print("  confusion matrix (rows = actual, columns = predicted):")
    print("                    predicted normal   predicted anomaly/cluster")
    print(f"    actual normal   {metrics['true_negatives']:>16}   {metrics['false_positives']:>25}")
    print(f"    actual positive {metrics['false_negatives']:>16}   {metrics['true_positives']:>25}")


def evaluate_isolation_forest() -> dict:
    print("\n=== Isolation Forest (patient vital signs) ===")
    train_events = pd.read_csv(VITALS_TRAIN)
    model = detection.train_isolation_forest(train_events)          # no labels are read here
    detection.save_model(model)
    print(f"Trained on {len(train_events)} synthetic readings; saved to {config.ISOLATION_FOREST_PATH}")

    test_events = pd.read_csv(VITALS_TEST)
    predictions = detection.score_vitals(model, test_events)
    test_labels = pd.read_csv(VITALS_TEST_LABELS)                   # used only for evaluation
    results = predictions.merge(test_labels, on="event_id", how="inner", validate="one_to_one")
    assert len(results) == len(test_events), "every test event should have exactly one label"

    prevalence = float(results["is_anomaly"].mean())
    metrics = classification_metrics(results["is_anomaly"], results["predicted_anomaly"])
    recall_by_type = (
        results[results["is_anomaly"]].groupby("anomaly_type")["predicted_anomaly"].mean().round(3).to_dict()
    )

    print(f"Held-out synthetic test set: {len(results)} readings, anomaly prevalence {prevalence:.1%}")
    print_metrics(metrics)
    print("  recall by synthetic anomaly type:", recall_by_type)
    return {
        "train_rows": len(train_events),
        "test_rows": len(results),
        "anomaly_prevalence": round(prevalence, 4),
        **metrics,
        "recall_by_anomaly_type": recall_by_type,
    }


def evaluate_dbscan() -> dict:
    """Replay the geo test events in time order, like the streaming pipeline will.

    At each simulated time step, DBSCAN runs on the events from the last
    DBSCAN_WINDOW_MINUTES. The events that arrived at that step are marked as
    predicted outbreak events if they fall inside a cluster.
    """
    print("\n=== DBSCAN (geographic clusters) ===")
    events = pd.read_csv(GEO_TEST)
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
    window = timedelta(minutes=config.DBSCAN_WINDOW_MINUTES)

    predicted = {}
    for step_time in sorted(events["event_time"].unique()):
        recent = events[(events["event_time"] > step_time - window) & (events["event_time"] <= step_time)]
        clustered = detection.find_clusters(recent)
        arrived_now = clustered[clustered["event_time"] == step_time]
        predicted.update(zip(arrived_now["event_id"], arrived_now["in_cluster"]))
    events["predicted_outbreak"] = events["event_id"].map(predicted)

    labels = pd.read_csv(GEO_TEST_LABELS)                           # used only for evaluation
    results = events.merge(labels, on="event_id", how="inner", validate="one_to_one")
    assert len(results) == len(events), "every geo test event should have exactly one label"

    metrics = classification_metrics(results["is_outbreak_event"], results["predicted_outbreak"])
    with open(GEO_TEST_OUTBREAKS, encoding="utf-8") as file:
        outbreaks = json.load(file)
    detected = results[results["predicted_outbreak"] & results["is_outbreak_event"]]["outbreak_id"].nunique()

    print(f"Held-out synthetic geo events: {len(results)} "
          f"({int(results['is_outbreak_event'].sum())} belong to {len(outbreaks)} synthetic outbreaks)")
    print(f"Settings: eps {config.DBSCAN_EPS_KM} km, min_samples {config.DBSCAN_MIN_SAMPLES}, "
          f"window {config.DBSCAN_WINDOW_MINUTES} simulated minutes")
    print_metrics(metrics)
    print(f"  synthetic outbreaks with at least one clustered event: {detected} of {len(outbreaks)}")
    return {
        "test_events": len(results),
        "outbreak_events": int(results["is_outbreak_event"].sum()),
        "outbreaks": len(outbreaks),
        "outbreaks_detected": int(detected),
        **metrics,
    }


def main() -> None:
    argparse.ArgumentParser(
        description="Train Isolation Forest and evaluate both detectors on held-out SYNTHETIC data."
    ).parse_args()
    for path in (VITALS_TRAIN, VITALS_TEST, VITALS_TEST_LABELS, GEO_TEST, GEO_TEST_LABELS, GEO_TEST_OUTBREAKS):
        if not path.exists():
            raise SystemExit(f"Missing {path}.\nRun first:  python generate_data.py --mode train")

    print("RESULTS BELOW USE SYNTHETIC DATA ONLY (academic demonstration, not clinical performance).")
    report = {
        "note": "Synthetic data only. Not a measure of real clinical or public-health performance.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": {"train": config.TRAIN_SEED, "test": config.TEST_SEED},
        "isolation_forest": evaluate_isolation_forest(),
        "dbscan": evaluate_dbscan(),
    }

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = config.RESULTS_DIR / "phase2_evaluation.json"
    with open(output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(f"\nSaved synthetic-data evaluation to {output}")


if __name__ == "__main__":
    main()
