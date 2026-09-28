"""Tests for the synthetic data generator.

Run from the project folder with:  .\\.venv\\Scripts\\python.exe -m pytest
"""

import json
from datetime import datetime

import config
import generate_data

LABEL_FIELDS = {"is_anomaly", "anomaly_type", "is_corrupt", "corruption_type", "is_outbreak_event", "outbreak_id"}


def generate_ticks(seed, num_ticks):
    simulator = generate_data.StreamSimulator(seed)
    return [simulator.next_tick() for _ in range(num_ticks)]


def test_same_seed_gives_same_data():
    first = generate_ticks(seed=7, num_ticks=40)
    second = generate_ticks(seed=7, num_ticks=40)
    assert first == second


def test_different_seed_gives_different_data():
    first = generate_ticks(seed=7, num_ticks=5)
    second = generate_ticks(seed=8, num_ticks=5)
    assert first[0]["vital_events"] != second[0]["vital_events"]


def test_events_do_not_contain_labels():
    for tick in generate_ticks(seed=1, num_ticks=40):
        for event in tick["vital_events"] + tick["geo_events"]:
            assert not LABEL_FIELDS & set(event)


def test_every_event_has_a_matching_label():
    for tick in generate_ticks(seed=1, num_ticks=40):
        assert [e["event_id"] for e in tick["vital_events"]] == [l["event_id"] for l in tick["vital_labels"]]
        assert [e["event_id"] for e in tick["geo_events"]] == [l["event_id"] for l in tick["geo_labels"]]


def test_valid_vitals_are_in_plausible_ranges():
    ticks = generate_ticks(seed=3, num_ticks=100)
    for tick in ticks:
        for event, label in zip(tick["vital_events"], tick["vital_labels"]):
            if label["is_corrupt"]:
                continue
            assert 20 <= event["heart_rate"] <= 250
            assert event["diastolic_bp"] < event["systolic_bp"]
            assert 50 <= event["oxygen_saturation"] <= 100
            assert 30 <= event["body_temperature"] <= 44
            assert 4 <= event["respiratory_rate"] <= 60


def test_training_data_has_expected_anomaly_rate():
    events, labels = generate_data.generate_vitals_dataset(seed=5, num_rows=3000, id_prefix="T")
    assert len(events) == len(labels) == 3000
    assert not LABEL_FIELDS & set(events.columns)
    anomaly_rate = labels["is_anomaly"].mean()
    assert abs(anomaly_rate - config.ANOMALY_RATE) < 0.015
    assert not labels["is_corrupt"].any()


def test_outbreak_events_stay_inside_outbreak_radius():
    simulator = generate_data.StreamSimulator(seed=11)
    ticks = [simulator.next_tick() for _ in range(config.FIRST_OUTBREAK_TICK + config.OUTBREAK_DURATION_TICKS)]
    outbreaks = {o["outbreak_id"]: o for o in simulator.outbreaks}
    assert outbreaks, "expected at least one synthetic outbreak"

    outbreak_events = 0
    for tick in ticks:
        for event, label in zip(tick["geo_events"], tick["geo_labels"]):
            if label["outbreak_id"] is None:
                continue
            outbreak = outbreaks[label["outbreak_id"]]
            distance = generate_data.distance_km(
                event["latitude"], event["longitude"], outbreak["center_latitude"], outbreak["center_longitude"]
            )
            assert distance <= outbreak["radius_km"] + 0.01
            assert event["symptom_category"] == outbreak["symptom_category"]
            outbreak_events += 1
    assert outbreak_events > 0


def test_stream_writes_events_and_labels_to_separate_folders(tmp_path):
    vitals_dir, geo_dir, labels_dir = tmp_path / "vitals", tmp_path / "geo", tmp_path / "labels"
    written = generate_data.run_stream(
        num_ticks=3, seconds_per_tick=0, seed=2, vitals_dir=vitals_dir, geo_dir=geo_dir, labels_dir=labels_dir
    )
    assert written == 3
    assert len(list(vitals_dir.glob("*.json"))) == 3
    assert not list(vitals_dir.glob("*.tmp"))
    assert (labels_dir / "vitals_labels.jsonl").exists()

    with open(vitals_dir / "vitals_000001.json", encoding="utf-8") as file:
        events = [json.loads(line) for line in file]
    assert len(events) == config.VITALS_PER_TICK
    for event in events:
        assert not LABEL_FIELDS & set(event)
        # Both timestamps must be timezone-aware ISO 8601 strings.
        assert datetime.fromisoformat(event["event_time"]).tzinfo is not None
        assert datetime.fromisoformat(event["created_at"]).tzinfo is not None
