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


def passes_pipeline_validation(event):
    """The same rules spark_pipeline.validate_vitals applies (ranges from config, systolic above diastolic)."""
    for field, (low, high) in config.VITAL_VALID_RANGES.items():
        if event[field] is None or not low <= event[field] <= high:
            return False
    return event["systolic_bp"] > event["diastolic_bp"]


def test_valid_readings_always_have_systolic_above_diastolic():
    # the stream seeds used in the documentation plus the training and test data
    readings = [(event, label) for seed in (config.STREAM_SEED, 101, 202) for tick in generate_ticks(seed, 300)
                for event, label in zip(tick["vital_events"], tick["vital_labels"])]
    for seed, rows in ((config.TRAIN_SEED, config.TRAIN_ROWS), (config.TEST_SEED, config.TEST_ROWS)):
        events, labels = generate_data.generate_vitals_dataset(seed, rows, "T")
        readings += zip(events.to_dict("records"), labels.to_dict("records"))

    valid = [event for event, label in readings if not label["is_corrupt"]]
    assert len(valid) > 15000
    for event in valid:
        assert event["systolic_bp"] - event["diastolic_bp"] >= config.MIN_PULSE_PRESSURE
        assert passes_pipeline_validation(event)


def test_patient_baselines_keep_a_realistic_pressure_gap():
    for seed in (config.TRAIN_SEED, config.TEST_SEED, config.STREAM_SEED, 101, 202):
        for patient in generate_data.StreamSimulator(seed).patients:
            gap = patient["base_systolic_bp"] - patient["base_diastolic_bp"]
            assert gap >= config.MIN_BASELINE_PULSE_PRESSURE - 1e-9


def test_intentional_corrupt_readings_are_still_generated_and_fail_validation():
    corrupt_types = set()
    for tick in generate_ticks(seed=config.STREAM_SEED, num_ticks=400):
        for event, label in zip(tick["vital_events"], tick["vital_labels"]):
            if label["is_corrupt"]:
                corrupt_types.add(label["corruption_type"])
                assert not passes_pipeline_validation(event)
    # all four deliberate corruption types, including diastolic above systolic, still occur
    assert corrupt_types == set(generate_data.CORRUPTION_TYPES)


def test_stream_seed_101_repeats_and_differs_from_seed_202():
    assert generate_ticks(seed=101, num_ticks=30) == generate_ticks(seed=101, num_ticks=30)
    assert generate_ticks(seed=202, num_ticks=30) != generate_ticks(seed=101, num_ticks=30)


def test_stream_command_uses_default_seed_44_or_the_given_seed(monkeypatch, capsys):
    used = []
    monkeypatch.setattr(generate_data, "run_stream", lambda *args, seed, **kwargs: used.append(seed) or 0)
    for argv in (["--mode", "stream", "--ticks", "1", "--seconds-per-tick", "0"],
                 ["--mode", "stream", "--ticks", "1", "--seconds-per-tick", "0", "--seed", "101"]):
        monkeypatch.setattr("sys.argv", ["generate_data.py", *argv])
        generate_data.main()
    assert config.STREAM_SEED == 44 and used == [44, 101]
    output = capsys.readouterr().out
    assert "Stream seed: 44" in output and "Stream seed: 101" in output


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
