"""Generate SYNTHETIC patient-vital events and geographic distress events.

All data produced here is simulated for academic demonstration only. It does
not describe real patients, real places or real outbreaks.

Modes
-----
train   Write small CSV datasets for training and testing the models:
            data/training/vitals_train.csv, data/training/vitals_test.csv
stream  Simulate a live feed. Every tick writes one JSON file of vital-sign
        events and one JSON file of geographic events:
            data/incoming/vitals/vitals_000001.json
            data/incoming/geo/geo_000001.json

Ground-truth labels (is_anomaly, outbreak_id, ...) are always written to
data/labels/ and never into the event files the detection pipeline reads.

Every event has two timestamps:
    event_time  simulated time of the reading (used for windows and charts)
    created_at  real wall-clock time the event file was written (stream mode
                only; used to measure processing latency)

Examples
--------
    python generate_data.py --mode train
    python generate_data.py --mode stream --minutes 2 --fresh
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config

KM_PER_DEGREE_LATITUDE = 111.32
EARTH_RADIUS_KM = 6371.0

# Each abnormal pattern overrides some vitals with values drawn from these ranges.
ANOMALY_PATTERNS = {
    "tachycardia_hypoxia": {"heart_rate": (125, 160), "oxygen_saturation": (82, 90), "respiratory_rate": (24, 32)},
    "fever": {"body_temperature": (38.8, 40.5), "heart_rate": (100, 125), "respiratory_rate": (20, 26)},
    "hypotension": {"systolic_bp": (75, 90), "diastolic_bp": (40, 55), "heart_rate": (105, 130)},
    "hypertensive_crisis": {"systolic_bp": (180, 210), "diastolic_bp": (115, 130)},
    "bradycardia": {"heart_rate": (35, 48)},
}

# Invalid records the Spark pipeline should later reject during validation.
CORRUPTION_TYPES = ["missing_heart_rate", "impossible_oxygen", "diastolic_above_systolic", "negative_respiratory_rate"]

BACKGROUND_SEVERITY_WEIGHTS = [0.50, 0.30, 0.15, 0.05]   # severity 1..4
OUTBREAK_SEVERITY_WEIGHTS = [0.20, 0.35, 0.30, 0.15]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def to_iso(moment: datetime) -> str:
    """Format a timezone-aware datetime as ISO 8601 with milliseconds."""
    return moment.isoformat(timespec="milliseconds")


def utc_now_iso() -> str:
    """Current real time in UTC, used for the created_at timestamp."""
    return to_iso(datetime.now(timezone.utc))


def simulated_time(tick: int) -> datetime:
    """Simulated time of a tick: start time + tick * minutes per tick."""
    start = datetime.fromisoformat(config.SIMULATION_START_TIME)
    return start + timedelta(minutes=tick * config.SIMULATED_MINUTES_PER_TICK)


def offset_position(latitude: float, longitude: float, north_km: float, east_km: float) -> tuple[float, float]:
    """Move a point by a distance in kilometres (good enough for a few km)."""
    new_latitude = latitude + north_km / KM_PER_DEGREE_LATITUDE
    new_longitude = longitude + east_km / (KM_PER_DEGREE_LATITUDE * math.cos(math.radians(latitude)))
    return round(new_latitude, 6), round(new_longitude, 6)


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle (haversine) distance between two points in kilometres."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def nearest_region(latitude: float, longitude: float) -> dict:
    """Return the region whose centre is closest to the point."""
    return min(config.REGIONS, key=lambda r: distance_km(latitude, longitude, r["latitude"], r["longitude"]))


# ---------------------------------------------------------------------------
# Patient vital signs
# ---------------------------------------------------------------------------
def create_patients(rng: np.random.Generator, num_patients: int) -> list[dict]:
    """Create synthetic patients with a home location and personal baseline vitals."""
    patients = []
    for number in range(1, num_patients + 1):
        home = config.REGIONS[int(rng.integers(len(config.REGIONS)))]
        north_km, east_km = rng.normal(0, config.BACKGROUND_SPREAD_KM, size=2)
        latitude, longitude = offset_position(home["latitude"], home["longitude"], north_km, east_km)
        patients.append({
            "patient_id": f"P-{number:04d}",
            "region_id": nearest_region(latitude, longitude)["region_id"],
            "latitude": latitude,
            "longitude": longitude,
            "base_heart_rate": rng.normal(75, 8),
            "base_systolic_bp": rng.normal(118, 10),
            "base_diastolic_bp": rng.normal(76, 7),
            "base_oxygen_saturation": rng.normal(97.5, 0.8),
            "base_body_temperature": rng.normal(36.8, 0.2),
            "base_respiratory_rate": rng.normal(16, 1.5),
        })
    return patients


def normal_vitals(rng: np.random.Generator, patient: dict) -> dict:
    """Normal readings: the patient's baseline plus small random variation."""
    return {
        "heart_rate": patient["base_heart_rate"] + rng.normal(0, 5),
        "systolic_bp": patient["base_systolic_bp"] + rng.normal(0, 6),
        "diastolic_bp": patient["base_diastolic_bp"] + rng.normal(0, 4),
        "oxygen_saturation": min(100.0, patient["base_oxygen_saturation"] + rng.normal(0, 0.7)),
        "body_temperature": patient["base_body_temperature"] + rng.normal(0, 0.2),
        "respiratory_rate": patient["base_respiratory_rate"] + rng.normal(0, 1.5),
    }


def round_vitals(vitals: dict) -> dict:
    """Round readings the way a monitor would report them."""
    return {
        "heart_rate": int(round(vitals["heart_rate"])),
        "systolic_bp": int(round(vitals["systolic_bp"])),
        "diastolic_bp": int(round(vitals["diastolic_bp"])),
        "oxygen_saturation": round(float(vitals["oxygen_saturation"]), 1),
        "body_temperature": round(float(vitals["body_temperature"]), 1),
        "respiratory_rate": int(round(vitals["respiratory_rate"])),
    }


def corrupt_vitals(rng: np.random.Generator, vitals: dict) -> str:
    """Make a reading invalid in one of a few simple ways. Returns the corruption type."""
    corruption = str(rng.choice(CORRUPTION_TYPES))
    if corruption == "missing_heart_rate":
        vitals["heart_rate"] = None
    elif corruption == "impossible_oxygen":
        vitals["oxygen_saturation"] = round(float(rng.uniform(101, 120)), 1)
    elif corruption == "diastolic_above_systolic":
        vitals["systolic_bp"], vitals["diastolic_bp"] = vitals["diastolic_bp"], vitals["systolic_bp"] + 20
    elif corruption == "negative_respiratory_rate":
        vitals["respiratory_rate"] = -abs(vitals["respiratory_rate"])
    return corruption


def make_vital_event(
    rng: np.random.Generator,
    patient: dict,
    event_id: str,
    event_time: datetime,
    anomaly_rate: float,
    corrupt_rate: float,
) -> tuple[dict, dict]:
    """Create one vital-sign event and its separate ground-truth label."""
    vitals = normal_vitals(rng, patient)

    anomaly_type = "none"
    if rng.random() < anomaly_rate:
        anomaly_type = str(rng.choice(list(ANOMALY_PATTERNS)))
        for field, (low, high) in ANOMALY_PATTERNS[anomaly_type].items():
            vitals[field] = rng.uniform(low, high)
    vitals = round_vitals(vitals)

    corruption_type = "none"
    if rng.random() < corrupt_rate:
        corruption_type = corrupt_vitals(rng, vitals)

    event = {
        "event_id": event_id,
        "patient_id": patient["patient_id"],
        "region_id": patient["region_id"],
        "event_time": to_iso(event_time),
        **vitals,
        "latitude": patient["latitude"],
        "longitude": patient["longitude"],
    }
    label = {
        "event_id": event_id,
        "is_anomaly": anomaly_type != "none",
        "anomaly_type": anomaly_type,
        "is_corrupt": corruption_type != "none",
        "corruption_type": corruption_type,
    }
    return event, label


def generate_vitals_dataset(seed: int, num_rows: int, id_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a vitals dataset (events, labels) for training or testing the models.

    Datasets contain no corrupt rows: the models are trained on data that has
    already passed validation.
    """
    rng = np.random.default_rng(seed)
    patients = create_patients(rng, config.NUM_PATIENTS)
    events, labels = [], []
    for row in range(num_rows):
        patient = patients[int(rng.integers(len(patients)))]
        event_time = simulated_time(row // config.VITALS_PER_TICK)
        event, label = make_vital_event(
            rng, patient, f"{id_prefix}-{row + 1:08d}", event_time, config.ANOMALY_RATE, corrupt_rate=0.0
        )
        events.append(event)
        labels.append(label)
    return pd.DataFrame(events), pd.DataFrame(labels)


# ---------------------------------------------------------------------------
# Geographic distress events and outbreaks
# ---------------------------------------------------------------------------
def outbreak_starts_at(tick: int) -> bool:
    """Outbreaks start on a fixed schedule so every demo run contains some."""
    return tick >= config.FIRST_OUTBREAK_TICK and (tick - config.FIRST_OUTBREAK_TICK) % config.OUTBREAK_EVERY_TICKS == 0


def create_outbreak(rng: np.random.Generator, number: int, tick: int) -> dict:
    """Create a synthetic outbreak near a random region centre."""
    region = config.REGIONS[int(rng.integers(len(config.REGIONS)))]
    north_km, east_km = rng.uniform(-2, 2, size=2)
    latitude, longitude = offset_position(region["latitude"], region["longitude"], north_km, east_km)
    end_tick = tick + config.OUTBREAK_DURATION_TICKS - 1
    return {
        "outbreak_id": f"OB-{number:03d}",
        "center_latitude": latitude,
        "center_longitude": longitude,
        "radius_km": config.OUTBREAK_RADIUS_KM,
        "symptom_category": str(rng.choice(config.SYMPTOM_CATEGORIES)),
        "start_tick": tick,
        "end_tick": end_tick,
        "start_time": to_iso(simulated_time(tick)),
        "end_time": to_iso(simulated_time(end_tick)),
    }


def make_geo_event(
    latitude: float, longitude: float, event_time: datetime, symptom: str, severity: int, case_count: int
) -> dict:
    """Create one geographic distress event (event_id is filled in later)."""
    region = nearest_region(latitude, longitude)
    return {
        "event_id": None,
        "event_time": to_iso(event_time),
        "region_id": region["region_id"],
        "region_name": region["region_name"],
        "latitude": latitude,
        "longitude": longitude,
        "symptom_category": symptom,
        "severity": severity,
        "case_count": case_count,
    }


def background_geo_event(rng: np.random.Generator, event_time: datetime) -> dict:
    """A normal, scattered distress event that is not part of any outbreak."""
    region = config.REGIONS[int(rng.integers(len(config.REGIONS)))]
    north_km, east_km = rng.normal(0, config.BACKGROUND_SPREAD_KM, size=2)
    latitude, longitude = offset_position(region["latitude"], region["longitude"], north_km, east_km)
    return make_geo_event(
        latitude,
        longitude,
        event_time,
        symptom=str(rng.choice(config.SYMPTOM_CATEGORIES)),
        severity=int(rng.choice([1, 2, 3, 4], p=BACKGROUND_SEVERITY_WEIGHTS)),
        case_count=1 + int(rng.poisson(0.5)),
    )


def outbreak_geo_event(rng: np.random.Generator, outbreak: dict, event_time: datetime) -> dict:
    """A distress event placed inside an outbreak's radius."""
    # sqrt() spreads points evenly over the circle instead of bunching them in the middle.
    distance = outbreak["radius_km"] * math.sqrt(rng.random())
    angle = rng.uniform(0, 2 * math.pi)
    latitude, longitude = offset_position(
        outbreak["center_latitude"], outbreak["center_longitude"], distance * math.cos(angle), distance * math.sin(angle)
    )
    return make_geo_event(
        latitude,
        longitude,
        event_time,
        symptom=outbreak["symptom_category"],
        severity=int(rng.choice([1, 2, 3, 4], p=OUTBREAK_SEVERITY_WEIGHTS)),
        case_count=1 + int(rng.poisson(1.5)),
    )


# ---------------------------------------------------------------------------
# Stream simulation
# ---------------------------------------------------------------------------
class StreamSimulator:
    """Produces synthetic events one tick at a time.

    For a given seed the events and labels are always the same (only the
    created_at timestamp, added when files are written, depends on real time).
    """

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.patients = create_patients(self.rng, config.NUM_PATIENTS)
        self.tick = 0
        self.next_vital_number = 1
        self.next_geo_number = 1
        self.outbreaks: list[dict] = []

    def next_tick(self) -> dict:
        """Generate all events and labels for the next tick."""
        event_time = simulated_time(self.tick)

        new_outbreak = None
        if outbreak_starts_at(self.tick):
            new_outbreak = create_outbreak(self.rng, len(self.outbreaks) + 1, self.tick)
            self.outbreaks.append(new_outbreak)

        vital_events, vital_labels = [], []
        for _ in range(config.VITALS_PER_TICK):
            patient = self.patients[int(self.rng.integers(len(self.patients)))]
            event, label = make_vital_event(
                self.rng,
                patient,
                f"V-{self.next_vital_number:08d}",
                event_time,
                config.ANOMALY_RATE,
                config.CORRUPT_RATE,
            )
            self.next_vital_number += 1
            vital_events.append(event)
            vital_labels.append(label)

        # Build geo events together with their outbreak_id, then shuffle so the
        # order of events in a file says nothing about which are outbreak events.
        geo_pairs = []
        for _ in range(int(self.rng.poisson(config.BACKGROUND_EVENTS_PER_TICK))):
            geo_pairs.append((background_geo_event(self.rng, event_time), None))
        for outbreak in self.outbreaks:
            if outbreak["start_tick"] <= self.tick <= outbreak["end_tick"]:
                for _ in range(int(self.rng.poisson(config.OUTBREAK_EVENTS_PER_TICK))):
                    geo_pairs.append((outbreak_geo_event(self.rng, outbreak, event_time), outbreak["outbreak_id"]))

        geo_events, geo_labels = [], []
        for index in self.rng.permutation(len(geo_pairs)):
            event, outbreak_id = geo_pairs[int(index)]
            event["event_id"] = f"G-{self.next_geo_number:08d}"
            self.next_geo_number += 1
            geo_events.append(event)
            geo_labels.append({
                "event_id": event["event_id"],
                "is_outbreak_event": outbreak_id is not None,
                "outbreak_id": outbreak_id,
            })

        self.tick += 1
        return {
            "vital_events": vital_events,
            "vital_labels": vital_labels,
            "geo_events": geo_events,
            "geo_labels": geo_labels,
            "new_outbreak": new_outbreak,
        }


# ---------------------------------------------------------------------------
# Writing files
# ---------------------------------------------------------------------------
def write_json_lines(path: Path, records: list[dict]) -> None:
    """Write one JSON object per line.

    The file is first written with a .tmp extension and then renamed, so Spark
    (which only reads *.json files) never sees a half-written file.
    """
    temporary_path = path.with_suffix(".tmp")
    with open(temporary_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    os.replace(temporary_path, path)


def append_json_lines(path: Path, records: list[dict]) -> None:
    """Append one JSON object per line (used for label files)."""
    with open(path, "a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


def clear_previous_stream(incoming_dirs: list[Path], labels_dir: Path) -> None:
    """Delete event and label files from an earlier stream run."""
    for folder in incoming_dirs:
        for pattern in ("*.json", "*.tmp"):
            for file in folder.glob(pattern):
                file.unlink()
    for name in ("vitals_labels.jsonl", "geo_labels.jsonl", "outbreaks.json"):
        (labels_dir / name).unlink(missing_ok=True)


def run_stream(
    num_ticks: int,
    seconds_per_tick: float = config.REAL_SECONDS_PER_TICK,
    seed: int = config.STREAM_SEED,
    fresh: bool = False,
    vitals_dir: Path = config.VITALS_INCOMING_DIR,
    geo_dir: Path = config.GEO_INCOMING_DIR,
    labels_dir: Path = config.LABELS_DIR,
) -> int:
    """Write num_ticks ticks of events, pausing seconds_per_tick between ticks.

    Returns the number of ticks written.
    """
    for folder in (vitals_dir, geo_dir, labels_dir):
        folder.mkdir(parents=True, exist_ok=True)

    if fresh:
        clear_previous_stream([vitals_dir, geo_dir], labels_dir)
    elif any(vitals_dir.glob("*.json")) or any(geo_dir.glob("*.json")):
        raise SystemExit(
            "Event files from an earlier run already exist in data/incoming/.\n"
            "Run again with --fresh to delete them and start a new simulation."
        )

    simulator = StreamSimulator(seed)
    ticks_written = 0
    try:
        for tick in range(1, num_ticks + 1):
            data = simulator.next_tick()

            # created_at is the real time the events are handed to the pipeline.
            created_at = utc_now_iso()
            for event in data["vital_events"] + data["geo_events"]:
                event["created_at"] = created_at

            write_json_lines(vitals_dir / f"vitals_{tick:06d}.json", data["vital_events"])
            if data["geo_events"]:
                write_json_lines(geo_dir / f"geo_{tick:06d}.json", data["geo_events"])

            append_json_lines(labels_dir / "vitals_labels.jsonl", data["vital_labels"])
            append_json_lines(labels_dir / "geo_labels.jsonl", data["geo_labels"])
            if data["new_outbreak"]:
                with open(labels_dir / "outbreaks.json", "w", encoding="utf-8") as file:
                    json.dump(simulator.outbreaks, file, indent=2)
                print(f"  tick {tick}: synthetic outbreak {data['new_outbreak']['outbreak_id']} started")

            ticks_written = tick
            if tick % 10 == 0 or tick == num_ticks:
                print(f"  wrote tick {tick}/{num_ticks} (simulated time {data['vital_events'][0]['event_time']})")
            if seconds_per_tick > 0 and tick < num_ticks:
                time.sleep(seconds_per_tick)
    except KeyboardInterrupt:
        print(f"\nStopped by user after {ticks_written} ticks.")
    return ticks_written


def run_train(train_dir: Path = config.TRAINING_DIR, labels_dir: Path = config.LABELS_DIR) -> None:
    """Write the training and test vitals datasets with separate label files."""
    train_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    for name, seed, rows, prefix in (
        ("train", config.TRAIN_SEED, config.TRAIN_ROWS, "VTR"),
        ("test", config.TEST_SEED, config.TEST_ROWS, "VTE"),
    ):
        events, labels = generate_vitals_dataset(seed, rows, prefix)
        events.to_csv(train_dir / f"vitals_{name}.csv", index=False)
        labels.to_csv(labels_dir / f"vitals_{name}_labels.csv", index=False)
        print(f"  {name}: {len(events)} rows, {int(labels['is_anomaly'].sum())} labelled anomalies (seed {seed})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SYNTHETIC healthcare events (academic demo only).")
    parser.add_argument("--mode", choices=["train", "stream"], required=True)
    parser.add_argument("--minutes", type=float, default=config.DEFAULT_STREAM_MINUTES,
                        help="stream mode: how many real minutes to run")
    parser.add_argument("--ticks", type=int, default=None,
                        help="stream mode: number of ticks to write (overrides --minutes)")
    parser.add_argument("--seconds-per-tick", type=float, default=config.REAL_SECONDS_PER_TICK,
                        help="stream mode: real seconds to wait between ticks")
    parser.add_argument("--fresh", action="store_true",
                        help="stream mode: delete event and label files from an earlier run first")
    args = parser.parse_args()

    print("Generating SYNTHETIC data for academic demonstration only.")
    if args.mode == "train":
        run_train()
    else:
        if args.ticks is not None:
            num_ticks = args.ticks
        else:
            num_ticks = max(1, int(args.minutes * 60 / max(args.seconds_per_tick, 0.001)))
        print(f"Streaming {num_ticks} ticks, {args.seconds_per_tick}s apart (Ctrl+C to stop).")
        run_stream(num_ticks, args.seconds_per_tick, fresh=args.fresh)
    print("Done.")


if __name__ == "__main__":
    main()
