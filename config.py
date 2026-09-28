"""Project settings in one place.

All data in this project is SYNTHETIC and intended only for academic
demonstration. Paths are built relative to this file, so the project does not
depend on any personal folder location.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders and files
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Spark reads new event files from these folders.
VITALS_INCOMING_DIR = DATA_DIR / "incoming" / "vitals"
GEO_INCOMING_DIR = DATA_DIR / "incoming" / "geo"

# Ground-truth labels are kept here, separate from the event files.
# The Spark pipeline and the models never read this folder.
LABELS_DIR = DATA_DIR / "labels"

# Small CSV datasets used to train and test the models.
TRAINING_DIR = DATA_DIR / "training"

# Spark Structured Streaming checkpoints.
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# The single SQLite database (Spark writes, Streamlit reads).
DATABASE_PATH = DATA_DIR / "healthcare.db"

# ---------------------------------------------------------------------------
# Random seeds (different seeds keep training, test and stream data separate)
# ---------------------------------------------------------------------------
TRAIN_SEED = 42
TEST_SEED = 43
STREAM_SEED = 44

# ---------------------------------------------------------------------------
# Simulated world
# ---------------------------------------------------------------------------
# Region names are fictional. Coordinates are arbitrary positions chosen only
# so the map has a background; they do not describe any real place's health.
REGIONS = [
    {"region_id": "R1", "region_name": "North District", "latitude": 13.030, "longitude": 77.590},
    {"region_id": "R2", "region_name": "South District", "latitude": 12.910, "longitude": 77.590},
    {"region_id": "R3", "region_name": "East District", "latitude": 12.970, "longitude": 77.660},
    {"region_id": "R4", "region_name": "West District", "latitude": 12.970, "longitude": 77.520},
    {"region_id": "R5", "region_name": "Central District", "latitude": 12.970, "longitude": 77.590},
    {"region_id": "R6", "region_name": "Lake District", "latitude": 13.020, "longitude": 77.660},
]

SYMPTOM_CATEGORIES = ["respiratory", "gastrointestinal", "fever", "rash", "neurological"]

NUM_PATIENTS = 100

# Simulated time: every tick moves the simulated clock forward by this much.
SIMULATION_START_TIME = "2026-01-01T00:00:00+00:00"
SIMULATED_MINUTES_PER_TICK = 5

# Real (wall-clock) seconds between ticks in stream mode.
REAL_SECONDS_PER_TICK = 1.0

# Default length of a stream run, in real minutes. Kept small on purpose.
DEFAULT_STREAM_MINUTES = 2

# Patient vitals
VITALS_PER_TICK = 10          # vital-sign readings written per tick
ANOMALY_RATE = 0.03           # share of readings with abnormal vitals
CORRUPT_RATE = 0.01           # share of stream readings with invalid values (tests validation)

# Geographic distress events
BACKGROUND_EVENTS_PER_TICK = 3    # average number of normal (non-outbreak) events per tick
BACKGROUND_SPREAD_KM = 4.0        # how far normal events spread around a region centre
FIRST_OUTBREAK_TICK = 20          # tick at which the first outbreak starts
OUTBREAK_EVERY_TICKS = 60         # a new outbreak starts every 60 ticks (5 simulated hours)
OUTBREAK_DURATION_TICKS = 24      # each outbreak lasts 24 ticks (2 simulated hours)
OUTBREAK_EVENTS_PER_TICK = 3      # average extra events per tick inside an active outbreak
OUTBREAK_RADIUS_KM = 1.0          # outbreak events fall within this distance of its centre

# Training / test dataset sizes (train mode)
TRAIN_ROWS = 5000
TEST_ROWS = 2000
GEO_TEST_TICKS = 288          # held-out geographic test events: 288 ticks = 1 simulated day

# ---------------------------------------------------------------------------
# Detection models
# ---------------------------------------------------------------------------
# Isolation Forest (unsupervised: trained on vital signs only, never on labels)
VITAL_FEATURES = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "oxygen_saturation",
    "body_temperature",
    "respiratory_rate",
]
ISOLATION_FOREST_PATH = MODELS_DIR / "isolation_forest.joblib"
MODEL_RANDOM_SEED = TRAIN_SEED
ISOLATION_FOREST_TREES = 200
# Expected share of anomalies. This is a design assumption taken from the
# simulation setting above, not a value tuned on the held-out test labels.
ISOLATION_FOREST_CONTAMINATION = ANOMALY_RATE

# DBSCAN (geographic clusters of distress events)
DBSCAN_EPS_KM = 0.5           # events closer than 0.5 km are neighbours
DBSCAN_MIN_SAMPLES = 5        # a cluster needs at least 5 nearby events
DBSCAN_WINDOW_MINUTES = 60    # cluster the events from the last 60 simulated minutes

# Risk score (simple academic heuristic, NOT a probability or clinical score)
RISK_ANOMALY_RATE_FOR_MAX = 0.10    # 10% abnormal vital readings in a region = full anomaly part
RISK_CLUSTER_CASES_FOR_MAX = 20     # 20 clustered cases in a region = full cluster part
RISK_MEDIUM_THRESHOLD = 30          # score 30-59 = Medium
RISK_HIGH_THRESHOLD = 60            # score 60+ = High

# Evaluation output (git-ignored because it is inside data/)
RESULTS_DIR = DATA_DIR / "results"

# ---------------------------------------------------------------------------
# Spark (kept small so it does not use all CPU cores on a shared computer)
# ---------------------------------------------------------------------------
SPARK_MASTER = "local[2]"
SPARK_SHUFFLE_PARTITIONS = 2
SPARK_UI_PORT = 4040
SPARK_BIND_ADDRESS = "127.0.0.1"   # listen on this computer only

# Windows compatibility (no winutils.exe or hadoop.dll needed).
# setup_windows.ps1 downloads this small pure-Java JAR from Maven Central.
# Spark must receive it through --driver-class-path before PySpark starts.
JARS_DIR = PROJECT_ROOT / "jars"
BARE_LOCAL_FS_JAR = JARS_DIR / "hadoop-bare-naked-local-fs-0.1.0.jar"
SPARK_FILE_SYSTEM_CLASS = "com.globalmentor.apache.hadoop.fs.BareLocalFileSystem"
SPARK_CHECKPOINT_MANAGER_CLASS = (
    "org.apache.spark.sql.execution.streaming.checkpointing.FileSystemBasedCheckpointFileManager"
)

# ---------------------------------------------------------------------------
# Streamlit
# ---------------------------------------------------------------------------
STREAMLIT_PORT = 8501
