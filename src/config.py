"""Central configuration management for Sales Prediction project."""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "advertising.csv"
FALLBACK_DATA_PATH = BASE_DIR / "advertising.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_sales_model.joblib"
METRIC_REGISTRY_PATH = MODELS_DIR / "model_metrics.joblib"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_CSV_PATH = OUTPUTS_DIR / "metrics.csv"
METRICS_MD_PATH = OUTPUTS_DIR / "metrics.md"
DATA_AUDIT_PATH = OUTPUTS_DIR / "data_audit.md"
LOGS_DIR = OUTPUTS_DIR / "logs"
LOG_FILE_PATH = LOGS_DIR / "sales_prediction.log"

# Modeling Hyperparameters & Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET_COLUMN = "Sales"
RAW_FEATURE_COLUMNS = ["TV", "Radio", "Newspaper"]

# Feature bounds observed in dataset (for input validation and warning bounds)
FEATURE_RANGES = {
    "TV": {"min": 0.7, "max": 296.4, "mean": 147.04},
    "Radio": {"min": 0.0, "max": 49.6, "mean": 23.26},
    "Newspaper": {"min": 0.3, "max": 114.0, "mean": 30.55},
}
