"""Data loader module for loading and auditing the sales dataset."""

import logging
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_PATH,
    FALLBACK_DATA_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    LOG_FILE_PATH,
    LOGS_DIR,
)

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("data_loader")


def locate_data_file() -> Path:
    """Locate the advertising dataset in project folder."""
    if RAW_DATA_PATH.exists():
        return RAW_DATA_PATH
    if FALLBACK_DATA_PATH.exists():
        return FALLBACK_DATA_PATH
    raise FileNotFoundError("Could not locate advertising.csv in data/raw/ or root directory.")


def load_raw_data() -> pd.DataFrame:
    """Load the raw advertising dataset as a pandas DataFrame."""
    data_file = locate_data_file()
    logger.info("Loading raw dataset from %s", data_file)
    df = pd.read_csv(data_file)
    logger.info("Loaded dataset successfully with shape %s", df.shape)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training (80%) and testing (20%) partitions.
    Case C applies: single dataset file with random shuffle split.
    """
    logger.info(
        "Splitting dataset: test_size=%.2f, random_state=%d (Case C)",
        test_size,
        random_state,
    )
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, shuffle=True
    )
    logger.info(
        "Train set shape: %s, Test set shape: %s",
        train_df.shape,
        test_df.shape,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_features_and_target(
    df: pd.DataFrame, target_col: str = TARGET_COLUMN
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target variable."""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
