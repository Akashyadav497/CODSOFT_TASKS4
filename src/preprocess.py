"""Data cleaning, validation, and outlier handling module."""

import logging
from typing import Tuple
import pandas as pd
import numpy as np

from src.config import TARGET_COLUMN, RAW_FEATURE_COLUMNS

logger = logging.getLogger("preprocess")


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Clean the sales dataset according to mandatory specifications:
    - Drop duplicate rows and report count.
    - Check for missing values and handle appropriately (drop if target missing).
    - Validate ranges (no negative values).
    - Check IQR outlier boundaries.
    """
    initial_len = len(df)
    report = {}

    # Check duplicates
    duplicate_count = df.duplicated().sum()
    report["duplicates_removed"] = int(duplicate_count)
    if duplicate_count > 0:
        df = df.drop_duplicates().copy()
        logger.info("Removed %d duplicate rows.", duplicate_count)
    else:
        logger.info("No duplicate rows found.")

    # Check missing values
    missing_counts = df.isnull().sum().to_dict()
    report["missing_values"] = {k: int(v) for k, v in missing_counts.items()}
    if df[TARGET_COLUMN].isnull().any():
        null_target_count = df[TARGET_COLUMN].isnull().sum()
        df = df.dropna(subset=[TARGET_COLUMN]).copy()
        logger.warning("Dropped %d rows with missing target '%s'.", null_target_count, TARGET_COLUMN)
        report["null_targets_dropped"] = int(null_target_count)

    # Validate non-negative numbers
    for col in RAW_FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            negative_mask = df[col] < 0
            neg_count = negative_mask.sum()
            if neg_count > 0:
                logger.warning("Found %d negative values in '%s'. Clipping to 0.0.", neg_count, col)
                df[col] = df[col].clip(lower=0.0)
            report[f"{col}_negatives_clipped"] = int(neg_count)

    # IQR outlier audit
    outliers_detected = {}
    for col in RAW_FEATURE_COLUMNS + [TARGET_COLUMN]:
        if col in df.columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outlier_rows = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outliers_detected[col] = {
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "count": int(len(outlier_rows)),
            }
    report["outliers_audit"] = outliers_detected

    logger.info("Data cleaning completed. Rows retained: %d/%d", len(df), initial_len)
    return df, report
