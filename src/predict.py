"""Prediction interface module providing functional API and CLI for sales forecasting."""

import argparse
import sys
import logging
from typing import Dict, Union, Tuple, Optional
import joblib
import pandas as pd
import numpy as np

from src.config import BEST_MODEL_PATH, FEATURE_RANGES, RAW_FEATURE_COLUMNS

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("predict")

_CACHED_MODEL = None


def load_model():
    """Load the trained machine learning pipeline from disk."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at '{BEST_MODEL_PATH}'. "
                "Please run training first: python -m src.train"
            )
        _CACHED_MODEL = joblib.load(BEST_MODEL_PATH)
    return _CACHED_MODEL


def validate_inputs(
    tv: Union[int, float, str],
    radio: Union[int, float, str],
    newspaper: Union[int, float, str],
) -> Tuple[Dict[str, float], list]:
    """
    Validate input spending values:
    - Check numeric conversion
    - Ensure not empty
    - Ensure non-negative
    - Warn if input is outside ranges observed during training
    """
    inputs = {"TV": tv, "Radio": radio, "Newspaper": newspaper}
    clean_vals = {}
    warnings = []

    for name, val in inputs.items():
        # Check empty or None
        if val is None or (isinstance(val, str) and val.strip() == ""):
            raise ValueError(f"Input for '{name}' advertising spend cannot be empty.")

        # Check numeric
        try:
            numeric_val = float(val)
        except (ValueError, TypeError):
            raise ValueError(f"Input for '{name}' must be a valid numeric value, received: '{val}'.")

        # Check negative
        if numeric_val < 0.0:
            raise ValueError(
                f"Advertising expenditure for '{name}' cannot be negative ({numeric_val})."
            )

        # Range check warning
        expected_min = FEATURE_RANGES[name]["min"]
        expected_max = FEATURE_RANGES[name]["max"]
        if numeric_val < expected_min or numeric_val > expected_max * 1.5:
            warnings.append(
                f"Notice: '{name}' spend ({numeric_val}) is outside the typical training range "
                f"[{expected_min}, {expected_max}]. Model predictions in extrapolation regions have higher uncertainty."
            )

        clean_vals[name] = numeric_val

    return clean_vals, warnings


def predict_sales(
    tv: Union[int, float, str],
    radio: Union[int, float, str],
    newspaper: Union[int, float, str],
) -> Dict[str, Union[float, list, Dict[str, float]]]:
    """
    Functional prediction API.
    Returns dictionary with predicted sales, cleaned input, and any range warnings.
    """
    clean_vals, warnings = validate_inputs(tv, radio, newspaper)
    model = load_model()

    input_df = pd.DataFrame([clean_vals])
    prediction = model.predict(input_df)[0]

    return {
        "predicted_sales": float(round(prediction, 3)),
        "inputs": clean_vals,
        "warnings": warnings,
    }


def main():
    """Command Line Interface for predicting sales from advertising spend."""
    parser = argparse.ArgumentParser(
        description="Predict product sales ($k) given TV, Radio, and Newspaper advertising spend."
    )
    parser.add_argument("--tv", type=str, required=True, help="TV advertising budget ($k)")
    parser.add_argument("--radio", type=str, required=True, help="Radio advertising budget ($k)")
    parser.add_argument("--newspaper", type=str, required=True, help="Newspaper advertising budget ($k)")

    args = parser.parse_args()

    try:
        result = predict_sales(tv=args.tv, radio=args.radio, newspaper=args.newspaper)
        print("\n================ SALES PREDICTION RESULT ================")
        print(f"  Inputs:")
        print(f"    - TV Spend:        ${result['inputs']['TV']:,.2f}k")
        print(f"    - Radio Spend:     ${result['inputs']['Radio']:,.2f}k")
        print(f"    - Newspaper Spend: ${result['inputs']['Newspaper']:,.2f}k")
        print("  -------------------------------------------------------")
        print(f"  Predicted Sales:     {result['predicted_sales']:,.2f}k units ($k)")
        print("=========================================================\n")
        if result["warnings"]:
            for w in result["warnings"]:
                print(f"[!] {w}")
    except ValueError as e:
        print(f"\n[Validation Error]: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n[Execution Error]: An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
