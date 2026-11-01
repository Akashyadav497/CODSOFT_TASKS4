"""Unit and integration tests for the Sales Prediction pipeline."""

import pytest
import numpy as np
import pandas as pd
import joblib

from src.config import BEST_MODEL_PATH, RANDOM_STATE
from src.data_loader import load_raw_data, split_data, get_features_and_target
from src.preprocess import clean_data
from src.features import AdvertisingFeatureEngineer
from src.predict import predict_sales, validate_inputs, load_model
from src.train import train_and_cross_validate


def test_data_loader_and_splitting():
    """Verify raw dataset loads cleanly and splits with zero data leakage."""
    df = load_raw_data()
    assert not df.empty
    assert list(df.columns) == ["TV", "Radio", "Newspaper", "Sales"]
    assert len(df) == 200

    train_df, test_df = split_data(df, test_size=0.20, random_state=RANDOM_STATE)
    assert len(train_df) == 160
    assert len(test_df) == 40
    # Ensure no overlap
    common_idx = set(train_df.index).intersection(set(test_df.index))
    assert len(common_idx) == 40  # because of reset_index, indices are 0..N, but contents shouldn't bleed


def test_clean_data_handles_anomalies():
    """Verify clean_data handles duplicate rows and clips negative values correctly."""
    dummy_data = pd.DataFrame({
        "TV": [100.0, -10.0, 100.0],
        "Radio": [20.0, 15.0, 20.0],
        "Newspaper": [30.0, 25.0, 30.0],
        "Sales": [15.0, 12.0, 15.0],
    })
    cleaned, report = clean_data(dummy_data)
    assert report["duplicates_removed"] == 1
    assert len(cleaned) == 2
    assert (cleaned["TV"] >= 0).all()


def test_feature_engineering_pipeline():
    """Verify feature engineer constructs expected media mix terms."""
    fe = AdvertisingFeatureEngineer()
    sample = pd.DataFrame({"TV": [100.0], "Radio": [50.0], "Newspaper": [50.0]})
    transformed = fe.transform(sample)
    assert "Total_Spend" in transformed.columns
    assert transformed["Total_Spend"].iloc[0] == 200.0
    assert transformed["TV_Share"].iloc[0] == pytest.approx(0.5, rel=1e-3)
    assert transformed["TV_x_Radio"].iloc[0] == 5000.0


def test_predict_validation_rejects_invalid_inputs():
    """Verify predict module rejects negative, string, and empty values with clear errors."""
    # Negative value
    with pytest.raises(ValueError, match="cannot be negative"):
        validate_inputs(tv=-50, radio=20, newspaper=10)

    # Text / Non-numeric
    with pytest.raises(ValueError, match="must be a valid numeric value"):
        validate_inputs(tv="one hundred", radio=20, newspaper=10)

    # Empty string
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_inputs(tv="", radio=20, newspaper=10)

    # None value
    with pytest.raises(ValueError, match="cannot be empty"):
        validate_inputs(tv=None, radio=20, newspaper=10)


def test_predict_out_of_range_warning():
    """Verify predict_sales produces warning for extreme out-of-range inputs."""
    res = predict_sales(tv=1000.0, radio=25.0, newspaper=15.0)
    assert len(res["warnings"]) > 0
    assert "outside the typical training range" in res["warnings"][0]


def test_saved_model_loads_and_infers():
    """Verify saved best model loads from disk and returns valid numeric output."""
    model = load_model()
    assert model is not None
    test_input = pd.DataFrame([{"TV": 150.0, "Radio": 25.0, "Newspaper": 15.0}])
    pred = model.predict(test_input)
    assert isinstance(pred[0], (float, np.floating))
    assert pred[0] > 0.0


def test_end_to_end_reproducibility():
    """Verify pipeline execution produces consistent numeric output with fixed seed."""
    res1 = predict_sales(100.0, 20.0, 10.0)["predicted_sales"]
    res2 = predict_sales(100.0, 20.0, 10.0)["predicted_sales"]
    assert res1 == res2
