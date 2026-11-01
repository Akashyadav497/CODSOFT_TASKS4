"""Training pipeline for multiple regression models with CV and hyperparameter tuning."""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.config import (
    RANDOM_STATE,
    CV_FOLDS,
    BEST_MODEL_PATH,
    METRIC_REGISTRY_PATH,
    PROCESSED_DATA_DIR,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    MODELS_DIR,
    LOG_FILE_PATH,
    LOGS_DIR,
)
from src.data_loader import load_raw_data, split_data, get_features_and_target
from src.preprocess import clean_data
from src.features import AdvertisingFeatureEngineer

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("train")


def build_models() -> Dict[str, Pipeline]:
    """
    Construct pipelines for all evaluated models:
    - Baseline: DummyRegressor (predicts mean)
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest Regressor
    - Gradient Boosting Regressor
    - XGBoost Regressor
    """
    models = {
        "Baseline (Mean)": Pipeline([
            ("regressor", DummyRegressor(strategy="mean"))
        ]),
        "Linear Regression": Pipeline([
            ("features", AdvertisingFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]),
        "Ridge Regression": Pipeline([
            ("features", AdvertisingFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ]),
        "Lasso Regression": Pipeline([
            ("features", AdvertisingFeatureEngineer()),
            ("scaler", StandardScaler()),
            ("regressor", Lasso(alpha=0.1, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("features", AdvertisingFeatureEngineer()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("features", AdvertisingFeatureEngineer()),
            ("regressor", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("features", AdvertisingFeatureEngineer()),
            ("regressor", XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE)),
        ]),
    }
    return models


def tune_gradient_boosting(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Tune Gradient Boosting hyperparameters using 5-fold GridSearchCV."""
    logger.info("Running GridSearchCV hyperparameter tuning on Gradient Boosting...")
    base_pipeline = Pipeline([
        ("features", AdvertisingFeatureEngineer()),
        ("regressor", GradientBoostingRegressor(random_state=RANDOM_STATE)),
    ])
    param_grid = {
        "regressor__n_estimators": [50, 100, 150],
        "regressor__learning_rate": [0.03, 0.05, 0.1],
        "regressor__max_depth": [2, 3, 4],
        "regressor__subsample": [0.8, 1.0],
    }
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    logger.info("Best Gradient Boosting params: %s (Best CV RMSE: %.4f)",
                grid_search.best_params_, -grid_search.best_score_)
    return grid_search.best_estimator_


def train_and_cross_validate() -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Train all models, perform 5-fold CV, tune top performer, and persist best model."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load and clean raw dataset
    raw_df = load_raw_data()
    cleaned_df, clean_report = clean_data(raw_df)

    # 2. Split train/test (80/20 Case C)
    train_df, test_df = split_data(cleaned_df)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    logger.info("Saved processed train and test splits to %s", PROCESSED_DATA_DIR)

    X_train, y_train = get_features_and_target(train_df)
    X_test, y_test = get_features_and_target(test_df)

    # 3. Cross-Validation across candidate models
    models = build_models()
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            },
            return_train_score=True,
        )
        cv_results[name] = {
            "cv_r2_mean": float(np.mean(scores["test_r2"])),
            "cv_r2_std": float(np.std(scores["test_r2"])),
            "cv_mae_mean": float(-np.mean(scores["test_mae"])),
            "cv_mae_std": float(np.std(scores["test_mae"])),
            "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
            "cv_rmse_std": float(np.std(scores["test_rmse"])),
            "train_r2_mean": float(np.mean(scores["train_r2"])),
        }
        logger.info(
            "Model [%s] -> CV R2: %.4f (+/- %.4f), RMSE: %.4f, MAE: %.4f",
            name,
            cv_results[name]["cv_r2_mean"],
            cv_results[name]["cv_r2_std"],
            cv_results[name]["cv_rmse_mean"],
            cv_results[name]["cv_mae_mean"],
        )

    # 4. Tune best model (Gradient Boosting)
    tuned_gb = tune_gradient_boosting(X_train, y_train)
    scores = cross_validate(
        tuned_gb,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        },
        return_train_score=True,
    )
    cv_results["Tuned Gradient Boosting"] = {
        "cv_r2_mean": float(np.mean(scores["test_r2"])),
        "cv_r2_std": float(np.std(scores["test_r2"])),
        "cv_mae_mean": float(-np.mean(scores["test_mae"])),
        "cv_mae_std": float(np.std(scores["test_mae"])),
        "cv_rmse_mean": float(-np.mean(scores["test_rmse"])),
        "cv_rmse_std": float(np.std(scores["test_rmse"])),
        "train_r2_mean": float(np.mean(scores["train_r2"])),
    }
    models["Tuned Gradient Boosting"] = tuned_gb

    # 5. Fit all models on full X_train
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model

    # Select best model based on CV RMSE
    best_model_name = min(
        [m for m in cv_results.keys() if m != "Baseline (Mean)"],
        key=lambda k: cv_results[k]["cv_rmse_mean"],
    )
    best_model = fitted_models[best_model_name]
    logger.info("Best model selected: %s", best_model_name)

    # Save best model pipeline and all fitted models for evaluation
    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(
        {
            "best_model_name": best_model_name,
            "best_model": best_model,
            "fitted_models": fitted_models,
            "cv_results": cv_results,
            "clean_report": clean_report,
        },
        METRIC_REGISTRY_PATH,
    )
    logger.info("Saved best model to %s and registry to %s", BEST_MODEL_PATH, METRIC_REGISTRY_PATH)

    return cv_results, train_df, test_df


if __name__ == "__main__":
    train_and_cross_validate()
