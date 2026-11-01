"""Evaluation module computing test metrics, comparison tables, and diagnostic plots."""

import logging
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    BEST_MODEL_PATH,
    METRIC_REGISTRY_PATH,
    METRICS_CSV_PATH,
    METRICS_MD_PATH,
    FIGURES_DIR,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    RANDOM_STATE,
    LOG_FILE_PATH,
    LOGS_DIR,
)
from src.data_loader import get_features_and_target

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("evaluate")


def evaluate_models():
    """Evaluate all trained models on held-out test data and generate reports/plots."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load registry and test data
    registry = joblib.load(METRIC_REGISTRY_PATH)
    fitted_models = registry["fitted_models"]
    cv_results = registry["cv_results"]

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train, y_train = get_features_and_target(train_df)
    X_test, y_test = get_features_and_target(test_df)

    logger.info("Evaluating %d models on held-out test set (%d samples)...", len(fitted_models), len(test_df))

    records = []
    test_preds = {}

    for name, model in fitted_models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        test_preds[name] = y_test_pred

        # Test metrics
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        # Train metrics for overfitting check
        train_r2 = r2_score(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

        # Cross-validation summary
        cv_info = cv_results.get(name, {})
        cv_r2_str = f"{cv_info.get('cv_r2_mean', np.nan):.4f} +/- {cv_info.get('cv_r2_std', 0.0):.4f}"
        cv_rmse_str = f"{cv_info.get('cv_rmse_mean', np.nan):.4f} +/- {cv_info.get('cv_rmse_std', 0.0):.4f}"

        records.append({
            "Model": name,
            "CV_R2_Score": cv_info.get("cv_r2_mean", np.nan),
            "CV_R2_Std": cv_info.get("cv_r2_std", np.nan),
            "CV_RMSE": cv_info.get("cv_rmse_mean", np.nan),
            "CV_RMSE_Std": cv_info.get("cv_rmse_std", np.nan),
            "CV_MAE": cv_info.get("cv_mae_mean", np.nan),
            "Train_R2": train_r2,
            "Test_R2": test_r2,
            "Train_RMSE": train_rmse,
            "Test_RMSE": test_rmse,
            "Test_MAE": test_mae,
            "Overfit_Delta_R2": train_r2 - test_r2,
        })

    metrics_df = pd.DataFrame(records).sort_values(by="Test_RMSE", ascending=True)

    # Save to CSV
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    logger.info("Saved metrics comparison table to %s", METRICS_CSV_PATH)

    # Generate Markdown Table
    md_content = "# Model Performance Evaluation & Comparison\n\n"
    md_content += "Evaluated on 5-Fold Cross-Validation and Held-Out Test Set (20% Split, Case C, random_state=42):\n\n"
    md_content += metrics_df[[
        "Model", "Test_R2", "Test_RMSE", "Test_MAE", "Train_R2", "Train_RMSE", "CV_R2_Score", "CV_RMSE"
    ]].to_markdown(index=False)
    md_content += "\n\n### Metric Interpretations:\n"
    md_content += "- **Test R²**: Proportion of sales variance explained by marketing spend (1.0 = perfect prediction).\n"
    md_content += "- **Test RMSE**: Root Mean Squared Error (penalizes large outlier misses heavily).\n"
    md_content += "- **Test MAE**: Mean Absolute Error (average magnitude of forecast errors in $k).\n"
    md_content += "- **Train vs Test**: Evaluates overfitting; close Train and Test R² indicates robust generalization.\n"

    with open(METRICS_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info("Saved metrics markdown summary to %s", METRICS_MD_PATH)

    # Best model evaluation
    best_model_name = registry["best_model_name"]
    best_model = registry["best_model"]
    y_best_pred = test_preds[best_model_name]
    residuals = y_test - y_best_pred

    # 1. Actual vs Predicted Sales Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_best_pred, color="#1f77b4", alpha=0.8, edgecolors="k", s=60, label="Predictions")
    min_val = min(y_test.min(), y_best_pred.min()) - 1
    max_val = max(y_test.max(), y_best_pred.max()) + 1
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Perfect Fit (y = x)")
    plt.title(f"Figure 8: Actual vs. Predicted Sales ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Actual Sales ($k)", fontsize=11)
    plt.ylabel("Predicted Sales ($k)", fontsize=11)
    plt.legend(frameon=True)
    plt.savefig(FIGURES_DIR / "eval_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Residual Plot (Residuals vs Predicted)
    plt.figure(figsize=(8, 5))
    plt.scatter(y_best_pred, residuals, color="#2ca02c", alpha=0.8, edgecolors="k", s=55)
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title(f"Figure 9: Residual Plot ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Sales ($k)", fontsize=11)
    plt.ylabel("Residuals (Actual - Predicted) ($k)", fontsize=11)
    plt.savefig(FIGURES_DIR / "eval_residuals_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Residual Distribution (Histogram & KDE)
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, color="#9467bd", bins=12)
    plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero Bias")
    plt.title(f"Figure 10: Residuals Distribution ({best_model_name})", fontsize=13, fontweight="bold")
    plt.xlabel("Residual Error ($k)", fontsize=11)
    plt.ylabel("Frequency", fontsize=11)
    plt.legend()
    plt.savefig(FIGURES_DIR / "eval_residuals_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Model Comparison Bar Chart
    plt.figure(figsize=(10, 5))
    chart_df = metrics_df[metrics_df["Model"] != "Baseline (Mean)"]
    sns.barplot(data=chart_df, x="Model", y="Test_RMSE", palette="viridis")
    plt.title("Figure 11: Test RMSE Comparison Across Machine Learning Models", fontsize=13, fontweight="bold")
    plt.ylabel("Test RMSE ($k - Lower is Better)", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    for idx, row in enumerate(chart_df.itertuples()):
        plt.text(idx, row.Test_RMSE + 0.03, f"{row.Test_RMSE:.3f}", ha="center", fontweight="bold")
    plt.savefig(FIGURES_DIR / "eval_model_comparison_rmse.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Feature Importance & Coefficients
    logger.info("Computing Feature Importances for Business Decision Section...")
    feature_names = best_model.named_steps["features"].get_feature_names_out()
    reg = best_model.named_steps["regressor"]
    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
            by="Importance", ascending=False
        )

        plt.figure(figsize=(9, 5))
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="Blues_r")
        plt.title(f"Figure 12: Tree Feature Importances ({best_model_name})", fontsize=13, fontweight="bold")
        plt.xlabel("Gini / Gain Importance")
        plt.savefig(FIGURES_DIR / "eval_feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Linear Model Coefficients
    lin_model = fitted_models["Linear Regression"]
    lin_reg = lin_model.named_steps["regressor"]
    lin_coefs = lin_reg.coef_
    coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": lin_coefs}).sort_values(
        by="Coefficient", ascending=False
    )
    plt.figure(figsize=(9, 5))
    sns.barplot(data=coef_df, x="Coefficient", y="Feature", palette="coolwarm")
    plt.title("Figure 13: Standardized Coefficients (Linear Regression)", fontsize=13, fontweight="bold")
    plt.xlabel("Standardized Coefficient Magnitude")
    plt.savefig(FIGURES_DIR / "eval_linear_coefficients.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Permutation Importance
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE)
    perm_df = pd.DataFrame({"Feature": X_test.columns, "Mean_Importance": perm.importances_mean}).sort_values(
        by="Mean_Importance", ascending=False
    )
    plt.figure(figsize=(8, 4))
    sns.barplot(data=perm_df, x="Mean_Importance", y="Feature", palette="magma")
    plt.title("Figure 14: Permutation Feature Importance on Held-Out Test Data", fontsize=13, fontweight="bold")
    plt.xlabel("Decrease in R² Score upon Feature Shuffle")
    plt.savefig(FIGURES_DIR / "eval_permutation_importance.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("Evaluation completed successfully. All figures and tables generated.")
    return metrics_df


if __name__ == "__main__":
    evaluate_models()
