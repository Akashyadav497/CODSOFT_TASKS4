# 📊 CodSoft Data Science Internship - Task 4: Sales Prediction Using Python

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B.svg)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/Tests-7%20Passed-brightgreen.svg)](https://pytest.org/)

---

## 📌 1. Project Title & Problem Statement

### **Sales Prediction using Machine Learning & Media Mix Modeling**
Sales prediction means forecasting the volume of a product that customers will purchase based on advertising allocations across multiple media platforms. In contemporary marketing, businesses operate under constrained capital and must determine how to allocate advertising budgets across promotional channels (such as **Television**, **Radio**, and **Newspapers**) to maximize return on investment (ROI).

The objective of this project is to build an end-to-end, production-ready, reproducible Machine Learning pipeline in Python that analyzes marketing spend data, interprets multi-channel interactions, predicts sales with high accuracy, and delivers actionable, non-causal decision support for marketing executives.

---

## 📁 2. Dataset Description & Audit Findings

The project uses the official advertising marketing dataset (`advertising.csv`) located directly in the project folder. Original files are treated strictly as read-only.

- **Total Samples**: 200 campaign records
- **Total Columns**: 4
- **Missing Values**: 0 across all columns (verified via automated audit)
- **Duplicate Records**: 0
- **Negative Values**: None detected
- **Time/Date Dimension**: **No date or timestamp column exists** in the dataset. Observations represent cross-sectional campaign markets rather than a continuous time series.

### Column Summary
| Column | Type | Role | Min | Median | Max | Mean | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TV** | `float64` | Predictor | 0.70 | 149.75 | 296.40 | 147.04 | Advertising investment in TV ($k) |
| **Radio** | `float64` | Predictor | 0.00 | 22.90 | 49.60 | 23.26 | Advertising investment in Radio ($k) |
| **Newspaper** | `float64` | Predictor | 0.30 | 25.75 | 114.00 | 30.55 | Advertising investment in Newspaper ($k) |
| **Sales** | `float64` | Target | 1.60 | 16.00 | 27.00 | 15.13 | Product sales volume ($k units) |

### Data Rule Case Applied: **Case C**
Because a single data file is present without sequential timestamps:
- We create an **80% Training partition (160 samples)** and a **20% Held-Out Testing partition (40 samples)** using `random_state=42`.
- Preprocessing transformers and scalers are fitted **strictly on the training partition** and then applied to transform test and inference inputs, guaranteeing zero data leakage.

---

## ⚙️ 3. Environment Setup & Execution Commands

### Prerequisites
- Python 3.10 to 3.13
- Git

### Step-by-Step Setup
```bash
# 1. Clone repository
git clone https://github.com/<your-username>/sales_prediction.git
cd sales_prediction

# 2. (Optional) Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Reproducing All Pipeline Steps
You can run individual commands directly or use the provided Windows batch script / Makefile:

```bash
# 1. Run Exploratory Data Analysis (generates figures in outputs/figures/)
python -m src.eda

# 2. Train baseline and ML models with 5-fold cross-validation
python -m src.train

# 3. Evaluate models on the held-out test split and create diagnostic plots
python -m src.evaluate

# 4. Make an interactive CLI prediction
python -m src.predict --tv 150 --radio 30 --newspaper 20

# 5. Run automated pytest test suite
pytest tests/ -v

# 6. Launch interactive Streamlit web dashboard
streamlit run app.py
```

*Or on Windows with `run.bat`:*
```cmd
run.bat test
run.bat eda
run.bat train
run.bat evaluate
run.bat app
```

---

## 🔬 4. Methodology & Pipeline Architecture

```
                      Raw Data (advertising.csv)
                                 │
                   Data Cleaning & Range Auditing
                                 │
                 80 / 20 Split (random_state=42)
                  ┌──────────────┴──────────────┐
                  ▼                             ▼
             Train Set                      Test Set
           (160 samples)                  (40 samples)
                  │                             │
    Feature Engineering Pipeline                │
   (Total_Spend, Shares, TV*Radio)              │
                  │                             │
       5-Fold Cross-Validation                  │
   (Linear, Ridge, Lasso, RF, GB, XGB)          │
                  │                             │
    Hyperparameter Grid Search (GB)             │
                  │                             │
       Final Fitted Pipeline ─────────► Held-Out Evaluation
                                                │
                                    Metrics (R², RMSE, MAE)
                                                │
                                    Diagnostic Plots & Reports
```

1. **Data Cleaning & Auditing**: Check for null values, duplicates, and negative spends. Verify Tukey IQR bounds (identifying 2 high newspaper spend points that are preserved as valid campaign allocations).
2. **Feature Engineering (`AdvertisingFeatureEngineer`)**:
   - `Total_Spend = TV + Radio + Newspaper`
   - `TV_Share`, `Radio_Share`, `Newspaper_Share`
   - `TV_x_Radio`: Non-linear marketing synergy interaction term.
   - `Radio_to_TV_Ratio`: Media balance proportion.
3. **Model Selection & Tuning**:
   - Baseline: `DummyRegressor(strategy='mean')`
   - Linear Models: Standard OLS, Ridge ($L_2$), Lasso ($L_1$)
   - Tree Ensembles: Random Forest, Gradient Boosting, XGBoost
   - 5-fold Cross-Validation with `random_state=42`
   - Hyperparameter tuning via `GridSearchCV` on learning rate, tree depth, and subsample ratio.
4. **Leakage-Free Persistence**: Whole preprocessing + estimator pipeline serialized to `models/best_sales_model.joblib`.

---

## 📈 5. Real Model Benchmark Results

> **All numbers below originate directly from executing the training and evaluation code on the real dataset split (`random_state=42`). No scores are fabricated.**

### Held-Out Test Set Performance (`outputs/metrics.csv`)
| Model | Test R² | Test RMSE | Test MAE | Train R² | Train RMSE | 5-Fold CV R² | 5-Fold CV RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.9578** | **1.1413** | **0.8401** | 0.9961 | 0.3220 | 0.9253 ± 0.0492 | 1.2784 ± 0.3540 |
| **Gradient Boosting** | **0.9560** | **1.1659** | **0.8375** | 0.9977 | 0.2505 | 0.9283 ± 0.0496 | 1.2415 ± 0.3477 |
| **Random Forest** | **0.9512** | **1.2283** | **0.8766** | 0.9886 | 0.5521 | 0.9357 ± 0.0252 | 1.2357 ± 0.2520 |
| **Linear Regression** | **0.9465** | **1.2854** | **0.9403** | 0.9354 | 1.3163 | 0.9076 ± 0.0338 | 1.5015 ± 0.2310 |
| **Ridge Regression** | **0.9461** | **1.2905** | **0.9545** | 0.9339 | 1.3305 | 0.9001 ± 0.0431 | 1.5579 ± 0.2820 |
| **Tuned Gradient Boosting**| **0.9444** | **1.3103** | **0.9104** | 0.9990 | 0.1672 | 0.9326 ± 0.0463 | 1.2204 ± 0.3297 |
| **Lasso Regression** | **0.9406** | **1.3549** | **1.0438** | 0.9313 | 1.3570 | 0.9151 ± 0.0188 | 1.4489 ± 0.2114 |
| **Baseline (Mean)** | **-0.0324** | **5.6482** | **4.9315** | 0.0000 | 5.1768 | -0.0712 ± 0.0683| 5.2061 ± 0.6978 |

### Key Evaluation Findings:
1. **Outperformance of Baseline**: The baseline mean model yields an $R^2$ of $-0.032$ and an RMSE of $5.65k$. Machine learning models improve accuracy dramatically, reducing error by over **79%**.
2. **Top Model**: Gradient tree boosting methods (**XGBoost** and **Gradient Boosting**) lead overall performance, achieving a **Test $R^2$ of ~0.958** and **Test RMSE of 1.14k units**.
3. **Overfitting Analysis**: While tree models attain high training $R^2$ (~0.99), the held-out test $R^2$ of 0.958 confirms that the models generalize exceptionally well to unseen marketing regimes.

---

## 📐 6. Evaluation Metrics Explained in Plain English

### 1. R-Squared ($R^2$ - Coefficient of Determination)
- **Concept**: Measures the proportion of variance in sales explained by the advertising features, relative to a naive model that always predicts the historical mean sales.
- **Formula**:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  - $y_i$: Actual sales
  - $\hat{y}_i$: Model prediction
  - $\bar{y}$: Mean of actual sales
- **Solved Example**:
  Suppose total variance around mean $\sum(y_i - \bar{y})^2 = 100$, and model squared error sum $\sum(y_i - \hat{y}_i)^2 = 5$.
  $$R^2 = 1 - \frac{5}{100} = 0.95 \quad (95\%\text{ of variance explained})$$

### 2. Mean Absolute Error (MAE)
- **Concept**: The average dollar/unit magnitude of prediction errors, giving equal weight to all mistakes.
- **Formula**:
  $$\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|$$
- **Solved Example**:
  If actual sales are $[10, 20]$ and predictions are $[12, 19]$:
  $$\text{MAE} = \frac{|10-12| + |20-19|}{2} = \frac{2 + 1}{2} = 1.5$$

### 3. Root Mean Squared Error (RMSE)
- **Concept**: Measures standard deviation of residuals. Because errors are squared before averaging, large errors are penalized much more heavily than small ones.
- **Formula**:
  $$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
- **Solved Example**:
  Using the same errors:
  $$\text{RMSE} = \sqrt{\frac{(10-12)^2 + (20-19)^2}{2}} = \sqrt{\frac{4 + 1}{2}} = \sqrt{2.5} \approx 1.581$$

---

## 💼 7. Business Insights: What This Means for Advertising Decisions

> **Important Caution**: All observations describe statistical associations observed in historical sample campaigns, **not guarantees of causality**.

1. **TV is the Strongest Sales Volume Driver**:
   - TV spend exhibits a strong Pearson correlation ($r = 0.901$) with sales volume.
   - Permutation importance confirms that shuffling TV values causes the single steepest decline in model predictive accuracy.
2. **Radio Provides Critical Synergistic Lift**:
   - While Radio accounts for a smaller total spend footprint, it exhibits substantial positive regression coefficients and high interaction importance (`TV_x_Radio`).
   - Historical data reveals that high TV spend paired with moderate-to-high Radio spend correlates with the highest sales returns.
3. **Newspaper Advertising Shows Negligible Direct Contribution**:
   - In multivariate modeling, Newspaper advertising yields a near-zero standardized coefficient ($~-0.01$ to $0.02$) and very low feature importance.
   - When TV and Radio spend are controlled for, variations in Newspaper spend do not meaningfully correlate with sales changes.

### Limitations & Decision Constraints
- **Sample Size**: The dataset contains 200 records. While sufficient for stable regression, findings should be continuously monitored as new campaign data arrives.
- **Missing Covariates**: Crucial commercial variables such as pricing changes, discount promotions, seasonality, competitor spending, and macroeconomic conditions were not captured in this dataset.
- **Extrapolation Boundaries**: The models are reliable within historical spend boundaries (TV: \$0.7k-\$296.4k, Radio: \$0.0k-\$49.6k, Newspaper: \$0.3k-\$114.0k). Predictions made far outside these intervals enter extrapolation regions with elevated uncertainty.

---

## 🖥️ 8. Interactive Streamlit Web Application

The project includes an interactive web dashboard (`app.py`) built with Streamlit and Plotly:
- **📊 Dataset Explorer**: Interactive sliders for filtering campaigns by spend range, dynamic scatter charts with OLS trend lines, budget share pie charts, and a CSV data export button.
- **🤖 Model Comparison**: Interactive benchmark bar charts comparing test RMSE and $R^2$ across all candidate algorithms, with an exportable metrics table.
- **🔮 Prediction Tool**: User input forms for TV, Radio, and Newspaper expenditures with automatic bounds validation and instant sales forecasting.
- **💡 What-If Simulator**: Allows marketers to isolate one advertising channel and sweep its budget across varying levels while holding other channels fixed to visualize model sensitivity curves.

To launch the dashboard:
```bash
streamlit run app.py
```

---

## 🧪 9. Automated Testing (`pytest`)

The project contains a test suite (`tests/test_pipeline.py`) covering data integrity, preprocessing, model inference, and input validation:
```bash
pytest tests/ -v
```
All **7 unit & integration tests pass**:
- `test_data_loader_and_splitting`: Verifies data integrity and leak-free splitting.
- `test_clean_data_handles_anomalies`: Validates duplicate removal and range clipping.
- `test_feature_engineering_pipeline`: Tests custom scikit-learn transformer feature creation.
- `test_predict_validation_rejects_invalid_inputs`: Ensures invalid inputs (negative values, strings, empty values) are rejected with clear user-friendly errors.
- `test_predict_out_of_range_warning`: Verifies domain warnings for out-of-distribution inputs.
- `test_saved_model_loads_and_infers`: Validates joblib model loading and numeric output generation.
- `test_end_to_end_reproducibility`: Guarantees deterministic outputs with fixed random seeds.

---

## 🗂️ 10. Project Directory Structure

```
Sales Prediction/
├── .gitignore                      # Git ignore file
├── Makefile                        # CLI shortcut automation commands
├── README.md                       # Comprehensive project documentation
├── advertising.csv                 # Original source dataset (read-only)
├── app.py                          # Multi-page interactive Streamlit web dashboard
├── requirements.txt                # Pinned production dependencies
├── run.bat                         # Windows quick runner script
├── data/
│   ├── raw/
│   │   └── advertising.csv         # Mirrored read-only raw data
│   └── processed/
│       ├── train.csv               # 80% Training partition
│       └── test.csv                # 20% Held-out testing partition
├── models/
│   ├── best_sales_model.joblib     # Serialized best ML pipeline
│   └── model_metrics.joblib        # Serialized evaluation registry
├── outputs/
│   ├── data_audit.md               # Data audit & EDA quality report
│   ├── metrics.csv                 # Model benchmark comparison table
│   ├── metrics.md                  # Markdown evaluation report
│   └── figures/                    # 14 Publication-quality diagnostic charts
│       ├── eda_sales_distribution.png
│       ├── eda_features_distribution.png
│       ├── eda_scatter_regression.png
│       ├── eda_correlation_heatmap.png
│       ├── eda_pairplot.png
│       ├── eda_missing_values.png
│       ├── eda_budget_shares.png
│       ├── eval_actual_vs_predicted.png
│       ├── eval_residuals_plot.png
│       ├── eval_residuals_distribution.png
│       ├── eval_model_comparison_rmse.png
│       ├── eval_feature_importance.png
│       ├── eval_linear_coefficients.png
│       └── eval_permutation_importance.png
├── src/
│   ├── __init__.py
│   ├── config.py                   # Central paths, constants & feature bounds
│   ├── data_loader.py              # Data ingestion and splitting
│   ├── preprocess.py               # Data cleaning & outlier auditing
│   ├── features.py                 # Pipeline feature engineering transformer
│   ├── eda.py                      # Exploratory data analysis chart generator
│   ├── train.py                    # Multi-model training, 5-fold CV & tuning
│   ├── evaluate.py                 # Diagnostic evaluation & metric extraction
│   └── predict.py                  # Functional prediction API & CLI tool
└── tests/
    └── test_pipeline.py            # Automated pytest test suite
```

---

## 👤 Author & Acknowledgements
- **Author**: Internship Candidate (Data Science Track)
- **Organization**: CodSoft Virtual Internship
- **Task**: Task 4 - Sales Prediction Using Python
- **Verification Statement**: All evaluation metrics, figures, and test logs included in this project were generated through direct execution on the provided data.
