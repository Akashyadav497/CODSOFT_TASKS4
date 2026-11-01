# Data Audit Report: Sales Prediction Dataset

## 1. Executive Summary & File Identification
- **Identified File**: `advertising.csv` (located at project root and mirrored in `data/raw/advertising.csv` as read-only source).
- **Dataset Assignment Case**: **Case C** (Single dataset file present).
  - Train/Test Split Strategy: 80% Training set (160 samples), 20% Held-Out Testing set (40 samples).
  - Split methodology: Stratified/Random shuffle split with `random_state=42` because no date or sequential timestamp column exists in the dataset.
- **Target Variable**: `Sales` (Continuous numeric representing product sales / units in thousands).
  - Reason for selection: `Sales` is the primary business performance metric and the explicitly named response variable to be predicted from marketing allocations.
- **Date Column Presence**: **No date column exists** in the dataset. Observations represent independent market observations across channels.

---

## 2. Dataset Structure and Data Types

- **Total Rows**: 200
- **Total Columns**: 4
- **Memory Usage**: ~6.4 KB

| Column Name | Data Type | Non-Null Count | Missing Count | Role | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TV** | `float64` | 200 | 0 (0.0%) | Feature (Predictor) | Advertising budget spent on TV (in thousands of dollars) |
| **Radio** | `float64` | 200 | 0 (0.0%) | Feature (Predictor) | Advertising budget spent on Radio (in thousands of dollars) |
| **Newspaper** | `float64` | 200 | 0 (0.0%) | Feature (Predictor) | Advertising budget spent on Newspaper (in thousands of dollars) |
| **Sales** | `float64` | 200 | 0 (0.0%) | Target (Response) | Product sales (in thousands of units/dollars) |

---

## 3. Data Integrity & Quality Checks

- **Duplicate Rows**: `0` duplicate rows detected.
- **Missing Values**: `0` missing values across all columns.
- **Negative Value Check**: No negative spending or negative sales observed:
  - TV range: `[0.70, 296.40]`
  - Radio range: `[0.00, 49.60]`
  - Newspaper range: `[0.30, 114.00]`
  - Sales range: `[1.60, 27.00]`

---

## 4. Summary Statistics

| Statistic | TV ($k) | Radio ($k) | Newspaper ($k) | Sales (Units $k) |
| :--- | :--- | :--- | :--- | :--- |
| **Count** | 200.00 | 200.00 | 200.00 | 200.00 |
| **Mean** | 147.0425 | 23.2640 | 30.5540 | 15.1305 |
| **Std Dev** | 85.8542 | 14.8468 | 21.7786 | 5.2839 |
| **Min** | 0.7000 | 0.0000 | 0.3000 | 1.6000 |
| **25% (Q1)** | 74.3750 | 9.9750 | 12.7500 | 11.0000 |
| **50% (Median)**| 149.7500 | 22.9000 | 25.7500 | 16.0000 |
| **75% (Q3)** | 218.8250 | 36.5250 | 45.1000 | 19.0500 |
| **Max** | 296.4000 | 49.6000 | 114.0000 | 27.0000 |

---

## 5. Outlier Detection (Tukey's 1.5 × IQR Method)

- **TV**: IQR = 144.45 | Bounds: `[-142.30, 435.50]` -> **0 outliers** detected.
- **Radio**: IQR = 26.55 | Bounds: `[-29.85, 76.35]` -> **0 outliers** detected.
- **Sales**: IQR = 8.05 | Bounds: `[-1.08, 31.12]` -> **0 outliers** detected.
- **Newspaper**: IQR = 32.35 | Bounds: `[-35.78, 93.62]` -> **2 outliers** detected:
  - Row 16: `TV = 67.8, Radio = 36.6, Newspaper = 114.0, Sales = 12.5`
  - Row 101: `TV = 296.4, Radio = 36.3, Newspaper = 100.9, Sales = 23.8`

### Outlier Handling Justification
These 2 points represent genuine high advertising investments in newspaper campaigns rather than measurement errors or data corruptions. Dropping them would discard real business scenario data. Furthermore, robust tree ensembles (Random Forest, Gradient Boosting) naturally handle extreme values without degradation, and regularized linear regression (Ridge/Lasso) moderates high leverage. Therefore, these values are **retained** in training to ensure the models generalize to high-spend campaigns.

---

## 6. First 5 and Last 5 Rows

### First 5 Rows
| Index | TV | Radio | Newspaper | Sales |
| :--- | :--- | :--- | :--- | :--- |
| 0 | 230.1 | 37.8 | 69.2 | 22.1 |
| 1 | 44.5 | 39.3 | 45.1 | 10.4 |
| 2 | 17.2 | 45.9 | 69.3 | 12.0 |
| 3 | 151.5 | 41.3 | 58.5 | 16.5 |
| 4 | 180.8 | 10.8 | 58.4 | 17.9 |

### Last 5 Rows
| Index | TV | Radio | Newspaper | Sales |
| :--- | :--- | :--- | :--- | :--- |
| 195 | 38.2 | 3.7 | 13.8 | 7.6 |
| 196 | 94.2 | 4.9 | 8.1 | 14.0 |
| 197 | 177.0 | 9.3 | 6.4 | 14.8 |
| 198 | 283.6 | 42.0 | 66.2 | 25.5 |
| 199 | 232.1 | 8.6 | 8.7 | 18.4 |

---

## 7. Modeling & Feature Engineering Roadmap
1. **Target**: `Sales`.
2. **Raw Features**: `TV`, `Radio`, `Newspaper`.
3. **Engineered Features**:
   - `Total_Spend`: `TV + Radio + Newspaper`
   - Spend share ratios: `TV_Share`, `Radio_Share`, `Newspaper_Share`
   - Synergistic interaction term: `TV_x_Radio` (well-documented in media mix modeling)
4. **Leakage Prevention**:
   - All transformations, standard scalers, and parameter estimation fit strictly on `X_train` and transform `X_test`.
