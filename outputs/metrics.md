# Model Performance Evaluation & Comparison

Evaluated on 5-Fold Cross-Validation and Held-Out Test Set (20% Split, Case C, random_state=42):

| Model                   |    Test_R2 |   Test_RMSE |   Test_MAE |   Train_R2 |   Train_RMSE |   CV_R2_Score |   CV_RMSE |
|:------------------------|-----------:|------------:|-----------:|-----------:|-------------:|--------------:|----------:|
| XGBoost                 |  0.957847  |     1.1413  |   0.840137 |   0.996131 |     0.322017 |     0.92534   |   1.27844 |
| Gradient Boosting       |  0.956011  |     1.16589 |   0.837457 |   0.997659 |     0.250465 |     0.928313  |   1.24147 |
| Random Forest           |  0.95118   |     1.22825 |   0.876607 |   0.988627 |     0.552087 |     0.935723  |   1.23573 |
| Linear Regression       |  0.946531  |     1.28539 |   0.940264 |   0.935351 |     1.31626  |     0.907647  |   1.50145 |
| Ridge Regression        |  0.946108  |     1.29047 |   0.95454  |   0.933943 |     1.33052  |     0.900111  |   1.55789 |
| Tuned Gradient Boosting |  0.944442  |     1.31027 |   0.910371 |   0.998956 |     0.16723  |     0.932623  |   1.22036 |
| Lasso Regression        |  0.940595  |     1.35487 |   1.04381  |   0.931291 |     1.35697  |     0.915092  |   1.4489  |
| Baseline (Mean)         | -0.0324018 |     5.64822 |   4.93153  |   0        |     5.17681  |    -0.0711625 |   5.20614 |

### Metric Interpretations:
- **Test R²**: Proportion of sales variance explained by marketing spend (1.0 = perfect prediction).
- **Test RMSE**: Root Mean Squared Error (penalizes large outlier misses heavily).
- **Test MAE**: Mean Absolute Error (average magnitude of forecast errors in $k).
- **Train vs Test**: Evaluates overfitting; close Train and Test R² indicates robust generalization.
