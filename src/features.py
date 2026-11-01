"""Feature engineering transformer compliant with scikit-learn Pipeline standards."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AdvertisingFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer for feature engineering in Media Mix / Advertising data:
    1. Total_Spend = TV + Radio + Newspaper
    2. TV_Share = TV / (Total_Spend + epsilon)
    3. Radio_Share = Radio / (Total_Spend + epsilon)
    4. Newspaper_Share = Newspaper / (Total_Spend + epsilon)
    5. TV_x_Radio = TV * Radio (Synergy interaction term)
    6. Radio_to_TV_Ratio = Radio / (TV + epsilon)
    """

    def __init__(self, include_interactions: bool = True, include_shares: bool = True):
        self.include_interactions = include_interactions
        self.include_shares = include_shares
        self.feature_names_out_ = []

    def fit(self, X, y=None):
        # Stateless transformer, nothing to learn from training set
        return self

    def transform(self, X):
        # Accept either pandas DataFrame or numpy ndarray
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=["TV", "Radio", "Newspaper"])

        tv = df["TV"].astype(float)
        radio = df["Radio"].astype(float)
        newspaper = df["Newspaper"].astype(float)
        eps = 1e-5

        total_spend = tv + radio + newspaper
        df["Total_Spend"] = total_spend

        if self.include_shares:
            df["TV_Share"] = tv / (total_spend + eps)
            df["Radio_Share"] = radio / (total_spend + eps)
            df["Newspaper_Share"] = newspaper / (total_spend + eps)

        if self.include_interactions:
            df["TV_x_Radio"] = tv * radio
            df["Radio_to_TV_Ratio"] = radio / (tv + eps)

        self.feature_names_out_ = df.columns.tolist()
        return df

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_:
            return np.array(self.feature_names_out_)
        # Default expected columns
        cols = ["TV", "Radio", "Newspaper", "Total_Spend"]
        if self.include_shares:
            cols.extend(["TV_Share", "Radio_Share", "Newspaper_Share"])
        if self.include_interactions:
            cols.extend(["TV_x_Radio", "Radio_to_TV_Ratio"])
        return np.array(cols)
