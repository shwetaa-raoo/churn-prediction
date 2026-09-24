"""
features.py
-----------
Feature engineering shared by train.py and predict.py, so the training path
and the prediction path can never drift apart.
"""

import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features to a DataFrame of raw customer columns."""
    df = df.copy()

    # Rupees paid per GB of data used (the +1 avoids division by zero).
    # High values = paying a lot for very little data (replaces the old ChargeRatio).
    df["recharge_per_gb"] = df["monthly_recharge_inr"] / (df["data_gb_month"] + 1)

    # Days the customer has gone past the end of their plan without recharging.
    df["overdue_days"] = (
        df["days_since_last_recharge"] - df["plan_validity_days"]
    ).clip(lower=0)

    return df
