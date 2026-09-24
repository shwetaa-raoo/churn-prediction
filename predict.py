"""
predict.py
----------
Loads the saved model and preprocessor, takes a dict of raw customer features,
and returns a churn probability + risk label.
"""

import os
from functools import lru_cache

import joblib
import pandas as pd

from features import add_features

MODEL_DIR = "models"

# Risk bands. With ~18% of customers churning, probabilities rarely go above 0.5,
# so the bands sit lower than a naive 0.4 / 0.7. On the held-out test set these
# bands had actual churn rates of roughly 10% (Low), 27% (Medium), 48% (High).
HIGH_RISK   = 0.35
MEDIUM_RISK = 0.20


@lru_cache(maxsize=1)
def load_artifacts():
    model        = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.pkl"))
    schema       = joblib.load(os.path.join(MODEL_DIR, "input_schema.pkl"))
    return model, preprocessor, schema


def predict_churn(customer: dict) -> dict:
    """
    Takes a dict of raw customer features (human-readable values such as
    "Prepaid" or 299, matching the dataset columns), returns churn probability
    and a risk label. Encoding and scaling are handled by the saved preprocessor.
    """
    model, preprocessor, schema = load_artifacts()

    missing = [c for c in schema["columns"] if c not in customer]
    if missing:
        raise ValueError(f"Missing customer fields: {missing}")

    df = pd.DataFrame([customer])[schema["columns"]]   # align to training order
    df = add_features(df)                              # engineered features

    proba = model.predict_proba(preprocessor.transform(df))[0][1]

    if proba >= HIGH_RISK:
        risk = "High Risk"
    elif proba >= MEDIUM_RISK:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {"churn_probability": round(float(proba), 4), "risk_label": risk}


if __name__ == "__main__":
    # Quick test with a sample prepaid customer who has gone past plan expiry
    sample = {
        "operator": "Jio", "circle": "Bihar", "city_tier": "Rural",
        "age": 22, "gender": "Male", "plan_type": "Prepaid",
        "device_type": "4G smartphone", "tenure_months": 4,
        "monthly_recharge_inr": 299, "plan_validity_days": 28,
        "days_since_last_recharge": 55, "payment_channel": "Retail shop",
        "ott_bundle": 0, "data_gb_month": 6.0, "voice_minutes_month": 200,
        "sms_per_month": 10, "recharge_change_pct": -25.0, "data_change_pct": -30.0,
        "dual_sim": 1, "five_g_area": 0, "call_drop_rate_pct": 6.0,
        "network_complaints_3m": 3, "care_calls_3m": 2,
    }
    result = predict_churn(sample)
    print(f"Churn probability: {result['churn_probability']:.2%}")
    print(f"Risk label: {result['risk_label']}")
