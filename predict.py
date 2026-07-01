"""
predict.py
----------
Loads the saved model and scaler, takes a dict of customer features,
and returns a churn probability + risk label.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = "models"


def load_model():
    model         = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler        = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    return model, scaler, feature_names


def predict_churn(customer: dict) -> dict:
    """
    Takes a dict of raw customer features (matching the training columns),
    returns churn probability and a risk label.
    """
    model, scaler, feature_names = load_model()

    df = pd.DataFrame([customer])

    # Add engineered feature
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Align columns to training order
    df = df[feature_names]

    X_scaled = scaler.transform(df)
    proba    = model.predict_proba(X_scaled)[0][1]

    if proba >= 0.7:
        risk = "High Risk"
    elif proba >= 0.4:
        risk = "Medium Risk"
    else:
        risk = "Low Risk"

    return {"churn_probability": round(float(proba), 4), "risk_label": risk}


if __name__ == "__main__":
    # Quick test with a sample customer
    sample = {
        "gender": 0, "SeniorCitizen": 0, "Partner": 1, "Dependents": 0,
        "tenure": 2, "PhoneService": 1, "MultipleLines": 0,
        "InternetService": 1, "OnlineSecurity": 0, "OnlineBackup": 0,
        "DeviceProtection": 0, "TechSupport": 0, "StreamingTV": 0,
        "StreamingMovies": 0, "Contract": 0, "PaperlessBilling": 1,
        "PaymentMethod": 2, "MonthlyCharges": 70.0, "TotalCharges": 140.0,
    }
    result = predict_churn(sample)
    print(f"Churn probability: {result['churn_probability']:.2%}")
    print(f"Risk label: {result['risk_label']}")
