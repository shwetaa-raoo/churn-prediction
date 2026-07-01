"""
app.py
------
Streamlit dashboard: model comparison, feature importance, and
an interactive form to predict churn for a new customer.

Run with: streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from predict import predict_churn

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Customer Churn Prediction Dashboard")
st.caption("Trained on Telco Customer Churn dataset · Random Forest / XGBoost / Logistic Regression")

MODEL_DIR = "models"

# ── Check model exists ────────────────────────────────────────────────────────
if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

# ── Load comparison results ───────────────────────────────────────────────────
comparison  = joblib.load(os.path.join(MODEL_DIR, "model_comparison.pkl"))
best_name   = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Predict Churn", "📊 Model Performance"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction form
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")
    st.caption(f"Using best model: **{best_name}**")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender          = st.selectbox("Gender", ["Male", "Female"])
        senior          = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner         = st.selectbox("Has Partner", ["No", "Yes"])
        dependents      = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        phone_service   = st.selectbox("Phone Service", ["No", "Yes"])

    with col2:
        multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet        = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_sec      = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

    with col3:
        streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_mov   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment         = st.selectbox("Payment Method", [
                            "Electronic check", "Mailed check",
                            "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly         = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0)
        total           = st.number_input("Total Charges ($)", 0.0, 10000.0, 780.0)

    # Encode inputs to match training encoding
    encode = {
        "No": 0, "Yes": 1,
        "Male": 0, "Female": 1,
        "DSL": 0, "Fiber optic": 1, "No": 2,
        "No internet service": 2, "No phone service": 2,
        "Month-to-month": 0, "One year": 1, "Two year": 2,
        "Electronic check": 0, "Mailed check": 1,
        "Bank transfer (automatic)": 2, "Credit card (automatic)": 3,
    }

    customer = {
        "gender":            encode.get(gender, 0),
        "SeniorCitizen":     encode.get(senior, 0),
        "Partner":           encode.get(partner, 0),
        "Dependents":        encode.get(dependents, 0),
        "tenure":            tenure,
        "PhoneService":      encode.get(phone_service, 0),
        "MultipleLines":     encode.get(multiple_lines, 0),
        "InternetService":   encode.get(internet, 0),
        "OnlineSecurity":    encode.get(online_sec, 0),
        "OnlineBackup":      encode.get(online_backup, 0),
        "DeviceProtection":  encode.get(device_prot, 0),
        "TechSupport":       encode.get(tech_support, 0),
        "StreamingTV":       encode.get(streaming_tv, 0),
        "StreamingMovies":   encode.get(streaming_mov, 0),
        "Contract":          encode.get(contract, 0),
        "PaperlessBilling":  encode.get(paperless, 0),
        "PaymentMethod":     encode.get(payment, 0),
        "MonthlyCharges":    monthly,
        "TotalCharges":      total,
    }

    if st.button("Predict Churn", type="primary"):
        result = predict_churn(customer)
        prob   = result["churn_probability"]
        risk   = result["risk_label"]

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{prob:.1%}")
        c2.metric("Risk Level", risk)
        c3.metric("Retention Probability", f"{1-prob:.1%}")

        if risk == "High Risk":
            st.error(f"⚠️ This customer is at **high risk** of churning ({prob:.1%}). Consider proactive retention offers.")
        elif risk == "Medium Risk":
            st.warning(f"🔶 This customer has **medium risk** of churning ({prob:.1%}). Monitor and engage.")
        else:
            st.success(f"✅ This customer is at **low risk** of churning ({prob:.1%}). Keep up the good service!")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Model comparison
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Model Comparison")

    comp_df = pd.DataFrame(comparison).T.reset_index()
    comp_df.columns = ["Model", "Accuracy", "ROC-AUC"]
    comp_df = comp_df.sort_values("ROC-AUC", ascending=False)

    st.dataframe(comp_df.style.highlight_max(
        subset=["Accuracy", "ROC-AUC"], color="#d4edda"), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(comp_df["Model"], comp_df["Accuracy"], color=["#2196F3","#4CAF50","#FF9800"])
    axes[0].set_title("Accuracy by Model")
    axes[0].set_ylim(0.7, 1.0)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(comp_df["Model"], comp_df["ROC-AUC"], color=["#2196F3","#4CAF50","#FF9800"])
    axes[1].set_title("ROC-AUC by Model")
    axes[1].set_ylim(0.7, 1.0)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    img_path = os.path.join(MODEL_DIR, "feature_importance.png")
    if os.path.exists(img_path):
        st.image(img_path, use_column_width=True)
    else:
        st.info("Feature importance chart not found. Run train.py to generate it.")
