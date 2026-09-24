"""
app.py
------
Streamlit dashboard: model comparison, feature importance, and
an interactive form to predict churn for a new customer.

The form is generated from models/input_schema.pkl (written by train.py),
so it always matches the columns the model was trained on.

Run with: streamlit run app.py
"""

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from predict import predict_churn

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="wide")
st.title("📉 Telecom Churn Prediction Dashboard")

MODEL_DIR = "models"

# ── Check model exists ────────────────────────────────────────────────────────
if not os.path.exists(os.path.join(MODEL_DIR, "best_model.pkl")):
    st.error("No trained model found. Run `python train.py` first.")
    st.stop()

# ── Load training artifacts ───────────────────────────────────────────────────
comparison = joblib.load(os.path.join(MODEL_DIR, "model_comparison.pkl"))
best_name  = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
schema     = joblib.load(os.path.join(MODEL_DIR, "input_schema.pkl"))
meta       = joblib.load(os.path.join(MODEL_DIR, "training_meta.pkl"))

st.caption(
    f"Trained on a **synthetic** Indian telecom dataset ({meta['n_rows']:,} customers) · "
    "Logistic Regression / Random Forest / Gradient Boosting. "
    "Operator names are labels only; predictions are illustrative."
)

# ── Form layout ───────────────────────────────────────────────────────────────
GROUPS = [
    ("Customer profile", ["operator", "circle", "city_tier", "age", "gender",
                          "device_type", "plan_type", "tenure_months"]),
    ("Plan & recharge",  ["monthly_recharge_inr", "plan_validity_days",
                          "days_since_last_recharge", "payment_channel",
                          "ott_bundle", "recharge_change_pct"]),
    ("Usage",            ["data_gb_month", "voice_minutes_month", "sms_per_month",
                          "data_change_pct", "dual_sim"]),
    ("Network experience", ["five_g_area", "call_drop_rate_pct",
                            "network_complaints_3m", "care_calls_3m"]),
]

LABELS = {
    "operator": "Operator", "circle": "Circle / State", "city_tier": "City Tier",
    "age": "Age", "gender": "Gender", "device_type": "Device Type",
    "plan_type": "Plan Type", "tenure_months": "Tenure (months)",
    "monthly_recharge_inr": "Monthly Recharge (₹)",
    "plan_validity_days": "Plan Validity (days)",
    "days_since_last_recharge": "Days Since Last Recharge",
    "payment_channel": "Payment Channel", "ott_bundle": "OTT Bundle",
    "recharge_change_pct": "Recharge Change vs Last 3 Months (%)",
    "data_gb_month": "Data Used (GB / month)",
    "voice_minutes_month": "Voice Minutes / month", "sms_per_month": "SMS / month",
    "data_change_pct": "Data Usage Change vs Last 3 Months (%)",
    "dual_sim": "Also Uses Another SIM", "five_g_area": "5G Available in Area",
    "call_drop_rate_pct": "Call Drop Rate (%)",
    "network_complaints_3m": "Network Complaints (last 3 months)",
    "care_calls_3m": "Customer Care Calls (last 3 months)",
}


def render_input(col: str, container):
    """Draw the right widget for a column, based on the saved schema."""
    label = LABELS.get(col, col.replace("_", " ").title())

    if col in schema["categorical"]:
        return container.selectbox(label, schema["categorical"][col])

    spec = schema["numeric"][col]
    if spec["kind"] == "binary":
        return int(container.selectbox(label, ["No", "Yes"]) == "Yes")
    if col == "tenure_months":
        return container.slider(label, spec["min"], spec["max"], spec["default"])

    step = 1 if spec["kind"] == "int" else 0.1
    return container.number_input(label, min_value=spec["min"], max_value=spec["max"],
                                  value=spec["default"], step=step)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Predict Churn", "📊 Model Performance"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction form
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Enter Customer Details")
    st.caption(f"Using best model: **{best_name}**")

    customer = {}
    for title, cols in GROUPS:
        st.markdown(f"**{title}**")
        grid = st.columns(4)
        for i, col in enumerate(cols):
            customer[col] = render_input(col, grid[i % 4])

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
            st.error(f"⚠️ This customer is at **high risk** of churning ({prob:.1%}). Consider a proactive retention offer, such as a bonus-data recharge.")
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
    comp_df.columns = ["Model", "Accuracy", "ROC-AUC", "Churn Recall", "Churn F1"]
    comp_df = comp_df.sort_values("ROC-AUC", ascending=False)

    metric_cols = ["Accuracy", "ROC-AUC", "Churn Recall", "Churn F1"]
    styled = (comp_df.style
              .format({c: "{:.3f}" for c in metric_cols})
              .highlight_max(subset=metric_cols, color="#2e7d32"))  # dark green: readable on light and dark themes
    st.dataframe(styled, hide_index=True)
    st.caption(
        f"Only {meta['churn_rate']:.0%} of customers churn, so always predicting \"no churn\" "
        f"already scores {meta['baseline_accuracy']:.1%} accuracy. Models are therefore ranked by "
        "ROC-AUC, and churn recall shows how many actual churners are caught at a 0.5 cut-off. "
        "The risk bands in the predictor use lower cut-offs to catch more of them."
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    axes[0].bar(comp_df["Model"], comp_df["Accuracy"], color=colors)
    axes[0].axhline(meta["baseline_accuracy"], color="red", linestyle="--", linewidth=1,
                    label="Always-predict-no-churn baseline")
    axes[0].set_title("Accuracy by Model")
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(comp_df["Model"], comp_df["ROC-AUC"], color=colors)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random guessing")
    axes[1].set_title("ROC-AUC by Model")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend(loc="lower right", fontsize=8)
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Feature Importance")
    st.caption("Permutation importance on the held-out test set: how much ROC-AUC drops when a feature is shuffled.")
    img_path = os.path.join(MODEL_DIR, "feature_importance.png")
    if os.path.exists(img_path):
        st.image(img_path)
    else:
        st.info("Feature importance chart not found. Run train.py to generate it.")
