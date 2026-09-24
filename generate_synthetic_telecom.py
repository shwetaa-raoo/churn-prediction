"""
Synthetic Indian telecom churn dataset generator.

ALL DATA IS SIMULATED. No real Airtel, Jio, Vi or BSNL customer data is used,
and churn is produced by the hand-written rules in `logit` below. Operator
is a label only (it has no direct effect on churn). Say this in your README.

Usage: python generate_synthetic_telecom.py  ->  indian_telecom_churn_synthetic.csv
"""
import numpy as np
import pandas as pd

SEED, N = 42, 10_000
rng = np.random.default_rng(SEED)

# --- Demographics / account -------------------------------------------------
operator = rng.choice(["Airtel", "Jio", "Vi", "BSNL"], N, p=[0.35, 0.42, 0.16, 0.07])
circle = rng.choice(["Delhi", "Mumbai", "Karnataka", "Tamil Nadu", "Uttar Pradesh",
                     "Maharashtra", "West Bengal", "Gujarat", "Rajasthan", "Bihar"], N)
city_tier = rng.choice(["Metro", "Tier-2", "Tier-3", "Rural"], N, p=[0.20, 0.30, 0.25, 0.25])
age = np.clip(rng.normal(32, 10, N), 18, 75).astype(int)
gender = rng.choice(["Male", "Female"], N, p=[0.52, 0.48])
plan_type = rng.choice(["Prepaid", "Postpaid"], N, p=[0.85, 0.15])
device = rng.choice(["Feature phone", "4G smartphone", "5G smartphone"], N, p=[0.12, 0.58, 0.30])
tenure_months = np.clip(rng.exponential(30, N), 1, 120).astype(int)
prepaid = plan_type == "Prepaid"

# --- Spend and plan ----------------------------------------------------------
tier_mult = pd.Series(city_tier).map({"Metro": 1.25, "Tier-2": 1.05, "Tier-3": 0.9, "Rural": 0.8}).values
base = np.where(prepaid, 240, 550)
monthly_recharge_inr = np.clip(rng.lognormal(np.log(base * tier_mult), 0.35), 99, 1500).astype(int)
plan_validity_days = np.where(prepaid, rng.choice([28, 56, 84, 365], N, p=[0.55, 0.15, 0.22, 0.08]), 30)
days_since_last_recharge = np.clip(rng.gamma(2, plan_validity_days / 4), 0, plan_validity_days * 2).astype(int)
overdue_days = np.maximum(days_since_last_recharge - plan_validity_days, 0)
payment_channel = np.where(
    prepaid,
    rng.choice(["UPI app", "Operator app", "Retail shop"], N, p=[0.5, 0.2, 0.3]),
    rng.choice(["UPI app", "Operator app", "Auto-debit"], N, p=[0.2, 0.25, 0.55]),
)
ott_bundle = (rng.random(N) < np.where(monthly_recharge_inr >= 400, 0.5, 0.1)).astype(int)

# --- Usage -------------------------------------------------------------------
data_mu = pd.Series(device).map({"Feature phone": 0.5, "4G smartphone": 16, "5G smartphone": 32}).values
data_gb_month = np.round(rng.lognormal(np.log(data_mu), 0.5), 1)
voice_minutes_month = np.clip(rng.normal(350, 150, N), 0, None).astype(int)
sms_per_month = rng.poisson(20, N)
recharge_change_pct = np.round(rng.normal(0, 20, N), 1)   # vs previous 3 months
data_change_pct = np.round(rng.normal(0, 25, N), 1)
dual_sim = (rng.random(N) < np.where(prepaid, 0.40, 0.15)).astype(int)  # uses another operator too

# --- Network experience ------------------------------------------------------
five_g_area = (rng.random(N) < pd.Series(city_tier).map(
    {"Metro": 0.85, "Tier-2": 0.60, "Tier-3": 0.30, "Rural": 0.10}).values).astype(int)
call_drop_rate_pct = np.round(np.clip(rng.gamma(3, 0.8, N) + (city_tier == "Rural"), 0, 15), 1)
network_complaints_3m = rng.poisson(0.4 + 0.15 * call_drop_rate_pct)
care_calls_3m = rng.poisson(0.6 + 0.4 * network_complaints_3m)

# --- Churn rules (this is what the model will "discover") --------------------
logit = (
    -3.6
    + 1.3 * prepaid
    - 0.025 * np.minimum(tenure_months, 60)
    + 0.35 * dual_sim
    + 0.25 * call_drop_rate_pct
    + 0.30 * network_complaints_3m
    + 0.20 * care_calls_3m
    + 0.035 * np.minimum(overdue_days, 60)
    - 0.02 * recharge_change_pct
    - 0.015 * data_change_pct
    - 0.40 * ott_bundle
    - 0.30 * (plan_validity_days >= 84)
    + 0.30 * (age < 25)
    + rng.normal(0, 0.7, N)           # unexplained noise so it isn't deterministic
)
churn = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

df = pd.DataFrame({
    "customer_id": [f"IN{100000 + i}" for i in range(N)],
    "operator": operator, "circle": circle, "city_tier": city_tier,
    "age": age, "gender": gender, "plan_type": plan_type, "device_type": device,
    "tenure_months": tenure_months, "monthly_recharge_inr": monthly_recharge_inr,
    "plan_validity_days": plan_validity_days, "days_since_last_recharge": days_since_last_recharge,
    "payment_channel": payment_channel, "ott_bundle": ott_bundle,
    "data_gb_month": data_gb_month, "voice_minutes_month": voice_minutes_month,
    "sms_per_month": sms_per_month, "recharge_change_pct": recharge_change_pct,
    "data_change_pct": data_change_pct, "dual_sim": dual_sim, "five_g_area": five_g_area,
    "call_drop_rate_pct": call_drop_rate_pct, "network_complaints_3m": network_complaints_3m,
    "care_calls_3m": care_calls_3m, "churn": churn,
})
df.to_csv("indian_telecom_churn_synthetic.csv", index=False)
print(df.shape, "churn rate:", round(df.churn.mean(), 3))
