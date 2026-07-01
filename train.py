"""
train.py
--------
Loads the Telco churn dataset, engineers features, handles class imbalance
with SMOTE, trains and compares 3 models, then saves the best one.

Run with: python train.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE

DATA_PATH  = os.path.join("data", "telco_churn.csv")
MODEL_DIR  = "models"


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ── 2. Clean & engineer features ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    df = df.copy()

    # TotalCharges has hidden spaces → convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop customerID — it's just an identifier
    df.drop(columns=["customerID"], inplace=True)

    # Feature engineering: charge per month ratio
    df["ChargeRatio"] = df["MonthlyCharges"] / (df["tenure"] + 1)

    # Binary target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Encode all remaining object columns
    le = LabelEncoder()
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    print(f"Class balance — No churn: {(y==0).sum()}, Churn: {(y==1).sum()}")
    return X, y, list(X.columns)


# ── 3. Train & evaluate ───────────────────────────────────────────────────────

def train_and_compare(X_train, X_test, y_train, y_test, feature_names):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting":             GradientBoostingClassifier(n_estimators=200,
                                               random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds  = model.predict(X_test)
        proba  = model.predict_proba(X_test)[:, 1]
        acc    = accuracy_score(y_test, preds)
        auc    = roc_auc_score(y_test, proba)
        results[name] = {"model": model, "accuracy": acc, "roc_auc": auc, "preds": preds}
        print(f"\n{'='*40}")
        print(f"{name}  |  Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}")
        print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    return results


# ── 4. Save best model + scaler ───────────────────────────────────────────────

def save_best(results, scaler, feature_names):
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best      = results[best_name]
    print(f"\nBest model: {best_name}  (ROC-AUC {best['roc_auc']:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["model"], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler,        os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    # Save model comparison results (for dashboard)
    summary = {k: {"accuracy": v["accuracy"], "roc_auc": v["roc_auc"]}
               for k, v in results.items()}
    joblib.dump(summary, os.path.join(MODEL_DIR, "model_comparison.pkl"))
    joblib.dump(best_name, os.path.join(MODEL_DIR, "best_model_name.pkl"))

    print("Saved to models/")
    return best_name, best["model"]


# ── 5. Feature importance plot ────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[::-1][:15]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices],
                y=[feature_names[i] for i in indices],
                palette="viridis")
    plt.title(f"Top 15 Feature Importances — {model_name}")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X, y, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE to balance classes...")
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE — No churn: {(y_train==0).sum()}, Churn: {(y_train==1).sum()}")

    results  = train_and_compare(X_train, X_test, y_train, y_test, feature_names)
    best_name, best_model = save_best(results, scaler, feature_names)
    plot_feature_importance(best_model, feature_names, best_name)

    print("\nTraining complete! Run: streamlit run app.py")