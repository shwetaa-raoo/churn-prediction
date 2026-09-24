"""
train.py
--------
Loads the synthetic Indian telecom churn dataset, engineers features, encodes
and scales them, trains and compares 3 models, then saves the best one.

Run with: python train.py
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save charts to file without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, recall_score, roc_auc_score
)

from features import add_features

DATA_PATH    = os.path.join("data", "indian_telecom_churn_synthetic.csv")
MODEL_DIR    = "models"
RANDOM_STATE = 42
ID_COL       = "customer_id"
TARGET       = "churn"

# Switches you can flip to experiment:
USE_SMOTE     = False      # SMOTE over-predicted churn in earlier runs, so it is off by default
SELECT_METRIC = "roc_auc"  # "roc_auc" (recommended for imbalanced churn) or "accuracy"

# Display order for dropdowns in the app (anything not listed is sorted A-Z)
PREFERRED_ORDER = {
    "operator":    ["Airtel", "Jio", "Vi", "BSNL"],
    "city_tier":   ["Metro", "Tier-2", "Tier-3", "Rural"],
    "device_type": ["Feature phone", "4G smartphone", "5G smartphone"],
    "plan_type":   ["Prepaid", "Postpaid"],
}


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# ── 2. Split raw features / target and describe the input schema ─────────────

def split_features_target(df: pd.DataFrame):
    df = df.drop(columns=[ID_COL])  # identifier only
    y = df[TARGET].astype(int)
    X_raw = df.drop(columns=[TARGET])
    print(f"Class balance — No churn: {(y==0).sum()}, Churn: {(y==1).sum()}")
    return X_raw, y


def build_schema(X_raw: pd.DataFrame) -> dict:
    """Record the raw input columns, their categories and numeric ranges,
    so the Streamlit form is generated from the data instead of hard-coded."""
    categorical, numeric = {}, {}
    for col in X_raw.columns:
        s = X_raw[col]
        if s.dtype == "object":
            cats = sorted(s.unique())
            order = PREFERRED_ORDER.get(col, [])
            categorical[col] = [c for c in order if c in cats] + [c for c in cats if c not in order]
        else:
            is_int = pd.api.types.is_integer_dtype(s)
            cast = int if is_int else float
            if set(s.unique()) <= {0, 1}:
                kind = "binary"
            else:
                kind = "int" if is_int else "float"
            numeric[col] = {"kind": kind, "min": cast(s.min()), "max": cast(s.max()),
                            "default": cast(s.median())}
    return {"columns": list(X_raw.columns), "categorical": categorical, "numeric": numeric}


# ── 3. Encode + scale ────────────────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
    ])


# ── 4. Train & evaluate ──────────────────────────────────────────────────────

def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "model":        model,
            "accuracy":     accuracy_score(y_test, preds),
            "roc_auc":      roc_auc_score(y_test, proba),
            "churn_recall": recall_score(y_test, preds),
            "churn_f1":     f1_score(y_test, preds),
        }
        r = results[name]
        print(f"\n{'='*40}")
        print(f"{name}  |  Accuracy: {r['accuracy']:.4f}  |  ROC-AUC: {r['roc_auc']:.4f}"
              f"  |  Churn recall: {r['churn_recall']:.4f}")
        print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

    return results


# ── 5. Save best model + preprocessing ───────────────────────────────────────

def save_best(results, preprocessor, feature_names, schema, meta):
    best_name = max(results, key=lambda k: results[k][SELECT_METRIC])
    best      = results[best_name]
    print(f"\nBest model by {SELECT_METRIC}: {best_name}  "
          f"(Accuracy {best['accuracy']:.4f}, ROC-AUC {best['roc_auc']:.4f})")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best["model"], os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(preprocessor,  os.path.join(MODEL_DIR, "preprocessor.pkl"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))
    joblib.dump(schema,        os.path.join(MODEL_DIR, "input_schema.pkl"))

    # Model comparison results (for the dashboard)
    summary = {k: {m: v[m] for m in ("accuracy", "roc_auc", "churn_recall", "churn_f1")}
               for k, v in results.items()}
    joblib.dump(summary,   os.path.join(MODEL_DIR, "model_comparison.pkl"))
    joblib.dump(best_name, os.path.join(MODEL_DIR, "best_model_name.pkl"))
    joblib.dump({**meta, "select_metric": SELECT_METRIC, "use_smote": USE_SMOTE},
                os.path.join(MODEL_DIR, "training_meta.pkl"))

    print("Saved to models/")
    return best_name, best["model"]


# ── 6. Feature importance plot ───────────────────────────────────────────────

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return

    indices = np.argsort(importances)[::-1][:15]
    labels  = [feature_names[i] for i in indices]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=labels, hue=labels,
                palette="viridis", legend=False)
    plt.title(f"Top 15 Feature Importances — {model_name}")
    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Feature importance chart saved to {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    X_raw, y = split_features_target(df)
    schema = build_schema(X_raw)

    X = add_features(X_raw)  # engineered features (recharge_per_gb, overdue_days)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    baseline = max(y_test.mean(), 1 - y_test.mean())
    print(f"Majority-class baseline accuracy: {baseline:.4f}")

    # Fit encoding/scaling on the training split only, then apply to both splits
    preprocessor  = build_preprocessor(X_train)
    X_train       = preprocessor.fit_transform(X_train)
    X_test        = preprocessor.transform(X_test)
    feature_names = [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]

    if USE_SMOTE:
        from imblearn.over_sampling import SMOTE
        print("\nApplying SMOTE to the training split only...")
        X_train, y_train = SMOTE(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
    else:
        print("\nTraining with the original class distribution...")

    results = train_and_compare(X_train, X_test, y_train, y_test)
    meta = {"baseline_accuracy": float(baseline), "n_rows": int(len(df)),
            "churn_rate": float(y.mean())}
    best_name, best_model = save_best(results, preprocessor, feature_names, schema, meta)
    plot_feature_importance(best_model, feature_names, best_name)

    print("\nTraining complete! Run: streamlit run app.py")
