# Customer Churn Prediction System

A machine learning system that predicts customer churn using the Telco Customer Churn dataset, with model comparison, feature importance analysis, and an interactive Streamlit dashboard.

## Stack
- **Scikit-learn** — Logistic Regression, Random Forest
- **XGBoost** — gradient boosting classifier
- **imbalanced-learn (SMOTE)** — handles class imbalance
- **Pandas / NumPy** — data processing and feature engineering
- **Streamlit** — interactive prediction dashboard
- **Matplotlib / Seaborn** — visualizations

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Step 1 — Train the models:**
```bash
python train.py
```

This will:
- Load and preprocess the dataset
- Engineer a `ChargeRatio` feature
- Apply SMOTE to handle class imbalance
- Train and compare 3 models (Logistic Regression, Random Forest, XGBoost)
- Save the best model based on ROC-AUC
- Generate a feature importance chart

**Step 2 — Launch the dashboard:**
```bash
streamlit run app.py
```

## How it works

1. **Preprocessing**: TotalCharges is cleaned (hidden spaces), customerID dropped, categorical columns label-encoded.
2. **Feature Engineering**: A `ChargeRatio` (MonthlyCharges / tenure+1) is added to capture spending intensity.
3. **Class Imbalance**: SMOTE (Synthetic Minority Oversampling) generates synthetic churn examples so the model doesn't just predict "no churn" for everything.
4. **Model Comparison**: All 3 models are evaluated on accuracy and ROC-AUC. The best ROC-AUC model is saved.
5. **Prediction**: The dashboard encodes user inputs, applies the saved scaler, and returns a churn probability with a risk label (High / Medium / Low).

## CV Bullet
*Built a customer churn prediction system comparing Logistic Regression, Random Forest, and XGBoost classifiers on the Telco dataset, applying SMOTE for class imbalance handling and feature engineering to optimize ROC-AUC.*
