"""
Customer Churn Prediction - Training Pipeline
Run: python train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier

DATA_PATH = "data/churn.csv"
MODEL_PATH = "models/churn_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.pkl"


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------
# 2. CLEAN & PREPROCESS
# ---------------------------------------------------------
def preprocess(df):
    df = df.copy()

    # Drop customer ID (not predictive)
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # TotalCharges has some blank strings -> convert to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # Target column
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Encode categorical columns
    encoders = {}
    cat_cols = df.select_dtypes(include="object").columns

    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


# ---------------------------------------------------------
# 3. TRAIN / COMPARE MODELS
# ---------------------------------------------------------
def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, probs)
        })
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    print("\n=== Model Comparison ===")
    print(results_df.to_string(index=False))

    best_name = results_df.iloc[0]["Model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name}")

    return best_model, best_name, results_df


# ---------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df, encoders = preprocess(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    best_model, best_name, results_df = train_and_compare(
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    # Detailed report for best model
    preds = best_model.predict(X_test_scaled)
    print("\n=== Classification Report (Best Model) ===")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))

    # Save artifacts
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_columns, FEATURES_PATH)
    results_df.to_csv("models/model_comparison.csv", index=False)

    # Save a small background sample for SHAP explainability in the app
    background_sample = X_train_scaled[:100]
    joblib.dump(background_sample, "models/background_sample.pkl")

    print(f"\nSaved best model ({best_name}) to {MODEL_PATH}")
    print("Saved encoders, scaler, feature list, and comparison table to models/")


if __name__ == "__main__":
    main()
