"""
Customer Churn Prediction - Streamlit App
Run from project root: streamlit run app/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

# ---------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    background = joblib.load("models/background_sample.pkl")
    return model, encoders, scaler, feature_columns, background

model, encoders, scaler, feature_columns, background = load_artifacts()

st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict the likelihood of churn.")

# ---------------------------------------------------------
# Model comparison chart (sidebar or expander)
# ---------------------------------------------------------
with st.expander("📈 View model comparison"):
    try:
        comparison_df = pd.read_csv("models/model_comparison.csv")
        st.dataframe(comparison_df, use_container_width=True)

        chart_df = comparison_df.set_index("Model")[["ROC-AUC", "Accuracy"]]
        st.bar_chart(chart_df)
    except FileNotFoundError:
        st.info("Model comparison data not found. Run train_model.py first.")

# ---------------------------------------------------------
# Input form
# ---------------------------------------------------------
def user_input_form():
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["No", "Yes"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with col2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges", min_value=0.0, value=840.0)

    data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    return pd.DataFrame([data])


input_df = user_input_form()

# ---------------------------------------------------------
# Predict
# ---------------------------------------------------------
if st.button("Predict Churn"):
    df = input_df.copy()

    # Apply saved label encoders to categorical columns
    for col, le in encoders.items():
        if col in df.columns and col != "Churn":
            df[col] = le.transform(df[col].astype(str))

    # Reorder columns to match training feature order
    df = df[feature_columns]

    scaled = scaler.transform(df)

    prob = model.predict_proba(scaled)[0][1]
    pred = model.predict(scaled)[0]

    st.subheader("Result")
    if pred == 1:
        st.error(f"⚠️ This customer is likely to CHURN (probability: {prob:.2%})")
    else:
        st.success(f"✅ This customer is likely to STAY (churn probability: {prob:.2%})")

    st.progress(float(prob))

    # -------------------------------------------------
    # SHAP Explainability
    # -------------------------------------------------
    st.subheader("Why this prediction? (SHAP explanation)")
    try:
        explainer = shap.Explainer(model, background, feature_names=feature_columns)
        shap_values = explainer(scaled)

        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"SHAP explanation not available: {e}")
