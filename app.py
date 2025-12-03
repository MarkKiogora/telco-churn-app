import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📡", layout="centered")

st.title("📡 Telco Customer Churn Prediction App")
st.write("Enter customer information below to predict churn probability.")

# Load model & preprocessor
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("logreg_churn_model.pkl")

# ---- FORM ----
st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

with col2:
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, step=0.1)
    total = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, step=0.1)
    payment = st.selectbox("Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    tenure = st.slider("Tenure (Months)", 1, 72)

# ---- PREDICT ----
if st.button("Predict Churn"):
    customer = {
        "Gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "PaymentMethod": payment,
        "PaperlessBilling": paperless,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    df = pd.DataFrame([customer])
    X = preprocessor.transform(df)
    proba = model.predict_proba(X)[0][1]

    st.subheader("🔮 Prediction Result")

    st.write(f"### Churn Probability: **{proba:.2%}**")

    if proba >= 0.50:
        st.error("⚠️ High Risk: Customer is likely to churn!")
    else:
        st.success("✅ Low Risk: Customer is likely to stay.")

    st.info("Model: Logistic Regression (balanced weights)")
