import streamlit as st
import joblib
import pandas as pd
import numpy as np


# 1. Efficiently load model and scaler
@st.cache_resource
def load_assets():
    model = joblib.load("./models/loan_model.pkl")
    scaler = joblib.load("./models/scaler.pkl")
    return model, scaler


model, scaler = load_assets()

# 2. UI Configuration
st.set_page_config(page_title="Loan Risk Predictor", page_icon="🏦", layout="centered")
st.title("🏦 Credit Loan Default Risk Predictor")
st.markdown("Enter borrower details below to assess the probability of default.")

# 3. User Inputs
st.subheader("Financial & Demographic Profile")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    income = st.number_input("Annual Income ($)", min_value=0, value=65000)
    loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=15000)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=710)
    months_employed = st.number_input("Months Employed", min_value=0, value=48)
    dti_ratio = st.number_input("DTI Ratio (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=0.3)

with col2:
    num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=4)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, value=7.5)
    loan_term = st.number_input("Loan Term (Months)", min_value=12, value=36)

    education = st.selectbox("Education Level", ["Bachelor's / Other", "High School", "Master's", "PhD"])
    employment = st.selectbox("Employment Type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital = st.selectbox("Marital Status", ["Divorced / Widowed", "Married", "Single"])
    purpose = st.selectbox("Loan Purpose", ["Personal / Auto", "Business", "Education", "Home", "Other"])

st.subheader("Additional Information")
col3, col4, col5 = st.columns(3)
with col3: has_mortgage = st.checkbox("Has Mortgage")
with col4: has_dependents = st.checkbox("Has Dependents")
with col5: has_cosigner = st.checkbox("Has Co-Signer")

st.divider()

# 4. Prediction Logic
if st.button("Predict Loan Risk", type="primary", use_container_width=True):

    # Map dropdowns to the exact one-hot encoded format
    features_dict = {
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'MonthsEmployed': months_employed,
        'NumCreditLines': num_credit_lines,
        'InterestRate': interest_rate,
        'LoanTerm': loan_term,
        'DTIRatio': dti_ratio,
        'Education_High School': 1 if education == "High School" else 0,
        "Education_Master's": 1 if education == "Master's" else 0,
        'Education_PhD': 1 if education == "PhD" else 0,
        'EmploymentType_Part-time': 1 if employment == "Part-time" else 0,
        'EmploymentType_Self-employed': 1 if employment == "Self-employed" else 0,
        'EmploymentType_Unemployed': 1 if employment == "Unemployed" else 0,
        'MaritalStatus_Married': 1 if marital == "Married" else 0,
        'MaritalStatus_Single': 1 if marital == "Single" else 0,
        'HasMortgage_Yes': 1 if has_mortgage else 0,
        'HasDependents_Yes': 1 if has_dependents else 0,
        'LoanPurpose_Business': 1 if purpose == "Business" else 0,
        'LoanPurpose_Education': 1 if purpose == "Education" else 0,
        'LoanPurpose_Home': 1 if purpose == "Home" else 0,
        'LoanPurpose_Other': 1 if purpose == "Other" else 0,
        'HasCoSigner_Yes': 1 if has_cosigner else 0
    }

    # Create DataFrame (Ensures exact column names and ordering)
    input_df = pd.DataFrame([features_dict])

    # CRITICAL: Scale the user input using the saved scaler
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

    # Predict
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Display Results
    if prediction[0] == 1:
        st.error(f"🚨 **High Risk Borrower** detected. \n\nEstimated Default Probability: **{probability:.1f}%**")
    else:
        st.success(f"✅ **Low Risk Borrower** detected. \n\nEstimated Default Probability: **{probability:.1f}%**")
