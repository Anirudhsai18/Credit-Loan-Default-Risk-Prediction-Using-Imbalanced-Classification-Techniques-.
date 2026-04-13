import joblib
import pandas as pd
import numpy as np

def run_test_prediction():
    print("Loading model and scaler...")
    model = joblib.load("./models/loan_model.pkl")
    scaler = joblib.load("./models/scaler.pkl")

    # The exact 24 features
    feature_names = [
        'Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed',
        'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio',
        'Education_High School', "Education_Master's", 'Education_PhD',
        'EmploymentType_Part-time', 'EmploymentType_Self-employed', 'EmploymentType_Unemployed',
        'MaritalStatus_Married', 'MaritalStatus_Single', 'HasMortgage_Yes', 'HasDependents_Yes',
        'LoanPurpose_Business', 'LoanPurpose_Education', 'LoanPurpose_Home', 'LoanPurpose_Other', 'HasCoSigner_Yes'
    ]

    print("Generating dummy data...")
    # Generate random data representing one borrower
    dummy_data = np.random.rand(1, 24)
    sample_df = pd.DataFrame(dummy_data, columns=feature_names)

    print("Scaling dummy data...")
    sample_scaled = pd.DataFrame(scaler.transform(sample_df), columns=feature_names)

    print("Making prediction...")
    prediction = model.predict(sample_scaled)
    probability = model.predict_proba(sample_scaled)[0][1] * 100

    print("\n--- Result ---")
    if prediction[0] == 1:
        print(f" High Risk: Loan Default ({probability:.1f}% probability)")
    else:
        print(f"✅ Low Risk: Safe Borrower ({probability:.1f}% probability)")

if __name__ == "__main__":
    run_test_prediction()