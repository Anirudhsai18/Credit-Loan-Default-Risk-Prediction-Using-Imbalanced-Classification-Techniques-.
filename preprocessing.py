import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(data_path="./data/Loan_default.csv", save_scaler=True):
    print("Loading and preprocessing data...")
    df = pd.read_csv(data_path)

    # 1. Drop the ID column (no predictive value)
    if "LoanID" in df.columns:
        df = df.drop("LoanID", axis=1)

    # 2. One-Hot Encoding (Creates the exact 24 features)
    df = pd.get_dummies(df, drop_first=True)

    # 3. Separate Features (X) and Target (y)
    X = df.drop("Default", axis=1)
    y = df["Default"]

    # 4. Scale the features
    scaler = StandardScaler()

    # CRITICAL: Keep feature names after scaling by wrapping in a DataFrame
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 5. Save the scaler for the Streamlit App
    if save_scaler:
        os.makedirs("./models", exist_ok=True)
        joblib.dump(scaler, "./models/scaler.pkl")
        print("✅ Scaler saved to ./models/scaler.pkl")

    # 6. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print(f"✅ Preprocessing complete. X_train shape: {X_train.shape}")