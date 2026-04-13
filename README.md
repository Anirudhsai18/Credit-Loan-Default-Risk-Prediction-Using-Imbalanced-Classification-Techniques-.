# Credit-Loan-Default-Risk-Prediction-Using-Imbalanced-Classification-Techniques-.
ML model to predict loan defaults using Lending Club data. Handles imbalanced data and uses Logistic Regression &amp; XGBoost to improve risk detection and lending decisions.
# 🏦 Credit Loan Default Risk Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Project Overview
Predicting loan defaults is one of the most critical challenges in the financial sector. However, credit datasets suffer from severe **class imbalance**—the vast majority of borrowers repay their loans, while only a small fraction default. Traditional machine learning models trained on this data suffer from the "Accuracy Paradox," where they boast 95% accuracy by simply predicting that *no one* defaults, completely failing their business purpose.

This project builds an end-to-end, production-ready machine learning pipeline that successfully neutralizes class imbalance to accurately identify high-risk borrowers. It features a fully trained **Random Forest Classifier**, data resampling via **SMOTE**, and an interactive **Streamlit web dashboard** for real-time risk assessment.

## ✨ Key Features
* **Imbalanced Classification Techniques:** Utilizes Synthetic Minority Over-sampling Technique (SMOTE) and algorithm-level class weighting to force the model to learn the minority "Default" class.
* **Specialized Evaluation Metrics:** Evaluates performance using Precision-Recall AUC (PR-AUC), F1-Score, and Minority Recall rather than misleading global accuracy.
* **Automated Preprocessing:** Handles one-hot encoding, missing values, and data scaling (StandardScaler), preserving feature names across the pipeline.
* **Interactive UI:** Includes a user-friendly Streamlit web application allowing loan officers to input borrower profiles and receive instant probability-based risk assessments.

## 📂 Project Structure
```text
├── data/
│   └── Loan_default.csv       # Raw dataset (Not uploaded to GitHub if too large)
├── models/
│   ├── loan_model.pkl         # Trained Random Forest model (Generated locally)
│   ├── scaler.pkl             # Fitted StandardScaler (Generated locally)
│   ├── confusion_matrix.png   # Evaluation graphs
│   └── feature_importance.png # Feature importance plot
├── preprocessing.py           # Data cleaning, encoding, and scaling pipeline
├── train.py                   # Model training, SMOTE application, and evaluation
├── predict.py                 # Command-line interface for quick testing
├── app.py                     # Streamlit web dashboard application
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
