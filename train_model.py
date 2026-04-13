# Force Matplotlib to show pop-up windows (Must be at the very top!)
import matplotlib
matplotlib.use('TkAgg')

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_data
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    auc
)

def train_model():
    # 1. Load preprocessed and scaled data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Handle Class Imbalance with SMOTE
    print("Applying SMOTE to balance training data...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 3. Train Model (Optimized with n_jobs=-1 for speed)
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced", # Extra layer of defense against imbalance
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # ==========================================
    # 4. Evaluate & Plot Graphs
    # ==========================================
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Set a nice visual style for the graphs
    sns.set_theme(style="whitegrid")

    # --- Graph 1: Confusion Matrix ---
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix", fontsize=14)
    plt.xlabel("Predicted Label (0=Safe, 1=Default)")
    plt.ylabel("Actual Label (0=Safe, 1=Default)")
    plt.tight_layout()
    plt.savefig("./models/confusion_matrix.png")  # Saves the graph
    plt.show()

    # --- Graph 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Receiver Operating Characteristic (ROC)", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("./models/roc_curve.png")
    plt.show()

    # --- Graph 3: Precision-Recall Curve (Crucial for Imbalanced Data) ---
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
    plt.title("Precision-Recall Curve (PR-AUC)", fontsize=14)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("./models/pr_curve.png")
    plt.show()

    # --- Graph 4: Feature Importances ---
    importances = model.feature_importances_
    features_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": importances
    }).sort_values("Importance", ascending=False).head(10)  # Top 10 features

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=features_df, palette="viridis")
    plt.title("Top 10 Feature Importances", fontsize=14)
    plt.tight_layout()
    plt.savefig("./models/feature_importance.png")
    plt.show()

    # ==========================================
    # 5. Save the Model
    # ==========================================
    joblib.dump(model, "./models/loan_model.pkl")
    print("✅ Model saved successfully to ./models/loan_model.pkl")


if __name__ == "__main__":
    train_model()