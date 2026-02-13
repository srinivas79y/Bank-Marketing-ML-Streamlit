# ==========================================
# Streamlit App - Bank Marketing Prediction
# ==========================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(page_title="Bank Marketing ML App", layout="wide")

st.title("📊 Bank Marketing Subscription Prediction")
st.markdown(
    """
This application allows you to:
- Select a trained classification model
- Upload a test dataset
- Evaluate performance using multiple metrics
"""
)

# ==========================================
# Download Sample Test Data
# ==========================================

st.markdown("### 📥 Download Sample Test Dataset")

with open("test_data.csv", "rb") as file:
    st.download_button(
        label="Download Sample Test CSV",
        data=file,
        file_name="test_data.csv",
        mime="text/csv"
    )

st.markdown("---")

# ==========================================
# Load Models & Artifacts
# ==========================================

models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "kNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl"),
    "XGBoost": joblib.load("model/xgboost.pkl")
}

scaler = joblib.load("model/scaler.pkl")
trained_columns = joblib.load("model/feature_columns.pkl")

# ==========================================
# Sidebar Controls
# ==========================================

st.sidebar.header("⚙️ Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (Bank Dataset Format)",
    type=["csv"]
)

# ==========================================
# Main Logic
# ==========================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=";")

    if "y" not in df.columns:
        st.error("Uploaded file must contain target column 'y'")
    else:

        # Convert target
        df["y"] = df["y"].map({"yes": 1, "no": 0})

        # Encode categorical variables
        categorical_cols = df.select_dtypes(include="object").columns
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        X = df.drop("y", axis=1)
        y_true = df["y"]

        # Align with training columns
        X = X.reindex(columns=trained_columns, fill_value=0)

        model = models[model_choice]

        # Scale if required
        if model_choice in ["Logistic Regression", "kNN"]:
            X_scaled = scaler.transform(X)
            y_prob = model.predict_proba(X_scaled)[:, 1]
        else:
            y_prob = model.predict_proba(X)[:, 1]

        # Custom threshold (same as training)
        y_pred = (y_prob > 0.4).astype(int)

        # ==========================================
        # Metrics
        # ==========================================

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        mcc = matthews_corrcoef(y_true, y_pred)

        # ==========================================
        # Display Metrics
        # ==========================================

        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Precision", f"{precision:.4f}")

        col2.metric("Recall", f"{recall:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")

        col3.metric("AUC Score", f"{auc:.4f}")
        col3.metric("MCC", f"{mcc:.4f}")

        # ==========================================
        # Confusion Matrix
        # ==========================================

        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)

else:
    st.info("Please upload a test CSV file using the sidebar to evaluate a model.")
