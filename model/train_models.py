# ==========================================
# Bank Marketing Classification Project
# ==========================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ==========================================
# 1. Load Dataset
# ==========================================

data_path = "C:/Users/91703/Desktop/ML Assignment-2/bank-full.csv"
df = pd.read_csv(data_path, sep=";")

# Convert target to binary
df["y"] = df["y"].map({"yes": 1, "no": 0})


# ==========================================
# 2. Encode Categorical Variables
# ==========================================

categorical_cols = df.select_dtypes(include="object").columns

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# ==========================================
# 3. Split Features and Target
# ==========================================

X = df.drop("y", axis=1)
y = df["y"]
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================
# 4. Feature Scaling (Only for LR & kNN)
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")


# ==========================================
# 5. Handle Class Imbalance
# ==========================================

neg, pos = np.bincount(y_train)
scale_pos_weight = neg / pos


# ==========================================
# 6. Initialize Models
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    ),

    "kNN": KNeighborsClassifier(
        n_neighbors=9
    ),

    "Naive Bayes": GaussianNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=80,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42
    )
}


results = []


# ==========================================
# 7. Train and Evaluate
# ==========================================

for name, model in models.items():

    print(f"\nTraining {name}...")

    if name in ["Logistic Regression", "kNN"]:
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

    # Custom threshold (improves recall for imbalance)
    y_pred = (y_prob > 0.4).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        round(accuracy, 4),
        round(auc, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        round(mcc, 4)
    ])

    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, f"model/{filename}")


# ==========================================
# 8. Save Results
# ==========================================

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
)

results_df.to_csv("model/model_results.csv", index=False)

print("\n✅ Training Completed Successfully!\n")
print(results_df)
