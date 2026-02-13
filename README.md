a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether a bank client will subscribe to a term deposit based on marketing campaign data.
The task is a binary classification problem, where:
Target Variable (y):
1 → Client subscribed to term deposit (Yes)
0 → Client did not subscribe (No)
The goal is to evaluate and compare the performance of different machine learning models using multiple evaluation metrics and identify the best-performing model.

The dataset used in this project is the Bank Marketing Dataset from the UCI Machine Learning Repository.


b. Dataset Description

Dataset Details:

Total Instances: 45,211
Total Features: 16 input features
Target Variable: y
Type: Binary Classification

Feature Categories:
Demographic Features: age, job, marital, education
Financial Features: balance, housing, loan, default
Campaign Features: duration, campaign, pdays, previous, poutcome
Contact Features: contact type, day, month
Categorical features were encoded using one-hot encoding and numerical features were retained. Class imbalance was handled using class weighting and threshold adjustment.


c.Models Used

The following 6 machine learning models were implemented and evaluated:
Logistic Regression
Decision Tree
k-Nearest Neighbors (kNN)
Naive Bayes
Random Forest (Ensemble)
XGBoost (Ensemble)

All models were evaluated using the following metrics:
Accuracy
AUC (Area Under ROC Curve)
Precision
Recall
F1 Score
MCC (Matthews Correlation Coefficient)

| ML Model Name            | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
| ------------------------ | -------- | ------ | --------- | ------ | ------ | ------ |
| Logistic Regression      | 0.8047   | 0.9079 | 0.3626    | 0.8828 | 0.5140 | 0.4822 |
| Decision Tree            | 0.8233   | 0.8610 | 0.3834    | 0.8393 | 0.5264 | 0.4864 |
| kNN                      | 0.8935   | 0.8422 | 0.5639    | 0.3960 | 0.4653 | 0.4160 |
| Naive Bayes              | 0.8597   | 0.8231 | 0.4206    | 0.5284 | 0.4684 | 0.3921 |
| Random Forest (Ensemble) | 0.9045   | 0.9367 | 0.6155    | 0.4887 | 0.5448 | 0.4963 |
| XGBoost (Ensemble)       | 0.8550   | 0.9292 | 0.4394    | 0.8667 | 0.5831 | 0.5494 |


| ML Model Name            | Observation about Model Performance                                                                                                                                                                                           |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Logistic Regression      | Achieved very high recall (0.8828), making it effective at identifying customers who subscribe. Suitable when minimizing false negatives is important. However, precision is relatively low, indicating more false positives. |
| Decision Tree            | Provided a balanced trade-off between precision and recall. It is interpretable but slightly lower AUC compared to ensemble methods.                                                                                          |
| kNN                      | Achieved high accuracy but relatively low recall. Performance is affected by high dimensionality after one-hot encoding.                                                                                                      |
| Naive Bayes              | Moderate performance overall. Assumption of feature independence limits its ability to capture complex relationships in the dataset.                                                                                          |
| Random Forest (Ensemble) | Achieved the highest accuracy and highest AUC (0.9367). Demonstrates strong overall predictive power and robustness due to ensemble learning.                                                                                 |
| XGBoost (Ensemble)       | Achieved the highest MCC and F1 score, indicating the best overall balance between precision and recall. Performs strongly on imbalanced data and captures complex patterns effectively.                                      |


Ensemble methods (Random Forest and XGBoost) outperform individual models in terms of predictive performance.
Random Forest achieved the highest AUC and accuracy.
XGBoost achieved the highest MCC and F1 score.
Logistic Regression achieved the highest recall.
Overall, XGBoost provides the best balance across all evaluation metrics and is the most suitable model for this dataset.