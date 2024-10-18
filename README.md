# Treatment Prediction Model Based on Symptoms
This repository contains a machine learning model that predicts the best possible treatment based on input symptoms and known disease conditions. The model is built using several classifiers, including Logistic Regression, Decision Trees, Random Forest, and Support Vector Machine (SVC), with Logistic Regression selected as the final model after hyperparameter tuning.

## Features
Input: A set of multiple symptoms.
Output: Suggested treatment(s).
Machine Learning Algorithm: Logistic Regression (after tuning for best performance).
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Model Overview
The model has undergone several steps of training and evaluation using various classifiers. After hyperparameter tuning, Logistic Regression provided the most optimal results, thus being chosen as the final model.

## Steps Involved:
Data Collection & Preprocessing: The dataset consists of patient records including symptoms and corresponding diagnoses and treatments. Data was cleaned, processed, and prepared for training.

Training Models: Multiple models were trained (Logistic Regression, Decision Tree, Random Forest, and SVC) using scikit-learn's API.

Hyperparameter Tuning: The models were tuned using GridSearchCV to find the optimal hyperparameters.

Model Evaluation: Metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC were used for performance evaluation.

Final Model: Logistic Regression was selected based on its performance after tuning.
