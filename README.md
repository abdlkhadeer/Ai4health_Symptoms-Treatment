# Disease Prediction Model Based on Symptoms
This repository contains a machine learning model that predicts the best possible disease based on input symptoms and known disease conditions. The model is built using Random Forest Classifier.

## Features
Input: A set of multiple symptoms.
Output: Suggested disease(s).
Machine Learning Algorithm: Random Forest Classifier.
Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
Model Overview
The model has undergone several steps of training and evaluation. 

## Steps Involved:
Data Collection & Preprocessing: The dataset consists of patient records including symptoms and corresponding disease (diagnoses). Data was cleaned, processed, and prepared for training.

Training Models: The model was trained with Random Forest Classifieir using scikit-learn's API.

Hyperparameter Tuning: The models were tuned using GridSearchCV to find the optimal hyperparameters.

Model Evaluation: Metrics such as Accuracy, Precision, Recall, F1-Score, and ROC-AUC were used for performance evaluation.
