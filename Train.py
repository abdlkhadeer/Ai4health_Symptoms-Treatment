#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import data loading and visualization libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random


# import machine learning libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier

# confusion_matrix is in sklearn.metrics, not sklearn.linear_model
from sklearn.metrics import confusion_matrix 
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc


# import warnings
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df = pd.read_csv('Training_Dataset.csv')

# Display the first few rows of the dataframe
df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


len(df['treatment'].unique())


# In[7]:


# Split The Data in Train-Test Split

X = df.drop(columns=['treatment'])  # Features
y = df['treatment']  # Target variable


# In[8]:


le = LabelEncoder()
y_encoded = le.fit_transform(y)


# In[9]:


y_encoded.shape


# In[10]:


# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42) 


# In[20]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# ## Defining a function to evaluate models and display metrics

# In[13]:


# Defining a function to evaluate the models and display metrics
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average='weighted'), 2)  
    recall = round(recall_score(y_test, y_pred, average='weighted'), 2)
    f1 = round(f1_score(y_test, y_pred, average='weighted'), 2)
    
    # Calculate ROC AUC for multi-class
    try:
        if y_pred_proba is not None:
            unique_classes = np.unique(y_test)
            if len(unique_classes) > 2:
                roc_auc = round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 2)  
            else:
                roc_auc = round(roc_auc_score(y_test, y_pred_proba[:, 1]), 2) 
        else:
            roc_auc = "Not Applicable"
    except Exception as e:
        roc_auc = "Not Calculable"
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


# ## Confusion Matrix Plotting Function

# In[14]:


def plot_confusion_matrix(y_test, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


# ## Evaluating The Multiple Models

# In[21]:


# Initializing the models
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "SVC" : SVC(kernel='linear', random_state=42, probability=True), 
}


# Initializing an empty dictionary to store results
results_dict = {}

# Loop through the models and evaluate each
for model_name, model in models.items():
    print(f"\nEvaluating: {model_name}")
    
    # Train and evaluate the model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
     # Store metrics in the results dictionary
    results_dict[model_name] = metrics # Store metrics for each model
    
    # Display metrics
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1 Score: {metrics['f1']}")
    print(f"ROC AUC: {metrics['roc_auc']}")
    
    # Plot confusion matrix
    y_pred = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, model_name)
    


# In[16]:


# Create DataFrame directly from the results_dict
results = pd.DataFrame.from_dict(results_dict, orient='index')

# Resetting index to make 'Model' a column
results = results.reset_index().rename(columns={'index': 'Model'})

results_sorted = results.sort_values(by='roc_auc', ascending=False) 

# Display the comparison table
print("\nModel Comparison:")
print(results_sorted)


# ## Bar Plot Showing The Comparison Bewtween Models and Evaluation Metric

# In[17]:


# Bar Plot for results comparison 
results_sorted.plot(x='Model', y=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'], kind='bar', figsize=(10,6))
plt.title('Model Performance Comparison')
plt.ylabel('Scores')
plt.xticks(rotation=45)
plt.show()


# ## Performing Hyperparameter Tuning To Get The Best Parameter For Each Model 

# In[18]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Hyperparameter grids for each model
param_grids = {
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10, 100],  
        'solver': ['liblinear', 'lbfgs']
    },
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVC': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'], 
        'gamma': ['scale', 'auto']  
    }
}


# In[19]:


# Initializing an empty dictionary to store the best models and their parameters
best_models = {}

# Loop through each model and apply GridSearchCV
for model_name, model in models.items():
    print(f"\nTuning hyperparameters for {model_name}...")
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=param_grids[model_name], 
                               scoring='accuracy',  
                               cv=5,  # 5-fold cross-validation
                               n_jobs=-1)  
    
    # Fit the GridSearchCV object to the data
    grid_search.fit(X_train, y_train)
    
    # Store the best model and its best parameters
    best_models[model_name] = grid_search.best_estimator_
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best accuracy: {grid_search.best_score_:.2f}")


# In[22]:


# Re-evaluating the Logistic Regression model as the best model selection 
def re_evaluate_logistic_regression(best_logreg_model, X_test, y_test):
    # Predict on the test set
    y_pred = best_logreg_model.predict(X_test)
    y_pred_proba = best_logreg_model.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Check if multiclass and calculate ROC AUC
    unique_classes = np.unique(y_test)
    if len(unique_classes) > 2:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    else:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    # Display confusion matrix
    print("\nConfusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.show()

    # Plot ROC Curve
    fpr = {}
    tpr = {}
    for i, class_label in enumerate(unique_classes):
        fpr[class_label], tpr[class_label], _ = roc_curve(y_test == class_label, y_pred_proba[:, i])
        plt.plot(fpr[class_label], tpr[class_label], label=f'Class {class_label} ROC (AUC = {roc_auc:.2f})')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    
    # Positioning the legend outside the plot for clearer visualization
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    
    plt.tight_layout()  
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }

# Get the best Logistic Regression model from the dictionary
best_logreg_model = best_models['Logistic Regression']

# Re-evaluate the tuned Logistic Regression model
print("\nRe-evaluation of Tuned Logistic Regression Model:")
logreg_metrics = re_evaluate_logistic_regression(best_logreg_model, X_test, y_test)

# Print the re-evaluated metrics
print("\nRe-evaluated Metrics for Tuned Logistic Regression:")
print(f"Accuracy: {logreg_metrics['accuracy']:.2f}")
print(f"Precision: {logreg_metrics['precision']:.2f}")
print(f"Recall: {logreg_metrics['recall']:.2f}")
print(f"F1 Score: {logreg_metrics['f1']:.2f}")
print(f"ROC AUC: {logreg_metrics['roc_auc']:.2f}")


# ## Saving the model

# In[23]:


# using joblib to save Logistic Regression model
import joblib

joblib.dump(best_logreg_model, 'logreg_model.pkl')


# ## Using The X_Test Data To Make Predictions on The Saved Model 

# In[25]:


# Load the saved model
loaded_model = joblib.load('logreg_model.pkl')


# In[27]:


# 2d arry convert
X_test.iloc[0].values.reshape(1,-1)


# In[28]:


# pred on the 2d array above to check if our model pred correctly or not

# test 1 :
print('Model Predictions :',best_logreg_model.predict(X_test.iloc[0].values.reshape(1,-1)))
print('Actual Labels :', y_test[0])


# In[34]:


# test 2 :
print('Model Predictions :',best_logreg_model.predict(X_test.iloc[40].values.reshape(1,-1)))
print('Actual Labels :', y_test[40])

