import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Load the dataset
url = 'https://docs.google.com/spreadsheets/d/1bN2C5iD8uNG4BQrtXE5WNpGBhEdEjpcWBEJ8alT1oks/export?format=csv'
df = pd.read_csv(url)

# Data Exploration and Preprocessing
# Check for missing values and fill them with the column mean (or use a more suitable method as needed)
df.fillna(df.mean(), inplace=True)

# Encode categorical columns if necessary
df = pd.get_dummies(df, drop_first=True)

# Convert date to datetime and extract the year
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Split data based on time periods
train_data = df[df['year'] < df['year'].max()]  # First 2 years for training
test_data = df[df['year'] == df['year'].max()]  # Last year for testing

X_train = train_data.drop(['Churn', 'date', 'year'], axis=1)
y_train = train_data['Churn']
X_test = test_data.drop(['Churn', 'date', 'year'], axis=1)
y_test = test_data['Churn']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Initial Model Training: Logistic Regression
log_model = LogisticRegression(max_iter=200, random_state=42)
log_model.fit(X_train, y_train)

# Evaluate Logistic Regression model on test data
y_pred_log = log_model.predict(X_test)
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall:", recall_score(y_test, y_pred_log))
print("F1 Score:", f1_score(y_test, y_pred_log))
print("AUC-ROC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# Handling Concept Drift with Time-Weighted Learning
weights = np.linspace(1, 2, len(y_train))  # Higher weights for recent data
log_model.fit(X_train, y_train, sample_weight=weights)

# Evaluate Time-Weighted Logistic Regression on test data
y_pred_weighted_log = log_model.predict(X_test)
print("\nTime-Weighted Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_weighted_log))
print("Precision:", precision_score(y_test, y_pred_weighted_log))
print("Recall:", recall_score(y_test, y_pred_weighted_log))
print("F1 Score:", f1_score(y_test, y_pred_weighted_log))
print("AUC-ROC:", roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1]))

# Ensemble Model: Combine Logistic Regression and Decision Tree
tree_model = DecisionTreeClassifier(random_state=42)
ensemble = VotingClassifier(estimators=[('log', log_model), ('tree', tree_model)], voting='soft')
ensemble.fit(X_train, y_train)

# Evaluate Ensemble Model on test data
y_pred_ensemble = ensemble.predict(X_test)
print("\nEnsemble Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Precision:", precision_score(y_test, y_pred_ensemble))
print("Recall:", recall_score(y_test, y_pred_ensemble))
print("F1 Score:", f1_score(y_test, y_pred_ensemble))
print("AUC-ROC:", roc_auc_score(y_test, ensemble.predict_proba(X_test)[:, 1]))

print("\nAll steps completed. Model is evaluated for potential concept drift and data shifts.")
