# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifiers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the Titanic dataset from seaborn as an example dataset
df = sns.load_dataset('titanic')

# Inspect for missing values
print("Missing values per column:\n", df.isnull().sum())

# Handle missing values
# For simplicity, fill missing 'age' with the median, 'embark_town' with the mode, and drop 'deck' due to high missing values
df['age'].fillna(df['age'].median(), inplace=True)
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)
df.drop(columns=['deck'], inplace=True)

# Identify outliers in 'fare' and 'age' using box plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['age']).set_title("Age Outliers")
plt.subplot(1, 2, 2)
sns.boxplot(y=df['fare']).set_title("Fare Outliers")
plt.show()

# Handle outliers by capping to percentiles
q1_age, q3_age = df['age'].quantile([0.05, 0.95])
q1_fare, q3_fare = df['fare'].quantile([0.05, 0.95])
df['age'] = np.clip(df['age'], q1_age, q3_age)
df['fare'] = np.clip(df['fare'], q1_fare, q3_fare)

# Use Min-Max scaling for 'age' and 'fare'
scaler = MinMaxScaler()
df[['age', 'fare']] = scaler.fit_transform(df[['age', 'fare']])

# Create family_size and title features
df['family_size'] = df['sibsp'] + df['parch']
df['title'] = df['name'].apply(lambda x: x.split(",")[1].split(".")[0].strip() if pd.notnull(x) else "Unknown")

# Drop columns that may be less useful
df.drop(columns=['name', 'ticket', 'embarked'], inplace=True)

# For correlation analysis, convert categorical columns to numeric where necessary
df = pd.get_dummies(df, columns=['sex', 'class', 'embark_town', 'title'], drop_first=True)

# Calculate feature importance using Random Forest for selection
X = df.drop('survived', axis=1)
y = df['survived']
model = RandomForestClassifiers(random_state=42)
model.fit(X, y)
importances = model.feature_importances_
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print("Feature importances:\n", feature_importances)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred = log_reg.predict(X_test)
print("Logistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
