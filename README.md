# heart_disease_predict
# Heart Disease Prediction Project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load the Dataset
df = pd.read_csv('C:/Users/Gnan Tejas D/OneDrive/Desktop/heart_disease.csv')
print("First 5 rows of data:\n", df.head())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# 2. Feature Engineering
# Encode 'Gender' and 'Heart Disease' columns
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male = 1, Female = 0
df['Heart Disease'] = df['Heart Disease'].map({'Yes': 1, 'No': 0})

# Separate features and target
X = df[['Age', 'Gender', 'Cholesterol', 'Blood Pressure']]
y = df['Heart Disease']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Model Training
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Accuracy, Precision, Recall, F1-Score
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Full Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
