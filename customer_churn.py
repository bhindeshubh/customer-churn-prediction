# -*- coding: utf-8 -*-
"""customer-churn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1we_sTMXSTrNMzdf8DmlMrZvRPjFkPKiG

# Customer Churn Prediction Model for the Telecommunication Industry

## Importing Necessary Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore') # This line is to hide the warnings

"""## Importing Data"""

data = pd.read_csv('Telco-Customer-Churn.csv') # Loading the data
data.head() # First 5 rows

"""## Exploratory Data Analysis (EDA)"""

# No. of rows & columnw
print(f'No. of Rows: {data.shape[0]} \nNo. of Columns: {data.shape[1]}')

# Information about data
data.info()

# Data Description
data.describe(include='all').fillna('-')

# Checking if there are any duplicate values
data.duplicated().sum()

# Checking if there are any null values
data.isnull().sum()

# Checking for outliers in Monthly Charges column (Since it is the only numerical column)
sns.boxplot(data.MonthlyCharges)

# Checking if it is normally distributed (if not, perform feature scaling)
plt.hist(x=data.MonthlyCharges)
plt.show()

# Creating a copy of data
data1 = data.copy()

# Data Preprocessing

# Label Encoding
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
categorical_columns = data1.select_dtypes(include=['object']).columns
for column in categorical_columns:
    data1[column] = labelencoder.fit_transform(data1[column])

data1

final_data = data1.copy() # Final data to be used for prediction

final_data.head()

final_data.to_csv('final_data.csv') # Downloading the data

"""## XGBoost Algorithm"""

# Importing XG Boost
from xgboost import XGBClassifier

# X: Features, Y: Target Variable
X, Y = final_data.drop(['customerID','Churn'], axis=1), final_data.Churn

from sklearn.model_selection import train_test_split, cross_val_score, KFold # Necessary functions

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.3)

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

"""## Cross Validation"""

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.001],
    'n_estimators': [100, 500, 1000]
}

# Perform Grid Search Cross Validation to find the best hyperparameters
grid_search = GridSearchCV(xgb, param_grid, cv=5)
grid_search.fit(x_train, y_train)

# Get the best model from the grid search
best_xgb = grid_search.best_estimator_

"""## Model Building"""

# Train the model using the training sets
model = best_xgb.fit(x_train, y_train)

# Predict the response for the test dataset
y_pred = best_xgb.predict(x_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

"""## Downloading the trained model"""

import joblib
joblib.dump(model, 'model.pkl')