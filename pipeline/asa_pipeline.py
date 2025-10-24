# -*- coding: utf-8 -*-
"""
Created on Wed Oct 22 12:46:08 2025

@author: steve
"""

# Import Required Python Packages

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Import Data

my_df = pd.read_csv("data/asa_prediction_data.csv")

# Split Data into Input and Output objects

X = my_df.drop(["asa"], axis = 1)
y = my_df["asa"]

# Split data into Training and Test sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
 

# Specify Numeric and Categorical Features

numeric_features = ["proj_abr", "super_v", "proj_aht", "proj-sl", "net_staff"]
# categorical_features = ["gender"]

#
# Set Up Pipelines
#


# Numerical Feature Transformer

numeric_transformer = Pipeline(steps = [("imputer", SimpleImputer()),
                                        ("scaler", StandardScaler())])

# Catergorical Feature Transformer

# categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "constant", fill_value = "U")),
#                                         ("ohe", OneHotEncoder(handle_unknown = "ignore"))])

# Preprocessing Pipeline

preprocessing_pipeline = ColumnTransformer(transformers = [("numeric", numeric_transformer, numeric_features)])

#
# Apply The Pipeline
#

# Logistic Regression for comparison

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", LogisticRegression(random_state = 42))])

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)

# Random Forest

clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", RandomForestClassifier(random_state = 42))])

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
accuracy_score(y_test, y_pred_class)


# Save the Pipeline

import joblib
joblib.dump(clf, "data/asa_superV_model.joblib")




