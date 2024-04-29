from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from joblib import dump


# print(data.columns)
df = pd.read_csv("../data/processedData.csv")

# print(len(df.columns))
# print(df['last_close_time'])
# print("==========")

if 'last_close_time' in df.columns:
    df['last_close_time'] = pd.to_datetime(df['last_close_time'], format='%d/%m/%Y %H:%M:%S')
    df['last_close_year'] = df['last_close_time'].dt.year
    df['last_close_month'] = df['last_close_time'].dt.month
    df['last_close_day'] = df['last_close_time'].dt.day
    df['last_close_hour'] = df['last_close_time'].dt.hour
    df.drop('last_close_time', axis=1, inplace=True) 

# print(len(df.columns))
# print(df['last_close_hour'])

# check for categorical cols
# for column in df.columns:
#     if df[column].dtype == 'object':  # Typically, object dtype implies categorical data
#         print(f"{column}: {df[column].unique()}")

df['contrib_gender'] = df['contrib_gender'].map({'Missing': 0, 'male': 1, 'female': 2})

column_transformer = ColumnTransformer([
    ('country_encoder', OneHotEncoder(handle_unknown='ignore'), ['contrib_country']),
    ('emo_encoder', OneHotEncoder(), ['contrib_first_emo']),
    ('lang_encoder', OneHotEncoder(), ['language']),
    ('ci_first_status_encoder', OneHotEncoder(), ['ci_first_build_status']),
    ('ci_last_status_encoder', OneHotEncoder(), ['ci_last_build_status'])
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocessor', column_transformer),
    ('classifier', DecisionTreeClassifier(random_state=42))
])

# Prepare the feature matrix X and target vector y
X = df.drop("merged_or_not", axis=1)
y = df["merged_or_not"]

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='weighted')  
print("F1 Score:", f1)
