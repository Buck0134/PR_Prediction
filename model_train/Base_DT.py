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

for column in df.columns:
    if df[column].dtype == 'object':  # Typically, object dtype implies categorical data
        print(f"{column}: {df[column].unique()}")

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

# tree_clf = DecisionTreeClassifier(random_state=42)
# tree_clf.fit(X_train, y_train)

# y_pred = tree_clf.predict(X_test)

# initial_accuracy = accuracy_score(y_test, y_pred)
# initial_precision = precision_score(y_test, y_pred, average='weighted')
# initial_f1 = f1_score(y_test, y_pred, average='weighted')

# print("Initial Accuracy:", initial_accuracy)
# print("Initial Precision:", initial_precision)
# print("Initial F1 Score:", initial_f1)


# # def train_decision_tree(data_path):
# #     data = pd.read_csv(data_path)
# #     # print(data.columns)

# #     # Prepare the feature matrix X and target vector y
# #     X = data.drop("merged_or_not", axis=1)
# #     y = data["merged_or_not"]

# #     start_time = time.time()

# #     # Splitting the dataset
# #     X_train, X_test, y_train, y_test = train_test_split(
# #         X, y, test_size=0.15, random_state=42, stratify=y
# #     )

# #     # Initialize the decision tree
# #     tree_clf = DecisionTreeClassifier()

# #     # Perform a quick cross-validation
# #     cv_scores = cross_val_score(
# #         tree_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1
# #     )
# #     tree_clf.fit(X_train, y_train)

# #     # Calculate training accuracy
# #     y_train_pred = tree_clf.predict(X_train)
# #     training_accuracy = accuracy_score(y_train, y_train_pred)

# #     # Calculate test accuracy for the default model
# #     y_test_pred_default = tree_clf.predict(X_test)
# #     test_accuracy_default = accuracy_score(y_test, y_test_pred_default)

# #     # Calculate precision and F1 score for the default model
# #     precision_default = precision_score(y_test, y_test_pred_default)
# #     f1_default = f1_score(y_test, y_test_pred_default)

# #     # Set up the grid search for hyperparameter tuning with a simplified parameter grid
# #     param_grid = {
# #         "criterion": ["gini"],
# #         "max_depth": [10, 15, 20],
# #         "min_samples_leaf": [50, 100],
# #     }
# #     grid_search = GridSearchCV(
# #         DecisionTreeClassifier(),
# #         param_grid,
# #         cv=3,
# #         scoring="accuracy",
# #         return_train_score=True,
# #         n_jobs=-1,
# #     )
# #     grid_search.fit(X_train, y_train)

# #     # Finding the best model
# #     best_model = grid_search.best_estimator_

# #     # Calculate test accuracy for the best model
# #     y_test_pred_best = best_model.predict(X_test)
# #     test_accuracy_best = accuracy_score(y_test, y_test_pred_best)

# #     # Calculate precision and F1 score for the best model
# #     precision_best = precision_score(y_test, y_test_pred_best)
# #     f1_best = f1_score(y_test, y_test_pred_best)

# #     # Reporting the metrics
# #     training_time = time.time() - start_time
# #     print(f"Decision Tree training took {training_time} seconds.")
# #     print(f"Training Accuracy: {training_accuracy}")
# #     print(f"Mean CV Score: {np.mean(cv_scores)}")
# #     print(f"Test Accuracy (Default Parameters): {test_accuracy_default}")
# #     print(f"Test Precision (Default Parameters): {precision_default}")
# #     print(f"Test F1 Score (Default Parameters): {f1_default}")
# #     print(f"Test Accuracy (Best Parameters): {test_accuracy_best}")
# #     print(f"Test Precision (Best Parameters): {precision_best}")
# #     print(f"Test F1 Score (Best Parameters): {f1_best}")

# #     model_path = "featureSelected_decision_tree_model_Kbest.joblib"
# #     dump(best_model, model_path)
# #     print(f"Model saved to {model_path}")

# #     return {
# #         "training_accuracy": training_accuracy,
# #         "cv_scores": cv_scores,
# #         "test_accuracy_default": test_accuracy_default,
# #         "precision_default": precision_default,
# #         "f1_default": f1_default,
# #         "test_accuracy_best": test_accuracy_best,
# #         "precision_best": precision_best,
# #         "f1_best": f1_best,
# #     }


# # train_decision_tree("data/processedDataNew.csv")
