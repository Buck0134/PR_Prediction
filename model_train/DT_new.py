from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import pandas as pd
import time

from joblib import dump


def train_decision_tree(train_data_path, test_data_path):
    # data = pd.read_csv(data_path)
    # print(data.columns)

    # Prepare the feature matrix X and target vector y
    train_data = pd.read_csv(train_data_path) 
    X_train = train_data.drop("merged_or_not", axis=1)
    y_train = train_data["merged_or_not"]
    # X = data.drop("merged_or_not", axis=1)
    # y = data["merged_or_not"]

    test_data = pd.read_csv(test_data_path) 
    X_test = test_data.drop("merged_or_not", axis=1)
    y_test = test_data["merged_or_not"]

    categorical_columns = X_train.select_dtypes(include=['object', 'category']).columns

    X_train = X_train.drop(categorical_columns, axis=1)
    X_test = X_test.drop(categorical_columns, axis=1)

    print("Categorical columns:")
    print(categorical_columns)

    start_time = time.time()

    # Splitting the dataset
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.15, random_state=42
    # )

    # Initialize the decision tree
    tree_clf = DecisionTreeClassifier()

    # Perform a quick cross-validation
    cv_scores = cross_val_score(
        tree_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1
    )
    tree_clf.fit(X_train, y_train)

    # Calculate training accuracy
    y_train_pred = tree_clf.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)

    # Calculate test accuracy for the default model
    y_test_pred_default = tree_clf.predict(X_test)
    test_accuracy_default = accuracy_score(y_test, y_test_pred_default)

    # Calculate precision and F1 score for the default model
    precision_default = precision_score(y_test, y_test_pred_default)
    f1_default = f1_score(y_test, y_test_pred_default)

    # Set up the grid search for hyperparameter tuning with a simplified parameter grid
    param_grid = {
        "criterion": ["gini"],
        "max_depth": [10, 15, 20],
        "min_samples_leaf": [50, 100],
    }
    grid_search = GridSearchCV(
        DecisionTreeClassifier(),
        param_grid,
        cv=3,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # Finding the best model
    best_model = grid_search.best_estimator_

    # Calculate test accuracy for the best model
    y_test_pred_best = best_model.predict(X_test)
    test_accuracy_best = accuracy_score(y_test, y_test_pred_best)

    # Calculate precision and F1 score for the best model
    precision_best = precision_score(y_test, y_test_pred_best)
    f1_best = f1_score(y_test, y_test_pred_best)

    # Reporting the metrics
    training_time = time.time() - start_time
    print(f"Decision Tree training took {training_time} seconds.")
    print(f"Training Accuracy: {training_accuracy}")
    print(f"Mean CV Score: {np.mean(cv_scores)}")
    print(f"Test Accuracy (Default Parameters): {test_accuracy_default}")
    print(f"Test Precision (Default Parameters): {precision_default}")
    print(f"Test F1 Score (Default Parameters): {f1_default}")
    print(f"Test Accuracy (Best Parameters): {test_accuracy_best}")
    print(f"Test Precision (Best Parameters): {precision_best}")
    print(f"Test F1 Score (Best Parameters): {f1_best}")

    model_path = "decision_tree_base.joblib"
    dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    return {
        "training_accuracy": training_accuracy,
        "cv_scores": cv_scores,
        "test_accuracy_default": test_accuracy_default,
        "precision_default": precision_default,
        "f1_default": f1_default,
        "test_accuracy_best": test_accuracy_best,
        "precision_best": precision_best,
        "f1_best": f1_best,
    }


train_decision_tree("data/processedData.csv", "data/processedData_test.csv")
