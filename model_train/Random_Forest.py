from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import pandas as pd
import time
from joblib import dump


def train_random_forest(data_path):
    data = pd.read_csv(data_path)

    X = data.drop("merged_or_not", axis=1)
    y = data["merged_or_not"]

    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    random_classic = RandomForestClassifier(n_estimators=20, min_samples_leaf=60)
    cv_scores = cross_val_score(
        random_classic, X_train, y_train, cv=2, scoring="accuracy", n_jobs=-1
    )

    param_grid = {
        "max_depth": [None],
        "min_samples_leaf": [60],
        "n_estimators": [20],
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(),
        param_grid,
        cv=2,
        scoring="accuracy",
        return_train_score=True,
        n_jobs=-1,
        verbose=3,
    )

    grid_search.fit(X_train, y_train)

    # Extracting and using the best model from grid search
    best_model = grid_search.best_estimator_

    # Default model predictions (Before Grid Search)
    random_classic.fit(X_train, y_train)
    y_test_pred_default = random_classic.predict(X_test)
    test_accuracy_default = accuracy_score(y_test, y_test_pred_default)
    precision_default = precision_score(y_test, y_test_pred_default)
    f1_default = f1_score(y_test, y_test_pred_default)

    # Best model predictions (After Grid Search)
    y_test_pred_best = best_model.predict(X_test)
    test_accuracy_best = accuracy_score(y_test, y_test_pred_best)
    precision_best = precision_score(y_test, y_test_pred_best)
    f1_best = f1_score(y_test, y_test_pred_best)

    training_time = time.time() - start_time
    print(f"Random Forest training took {training_time} seconds.")
    print(f"Training Accuracy: {np.mean(cv_scores)}")
    print(f"Mean CV Score: {np.mean(cv_scores)}")
    print(f"Test Accuracy (Default Parameters): {test_accuracy_default}")
    print(f"Test Precision (Default Parameters): {precision_default}")
    print(f"Test F1 Score (Default Parameters): {f1_default}")
    print(f"Test Accuracy (Best Parameters): {test_accuracy_best}")
    print(f"Test Precision (Best Parameters): {precision_best}")
    print(f"Test F1 Score (Best Parameters): {f1_best}")

    model_path = "random_forest_kBest.joblib"
    dump(best_model, model_path)
    print(f"Model saved to {model_path}")


train_random_forest("data/processedDataNew.csv")