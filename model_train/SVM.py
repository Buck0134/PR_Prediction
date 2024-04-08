from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, f1_score
import numpy as np
import pandas as pd
import time
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline


def train_svm(data_path):
    data = pd.read_csv(data_path)
    X = data.drop("merged_or_not", axis=1)
    y = data["merged_or_not"]

    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    imputer = SimpleImputer(strategy="mean")

    svm_clf = make_pipeline(imputer, SVC())

    # cross-validation
    cv_scores = cross_val_score(
        svm_clf, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1
    )
    svm_clf.fit(X_train, y_train)
    # training accurcy
    y_train_pred = svm_clf.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)

    # test accuracy for  default model
    y_test_pred_default = svm_clf.predict(X_test)
    test_accuracy_default = accuracy_score(y_test, y_test_pred_default)

    # precision and F1 score for the default model
    precision_default = precision_score(y_test, y_test_pred_default)
    f1_default = f1_score(y_test, y_test_pred_default)

    # grid search for hyperparameter tuning
    param_grid = {
        "C": [1, 10],
        "gamma": ["scale"],
        "kernel": ["linear"],
    }

    grid_search = GridSearchCV(
        SVC(), param_grid, cv=2, scoring="accuracy", return_train_score=True, n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    # the best model
    best_model = grid_search.best_estimator_

    # test accuracy for the best model
    y_test_pred_best = best_model.predict(X_test)
    test_accuracy_best = accuracy_score(y_test, y_test_pred_best)

    # precision and F1 score for the best model
    precision_best = precision_score(y_test, y_test_pred_best)
    f1_best = f1_score(y_test, y_test_pred_best)

    # metrics
    training_time = time.time() - start_time
    print(f"SVM training took {training_time} seconds.")
    print(f"Training Accuracy: {training_accuracy}")
    print(f"Mean CV Score: {np.mean(cv_scores)}")
    print(f"Test Accuracy (Default Parameters): {test_accuracy_default}")
    print(f"Test Precision (Default Parameters): {precision_default}")
    print(f"Test F1 Score (Default Parameters): {f1_default}")
    print(f"Test Accuracy (Best Parameters): {test_accuracy_best}")
    print(f"Test Precision (Best Parameters): {precision_best}")
    print(f"Test F1 Score (Best Parameters): {f1_best}")

    model_path = "featureSelected_svm_Kbest.joblib"
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


train_svm("data/processedDataNew.csv")
