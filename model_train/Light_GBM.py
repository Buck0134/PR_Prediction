from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, f1_score
import pandas as pd
import time
from joblib import dump


def train_lightgbm(train_data_path, test_data_path):
    # Load training data
    train_data = pd.read_csv(train_data_path)
    X_train = train_data.drop("merged_or_not", axis=1)
    y_train = train_data["merged_or_not"]

    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop("merged_or_not", axis=1)
    y_test = test_data["merged_or_not"]

    start_time = time.time()

    lgbm_model = LGBMClassifier(n_estimators=100, learning_rate=0.1)

    # Parameter grid for grid search
    param_grid = {
        "num_leaves": [31, 41],
        "reg_alpha": [0.1, 0.5],
        "min_data_in_leaf": [20, 50, 100],
        "learning_rate": [0.05, 0.1, 0.2],
    }

    # Grid search
    grid_search = GridSearchCV(
        estimator=LGBMClassifier(),
        param_grid=param_grid,
        cv=2,
        scoring="accuracy",
        n_jobs=-1,
        verbose=3,
        return_train_score=True,  # calculate training accuracy
    )

    grid_search.fit(X_train, y_train)

    # Best model from grid search
    best_model = grid_search.best_estimator_

    # Training accuracy
    y_train_pred = best_model.predict(X_train)
    training_accuracy = accuracy_score(y_train, y_train_pred)

    # Making predictions using the default parameters for comparison
    y_test_pred_default = lgbm_model.fit(X_train, y_train).predict(X_test)
    test_accuracy_default = accuracy_score(y_test, y_test_pred_default)
    precision_default = precision_score(y_test, y_test_pred_default, average="macro")
    f1_default = f1_score(y_test, y_test_pred_default, average="macro")

    # Making predictions using the best model from grid search
    y_test_pred_best = best_model.predict(X_test)
    test_accuracy_best = accuracy_score(y_test, y_test_pred_best)
    precision_best = precision_score(y_test, y_test_pred_best, average="macro")
    f1_best = f1_score(y_test, y_test_pred_best, average="macro")

    training_time = time.time() - start_time
    print(f"LightGBM training took {training_time} seconds.")
    print(f"Training Accuracy: {training_accuracy}")
    print(f"Mean CV Score: {grid_search.best_score_}")
    print(f"Test Accuracy (Default Parameters): {test_accuracy_default}")
    print(f"Test Precision (Default Parameters): {precision_default}")
    print(f"Test F1 Score (Default Parameters): {f1_default}")
    print(f"Test Accuracy (Best Parameters): {test_accuracy_best}")
    print(f"Test Precision (Best Parameters): {precision_best}")
    print(f"Test F1 Score (Best Parameters): {f1_best}")

    model_path = "lightgbm_PSO.joblib"
    dump(best_model, model_path)
    print(f"Model saved to {model_path}")


train_lightgbm("data/processedDataNew.csv", "data/processedDataNew_test.csv")
