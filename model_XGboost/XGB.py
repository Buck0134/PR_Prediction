import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from joblib import dump
import time

import time

def get_oversample_model(file_path):
    df = pd.read_csv(file_path)
    # Split data into features and target
    X = df.drop('merged_or_not', axis=1)
    y = df['merged_or_not']

    smote = SMOTE(random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_imputed, y_train)

    return X_train_oversampled, y_train_oversampled, X_test_imputed, y_test

def train_xgboost_model(file_path):
    X_train_oversampled, y_train_oversampled, X_test_imputed, y_test = get_oversample_model(file_path)

    start_time = time.time()
    
    parameter_grid = {
        'max_depth': [6, 9, 12, 15],
        'n_estimators': [100, 200, 300, 350],
        'learning_rate': [0.01, 0.2, 0.5, 0.7],

    }

    xgbc = XGBClassifier(use_label_encoder=False, random_state=42 )
    clf = GridSearchCV(xgbc, parameter_grid, n_jobs=-1, cv=3, scoring='f1_weighted',return_train_score=True)
    clf.fit(X_train_oversampled, y_train_oversampled)

    print("Best parameters found for XGBoost:\n", clf.best_params_)

    best_model = clf.best_estimator_

    #  training accuracy
    y_train_pred = best_model.predict(X_train_oversampled)
    training_accuracy = accuracy_score(y_train_oversampled, y_train_pred)

    # Making predictions using the default parameters for comparison
    y_test_pred_default = xgbc.fit(X_train_oversampled, y_train_oversampled).predict(X_test_imputed)
    test_accuracy_default = accuracy_score(y_test, y_test_pred_default)
    precision_default = precision_score(y_test, y_test_pred_default, average="weighted")
    f1_default = f1_score(y_test, y_test_pred_default, average="weighted")

    # Making predictions using the best model from grid search

    y_test_pred_best = best_model.predict(X_test_imputed)
    test_accuracy_best = accuracy_score(y_test, y_test_pred_best)
    precision_best = precision_score(y_test, y_test_pred_best, average="weighted")
    f1_best = f1_score(y_test, y_test_pred_best, average="weighted")

    training_time = time.time() - start_time
    print(f"XGBoost Classifier training took {training_time} seconds.")
    print(f"Training Accuracy: {training_accuracy}")
    print(f"Mean CV Score: {clf.best_score_}")
    print(f"Test Accuracy (Default Parameters): {test_accuracy_default}")
    print(f"Test Precision (Default Parameters): {precision_default}")
    print(f"Test F1 Score (Default Parameters): {f1_default}")
    print(f"Test Accuracy (Best Parameters): {test_accuracy_best}")
    print(f"Test Precision (Best Parameters): {precision_best}")
    print(f"Test F1 Score (Best Parameters): {f1_best}")

    model_path = "_kBest.joblib"
    dump(best_model, model_path)
    print(f"Model saved to {model_path}")

train_xgboost_model("../data/processedDataNew.csv")
# print("============Below is result of pso:===============")
# train_xgboost_model("../data/filteredData_selected_pso.csv")
# print("hello")