from pso import PSO
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import time
from tqdm import tqdm


def load_and_prepare_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    print("Finished loading data")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    print("Split finished")
    merge_distribution = {"Not Merged": list(y).count(0), "Merged": list(y).count(1)}
    feature_names = df.columns[:-1].tolist()

    return X_train, X_test, y_train, y_test, merge_distribution, feature_names


def evaluate_model(
    X_train, X_test, y_train, y_test, selected_features_indices, feature_names
):
    # Ensure at least one feature is selected
    if len(selected_features_indices) == 0:
        return 0, 0, 0, 0

    # Select features for training and testing sets
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Train and predict with Decision Tree Classifier
    model = DecisionTreeClassifier()
    model.fit(X_train_selected, y_train)
    predictions = model.predict(X_test_selected)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, pos_label=1)
    recall = recall_score(y_test, predictions, pos_label=1)
    f1 = f1_score(y_test, predictions, pos_label=1)

    selected_feature_names = [feature_names[i] for i in selected_features_indices]

    return accuracy, precision, recall, f1, selected_feature_names


def main():

    datasets = {"data": load_and_prepare_dataset("data/filteredData.csv")}

    num_iterations = [5, 10, 15]

    for dataset_name, dataset_data in datasets.items():
        X_train, X_test, y_train, y_test, merged_distribution, feature_names = (
            dataset_data
        )
        num_features = X_train.shape[1]

        (
            accuracy_list,
            precision_list,
            recall_list,
            f1_list,
            time_list,
            num_selected_features_list,
            feature_selection_time_list,
        ) = ([], [], [], [], [], [], [])
        for run_index in range(3):  # Perform three runs for each iteration count
            with tqdm(
                total=num_iterations[run_index],
                desc=f"PSO optimization for {dataset_name} with {num_iterations[run_index]} iterations",
            ) as pbar:
                print("initializing PSO...")
                pso = PSO(
                    num_particles=20,
                    num_features=num_features,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    num_iterations=num_iterations[run_index],
                )
                print("PSO initialized, start pso.optimize...")
                (
                    _,
                    selected_features_indices,
                    num_selected_features,
                    feature_selection_time,
                ) = pso.optimize(pbar=pbar)
                print("pso.optimize finished")
                run_time = feature_selection_time

            accuracy, precision, recall, f1, selected_feature_names = evaluate_model(
                X_train,
                X_test,
                y_train,
                y_test,
                selected_features_indices,
                feature_names,
            )
            print(f"Selected features for run {run_index+1}: {selected_feature_names}")

            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            time_list.append(run_time)
            num_selected_features_list.append(num_selected_features)
            feature_selection_time_list.append(feature_selection_time)

        # Calculate averages
        avg_results = {
            "accuracy": sum(accuracy_list) / len(accuracy_list),
            "precision": sum(precision_list) / len(precision_list),
            "recall": sum(recall_list) / len(recall_list),
            "f1": sum(f1_list) / len(f1_list),
            "time": sum(time_list) / len(time_list),
            "num_selected_features": sum(num_selected_features_list)
            / len(num_selected_features_list),
            "feature_selection_time": sum(feature_selection_time_list)
            / len(feature_selection_time_list),
        }

        print(f"Results for {dataset_name}:")
        print(f"Avg_accuracy: {avg_results['accuracy']:.4f}")
        print(f"Avg_precision: {avg_results['precision']:.4f}")
        print(f"Avg_recall: {avg_results['recall']:.4f}")
        print(f"Avg_f1: {avg_results['f1']:.4f}")
        print(f"Avg_time: {avg_results['time']:.4f}s")
        print(f"Avg_num_selected_features: {avg_results['num_selected_features']:.4f}")
        print(
            f"Avg_feature_selection_time: {avg_results['feature_selection_time']:.4f}s"
        )
        print("===========================================================")


if __name__ == "__main__":
    main()
