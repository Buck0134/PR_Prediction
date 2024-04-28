from population import Population
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import time
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

def load_and_prepare_dataset(file_path):
    print(f"Loading dataset from {file_path}")
    df = pd.read_csv(file_path)

    print("Finished loading data")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    print("Split finished")
    merge_distribution = {'Not Merged': list(y).count(0), 'Merged': list(y).count(1)}
    feature_names = df.columns[:-1].tolist()

    return X_train, X_test, y_train, y_test, merge_distribution, feature_names

def genetic_algorithm(run_index, max_generation, pop_size, mutation_rate, crossover_rate, num_features, X_train, X_test, y_train, y_test):
    pop = Population(pop_size, mutation_rate, crossover_rate, num_features, X_train, X_test, y_train, y_test, max_generation)
    pop.create_initial_population()
    print(f"Initial population of size {pop_size} created.")  # New print statement
    with tqdm(total=max_generation, desc=f"Run {run_index + 1}: Generations") as pbar:
        while not pop.finished and pop.generations < max_generation:
            print(f"Generation {pop.generations + 1}/{max_generation} processing...")  # New print statement
            pop.evolve(pbar)
    print("Genetic algorithm finished. Evaluating best individual...")  # New print statement
    return pop.best_ind.genes, pop.best_ind.fitness, pop.best_ind, pop.generations, pop


def evaluate_model(X_train, X_test, y_train, y_test, selected_genes, feature_names):
    selected_features_indices = [i for i, gene in enumerate(selected_genes) if gene == 1]
    if len(selected_features_indices) == 0:
        return 0, 0, 0, 0, []

    # Select features based on the genes
    X_train_selected = X_train[:, selected_features_indices]
    X_test_selected = X_test[:, selected_features_indices]

    # Train and predict with the model
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

def run_genetic_algorithm(n_runs, max_generations, feature_names, **kwargs):
    all_features = []
    all_fitness = []
    all_times = []
    all_feature_selection_times = []
    all_generations = []
    model_metrics = []

    for run_index in range(n_runs):
        print("Running genetic algorithm for run", run_index + 1)
        start_time = time.time()
        selected_genes, best_fitness, best_individual, generations, pop = genetic_algorithm(run_index=run_index, max_generation=max_generations[run_index], **kwargs)
        print("genetic algo finished")
        elapsed_time = time.time() - start_time

        feature_selection_time = pop.total_feature_selection_time
        num_selected_features = np.sum(selected_genes)
        if num_selected_features == 0:
            continue

        accuracy, precision, recall, f1, selected_feature_names = evaluate_model(kwargs['X_train'], kwargs['X_test'], kwargs['y_train'], kwargs['y_test'], selected_genes, feature_names=feature_names)
        print(f"Selected features for run {run_index+1}: {selected_feature_names}")
        all_features.append(num_selected_features)
        all_fitness.append(best_fitness)
        all_times.append(elapsed_time)
        all_feature_selection_times.append(feature_selection_time)
        all_generations.append(generations)
        model_metrics.append((accuracy, precision, recall, f1))
            

    avg_features = np.mean(all_features)
    avg_fitness = np.mean(all_fitness)
    avg_time = np.mean(all_times)
    avg_feature_selection_time = np.mean(all_feature_selection_times)
    avg_generations = np.mean(all_generations)

    avg_accuracy, avg_precision, avg_recall, avg_f1 = np.mean(model_metrics, axis=0)

    metrics = {
        'avg_features': avg_features,
        'avg_fitness': avg_fitness,
        'avg_time': avg_time,
        'avg_feature_selection_time': avg_feature_selection_time,
        'avg_generations': avg_generations,
        'avg_accuracy': avg_accuracy,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
    }

    return metrics

def main():

    # Default parameters
    default_pop_size = 20
    default_mutation_rate = 0.01
    default_crossover_rate = 0.6

    # Parameter ranges
    mutation_rates = [0.005, 0.02]
    population_sizes = [200, 300]
    crossover_rates = [0.5, 0.7]

    n_runs = 3
    max_generations = [5,10,15]

    X_train, X_test, y_train, y_test, merge_distribution, feature_names = load_and_prepare_dataset('../../data/filteredData.csv')

    num_features = X_train.shape[1]
    print('merge_distribution:', merge_distribution)

    avg_metrics = run_genetic_algorithm(
                n_runs=n_runs,
                max_generations=max_generations,
                pop_size=default_pop_size,
                mutation_rate=default_mutation_rate,
                crossover_rate=default_crossover_rate,
                num_features=num_features,
                feature_names=feature_names,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test
            )
    print_results(avg_metrics)

    # # Test different mutation rates
    # for mutation_rate in mutation_rates:
    #     print('running run_genetic_algorithm with mutation_rate:', mutation_rate)
    #     avg_metrics = run_genetic_algorithm(
    #             n_runs=n_runs,
    #             max_generations=max_generations,
    #             pop_size=default_pop_size,
    #             mutation_rate=mutation_rate,
    #             crossover_rate=default_crossover_rate,
    #             num_features=num_features,
    #             X_train=X_train,
    #             X_test=X_test,
    #             y_train=y_train,
    #             y_test=y_test
    #         )
    #     print(f"Results for with Mutation Rate: {mutation_rate}")
    #     print_results(avg_metrics)

    #     # Test different population sizes
    # for pop_size in population_sizes:
    #     avg_metrics = run_genetic_algorithm(
    #             n_runs=n_runs,
    #             max_generations=max_generations,
    #             pop_size=pop_size,
    #             mutation_rate=default_mutation_rate,
    #             crossover_rate=default_crossover_rate,
    #             num_features=num_features,
    #             X_train=X_train,
    #             X_test=X_test,
    #             y_train=y_train,
    #             y_test=y_test
    #         )
    #     print(f"Results for with Population Size: {pop_size}")
    #     print_results(avg_metrics)

    # # Test different crossover rates
    # for crossover_rate in crossover_rates:
    #     avg_metrics = run_genetic_algorithm(
    #             n_runs=n_runs,
    #             max_generations=max_generations,
    #             pop_size=default_pop_size,
    #             mutation_rate=default_mutation_rate,
    #             crossover_rate=crossover_rate,
    #             num_features=num_features,
    #             X_train=X_train,
    #             X_test=X_test,
    #             y_train=y_train,
    #             y_test=y_test
    #         )
    #     print(f"Results for with Crossover Rate: {crossover_rate}")
    #     print_results(avg_metrics)

def print_results(metrics):
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")
    print("=====================================================")


if __name__ == "__main__":
    main()
