import random
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import time

class Individual:
    def __init__(self, num_features, X_train, X_test, y_train, y_test):
        self.num_features = num_features
        self.genes = self.generate_random_genes(num_features)
        self.fitness = 0
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.calculate_fitness()

    @staticmethod
    def generate_random_genes(size):
        return [random.randint(0, 1) for _ in range(size)]

    def calculate_fitness(self):
        start_time = time.time()
        selected_features = [i for i, gene in enumerate(self.genes) if gene == 1]

        # Ensure there's at least one feature selected; if not, randomly select one feature
        if not selected_features:
            random_feature = random.randint(0, self.num_features - 1)
            selected_features = [random_feature]
            self.genes[random_feature] = 1  # Update the genes to reflect this random selection

        X_train_selected = self.X_train[:, selected_features]
        X_test_selected = self.X_test[:, selected_features]

        model = DecisionTreeClassifier()
        model.fit(X_train_selected, self.y_train)
        predictions = model.predict(X_test_selected)
        end_time = time.time()
        self.feature_selection_time = end_time - start_time
        self.fitness = accuracy_score(self.y_test, predictions)
        
    def mutate(self, mutation_rate):
        for i in range(self.num_features):
            if random.random() < mutation_rate:
                self.genes[i] = 1 - self.genes[i]
    
    def crossover(self, partner, crossover_rate=0.6):
        child = Individual(self.num_features, self.X_train, self.X_test, self.y_train, self.y_test)
        for i in range(self.num_features):
            if random.random() < crossover_rate:
                child.genes[i] = self.genes[i]
            else:
                child.genes[i] = partner.genes[i]
        return child
    
