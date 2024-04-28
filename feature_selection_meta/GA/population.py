from individual import Individual
import random
import numpy as np

class Population:
    def __init__(self, size, mutation_rate, crossover_rate, num_features, X_train, X_test, y_train, y_test, max_generation):
        self.population = []
        self.size = size
        self.generations = 0
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.best_ind = None
        self.finished = False
        self.perfect_score = 1.0
        self.average_fitness = 0.0
        self.mating_pool = []
        self.num_features = num_features
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.max_generations = max_generation
        self.total_feature_selection_time = 0

    def create_initial_population(self):
        print("Creating initial population...")
        for index in range(self.size):
            print(f"Creating individual {index + 1} of {self.size}")
            ind = Individual(self.num_features, self.X_train, self.X_test, self.y_train, self.y_test)
            self.population.append(ind)
        self.evaluate_population()

    def evolve(self, pbar=None):
        self.mating_pool = self._create_mating_pool()
        self._generate_new_population()
        self.evaluate_population()
        self.generations += 1
        if self.generations >= self.max_generations:
            self.finished = True
        if pbar is not None:
            pbar.update(1)

    def _create_mating_pool(self):
        mating_pool = []
        max_fitness = max([ind.fitness for ind in self.population])
        for ind in self.population:
            fitness_normal = ind.fitness / max_fitness
            n = int(fitness_normal * 100)  # Arbitrary multiplier to increase selection chances
            mating_pool.extend([ind for _ in range(n)])
        return mating_pool

    def _generate_new_population(self):
        new_population = []
        for _ in range(self.size):
            parent_a = random.choice(self.mating_pool)
            parent_b = random.choice(self.mating_pool)
            child = parent_a.crossover(parent_b, self.crossover_rate)
            child.mutate(self.mutation_rate)
            new_population.append(child)
        self.population = new_population

    def evaluate_population(self):
        self.average_fitness = np.mean([ind.fitness for ind in self.population])
        self.best_ind = max(self.population, key=lambda ind: ind.fitness)
        self.total_feature_selection_time = sum(ind.feature_selection_time for ind in self.population)
        if self.best_ind.fitness >= self.perfect_score:
            self.finished = True

    def print_population_status(self):
        print(f"Generation: {self.generations}")
        print(f"Average fitness: {self.average_fitness:.4f}")
        print("Best individual fitness:", self.best_ind.fitness)
