from particle import Particle
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time

class PSO:
    def __init__(self, num_particles, num_features, X_train, X_test, y_train, y_test, num_iterations, w=0.5, c1=0.8, c2=0.9):
        self.num_particles = num_particles
        self.particles = [Particle(num_features) for _ in range(num_particles)]
        self.gbest_position = np.random.rand(num_features) > 0.5
        self.gbest_value = -float('inf')
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_iterations = num_iterations
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive coefficient
        self.c2 = c2  # Social coefficient

    def evaluate_fitness(self, particle):
        print("start evaluate_fitness inside pso.optimize()")
        start_time = time.time()
        selected_features = particle.position > 0.5
        if not np.any(selected_features):  # Avoid having no selected features
            particle.fitness = -float('inf')
            return
        X_train_selected = self.X_train[:, selected_features]
        X_test_selected = self.X_test[:, selected_features]
        feature_selection_time = time.time() - start_time

        model = DecisionTreeClassifier()
        print("evaluate_fitness model fitting...")
        model.fit(X_train_selected, self.y_train)

        print("evaluate_fitness model prediction...")
        predictions = model.predict(X_test_selected)
        particle.fitness = accuracy_score(self.y_test, predictions)
        particle.feature_selection_time = feature_selection_time

    def update_velocity_position(self, particle):
        for i in range(len(particle.velocity)):
            r1, r2 = np.random.rand(), np.random.rand()

            # Convert boolean positions to integers for the subtraction operation
            pbest_pos_int = particle.pbest_position.astype(int)
            current_pos_int = particle.position.astype(int)
            gbest_pos_int = self.gbest_position.astype(int)

            cognitive_velocity = self.c1 * r1 * (pbest_pos_int[i] - current_pos_int[i])
            social_velocity = self.c2 * r2 * (gbest_pos_int[i] - current_pos_int[i])

            particle.velocity[i] = self.w * particle.velocity[i] + cognitive_velocity + social_velocity

            # Update position using sigmoid to map velocity to [0, 1], then threshold
            particle.position[i] = particle.sigmoid(particle.velocity[i]) > np.random.rand()


    def optimize(self, pbar=None):
        for iteration in range(self.num_iterations):
            print("Iteration: ", iteration + 1)
            for particle in self.particles:
                self.evaluate_fitness(particle)
                particle.update_personal_best()
                if particle.fitness > self.gbest_value:
                    self.gbest_position = particle.position.copy()
                    self.gbest_value = particle.fitness

            for particle in self.particles:
                self.update_velocity_position(particle)

            total_feature_selection_time = sum(p.feature_selection_time for p in self.particles)
            if pbar is not None:
                pbar.update(1)
        
            # Print the progress at each iteration
            # print(f"Iteration {iteration + 1}/{self.num_iterations} - Best fitness so far: {self.gbest_value:.4f}")
            
            # Optionally, print the global best position (selected features) for each iteration
            # Convert binary positions to feature indices for readability
            selected_features_indices = [index for index, value in enumerate(self.gbest_position) if value > 0.5]
            # print(f"Selected features at iteration {iteration + 1}: {selected_features_indices}")
        
        num_selected_features = np.sum(self.gbest_position > 0.5)
        # After all iterations, print the final best solution
        # print("\nOptimization Finished.")
        # print(f"Best fitness achieved: {self.gbest_value:.4f}")
        # print(f"Best feature subset: {selected_features_indices}")
        # print(f"Feature selection time: {total_feature_selection_time:.4f} seconds")
        # print("\n")
        return self.gbest_value, selected_features_indices, num_selected_features, total_feature_selection_time

