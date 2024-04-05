import numpy as np

class Particle:
    def __init__(self, num_features):
        self.position = np.random.rand(num_features) > 0.5  # Random binary position
        self.velocity = np.random.uniform(low=-1, high=1, size=num_features)  # Random velocity
        self.pbest_position = self.position.copy()
        self.pbest_value = -float('inf')
        self.fitness = -float('inf')
        self.feature_selection_time = 0

    def update_personal_best(self):
        if self.fitness > self.pbest_value:
            self.pbest_position = self.position.copy()
            self.pbest_value = self.fitness

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
