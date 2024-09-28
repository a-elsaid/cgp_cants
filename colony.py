from ant import Ant
from search_space import Space
import numpy as np


function_dict = {0: 'add', 1: 'subtract', 2: 'multiply', 3: 'divide'}

class Colony():
    def __init__(self, space: Space, num_ants: int, function_dict: dict):
        self.space = space
        self.ants = [Ant(space) for _ in range(num_ants)]
        self.best_solutions = {}
        self.best_score = float('inf')


    def ants_go(self):
        paths = []
        for ant in self.ants:
            # Each ant explores the search space
            ant.march()
            paths.append(ant.get_path())
        return paths

    def evaluate_ant(self, ant: Ant):
        # Implement the evaluation logic for the ant's solution
        # This is a placeholder implementation
        return np.random.uniform(0, 1)

    def update_pheromones(self):
        # Implement pheromone update logic
        pass

    def run(self, iterations: int):
        for _ in range(iterations):
            self.evaluate()
            self.update_pheromones()  # Update pheromones after evaluating all ants


    # Call the merge_points method in the run method
    def run(self, iterations: int):
        for _ in range(iterations):
            self.evaluate()
            self.update_pheromones()
            self.merge_points()