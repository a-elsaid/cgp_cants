from ant import Ant
from search_space import Space
import numpy as np
from graph import Graph
from typing import List, Dict
from util import function_dict, function_names
from matplotlib import pyplot as plt
import sys

import loguru
logger = loguru.logger
logger.remove()
logger.add(sys.stdout, level="TRACE")

class Colony():
    count = 0
    def __init__(   self,
                    num_ants: int, 
                    population_size: int, 
                    input_names: List[str], 
                    output_names: List[str],
                    train_input: np.ndarray,
                    train_target: np.ndarray,
                    test_input: np.ndarray = None,
                    test_target: np.ndarray = None,
                    num_itrs: int = 10,
    ):
        self.id = Colony.count + 1
        Colony.count += 1
        self.num_itrs = num_itrs
        self.num_ants = num_ants
        self.train_input = train_input
        self.train_target = train_target
        self.test_input = test_input
        self.test_target = test_target
        self.best_solutions = []
        self.best_score = None
        self.avg_col_score = None
        self.bst_col_score = None
        self.boost_exploration = True
        self.mortality_rate = np.random.uniform(0.1, 0.5)
        self.evaporation_rate = np.random.uniform(0.1, 0.5)
        self.original_evaporation_rate = self.evaporation_rate  # Store the original evaporation rate
        self.population_size = population_size
        self.life_count = 0
        self.space = Space(
                            input_names=input_names, 
                            output_names=output_names, 
                            evap_rate=self.evaporation_rate
        )
        self.ants = [Ant(self.space) for _ in range(num_ants)]
        
        self.pso_position = [self.num_ants, self.mortality_rate, self.evaporation_rate]
        self.pso_velocity = np.random.uniform(low=-1, high=1, size=len(self.pso_position))
        self.pso_best_position = self.pso_position
        self.pso_bounds = [[5, 20], [0.01, 0.1], [0.15, 0.95]] # Number of ants, mortality rate, evaporation rate

    def set_evaporation_rate(self, rate):
        self.evaporation_rate = rate
        self.space.evaporation_rate = rate
        
    def check_explored_space(self):
        no_passed = False
        prev_i = 0
        steps = list(np.linspace(0, 1, 5))[1:]
        for i in steps:
            prev_j = 0
            for j in steps:
                prev_k = 0
                for k in steps:
                    prev_l = 0
                    for l in steps:
                        no_passed = True
                        for p in self.space.points:
                            if (
                                (prev_i <= p.get_x() <= i) and 
                                (prev_j <= p.get_y() <= j) and 
                                (prev_k <= p.get_z() <= k) and 
                                (prev_l <= p.get_f() <= l)
                            ):
                                no_passed = False
                                break
                        if no_passed:
                            break
                        prev_l = l
                    if no_passed:
                        break
                    prev_k = k
                if no_passed:
                    break
                prev_j = j
            if no_passed:
                break
            prev_i = i
        return no_passed

    def ants_go(self, increase_exploration=False):
        paths = []
        for ant in self.ants:
            ant.reset()
            if increase_exploration:
                ant.explore_rate = 0.999
            ant.march()
            paths.append(ant.path)
            self.space.add_new_points(ant.new_points)
            self.space.add_input_points(ant.new_in_points)
        graph = Graph(ants_paths = paths, space=self.space)
        return graph


    def update_scores(self):
        self.avg_col_score = np.mean([x[0] for x in self.best_solutions])
        self.bst_col_score = self.best_solutions[0][0]



    def insert_to_population(self, score, solution):
        inserted = False
        if len(self.best_solutions) < self.population_size:
            self.best_solutions.append([score, solution])
            inserted = True
        elif score > self.best_solutions[0][0]:
            self.best_solutions[-1] = [score, solution]
            inserted = True
        self.best_solutions.sort(key=lambda x: x[0])
        return inserted

    def evolve_ants(self, fit):
        for ant in self.ants:
            ant.update_best_behaviors(fit)
            ant.evolve_behavior()

    def life(self, num_itrs=None, total_itrs=None):
        if num_itrs:
            self.num_itrs = num_itrs
        for itr in range(self.num_itrs):
            logger.info(f"Colony({self.id}): Iteration: {self.life_count}{f'/{total_itrs}' if total_itrs else ''}")
            self.life_count+=1
            if self.boost_exploration:
                space_is_not_explored = self.check_explored_space()
                if space_is_not_explored:
                    logger.info(f"Colony({self.id}): Space is not fully explored: Boosting Exploration")
                    self.evaporation_rate = 0.999
                else:
                    logger.info(f"Colony({self.id}): Space is fully explored: Resetting Exploration")
                    self.boost_exploration = False
                    self.evaporation_rate = self.original_evaporation_rate
            graph = self.ants_go(increase_exploration=self.boost_exploration)
            

            fig = plt.figure(figsize=(40, 40))
            ax = fig.add_subplot(111, projection='3d')
            
            graph.plot_path_points(ax=ax, plt=plt)
            graph.plot_nodes(ax=ax, plt=plt)
            graph.plot_pheromones(ax=ax, plt=plt)
            plt.savefig(f"colony_{self.id}_graph_{graph.id}.png")
            plt.cla()
            plt.clf()
            plt.close()
            graph.visualize_graph(f"colony_{self.id}_graph_{graph.id}")  
            '''
            '''
            

            fit, _ = graph.evaluate(self.train_input, self.train_target)
            fit = np.mean(fit)
            logger.info(f"Colony({self.id}): Fitness: {fit}")
            inserted = self.insert_to_population(fit, graph)
            self.evolve_ants(fit)
            if inserted:
                self.update_scores()
                self.space.deposited_pheromone(graph)
            self.space.add_new_points(graph.added_points)
            self.space.add_input_points(graph.added_in_points)


            self.space.evaporate_pheromone(self.evaporation_rate)


    def get_col_fit(self, rank=None, avg:bool=False) -> float:
        """return the population best fitness"""
        self.update_best_colony_score(rank, avg)
        if avg:
            return self.avg_population_fit, self.pso_best_position
        else:
            return self.bst_col_score, self.pso_best_position



    def update_best_colony_score(self, rank=None, avg:bool=True) -> None:
        best_solutions = np.array(self.best_solutions)
        logger.trace(f"Worker({rank}:: Collecting Fitnees from Colony({self.id})")
        best_scores = best_solutions[:, 0]
        best_score  = np.sort(best_scores)[0]

        '''Get avg of colony fits as measure of overall colony-fit'''
        avg_col_score = sum(best_scores) / len(best_scores) 


        if self.avg_col_score is None or avg_col_score < self.avg_col_score:
            self.avg_col_score = avg_col_score
            if avg:
                self.pso_best_position = self.pso_position

        if self.bst_col_score is None or best_score < self.bst_col_score:
            self.bst_col_score = best_score
            if not avg:
                self.pso_best_position = self.pso_position

    def update_velocity(self, pos_best_g):
        """update new particle velocity"""

        logger.info(f"COLONY({self.id}):: Updating Colony PSO velocity")

        w = 0.5  # constant inertia weight (how much to weigh the previous velocity)
        c1 = 1.7  # cognative constant
        c2 = 1.7  # social constant

        for i, pos in enumerate(self.pso_position):
            r1 = np.random.random()
            r2 = np.random.random()
            vel_cognitive = c1 * r1 * (self.pso_best_position[i] - pos)
            vel_social = c2 * r2 * (pos_best_g[i] - pos)
            self.pso_velocity[i] = w * self.pso_velocity[i] + vel_cognitive + vel_social

    def update_position(self):
        """update the particle position based off new velocity updates"""
        logger.info(f"COLONY({self.id}):: Updating Colony PSO position")
        
        self.num_ants+= self.pso_velocity[0]
        self.num_ants = int(self.num_ants)
        self.mortality_rate+= self.pso_velocity[1]
        self.evaporation_rate+= self.pso_velocity[2]

        if self.num_ants < self.pso_bounds[0][0] or self.num_ants > self.pso_bounds[0][1]:
            self.num_ants = np.random.randint(
                low=self.pso_bounds[0][0], high=self.pso_bounds[0][1]
            )
            
        if self.mortality_rate < self.pso_bounds[1][0] or self.mortality_rate > self.pso_bounds[1][1]:
            self.mortality_rate = np.random.uniform(
                low=self.pso_bounds[1][0], high=self.pso_bounds[1][1]
            )
            
        if self.evaporation_rate < self.pso_bounds[2][0] or self.evaporation_rate > self.pso_bounds[2][1]:
            self.evaporation_rate = np.random.uniform(
                low=self.pso_bounds[2][0], high=self.pso_bounds[2][1]
            )
            
        self.pso_position[0] = self.num_ants
        self.pso_position[1] = self.mortality_rate
        self.pso_position[2] = self.evaporation_rate

        '''
        for i, pos in enumerate(self.pso_position):
            self.pso_position[i] = pos + self.pso_velocity[i]

            # adjust position if necessary
            if pos < self.pso_bounds[i][0] or pos > self.pso_bounds[i][1]:
                if i < 1:
                    self.pso_position[i] = np.random.randint(
                        low=self.pso_bounds[i][0], high=self.pso_bounds[i][1]
                    )
                else:
                    self.pso_position[i] = np.random.uniform(
                        low=self.pso_bounds[i][0], high=self.pso_bounds[i][1]
                    )
        '''

        self.space.evaporation_rate = self.evaporation_rate
