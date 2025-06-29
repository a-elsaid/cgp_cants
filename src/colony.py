from ant import Ant
from search_space import Space
import numpy as np
from graph import Graph
from typing import List, Dict
from util import function_dict, function_names
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
from timeseries import Timeseries
import pickle


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
                    data: Timeseries,
                    num_itrs: int = 10,
                    worker_id: int = None,
    ):
        self.id = Colony.count + 1
        Colony.count += 1
        self.num_itrs = num_itrs
        self.num_ants = num_ants
        self.data = data
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
        logger.info(f"Colony({self.id}) (Worker_{worker_id}):: Created with {num_ants} ants and {population_size} population size")

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
        graph = Graph(ants_paths = paths, space=self.space, colony_id=self.id)
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

    def life(self, num_itrs=None, total_itrs=None, cost_type="mse"):
        if num_itrs:
            self.num_itrs = num_itrs
        patience = 10
        for itr in range(self.num_itrs):
            logger.info(f"Colony({self.id}): Iteration: {self.life_count}{f'/{total_itrs}' if total_itrs else ''}")
            self.life_count+=1
            if self.boost_exploration:
                space_is_not_explored = self.check_explored_space()
                if space_is_not_explored:
                    logger.info(f"Colony({self.id}): Space is not fully explored: Boosting Exploration")
                    logger.info(f"Colony({self.id}): Setting Evaporation Rate to 0.999: ON HOLD FOR NOW")
                    # self.evaporation_rate = 0.999
                else:
                    logger.info(f"Colony({self.id}): Space is fully explored: Resetting Exploration")
                    self.boost_exploration = False
                    self.evaporation_rate = self.original_evaporation_rate
            graph = self.ants_go(increase_exploration=self.boost_exploration)
            

            fit, _ = graph.evaluate(self.data, cost_type=cost_type)
            wait = 10
            while fit > 5 and wait > 0:
                wait-=1
                logger.info(f"Colony({self.id}): Fitness is None, ReGenerating & ReEvaluating Graph {wait} times left")
                graph = self.ants_go(increase_exploration=self.boost_exploration)
                fit, _ = graph.evaluate(self.data)
                
            logger.info(f"Colony({self.id}) - Graph({graph.id}): Fitness: {fit}")
            inserted = self.insert_to_population(fit, graph)

            self.evolve_ants(fit)
            if inserted:
                self.update_scores()
                self.space.deposited_pheromone(graph)
                graph.visualize_graph(f"colony_{self.id}_graph_{graph.id}_fit_{fit}.gv")
                graph.generate_eqn(f"colony_{self.id}_graph_{graph.id}_fit_{fit}.eqn")
                graph.write_structure(f"colony_{self.id}_graph_{graph.id}_fit_{fit}.strct")
                fig = plt.figure(figsize=(40, 40))
                ax = fig.add_subplot(111, projection='3d')
                # graph.plot_path_points(ax=ax, plt=plt)
                graph.plot_paths(ax=ax, plt=plt)
                # graph.plot_nodes(ax=ax, plt=plt)
                graph.plot_pheromones(ax=ax, plt=plt)

                function_colors = {
                    0: 'red',
                    1: 'blue',
                    2: 'green',
                    3: 'orange',
                    4: 'purple',
                    5: 'brown',
                    6: 'cyan',
                    7: 'magenta',
                    8: 'gold',
                }

                # Manually create legend entries for whatever you added
                custom_legend_items = [
                    Line2D([0], [0], marker='*', linestyle='None', markeredgecolor='red', markerfacecolor='red', markersize=80, label='Node'),
                    Line2D([0], [0], linestyle='-', color='gray', label='Ant Path', linewidth=6)
                    # Line2D([0], [0], marker='o', linestyle='None', markeredgecolor='gray', markerfacecolor='gray', markersize=80, label='Space Point'),
                ]
                
                for i, func_name in enumerate(function_names.values()):
                    custom_legend_items.append(
                        Line2D([0], [0], marker='o', linestyle='None', markeredgecolor=function_colors[i], markerfacecolor=function_colors[i], markersize=80, label=func_name)
                    )

                # Add the custom legend
                ax.legend( 
                            handles=custom_legend_items,
                            loc='upper left',
                            fontsize=40,               # Increase font size
                            handlelength=4,            # Length of the legend handles
                            borderpad=1.0,             # Padding inside the legend box
                            labelspacing=1.2,          # Space between labels
                            handletextpad=1.5,         # Space between handle and text
                            frameon=True,              # Show legend frame
                            framealpha=1.0,            # Opaque box
                            borderaxespad=1.5          # Padding between legend and axes
                )
                plt.savefig(f"colony_{self.id}_graph_{graph.id}_fit_{fit}.png")
                plt.cla(); plt.clf(); plt.close()
                graph.plot_target_predict(data=self.data, file_name=f"colony_{self.id}_graph_{graph.id}_fit_{fit}_target_predict", cost_type=cost_type)
                self.save_graph(graph, f"colony_{self.id}_graph_{graph.id}_fit_{fit}.graph")
                
                fig = plt.figure(figsize=(40, 40))
                ax = fig.add_subplot(111, projection='3d')
                graph.plot_nodes(ax=ax, plt=plt)
                plt.savefig(f"colony_{self.id}_nn_{graph.id}_fit_{fit}.png")
                plt.cla(); plt.clf(); plt.close()

            self.space.add_new_points(graph.added_points)
            self.space.add_input_points(graph.added_in_points)


            self.space.evaporate_pheromone(self.evaporation_rate)

            '''
            Resting Colony's Evaporation Rate to Max
            if no better graphs are found
            '''
            if (not self.boost_exploration) and (not inserted):
                logger.info(f"Colony({self.id}): No Improvemnet->Resetting Evaporation Rate")
                patience-=1
            if patience == 0:
                self.boost_exploration = True
                patience = 10
            prev_inserted = inserted


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


    def save_graph(self, graph, name):
        with open(name, "wb") as f:
            pickle.dump(graph, f)

    def load_graph(self, name):
        with open(name, "rb") as f:
            return pickle.load(f)