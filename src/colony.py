# includes:
#  - Single-threaded life() loop
#  - A newer threaded life_threads() loop using ThreadPoolExecutor
#  - Utilities to save/load graphs and expose the best model
from ant import Ant
from search_space import Space
import numpy as np
from graph import Graph
from util import function_dict, function_names
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
from timeseries import Timeseries
import pickle
from typing import List

import multiprocessing as mp  # currently unused (kept for optional process mode)

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED  # threaded parallelism
import threading  # locks and thread naming

# from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
# import os

import loguru

# ----------------------------- logging setup ----------------------------------
logger = loguru.logger
logger.remove()
logger.add(sys.stdout, level="INFO")

# ---------------------- (optional) process worker -----------------------------
# This helper function shows how a child process could evaluate a Graph and
# dump artifacts. Crrent code uses threads (life_threads()).
def _worker(graph: Graph, data, cost_type: str, colony_id: int):
        """
        Child process:
        - evaluate the graph
        - dump its files (.gv, .eqn, .strct, .png, .graph)
        - return (fit, graph_filename, pickled_added_pts, pickled_added_in_pts)
        """
        # 1) evaluate the candidate graph on 'data'
        fit, _ = graph.evaluate(data, cost_type=cost_type)

        # 2) dump artifacts with a conventional name carrying IDs and fitness
        base = f"colony_{colony_id}_graph_{graph.id}_fit_{fit:.6f}"
        graph.visualize_graph(f"{base}.gv")
        graph.generate_eqn(f"{base}.eqn")
        graph.write_structure(f"{base}.strct}")
        graph.plot_target_predict(data=data, file_name=f"{base}_target_predict", cost_type=cost_type)

        # 3) serialize the full graph; master can reload it later
        fname = f"{base}.graph"
        with open(fname, "wb") as f:
            pickle.dump(graph, f)

        # 4) return lists of points so the master can merge space updates
        added_pts    = graph.get_added_points()     # List[Point]
        added_in_pts = graph.get_added_in_points()  # List[Point]
        return float(fit), fname, added_pts, added_in_pts

# ------------------------------- Colony class ---------------------------------
class Colony():
    count = 0  # class-level counter for unique colony IDs

    def __init__(   self,
                    num_ants: int, 
                    population_size: int, 
                    input_names: List[str], 
                    output_names: List[str],
                    data: Timeseries,
                    num_itrs: int = 10,
                    worker_id: int = None,
                    out_dir: str = "./OUT",
                    use_torch: bool = True,
    ):
        # ----- book-keeping / configuration -----
        self.id = Colony.count + 1
        Colony.count += 1
        self.num_itrs = num_itrs                  # how many iterations to run inside life()/life_threads()
        self.num_ants = num_ants                  # ants per iteration
        self.data = data                          # dataset wrapper
        self.best_solutions = []                  # list of [fitness, Graph] (sorted ascending by fitness)
        self.best_score = None                    # (legacy; not used elsewhere)
        self.avg_col_score = None                 # cache of average fitness across population
        self.bst_col_score = None                 # cache of best (lowest) fitness in population
        self.boost_exploration = True             # flag to push exploration early on
        self.mortality_rate = np.random.uniform(0.1, 0.5)   # parameter that affects ant/graph survival
        self.evaporation_rate = np.random.uniform(0.1, 0.5) # pheromone evaporation rate in Space
        self.original_evaporation_rate = self.evaporation_rate  # to restore after boosting
        self.population_size = population_size    # cap on best_solutions length
        self.life_count = 0                       # iteration counter
        self.__use_torch = use_torch              # Graphs can run torch ops if True

        # ----- pheromone space & ants -----
        self.space = Space(
                            input_names=input_names, 
                            output_names=output_names, 
                            evap_rate=self.evaporation_rate
        )
        self.ants = [Ant(self.space) for _ in range(num_ants)]  # build the swarm
        
        # ----- PSO state for meta-params (optional tuning across iterations) -----
        self.pso_position = [self.num_ants, self.mortality_rate, self.evaporation_rate]
        self.pso_velocity = np.random.uniform(low=-1, high=1, size=len(self.pso_position))
        self.pso_best_position = self.pso_position
        # bounds: [num_ants_min,max], [mortality_min,max], [evaporation_min,max]
        self.pso_bounds = [[5, 20], [0.01, 0.1], [0.15, 0.95]]

        logger.info(f"Colony({self.id}) (Worker_{worker_id}):: Created with {num_ants} ants and {population_size} population size")

        # ----- threading support -----
        self._lock = threading.Lock()   # protects shared state (Space, best_solutions, etc.)
        self.out_dir = out_dir          # directory to write artifacts

    # Pickle hooks: thread locks are not picklable, so remove/recreate them.
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_lock']  # remove unpicklable lock
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()  # recreate the lock

    # Update evaporation rate consistently in both colony and Space
    def set_evaporation_rate(self, rate):
        self.evaporation_rate = rate
        self.space.evaporation_rate = rate
        
    # Coarse coverage check: splits [0,1]^4 into bins and verifies at least one visited point per bin.
    def check_explored_space(self):
        no_passed = False
        prev_i = 0
        steps = list(np.linspace(0, 1, 5))[1:]  # [0.25, 0.5, 0.75, 1.0]
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

    # March all ants (optionally cranking explore_rate), aggregate their paths, and build a Graph.
    def ants_forage(self, increase_exploration=False):
        with self._lock:    # ensure thread-safe updates to Space and shared lists
            paths = []
            for ant in self.ants:
                ant.reset()
                if increase_exploration:
                    ant.explore_rate = 0.999  # push exploration extremely high
                ant.march()                   # build a path from input to output
                paths.append(ant.path)
                # incorporate new discovered points into global Space
                self.space.add_new_points(ant.new_points)
                self.space.add_input_points(ant.new_in_points)
        # build a candidate graph from all ants paths
        graph = Graph(ants_paths = paths, space=self.space, colony_id=self.id, use_torch=self.__use_torch)
        return graph

    # Cache average and best fitness across population (assuming best_solutions sorted)
    def update_scores(self):
        self.avg_col_score = np.mean([x[0] for x in self.best_solutions])
        self.bst_col_score = self.best_solutions[0][0]

    # Insert a (fitness, graph) into population if there is room or it improves worst.
    # Preserves original logic (condition uses '>'); not changed.
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

    # Let ants update their behavior archive and self-evolve (mutate/crossover)
    def evolve_ants(self, fit):
        for ant in self.ants:
            ant.update_best_behaviors(fit)
            ant.evolve_behavior()

    # Persist graph artifacts; optionally produce heavy 3D plots and target-vs-predict plots.
    def _dump_graph(self, graph, fit, cost_type="mse", plot=False):
        
        graph.visualize_graph(f"{self.out_dir}/colony_{self.id}_graph_{graph.id}_fit_{fit}.gv")
        graph.generate_eqn(f"{self.out_dir}/colony_{self.id}_graph_{graph.id}_fit_{fit}.eqn")
        self.save_graph(graph, f"{self.out_dir}/colony_{self.id}_graph_{graph.id}_fit_{fit}.graph")
        if not plot:
            return
        graph.write_structure(f"{self.out_dir}/colony_{self.id}_graph_{graph.id}_fit_{fit}.strct")
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

        # Build a custom legend with consistent styling for saved figures
        custom_legend_items = [
            Line2D([0], [0], marker='*', linestyle='None', markeredgecolor='red', markerfacecolor='red', markersize=80, label='Node'),
            Line2D([0], [0], linestyle='-', color='gray', label='Ant Path', linewidth=6)
            # Line2D([0], [0], marker='o', linestyle='None', markeredgecolor='gray', markerfacecolor='gray', markersize=80, label='Space Point'),
        ]

        for i, func_name in enumerate(function_names.values()):
            custom_legend_items.append(
                Line2D([0], [0], marker='o', linestyle='None', markeredgecolor=function_colors[i], markerfacecolor=function_colors[i], markersize=80, label=func_name)
            )

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
        plt.savefig(f"{self.out_dir}/colony_{self.id}_graph_{graph.id}_fit_{fit}.png")
        plt.cla(); plt.clf(); plt.close()
        graph.plot_target_predict(data=self.data, file_name=f"{self.out_dir}/colony_{self.id}_graph_{graph.id}_fit_{fit}_target_predict", cost_type=cost_type)

        fig = plt.figure(figsize=(40, 40))
        ax = fig.add_subplot(111, projection='3d')
        graph.plot_nodes(ax=ax, plt=plt)
        plt.savefig(f"{self.out_dir}/colony_{self.id}_nn_{graph.id}_fit_{fit}.png")
        plt.cla(); plt.clf(); plt.close()

    # Build a graph and evaluate it once (used by the threaded scheduler).
    def _gen_and_eval(self, increase_exploration: bool, cost_type: str, train_epochs: int = 10):
        """
        Build a graph once, evaluate it, return (fit, graph).
        """
        thd = threading.current_thread()
        logger.info(f"Colony({self.id:2d}) -- Thread({thd.name}) -- Generating and Evaluating Graph")
        graph = self.ants_forage(increase_exploration=increase_exploration)  # build from ants' paths
        th_id = threading.current_thread().name
        fit, _ = graph.evaluate(self.data, cost_type=cost_type, num_epochs=train_epochs, thread_id=th_id)
        return fit, graph

    # Threaded life-cycle that fills N "slots" concurrently and retries weak graphs up to 10×.
    def life_threads(self, num_itrs=None, total_itrs=None, cost_type="mse", train_epochs=10):
        
        if num_itrs:
            self.num_itrs = num_itrs

        executor = ThreadPoolExecutor(max_workers=num_itrs, thread_name_prefix=f"Colony{self.id:2d}-Thread")
        # each slot is one final graph you want; keeps trying until it passes
        pending = []           # list of futures
        attempts = dict()      # future -> how many times we tried

        # 1) launch one task per iteration slot
        for slot in range(self.num_itrs):
            fut = executor.submit(self._gen_and_eval, self.boost_exploration, cost_type, train_epochs=train_epochs)
            pending.append(fut)
            attempts[fut] = 1

        done_results = []  # will hold (fit, graph) pairs that passed

        # 2) as soon as any future completes, check it:
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in done:
                pending.remove(fut)
                count = attempts.pop(fut)
                fit, graph = fut.result()

                # if it’s too weak and we haven’t retried 10× yet → re-submit
                if fit > 5 and count < 10:
                    new_fut = executor.submit(self._gen_and_eval, self.boost_exploration, cost_type)
                    pending.append(new_fut)
                    attempts[new_fut] = count + 1
                else:
                    # it’s good (or we’re out of retries)
                    done_results.append((fit, graph))

        # 3) now process them in arrival order (or sort by fitness, whatever)
        for fit, graph in done_results:
            logger.info(f"Colony({self.id}) - Graph({graph.id}) fitness={fit:.4e}")

            # *** shared‐state updates under a lock ***
            with self._lock:
                inserted = self.insert_to_population(fit, graph)
                self.evolve_ants(fit)
                self.space.evaporate_pheromone(self.evaporation_rate)
                self.space.add_new_points(graph.added_points)
                self.space.add_input_points(graph.added_in_points)
                if inserted:
                    self.update_scores()
                    self.space.deposited_pheromone(graph)
                    self._dump_graph(graph, fit, cost_type=cost_type)

        # 4) clean up (cancel or harvest remaining futures)
        for fut in pending:
            # if it’s still running, cancel it
            if not fut.done():
                fut.cancel()
            else:
                # if it’s done, we can still get the result
                try:
                    fut.result()
                except Exception as e:
                    logger.error(f"Colony({self.id}): Error in future: {e}")

        executor.shutdown()
    
    """
    def life_processes(self, num_itrs=None, total_itrs=None, cost_type="mse"):
        if num_itrs:
            self.num_itrs = num_itrs

        executor = ProcessPoolExecutor(max_workers=self.num_itrs)
        futures  = {}

        # build & submit
        for _ in range(self.num_itrs):
            graph = self.ants_forage(increase_exploration=self.boost_exploration)
            fut = executor.submit(_worker, graph, self.data, cost_type, self.id)
            futures[fut] = 1

        done = []

        # retry or collect
        while futures:
            finished, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in finished:
                tries = futures.pop(fut)
                fit, fname, added_pts, added_in_pts = fut.result()

                if fit > 5 and tries < 10:
                    graph = self.ants_forage(increase_exploration=self.boost_exploration)
                    new_fut = executor.submit(_worker, graph, self.data, cost_type, self.id)
                    futures[new_fut] = tries + 1
                else:
                    done.append((fit, fname, added_pts, added_in_pts))

        # merge back
        for fit, fname, added_pts, added_in_pts in done:
            # load the full Graph for population insertion & I/O
            graph = self.load_graph(fname)

            self.space.add_new_points(added_pts)
            self.space.add_input_points(added_in_pts)
            self.space.evaporate_pheromone(self.evaporation_rate)

            # standard ACO updates
            inserted = self.insert_to_population(fit, graph)
            self.evolve_ants(fit)
            self.space.evaporate_pheromone(self.evaporation_rate)
            if inserted:
                self.update_scores()
                self.space.deposited_pheromone(graph)
                # dumps already done in worker

        executor.shutdown(wait=True)
    """
    
    # Original single-threaded loop: generate/evaluate graphs sequentially with a patience schedule.
    def life(self, num_itrs=None, total_itrs=None, cost_type="mse", train_epochs=10):
        if num_itrs:
            self.num_itrs = num_itrs
        patience = 10
        for itr in range(self.num_itrs):
            logger.info(f"Colony({self.id}): Iteration: {self.life_count+1}{f'/{total_itrs}' if total_itrs else ''}")
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
            graph = self.ants_forage(increase_exploration=self.boost_exploration)
            

            fit, _ = graph.evaluate(self.data, cost_type=cost_type, num_epochs=train_epochs)
            wait = 10
            while fit > 5 and wait > 0:
                wait-=1
                logger.info(f"Colony({self.id}): Fitness is None, ReGenerating & ReEvaluating Graph {wait} times left")
                graph = self.ants_forage(increase_exploration=self.boost_exploration)
                fit, _ = graph.evaluate(self.data)
                
            logger.info(f"Colony({self.id}) - Graph({graph.id}): Fitness: {fit}")
            inserted = self.insert_to_population(fit, graph)

            self.evolve_ants(fit)
            if inserted:
                self.update_scores()
                self.space.deposited_pheromone(graph)
                self._dump_graph(graph, fit, cost_type=cost_type)

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

    # Interface used by MPI/PSO orchestration to fetch colony fitness and position
    def get_col_fit(self, rank=None, avg:bool=False) -> float:
        """return the population best fitness"""
        self.update_best_colony_score(rank, avg)
        if avg:
            return self.avg_population_fit, self.pso_best_position
        else:
            return self.bst_col_score, self.pso_best_position

    # Update cached metrics; also update PSO personal-best depending on avg/best mode.
    def update_best_colony_score(self, rank=None, avg:bool=True) -> None:
        best_solutions = np.array(self.best_solutions)
        logger.trace(f"Worker({rank}:: Collecting Fitnees from Colony({self.id})")
        best_scores = best_solutions[:, 0]
        best_score  = np.sort(best_scores)[0]

        '''Get avg of colony fits as measure of overall colony-fit'''
        avg_col_score = sum(best_scores) / len(best_scores) 

        # record best/avg, and update pso_best_position based on which metric is used
        if self.avg_col_score is None or avg_col_score < self.avg_col_score:
            self.avg_col_score = avg_col_score
            if avg:
                self.pso_best_position = self.pso_position

        if self.bst_col_score is None or best_score < self.bst_col_score:
            self.bst_col_score = best_score
            if not avg:
                self.pso_best_position = self.pso_position

    # PSO velocity update step (classic w/c1/c2 and random factors r1,r2)
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

    # PSO position update step with bounds and consistency with Space’s evaporation rate
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

        self.space.evaporation_rate = self.evaporation_rate  # keep Space in sync

    # Robust serialization with a small diagnostic: prints non-picklable attributes
    def save_graph(self, graph, name):
        for attr, value in graph.__dict__.items():
            try:
                pickle.dumps(value)
            except Exception as e:
                print(f"Cannot pickle attribute '{attr}': {type(value)} — {e}")
        with open(name, "wb") as f:
            pickle.dump(graph, f)

    # Load a pickled Graph back to memory
    def load_graph(self, name):
        with open(name, "rb") as f:
            return pickle.load(f)

    # Convenience accessors for the best candidate
    def get_best_graph(self):
        """Return the current best Graph (lowest fitness) or None if empty."""
        if not self.best_solutions:
            return None
        try:
            self.best_solutions.sort(key=lambda x: x[0])  # lower fitness is better
        except Exception:
            pass
        try:
            return self.best_solutions[0][1]
        except Exception:
            return None

    def get_best_model(self):
        """Alias for get_best_graph(), for downstream code that expects 'model'."""
        return self.get_best_graph()

    def get_best_fitness(self):
        """Return the best (lowest) fitness value or None if unknown."""
        if not self.best_solutions:
            return None
        try:
            self.best_solutions.sort(key=lambda x: x[0])
        except Exception:
            pass
        try:
            return float(self.best_solutions[0][0])
        except Exception:
            return None
