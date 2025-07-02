"""
to run the colonies in parallel and evolve them
using PSO
"""
import sys

import pickle
import threading as th
import numpy as np
import loguru
from colony import Colony
from timeseries import Timeseries
from helper import Args_Parser
from search_space import Space
from mpi4py import MPI
logger = loguru.logger


comm_mpi = MPI.COMM_WORLD
comm_size = comm_mpi.Get_size()
rank = comm_mpi.Get_rank()

worker_group = np.arange(1,comm_size)
num_colonies = comm_size - 1

def create_colony(data=None, living_time=None):
    num_ants = np.random.randint(low=1, high=20)
    population_size = np.random.randint(low=5, high=25)
    evaporation_rate = np.random.uniform(low=0.7, high=0.9)
    colony = Colony(
                    num_ants=num_ants, 
                    population_size=population_size, 
                    input_names=data.input_names,
                    output_names=data.output_names,
                    data=data,
                    num_itrs=living_time,
                    worker_id=rank,
    )
    return colony

def living_colony(data, living_time, intervals, cost_type="mse", train_epochs=10):
    """
    used by threads to get the colonies to live in parallel
    """
    logger.info(f"Starting Colony: Lead Worker({rank}) reporting for duty")
    colony = create_colony(data=data, living_time=living_time)

    worker, best_position_global, fitness_global = comm_mpi.recv(source=0)
    colony.id = worker
    logger.info(f"Worker {rank} Received Main's Kickoff Msg")


    for tim in range(intervals, living_time + 1, intervals):
        # colony.life(num_itrs=intervals, total_itrs=living_time, cost_type=cost_type)
        # colony.life_processes(num_itrs=intervals, total_itrs=living_time, cost_type=cost_type)
        colony.life_threads(num_itrs=intervals, total_itrs=living_time, cost_type=cost_type, train_epochs=train_epochs)
        (   colony_fit, 
            colony_position,
        ) = colony.get_col_fit(rank=rank, avg=False)

        logger.info( 
                        f"Worker({rank}) reporting " +
                        f"its Overall Fitness: {fitness_global:.3f} " +
                        f"for Colony {colony.id} " +
                        f"No. Ants ({colony.num_ants}) " +
                        f"ER ({colony.evaporation_rate:.3f}) " +
                        f"MR ({colony.mortality_rate:.3f})  " +
                        f"({tim}/{living_time} Living Time)"
                    )
        comm_mpi.send((tim, colony_position, colony_fit), dest=0)
        best_position_global, fitness_global = comm_mpi.recv(source=0)
        colony.update_velocity(best_position_global)
        colony.update_position()
        logger.info(
                        f"\n***>>>===---\n"
                        f"Colony({colony.id})::\n" +
                        f"\tBest Global Pos: (Ants:{best_position_global[0]}, MortRate:{best_position_global[1]:.2f}, EvapRate:{best_position_global[2]:.2f})\n" +
                        f"\tBest Col Pos: (Ants:{colony.pso_best_position[0]}, MortRate:{colony.pso_best_position[1]:.2f}, EvapRate:{colony.pso_best_position[2]:.2f})\n" +
                        f"\tNo Ants: {colony.num_ants} " +
                        f"\tER: {colony.evaporation_rate:.3f}  " +
                        f"\tMR: {colony.mortality_rate:.3f}\n" +
                        f"---===<<<***"
                    )

    comm_mpi.send(None, dest=0)
    comm_mpi.send(colony, dest=0)

''' 
    First CPU: Environment
    Other CPUs grouped in groups = num of colonies
    First CPU in a group is Manager of group (Colony)
    Other group-CPUs: Workers
'''


def environment(living_time):
    best_position_global = None
    fitness_global = -1
    BEST_POS_GOL = [0] * num_colonies
    FIT_GOL = np.zeros(num_colonies)
    logger.info(f"Main reporting for duty")

    for w in worker_group:
        logger.info(f"Main sending Worker {w} its' kickoff msg") 
        comm_mpi.send((w, best_position_global, fitness_global), dest=w)
        logger.info(f"Main finished sending Worker {w} its' kickoff msg") 

    done_workers = 0
    best_colonies = []
    while True:
        for c in range(1, num_colonies+1):
            msg = comm_mpi.recv(source=c)
            if msg:
                tim, best_position, fitness_global = msg
                BEST_POS_GOL[c-1] = best_position
                FIT_GOL[c-1] = fitness_global
            else:
                done_workers+=1
                best_colonies.append(comm_mpi.recv(source=c))
        if done_workers==num_colonies:
            break
        elif 0<done_workers<num_colonies:
            logger.error("SOMETHING IS WRONG")
            sys.exit()
        fitness_global = np.min(FIT_GOL)
        best_position_global = BEST_POS_GOL[np.argmin(FIT_GOL)]
        logger.info(f"*** Finished {tim}/{living_time} Living Time ** Best Global Fitness: {fitness_global:.7e} ***")
        for c in range(1,num_colonies+1):
            comm_mpi.send((best_position_global, fitness_global), dest=c)
        
    '''
        **** TODO ****
        add code to save the best performing GRAPH in each round of intervals    
        this can be done by sending a signal to the lead-worker to save its
        best GRAPH if its group did the best job
    '''
    
    best_colony = best_colonies[0]
    for coln in best_colonies[1:]:
        if coln.best_solutions[0][0] < best_colony.best_solutions[0][0]:
            best_colony = coln


    # logger.info(f"** Evaluating Best RNN in Best Colony({best_rnn_colony.id}) **")
    # best_rnn_colony.use_bp = True
    # best_rnn_colony.num_epochs = 500
    # best_rnn_colony.evaluate_rnn(best_rnn_colony.best_rnns[0][1])


    # with open(
    #     "{}/{}_best_rnn.nn".format(args.out_dir, args.log_file_name), "bw"
    # ) as file_obj:
    #     pickle.dump(best_rnn_colony.best_rnns[0][1], file_obj)

def logger_setup(term_log_level="INFO", file_log_level="DEBUG", log_dir="logs", log_file_name="colonies"):
    logger.remove()
    logger.add(sys.stdout, level=term_log_level)
    logger.add(f"{log_dir}/{log_file_name}_cants.log", level=file_log_level)

def main():
    """
    main function to run the colonies
    """
    args = Args_Parser(sys.argv)
    logger_setup()

    if rank == 0: # Main Process
        logger.info(f"Main reporting for duty")
        environment(args.living_time)
        return
    else:   # Worker Process
        data = Timeseries(
                    data_files=args.data_files,
                    input_params=args.input_names,
                    output_params=args.output_names,
                    norm_type=args.normalization,
                    future_time=args.future_time,
                    data_dir=args.data_dir,
                )
        logger.info(f"Worker {rank} reporting for duty")
        intervals = args.communication_intervals
        if intervals > args.living_time + 1:
            logger.error(
                f"""
                    Colonies evolution intervals ({intervals}) less 
                    than the total number of iterations ({args.living_time+1})
                """
            )
            sys.exit()
        living_colony(data=data, living_time=args.living_time, intervals=intervals, cost_type=args.loss_fun, train_epochs=args.bp_epochs)


if __name__ == "__main__":
    main()


def kickoff_colonies(
                    data_files,
                    input_params,
                    output_params,
                    data_dir,
                    future_time=0,
                    term_log_level="INFO",
                    log_file_name="cants",
                    log_dir="logs",
                    file_log_level="INFO",
                    norm_type="minmax",
                    communication_intervals=2,
                    living_time=300,
                    cost_type="mse",
                    ):
    """
    kick off the colonies
    """
    logger_setup()
    if rank == 0:
        logger.info(f"Main reporting for duty")
        environment(living_time=living_time)
        return
    else:
        data = Timeseries(
                    data_files=data_files,
                    input_params=input_params,
                    output_params=output_params,
                    norm_type=norm_type,
                    future_time=future_time,
                    data_dir=data_dir,
                )
        logger.info(f"Worker {rank} reporting for duty")
        intervals = communication_intervals
        if intervals > living_time + 1:
            logger.error(
                f"""
                    Colonies evolution intervals ({intervals}) less 
                    than the total number of iterations ({living_time+1})
                """
            )
            sys.exit()
        living_colony(data=data, living_time=living_time, cost_type=cost_type)


# if rank==0:
#     environment()
# else:
#     intervals = args.communication_intervals
#     if intervals > args.living_time + 1:
#         logger.error(
#             f"Colonies evolution intervals ({intervals}) less " +
#             f"than the total number of iterations ({args.living_time+1})"
#         )
#         sys.exit()
#     living_colony()
