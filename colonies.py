"""
to run the colonies in parallel and evolve them
using PSO
"""
import sys

import pickle
import threading as th
import numpy as np
from loguru import logger
from colony import Colony
from timeseries import Timeseries
from helper import Args_Parser
from search_space import Space
from mpi4py import MPI

comm_mpi = MPI.COMM_WORLD
comm_size = comm_mpi.Get_size()
rank = comm_mpi.Get_rank()

sys.path.insert(1, "/home/aaevse/loguru")

args = Args_Parser(sys.argv)

# NUM_COLONIES = 10
# LIVING_TIME = 100

def logger_setup():
    logger.remove()
    logger.add(sys.stdout, level=args.term_log_level)
    logger.add(f"{args.log_dir}/{args.log_file_name}_cants.log", level=args.file_log_level)

def create_colony():
    data = Timeseries(
        data_files=args.data_files,
        input_params=args.input_names,
        output_params=args.output_names,
        norm_type=args.normalization,
        future_time=args.future_time,
        data_dir=args.data_dir,
    )

    num_ants = np.random.randint(low=1, high=20)
    population_size = np.random.randint(low=10, high=100)
    evaporation_rate = np.random.uniform(low=0.7, high=0.9)
    colony = Colony(
                    num_ants=num_ants, 
                    population_size=population_size, 
                    input_names=data.input_names,
                    output_names=data.output_names,
                    train_input=data.train_input,
                    train_target=data.train_output,
                    num_itrs=args.living_time,
    )

    return colony

intervals = args.communication_intervals
if intervals > args.living_time + 1:
    logger.error(
        f"Colonies evolution intervals ({intervals}) less " +
        f"than the total number of iterations ({args.living_time+1})"
    )
    sys.exit()


def living_colony():
    """
    used by threads to get the colonies to live in parallel
    """
    logger_setup()
    logger.info(f"Starting Colony: Lead Worker({rank}) reporting for duty")
    colony = create_colony()
    worker, best_position_global, fitness_global = comm_mpi.recv(source=0)
    colony.id = worker
    logger.info(f"Worker {rank} Received Main's Kickoff Msg")


    for tim in range(intervals, args.living_time + 1, intervals):
        colony.life(num_itrs=intervals, total_itrs=args.living_time)
        (   colony_fit, 
            colony_position
        ) = colony.get_col_fit(rank=rank, avg=False)

        logger.info( 
                        f"Worker({rank}) reporting " +
                        f"its OverallFitness: {fitness_global} " +
                        f"for Colony {colony.id} " +
                        f"No. Ants ({colony.num_ants}) " +
                        f"ER ({colony.evaporation_rate}) " +
                        f"MR ({colony.mortality_rate})  " +
                        f"({tim}/{args.living_time})"
                    )
        comm_mpi.send((tim, colony_position, colony_fit), dest=0)
        best_position_global, fitness_global = comm_mpi.recv(source=0)
        colony.update_velocity(best_position_global)
        colony.update_position()
        logger.info(f"***---===>>> Colony({colony.id}) Best Global Pos: {best_position_global})  Best Col Pos: {colony.pso_best_position} No Ants: {colony.num_ants} ER: {colony.evaporation_rate}  MR: {colony.mortality_rate}")

    comm_mpi.send(None, dest=0)
    comm_mpi.send(colony, dest=0)

''' 
    First CPU: Environment
    Other CPUs grouped in groups = num of colonies
    First CPU in a group is Manager of group (Colony)
    Other group-CPUs: Workers
'''
worker_group = np.arange(1,comm_size)
num_colonies = comm_size - 1

def environment():
    logger_setup()
    best_position_global = None
    fitness_global = -1
    BEST_POS_GOL = [0] * args.num_colonies
    FIT_GOL = np.zeros(args.num_colonies)
    logger.info(f"Main reporting for duty")

    for w in worker_group:
        logger.info(f"Main sending Worker {w} its' kickoff msg") 
        comm_mpi.send((w, best_position_global, fitness_global), dest=w)
        logger.info(f"Main finished sending Worker {w} its' kickoff msg") 

    done_workers = 0
    best_colonies = []
    while True:
        for c in range(1, args.num_colonies+1):
            msg = comm_mpi.recv(source=c)
            if msg:
                tim, best_position, fitness_global = msg
                BEST_POS_GOL[c-1] = best_position
                FIT_GOL[c-1] = fitness_global
            else:
                done_workers+=1
                best_colonies.append(comm_mpi.recv(source=c))
        if done_workers==args.num_colonies:
            break
        elif 0<done_workers<args.num_colonies:
            logger.error("SOMETHING IS WRONG")
            sys.exit()
        fitness_global = np.min(FIT_GOL)
        best_position_global = BEST_POS_GOL[np.argmin(FIT_GOL)]
        logger.info(f"*** Finished {tim}/{args.living_time} Living Time ** Best Global Fitness: {fitness_global} ***")
        for c in range(1,args.num_colonies+1):
            comm_mpi.send((best_position_global, fitness_global), dest=c)
        
    '''
        **** TODO ****
        add code to save the best performing RNN in each round of intervals    
        this can be done by sending a signal to the lead-worker to save its
        best RNN if its group did the best job
    '''
    """
    """
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

if rank==0:
    environment()
else:
    living_colony()
