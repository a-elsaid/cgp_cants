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
from loguru import logger
from mpi4py import MPI

# new libraries to apply new class
import os
import time
from pathlib import Path
import pandas as pd
import torch

logger = loguru.logger

comm_mpi = MPI.COMM_WORLD
comm_size = comm_mpi.Get_size()
rank = comm_mpi.Get_rank()

worker_group = np.arange(1, comm_size)
num_colonies = comm_size - 1

# -------------------- NEW: module-level handle to expose best colony/model --------------------
colony = None  # after environment() or Case-3, this will point to the best colony/controller

# -------------------- NEW: read threshold from env (set this in run.sh) ----------------------
# If FITNESS_GLOBAL is set (e.g., 90.0), treat it as the accuracy threshold (%)
def _read_threshold_default(default_val=90.0) -> float:
    for k in ("FITNESS_GLOBAL", "fitness_global", "ACC_THRESHOLD", "CANTS_CASE3_ACC_THRESH"):
        v = os.getenv(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return float(default_val)

FITNESS_THRESHOLD = _read_threshold_default(90.0)  # percent

def create_colony(data=None, living_time=None, out_dir="./OUT"):
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
        out_dir=out_dir,
    )
    return colony

def living_colony(data, living_time, intervals, cost_type="mse", train_epochs=10, out_dir="./OUT"):
    """
    used by threads to get the colonies to live in parallel
    """
    logger.info(f"Starting Colony: Lead Worker({rank}) reporting for duty")
    colony = create_colony(data=data, living_time=living_time, out_dir=out_dir)

    # Receive kickoff (worker id, best_position_global, threshold_as_fitness_global)
    worker, best_position_global, fitness_global = comm_mpi.recv(source=0)
    colony.id = worker
    logger.info(f"Worker {rank} Received Main's Kickoff Msg")

    for tim in range(intervals, living_time + 1, intervals):
        # evolve for this interval
        colony.life_threads(
            num_itrs=intervals,
            total_itrs=living_time,
            cost_type=cost_type,
            train_epochs=train_epochs,
        )
        (colony_fit, colony_position) = colony.get_col_fit(rank=rank, avg=False)

        # NOTE: fitness_global is a fixed threshold broadcast by rank 0
        logger.info(
            f"Worker({rank}) reporting "
            f"Threshold(Acc%): {fitness_global:.3f} "
            f"| Colony #{colony.id} "
            f"| Ants ({colony.num_ants}) "
            f"| ER ({colony.evaporation_rate:.3f}) "
            f"| MR ({colony.mortality_rate:.3f})  "
            f"({tim}/{living_time} Living Time)"
        )

        # send this colony's status back to main
        comm_mpi.send((tim, colony_position, colony_fit), dest=0)

        # receive PSO best position + (still) the threshold
        best_position_global, fitness_global = comm_mpi.recv(source=0)

        colony.update_velocity(best_position_global)
        colony.update_position()
        logger.info(
            f"\n***>>>===---\n"
            f"Colony({colony.id})::\n"
            f"\tBest Global Pos: (Ants:{best_position_global[0]}, "
            f"MortRate:{best_position_global[1]:.2f}, "
            f"EvapRate:{best_position_global[2]:.2f})\n"
            f"\tBest Col Pos: (Ants:{colony.pso_best_position[0]}, "
            f"MortRate:{colony.pso_best_position[1]:.2f}, "
            f"EvapRate:{colony.pso_best_position[2]:.2f})\n"
            f"\tNo Ants: {colony.num_ants} "
            f"\tER: {colony.evaporation_rate:.3f}  "
            f"\tMR: {colony.mortality_rate:.3f}\n"
            f"---===<<<***"
        )

    # signal done + send back the colony object
    comm_mpi.send(None, dest=0)
    comm_mpi.send(colony, dest=0)

''' 
    First CPU: Environment
    Other CPUs grouped in groups = num of colonies
    First CPU in a group is Manager of group (Colony)
    Other group-CPUs: Workers
'''
def environment(living_time):
    # IMPORTANT: fitness_global now carries the ACCURACY THRESHOLD (constant),
    #            not the evolving best fitness. We still compute the evolving
    #            best fitness separately below for logging/PSO purposes.
    fitness_global = FITNESS_THRESHOLD  # percent, e.g., 90.0

    best_position_global = None
    BEST_POS_GOL = [0] * num_colonies
    FIT_GOL = np.zeros(num_colonies)
    logger.info(f"Main reporting for duty (threshold/fitness_global={fitness_global}%)")

    # Kick off workers with the threshold value
    for w in worker_group:
        logger.info(f"Main sending Worker {w} its kickoff msg")
        comm_mpi.send((w, best_position_global, fitness_global), dest=w)
        logger.info(f"Main finished sending Worker {w} its kickoff msg")

    done_workers = 0
    best_colonies = []
    while True:
        for c in range(1, num_colonies + 1):
            msg = comm_mpi.recv(source=c)
            if msg:
                tim, best_position, col_fitness = msg
                BEST_POS_GOL[c - 1] = best_position
                FIT_GOL[c - 1] = col_fitness
            else:
                done_workers += 1
                best_colonies.append(comm_mpi.recv(source=c))

        if done_workers == num_colonies:
            break
        elif 0 < done_workers < num_colonies:
            logger.error("SOMETHING IS WRONG")
            sys.exit()

        # Compute current best fitness across colonies (for logging/PSO)
        current_best_fitness = float(np.min(FIT_GOL))
        best_position_global = BEST_POS_GOL[int(np.argmin(FIT_GOL))]

        logger.info(
            f"*** Finished {tim}/{living_time} Living Time "
            f"** Best Global Fitness (cost): {current_best_fitness:.7e} "
            f"** Threshold(Acc%): {fitness_global:.3f} ***"
        )

        # Broadcast the best position + the SAME threshold to all workers
        for c in range(1, num_colonies + 1):
            comm_mpi.send((best_position_global, fitness_global), dest=c)

    # -------------------- best performing model selection --------------------
    if not best_colonies:
        logger.error("No colonies returned from workers.")
        return None

    best_colony = best_colonies[0]
    for coln in best_colonies[1:]:
        if coln.best_solutions[0][0] < best_colony.best_solutions[0][0]:
            best_colony = coln

    # -------------------- expose & return the model ----------------
    global colony
    colony = best_colony                         # users can call colonies.colony.get_best_model()
    best_model = best_colony.get_best_model()
    return best_model

def logger_setup(term_log_level="INFO", file_log_level="DEBUG", log_dir="logs", log_file_name="colonies"):
    logger.remove()
    logger.add(sys.stdout, level=term_log_level)
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception:
        pass
    logger.add(f"{log_dir}/{log_file_name}_cants.log", level=file_log_level)

def main():
    """
    main function to run the colonies
    """
    args = Args_Parser(sys.argv)
    logger_setup()

    if rank == 0: # Main Process
        logger.info(f"Main reporting for duty")
        best_model = environment(args.living_time)
        # best_model is returned; global "colony" is set too
        return best_model
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
        living_colony(
            data=data, 
            living_time=args.living_time, 
            intervals=intervals, 
            cost_type=args.loss_fun, 
            train_epochs=args.bp_epochs, 
            out_dir=args.out_dir,
        )

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
        best_model = environment(living_time=living_time)   # <-- return it here
        return best_model
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

################################## NEW CODE ##############################################
# ======== Case 3 (Pickle): Threshold-Triggered Window Retraining ============
# CLI stays the same; enable with:  export CANTS_CASE3=1
# Optional envs:
#   CANTS_CASE3_ACC_THRESH=90.0  (also honors FITNESS_GLOBAL if set)
#   CANTS_CASE3_START=2012-01-01
#   CANTS_CASE3_END=2019-12-31
#   CANTS_CASE3_LOG_PREFIX=case3

def _split_space_list(s: str):
    return [x.strip() for x in s.split() if x.strip()]

def _load_pickle_df_list(data_dir: str, files: list) -> pd.DataFrame:
    dfs = []
    for f in files:
        p = Path(f)
        if not p.is_absolute():
            p = Path(data_dir) / f
        try:
            df = pd.read_pickle(p)
        except Exception:
            with open(p, "rb") as fh:
                df = pickle.load(fh)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Pickle {p} did not contain a DataFrame.")
        dfs.append(df)
    if not dfs:
        raise ValueError("No pickle dataframes loaded.")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    return df

def _ensure_datetime(df: pd.DataFrame, col: str = "HighestModDate") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    df.sort_values(by=col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _three_month_intervals(df: pd.DataFrame, start=None, end=None, date_col="HighestModDate"):
    if start is None:
        start = df[date_col].min().normalize()
    else:
        start = pd.to_datetime(start)
    if end is None:
        end = df[date_col].max().normalize()
    else:
        end = pd.to_datetime(end)
    intervals = []
    cur = start
    while cur <= end:
        nxt = (cur + pd.DateOffset(months=3)) - pd.DateOffset(days=1)
        if nxt > end:
            nxt = end
        intervals.append((cur, nxt))
        cur = nxt + pd.DateOffset(days=1)
    return intervals

def _mask_between(df: pd.DataFrame, start, end, date_col="HighestModDate"):
    m = (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[m]

def _binary_acc(preds: np.ndarray, y: np.ndarray, threshold=0.5, lags:int=0) -> float:
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    y_eff = y[lags:]
    if len(y_eff) == 0:
        return float("nan")
    labels = (preds >= threshold).astype(np.float32)
    y_labels = (y_eff >= 0.5).astype(np.float32)
    return float((labels == y_labels).mean()) * 100.0

def _compute_norm_stats(X_train: np.ndarray, mode: str):
    if mode == "minmax":
        return {"mode":"minmax","min":X_train.min(axis=0), "max":X_train.max(axis=0)}
    elif mode == "mean_std":
        return {"mode":"mean_std","mean":X_train.mean(axis=0), "std":X_train.std(axis=0)}
    return {"mode":"none"}

def _apply_norm(X: np.ndarray, st: dict):
    m = st.get("mode","none")
    if m == "minmax":
        denom = (st["max"] - st["min"]).copy()
        denom[denom==0.0] = 1.0
        return (X - st["min"]) / denom
    elif m == "mean_std":
        denom = st["std"].copy()
        denom[denom==0.0] = 1.0
        return (X - st["mean"]) / denom
    return X

class Colonies:
    """
    Case 3 controller over 3-month windows for a single pickled DataFrame.
    """
    def __init__(self, args):
        self.args = args
        self.data_dir = getattr(args, "data_dir", getattr(args, "data_dirs",".")) 
        files = _split_space_list(getattr(args, "data_files", ""))
        if not files:
            logger.error("Please pass your pickle path in --data_files")
            raise SystemExit(1)
        df = _load_pickle_df_list(self.data_dir, files)
        df = _ensure_datetime(df, "HighestModDate")

        start_env = os.getenv("CANTS_CASE3_START")
        end_env   = os.getenv("CANTS_CASE3_END")
        if start_env or end_env:
            start = pd.to_datetime(start_env) if start_env else df["HighestModDate"].min()
            end   = pd.to_datetime(end_env)   if end_env   else df["HighestModDate"].max()
            df = df[(df["HighestModDate"] >= start) & (df["HighestModDate"] <= end)].copy()

        self.df_all = df
        self.in_names  = _split_space_list(getattr(args, "input_names",  ""))
        self.out_names = _split_space_list(getattr(args, "output_names", ""))
        if not self.in_names or not self.out_names:
            raise SystemExit("Please provide --input_names and --output_names matching your pickle columns.")
        self.norm_type = getattr(args, "norm_type", "minmax")
        self.time_lag  = getattr(args, "time_lag", 0)

        # Threshold also honored here (either FITNESS_GLOBAL or CANTS_CASE3_ACC_THRESH)
        self.acc_threshold = float(os.getenv("FITNESS_GLOBAL", os.getenv("CANTS_CASE3_ACC_THRESH", "90.0")))
        self.log_prefix    = os.getenv("CANTS_CASE3_LOG_PREFIX", "case3")

        self.intervals = _three_month_intervals(self.df_all)
        if len(self.intervals) < 1:
            raise SystemExit("No intervals found after filtering.")

        self.interval_numbers     = []
        self.before_accuracies    = []
        self.retrained_accuracies = []
        self.base_times           = []
        self.train_times          = []
        self.after_times          = []

        self.current_graph = None
        self.current_lags  = 0
        self.norm_stats_in = None

    def get_best_model(self):
        return self.current_graph

    def _window_df(self, idx: int) -> pd.DataFrame:
        s, e = self.intervals[idx]
        return _mask_between(self.df_all, s, e, "HighestModDate")

    def _to_arrays(self, df_win: pd.DataFrame):
        X = df_win[self.in_names].astype(np.float32).values
        y = df_win[self.out_names].astype(np.float32).values
        return X, y

    def _train_on_window(self, df_win: pd.DataFrame):
        X_raw, y_raw = self._to_arrays(df_win)
        if len(X_raw) < 10:
            return None, None
        split = int(len(X_raw) * 0.8)
        X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
        y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

        self.norm_stats_in = _compute_norm_stats(X_train_raw, self.norm_type)
        X_train = _apply_norm(X_train_raw, self.norm_stats_in)
        X_test  = _apply_norm(X_test_raw,  self.norm_stats_in)

        class _TS: pass
        ts = _TS()
        ts.input_names  = self.in_names
        ts.output_names = self.out_names
        ts.train_input  = X_train
        ts.test_input   = X_test
        ts.train_output = y_train_raw
        ts.test_output  = y_test_raw
        ts.input_data   = X_raw
        ts.output_data  = y_raw

        colony = Colony(
            num_ants=np.random.randint(5, 15),
            population_size=np.random.randint(10, 20),
            input_names=ts.input_names,
            output_names=ts.output_names,
            data=ts,
            num_itrs=getattr(self.args,"living_time", 10),
            worker_id=0,
            out_dir=getattr(self.args,"out_dir","./OUT"),
            use_torch=True,
        )

        t0 = time.time()
        colony.life_threads(
            num_itrs=getattr(self.args,"communication_intervals", 2),
            total_itrs=getattr(self.args,"living_time", 10),
            cost_type=getattr(self.args,"loss_fun","mse"),
            train_epochs=getattr(self.args,"bp_epochs", 10),
        )
        t1 = time.time()
        self.train_times.append(t1 - t0)

        if not getattr(colony, "best_solutions", None):
            logger.error("No graphs produced by colony.")
            return None, ts

        best_graph = colony.best_solutions[0][1]
        self.current_graph = best_graph
        self.current_lags  = getattr(best_graph, "lags", 0)
        return best_graph, ts

    def _eval_on_full_window(self, df_win: pd.DataFrame) -> float:
        if self.current_graph is None:
            return float("nan")
        X_raw, y_raw = self._to_arrays(df_win)
        if len(X_raw) == 0:
            return float("nan")
        X = _apply_norm(X_raw, self.norm_stats_in)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_raw, dtype=torch.float32)

        t0 = time.time()
        preds, _, _ = self.current_graph.single_thrust(
            X_t, y_t, prt=False, cal_gradient=False,
            cost_type=getattr(self.args, "loss_fun", "mse")
        )
        if isinstance(preds[0], torch.Tensor):
            P = torch.stack(preds).detach().cpu().numpy()
        else:
            P = np.stack(preds)
        t1 = time.time()
        self.base_times.append(t1 - t0)

        return _binary_acc(P, y_raw, threshold=0.5, lags=self.current_lags)

    def run_case3(self):
        logger.info(f"[Case3-Pickle] {len(self.intervals)} intervals | acc threshold = {self.acc_threshold}%")

        w1_df = self._window_df(0)
        graph, ts = self._train_on_window(w1_df)
        if graph is None:
            logger.error("Training on Window 1 failed or window too small.")
            return

        X_test = _apply_norm(ts.test_input, self.norm_stats_in) if ts.test_input is not None else None
        y_test = ts.test_output
        if X_test is not None and len(X_test) > 0:
            X_t = torch.tensor(X_test, dtype=torch.float32)
            y_t = torch.tensor(y_test, dtype=torch.float32)
            preds, _, _ = self.current_graph.single_thrust(
                X_t, y_t, prt=False, cal_gradient=False,
                cost_type=getattr(self.args,"loss_fun","mse")
            )
            P = torch.stack(preds).detach().cpu().numpy() if isinstance(preds[0], torch.Tensor) else np.stack(preds)
            acc_w1 = _binary_acc(P, y_test, threshold=0.5, lags=self.current_lags)
        else:
            acc_w1 = float("nan")

        self.interval_numbers.append(1)
        self.before_accuracies.append(acc_w1)
        self.retrained_accuracies.append(np.nan)
        self.after_times.append(np.nan)
        logger.info(f"[Case-3] Window 1 accuracy (held-out): {acc_w1:.2f}%")

        for i in range(1, len(self.intervals)):
            df_win = self._window_df(i)
            label  = f"Window {i+1}"
            acc_before = self._eval_on_full_window(df_win)
            logger.info(f"[Case-3] {label}: accuracy before retrain = {acc_before:.2f}%")
            self.interval_numbers.append(i+1)
            self.before_accuracies.append(acc_before)
            retrained_acc = np.nan

            if np.isnan(acc_before) or acc_before < self.acc_threshold:
                logger.info(f"[Case-3] {label}: acc {acc_before:.2f}% < {self.acc_threshold}%; retraining on this window.")
                _ = self._train_on_window(df_win)
                t4 = time.time()
                retrained_acc = self._eval_on_full_window(df_win)
                t5 = time.time()
                self.after_times.append(t5 - t4)
                logger.info(f"[Case-3] {label}: accuracy after retrain = {retrained_acc:.2f}%")
            else:
                self.train_times.append(np.nan)
                self.after_times.append(np.nan)

            self.retrained_accuracies.append(retrained_acc)

        finals = []
        for b, r in zip(self.before_accuracies, self.retrained_accuracies):
            finals.append(r if not (isinstance(r, float) and np.isnan(r)) else b)
        avg_final  = float(np.nanmean(finals))
        avg_before = float(np.nanmean(self.before_accuracies))
        logger.info(f"[Case-3] Avg final acc = {avg_final:.2f}% | Avg before = {avg_before:.2f}%")
        if np.any(~np.isnan(np.array(self.retrained_accuracies, dtype=float))):
            avg_after = float(np.nanmean([r for r in self.retrained_accuracies if not np.isnan(r)]))
            logger.info(f"[Case-3] Avg acc after retrain (triggered windows): {avg_after:.2f}%")

        try:
            out_dir = getattr(self.args, "out_dir", "./OUT")
            os.makedirs(out_dir, exist_ok=True)
            df_sum = pd.DataFrame({
                "Interval": self.interval_numbers,
                "Acc_Before": self.before_accuracies,
                "Acc_After": self.retrained_accuracies,
                "Base_s": self.base_times + [np.nan]*(len(self.interval_numbers)-len(self.base_times)),
                "Train_s": self.train_times + [np.nan]*(len(self.interval_numbers)-len(self.train_times)),
                "After_s": self.after_times,
            })
            p = Path(out_dir) / f"{self.log_prefix}_case3_pickle_summary.csv"
            df_sum.to_csv(p, index=False)
            logger.info(f"[Case-3] Wrote summary CSV: {p}")
        except Exception as e:
            logger.warning(f"[Case-3] Could not write summary CSV: {e}")

# ---- Entry switch: choose Case-3 path only when requested ----
if os.getenv("CANTS_CASE3","0") == "1":
    args = Args_Parser(sys.argv)
    logger_setup()
    _case3 = Colonies(args)
    # expose controller so you can call colonies.colony.get_best_model()
    global colony
    colony = _case3
    _case3.run_case3()
    sys.exit(0)
