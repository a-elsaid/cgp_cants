"""
to run the colonies in parallel and evolve them
using PSO
"""
# ^ Module docstring: describes purpose of the file.

import sys                # stdlib: access argv, stdout, etc.
import pickle             # stdlib: loading pickle files (used in Case-3)
import threading as th    # stdlib: threads (used inside Colony.life_threads)
import numpy as np        # numeric arrays
import loguru             # logging framework
from colony import Colony # core evolutionary/training logic per colony
from timeseries import Timeseries   # data wrapper for CSV (classic MPI path)
from helper import Args_Parser      # CLI parser
from search_space import Space      # search-space primitives (not directly used here)
from loguru import logger           # logger handle
from mpi4py import MPI             # MPI for multi-process orchestration

# new libraries to apply new class
import os                 # stdlib: environment variables, paths
import time               # stdlib: timing for perf logs
from pathlib import Path  # cross-platform filesystem paths
import pandas as pd       # dataframes (used in Case-3 pickle path)
import torch              # tensor ops / inference in graphs

logger = loguru.logger    # unify logging handle

# -------------------- MPI world --------------------
comm_mpi = MPI.COMM_WORLD        # MPI communicator spanning all ranks
comm_size = comm_mpi.Get_size()  # total number of MPI processes
rank = comm_mpi.Get_rank()       # this process's rank id (0 = master)

worker_group = np.arange(1, comm_size)      # ranks 1..N-1 are workers
num_colonies = max(0, comm_size - 1)        # number of worker colonies

# -------------------- module-level handle --------------------
colony = None  # after environment() or Case-3, points to the best colony/controller

# -------------------- helpers --------------------
# NEW: accept ACC threshold from several env var spellings.
def _read_threshold_default(default_val=90.0) -> float:
    for k in ("FITNESS_GLOBAL", "fitness_global", "ACC_THRESHOLD", "CANTS_CASE3_ACC_THRESH"):
        v = os.getenv(k)          # read env var k
        if v is not None:
            try:
                return float(v)   # parse to float if present
            except Exception:
                pass              # ignore bad values, fall through
    return float(default_val)     # fallback default (percent)

FITNESS_THRESHOLD = _read_threshold_default(90.0)  # percent; global accuracy trigger

# NEW: normalize loss names from env/CLI into the internal identifiers used by Graph/Colony.
def _map_loss_name(name_from_env: str) -> str:
    n = (name_from_env or "").strip().lower()
    if n in ("bce", "binary_cross_entropy", "bi_cross_entropy", "bicross", "bicross_entropy"):
        return "bicross_entropy"
    if n in ("ce", "cross_entropy", "softmax_ce"):
        return "cross_entropy"
    if n in ("mse", "l2"):
        return "mse"
    return "bicross_entropy"  # default if unknown

def logger_setup(term_log_level="INFO", file_log_level="DEBUG", log_dir="logs", log_file_name="colonies"):
    logger.remove()                                  # clear existing handlers
    logger.add(sys.stdout, level=term_log_level)     # log to console
    try:
        os.makedirs(log_dir, exist_ok=True)          # ensure log directory
    except Exception:
        pass
    logger.add(f"{log_dir}/{log_file_name}_cants.log", level=file_log_level)  # log to file

# ======================================================================
# =========================== MPI (classic) =============================
# ======================================================================

def create_colony(data=None, living_time=None, out_dir="./OUT"):
    num_ants = np.random.randint(low=1, high=20)     # randomize ants (diversity)
    population_size = np.random.randint(low=5, high=25)  # number of graphs
    c = Colony(
        num_ants=num_ants,
        population_size=population_size,
        input_names=data.input_names,    # features
        output_names=data.output_names,  # labels
        data=data,                       # Timeseries or TS-like object
        num_itrs=living_time,            # total evolution steps
        worker_id=rank,                  # tie to MPI rank
        out_dir=out_dir,                 # output dir
    )
    return c

def living_colony(data, living_time, intervals, cost_type="mse", train_epochs=10, out_dir="./OUT"):
    """
    used by threads to get the colonies to live in parallel
    """
    logger.info(f"Starting Colony: Lead Worker({rank}) reporting for duty")
    c = create_colony(data=data, living_time=living_time, out_dir=out_dir)

    # Receive kickoff (worker id, best_position_global, threshold_as_fitness_global)
    worker, best_position_global, fitness_global = comm_mpi.recv(source=0)  # blocking recv from master
    c.id = worker                         # assign a stable ID to this colony
    logger.info(f"Worker {rank} Received Main's Kickoff Msg")

    for tim in range(intervals, living_time + 1, intervals):  # evolve in communication chunks
        c.life_threads(
            num_itrs=intervals,          # iterations this chunk
            total_itrs=living_time,      # total iterations target
            cost_type=str(cost_type),    # e.g., "mse" | "bicross_entropy" | "cross_entropy"
            train_epochs=train_epochs,   # backprop epochs inside each eval
        )
        (colony_fit, colony_position) = c.get_col_fit(rank=rank, avg=False)  # current fitness & PSO pos

        logger.info(
            f"Worker({rank}) reporting "
            f"Threshold(Acc%): {fitness_global:.3f} "  # note: carried as accuracy threshold in this fork
            f"| Colony #{c.id} "
            f"| Ants ({c.num_ants}) "
            f"({tim}/{living_time} Living Time)"
        )

        comm_mpi.send((tim, colony_position, colony_fit), dest=0)   # send progress to master
        best_position_global, fitness_global = comm_mpi.recv(source=0)  # receive updated PSO global best & threshold

        c.update_velocity(best_position_global)  # pull towards global best
        c.update_position()                      # apply PSO step

    comm_mpi.send(None, dest=0)  # signal completion
    comm_mpi.send(c, dest=0)     # send back the final colony object for selection

def environment(living_time):
    fitness_global = FITNESS_THRESHOLD     # global accuracy threshold (repurposed)
    best_position_global = None            # best PSO point known so far
    BEST_POS_GOL = [0] * num_colonies      # store best positions per worker
    FIT_GOL = np.zeros(num_colonies)       # store fitness per worker
    logger.info(f"Main reporting for duty (threshold/fitness_global={fitness_global}%)")

    for w in worker_group:
        comm_mpi.send((w, best_position_global, fitness_global), dest=w)  # kickoff message to each worker

    done_workers = 0
    best_colonies = []
    while True:
        for c in range(1, num_colonies + 1):       # one mailbox per worker rank
            msg = comm_mpi.recv(source=c)          # receive either progress or 'None' (done signal)
            if msg:
                tim, best_position, col_fitness = msg
                BEST_POS_GOL[c - 1] = best_position
                FIT_GOL[c - 1] = col_fitness
            else:
                done_workers += 1
                best_colonies.append(comm_mpi.recv(source=c))  # then receive the colony object

        if done_workers == num_colonies:
            break
        elif 0 < done_workers < num_colonies:
            logger.error("SOMETHING IS WRONG")
            sys.exit()

        current_best_fitness = float(np.min(FIT_GOL))           # pick best current fitness among workers
        best_position_global = BEST_POS_GOL[int(np.argmin(FIT_GOL))]  # corresponding PSO position
        for c in range(1, num_colonies + 1):
            comm_mpi.send((best_position_global, fitness_global), dest=c)  # broadcast PSO best & fixed threshold

    if not best_colonies:        # safety: no worker returned a colony (early exit)
        logger.error("No colonies returned from workers.")
        return None

    best_colony = best_colonies[0]                       # select the best colony by lowest fitness of best_solutions[0]
    for coln in best_colonies[1:]:
        if coln.best_solutions[0][0] < best_colony.best_solutions[0][0]:
            best_colony = coln

    global colony
    colony = best_colony                                 # expose best colony at module level
    best_model = best_colony.get_best_model()            # return its best graph/model
    return best_model

# ======================================================================
# ============================= Case-3 =================================
# ======================================================================

# NEW: helper to split space-separated CLI strings into a list
def _split_space_list(s: str):
    return [x.strip() for x in s.split() if x.strip()]

# NEW: load one or more pickled DataFrames and concatenate
def _load_pickle_df_list(data_dir: str, files: list) -> pd.DataFrame:
    dfs = []
    for f in files:
        p = Path(f)
        if not p.is_absolute():
            p = Path(data_dir) / f              # make path absolute
        try:
            df = pd.read_pickle(p)              # fast path
        except Exception:
            with open(p, "rb") as fh:           # fallback to pickle.load for older pickles
                df = pickle.load(fh)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Pickle {p} did not contain a DataFrame.")
        dfs.append(df)
    if not dfs:
        raise ValueError("No pickle dataframes loaded.")
    df = pd.concat(dfs, axis=0, ignore_index=True)  # stack rows
    return df

# NEW: ensure a datetime column exists, is parsed, and data is sorted by it
def _ensure_datetime(df: pd.DataFrame, col: str = "HighestModDate") -> pd.DataFrame:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    df = df.copy()
    df[col] = pd.to_datetime(df[col])    # parse to datetime
    df.sort_values(by=col, inplace=True) # ascending time
    df.reset_index(drop=True, inplace=True)
    return df

# NEW: split overall date range into contiguous 3-month intervals
def _three_month_intervals(df: pd.DataFrame, start=None, end=None, date_col="HighestModDate"):
    if start is None:
        start = df[date_col].min().normalize()  # floor to day
    else:
        start = pd.to_datetime(start)
    if end is None:
        end = df[date_col].max().normalize()
    else:
        end = pd.to_datetime(end)
    intervals = []
    cur = start
    while cur <= end:
        nxt = (cur + pd.DateOffset(months=3)) - pd.DateOffset(days=1)  # inclusive end of 3-month block
        if nxt > end:
            nxt = end
        intervals.append((cur, nxt))
        cur = nxt + pd.DateOffset(days=1)       # next block starts the day after
    return intervals

# NEW: row filter for [start, end] inclusive
def _mask_between(df: pd.DataFrame, start, end, date_col="HighestModDate"):
    m = (df[date_col] >= start) & (df[date_col] <= end)
    return df.loc[m]

# NEW: utilities to align predictions with labels (considering lags) and compute accuracy
def _align_pred_y(preds: np.ndarray, y: np.ndarray, lags:int=0):
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    y_eff = y[lags:]                            # drop initial lags from labels
    n = min(len(preds), len(y_eff))             # clip to common length
    return preds[:n], y_eff[:n]

def _binary_acc(preds: np.ndarray, y: np.ndarray, threshold=0.5, lags:int=0) -> float:
    P, Y = _align_pred_y(preds, y, lags)
    if len(Y) == 0:
        return float("nan")
    labels = (P >= threshold).astype(np.float32)
    y_labels = (Y >= 0.5).astype(np.float32)
    return float((labels == y_labels).mean()) * 100.0   # percentage

# NEW: small search over thresholds to maximize accuracy on validation
def _best_threshold(preds: np.ndarray, y: np.ndarray, lags:int=0):
    P, Y = _align_pred_y(preds, y, lags)
    if len(Y) == 0:
        return 0.5
    ts = np.linspace(0.05, 0.95, 19)           # 0.05..0.95 step 0.05
    best_t, best_s = 0.5, -1.0
    Yb = (Y >= 0.5).astype(np.float32)
    for t in ts:
        s = ( (P >= t).astype(np.float32) == Yb ).mean()
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t)

# NEW: compute normalization statistics on train split
def _compute_norm_stats(X_train: np.ndarray, mode: str):
    if mode == "minmax":
        return {"mode":"minmax","min":X_train.min(axis=0), "max":X_train.max(axis=0)}
    elif mode == "mean_std":
        return {"mode":"mean_std","mean":X_train.mean(axis=0), "std":X_train.std(axis=0)}
    return {"mode":"none"}

# NEW: apply normalization to a matrix using given stats
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

# -------------------- Candidate training (used by MPI workers and local explore) --------------------
# NEW: train a single candidate (graph) on the given window data and return metrics + artifacts.
def _train_candidate(X_raw: np.ndarray, y_raw: np.ndarray, in_names, out_names,
                     norm_type: str, loss_pref: str, living_time: int, comm_intervals: int, bp_epochs: int, out_dir: str):
    """
    Train a single candidate on this window; return dict with graph, stats, lags, loss, threshold, acc.
    """
    if len(X_raw) < 10:
        return None
    split = int(len(X_raw) * 0.8)           # 80/20 split
    X_tr_raw, X_te_raw = X_raw[:split], X_raw[split:]
    y_tr_raw, y_te_raw = y_raw[:split], y_raw[split:]

    norm_stats = _compute_norm_stats(X_tr_raw, norm_type)  # fit scaler on train
    X_tr = _apply_norm(X_tr_raw, norm_stats)
    X_te = _apply_norm(X_te_raw, norm_stats)

    class _TS: pass                      # minimal Timeseries-like container
    ts = _TS()
    ts.input_names  = in_names
    ts.output_names = out_names
    ts.train_input  = X_tr
    ts.test_input   = X_te
    ts.train_output = y_tr_raw
    ts.test_output  = y_te_raw
    ts.input_data   = X_raw
    ts.output_data  = y_raw

    c = Colony(
        num_ants=np.random.randint(5, 15),         # random colony size
        population_size=np.random.randint(10, 20),
        input_names=ts.input_names,
        output_names=ts.output_names,
        data=ts,
        num_itrs=living_time,
        worker_id=rank,
        out_dir=out_dir,
        use_torch=True,                             # torch-mode for graphs
    )

    # Loss try order (robust to incompatible choices)
    mapped = _map_loss_name(loss_pref)
    if mapped == "mse":
        try_order = ["mse"]
    elif mapped == "cross_entropy":
        try_order = ["cross_entropy", "bicross_entropy", "mse"]
    else:
        try_order = ["bicross_entropy", "cross_entropy", "mse"]

    last_err = None
    for loss_name in try_order:
        try:
            c.life_threads(
                num_itrs=comm_intervals,      # one comm chunk per candidate
                total_itrs=living_time,
                cost_type=str(loss_name),     # chosen loss
                train_epochs=bp_epochs,
            )
            used_loss = loss_name             # success; keep the one that worked
            break
        except Exception as e:
            last_err = e
            used_loss = None                  # try the next fallback
    if used_loss is None:
        return {"error": f"train_failed: {last_err}"}

    if not getattr(c, "best_solutions", None):  # ensure colony produced any graph
        return {"error": "no_graph"}

    best_graph = c.best_solutions[0][1]         # take top-1 graph
    lags = getattr(best_graph, "lags", 0)       # time lag used inside graph (if any)

    # Learn a decision threshold on the validation split
    X_t = torch.tensor(X_te, dtype=torch.float32)
    y_t = torch.tensor(y_te_raw, dtype=torch.float32)
    preds, _, _ = best_graph.single_thrust(X_t, y_t, prt=False, cal_gradient=False, cost_type=used_loss)
    P = torch.stack(preds).detach().cpu().numpy() if isinstance(preds[0], torch.Tensor) else np.stack(preds)
    t_star = _best_threshold(P, y_te_raw, lags=lags)

    # Evaluate on full window using the candidate's own scaler and learned threshold
    X_full = _apply_norm(X_raw, norm_stats)
    Xf_t = torch.tensor(X_full, dtype=torch.float32)
    yf_t = torch.tensor(y_raw, dtype=torch.float32)
    preds_f, _, _ = best_graph.single_thrust(Xf_t, yf_t, prt=False, cal_gradient=False, cost_type=used_loss)
    Pf = torch.stack(preds_f).detach().cpu().numpy() if isinstance(preds_f[0], torch.Tensor) else np.stack(preds_f)
    acc_full = _binary_acc(Pf, y_raw, threshold=t_star, lags=lags)

    return {
        "graph": best_graph,          # trained candidate graph
        "norm_stats": norm_stats,     # its scaler
        "lags": int(lags),            # lag used
        "loss": used_loss,            # loss that worked
        "threshold": float(t_star),   # decision threshold tuned on holdout
        "acc": float(acc_full),       # accuracy on the full window
    }

# -------------------- Case-3 controller --------------------
# NEW: main controller implementing time-windowed, threshold-triggered retraining with optional MPI exploration.
class Colonies:
    """
    Case 3 controller over 3-month windows for a single pickled DataFrame.
    """
    def __init__(self, args):
        self.args = args
        self.data_dir = getattr(args, "data_dir", getattr(args, "data_dirs","."))  # prefer single dir
        files = _split_space_list(getattr(args, "data_files", ""))                 # listify data_files
        if not files:
            logger.error("Please pass your pickle path in --data_files")
            raise SystemExit(1)
        df = _load_pickle_df_list(self.data_dir, files)   # read pickle(s)
        df = _ensure_datetime(df, "HighestModDate")       # sort by time

        start_env = os.getenv("CANTS_CASE3_START")        # optional date slicing
        end_env   = os.getenv("CANTS_CASE3_END")
        if start_env or end_env:
            start = pd.to_datetime(start_env) if start_env else df["HighestModDate"].min()
            end   = pd.to_datetime(end_env)   if end_env   else df["HighestModDate"].max()
            df = df[(df["HighestModDate"] >= start) & (df["HighestModDate"] <= end)].copy()

        self.df_all = df
        self.in_names  = _split_space_list(getattr(args, "input_names",  ""))   # features
        self.out_names = _split_space_list(getattr(args, "output_names", ""))   # target(s)
        if not self.in_names or not self.out_names:
            raise SystemExit("Please provide --input_names and --output_names matching your pickle columns.")
        self.norm_type = getattr(args, "norm_type", "minmax")   # scaling mode
        self.time_lag  = getattr(args, "time_lag", 0)           # not used directly here

        # NEW: Case-3 knobs from env
        self.acc_threshold    = float(os.getenv("FITNESS_GLOBAL", os.getenv("CANTS_CASE3_ACC_THRESH", "90.0")))
        self.log_prefix       = os.getenv("CANTS_CASE3_LOG_PREFIX", "case3")
        self.desired_loss     = _map_loss_name(os.getenv("CANTS_CASE3_LOSS", "bicross_entropy"))
        self.promote_delta    = float(os.getenv("CANTS_CASE3_PROMOTE_DELTA", "0.5"))  # min improvement (pp) to promote
        self.local_candidates = int(os.getenv("CANTS_CASE3_CANDIDATES", "2"))         # local (non-MPI) trials per window

        self.intervals = _three_month_intervals(self.df_all)   # build 3-month windows
        if len(self.intervals) < 1:
            raise SystemExit("No intervals found after filtering.")

        # Tracking arrays for summary report
        self.interval_numbers     = []
        self.before_accuracies    = []
        self.retrained_accuracies = []
        self.base_times           = []
        self.train_times          = []
        self.after_times          = []

        # Deployed (frozen) model state
        self.current_graph = None
        self.current_lags  = 0
        self.norm_stats_in = None
        self.current_loss  = self.desired_loss
        self.current_threshold = 0.5

    def get_best_model(self):
        return self.current_graph

    def _window_df(self, idx: int) -> pd.DataFrame:
        s, e = self.intervals[idx]                     # (start, end)
        return _mask_between(self.df_all, s, e, "HighestModDate")  # slice rows

    def _to_arrays(self, df_win: pd.DataFrame):
        X = df_win[self.in_names].astype(np.float32).values   # features matrix
        y = df_win[self.out_names].astype(np.float32).values  # label(s)
        return X, y

    # ---------- Window-1: train and FREEZE ----------
    def _fit_window1(self, df_win: pd.DataFrame):
        X_raw, y_raw = self._to_arrays(df_win)
        cand = _train_candidate(
            X_raw, y_raw,
            self.in_names, self.out_names,
            self.norm_type, self.desired_loss,
            living_time=getattr(self.args,"living_time", 10),
            comm_intervals=getattr(self.args,"communication_intervals", 2),
            bp_epochs=getattr(self.args,"bp_epochs", 10),
            out_dir=getattr(self.args,"out_dir","./OUT"),
        )
        if cand is None or "error" in cand:
            logger.error(f"W1 training failed: {cand.get('error') if cand else 'unknown'}")
            return None

        # Freeze deployed state (parameters do not change unless promoted)
        self.current_graph     = cand["graph"]
        self.current_lags      = cand["lags"]
        self.norm_stats_in     = cand["norm_stats"]
        self.current_loss      = cand["loss"]
        self.current_threshold = cand["threshold"]

        # Return W1 accuracy (already computed over full window)
        return float(cand["acc"])

    def _eval_deployed_on_window(self, X_raw: np.ndarray, y_raw: np.ndarray) -> float:
        if self.current_graph is None or self.norm_stats_in is None:
            return float("nan")

        X = _apply_norm(X_raw, self.norm_stats_in)               # use deployed scaler
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_raw, dtype=torch.float32)

        t0 = time.time()
        preds, _, _ = self.current_graph.single_thrust(          # forward pass only
            X_t, y_t, prt=False, cal_gradient=False,
            cost_type=self.current_loss
        )
        if isinstance(preds[0], torch.Tensor):
            P = torch.stack(preds).detach().cpu().numpy()
        else:
            P = np.stack(preds)
        t1 = time.time()
        self.base_times.append(t1 - t0)                          # record inference time

        return _binary_acc(P, y_raw, threshold=self.current_threshold, lags=self.current_lags)

    # ---------- MPI exploration ----------
    # NEW: ask workers to each train a candidate and return the best by accuracy.
    def _mpi_explore(self, X_raw, y_raw):
        """Ask each worker to train a candidate and return the best dict."""
        if num_colonies <= 0:
            return None  # no workers available

        payload = {
            "cmd": "EXPLORE",
            "X": X_raw,
            "Y": y_raw,
            "in_names": self.in_names,
            "out_names": self.out_names,
            "norm_type": self.norm_type,
            "loss_pref": self.desired_loss,
            "living_time": getattr(self.args,"living_time", 10),
            "comm_intervals": getattr(self.args,"communication_intervals", 2),
            "bp_epochs": getattr(self.args,"bp_epochs", 10),
            "out_dir": getattr(self.args,"out_dir","./OUT"),
        }
        # dispatch jobs to all workers
        for w in worker_group:
            comm_mpi.send(payload, dest=w)
        # collect results
        results = []
        for w in worker_group:
            res = comm_mpi.recv(source=w)
            results.append(res)
        # choose best by 'acc'
        best = None
        for r in results:
            if r is None or "error" in r:
                continue
            if best is None or r["acc"] > best["acc"]:
                best = r
        return best

    # NEW: local (single-process) fallback exploration when no MPI workers present
    def _local_explore(self, X_raw, y_raw):
        """Train a handful of local candidates (no MPI); return the best dict."""
        best = None
        for _ in range(max(1, self.local_candidates)):
            c = _train_candidate(
                X_raw, y_raw,
                self.in_names, self.out_names,
                self.norm_type, self.desired_loss,
                living_time=getattr(self.args,"living_time", 10),
                comm_intervals=getattr(self.args,"communication_intervals", 2),
                bp_epochs=getattr(self.args,"bp_epochs", 10),
                out_dir=getattr(self.args,"out_dir","./OUT"),
            )
            if c is None or "error" in c:
                continue
            if best is None or c["acc"] > best["acc"]:
                best = c
        return best

    # NEW: promotion rule — only swap the deployed model if candidate is clearly better
    def _maybe_promote(self, acc_before: float, candidate: dict) -> (bool, float):
        """Decide if we should swap in the candidate. Returns (promoted?, new_acc)."""
        if candidate is None or "acc" not in candidate:
            return False, float("nan")
        cand_acc = float(candidate["acc"])
        target = max(self.acc_threshold, acc_before + self.promote_delta)  # min required improvement
        if np.isnan(acc_before) or cand_acc >= target:
            # promote: replace deployed model and scaler
            self.current_graph     = candidate["graph"]
            self.current_lags      = candidate["lags"]
            self.norm_stats_in     = candidate["norm_stats"]
            self.current_loss      = candidate["loss"]
            self.current_threshold = candidate["threshold"]
            return True, cand_acc
        return False, cand_acc

    # ---------- Worker loop for MPI Case-3 ----------
    # NEW: worker-side loop that receives EXPLORE jobs and replies with a candidate result
    def worker_loop(self):
        while True:
            job = comm_mpi.recv(source=0)    # wait for master
            if job is None:
                continue
            cmd = job.get("cmd", "")
            if cmd == "EXIT":
                break
            if cmd == "EXPLORE":
                res = _train_candidate(
                    job["X"], job["Y"],
                    job["in_names"], job["out_names"],
                    job["norm_type"], job["loss_pref"],
                    job["living_time"], job["comm_intervals"], job["bp_epochs"], job["out_dir"],
                )
                comm_mpi.send(res, dest=0)   # send result back

    # ---------- Main run ----------
    # NEW: orchestrates the entire Case-3 workflow (train W1, evaluate subsequent windows, explore, maybe promote)
    def run_case3(self):
        if rank == 0:
            logger.info(f"[Case3-Pickle] intervals={len(self.intervals)} | acc_thresh={self.acc_threshold}% | loss_pref={self.desired_loss} | MPI_workers={num_colonies}")

            # Window 1: train and freeze
            w1_df = self._window_df(0)
            acc_w1 = self._fit_window1(w1_df)
            if acc_w1 is None or np.isnan(acc_w1):
                logger.error("Training on Window 1 failed or window too small.")
                # tell workers to exit if any
                for w in worker_group:
                    comm_mpi.send({"cmd":"EXIT"}, dest=w)
                return

            self.interval_numbers.append(1)
            self.before_accuracies.append(acc_w1)
            self.retrained_accuracies.append(np.nan)
            self.after_times.append(np.nan)
            logger.info(f"[Case-3] Window 1 accuracy (frozen model, @t*={self.current_threshold:.3f}): {acc_w1:.2f}%")
            logger.info("[Case-3] Deployed model FROZEN. Later windows will explore with MPI but not modify this model unless promotion happens.")

            # Subsequent windows
            for i in range(1, len(self.intervals)):
                df_win = self._window_df(i)
                X_raw, y_raw = self._to_arrays(df_win)
                label  = f"Window {i+1}"

                acc_before = self._eval_deployed_on_window(X_raw, y_raw)   # evaluate deployed model
                self.interval_numbers.append(i+1)
                self.before_accuracies.append(acc_before)
                logger.info(f"[Case-3] {label}: deployed accuracy = {acc_before:.2f}%")

                post_acc = np.nan
                if np.isnan(acc_before) or acc_before < self.acc_threshold:
                    logger.info(f"[Case-3] {label}: below threshold ({self.acc_threshold}%). Exploring candidates{' with MPI' if num_colonies>0 else ''}...")
                    t4 = time.time()
                    best_cand = self._mpi_explore(X_raw, y_raw) if num_colonies>0 else self._local_explore(X_raw, y_raw)
                    promoted, cand_acc = self._maybe_promote(acc_before, best_cand)  # decide promotion
                    t5 = time.time()
                    self.after_times.append(t5 - t4)

                    if best_cand is None:
                        logger.warning(f"[Case-3] {label}: no valid candidate returned.")
                    else:
                        logger.info(f"[Case-3] {label}: best candidate acc = {cand_acc:.2f}% ; "
                                    f"decision target = {max(self.acc_threshold, (acc_before if not np.isnan(acc_before) else 0)+self.promote_delta):.2f}% ; "
                                    f"{'PROMOTED' if promoted else 'kept deployed'}")
                        post_acc = cand_acc if promoted else np.nan
                else:
                    self.train_times.append(np.nan)   # no exploration cost
                    self.after_times.append(np.nan)

                self.retrained_accuracies.append(post_acc)

            # Aggregate summary stats
            finals = []
            for b, r in zip(self.before_accuracies, self.retrained_accuracies):
                finals.append(r if not (isinstance(r, float) and np.isnan(r)) else b)
            avg_final  = float(np.nanmean(finals))
            avg_before = float(np.nanmean(self.before_accuracies))
            logger.info(f"[Case-3] Avg final acc = {avg_final:.2f}% | Avg before = {avg_before:.2f}%")

            # write summary CSV
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
            finally:
                # Tell workers to exit cleanly
                for w in worker_group:
                    comm_mpi.send({"cmd":"EXIT"}, dest=w)

        else:
            # Worker ranks: sit in the explore loop (wait for EXPLORE/EXIT commands)
            self.worker_loop()

# ======================================================================
# ============================== Entrypoints ============================
# ======================================================================

def main():
    args = Args_Parser(sys.argv)     # parse CLI
    logger_setup()                   # configure logging
    if rank == 0: # Main Process
        logger.info(f"Main reporting for duty")
        best_model = environment(args.living_time)  # run classic MPI orchestration
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
        living_colony(                     # run this worker's colony loop
            data=data, 
            living_time=args.living_time, 
            intervals=intervals, 
            cost_type=args.loss_fun, 
            train_epochs=args.bp_epochs, 
            out_dir=args.out_dir,
        )

# ---- Case-3 switch ----
# NEW: if CANTS_CASE3=1, use the Case-3 controller instead of classic MPI flow.
if os.getenv("CANTS_CASE3","0") == "1":
    args = Args_Parser(sys.argv)
    logger_setup()
    _case3 = Colonies(args)   # build controller (rank 0 = master; others = workers)
    colony = _case3           # expose controller at module level
    _case3.run_case3()        # start Case-3 run (rank 0 returns; workers loop until EXIT)
    # rank 0 returns from run_case3; workers exit inside
    sys.exit(0)
else:
    if __name__ == "__main__":
        main()                # fallback: classic CSV/MPI entrypoint
