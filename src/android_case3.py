#!/usr/bin/env python3
"""
Android Case-3 (CSV, first 3 intervals): windowed testing with threshold-triggered retraining.
- Train on interval_1 (80/20 split via Timeseries.split_data), record Window-1 accuracy.
- Test that deployed model on interval_2 full data; if acc < FITNESS_GLOBAL, retrain on interval_2 (same params).
- Repeat for interval_3.
Writes OUT/android_case3_summary.csv with before/after accuracies and timing.
"""

import os
import sys
import time
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch


from colony import Colony
from timeseries import Timeseries as _TSClass  # to reuse split_data()

# --- auto-tag support (no per-window typing) ---
_ARRAY_SOURCE = {}
_TRAIN_CALL = 0
def _remember_source(X, path):
    _ARRAY_SOURCE[id(X)] = os.fspath(path)
    
# --- hotfix: patch Node.adjust_lag without touching node.py ---
import node as _node

def _adjust_lag_patched(self, lags):
    # original intent: self.lag = round(self.point.get_z() * lags)
    self.lag = round(self.point.get_z() * lags)

# Replace the buggy method (fixes NameError: segslf)
_node.Node.adjust_lag = _adjust_lag_patched

import colony as _colony
import graphCants as _graph_cants
_colony.Graph = _graph_cants.Graph
# --- end hotfix ---


# ----------------------- small local helpers (inlined) -----------------------

def _read_threshold(default_val=90.0) -> float:
    for k in ("FITNESS_GLOBAL", "fitness_global", "ACC_THRESHOLD", "CANTS_CASE3_ACC_THRESH"):
        v = os.getenv(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                pass
    return default_val

def _map_loss_name(name: str) -> str:
    """Normalize user loss aliases to Graph/Colony cost_type names."""
    if not name:
        return "bicross_entropy"
    n = str(name).lower()
    if "bce" in n or "bicross" in n:
        return "bicross_entropy"
    if "cross" in n and "bi" not in n:
        return "cross_entropy"
    if "mse" in n:
        return "mse"
    return "bicross_entropy"

def _compute_norm_stats(X_train: np.ndarray, mode: str = "none") -> dict:
    mode = (mode or "none").lower()
    if mode == "minmax":
        return {"mode": "minmax", "min": X_train.min(axis=0), "max": X_train.max(axis=0)}
    if mode in ("mean_std", "zscore", "standard"):
        std = X_train.std(axis=0)
        std[std == 0.0] = 1.0
        return {"mode": "mean_std", "mean": X_train.mean(axis=0), "std": std}
    return {"mode": "none"}

def _apply_norm(X: np.ndarray, st: dict) -> np.ndarray:
    m = st.get("mode", "none")
    if m == "minmax":
        denom = (st["max"] - st["min"]).copy()
        denom[denom == 0.0] = 1.0
        return (X - st["min"]) / denom
    if m == "mean_std":
        return (X - st["mean"]) / st["std"]
    return X

def _align_pred_y(preds: np.ndarray, y: np.ndarray, lags: int = 0):
    """Trim labels by lags so preds and y line up."""
    preds = np.asarray(preds).reshape(-1)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    y_eff = y[lags:]
    n = min(len(preds), len(y_eff))
    return preds[:n], y_eff[:n]

def _binary_acc(preds: np.ndarray, y: np.ndarray, threshold: float = 0.5, lags: int = 0) -> float:
    """Binary accuracy (%) after thresholding sigmoid outputs."""
    P, Y = _align_pred_y(preds, y, lags)
    if len(Y) == 0:
        return float("nan")
    labels = (P >= threshold).astype(np.float32)
    y_labels = (Y >= 0.5).astype(np.float32)
    return float((labels == y_labels).mean()) * 100.0

def _best_threshold(preds: np.ndarray, y: np.ndarray, lags: int = 0) -> float:
    """Pick t* in [0.05, 0.95] that maximizes accuracy on a holdout."""
    P, Y = _align_pred_y(preds, y, lags)
    if len(Y) == 0:
        return 0.5
    Yb = (Y >= 0.5).astype(np.float32)
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        s = ((P >= t).astype(np.float32) == Yb).mean()
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t

def _split_80_20_with_repo_logic(X: np.ndarray, y: np.ndarray):
    """Use the repo’s Timeseries.split_data (80/20 chronological)."""
    # Timeseries.split_data(self, in_data, out_data, split_ratio=0.8, norm_fun=None)
    train_in, test_in, train_out, test_out = _TSClass.split_data(None, X, y, split_ratio=0.8, norm_fun=None)
    return (np.asarray(train_in), np.asarray(test_in),
            np.asarray(train_out), np.asarray(test_out))

def _train_candidate(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    in_names,
    out_names,
    norm_type: str,
    loss_pref: str,
    living_time: int,
    comm_intervals: int,  # kept for API parity (unused here)
    bp_epochs: int,
    out_dir: str,
):
    """
    Train one Colony on a single interval (80/20 split via Timeseries.split_data).
    Returns the best Graph and metadata.
    """
    if X_raw is None or y_raw is None or len(X_raw) < 10:
        return {"error": "insufficient_data"}

    # 80/20 chronological split (repo function)
    X_tr_raw, X_te_raw, y_tr_raw, y_te_raw = _split_80_20_with_repo_logic(X_raw, y_raw)

    # Fit normalization on train only; apply to both
    norm_stats = _compute_norm_stats(X_tr_raw, norm_type)
    X_tr = _apply_norm(X_tr_raw, norm_stats).astype(np.float32)
    X_te = _apply_norm(X_te_raw, norm_stats).astype(np.float32)

    
    global _TRAIN_CALL
    _TRAIN_CALL += 1
    src = _ARRAY_SOURCE.get(id(X_raw), "")
    src_name = Path(src).name if src else ""
    
    
    # Minimal Timeseries-like container the Graph/Colony can consume
    class _TS:
        pass
    ts = _TS()
    ts._interval  = src_name
    ts._stage     = f"train#{_TRAIN_CALL}"
    ts.input_names  = list(in_names)
    ts.output_names = list(out_names)
    ts.train_input  = X_tr
    ts.train_output = y_tr_raw.astype(np.float32)
    ts.test_input   = X_te
    ts.test_output  = y_te_raw.astype(np.float32)

    # ---- Fixed colony params (kept constant across retraining). Colonies can change, params stay. ----
    NUM_ANTS = 6
    POP_SIZE = 10
    NUM_ITRS = max(3, int(living_time))   # number of final graphs to try
    USE_TORCH = True

    c = Colony(
        num_ants=NUM_ANTS,
        population_size=POP_SIZE,
        input_names=ts.input_names,
        output_names=ts.output_names,
        data=ts,
        num_itrs=NUM_ITRS,
        worker_id=0,
        out_dir=out_dir,
        use_torch=USE_TORCH,
    )
    c.life_threads(num_itrs=NUM_ITRS, cost_type=_map_loss_name(loss_pref), train_epochs=int(bp_epochs))

    if not c.best_solutions:
        return {"error": "no_solution"}

    # best_solutions: list of (fit, graph), lower fit is better
    best_graph = c.best_solutions[0][1]

    # Learn a decision threshold t* on the train split (using sigmoid outputs)
    Xtr_t = torch.tensor(ts.train_input, dtype=torch.float32)
    ytr_t = torch.tensor(ts.train_output, dtype=torch.float32)
    preds_tr, _, _ = best_graph.single_thrust(Xtr_t, ytr_t, prt=False, cal_gradient=False, cost_type=_map_loss_name(loss_pref))
    if isinstance(preds_tr[0], torch.Tensor):
        P_tr = torch.stack(preds_tr).detach().cpu().numpy().reshape(-1)
    else:
        P_tr = np.stack(preds_tr).reshape(-1)

    t_star = _best_threshold(P_tr, ts.train_output, lags=best_graph.lags)

    # Report accuracy on the FULL interval (train+test) using t*
    X_full = _apply_norm(X_raw, norm_stats).astype(np.float32)
    Xf_t   = torch.tensor(X_full, dtype=torch.float32)
    yf_t   = torch.tensor(y_raw.astype(np.float32), dtype=torch.float32)
    preds_full, _, _ = best_graph.single_thrust(Xf_t, yf_t, prt=False, cal_gradient=False, cost_type=_map_loss_name(loss_pref))
    if isinstance(preds_full[0], torch.Tensor):
        P_full = torch.stack(preds_full).detach().cpu().numpy().reshape(-1)
    else:
        P_full = np.stack(preds_full).reshape(-1)
    acc_full = _binary_acc(P_full, y_raw, threshold=t_star, lags=best_graph.lags)

    return {
        "graph": best_graph,
        "norm_stats": norm_stats,
        "lags": int(best_graph.lags),
        "loss": _map_loss_name(loss_pref),
        "threshold": float(t_star),
        "acc": float(acc_full),
    }

# ----------------------- end helpers -----------------------

def _load_interval_csv(path: Path, in_cols, out_col="Malware"):
    df = pd.read_csv(path)
    X = df[in_cols].astype(np.float32).values
    y = df[[out_col]].astype(np.float32).values
    _remember_source(X, path)  
    return X, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",      type=str, required=True)
    ap.add_argument("--interval_files",type=str, required=True, help="space-separated file names for intervals (we will use first 3)")
    ap.add_argument("--input_names",   type=str, required=True)
    ap.add_argument("--output_names",  type=str, default="Malware")
    ap.add_argument("--out_dir",       type=str, default="OUT")
    ap.add_argument("--living_time",   type=int, default=60)
    ap.add_argument("--comm_interval", type=int, default=2)
    ap.add_argument("--bp_epochs",     type=int, default=9)
    ap.add_argument("--norm_type",     type=str, default="minmax")
    ap.add_argument("--loss_fun",      type=str, default="bicross_entropy")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs/outputs
    in_names  = [x.strip() for x in args.input_names.split()]
    out_names = [x.strip() for x in args.output_names.split()]
    assert len(out_names) == 1 and out_names[0] == "Malware", "Expect a single output: Malware"

    # Pick first 3 intervals
    files_all = [x.strip() for x in args.interval_files.split() if x.strip()]
    if len(files_all) < 3:
        print(f"Need at least 3 files; got {files_all}", file=sys.stderr)
        sys.exit(1)
    f1, f2, f3 = [data_dir / files_all[i] for i in range(3)]
    for p in (f1, f2, f3):
        if not p.exists():
            print(f"Missing interval file: {p}", file=sys.stderr); sys.exit(1)

    # Settings
    acc_thresh   = _read_threshold(90.0)  # percent
    loss_pref    = _map_loss_name(args.loss_fun)
    lt           = int(args.living_time)
    ci           = int(args.comm_interval)
    bp           = int(args.bp_epochs)
    norm_type    = args.norm_type

    # Tracking
    interval_numbers     = []
    before_accuracies    = []
    retrained_accuracies = []
    base_times, train_times, after_times = [], [], []

    # ===== Window 1: train/freeze =====
    X1_raw, y1_raw = _load_interval_csv(f1, in_names, out_col="Malware")
    t0 = time.time()
    cand1 = _train_candidate(
        X1_raw, y1_raw, in_names, out_names,
        norm_type=norm_type, loss_pref=loss_pref,
        living_time=lt, comm_intervals=ci, bp_epochs=bp, out_dir=str(out_dir),
    )
    t1 = time.time()
    if cand1 is None or "error" in cand1:
        print(f"[Case-3] Window 1 training failed: {cand1}", file=sys.stderr); sys.exit(1)

    graph       = cand1["graph"]
    norm_stats  = cand1["norm_stats"]
    lags        = int(cand1["lags"])
    used_loss   = cand1["loss"]
    t_star      = float(cand1["threshold"])
    acc_w1_full = float(cand1["acc"])
    interval_numbers.append(1)
    before_accuracies.append(acc_w1_full)
    retrained_accuracies.append(np.nan)
    base_times.append(np.nan)
    train_times.append(t1 - t0)
    after_times.append(np.nan)

    print(f"[Case-3] Window 1: trained ({used_loss}) @t*={t_star:.3f} ; acc={acc_w1_full:.2f}%")

    # ===== Window 2: evaluate deployed; maybe retrain on W2 =====
    X2_raw, y2_raw = _load_interval_csv(f2, in_names, out_col="Malware")
    X2 = _apply_norm(X2_raw, norm_stats)
    t0 = time.time()
    X2_t = torch.tensor(X2, dtype=torch.float32)
    y2_t = torch.tensor(y2_raw, dtype=torch.float32)
    preds2, _, _ = graph.single_thrust(X2_t, y2_t, prt=False, cal_gradient=False, cost_type=used_loss)
    P2 = (torch.stack(preds2).detach().cpu().numpy() if isinstance(preds2[0], torch.Tensor) else np.stack(preds2)).reshape(-1)
    acc2_before = _binary_acc(P2, y2_raw, threshold=t_star, lags=lags)
    t1 = time.time()
    base_times.append(t1 - t0)

    interval_numbers.append(2)
    before_accuracies.append(acc2_before)
    post_acc2 = np.nan

    if np.isnan(acc2_before) or acc2_before < acc_thresh:
        print(f"[Case-3] Window 2: {acc2_before:.2f}% < {acc_thresh:.2f}% → retraining on W2")
        t2 = time.time()
        cand2 = _train_candidate(
            X2_raw, y2_raw, in_names, out_names,
            norm_type=norm_type, loss_pref=loss_pref,
            living_time=lt, comm_intervals=ci, bp_epochs=bp, out_dir=str(out_dir),
        )
        t3 = time.time()
        train_times.append(t3 - t2)
        if cand2 and "error" not in cand2:
            graph      = cand2["graph"]
            norm_stats = cand2["norm_stats"]
            lags       = int(cand2["lags"])
            used_loss  = cand2["loss"]
            t_star     = float(cand2["threshold"])
            post_acc2  = float(cand2["acc"])
            print(f"[Case-3] Window 2 retrained: acc={post_acc2:.2f}% @t*={t_star:.3f}")
        else:
            print(f"[Case-3] Window 2 retraining failed: {cand2}", file=sys.stderr)
        after_times.append(t3 - t2)
        retrained_accuracies.append(post_acc2)
    else:
        print(f"[Case-3] Window 2: meets threshold → no retraining")
        train_times.append(np.nan)
        after_times.append(np.nan)
        retrained_accuracies.append(np.nan)

    # ===== Window 3: evaluate deployed (possibly updated); maybe retrain on W3 =====
    X3_raw, y3_raw = _load_interval_csv(f3, in_names, out_col="Malware")
    X3 = _apply_norm(X3_raw, norm_stats)
    t0 = time.time()
    X3_t = torch.tensor(X3, dtype=torch.float32)
    y3_t = torch.tensor(y3_raw, dtype=torch.float32)
    preds3, _, _ = graph.single_thrust(X3_t, y3_t, prt=False, cal_gradient=False, cost_type=used_loss)
    P3 = (torch.stack(preds3).detach().cpu().numpy() if isinstance(preds3[0], torch.Tensor) else np.stack(preds3)).reshape(-1)
    acc3_before = _binary_acc(P3, y3_raw, threshold=t_star, lags=lags)
    t1 = time.time()
    base_times.append(t1 - t0)

    interval_numbers.append(3)
    before_accuracies.append(acc3_before)
    post_acc3 = np.nan

    if np.isnan(acc3_before) or acc3_before < acc_thresh:
        print(f"[Case-3] Window 3: {acc3_before:.2f}% < {acc_thresh:.2f}% → retraining on W3")
        t2 = time.time()
        cand3 = _train_candidate(
            X3_raw, y3_raw, in_names, out_names,
            norm_type=norm_type, loss_pref=loss_pref,
            living_time=lt, comm_intervals=ci, bp_epochs=bp, out_dir=str(out_dir),
        )
        t3 = time.time()
        train_times.append(t3 - t2)
        if cand3 and "error" not in cand3:
            graph      = cand3["graph"]
            norm_stats = cand3["norm_stats"]
            lags       = int(cand3["lags"])
            used_loss  = cand3["loss"]
            t_star     = float(cand3["threshold"])
            post_acc3  = float(cand3["acc"])
            print(f"[Case-3] Window 3 retrained: acc={post_acc3:.2f}% @t*={t_star:.3f}")
        else:
            print(f"[Case-3] Window 3 retraining failed: {cand3}", file=sys.stderr)
        after_times.append(t3 - t2)
        retrained_accuracies.append(post_acc3)
    else:
        print(f"[Case-3] Window 3: meets threshold → no retraining")
        train_times.append(np.nan)
        after_times.append(np.nan)
        retrained_accuracies.append(np.nan)

    # ===== Summary =====
    finals = []
    for b, r in zip(before_accuracies, retrained_accuracies):
        finals.append(r if not (isinstance(r, float) and np.isnan(r)) else b)
    avg_final  = float(np.nanmean(finals))
    avg_before = float(np.nanmean(before_accuracies))
    print(f"[Case-3] Avg final acc={avg_final:.2f}% | Avg before={avg_before:.2f}%")

    df_sum = pd.DataFrame({
        "Interval":      interval_numbers,
        "Acc_Before":    before_accuracies,
        "Acc_After":     retrained_accuracies,
        "Base_s":        base_times[:len(interval_numbers)],
        "Train_s":       train_times[:len(interval_numbers)],
        "After_s":       after_times[:len(interval_numbers)],
    })
    out_csv = out_dir / "android_case3_summary.csv"
    df_sum.to_csv(out_csv, index=False)
    print(f"[Case-3] Wrote summary CSV → {out_csv}")

if __name__ == "__main__":
    main()
