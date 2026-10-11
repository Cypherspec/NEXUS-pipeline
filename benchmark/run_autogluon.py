"""AutoGluon-Tabular on the same CC18 splits, time-matched to NEXUS.

Runs in a separate environment that has autogluon.tabular installed:

    /opt/sota/bin/python -m benchmark.run_autogluon --workers 4

Per fold, AutoGluon (presets="best_quality", 1 CPU, eval_metric="log_loss")
gets a time limit equal to the wall-clock time NEXUS v2 needed to fit that
same fold (from results/raw_v2/<task>/fold<k>.npz), with a 60 s floor.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
RAW_V2 = ROOT / "results" / "raw_v2"
OUT = ROOT / "results" / "raw_autogluon"
MIN_BUDGET = 60
BUDGET_SCALE = float(os.environ.get("AG_BUDGET_SCALE", "1.0"))


def run_fold(task_id: int, fold: int) -> str:
    from autogluon.tabular import TabularPredictor
    from sklearn.preprocessing import LabelEncoder

    from benchmark.suite import load_task

    warnings.filterwarnings("ignore")
    tag = "" if BUDGET_SCALE == 1.0 else f"_x{BUDGET_SCALE:g}"
    out = OUT.with_name(OUT.name + tag) / str(task_id) / f"fold{fold}.npz"
    if out.exists():
        return f"{task_id}/{fold} cached"
    ref = np.load(RAW / str(task_id) / f"fold{fold}.npz")
    v2 = np.load(RAW_V2 / str(task_id) / f"fold{fold}.npz")
    budget = max(MIN_BUDGET, float(v2["nexus_fit_seconds"])) * BUDGET_SCALE

    X, y, folds, _ = load_task(task_id)
    y = LabelEncoder().fit_transform(y.astype(str))
    k = int(y.max()) + 1
    tr, te = folds[fold]
    train = X.iloc[tr].reset_index(drop=True).copy()
    train["__label__"] = y[tr]
    X_te = X.iloc[te].reset_index(drop=True)
    assert np.array_equal(y[te], ref["y_test"]), "label encoding mismatch with NEXUS run"

    tmp = tempfile.mkdtemp(prefix=f"ag_{task_id}_{fold}_")
    try:
        t0 = time.perf_counter()
        pred = TabularPredictor(label="__label__", eval_metric="log_loss", path=tmp, verbosity=0,
                                problem_type="binary" if k == 2 else "multiclass").fit(
            train, presets="best_quality", time_limit=budget, num_cpus=1, num_gpus=0)
        fit_s = time.perf_counter() - t0
        p = pred.predict_proba(X_te)
        proba = np.zeros((len(X_te), k))
        for c in p.columns:
            proba[:, int(c)] = p[c].to_numpy()
        proba /= proba.sum(axis=1, keepdims=True)
        best = pred.model_best
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, proba=proba.astype(np.float32), y_test=y[te].astype(np.int16),
                        fit_seconds=fit_s, budget=budget, best_model=best)
    return f"{task_id}/{fold} done in {fit_s:.0f}s (budget {budget:.0f}s, best {best})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks", type=int, nargs="*")
    ap.add_argument("--folds", type=int, default=10)
    args = ap.parse_args()
    tasks = pd.read_csv(ROOT / "benchmark" / "tasks.csv")
    tasks = tasks[tasks.included]
    if args.tasks:
        tasks = tasks[tasks.task_id.isin(args.tasks)]
    jobs = [(int(t), f) for t in tasks.task_id for f in range(args.folds)
            if (RAW / str(int(t)) / f"fold{f}.npz").exists() and (RAW_V2 / str(int(t)) / f"fold{f}.npz").exists()]
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(run_fold, t, f): (t, f) for t, f in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            t, f = futs[fut]
            try:
                msg = fut.result()
            except Exception:
                msg = f"{t}/{f} FAILED\n{traceback.format_exc()}"
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
