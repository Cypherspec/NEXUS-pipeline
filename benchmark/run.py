"""Run the benchmark: NEXUS + baselines on the official CC18 10-fold splits.

Each (task, fold) writes results/raw/<task_id>/fold<k>.npz holding every
prediction needed to recompute all methods and ablations. Completed folds
are skipped, so the run can be interrupted and resumed.

    python -m benchmark.run --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
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
V2_STACK = ("lightgbm", "logreg")


def run_fold_v2(task_id: int, fold: int) -> str:
    """NEXUS v2 only; baselines are shared with the v1 run on identical splits."""
    from sklearn.preprocessing import LabelEncoder

    from benchmark.suite import load_task
    from nexus import NexusClassifier

    warnings.filterwarnings("ignore")
    out = RAW_V2 / str(task_id) / f"fold{fold}.npz"
    if out.exists():
        return f"{task_id}/{fold} cached"
    X, y, folds, _ = load_task(task_id)
    y = LabelEncoder().fit_transform(y.astype(str))
    k = int(y.max()) + 1
    tr, te = folds[fold]
    X_tr, X_te, y_tr, y_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y[tr], y[te]
    t0 = time.perf_counter()
    m = NexusClassifier(stack_models=V2_STACK, random_state=fold, n_jobs=1).fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0
    assert list(m.classes_) == list(range(k))
    t0 = time.perf_counter()
    proba = m.predict_proba(X_te)
    pred_s = time.perf_counter() - t0
    members = m.predict_proba_members(X_te)
    names = list(members)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp, names=np.array(names), y_train=y_tr.astype(np.int16), y_test=y_te.astype(np.int16),
        oof=np.stack([m.report_.oof_proba[n] for n in names]).astype(np.float32),
        bagged_test=np.stack([members[n] for n in names]).astype(np.float32),
        nexus_test=proba.astype(np.float32), nexus_weights=np.array([m.weights_[n] for n in names]),
        nexus_temperature=m.temperature_,
        member_fit_seconds=np.array([m.report_.fit_seconds[n] for n in names]),
        nexus_fit_seconds=fit_s, nexus_predict_seconds=pred_s,
    )
    tmp.rename(out)
    return f"{task_id}/{fold} v2 done in {fit_s:.0f}s"


def run_fold(task_id: int, fold: int) -> str:
    from sklearn.preprocessing import LabelEncoder

    from benchmark.baselines import fit_single, fit_tuned_lightgbm
    from benchmark.suite import load_task
    from nexus import NexusClassifier
    from nexus.zoo import DEFAULT_ZOO

    warnings.filterwarnings("ignore")
    out = RAW / str(task_id) / f"fold{fold}.npz"
    if out.exists():
        return f"{task_id}/{fold} cached"
    X, y, folds, meta = load_task(task_id)
    y = LabelEncoder().fit_transform(y.astype(str))
    k = int(y.max()) + 1
    tr, te = folds[fold]
    X_tr, X_te, y_tr, y_te = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y[tr], y[te]

    t0 = time.perf_counter()
    nexus = NexusClassifier(stack_models=None, random_state=fold, n_jobs=1).fit(X_tr, y_tr)
    nexus_fit = time.perf_counter() - t0
    assert list(nexus.classes_) == list(range(k)), "class missing from training fold"
    t0 = time.perf_counter()
    nexus_test = nexus.predict_proba(X_te)
    nexus_predict = time.perf_counter() - t0
    members = nexus.predict_proba_members(X_te)
    names = [s.name for s in DEFAULT_ZOO]

    single_test, single_time = [], []
    for spec in DEFAULT_ZOO:
        t0 = time.perf_counter()
        m = fit_single(spec, X_tr, y_tr, k, seed=fold)
        single_time.append(time.perf_counter() - t0)
        single_test.append(m.predict_proba(X_te))

    t0 = time.perf_counter()
    tuned, tuned_cfg = fit_tuned_lightgbm(X_tr, y_tr, k, seed=fold)
    tuned_time = time.perf_counter() - t0
    tuned_test = tuned.predict_proba(X_te)

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(
        tmp,
        names=np.array(names),
        y_train=y_tr.astype(np.int16), y_test=y_te.astype(np.int16),
        oof=np.stack([nexus.report_.oof_proba[n] for n in names]).astype(np.float32),
        bagged_test=np.stack([members[n] for n in names]).astype(np.float32),
        single_test=np.stack(single_test).astype(np.float32),
        tuned_lgbm_test=tuned_test.astype(np.float32),
        nexus_test=nexus_test.astype(np.float32),
        nexus_weights=np.array([nexus.weights_[n] for n in names]),
        nexus_temperature=nexus.temperature_,
        member_fit_seconds=np.array([nexus.report_.fit_seconds[n] for n in names]),
        single_fit_seconds=np.array(single_time),
        nexus_fit_seconds=nexus_fit, nexus_predict_seconds=nexus_predict,
        tuned_lgbm_seconds=tuned_time, tuned_lgbm_config=json.dumps(tuned_cfg),
    )
    tmp.rename(out)
    return f"{task_id}/{fold} done in {nexus_fit + sum(single_time) + tuned_time:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks", type=int, nargs="*", help="subset of task ids")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--variant", choices=["v1", "v2"], default="v1")
    args = ap.parse_args()

    tasks = pd.read_csv(ROOT / "benchmark" / "tasks.csv")
    tasks = tasks[tasks.included]
    if args.tasks:
        tasks = tasks[tasks.task_id.isin(args.tasks)]
    # Download everything up front in the parent so workers never race on cache.
    from benchmark.suite import load_task
    for tid in tasks.task_id:
        load_task(int(tid))
    jobs = [(int(t), f) for t in tasks.task_id for f in range(args.folds)]
    with ProcessPoolExecutor(args.workers) as pool:
        fn = run_fold_v2 if args.variant == "v2" else run_fold
        futs = {pool.submit(fn, t, f): (t, f) for t, f in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            t, f = futs[fut]
            try:
                msg = fut.result()
            except Exception:
                msg = f"{t}/{f} FAILED\n{traceback.format_exc()}"
            print(f"[{i}/{len(jobs)}] {msg}", flush=True)


if __name__ == "__main__":
    main()
