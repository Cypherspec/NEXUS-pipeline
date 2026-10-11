"""Development harness on datasets disjoint from OpenML-CC18.

All NEXUS v2 design decisions are made here, never on the CC18 test suite.
Member predictions are cached per (dataset, fold, member), so ensembling
strategies can be compared offline without refitting.

    python -m benchmark.dev members --workers 2 --members lightgbm catboost_es ...
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "dev" / "cache"

# OpenML data ids, none of which is (or is derived from) a CC18 dataset.
DEV_DATASETS = {
    40981: "Australian", 41143: "jasmine", 41146: "sylvine", 40945: "Titanic",
    1479: "hill-valley", 334: "monks-problems-2", 40691: "wine-quality-red",
    59: "ionosphere", 40: "sonar", 181: "yeast", 30: "page-blocks", 41: "glass",
    1460: "banana", 1069: "pc2", 979: "waveform-5000", 53: "heart-statlog",
}
N_OUTER = 5
N_INNER = 5


def load_dev(did: int):
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelEncoder

    X, y = fetch_openml(data_id=did, return_X_y=True, as_frame=True, parser="auto",
                        data_home=str(ROOT / "data" / "openml"))
    X = X.reset_index(drop=True)
    y = LabelEncoder().fit_transform(pd.Series(y).astype(str))
    # Merge classes too rare to appear in every outer training fold.
    counts = np.bincount(y)
    rare = np.flatnonzero(counts < 2 * N_OUTER)
    if len(rare):
        y = np.where(np.isin(y, rare), rare[0], y)
        y = LabelEncoder().fit_transform(y)
    folds = list(StratifiedKFold(N_OUTER, shuffle=True, random_state=0).split(X, y))
    return X, y, folds


def member_path(did, fold, name):
    return CACHE / str(did) / f"fold{fold}" / f"{name.replace(':', '__')}.npz"


def fit_member(did: int, fold: int, name: str) -> str:
    """Cross-fit one member on the outer-train part; store OOF + bagged test preds."""
    from nexus.classifier import _fit_one, _make_folds
    from nexus.preprocessing import dense_view, split_columns, tree_view
    from nexus.zoo import ALL_SPECS

    warnings.filterwarnings("ignore")
    out = member_path(did, fold, name)
    if out.exists():
        return f"{did}/{fold}/{name} cached"
    X, y, folds = load_dev(did)
    tr, te = folds[fold]
    X_tr, X_te, y_tr = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True), y[tr]
    k = int(y.max()) + 1

    if name.startswith("L2:"):
        # L2:<member>@<l1set>: member trained on [X, L1 OOF probs]; test uses bagged L1 test preds.
        member, l1set = name[3:].split("@")
        l1 = L1_SETS[l1set]
        oof_l1 = np.concatenate([np.load(member_path(did, fold, m))["oof"] for m in l1], axis=1)
        te_l1 = np.concatenate([np.load(member_path(did, fold, m))["test"] for m in l1], axis=1)
        cols = [f"__l1_{i}" for i in range(oof_l1.shape[1])]
        X_tr = pd.concat([X_tr, pd.DataFrame(oof_l1, columns=cols)], axis=1)
        X_te = pd.concat([X_te, pd.DataFrame(te_l1, columns=cols)], axis=1)
        spec = ALL_SPECS[member]
    else:
        spec = ALL_SPECS[name.split("#")[0]]
    # "<member>#r<j>": an independent repetition of cross-fitting with new fold seeds.
    repeat = int(name.split("#r")[1]) if "#r" in name else 0

    numeric, categorical = split_columns(X_tr)
    view = tree_view(numeric, categorical) if spec.view == "tree" else dense_view(numeric, categorical)
    inner = _make_folds(y_tr, N_INNER, seed=fold + 1000 * repeat)
    oof = np.zeros((len(X_tr), k))
    test = np.zeros((len(X_te), k))
    t0 = time.perf_counter()
    for f, (a, b) in enumerate(inner):
        model, pred, _ = _fit_one(spec, view, X_tr.iloc[a], y_tr[a], X_tr.iloc[b], k, fold * 100 + f + 1000 * repeat)
        oof[b] = pred
        test += model.predict_proba(X_te) / len(inner)
    secs = time.perf_counter() - t0
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, oof=oof.astype(np.float32), test=test.astype(np.float32),
                        y_train=y_tr.astype(np.int16), y_test=y[te].astype(np.int16), seconds=secs)
    return f"{did}/{fold}/{name} {secs:.0f}s"


V1 = ["lightgbm", "xgboost", "catboost", "random_forest", "extra_trees", "logreg", "knn", "mlp"]
L1_SETS = {"v1": V1}


def run_members(names, workers):
    jobs = [(d, f, n) for n in names for d in DEV_DATASETS for f in range(N_OUTER)]
    for d in DEV_DATASETS:
        load_dev(d)
    with ProcessPoolExecutor(workers) as pool:
        futs = {pool.submit(fit_member, *j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                msg = fut.result()
            except Exception as e:
                msg = f"{futs[fut]} FAILED {type(e).__name__}: {e}"
            if "cached" not in msg:
                print(f"[{i}/{len(jobs)}] {msg}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["members"])
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    run_members(a.members, a.workers)
