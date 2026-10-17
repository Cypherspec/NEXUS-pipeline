"""MoleculeNet benchmark: D-MPNN against fingerprint and descriptor baselines.

Protocol: scaffold-balanced 80/10/10 splits (Yang et al. 2019), three seeds per
dataset, early stopping on the validation metric, and a single evaluation of
the selected model on the held-out test set. Classification is scored by
ROC-AUC (averaged over tasks), regression by RMSE.

scikit-learn is used only for the baseline models; the NEXUS package itself
depends on NumPy alone.

    python -m benchmarks.molnet --datasets bace bbbp esol --seeds 3 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "molnet"

CLASSIFICATION = ["bace", "bbbp", "clintox", "tox21", "hiv"]
REGRESSION = ["esol", "freesolv", "lipo"]

# Published reference points for the same datasets and scaffold-split protocol,
# quoted for orientation only: splits are regenerated here with our own seeds,
# so numbers are comparable in protocol but not identical in data.
REFERENCE = {
    "bace":     {"metric": "roc_auc", "chemprop_dmpnn": 0.857, "source": "Yang et al. 2019 (scaffold)"},
    "bbbp":     {"metric": "roc_auc", "chemprop_dmpnn": 0.913, "source": "Yang et al. 2019 (scaffold)"},
    "clintox":  {"metric": "roc_auc", "chemprop_dmpnn": 0.897, "source": "Yang et al. 2019 (scaffold)"},
    "tox21":    {"metric": "roc_auc", "chemprop_dmpnn": 0.754, "source": "Yang et al. 2019 (scaffold)"},
    "hiv":      {"metric": "roc_auc", "chemprop_dmpnn": 0.776, "source": "Yang et al. 2019 (scaffold)"},
    "esol":     {"metric": "rmse",    "chemprop_dmpnn": 0.980, "source": "Yang et al. 2019 (scaffold)"},
    "freesolv": {"metric": "rmse",    "chemprop_dmpnn": 2.180, "source": "Yang et al. 2019 (scaffold)"},
    "lipo":     {"metric": "rmse",    "chemprop_dmpnn": 0.655, "source": "Yang et al. 2019 (scaffold)"},
}

# Per-dataset training budgets, set from measured epoch times so a full
# 3-seed sweep finishes on four CPU cores. Message passing here runs on NumPy,
# so width is the dominant cost: hidden=128 is roughly four times faster per
# epoch than 256 and was chosen for that reason, not tuned on test scores.
BUDGETS = {
    "freesolv": dict(epochs=100, hidden=128, batch_size=32),
    "esol":     dict(epochs=100, hidden=128, batch_size=32),
    "bace":     dict(epochs=80,  hidden=128, batch_size=50),
    "bbbp":     dict(epochs=80,  hidden=128, batch_size=50),
    "clintox":  dict(epochs=80,  hidden=128, batch_size=50),
    "lipo":     dict(epochs=60,  hidden=128, batch_size=64),
    "tox21":    dict(epochs=40,  hidden=128, batch_size=64),
    "hiv":      dict(epochs=15,  hidden=128, batch_size=128),
}


def evaluate_split(name: str, seed: int) -> dict:
    """Train every method on one scaffold split and score it on the test set."""
    from nexus.chem.descriptors import DESCRIPTOR_NAMES, descriptor_matrix
    from nexus.chem.fingerprint import fingerprint_matrix
    from nexus.chem.scaffold import scaffold_overlap, scaffold_split
    from nexus.data import DATASETS, load_dataset
    from nexus.models.metrics import multitask_score
    from nexus.models.train import TrainConfig, predict_calibrated, train_model

    spec = DATASETS[name]
    task_type = spec.task_type
    metric = "roc_auc" if task_type == "classification" else "rmse"
    mols, Y, _, _ = load_dataset(name)
    train_idx, valid_idx, test_idx = scaffold_split(mols, seed=seed)

    def take(idx):
        return [mols[i] for i in idx], Y[idx]

    train_mols, Y_tr = take(train_idx)
    valid_mols, Y_va = take(valid_idx)
    test_mols, Y_te = take(test_idx)

    out: dict = {
        "dataset": name, "seed": seed, "task_type": task_type, "metric": metric,
        "n_total": len(mols), "n_train": len(train_idx), "n_valid": len(valid_idx),
        "n_test": len(test_idx), "n_tasks": Y.shape[1],
        "scaffold_leakage_test": scaffold_overlap(train_idx, test_idx, mols),
        "methods": {},
    }

    # ---- baselines on fixed featurisations
    fp = fingerprint_matrix(mols, kind="morgan", n_bits=2048).astype(np.float32)
    desc = descriptor_matrix(mols)
    desc = np.nan_to_num(desc, nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = desc[train_idx].mean(0), desc[train_idx].std(0)
    sd[sd == 0] = 1.0
    desc_z = (desc - mu) / sd
    combo = np.hstack([fp, desc_z])

    feature_sets = {"ECFP": fp, "descriptors": desc_z, "ECFP+descriptors": combo}
    for feat_name, X in feature_sets.items():
        for model_name in ("RF", "linear"):
            t0 = time.time()
            pred = _baseline(X, Y, train_idx, valid_idx, test_idx, task_type,
                             model_name, seed)
            out["methods"][f"{feat_name} + {model_name}"] = {
                "test_score": multitask_score(Y_te, pred, metric),
                "seconds": round(time.time() - t0, 1),
            }

    # ---- D-MPNN
    budget = BUDGETS.get(name, dict(epochs=60, hidden=256, batch_size=50))
    for variant, extra in (("D-MPNN", None), ("D-MPNN + descriptors", desc_z)):
        cfg = TrainConfig(seed=seed, depth=3, dropout=0.0, patience=15, **budget)
        t0 = time.time()
        result = train_model(
            train_mols, Y_tr, valid_mols, Y_va, task_type, cfg,
            extra_train=None if extra is None else extra[train_idx],
            extra_valid=None if extra is None else extra[valid_idx])
        pred = predict_calibrated(
            result, test_mols, task_type,
            extra=None if extra is None else extra[test_idx])
        out["methods"][variant] = {
            "test_score": multitask_score(Y_te, pred, metric),
            "valid_score": result.best_valid,
            "best_epoch": result.best_epoch,
            "seconds": round(time.time() - t0, 1),
            "n_parameters": result.model.n_parameters(),
        }
    return out


def _baseline(X, Y, train_idx, valid_idx, test_idx, task_type, model_name, seed):
    """Fit a scikit-learn baseline per task; returns test predictions."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge

    fit_idx = np.concatenate([train_idx, valid_idx])
    preds = np.zeros((len(test_idx), Y.shape[1]))
    for t in range(Y.shape[1]):
        y = Y[:, t]
        keep = fit_idx[np.isfinite(y[fit_idx])]
        if len(keep) == 0:
            preds[:, t] = np.nan
            continue
        y_fit = y[keep]
        if task_type == "classification":
            if len(np.unique(y_fit)) < 2:
                preds[:, t] = float(y_fit[0])
                continue
            model = (RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)
                     if model_name == "RF" else
                     LogisticRegression(max_iter=2000, C=1.0))
            model.fit(X[keep], y_fit)
            preds[:, t] = model.predict_proba(X[test_idx])[:, 1]
        else:
            model = (RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=seed)
                     if model_name == "RF" else Ridge(alpha=1.0))
            model.fit(X[keep], y_fit)
            preds[:, t] = model.predict(X[test_idx])
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=CLASSIFICATION + REGRESSION)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(d, s) for d in args.datasets for s in range(args.seeds)
            if not (OUT / f"{d}_seed{s}.json").exists()]
    print(f"{len(jobs)} runs to do ({len(args.datasets)} datasets x {args.seeds} seeds)")

    with ProcessPoolExecutor(args.workers) as pool:
        futures = {pool.submit(evaluate_split, d, s): (d, s) for d, s in jobs}
        for i, fut in enumerate(as_completed(futures), 1):
            d, s = futures[fut]
            try:
                res = fut.result()
                (OUT / f"{d}_seed{s}.json").write_text(json.dumps(res, indent=2))
                best = max(res["methods"].items(),
                           key=lambda kv: kv[1]["test_score"] * (1 if res["metric"] == "roc_auc" else -1))
                print(f"[{i}/{len(jobs)}] {d} seed{s}: "
                      f"D-MPNN {res['methods']['D-MPNN']['test_score']:.4f}, "
                      f"best={best[0]} {best[1]['test_score']:.4f}", flush=True)
            except Exception:
                print(f"[{i}/{len(jobs)}] {d} seed{s} FAILED\n{traceback.format_exc()}",
                      flush=True)


if __name__ == "__main__":
    main()
