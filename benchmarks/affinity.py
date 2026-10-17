"""Drug-target binding affinity prediction on the Davis kinase panel.

Davis et al. (2011): 68 inhibitors against 442 kinases, dissociation constants
converted to pKd = -log10(Kd / 1e9). Three splits are reported because they
answer different questions:

  random      -- interpolation within known drugs and targets
  cold-drug   -- unseen compounds against known targets (the lead-discovery case)
  cold-target -- known compounds against an unseen kinase (the selectivity case)

Metrics: RMSE, Pearson, Spearman and the concordance index (the probability
that two pairs are ranked in the correct order), which is the standard DTA
metric.

    python -m benchmarks.affinity
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nexus.bio.protein import protein_feature_matrix
from nexus.chem.fingerprint import fingerprint_matrix
from nexus.chem.smiles import read_smiles
from nexus.data import load_davis
from nexus.models.metrics import pearson, rmse, spearman

ROOT = Path(__file__).resolve().parent.parent
SEEDS = (0, 1, 2)


def concordance_index(y: np.ndarray, pred: np.ndarray, max_pairs: int = 2_000_000,
                      seed: int = 0) -> float:
    """P(correctly ordered | the pair differs), sampled for large sets."""
    n = len(y)
    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 > max_pairs:
        i = rng.integers(0, n, max_pairs)
        j = rng.integers(0, n, max_pairs)
    else:
        i, j = np.triu_indices(n, k=1)
    different = y[i] != y[j]
    if not different.any():
        return float("nan")
    i, j = i[different], j[different]
    correct = np.sign(y[i] - y[j]) == np.sign(pred[i] - pred[j])
    tied = pred[i] == pred[j]
    return float((correct.sum() + 0.5 * tied.sum()) / len(i))


def make_splits(n_drugs: int, n_targets: int, drug_idx: np.ndarray,
                target_idx: np.ndarray, kind: str, seed: int):
    rng = np.random.default_rng(seed)
    n = len(drug_idx)
    if kind == "random":
        order = rng.permutation(n)
        cut = int(0.8 * n)
        return order[:cut], order[cut:]
    if kind == "cold-drug":
        held = set(rng.choice(n_drugs, max(1, n_drugs // 5), replace=False))
        test = np.array([k for k in range(n) if drug_idx[k] in held])
    elif kind == "cold-target":
        held = set(rng.choice(n_targets, max(1, n_targets // 5), replace=False))
        test = np.array([k for k in range(n) if target_idx[k] in held])
    else:
        raise ValueError(kind)
    train = np.setdiff1d(np.arange(n), test)
    return train, test


def main() -> None:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    smiles, sequences, Y = load_davis()
    mols = [read_smiles(s) for s in smiles]
    ok = [i for i, m in enumerate(mols) if m is not None]
    print(f"Davis: {len(smiles)} drugs ({len(ok)} parsed), {len(sequences)} targets, "
          f"{Y.shape} affinity matrix")

    drug_features = fingerprint_matrix([mols[i] for i in ok], kind="morgan",
                                       n_bits=1024).astype(np.float64)
    target_features = protein_feature_matrix(sequences, k=3, n_bits=512)

    drug_idx, target_idx, values = [], [], []
    for a, i in enumerate(ok):
        for j in range(Y.shape[1]):
            if np.isfinite(Y[i, j]):
                drug_idx.append(a)
                target_idx.append(j)
                values.append(Y[i, j])
    drug_idx = np.array(drug_idx)
    target_idx = np.array(target_idx)
    # Davis ships Kd in nM; the standard transform is pKd = -log10(Kd / 1e9).
    y = -np.log10(np.array(values) / 1e9)
    X = np.hstack([drug_features[drug_idx], target_features[target_idx]])
    print(f"{len(y)} measured pairs, feature dim {X.shape[1]}, "
          f"pKd range {y.min():.2f}-{y.max():.2f}")

    report = {"n_pairs": int(len(y)), "n_drugs": len(ok),
              "n_targets": len(sequences), "feature_dim": int(X.shape[1]),
              "splits": {}}

    for split in ("random", "cold-drug", "cold-target"):
        per_model = {}
        for model_name in ("mean", "ridge", "random forest"):
            scores = []
            for seed in SEEDS:
                train, test = make_splits(len(ok), len(sequences), drug_idx,
                                          target_idx, split, seed)
                t0 = time.time()
                if model_name == "mean":
                    pred = np.full(len(test), y[train].mean())
                elif model_name == "ridge":
                    model = Ridge(alpha=1.0).fit(X[train], y[train])
                    pred = model.predict(X[test])
                else:
                    model = RandomForestRegressor(n_estimators=200, n_jobs=4,
                                                  random_state=seed,
                                                  min_samples_leaf=2)
                    model.fit(X[train], y[train])
                    pred = model.predict(X[test])
                scores.append({
                    "rmse": rmse(y[test], pred),
                    "pearson": pearson(y[test], pred),
                    "spearman": spearman(y[test], pred),
                    "ci": concordance_index(y[test], pred, seed=seed),
                    "seconds": time.time() - t0,
                    "n_test": int(len(test)),
                })
            per_model[model_name] = {
                metric: {"mean": float(np.mean([s[metric] for s in scores])),
                         "std": float(np.std([s[metric] for s in scores]))}
                for metric in ("rmse", "pearson", "spearman", "ci")
            }
            per_model[model_name]["n_test"] = scores[0]["n_test"]
            m = per_model[model_name]
            print(f"  {split:12s} {model_name:15s} RMSE {m['rmse']['mean']:.3f} "
                  f"Pearson {m['pearson']['mean']:.3f} CI {m['ci']['mean']:.3f}")
        report["splits"][split] = per_model

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "affinity.json").write_text(json.dumps(report, indent=2))
    print("\nwrote results/affinity.json")


if __name__ == "__main__":
    main()
