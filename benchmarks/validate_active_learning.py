"""Retrospective virtual screening: does active learning beat random picking?

The library is a real screening set with known labels (MoleculeNet HIV, which
is ~3.5% active). Each strategy starts from the same random seed batch and may
only see labels for compounds it has selected. The metric is the fraction of
all actives recovered after screening a fixed fraction of the library.

    python -m benchmarks.validate_active_learning
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nexus.al.loop import active_learning_screen
from nexus.chem.fingerprint import fingerprint_matrix
from nexus.data import load_dataset

ROOT = Path(__file__).resolve().parent.parent
LIBRARY_SIZE = 8000
N_ROUNDS = 10
BATCH = 100
INITIAL = 100
SEEDS = (0, 1, 2)
STRATEGIES = ["random", "greedy", "ucb", "ei", "uncertainty"]


def main() -> None:
    mols, Y, spec, _ = load_dataset("hiv")
    y = Y[:, 0]
    keep = np.isfinite(y)
    mols = [m for m, k in zip(mols, keep) if k]
    y = y[keep]

    rng = np.random.default_rng(0)
    idx = rng.choice(len(mols), min(LIBRARY_SIZE, len(mols)), replace=False)
    mols = [mols[i] for i in idx]
    y = y[idx]
    features = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)

    n_actives = int(y.sum())
    screened = INITIAL + N_ROUNDS * BATCH
    print(f"library {len(y)} compounds, {n_actives} actives "
          f"({100 * n_actives / len(y):.2f}%), screening {screened} "
          f"({100 * screened / len(y):.1f}%)")

    report = {
        "library_size": int(len(y)), "n_actives": n_actives,
        "n_screened": int(screened), "rounds": N_ROUNDS, "batch_size": BATCH,
        "initial_size": INITIAL, "seeds": list(SEEDS), "strategies": {},
    }

    for strategy in STRATEGIES:
        recalls, enrichments, curves, times = [], [], [], []
        for seed in SEEDS:
            t0 = time.time()
            res = active_learning_screen(
                features, oracle=lambda i: y[i], n_rounds=N_ROUNDS,
                batch_size=BATCH, initial_size=INITIAL, acquisition=strategy,
                surrogate="rf", hit_threshold=0.5, seed=seed)
            recalls.append(res.total_hits / n_actives)
            enrichments.append(res.enrichment_over_random(len(y), n_actives))
            curves.append(res.cumulative_hits)
            times.append(time.time() - t0)
        report["strategies"][strategy] = {
            "recall_mean": float(np.mean(recalls)),
            "recall_std": float(np.std(recalls)),
            "enrichment_mean": float(np.mean(enrichments)),
            "enrichment_std": float(np.std(enrichments)),
            "cumulative_hits_mean": np.mean(curves, axis=0).round(1).tolist(),
            "seconds_mean": float(np.mean(times)),
        }
        print(f"  {strategy:12s} recall {np.mean(recalls):.3f} +/- {np.std(recalls):.3f}"
              f"   enrichment {np.mean(enrichments):.2f}x"
              f"   ({np.mean(times):.0f}s/run)")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "active_learning_validation.json").write_text(
        json.dumps(report, indent=2))
    print("\nwrote results/active_learning_validation.json")


if __name__ == "__main__":
    main()
