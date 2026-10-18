"""What each stage of the pipeline costs, and how that cost grows.

A tool paper should say what it is like to run the thing. Each stage is timed
over a range of input sizes on the same machine, repeated, and reported with
the slope of a log-log fit so the growth is a measured exponent rather than a
claim. Stages that are per-molecule (parsing, descriptors, fingerprints) should
come out near linear; the Gaussian process is cubic in the training set and
should say so.

    python -m benchmarks.scaling --sizes 50 100 200 400 800
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def timed(fn, repeats: int = 3) -> float:
    """Best-of-N wall time, in seconds.

    The minimum is reported rather than the mean: it is the measurement least
    contaminated by whatever else the machine was doing.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 400, 800])
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    from nexus.al.gp import GaussianProcess
    from nexus.chem.conformer import embed_molecule
    from nexus.chem.descriptors import compute_descriptors
    from nexus.chem.fingerprint import fingerprint_matrix
    from nexus.chem.scaffold import murcko_scaffold
    from nexus.chem.smiles import parse_smiles, write_smiles
    from nexus.data import load_dataset
    from nexus.topo.homology import rips_persistence

    dataset, labels, _, _ = load_dataset("esol")
    smiles = [write_smiles(m) for m in dataset]
    y_all = np.asarray(labels, dtype=float).ravel()

    report: dict = {"sizes": args.sizes, "repeats": args.repeats, "stages": {}}

    def record(name: str, size: int, seconds: float) -> None:
        report["stages"].setdefault(name, []).append(
            {"n": size, "seconds": seconds, "per_item_ms": 1000 * seconds / size})

    for n in args.sizes:
        if n > len(smiles):
            continue
        subset = smiles[:n]
        mols = [parse_smiles(s) for s in subset]

        record("parse", n, timed(lambda: [parse_smiles(s) for s in subset], args.repeats))
        record("descriptors", n,
               timed(lambda: [compute_descriptors(m) for m in mols], args.repeats))
        record("fingerprints", n,
               timed(lambda: fingerprint_matrix(mols, kind="morgan", n_bits=1024),
                     args.repeats))
        record("scaffolds", n,
               timed(lambda: [murcko_scaffold(m) for m in mols], args.repeats))

        X = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)
        y = y_all[:n]
        record("gp_fit", n,
               timed(lambda: GaussianProcess(kernel="tanimoto", noise=1e-2).fit(X, y),
                     max(1, args.repeats - 1)))

        # The per-molecule 3D stages are timed on a fixed-size sample, since
        # their cost is driven by atom count rather than by how many molecules
        # are in scope.
        if n == args.sizes[0]:
            small = [m for m in mols if 5 <= m.n_atoms <= 25][:20]
            if small:
                report["per_molecule"] = {
                    "n_molecules": len(small),
                    "conformer_ms": 1000 * timed(
                        lambda: [embed_molecule(m, seed=0) for m in small],
                        args.repeats) / len(small),
                    "persistence_ms": 1000 * timed(
                        lambda: [rips_persistence(
                            np.asarray(embed_molecule(m, seed=0)[0]), max_dim=1)
                            for m in small], 1) / len(small),
                }
        print(f"  n={n:<5} parsed, described, fingerprinted, scaffolded, GP-fitted",
              flush=True)

    # Growth exponent from a log-log fit: 1.0 is linear, 3.0 cubic.
    for name, rows in report["stages"].items():
        if len(rows) < 3:
            continue
        logs_n = np.log(np.array([r["n"] for r in rows], dtype=float))
        logs_t = np.log(np.array([max(r["seconds"], 1e-9) for r in rows]))
        slope = float(np.polyfit(logs_n, logs_t, 1)[0])
        report.setdefault("exponents", {})[name] = slope
        print(f"  {name:<14} growth exponent {slope:.2f}", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "scaling.json").write_text(json.dumps(report, indent=2))
    print("wrote results/scaling.json")


if __name__ == "__main__":
    main()
