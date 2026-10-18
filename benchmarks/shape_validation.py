"""Does the shape layer measure what it claims to measure?

Three checks, each against something known independently of the code:

  probes        benzene must give exactly one long-lived loop and curvature
                zero on every bond, an acyclic chain neither
  cycle rule    Ollivier-Ricci curvature of C_n is exactly zero for n >= 6,
                which is a theorem, not a fit
  recovery      across a real dataset, long-lived H1 features should recover
                the ring count perceived from the bond graph -- the topology
                is computed from 3D coordinates alone and never sees a ring

    python -m benchmarks.shape_validation --n 200
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np

from nexus.chem.conformer import embed_molecule
from nexus.chem.descriptors import num_rings
from nexus.chem.smiles import parse_smiles
from nexus.data import load_dataset
from nexus.topo.homology import rips_persistence
from nexus.topo.ricci import ollivier_ricci_curvature

ROOT = Path(__file__).resolve().parent.parent

# A feature shorter-lived than this is embedding noise rather than a ring.
# Fixed once, here, and used by every count below.
LIFE_THRESHOLD = 0.35

PROBES = [
    ("benzene", "c1ccccc1", 1),
    ("octane", "CCCCCCCC", 0),
    ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 2),
    ("naphthalene", "c1ccc2ccccc2c1", 2),
    ("cyclohexane", "C1CCCCC1", 1),
]


def long_lived_loops(mol, seed: int = 0) -> tuple[int, float]:
    """Counts H1 features that survive a meaningful range of scales.

    @param mol Molecule to embed and analyse.
    @param seed Embedding seed.
    @returns The loop count and the longest lifetime.
    """
    xyz, _ = embed_molecule(mol, seed=seed)
    diagram = rips_persistence(np.asarray(xyz, dtype=float), max_dim=1)
    pairs = np.asarray(diagram.points.get(1, []), dtype=float).reshape(-1, 2)
    finite = pairs[np.isfinite(pairs[:, 1])]
    if not len(finite):
        return 0, 0.0
    lives = finite[:, 1] - finite[:, 0]
    return int((lives > LIFE_THRESHOLD).sum()), float(lives.max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dataset", default="esol")
    args = ap.parse_args()

    report: dict = {"life_threshold": LIFE_THRESHOLD, "probes": [], "cycles": [],
                    "recovery": {}}

    for name, smiles, expected in PROBES:
        mol = parse_smiles(smiles)
        loops, longest = long_lived_loops(mol)
        curvature = list(ollivier_ricci_curvature(mol).values())
        report["probes"].append({
            "name": name, "smiles": smiles, "expected_loops": expected,
            "loops": loops, "matches": loops == expected,
            "longest_life": longest,
            "curvature_min": float(min(curvature)) if curvature else 0.0,
            "curvature_max": float(max(curvature)) if curvature else 0.0,
        })
        print(f"  {name:<12} loops {loops} (expected {expected}) "
              f"{'ok' if loops == expected else 'MISMATCH'}", flush=True)

    # Cycle graphs: the theorem says curvature is exactly zero for n >= 6.
    for n in (6, 7, 8, 10, 12):
        ring = parse_smiles("C1" + "C" * (n - 1) + "1")
        values = list(ollivier_ricci_curvature(ring).values())
        worst = max(abs(v) for v in values) if values else 0.0
        report["cycles"].append({"n": n, "max_abs_curvature": float(worst),
                                 "is_zero": bool(worst < 1e-9)})
        print(f"  C{n:<11} max |curvature| {worst:.2e} "
              f"{'exactly zero' if worst < 1e-9 else 'NONZERO'}", flush=True)

    dataset, _, _, _ = load_dataset(args.dataset)
    loops, rings, skipped = [], [], 0
    for mol in dataset[: args.n]:
        if not (3 <= mol.n_atoms <= 40):
            skipped += 1
            continue
        try:
            count, _ = long_lived_loops(mol)
        except Exception:                                      # noqa: BLE001
            skipped += 1
            continue
        loops.append(count)
        rings.append(num_rings(mol))

    correlation = float(np.corrcoef(rings, loops)[0, 1]) if len(rings) > 2 else float("nan")
    exact = float(np.mean([a == b for a, b in zip(loops, rings)])) if rings else 0.0
    report["recovery"] = {
        "dataset": args.dataset, "n_molecules": len(rings), "n_skipped": skipped,
        "pearson_r": correlation, "exact_match_fraction": exact,
        "mean_absolute_error": float(np.mean(np.abs(np.array(loops) - np.array(rings))))
        if rings else 0.0,
    }
    print(f"\n  ring recovery over {len(rings)} molecules: r = {correlation:.3f}, "
          f"exact on {100 * exact:.1f}%", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "shape_validation.json").write_text(json.dumps(report, indent=2))
    print("wrote results/shape_validation.json")


if __name__ == "__main__":
    main()
