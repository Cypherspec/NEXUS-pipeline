"""Train the graph diffusion model and evaluate generated molecules.

Metrics are the standard ones for de novo generation:
  validity   -- fraction of samples that parse and pass a valence check
  uniqueness -- distinct molecules among the valid ones
  novelty    -- valid molecules not present in the training set
Property distributions of the generated set are compared with the training set
so "valid" is not mistaken for "drug-like".

    python -m benchmarks.generate --steps 4000 --samples 300
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nexus.chem.descriptors import drug_likeness, num_rings, tpsa
from nexus.chem.smiles import write_smiles
from nexus.models.ddpm import (
    MoleculeDDPM,
    graph_to_molecule,
    molecule_to_graph,
)
from nexus.models.nn import Adam

ROOT = Path(__file__).resolve().parent.parent
MAX_ATOMS = 18


def load_training_graphs(max_atoms: int = MAX_ATOMS, limit: int = 20000):
    """Small, sane, single-fragment molecules pooled across MoleculeNet."""
    from nexus.data import load_dataset

    mols = []
    for name in ("esol", "bbbp", "clintox", "tox21", "lipo", "freesolv"):
        try:
            dataset, _, _, _ = load_dataset(name)
        except Exception:
            continue
        mols.extend(dataset)
    graphs, reference = [], []
    seen: set[str] = set()
    for mol in mols:
        if len(mol.components()) != 1:
            continue
        if not (3 <= mol.n_atoms <= max_atoms) or not mol.is_sane():
            continue
        g = molecule_to_graph(mol)
        if g is None:
            continue
        try:
            smi = write_smiles(mol)
        except Exception:
            continue
        if smi in seen:
            continue
        seen.add(smi)
        graphs.append(g)
        reference.append(smi)
        if len(graphs) >= limit:
            break
    return graphs, reference


def evaluate(samples, training_smiles: set[str]) -> dict:
    valid, smiles = [], []
    for atoms, bonds in samples:
        mol = graph_to_molecule(atoms, bonds)
        if mol is None or mol.n_atoms == 0:
            continue
        if not mol.is_sane():
            continue
        if len(mol.components()) != 1:
            continue
        try:
            smi = write_smiles(mol)
        except Exception:
            continue
        valid.append(mol)
        smiles.append(smi)
    n = len(samples)
    unique = set(smiles)
    novel = unique - training_smiles
    return {
        "n_sampled": n,
        "validity": len(valid) / n if n else 0.0,
        "uniqueness": len(unique) / len(smiles) if smiles else 0.0,
        "novelty": len(novel) / len(unique) if unique else 0.0,
        "valid_molecules": valid,
        "smiles": smiles,
    }


def property_summary(mols) -> dict:
    if not mols:
        return {}
    return {
        "mean_atoms": float(np.mean([m.n_atoms for m in mols])),
        "mean_mw": float(np.mean([m.molecular_weight() for m in mols])),
        "mean_rings": float(np.mean([num_rings(m) for m in mols])),
        "mean_tpsa": float(np.mean([tpsa(m) for m in mols])),
        "mean_drug_likeness": float(np.mean([drug_likeness(m) for m in mols])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--diffusion-steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=96)
    args = ap.parse_args()

    graphs, reference_smiles = load_training_graphs()
    training_set = set(reference_smiles)
    print(f"training on {len(graphs)} molecules (<= {MAX_ATOMS} heavy atoms)")

    model = MoleculeDDPM(n_steps=args.diffusion_steps, hidden=args.hidden,
                         depth=3, max_atoms=MAX_ATOMS, seed=0)
    model.fit_marginals(graphs)
    optimizer = Adam(model.model.parameters(), lr=2e-3)
    rng = np.random.default_rng(0)

    t0 = time.time()
    losses = []
    for step in range(1, args.steps + 1):
        atoms, bonds = graphs[rng.integers(len(graphs))]
        t = int(rng.integers(1, model.n_steps + 1))
        optimizer.zero_grad()
        loss = model.loss(atoms, bonds, t, rng)
        loss.backward()
        optimizer.clip_grad_norm(5.0)
        optimizer.step()
        losses.append(float(loss.data))
        if step % max(1, args.steps // 10) == 0:
            print(f"  step {step:5d}  loss {np.mean(losses[-200:]):.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
    train_seconds = time.time() - t0

    print("sampling...")
    t1 = time.time()
    samples = [model.sample(seed=1000 + i, constrain_valence=True)
               for i in range(args.samples)]
    sample_seconds = time.time() - t1
    trained = evaluate(samples, training_set)

    # Ablation: the same trained model sampled without the valence constraint,
    # which isolates how much of the validity comes from the constraint.
    unconstrained = evaluate(
        [model.sample(seed=2000 + i, constrain_valence=False)
         for i in range(min(args.samples, 100))], training_set)

    # Control: samples drawn straight from the marginals, with no model. Any
    # validity above this is what the learned denoiser actually contributes.
    rng2 = np.random.default_rng(7)
    baseline_samples = []
    for _ in range(args.samples):
        n_atoms = int(rng2.choice(len(model.size_distribution), p=model.size_distribution))
        n_atoms = max(2, min(n_atoms, MAX_ATOMS))
        a = rng2.choice(len(model.atom_marginal), size=n_atoms, p=model.atom_marginal)
        b = rng2.choice(len(model.bond_marginal), size=(n_atoms, n_atoms),
                        p=model.bond_marginal)
        b = np.triu(b, 1); b = b + b.T
        baseline_samples.append((a, b))
    baseline = evaluate(baseline_samples, training_set)

    training_mols = [graph_to_molecule(a, b) for a, b in graphs]
    training_mols = [m for m in training_mols if m is not None]

    report = {
        "n_training_molecules": len(graphs),
        "max_atoms": MAX_ATOMS,
        "training_steps": args.steps,
        "diffusion_steps": args.diffusion_steps,
        "final_loss": float(np.mean(losses[-200:])),
        "train_seconds": round(train_seconds, 1),
        "sample_seconds": round(sample_seconds, 1),
        "model": {k: v for k, v in trained.items()
                  if k not in ("valid_molecules", "smiles")},
        "model_unconstrained_sampling": {
            k: v for k, v in unconstrained.items()
            if k not in ("valid_molecules", "smiles")},
        "marginal_baseline": {k: v for k, v in baseline.items()
                              if k not in ("valid_molecules", "smiles")},
        "properties_generated": property_summary(trained["valid_molecules"]),
        "properties_training": property_summary(training_mols),
        "example_smiles": trained["smiles"][:25],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "generation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "example_smiles"}, indent=2))
    print("examples:", trained["smiles"][:10])


if __name__ == "__main__":
    main()
