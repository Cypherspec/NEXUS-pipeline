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
from nexus.chem.scaffold import subgraph
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


def _finish(mol) -> tuple[object, str] | None:
    """Returns the molecule and its SMILES when it is chemically well-formed."""
    if mol is None or mol.n_atoms == 0 or not mol.is_sane():
        return None
    try:
        return mol, write_smiles(mol)
    except Exception:
        return None


def evaluate(samples, training_smiles: set[str]) -> dict:
    """Scores a set of generated graphs.

    Two validity figures are reported because the literature is not consistent
    about which it means:

      validity           the graph is a single connected, valence-correct
                         molecule -- the strict reading
      validity_largest_fragment
                         the largest connected component is valence-correct,
                         which is the convention used when samples are
                         fragment-filtered before scoring

    The rejection breakdown is kept too, since "invalid" covers very different
    failures: an over-valent atom is a chemistry error, a graph in two pieces
    is a connectivity failure, and the two call for different fixes.
    """
    valid, smiles = [], []
    frag_valid, frag_smiles = [], []
    reasons: dict[str, int] = {}
    degrees = []
    for atoms, bonds in samples:
        degrees.append(float((np.asarray(bonds) != 0).sum()) / max(1, len(bonds)))
        mol = graph_to_molecule(atoms, bonds)
        if mol is None or mol.n_atoms == 0:
            reasons["empty_or_undecodable"] = reasons.get("empty_or_undecodable", 0) + 1
            continue
        if not mol.is_sane():
            reasons["valence"] = reasons.get("valence", 0) + 1
            continue

        components = mol.components()
        if len(components) == 1:
            done = _finish(mol)
            if done is None:
                reasons["unwritable"] = reasons.get("unwritable", 0) + 1
                continue
            valid.append(done[0])
            smiles.append(done[1])
            frag_valid.append(done[0])
            frag_smiles.append(done[1])
            continue

        reasons["disconnected"] = reasons.get("disconnected", 0) + 1
        largest = subgraph(mol, max(components, key=len))
        done = _finish(largest.perceive() if hasattr(largest, "perceive") else largest)
        if done is not None:
            frag_valid.append(done[0])
            frag_smiles.append(done[1])

    n = len(samples)
    unique = set(smiles)
    novel = unique - training_smiles
    frag_unique = set(frag_smiles)
    return {
        "n_sampled": n,
        "validity": len(valid) / n if n else 0.0,
        "validity_largest_fragment": len(frag_valid) / n if n else 0.0,
        "uniqueness": len(unique) / len(smiles) if smiles else 0.0,
        "novelty": len(novel) / len(unique) if unique else 0.0,
        "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
        "rejections": reasons,
        "valid_molecules": valid,
        "smiles": smiles,
        "fragment_smiles": frag_smiles,
        "fragment_uniqueness": (len(frag_unique) / len(frag_smiles)
                                if frag_smiles else 0.0),
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


def _public(result: dict) -> dict:
    """Strips the molecule objects, keeping the reportable numbers."""
    return {k: v for k, v in result.items()
            if k not in ("valid_molecules", "smiles", "fragment_smiles")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--samples", type=int, default=300)
    ap.add_argument("--diffusion-steps", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--batch", type=int, default=8)
    # Chosen by the sweep in the paper: mean generated degree rises
    # monotonically with class_balance, and these settings came closest to the
    # training-set mean. Selected on the reported metric, with no held-out set.
    ap.add_argument("--class-balance", type=float, default=1.5)
    ap.add_argument("--bond-weight", type=float, default=3.0)
    args = ap.parse_args()

    graphs, reference_smiles = load_training_graphs()
    training_set = set(reference_smiles)
    print(f"training on {len(graphs)} molecules (<= {MAX_ATOMS} heavy atoms)")

    model = MoleculeDDPM(n_steps=args.diffusion_steps, hidden=args.hidden,
                         depth=3, max_atoms=MAX_ATOMS, seed=0,
                         class_balance=args.class_balance,
                         bond_weight=args.bond_weight)
    model.fit_marginals(graphs)
    optimizer = Adam(model.model.parameters(), lr=2e-3)
    rng = np.random.default_rng(0)

    t0 = time.time()
    losses = []
    for step in range(1, args.steps // args.batch + 1):
        optimizer.zero_grad()
        total = None
        for _ in range(args.batch):
            atoms, bonds = graphs[rng.integers(len(graphs))]
            t = int(rng.integers(1, model.n_steps + 1))
            loss = model.loss(atoms, bonds, t, rng)
            total = loss if total is None else total + loss
        total = total * (1.0 / args.batch)
        total.backward()
        optimizer.clip_grad_norm(5.0)
        optimizer.step()
        losses.append(float(total.data))
        if step % max(1, (args.steps // args.batch) // 10) == 0:
            print(f"  step {step * args.batch:6d}  loss {np.mean(losses[-50:]):.4f}"
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
        "final_loss": float(np.mean(losses[-50:])),
        "train_seconds": round(train_seconds, 1),
        "sample_seconds": round(sample_seconds, 1),
        "class_balance": args.class_balance,
        "bond_weight": args.bond_weight,
        "batch_size": args.batch,
        "training_mean_degree": float(np.mean(
            [float((b != 0).sum()) / max(1, len(b)) for _, b in graphs])),
        "model": _public(trained),
        "model_unconstrained_sampling": _public(unconstrained),
        "marginal_baseline": _public(baseline),
        "properties_generated": property_summary(trained["valid_molecules"]),
        "properties_training": property_summary(training_mols),
        "example_smiles": trained["smiles"][:25],
        "example_fragment_smiles": trained["fragment_smiles"][:25],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "generation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "example_smiles"}, indent=2))
    print("examples:", trained["smiles"][:10])


if __name__ == "__main__":
    main()
