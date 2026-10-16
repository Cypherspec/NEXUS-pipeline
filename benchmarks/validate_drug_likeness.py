"""Measure what the drug-likeness score does and does not discriminate.

Uses ClinTox (FDA_APPROVED flag) and the HIV screening library as a contrast
set. Result: the score does not separate approved drugs from clinical-trial
failures (AUC ~0.52) -- both classes are drug-like molecules that reached
trials -- while Lipinski violation counts do differ between them. Recorded so
the limitation is visible rather than assumed away.

    python -m benchmarks.validate_drug_likeness
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nexus.chem.descriptors import drug_likeness, lipinski_violations
from nexus.data import load_dataset

ROOT = Path(__file__).resolve().parent.parent


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive outscores a random negative (Mann-Whitney)."""
    scores = np.concatenate([pos, neg])
    ranks = scores.argsort().argsort().astype(float) + 1
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> None:
    mols, Y, spec, _ = load_dataset("clintox")
    approved = Y[:, spec.task_columns.index("FDA_APPROVED")]
    scores = np.array([drug_likeness(m) for m in mols])
    viol = np.array([lipinski_violations(m) for m in mols])

    pos = scores[approved == 1]
    neg = scores[approved == 0]

    hiv_mols, _, _, _ = load_dataset("hiv")
    rng = np.random.default_rng(0)
    sample = rng.choice(len(hiv_mols), 3000, replace=False)
    hiv_scores = np.array([drug_likeness(hiv_mols[i]) for i in sample])

    report = {
        "clintox": {
            "n_approved": int(len(pos)),
            "n_not_approved": int(len(neg)),
            "mean_score_approved": float(pos.mean()),
            "mean_score_not_approved": float(neg.mean()),
            "median_score_approved": float(np.median(pos)),
            "median_score_not_approved": float(np.median(neg)),
            "auc_approved_vs_not": auc(pos, neg),
            "mean_lipinski_violations_approved": float(viol[approved == 1].mean()),
            "mean_lipinski_violations_not_approved": float(viol[approved == 0].mean()),
        },
        "hiv_library_control": {
            "n_sampled": int(len(hiv_scores)),
            "mean_score": float(hiv_scores.mean()),
            "auc_approved_vs_hiv_library": auc(pos, hiv_scores),
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "drug_likeness_validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
