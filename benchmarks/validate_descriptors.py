"""Validate descriptor implementations against reference values shipped with
public datasets (ESOL and BACE carry precomputed descriptor columns).

    python -m benchmarks.validate_descriptors

Writes results/descriptor_validation.json.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from nexus.chem.descriptors import (
    num_hbond_acceptors,
    num_hbond_donors,
    num_lipinski_hba,
    num_lipinski_hbd,
    num_rings,
    num_rotatable_bonds,
    tpsa,
)
from nexus.chem.smiles import read_smiles

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "molnet"

# (file, smiles column, {reference column: [candidate implementations]}).
# Several descriptors have more than one accepted convention; we report the
# closest-matching one so the comparison is like-for-like.
SOURCES = [
    ("delaney-processed.csv", "smiles", {
        "Molecular Weight": ["mw"],
        "Number of H-Bond Donors": ["hbd", "hbd_strict"],
        "Number of Rings": ["rings"],
        "Number of Rotatable Bonds": ["rotatable_bonds"],
        "Polar Surface Area": ["tpsa", "tpsa_sp"],
    }),
    ("bace.csv", "mol", {
        "MW": ["mw"],
        "HBD": ["hbd", "hbd_strict"],
        "HBA": ["hba", "hba_strict"],
        "RB": ["rotatable_bonds"],
        "HeavyAtomCount": ["heavy_atoms"],
        "RingCount": ["rings"],
        "PSA": ["tpsa", "tpsa_sp"],
    }),
]

OURS = {
    "mw": lambda m: m.molecular_weight(),
    "heavy_atoms": lambda m: m.heavy_atom_count(),
    "hbd": num_lipinski_hbd,
    "hba": num_lipinski_hba,
    "hba_strict": num_hbond_acceptors,
    "hbd_strict": num_hbond_donors,
    "rotatable_bonds": num_rotatable_bonds,
    "rings": num_rings,
    "tpsa": tpsa,
    "tpsa_sp": lambda m: tpsa(m, include_s_p=True),
}


def main() -> None:
    report = {}
    for filename, smi_col, mapping in SOURCES:
        rows = list(csv.DictReader((DATA / filename).read_text().splitlines()))
        candidates = sorted({c for v in mapping.values() for c in v})
        ours: dict[str, list[float]] = {c: [] for c in candidates}
        theirs: dict[str, list[float]] = {c: [] for c in mapping}
        n_parsed = 0
        for row in rows:
            mol = read_smiles((row.get(smi_col) or "").strip())
            if mol is None:
                continue
            n_parsed += 1
            for c in candidates:
                ours[c].append(float(OURS[c](mol)))
            for ref_col in mapping:
                raw = (row.get(ref_col) or "").strip()
                theirs[ref_col].append(float(raw) if raw else np.nan)

        stats = {}
        for ref_col, options in mapping.items():
            b = np.array(theirs[ref_col])
            mask = ~np.isnan(b)
            best = None
            for name in options:
                a = np.array(ours[name])[mask]
                bb = b[mask]
                diff = a - bb
                tol = 0.6 if "tpsa" in name or name == "mw" else 0.5
                entry = {
                    "implementation": name,
                    "n": int(mask.sum()),
                    "pearson_r": float(np.corrcoef(a, bb)[0, 1]),
                    "mean_abs_error": float(np.mean(np.abs(diff))),
                    "median_abs_error": float(np.median(np.abs(diff))),
                    "max_abs_error": float(np.max(np.abs(diff))),
                    "fraction_exact": float(np.mean(np.abs(diff) < 0.05)),
                    "fraction_within_tol": float(np.mean(np.abs(diff) <= tol)),
                    "tolerance": tol,
                }
                if best is None or entry["mean_abs_error"] < best["mean_abs_error"]:
                    best = entry
                if len(options) > 1:
                    entry_key = f"{ref_col} [{name}]"
                    stats.setdefault("_all", {})[entry_key] = entry
            stats[ref_col] = best
        report[filename] = {"n_molecules": n_parsed, "descriptors": stats}

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "descriptor_validation.json").write_text(json.dumps(report, indent=2))
    for filename, block in report.items():
        print(f"\n{filename}  ({block['n_molecules']} molecules)")
        print(f"  {'reference column':<26}{'ours':<16}{'n':>6}{'r':>8}{'MAE':>9}{'exact':>9}{'within tol':>12}")
        for name, s in block["descriptors"].items():
            if name == "_all":
                continue
            print(f"  {name:<26}{s['implementation']:<16}{s['n']:>6}{s['pearson_r']:>8.4f}"
                  f"{s['mean_abs_error']:>9.3f}{100*s['fraction_exact']:>8.1f}%"
                  f"{100*s['fraction_within_tol']:>11.1f}%")


if __name__ == "__main__":
    main()
