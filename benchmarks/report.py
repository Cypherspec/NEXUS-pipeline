"""Aggregate MoleculeNet benchmark runs into a comparison table.

    python -m benchmarks.report

Reads results/molnet/*.json and writes results/molnet_summary.json plus a
markdown table. Reference values from the literature are shown for orientation;
they use the same protocol but not identical splits, so they are not a
like-for-like comparison of the same numbers.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "molnet"


def load_runs() -> dict:
    runs = defaultdict(list)
    for path in sorted(RESULTS.glob("*_seed*.json")):
        data = json.loads(path.read_text())
        runs[data["dataset"]].append(data)
    return dict(runs)


def main() -> None:
    from benchmarks.molnet import REFERENCE

    runs = load_runs()
    if not runs:
        print("no results yet")
        return

    summary = {}
    lines = []
    for dataset in sorted(runs):
        entries = runs[dataset]
        metric = entries[0]["metric"]
        higher_better = metric == "roc_auc"
        per_method = defaultdict(list)
        for entry in entries:
            for name, block in entry["methods"].items():
                score = block["test_score"]
                if np.isfinite(score):
                    per_method[name].append(score)

        stats = {}
        for name, scores in per_method.items():
            stats[name] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "n_seeds": len(scores),
            }
        ordered = sorted(stats.items(),
                         key=lambda kv: -kv[1]["mean"] if higher_better else kv[1]["mean"])
        summary[dataset] = {
            "metric": metric,
            "n_seeds": len(entries),
            "n_total": entries[0]["n_total"],
            "n_tasks": entries[0]["n_tasks"],
            "scaffold_leakage": float(np.mean([e["scaffold_leakage_test"] for e in entries])),
            "methods": dict(ordered),
            "reference": REFERENCE.get(dataset, {}),
        }

        arrow = "higher is better" if higher_better else "lower is better"
        ref = REFERENCE.get(dataset, {})
        lines.append(f"\n### {dataset}  ({entries[0]['n_total']} molecules, "
                     f"{entries[0]['n_tasks']} task(s), {metric}, {arrow})")
        if ref:
            lines.append(f"Reference D-MPNN (Yang et al. 2019, scaffold split): "
                         f"{ref['chemprop_dmpnn']:.3f}")
        lines.append("")
        lines.append("| Method | mean | std | seeds |")
        lines.append("|---|---|---|---|")
        for name, s in ordered:
            lines.append(f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | {s['n_seeds']} |")

    out = ROOT / "results" / "molnet_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    text = "\n".join(lines)
    (ROOT / "results" / "molnet_summary.md").write_text(text)
    print(text)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
