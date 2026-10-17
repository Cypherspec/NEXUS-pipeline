"""Validate the Gene Ontology layer on real human annotation data.

Three checks, all falsifiable:
  1. Positive control -- a gene set drawn from a known GO term must recover
     that term (or a close relative) at the top of the enrichment ranking.
  2. False-positive control -- random gene sets must yield few significant
     terms, at roughly the nominal FDR level.
  3. Semantic similarity -- genes sharing a pathway must score higher than
     random gene pairs.

    python -m benchmarks.validate_go
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from nexus.bio.enrichment import enrichment, format_results
from nexus.bio.gaf import gene_to_terms, parse_gaf
from nexus.bio.ontology import GeneOntology

ROOT = Path(__file__).resolve().parent.parent
GO_DIR = ROOT / "data" / "go"

# Well-characterised terms used as positive controls.
POSITIVE_CONTROLS = [
    ("GO:0006281", "DNA repair"),
    ("GO:0006915", "apoptotic process"),
    ("GO:0007049", "cell cycle"),
    ("GO:0006955", "immune response"),
    ("GO:0006096", "glycolytic process"),
]
N_DRAW = 60
N_RANDOM_TRIALS = 40
ALPHA = 0.05


def main() -> None:
    t0 = time.time()
    go = GeneOntology.load(GO_DIR / "go-basic.obo")
    annotations = parse_gaf(GO_DIR / "goa_human.gaf.gz")
    go.annotate(annotations, aspect="P")
    g2t = gene_to_terms(annotations, aspect="P")
    load_s = time.time() - t0
    rng = np.random.default_rng(0)
    all_genes = sorted({g for genes in go._term_genes.values() for g in genes})
    report = {
        "go_version": go.header.get("data-version", ""),
        "n_terms": len(go),
        "n_annotations": len(annotations),
        "n_genes": go.n_annotated_genes,
        "load_seconds": round(load_s, 2),
        "positive_controls": [],
        "false_positive_control": {},
        "semantic_similarity": {},
    }

    # ---- 1. positive controls
    for go_id, label in POSITIVE_CONTROLS:
        members = sorted(go.genes_for(go_id))
        if len(members) < N_DRAW:
            continue
        draw = list(rng.choice(members, N_DRAW, replace=False))
        res = enrichment(go, draw, aspect="P", method="classic")
        rank = next((i for i, r in enumerate(res) if r.go_id == go_id), None)
        top = res[0] if res else None
        in_top10_related = any(
            r.go_id == go_id or go_id in go.ancestors(r.go_id) or r.go_id in go.ancestors(go_id)
            for r in res[:10])
        report["positive_controls"].append({
            "target": go_id, "label": label, "n_drawn": N_DRAW,
            "rank_of_target": rank,
            "target_fdr": res[rank].fdr if rank is not None else None,
            "top_term": top.go_id if top else None,
            "top_term_name": top.name if top else None,
            "top_term_fdr": top.fdr if top else None,
            "target_or_relative_in_top10": bool(in_top10_related),
            "n_significant_fdr05": int(sum(1 for r in res if r.fdr <= ALPHA)),
        })
        print(f"[+] {label:22s} target rank {rank}  top={top.name[:40] if top else None!r}")

    # ---- 2. false-positive control on random gene sets
    counts, min_fdrs = [], []
    for trial in range(N_RANDOM_TRIALS):
        draw = list(rng.choice(all_genes, N_DRAW, replace=False))
        res = enrichment(go, draw, aspect="P", method="classic")
        sig = [r for r in res if r.fdr <= ALPHA]
        counts.append(len(sig))
        min_fdrs.append(min((r.fdr for r in res), default=1.0))
    report["false_positive_control"] = {
        "n_trials": N_RANDOM_TRIALS, "genes_per_trial": N_DRAW, "alpha": ALPHA,
        "mean_significant_terms": float(np.mean(counts)),
        "median_significant_terms": float(np.median(counts)),
        "max_significant_terms": int(np.max(counts)),
        "fraction_trials_with_any_hit": float(np.mean(np.array(counts) > 0)),
        "median_min_fdr": float(np.median(min_fdrs)),
    }
    print(f"[+] random sets: mean {np.mean(counts):.2f} significant terms, "
          f"{100*np.mean(np.array(counts)>0):.0f}% of trials had any hit")

    # ---- 3. semantic similarity: same-term pairs vs random pairs
    sim_same, sim_random = [], []
    pool = [g for g in all_genes if g in g2t and 2 <= len(g2t[g]) <= 60]
    for go_id, _ in POSITIVE_CONTROLS:
        members = [g for g in go.genes_for(go_id) if g in set(pool)]
        if len(members) < 20:
            continue
        for _ in range(40):
            a, b = rng.choice(members, 2, replace=False)
            sim_same.append(go.gene_similarity(g2t[a], g2t[b], method="lin"))
    for _ in range(200):
        a, b = rng.choice(pool, 2, replace=False)
        sim_random.append(go.gene_similarity(g2t[a], g2t[b], method="lin"))
    sim_same, sim_random = np.array(sim_same), np.array(sim_random)
    scores = np.concatenate([sim_same, sim_random])
    ranks = scores.argsort().argsort().astype(float) + 1
    auc = float((ranks[:len(sim_same)].sum() - len(sim_same) * (len(sim_same) + 1) / 2)
                / (len(sim_same) * len(sim_random)))
    report["semantic_similarity"] = {
        "n_same_pathway_pairs": int(len(sim_same)),
        "n_random_pairs": int(len(sim_random)),
        "mean_same_pathway": float(sim_same.mean()),
        "mean_random": float(sim_random.mean()),
        "auc_same_vs_random": auc,
    }
    print(f"[+] semantic similarity: same-pathway {sim_same.mean():.3f} vs "
          f"random {sim_random.mean():.3f}, AUC {auc:.3f}")

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "go_validation.json").write_text(json.dumps(report, indent=2))
    print("\nwrote results/go_validation.json")


if __name__ == "__main__":
    main()
