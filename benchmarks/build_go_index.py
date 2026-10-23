"""Prune the Gene Ontology into something an extension package can carry.

The full ontology is a 32 MB OBO beside a 15 MB gzipped annotation file. A
Mantis package is capped at 25 MB, every file in it must be valid UTF-8 text,
and a gzip is neither -- so shipping GO as distributed is not an option.

What ships instead is the slice enrichment actually uses: Biological Process
terms annotated to between `--min-size` and `--max-size` human genes, with the
true-path rule already applied. That window is the standard one for
over-representation analysis -- a term with three genes cannot reach
significance and a term with eight thousand says nothing about what came back
-- so the pruning is the analysis's own convention rather than a size
compromise dressed up as one. Everything outside the window is dropped, and the
index records exactly what was kept so a reader can tell.

    python -m benchmarks.build_go_index
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "extension" / "backend" / "go_index.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obo", default=str(ROOT / "data" / "go" / "go-basic.obo"))
    ap.add_argument("--gaf", default=str(ROOT / "data" / "go" / "goa_human.gaf.gz"))
    ap.add_argument("--aspect", default="P", help="GAF aspect code; P is biological process")
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--max-size", type=int, default=500)
    args = ap.parse_args()

    from nexus.bio.ontology import load_human_ontology

    go = load_human_ontology(args.obo, args.gaf, aspect=args.aspect)
    print(f"loaded {len(go)} terms, {go.n_annotated_genes} annotated genes", flush=True)

    kept = [gid for gid in go.topological_order()
            if args.min_size <= go.term_size(gid) <= args.max_size]

    # Gene symbols are interned into a list and referenced by index: the symbol
    # strings are a small fraction of the file, the 400k references are not.
    gene_index: dict[str, int] = {}
    terms = []
    for gid in kept:
        members = sorted(go.genes_for(gid))
        ids = []
        for gene in members:
            idx = gene_index.get(gene)
            if idx is None:
                idx = len(gene_index)
                gene_index[gene] = idx
            ids.append(idx)
        terms.append({"id": gid, "name": go.name(gid), "genes": ids})

    genes = [""] * len(gene_index)
    for gene, idx in gene_index.items():
        genes[idx] = gene

    index = {
        "source": {
            "obo": Path(args.obo).name, "gaf": Path(args.gaf).name,
            "aspect": args.aspect, "propagated": True,
        },
        "window": [args.min_size, args.max_size],
        "n_terms_total": len(go),
        "n_genes_annotated": go.n_annotated_genes,
        "genes": genes,
        "terms": terms,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Separators without spaces: this file is read by machine and its size is
    # the constraint the whole exercise is about.
    OUT.write_text(json.dumps(index, separators=(",", ":")))
    size_mb = OUT.stat().st_size / 1e6
    pairs = sum(len(t["genes"]) for t in terms)
    print(f"kept {len(terms)} terms, {len(genes)} genes, {pairs} gene-term pairs")
    print(f"wrote {OUT.relative_to(ROOT)}  {size_mb:.2f} MB")
    if size_mb > 5.0:
        raise SystemExit("index exceeds the 5 MB per-file cap Mantis enforces")


if __name__ == "__main__":
    main()
