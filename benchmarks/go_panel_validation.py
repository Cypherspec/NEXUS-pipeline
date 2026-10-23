"""Validates the Gene Ontology enrichment the Mantis panel runs.

    python -m benchmarks.go_validation

Two questions, because an enrichment that answers only the second is the kind
that decorates a figure with biology nobody measured:

  calibration  Random gene sets must come back empty. This is where the
               implementation was wrong: restricting Benjamini-Hochberg to the
               terms that happened to have hits selects tests by their own
               outcome. Both orderings are run here on identical sets, swept
               over set size, because the defect's severity depends on it --
               a small list leaves few terms with hits, so the correction's
               denominator collapses to the handful most likely to look
               significant. Small focused lists are also the common case.
  sensitivity  How many co-annotated genes a set needs before the term shows
               up at all. Diluting a whole term barely dents it -- 12 of 12
               terms still rank first with 80% of the set replaced by noise --
               so the informative measurement is the floor, not the curve.

Writes results/go_panel_validation.json.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "extension" / "backend" / "main.py"

ALPHA = 0.05
N_RANDOM = 40
RANDOM_SIZES = [20, 40, 80, 160, 320]
STUDY_SIZE = 200
PLANTED_K = [2, 3, 5, 8, 13, 21]
N_PLANTED = 12


def _load_backend():
    """Loads the shipped backend, not a copy of its logic."""
    spec = importlib.util.spec_from_file_location("nexus_extension_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _p_values(backend, index, genes) -> tuple[list, int, int]:
    """Per-term hypergeometric p-values, exactly as the backend computes them.

    Returned before any multiple-testing correction, so the two orderings
    below can be compared on identical inputs.
    """
    from nexus.bio.enrichment import hypergeometric_sf

    lookup = index["gene_lookup"]
    study = {i for i in (backend._resolve_symbol(g, lookup)[0] for g in genes)
             if i is not None}
    pool = set(range(len(index["genes"])))
    population, drawn = len(pool), len(study)

    rows = []
    for term in index["terms"]:
        members = term["gene_set"] & pool
        if len(members) < 2:
            continue
        hits = members & study
        p = (hypergeometric_sf(len(hits), population, len(members), drawn)
             if hits else 1.0)
        rows.append((p, len(hits)))
    return rows, population, drawn


def _significant(rows, *, filter_first: bool) -> int:
    """Counts significant terms under one of the two orderings.

    filter_first reproduces the defect: terms are dropped for having too few
    hits and only the survivors enter Benjamini-Hochberg, which makes the
    correction's denominator depend on the outcomes being corrected.
    """
    from nexus.bio.enrichment import benjamini_hochberg

    if filter_first:
        kept = [(p, h) for p, h in rows if h >= 2]
        if not kept:
            return 0
        fdr = benjamini_hochberg([p for p, _ in kept])
        return sum(1 for q in fdr if q <= ALPHA)

    fdr = benjamini_hochberg([p for p, _ in rows])
    return sum(1 for q, (_, h) in zip(fdr, rows) if q <= ALPHA and h >= 2)


def calibration(backend, index, rng) -> dict:
    """False positives on gene sets with no biology in them.

    Both orderings run on the same random sets, so the comparison is paired
    and the defect's cost is measured rather than recalled.
    """
    sweep = []
    for size in RANDOM_SIZES:
        shipped, filtered = [], []
        for _ in range(N_RANDOM):
            genes = rng.sample(index["genes"], size)
            rows, _, _ = _p_values(backend, index, genes)
            shipped.append(_significant(rows, filter_first=False))
            filtered.append(_significant(rows, filter_first=True))

            # The shipped path must agree with the reconstruction it is
            # compared to, or the comparison is between two things of mine.
            assert shipped[-1] == backend.invoke(
                "enrich", {"genes": genes}, {})["nSignificant"]

        sweep.append({
            "setSize": size,
            "nTrials": N_RANDOM,
            "counts": shipped,
            "countsFilterFirst": filtered,
            "mean": sum(shipped) / len(shipped),
            "meanFilterFirst": sum(filtered) / len(filtered),
            "maxCount": max(shipped),
            "maxFilterFirst": max(filtered),
            "trialsWithAny": sum(1 for c in shipped if c > 0),
            "trialsWithAnyFilterFirst": sum(1 for c in filtered if c > 0),
        })
    return {"alpha": ALPHA, "nTrials": N_RANDOM, "sweep": sweep}


def sensitivity(backend, index, rng) -> dict:
    """The detection floor: planted members needed before the term surfaces.

    The study set is held at a fixed size and only the number of genes drawn
    from the planted term varies, so the sweep measures sensitivity rather
    than set size. Rank matters as much as detection -- a term buried at
    position 400 is found but not reported.
    """
    pool = index["genes"]
    candidates = [t for t in index["terms"] if 40 <= len(t["gene_set"]) <= 150]
    terms = rng.sample(candidates, min(N_PLANTED, len(candidates)))

    rows = []
    for k in PLANTED_K:
        found, top10, ranks = 0, 0, []
        for term in terms:
            members = [index["genes"][i] for i in sorted(term["gene_set"])]
            planted = rng.sample(members, k)
            filler = [g for g in rng.sample(pool, STUDY_SIZE * 2)
                      if g not in term["gene_set"]][:STUDY_SIZE - k]

            out = backend.invoke("enrich", {"genes": planted + filler}, {})
            hit = next((i for i, r in enumerate(out["terms"])
                        if r["id"] == term["id"]), None)
            if hit is not None:
                found += 1
                ranks.append(hit + 1)
                if hit < 10:
                    top10 += 1
        rows.append({
            "planted": k,
            "studySize": STUDY_SIZE,
            "nTerms": len(terms),
            "recovered": found,
            "rate": found / len(terms),
            "inTop10": top10,
            "medianRank": sorted(ranks)[len(ranks) // 2] if ranks else None,
        })
    return {"nPlanted": len(terms), "curve": rows}


def davis_coverage(backend, index) -> dict:
    """How much of a real target panel the ontology actually covers.

    Taken from the fixture the Mantis panel is driven with, so the number in
    the paper is the number on screen.
    """
    import re

    text = (ROOT / "extension" / "harness" / "points_davis.js").read_text()
    points = json.loads(text[text.index("=") + 1:].rstrip().rstrip(";"))

    symbols = set()
    for point in points:
        for piece in re.split(r"[;,|/]+", point["metadata"].get("target_gene", "")):
            if piece.strip():
                symbols.add(piece.strip())

    out = backend.invoke("enrich", {"genes": sorted(symbols)}, {})
    return {
        "nSymbols": len(symbols),
        "nMatched": out["nStudyGenes"],
        "nResolvedNames": out["nResolvedNames"],
        "nRewritten": out["nRewritten"],
        "nUnmatched": out["nUnmatched"],
        "unmatchedExamples": out["unmatched"][:12],
        "nSignificant": out["nSignificant"],
        "nTested": out["nTested"],
        "topTerms": [{"name": r["name"], "fold": r["foldEnrichment"],
                      "fdr": r["fdr"], "hits": r["studyHits"],
                      "termSize": r["termSize"]}
                     for r in out["terms"][:5]],
    }


def main() -> None:
    backend = _load_backend()
    index = backend._go_index()
    rng = random.Random(20251023)

    out = {
        "ontology": {
            "source": index.get("source"),
            "window": index.get("window"),
            "nTermsTotal": index.get("n_terms_total"),
            "nTermsKept": len(index["terms"]),
            "nGenes": len(index["genes"]),
            "indexBytes": BACKEND.with_name("go_index.json").stat().st_size,
        },
        "calibration": calibration(backend, index, rng),
        "sensitivity": sensitivity(backend, index, rng),
        "davis": davis_coverage(backend, index),
    }

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "go_panel_validation.json").write_text(json.dumps(out, indent=2))

    for row in out["calibration"]["sweep"]:
        print(f"calibration  {row['setSize']:3d}-gene sets  "
              f"shipped {row['mean']:5.2f} (max {row['maxCount']:2d})   "
              f"filter-first {row['meanFilterFirst']:6.2f} "
              f"(max {row['maxFilterFirst']:3d}, "
              f"{row['trialsWithAnyFilterFirst']:2d}/{row['nTrials']} trials)")
    d = out["davis"]
    print(f"davis        {d['nMatched']} genes from {d['nResolvedNames']}/{d['nSymbols']} names "
          f"({d['nRewritten']} normalised), {d['nSignificant']} terms enriched")
    for row in out["sensitivity"]["curve"]:
        rank = row["medianRank"]
        print(f"sensitivity  {row['planted']:3d} of {row['studySize']} planted  "
              f"{row['recovered']}/{row['nTerms']} detected, "
              f"{row['inTop10']} in top 10"
              f"{f', median rank {rank}' if rank else ''}")


if __name__ == "__main__":
    main()
