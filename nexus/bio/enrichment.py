"""GO enrichment analysis: hypergeometric tests with multiple-testing control.

Includes the classic per-term test and the `elim` decorrelation strategy
(Alexa et al. 2006), which tests terms from the leaves upward and removes the
genes of already-significant descendants before testing an ancestor, so broad
parent terms are not reported merely because a specific child is enriched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from nexus.bio.ontology import GeneOntology


def log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def hypergeometric_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) for X ~ Hypergeometric(N population, K successes, n draws).

    Computed by summing exact log-probabilities, which stays accurate for the
    small tail probabilities that enrichment analysis depends on.
    """
    if k <= 0:
        return 1.0
    if K <= 0 or n <= 0 or k > min(K, n):
        return 0.0
    denom = log_choose(N, n)
    total = 0.0
    for x in range(k, min(K, n) + 1):
        term = log_choose(K, x) + log_choose(N - K, n - x) - denom
        total += math.exp(term)
    return float(min(max(total, 0.0), 1.0))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR), monotone and clipped to 1."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def bonferroni(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    return np.clip(p * len(p), 0, 1)


@dataclass
class EnrichmentResult:
    go_id: str
    name: str
    namespace: str
    depth: int
    n_study_hits: int           # k
    n_study: int                # n
    n_term_genes: int           # K
    n_population: int           # N
    p_value: float
    fdr: float = 1.0
    bonferroni: float = 1.0
    fold_enrichment: float = 0.0
    genes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def expected(self) -> float:
        return self.n_study * self.n_term_genes / max(self.n_population, 1)


def enrichment(
    go: GeneOntology,
    study_genes,
    population_genes=None,
    aspect: str | None = None,
    min_term_size: int = 5,
    max_term_size: int = 2000,
    method: str = "classic",
    alpha: float = 0.05,
) -> list[EnrichmentResult]:
    """Test every GO term for over-representation in `study_genes`.

    method: "classic" tests each term independently; "elim" removes the genes
    of significant descendant terms before testing an ancestor.
    """
    study = {g for g in study_genes if g in _all_genes(go)}
    population = (set(population_genes) if population_genes is not None
                  else set(_all_genes(go)))
    population |= study
    N = len(population)
    n = len(study)
    if n == 0 or N == 0:
        return []

    candidates = []
    for go_id, genes in go._term_genes.items():
        if aspect and _aspect_letter(go.namespace(go_id)) != aspect:
            continue
        in_pop = genes & population
        if not (min_term_size <= len(in_pop) <= max_term_size):
            continue
        candidates.append((go_id, in_pop))

    if method == "elim":
        results = _elim(go, candidates, study, population, alpha)
    else:
        results = []
        for go_id, in_pop in candidates:
            hits = in_pop & study
            if not hits:
                continue
            results.append(_make_result(go, go_id, hits, in_pop, n, N))

    if results:
        p = np.array([r.p_value for r in results])
        fdr = benjamini_hochberg(p)
        bon = bonferroni(p)
        for r, f, b in zip(results, fdr, bon):
            r.fdr, r.bonferroni = float(f), float(b)
    results.sort(key=lambda r: (r.p_value, -r.fold_enrichment))
    return results


def _make_result(go, go_id, hits, in_pop, n, N) -> EnrichmentResult:
    k, K = len(hits), len(in_pop)
    p = hypergeometric_sf(k, N, K, n)
    expected = n * K / N if N else 0.0
    return EnrichmentResult(
        go_id=go_id, name=go.name(go_id), namespace=go.namespace(go_id),
        depth=go.depth(go_id), n_study_hits=k, n_study=n, n_term_genes=K,
        n_population=N, p_value=p,
        fold_enrichment=(k / expected) if expected > 0 else 0.0,
        genes=tuple(sorted(hits)),
    )


def _elim(go, candidates, study, population, alpha) -> list[EnrichmentResult]:
    """Alexa et al. (2006) elim: test leaves first, then remove their genes."""
    by_id = dict(candidates)
    # Deepest terms first, so children are tested before their ancestors.
    ordered = sorted(by_id, key=lambda t: -go.depth(t))
    removed: dict[str, set[str]] = {t: set() for t in by_id}
    results = []
    N_all = len(population)
    for go_id in ordered:
        in_pop = by_id[go_id] - removed[go_id]
        study_local = study - removed[go_id]
        hits = in_pop & study_local
        if not hits or not study_local:
            continue
        r = _make_result(go, go_id, hits, in_pop, len(study_local), N_all)
        results.append(r)
        if r.p_value < alpha:
            for anc in go.ancestors(go_id, include_self=False):
                if anc in removed:
                    removed[anc] |= hits
    return results


def _all_genes(go: GeneOntology) -> set[str]:
    key = "_all_genes_cache"
    cached = getattr(go, key, None)
    if cached is None:
        cached = set()
        for genes in go._term_genes.values():
            cached |= genes
        setattr(go, key, cached)
    return cached


def _aspect_letter(namespace: str) -> str:
    return {"biological_process": "P", "molecular_function": "F",
            "cellular_component": "C"}.get(namespace, "?")


def format_results(results: list[EnrichmentResult], top: int = 20,
                   max_fdr: float = 0.05) -> str:
    lines = [f"{'GO term':<12}{'name':<46}{'k/n':>10}{'fold':>7}{'p':>11}{'FDR':>11}"]
    shown = 0
    for r in results:
        if r.fdr > max_fdr:
            continue
        lines.append(
            f"{r.go_id:<12}{r.name[:44]:<46}{f'{r.n_study_hits}/{r.n_term_genes}':>10}"
            f"{r.fold_enrichment:>7.1f}{r.p_value:>11.2e}{r.fdr:>11.2e}")
        shown += 1
        if shown >= top:
            break
    if shown == 0:
        lines.append(f"(no terms with FDR <= {max_fdr})")
    return "\n".join(lines)
