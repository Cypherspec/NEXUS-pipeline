"""Structure-activity landscape analysis over a set of molecules.

A Mantis map places molecules by similarity, which makes two questions natural
that no single-molecule calculation can answer:

    where is the landscape rough?   Two molecules that sit next to each other
                                    but behave differently are an activity
                                    cliff. They are where the structure-activity
                                    relationship actually lives, and where a
                                    model fitted on neighbourhood similarity
                                    will be wrong.

    what makes this cluster?        A region picked out on the map differs from
                                    the rest in some way. Which scaffolds and
                                    functional groups are over-represented in
                                    it, and is the difference larger than
                                    chance would give.

Both are computed from the package's own fingerprints and descriptors, and both
return pairs or groups a chemist can open and check rather than a single score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus.bio.enrichment import benjamini_hochberg, hypergeometric_sf
from nexus.chem.descriptors import count_functional_groups
from nexus.chem.fingerprint import morgan_fingerprint
from nexus.chem.scaffold import murcko_scaffold


@dataclass
class Cliff:
    """A pair of similar molecules whose measured values differ sharply."""

    i: int
    j: int
    similarity: float
    delta: float
    #: Structure-Activity Landscape Index: the change in value per unit of
    #: structural change. Large where a small edit flips behaviour.
    sali: float


def similarity_matrix(mols, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """All-pairs MinMax similarity over Morgan *count* fingerprints.

    Counts rather than bits, because binary ECFP saturates along a homologous
    series: hexanol and hexadecanol set the same bits -- every CH2 environment
    they contain is identical -- and score a Tanimoto of exactly 1.0 despite
    differing by ten carbons and six log units of solubility. Counting the
    occurrences separates them, and MinMax,

        sum_k min(a_k, b_k) / sum_k max(a_k, b_k)

    is the count-vector generalisation of Tanimoto that reduces to it when
    every count is 0 or 1. Getting this right matters most exactly where it is
    used: a cliff detector built on saturating fingerprints reports chain
    extensions as its strongest cliffs.

    Quadratic in the number of molecules, which is the point: the caller is
    asking about the landscape, not about one molecule. The panel bounds the
    set before it reaches here.
    """
    fps = np.array([morgan_fingerprint(m, n_bits=n_bits, radius=radius, counts=True)
                    for m in mols], dtype=np.float64)
    n = len(fps)
    similarity = np.ones((n, n), dtype=float)
    # min and max over a pair of count vectors have no matrix-product form, so
    # this is a row at a time: still vectorised over the other n molecules.
    for i in range(n):
        row = fps[i]
        lower = np.minimum(fps, row).sum(axis=1)
        upper = np.maximum(fps, row).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity[i] = np.where(upper > 0, lower / upper, 0.0)
    return similarity


def activity_cliffs(mols, values: np.ndarray, similarity_min: float = 0.6,
                    top: int = 40, sim: np.ndarray | None = None) -> list[Cliff]:
    """Pairs that are structurally close but far apart in measured value.

    @param similarity_min Tanimoto floor for a pair to count as "the same
                          neighbourhood". 0.6 on Morgan counts is the
                          conventional line for related compounds; below it a
                          large delta is unremarkable.
    @param top How many pairs to return, ranked by SALI.

    SALI is delta / (1 - similarity), so a pair that is nearly identical and
    still differs is ranked above a merely similar one with the same delta.
    Identical fingerprints would divide by zero and are floored instead.
    """
    values = np.asarray(values, dtype=float)
    S = similarity_matrix(mols) if sim is None else sim
    n = len(mols)
    iu, ju = np.triu_indices(n, k=1)
    s = S[iu, ju]
    # A pair that is identical to the fingerprint is not a cliff: it is the
    # same structure measured twice, and the delta is disagreement between
    # assays rather than chemistry. `duplicate_values` reports those.
    keep = (s >= similarity_min) & (s < 1.0)
    if not np.any(keep):
        return []
    iu, ju, s = iu[keep], ju[keep], s[keep]
    delta = np.abs(values[iu] - values[ju])
    sali = delta / (1.0 - s)
    order = np.argsort(-sali)[:top]
    return [Cliff(int(iu[k]), int(ju[k]), float(s[k]), float(delta[k]), float(sali[k]))
            for k in order]


def duplicate_values(mols, values: np.ndarray, sim: np.ndarray | None = None,
                     tolerance: float = 0.0) -> list[tuple[int, int, float]]:
    """Pairs that this fingerprint cannot tell apart but whose values differ.

    Worth separating from cliffs for two reasons: a repeated structure with
    two different measurements bounds how well any model can do on that
    endpoint, and a pair that differs only in stereochemistry will land here,
    since the fingerprint is computed without it.
    """
    values = np.asarray(values, dtype=float)
    S = similarity_matrix(mols) if sim is None else sim
    iu, ju = np.triu_indices(len(mols), k=1)
    s = S[iu, ju]
    delta = np.abs(values[iu] - values[ju])
    hit = (s >= 1.0) & (delta > tolerance)
    order = np.argsort(-delta[hit])
    return [(int(iu[hit][k]), int(ju[hit][k]), float(delta[hit][k])) for k in order]


def landscape_roughness(mols, values: np.ndarray, similarity_min: float = 0.6,
                        sim: np.ndarray | None = None) -> dict:
    """How rough the structure-activity landscape is over the whole set.

    The correlation between pairwise structural similarity and pairwise value
    similarity. Near 1 means the property varies smoothly with structure, so a
    neighbourhood model will work; near 0 means neighbours tell you little, and
    the cliffs below are where the information is.
    """
    values = np.asarray(values, dtype=float)
    S = similarity_matrix(mols) if sim is None else sim
    n = len(mols)
    iu, ju = np.triu_indices(n, k=1)
    s = S[iu, ju]
    d = np.abs(values[iu] - values[ju])
    spread = float(np.ptp(values))
    if spread <= 0 or len(s) < 3 or np.std(s) < 1e-12 or np.std(d) < 1e-12:
        return {"correlation": float("nan"), "nPairs": int(len(s)),
                "cliffPairs": 0, "meanSimilarity": float(np.mean(s)) if len(s) else 0.0}
    close = s >= similarity_min
    return {
        # Negative correlation between similarity and |delta| is the smooth
        # case; it is reported with the sign flipped so that larger is smoother.
        "correlation": float(-np.corrcoef(s, d)[0, 1]),
        "nPairs": int(len(s)),
        "nNeighbourPairs": int(close.sum()),
        "meanSimilarity": float(np.mean(s)),
        "meanNeighbourDelta": float(np.mean(d[close])) if close.any() else float("nan"),
        "valueRange": spread,
    }


# ------------------------------------------------------------------ regions

@dataclass
class EnrichedFeature:
    """A scaffold or functional group over-represented in a region."""

    feature: str
    kind: str                 # "scaffold" or "group"
    in_region: int
    in_background: int
    region_size: int
    background_size: int
    fold: float
    p_value: float
    q_value: float = float("nan")


def _feature_table(mols) -> list[set[str]]:
    """Each molecule's scaffold and its functional groups, as a feature set."""
    rows = []
    for mol in mols:
        features = set()
        scaffold = murcko_scaffold(mol)
        features.add(f"scaffold:{scaffold or '(acyclic)'}")
        for name, count in count_functional_groups(mol).items():
            if count > 0:
                features.add(f"group:{name}")
        rows.append(features)
    return rows


def region_enrichment(mols, in_region: np.ndarray, min_count: int = 3,
                      alpha: float = 0.05) -> list[EnrichedFeature]:
    """What distinguishes a selected region of the map from the rest.

    A one-sided hypergeometric test per feature, Benjamini-Hochberg corrected,
    which is the same machinery the Targets tab uses for Gene Ontology. Features
    appearing fewer than `min_count` times anywhere are dropped before
    correction rather than after, so they do not inflate the number of tests.

    @param in_region Boolean mask over `mols`.
    """
    mask = np.asarray(in_region, dtype=bool)
    rows = _feature_table(mols)
    N, n = len(rows), int(mask.sum())
    if n == 0 or n == N:
        return []

    totals: dict[str, int] = {}
    hits: dict[str, int] = {}
    for features, inside in zip(rows, mask):
        for feature in features:
            totals[feature] = totals.get(feature, 0) + 1
            if inside:
                hits[feature] = hits.get(feature, 0) + 1

    candidates = [f for f, total in totals.items()
                  if total >= min_count and hits.get(f, 0) >= 1]
    if not candidates:
        return []

    results = []
    for feature in candidates:
        k, K = hits.get(feature, 0), totals[feature]
        expected = n * K / N
        results.append(EnrichedFeature(
            feature=feature.split(":", 1)[1], kind=feature.split(":", 1)[0],
            in_region=k, in_background=K - k, region_size=n, background_size=N - n,
            fold=float(k / expected) if expected > 0 else float("inf"),
            p_value=float(hypergeometric_sf(k, N, K, n))))

    q = benjamini_hochberg(np.array([r.p_value for r in results]))
    for result, value in zip(results, q):
        result.q_value = float(value)

    results = [r for r in results if r.q_value <= alpha and r.fold > 1.0]
    results.sort(key=lambda r: (r.q_value, -r.fold))
    return results


def diversity(mols, sim: np.ndarray | None = None) -> dict:
    """How well a set covers its own chemical space.

    Mean nearest-neighbour distance is the useful summary: a set where every
    molecule has a near-twin is redundant however many compounds it holds, and
    that is a property of the set, not of any molecule in it.
    """
    S = similarity_matrix(mols) if sim is None else sim
    n = len(mols)
    if n < 2:
        return {"n": n, "meanNearestNeighbour": float("nan"), "redundantFraction": 0.0}
    np.fill_diagonal(S, -1.0)
    nearest = S.max(axis=1)
    np.fill_diagonal(S, 1.0)
    return {
        "n": n,
        "meanNearestNeighbour": float(np.mean(nearest)),
        # A conventional line for "the same compound series".
        "redundantFraction": float(np.mean(nearest >= 0.85)),
        "singletonFraction": float(np.mean(nearest < 0.4)),
    }
