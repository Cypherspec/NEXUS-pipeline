"""NEXUS backend actions for the Mantis extension host.

Mantis loads this module and calls ``invoke(action, payload, ctx)``. The
analysis itself is the ``nexus`` package, vendored beside this file by
``build.mjs`` so the extension carries no import path outside its own
directory; backend files are never exposed to the panel.

Two limits shape every action here. Mantis caps a backend invocation at 30
seconds, and the panel RPC at the same, so nothing may block indefinitely:
each action bounds its own input and returns a count of what it dropped
rather than working through an unbounded selection. And results must be
JSON-serialisable, so molecules cross the boundary as SMILES strings and
point ids, never as objects.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:                      # vendored package sits beside this file
    sys.path.insert(0, _HERE)

import numpy as np

# Largest input any single action will consider. Chosen so the slowest action
# (a Gaussian process over Morgan fingerprints) stays well inside the 30 s cap
# on this hardware; anything beyond it is reported as skipped, not silently cut.
MAX_MOLECULES = 1500
MAX_PREDICT = 600


class PayloadError(ValueError):
    """Raised when the panel sends something this backend cannot act on."""


def _structures(payload: dict, limit: int = MAX_MOLECULES) -> tuple[list, list[str], list[str], int]:
    """Parses the payload's molecules.

    @param payload Panel payload carrying ``molecules``: ``{id, smiles}`` records.
    @param limit Largest number of molecules to parse.
    @returns Parsed molecules, their SMILES, their point ids, and the number
             of records dropped because they were unparseable or over the limit.
    """
    from nexus.chem.smiles import parse_smiles

    records = payload.get("molecules")
    if not isinstance(records, list):
        raise PayloadError("'molecules' must be a list of {id, smiles} records")

    over_limit = max(0, len(records) - limit)
    mols, smiles, ids, failed = [], [], [], 0
    for record in records[:limit]:
        if not isinstance(record, dict):
            failed += 1
            continue
        text = record.get("smiles")
        if not isinstance(text, str) or not text.strip():
            failed += 1
            continue
        try:
            mol = parse_smiles(text.strip())
        except Exception:                                      # noqa: BLE001
            failed += 1
            continue
        mols.append(mol)
        smiles.append(text.strip())
        ids.append(str(record.get("id", "")))
    return mols, smiles, ids, failed + over_limit


def analyze(payload: dict, ctx: dict) -> dict:
    """Descriptors, rule-of-five status and scaffold families."""
    from nexus.chem.descriptors import compute_descriptors, veber_pass
    from nexus.chem.scaffold import murcko_scaffold

    started = time.time()
    mols, smiles, ids, dropped = _structures(payload)
    if not mols:
        raise PayloadError("no parseable molecules in the selection")

    rows, scaffolds = [], {}
    for mol, text, point_id in zip(mols, smiles, ids):
        descriptors = compute_descriptors(mol)
        scaffold = murcko_scaffold(mol)
        scaffolds.setdefault(scaffold, []).append(point_id)
        row = {"id": point_id, "smiles": text, "formula": mol.formula(),
               "veberPass": bool(veber_pass(mol))}
        # Every descriptor the package computes crosses as a plain float, so
        # the panel can offer any of them without the backend choosing first.
        row.update({key: (None if value is None else float(value))
                    for key, value in descriptors.items()})
        rows.append(row)

    families = sorted(
        ({"scaffold": key or "(acyclic)", "count": len(members), "pointIds": members}
         for key, members in scaffolds.items()),
        key=lambda f: -f["count"])

    return {
        "molecules": rows,
        "scaffoldFamilies": families[:40],
        "nScaffolds": len(scaffolds),
        "nAnalyzed": len(rows),
        "nDropped": dropped,
        "elapsedSeconds": round(time.time() - started, 3),
    }


def admet(payload: dict, ctx: dict) -> dict:
    """ADMET screening flags per molecule, plus counts over the set.

    Every endpoint is a published rule evaluated on this package's descriptors;
    see nexus/chem/admet.py for the sources and for what each one does and does
    not claim. Molecules are returned with their flags so the panel can rank
    the set, and `summary` carries the aggregate the header reports.
    """
    from nexus.chem.admet import admet_profile, admet_summary

    started = time.time()
    mols, smiles, ids, dropped = _structures(payload)
    if not mols:
        raise PayloadError("no parseable molecules in the selection")

    rows = []
    for mol, text, point_id in zip(mols, smiles, ids):
        row = {"id": point_id, "smiles": text, "formula": mol.formula()}
        row.update(admet_profile(mol).as_dict())
        rows.append(row)

    # Worst first: a triage list is read from the top.
    rows.sort(key=lambda r: (-r["riskFlags"], r["logSolubility"]))

    return {
        "molecules": rows,
        "summary": admet_summary(mols),
        "nAnalyzed": len(rows),
        "nDropped": dropped,
        "elapsedSeconds": round(time.time() - started, 3),
    }


def neighbors(payload: dict, ctx: dict) -> dict:
    """Nearest neighbours of one molecule by Tanimoto similarity.

    Drives the edge overlay: the panel draws an edge from the clicked point to
    each neighbour, so chemical similarity is visible as map structure rather
    than as a list of numbers.
    """
    from nexus.chem.fingerprint import bulk_tanimoto, fingerprint_matrix

    mols, smiles, ids, dropped = _structures(payload)
    focus_id = str(payload.get("focusId", ""))
    if focus_id not in ids:
        raise PayloadError("'focusId' is not among the supplied molecules")
    k = int(payload.get("k", 8))
    if not 1 <= k <= 50:
        raise PayloadError("'k' must be between 1 and 50")

    matrix = fingerprint_matrix(mols, kind="morgan", n_bits=1024)
    index = ids.index(focus_id)
    scores = bulk_tanimoto(matrix[index], matrix)
    order = np.argsort(scores)[::-1]
    out = []
    for position in order:
        if ids[position] == focus_id:
            continue
        out.append({"id": ids[position], "smiles": smiles[position],
                    "similarity": float(scores[position])})
        if len(out) >= k:
            break
    return {"focusId": focus_id, "neighbors": out, "nDropped": dropped}


def screen(payload: dict, ctx: dict) -> dict:
    """Ranks unmeasured molecules by acquisition value."""
    from nexus.al.acquisition import acquire, select_batch
    from nexus.al.gp import GaussianProcess
    from nexus.chem.fingerprint import fingerprint_matrix

    mols, smiles, ids, dropped = _structures(payload, MAX_PREDICT)
    measured = payload.get("measured") or {}
    if not isinstance(measured, dict):
        raise PayloadError("'measured' must be a mapping of point id to value")
    batch_size = int(payload.get("batchSize", 24))
    if not 1 <= batch_size <= 200:
        raise PayloadError("'batchSize' must be between 1 and 200")
    strategy = str(payload.get("strategy", "ei"))

    X = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)
    known = [i for i, point_id in enumerate(ids)
             if isinstance(measured.get(point_id), (int, float))]
    pool = [i for i in range(len(ids)) if i not in set(known)]
    if not pool:
        raise PayloadError("every molecule already carries a measurement")

    if len(known) >= 5:
        y = np.array([float(measured[ids[i]]) for i in known])
        model = GaussianProcess(kernel="tanimoto", noise=1e-2).fit(X[known], y)
        mean, std = model.predict(X[pool])
        scores = acquire(strategy, mean, std, best=float(np.max(y)))
        rationale = (f"{strategy} acquisition over {len(pool)} unmeasured molecules, "
                     f"informed by {len(known)} measurements")
    else:
        # Cold start: no model is credible yet, so rank by how far each molecule
        # sits from the rest rather than pretending to a posterior.
        scores = np.ones(len(pool))
        rationale = (f"{len(known)} measurements is too few to fit a model; "
                     f"ranking {len(pool)} molecules by diversity alone")

    chosen = select_batch(scores, batch_size, features=X[pool], diversity_weight=0.3)
    return {
        "batch": [{"id": ids[pool[i]], "smiles": smiles[pool[i]],
                   "score": float(scores[i])} for i in chosen],
        "nMeasured": len(known), "nPool": len(pool),
        "rationale": rationale, "nDropped": dropped,
    }


def predict(payload: dict, ctx: dict) -> dict:
    """Predicts the unlabelled molecules with conformal intervals."""
    from nexus.al.gp import GaussianProcess
    from nexus.chem.fingerprint import fingerprint_matrix
    from nexus.uq.conformal import ConformalRegressor

    mols, smiles, ids, dropped = _structures(payload, MAX_PREDICT)
    measured = payload.get("measured") or {}
    if not isinstance(measured, dict):
        raise PayloadError("'measured' must be a mapping of point id to value")
    alpha = float(payload.get("alpha", 0.1))
    if not 0.0 < alpha < 0.5:
        raise PayloadError("'alpha' must lie strictly between 0 and 0.5")
    adaptive = bool(payload.get("adaptive", False))

    X = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)
    known = [i for i, point_id in enumerate(ids)
             if isinstance(measured.get(point_id), (int, float))]
    if len(known) < 20:
        raise PayloadError("at least 20 measured molecules are needed to both "
                           "fit a model and calibrate its intervals")
    y = np.array([float(measured[ids[i]]) for i in known])

    rng = np.random.default_rng(int(payload.get("seed", 0)))
    shuffled = rng.permutation(len(known))
    n_calib = max(10, int(0.3 * len(shuffled)))
    calib = [known[i] for i in shuffled[:n_calib]]
    fit = [known[i] for i in shuffled[n_calib:]]
    y_calib = np.array([float(measured[ids[i]]) for i in calib])

    model = GaussianProcess(kernel="tanimoto", noise=1e-2).fit(
        X[fit], np.array([float(measured[ids[i]]) for i in fit]))
    mean_calib, std_calib = model.predict(X[calib])

    target = [i for i in range(len(ids)) if i not in set(known)] or list(range(len(ids)))
    mean, std = model.predict(X[target])

    conformal = ConformalRegressor(alpha=alpha, normalize=adaptive)
    conformal.fit(mean_calib, y_calib, std_calib if adaptive else None)
    low, high = conformal.predict_interval(mean, std if adaptive else None)

    rows = [{"id": ids[t], "smiles": smiles[t], "prediction": float(mean[k]),
             "std": float(std[k]), "low": float(low[k]), "high": float(high[k])}
            for k, t in enumerate(target)]
    widths = [r["high"] - r["low"] for r in rows]
    return {
        "predictions": rows,
        "nLabelled": len(known), "nCalibration": len(calib),
        "nominalCoverage": 1.0 - alpha,
        "calibrationCoverage": float(conformal.coverage(
            mean_calib, y_calib, std_calib if adaptive else None)),
        "intervalMode": "adaptive" if adaptive else "constant-width",
        "meanIntervalWidth": float(np.mean(widths)) if widths else None,
        "nDropped": dropped,
    }


def conformer(payload: dict, ctx: dict) -> dict:
    """A 3D conformer for one molecule, for the rotating structure view.

    The geometry is distance-geometry embedding followed by relaxation under an
    explicit out-of-plane term -- the same code the paper validates on benzene,
    where distance restraints alone leave the ring puckered. Coordinates are
    centred and scaled here so the panel can draw them without knowing the
    chemistry.
    """
    from nexus.chem.conformer import embed_molecule
    from nexus.chem.periodic import covalent_radius
    from nexus.chem.smiles import parse_smiles

    text = payload.get("smiles")
    if not isinstance(text, str) or not text.strip():
        raise PayloadError("'smiles' must be a non-empty string")
    try:
        mol = parse_smiles(text.strip())
    except Exception as exc:                                   # noqa: BLE001
        raise PayloadError(f"could not parse that structure: {exc}") from exc
    if mol.n_atoms > 120:
        raise PayloadError("structure is too large for the 3D view")

    xyz, restraint_error = embed_molecule(mol, seed=int(payload.get("seed", 0)))
    xyz = np.asarray(xyz, dtype=float)
    xyz -= xyz.mean(axis=0, keepdims=True)
    extent = float(np.abs(xyz).max()) or 1.0
    xyz = xyz / extent                                          # unit box

    atoms = [{"symbol": atom.symbol,
              "radius": float(covalent_radius(atom.symbol) or 0.7),
              "aromatic": bool(atom.aromatic),
              "hydrogens": int(atom.total_h),
              "x": float(xyz[i, 0]), "y": float(xyz[i, 1]), "z": float(xyz[i, 2])}
             for i, atom in enumerate(mol.atoms)]
    bonds = [{"a": int(b.a), "b": int(b.b), "order": int(b.order),
              "inRing": bool(b.in_ring)} for b in mol.bonds]
    return {
        "smiles": text.strip(), "formula": mol.formula(),
        "atoms": atoms, "bonds": bonds,
        "restraintError": float(restraint_error),
        "extentAngstrom": extent,
    }


def space3d(payload: dict, ctx: dict) -> dict:
    """Three-component projection of the selection's chemical space.

    The map itself is two-dimensional; this returns the third component so the
    panel can rotate the same molecules in depth. Components come from an SVD
    of the centred fingerprint matrix, which is the same construction the
    seeded space's own layout uses, carried one axis further.
    """
    from nexus.chem.fingerprint import fingerprint_matrix

    mols, smiles, ids, dropped = _structures(payload, MAX_PREDICT)
    if len(mols) < 4:
        raise PayloadError("at least four molecules are needed to project a space")

    X = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(X, full_matrices=False)
    coords = X @ vt[:3].T
    scale = float(np.abs(coords).max()) or 1.0
    coords = coords / scale

    variance = (singular ** 2)
    explained = (variance[:3] / variance.sum()).tolist() if variance.sum() else [0.0, 0.0, 0.0]

    colour_key = str(payload.get("colourBy", "drug_likeness"))
    values = []
    if colour_key:
        from nexus.chem.descriptors import compute_descriptors
        for mol in mols:
            descriptors = compute_descriptors(mol)
            value = descriptors.get(colour_key)
            values.append(None if value is None else float(value))

    points = [{"id": ids[i], "smiles": smiles[i],
               "x": float(coords[i, 0]), "y": float(coords[i, 1]),
               "z": float(coords[i, 2]),
               "value": values[i] if i < len(values) else None}
              for i in range(len(ids))]
    return {"points": points, "explainedVariance": explained,
            "colourBy": colour_key, "nDropped": dropped}


def shape(payload: dict, ctx: dict) -> dict:
    """Topology and curvature of one molecule.

    Two shape descriptions the pipeline computes and nothing in the map shows:

    Persistent homology of the 3D conformer, where a long-lived H1 feature is a
    ring -- benzene gives exactly one, acyclic octane none, which is the check
    the paper reports.

    Ollivier-Ricci curvature of the bond graph, computed with an exact
    transportation simplex rather than an entropic approximation. It is signed,
    and zero is meaningful: a cycle graph has curvature exactly zero, so a
    benzene ring reads flat while a fused or branched position does not.
    """
    import numpy as _np

    from nexus.chem.conformer import embed_molecule
    from nexus.chem.smiles import parse_smiles
    from nexus.topo.homology import diagram_statistics, rips_persistence
    from nexus.topo.ricci import forman_ricci_curvature, ollivier_ricci_curvature

    text = payload.get("smiles")
    if not isinstance(text, str) or not text.strip():
        # No molecule named: answer about the most structurally interesting one
        # in scope rather than the first, which is an arbitrary choice and is
        # usually a small acyclic fragment with no topology to show. Ring bonds
        # rank it, with size as the tie-break.
        candidates, candidate_smiles, _, _ = _structures(payload, 400)
        if not candidates:
            raise PayloadError("'smiles' must be given, or 'molecules' to choose from")
        ranked = sorted(
            zip(candidates, candidate_smiles),
            key=lambda pair: (sum(1 for b in pair[0].bonds if b.in_ring),
                              pair[0].n_atoms),
            reverse=True)
        mol, text = ranked[0][0], ranked[0][1]
        if mol.n_atoms > 60:
            # Fall back to the largest structure that fits the budget.
            inside = [pair for pair in ranked if pair[0].n_atoms <= 60]
            if not inside:
                raise PayloadError("every molecule in scope is too large for the shape analysis")
            mol, text = inside[0]
    else:
        text = text.strip()
        try:
            mol = parse_smiles(text)
        except Exception as exc:                               # noqa: BLE001
            raise PayloadError(f"could not parse that structure: {exc}") from exc
    # Rips persistence builds simplices over every point, so the cost climbs
    # steeply; the cap keeps a single call well inside the 30 s budget.
    if mol.n_atoms > 60:
        raise PayloadError("structure is too large for the shape analysis")

    xyz, restraint_error = embed_molecule(mol, seed=int(payload.get("seed", 0)))
    xyz = _np.asarray(xyz, dtype=float)

    diagram = rips_persistence(xyz, max_dim=1)
    # diagram.points is keyed by homology dimension, each value an array of
    # (birth, death) pairs. An infinite death is the component that never
    # merges; it carries no scale and is dropped rather than drawn at the edge.
    features = []
    for dim, pairs in diagram.points.items():
        for birth, death in _np.asarray(pairs, dtype=float).reshape(-1, 2):
            if not _np.isfinite(death):
                continue
            features.append({"dim": int(dim), "birth": float(birth),
                             "death": float(death), "life": float(death - birth)})
    features.sort(key=lambda f: -f["life"])

    ollivier = ollivier_ricci_curvature(mol)
    forman = forman_ricci_curvature(mol)
    bonds = []
    for bond in mol.bonds:
        key = (bond.a, bond.b) if (bond.a, bond.b) in ollivier else (bond.b, bond.a)
        bonds.append({
            "a": int(bond.a), "b": int(bond.b), "order": int(bond.order),
            "inRing": bool(bond.in_ring),
            "ollivier": float(ollivier.get(key, 0.0)),
            "forman": float(forman.get(key, 0.0)),
        })

    curvatures = [b["ollivier"] for b in bonds]
    centred = xyz - xyz.mean(axis=0, keepdims=True)
    extent = float(_np.abs(centred).max()) or 1.0
    return {
        "smiles": text, "formula": mol.formula(),
        "atoms": [{"symbol": a.symbol, "x": float(centred[i, 0] / extent),
                   "y": float(centred[i, 1] / extent),
                   "z": float(centred[i, 2] / extent)}
                  for i, a in enumerate(mol.atoms)],
        "bonds": bonds,
        # Cap per dimension, not globally. A 42-atom molecule has 41 H0
        # features whose lifetimes all exceed any loop's, so a single
        # most-persistent-40 cut silently drops every H1 feature -- the one
        # thing this view exists to show.
        "features": ([f for f in features if f["dim"] == 1][:30]
                     + [f for f in features if f["dim"] != 1][:30]),
        "h1": {k: float(v) for k, v in diagram_statistics(diagram, 1).items()},
        "h0": {k: float(v) for k, v in diagram_statistics(diagram, 0).items()},
        "curvature": {
            "min": min(curvatures) if curvatures else 0.0,
            "max": max(curvatures) if curvatures else 0.0,
            "mean": float(_np.mean(curvatures)) if curvatures else 0.0,
            "magnitude": max((abs(c) for c in curvatures), default=0.0),
        },
        "restraintError": float(restraint_error),
        "ringBonds": int(sum(1 for b in mol.bonds if b.in_ring)),
    }


# The pruned ontology is loaded once per process and kept: it is a few hundred
# thousand integers, and an action that reloaded it would spend most of its
# 30 s budget parsing JSON.
_GO_INDEX: dict | None = None


def _go_index() -> dict:
    """Loads the shipped Gene Ontology slice, once."""
    global _GO_INDEX
    if _GO_INDEX is None:
        path = os.path.join(_HERE, "go_index.json")
        if not os.path.exists(path):
            raise PayloadError("the Gene Ontology index is not present in this build")
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        raw["gene_lookup"] = {gene.upper(): i for i, gene in enumerate(raw["genes"])}
        # Membership tests dominate the enrichment loop, so each term's gene
        # list becomes a set once rather than on every call.
        for term in raw["terms"]:
            term["gene_set"] = set(term["genes"])
        _GO_INDEX = raw
    return _GO_INDEX



def _resolve_symbol(name: str, lookup: dict) -> tuple[int | None, str | None]:
    """Resolves a gene symbol, trying only transformations that must land on a
    real gene.

    Target panels write names the ontology does not use: a mutation in
    parentheses, a trailing ``p`` for the phosphorylated form, a ``-cyclinD1``
    suffix for a complex. Each candidate below is accepted only if it resolves,
    so a transformation can never silently substitute a different gene -- and
    anything still unresolved is reported rather than dropped. No alias table
    is hand-written here: inventing a mapping is how an enrichment ends up
    describing biology that was never assayed.

    @param name Symbol as the map wrote it.
    @param lookup Uppercased symbol to gene index.
    @returns The gene index and the form that matched, or (None, None).
    """
    raw = name.strip()
    if not raw:
        return None, None

    candidates = [raw]
    stripped = re.sub(r"\(.*?\)", "", raw).strip()
    if stripped and stripped != raw:
        candidates.append(stripped)
    for base in list(candidates):
        # Trailing lowercase p: the phosphorylated form of the same protein.
        if len(base) > 2 and base.endswith("p") and not base.endswith("PP"):
            candidates.append(base[:-1])
        # A complex written as kinase-partner.
        if "-" in base:
            candidates.append(base.split("-", 1)[0])

    for candidate in candidates:
        idx = lookup.get(candidate.strip().upper())
        if idx is not None:
            return idx, candidate.strip()
    return None, None


def enrich(payload: dict, ctx: dict) -> dict:
    """Gene Ontology over-representation for the targets behind a selection.

    The map holds molecules; this answers a question about the proteins they
    are annotated against. Genes are matched case-insensitively by symbol, and
    anything the ontology does not carry is returned as unmatched rather than
    quietly dropped -- an enrichment run against a silently halved gene list is
    worse than no enrichment at all.

    The background is the ontology's own annotated gene set unless the caller
    supplies one. Supplying the assayed panel is the more defensible choice
    when the study genes were drawn from it, and the response records which
    was used.
    """
    from nexus.bio.enrichment import benjamini_hochberg, hypergeometric_sf

    index = _go_index()
    lookup = index["gene_lookup"]

    requested = payload.get("genes")
    if not isinstance(requested, list) or not requested:
        raise PayloadError("'genes' must be a non-empty list of gene symbols")
    if len(requested) > 5000:
        raise PayloadError("at most 5000 genes can be tested at once")

    study, unmatched, rewritten, resolved = set(), [], 0, 0
    for name in requested:
        if not isinstance(name, str) or not name.strip():
            continue
        idx, matched_as = _resolve_symbol(name, lookup)
        if idx is None:
            unmatched.append(name.strip())
            continue
        if matched_as and matched_as.upper() != name.strip().upper():
            rewritten += 1
        # Names and genes are different counts: a kinase panel writes one gene
        # under several names, so these must be reported separately or the
        # matched and unmatched totals will not add up to what was sent.
        resolved += 1
        study.add(idx)
    if not study:
        raise PayloadError("none of those gene symbols appear in the ontology")

    background = payload.get("background")
    if isinstance(background, list) and background:
        pool = {lookup[n.strip().upper()] for n in background
                if isinstance(n, str) and n.strip().upper() in lookup}
        pool |= study
        background_label = "supplied"
    else:
        pool = set(range(len(index["genes"])))
        background_label = "all annotated genes"

    population = len(pool)
    drawn = len(study)
    alpha = float(payload.get("alpha", 0.05))
    if not 0.0 < alpha < 1.0:
        raise PayloadError("'alpha' must lie strictly between 0 and 1")

    # Every term in the window is a test, including the ones that came back
    # with nothing. Restricting the correction to terms that happened to have
    # hits selects the tests by their own outcome and shrinks the denominator
    # to the handful most likely to look significant: on random 20-gene sets
    # that produced a mean of 17.8 "significant" terms where the correct
    # procedure gives well under one.
    rows, pvalues = [], []
    for term in index["terms"]:
        members = term["gene_set"] & pool
        if len(members) < 2:
            continue                                   # not a test either way
        hits = members & study
        p = (hypergeometric_sf(len(hits), population, len(members), drawn)
             if hits else 1.0)
        pvalues.append(p)
        expected = drawn * len(members) / population if population else 0.0
        rows.append({
            "id": term["id"], "name": term["name"],
            "studyHits": len(hits), "termSize": len(members),
            "expected": expected,
            "foldEnrichment": (len(hits) / expected) if expected > 0 else 0.0,
            "pValue": p,
            "genes": sorted(index["genes"][i] for i in hits)[:40],
        })

    if rows:
        adjusted = benjamini_hochberg(np.asarray(pvalues, dtype=float))
        for row, q in zip(rows, adjusted):
            row["fdr"] = float(q)
        # Reported rows still need at least two study genes to be worth
        # reading, but that filter is applied after the correction, not before.
        rows = [r for r in rows if r["studyHits"] >= 2]
        rows.sort(key=lambda r: (r["fdr"], r["pValue"], -r["foldEnrichment"]))

    return {
        "terms": rows[:40],
        "nTested": len(pvalues),
        "nSignificant": sum(1 for r in rows if r["fdr"] <= alpha),
        "nStudyGenes": drawn,
        "nUnmatched": len(unmatched),
        "nRewritten": rewritten,
        "nResolvedNames": resolved,
        "unmatched": unmatched[:25],
        "population": population,
        "background": background_label,
        "alpha": alpha,
        "window": index["window"],
        "aspect": index["source"]["aspect"],
    }


_ACTIONS = {
    "analyze": analyze,
    "admet": admet,
    "neighbors": neighbors,
    "screen": screen,
    "predict": predict,
    "conformer": conformer,
    "space3d": space3d,
    "shape": shape,
    "enrich": enrich,
}


def invoke(action: str, payload: dict, ctx: dict):
    """Single dispatcher, so every action shares the same payload guards."""
    handler = _ACTIONS.get(action)
    if handler is None:
        raise ValueError(f"Unknown action: {action}")
    return handler(payload or {}, ctx or {})
