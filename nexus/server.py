"""HTTP service exposing NEXUS analysis to a front end.

Uses only the Python standard library, so the pipeline keeps its NumPy-only
dependency footprint. Start it with:

    python -m nexus.server --port 8765

Endpoints (all JSON, all POST unless noted):
    GET  /health              service and capability report
    POST /analyze             descriptors, drug-likeness, alerts, scaffolds
    POST /scaffolds           scaffold grouping for a set of molecules
    POST /similarity          Tanimoto ranking of a library against a query
    POST /predict             property prediction with conformal intervals
    POST /screen              next batch to measure, by active learning
    POST /topology            persistent homology and curvature descriptors
    POST /enrichment          GO enrichment for a gene list
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import numpy as np

logger = logging.getLogger("nexus.server")

MAX_BODY_BYTES = 16 * 1024 * 1024
MAX_MOLECULES = 20_000


class NexusError(ValueError):
    """A client-visible error; reported as HTTP 400 with its message."""


# --------------------------------------------------------------------- helpers

def _parse_molecules(payload: dict, key: str = "smiles") -> tuple[list, list[str], list[int]]:
    """Parse a SMILES list, returning (molecules, smiles, indices of failures)."""
    from nexus.chem.smiles import read_smiles

    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise NexusError(f"'{key}' must be a non-empty list of SMILES strings")
    if len(raw) > MAX_MOLECULES:
        raise NexusError(f"at most {MAX_MOLECULES} molecules per request")
    mols, smiles, failed = [], [], []
    for i, item in enumerate(raw):
        if not isinstance(item, str):
            failed.append(i)
            continue
        mol = read_smiles(item)
        if mol is None or mol.n_atoms == 0:
            failed.append(i)
            continue
        mols.append(mol)
        smiles.append(item)
    if not mols:
        raise NexusError("no valid molecules in request")
    return mols, smiles, failed


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


# -------------------------------------------------------------------- handlers

def handle_health(_: dict) -> dict:
    from nexus import __version__

    return {
        "status": "ok",
        "version": __version__,
        "capabilities": [
            "analyze", "scaffolds", "similarity", "predict", "screen",
            "topology", "enrichment",
        ],
        "max_molecules": MAX_MOLECULES,
    }


def handle_analyze(payload: dict) -> dict:
    """Descriptors, rule compliance and alerts for each molecule."""
    from nexus.chem.descriptors import (
        compute_descriptors,
        count_structural_alerts,
        count_functional_groups,
        lipinski_violations,
        veber_pass,
    )
    from nexus.chem.scaffold import murcko_scaffold

    mols, smiles, failed = _parse_molecules(payload)
    include_groups = bool(payload.get("functional_groups", False))

    rows = []
    for mol, smi in zip(mols, smiles):
        descriptors = compute_descriptors(mol)
        row: dict[str, Any] = {
            "smiles": smi,
            "formula": mol.formula(),
            "n_atoms": mol.n_atoms,
            "descriptors": {k: _finite(v) for k, v in descriptors.items()},
            "lipinski_violations": lipinski_violations(mol),
            "veber_pass": veber_pass(mol),
            "structural_alerts": count_structural_alerts(mol),
            "scaffold": murcko_scaffold(mol),
            "valence_ok": mol.is_sane(),
        }
        if include_groups:
            row["functional_groups"] = {k: v for k, v in
                                        count_functional_groups(mol).items() if v}
        rows.append(row)

    values = {k: [r["descriptors"][k] for r in rows
                  if r["descriptors"][k] is not None]
              for k in rows[0]["descriptors"]}
    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)),
                   "min": float(np.min(v)), "max": float(np.max(v))}
               for k, v in values.items() if v}
    return {"n_analyzed": len(rows), "n_failed": len(failed),
            "failed_indices": failed, "molecules": rows, "summary": summary}


def handle_scaffolds(payload: dict) -> dict:
    """Group molecules by Bemis-Murcko scaffold."""
    from nexus.chem.scaffold import scaffold_groups

    mols, smiles, failed = _parse_molecules(payload)
    generic = bool(payload.get("generic", False))
    groups = scaffold_groups(mols, generic=generic)
    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    return {
        "n_molecules": len(mols),
        "n_scaffolds": len(groups),
        "n_failed": len(failed),
        "groups": [
            {"scaffold": scaffold or "(acyclic)",
             "count": len(members),
             "members": [smiles[i] for i in members[:200]]}
            for scaffold, members in ordered
        ],
    }


def handle_similarity(payload: dict) -> dict:
    """Rank a library by Tanimoto similarity to a query molecule."""
    from nexus.chem.fingerprint import bulk_tanimoto, fingerprint_matrix, morgan_fingerprint
    from nexus.chem.smiles import read_smiles

    query_smiles = payload.get("query")
    if not isinstance(query_smiles, str):
        raise NexusError("'query' must be a SMILES string")
    query = read_smiles(query_smiles)
    if query is None:
        raise NexusError(f"could not parse query SMILES {query_smiles!r}")

    mols, smiles, failed = _parse_molecules(payload, key="library")
    top_k = int(payload.get("top_k", 50))
    radius = int(payload.get("radius", 2))
    n_bits = int(payload.get("n_bits", 2048))

    library = fingerprint_matrix(mols, kind="morgan", radius=radius, n_bits=n_bits)
    scores = bulk_tanimoto(morgan_fingerprint(query, radius=radius, n_bits=n_bits),
                           library)
    order = np.argsort(-scores)[:top_k]
    return {
        "query": query_smiles,
        "n_library": len(mols),
        "n_failed": len(failed),
        "hits": [{"smiles": smiles[i], "tanimoto": float(scores[i]),
                  "index": int(i)} for i in order],
    }


def handle_predict(payload: dict) -> dict:
    """Train on labelled molecules and predict the rest, with conformal intervals.

    Expects `smiles` plus `labels` (null for molecules to predict). A fraction
    of the labelled data is held out to calibrate the intervals, so the stated
    coverage is a real guarantee rather than a model confidence score.
    """
    from nexus.al.gp import GaussianProcess
    from nexus.chem.fingerprint import fingerprint_matrix
    from nexus.uq.conformal import ConformalClassifier, ConformalRegressor

    mols, smiles, failed = _parse_molecules(payload)
    labels = payload.get("labels")
    if not isinstance(labels, list) or len(labels) != len(payload["smiles"]):
        raise NexusError("'labels' must be a list the same length as 'smiles', "
                         "with null for molecules to predict")
    # Re-index labels onto successfully parsed molecules.
    keep = [i for i in range(len(payload["smiles"])) if i not in set(failed)]
    labels = [labels[i] for i in keep]

    alpha = float(payload.get("alpha", 0.1))
    task = payload.get("task", "auto")
    y = np.array([np.nan if v is None else float(v) for v in labels])
    known = np.flatnonzero(np.isfinite(y))
    unknown = np.flatnonzero(~np.isfinite(y))
    if len(known) < 10:
        raise NexusError("at least 10 labelled molecules are required")
    if task == "auto":
        task = "classification" if set(np.unique(y[known])) <= {0.0, 1.0} else "regression"

    X = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)
    rng = np.random.default_rng(int(payload.get("seed", 0)))
    shuffled = rng.permutation(known)
    n_calib = max(10, int(0.3 * len(shuffled)))
    calib_idx, fit_idx = shuffled[:n_calib], shuffled[n_calib:]
    if len(fit_idx) < 5:
        raise NexusError("not enough labelled molecules to both fit and calibrate")

    model = GaussianProcess(kernel="tanimoto", noise=1e-2).fit(X[fit_idx], y[fit_idx])
    mean_calib, std_calib = model.predict(X[calib_idx])
    target = unknown if len(unknown) else known
    mean, std = model.predict(X[target])

    rows = []
    if task == "regression":
        # With the default absolute-residual score every interval has the same
        # width: valid, but it cannot say which predictions to trust. Setting
        # `adaptive` scales the score by the GP posterior sd, which keeps the
        # same marginal guarantee and lets width vary. Measured on ESOL,
        # FreeSolv and Lipo it improves coverage in the hardest quintile on two
        # of the three and leaves marginal coverage unchanged, so it is offered
        # rather than imposed (see results/conformal_adaptive.json).
        adaptive = bool(payload.get("adaptive", False))
        conformal = ConformalRegressor(alpha=alpha, normalize=adaptive)
        sigma_calib = std_calib if adaptive else None
        conformal.fit(mean_calib, y[calib_idx], sigma_calib)
        low, high = conformal.predict_interval(mean, std if adaptive else None)
        for k, idx in enumerate(target):
            rows.append({"smiles": smiles[idx], "prediction": float(mean[k]),
                         "std": float(std[k]), "interval_low": float(low[k]),
                         "interval_high": float(high[k])})
        coverage_check = conformal.coverage(mean_calib, y[calib_idx], sigma_calib)
    else:
        probs_calib = np.clip(np.c_[1 - mean_calib, mean_calib], 1e-6, 1 - 1e-6)
        probs = np.clip(np.c_[1 - mean, mean], 1e-6, 1 - 1e-6)
        conformal = ConformalClassifier(alpha=alpha).fit(
            probs_calib / probs_calib.sum(1, keepdims=True), y[calib_idx].astype(int))
        sets = conformal.predict_set(probs / probs.sum(1, keepdims=True))
        for k, idx in enumerate(target):
            included = [int(c) for c in range(2) if sets[k, c]]
            rows.append({"smiles": smiles[idx], "prediction": float(mean[k]),
                         "std": float(std[k]), "prediction_set": included,
                         "ambiguous": len(included) != 1})
        coverage_check = conformal.coverage(
            probs_calib / probs_calib.sum(1, keepdims=True), y[calib_idx].astype(int))

    widths = [r["interval_high"] - r["interval_low"] for r in rows
              if r.get("interval_high") is not None and r.get("interval_low") is not None]
    return {
        "task": task, "alpha": alpha, "nominal_coverage": 1 - alpha,
        "calibration_coverage": float(coverage_check),
        "interval_mode": ("adaptive" if bool(payload.get("adaptive", False))
                          else "constant-width") if task == "regression" else None,
        "mean_interval_width": float(np.mean(widths)) if widths else None,
        "n_labelled": int(len(known)), "n_calibration": int(len(calib_idx)),
        "n_predicted": len(rows), "predictions": rows,
    }


def handle_screen(payload: dict) -> dict:
    """Choose the next batch of molecules to measure."""
    from nexus.al.acquisition import acquire, select_batch
    from nexus.al.gp import GaussianProcess
    from nexus.chem.fingerprint import fingerprint_matrix

    mols, smiles, failed = _parse_molecules(payload)
    measured = payload.get("measured") or {}
    if not isinstance(measured, dict):
        raise NexusError("'measured' must be an object mapping SMILES to values")
    batch_size = int(payload.get("batch_size", 32))
    strategy = payload.get("strategy", "ei")
    diversity = float(payload.get("diversity", 0.0))

    X = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)
    known = [i for i, s in enumerate(smiles) if s in measured]
    pool = np.array([i for i in range(len(smiles)) if i not in set(known)])
    if len(pool) == 0:
        raise NexusError("every molecule has already been measured")

    if len(known) < 5:
        # Too little data to model: spread the first batch over the library.
        rng = np.random.default_rng(int(payload.get("seed", 0)))
        picks = rng.choice(pool, min(batch_size, len(pool)), replace=False)
        reason = "random seed batch (fewer than 5 measurements available)"
        scores = np.zeros(len(pool))
    else:
        y = np.array([float(measured[smiles[i]]) for i in known])
        model = GaussianProcess(kernel="tanimoto", noise=1e-2).fit(X[known], y)
        mean, std = model.predict(X[pool])
        scores = acquire(strategy, mean, std, float(y.max()))
        local = select_batch(scores, min(batch_size, len(pool)),
                             features=X[pool], diversity_weight=diversity)
        picks = pool[local]
        reason = f"{strategy} acquisition over {len(pool)} unmeasured molecules"

    lookup = {int(p): k for k, p in enumerate(pool)}
    return {
        "strategy": strategy, "batch_size": int(len(picks)),
        "n_measured": len(known), "n_pool": int(len(pool)),
        "rationale": reason, "n_failed": len(failed),
        "batch": [{"smiles": smiles[int(i)], "index": int(i),
                   "score": float(scores[lookup[int(i)]])} for i in picks],
    }


def handle_topology(payload: dict) -> dict:
    """Topological and curvature descriptors for each molecule."""
    from nexus.chem.conformer import embed_molecule
    from nexus.topo.homology import diagram_statistics, rips_persistence
    from nexus.topo.ricci import curvature_features, ollivier_ricci_curvature

    mols, smiles, failed = _parse_molecules(payload)
    if len(mols) > 500:
        raise NexusError("topology is limited to 500 molecules per request")
    include_3d = bool(payload.get("conformer", True))

    rows = []
    for mol, smi in zip(mols, smiles):
        curvature = np.array(list(ollivier_ricci_curvature(mol).values()))
        row: dict[str, Any] = {
            "smiles": smi,
            "n_rings": mol.n_rings,
            "ollivier_mean": _finite(curvature.mean()) if len(curvature) else 0.0,
            "ollivier_min": _finite(curvature.min()) if len(curvature) else 0.0,
            "curvature_features": [float(v) for v in curvature_features(mol)],
        }
        if include_3d and 3 <= mol.n_atoms <= 60:
            coords, energy = embed_molecule(mol, seed=0, n_tries=2, steps=200)
            diagram = rips_persistence(coords, max_dim=1)
            row["conformer_energy"] = float(energy)
            row.update({k: float(v) for k, v in diagram_statistics(diagram, 0).items()})
            row.update({k: float(v) for k, v in diagram_statistics(diagram, 1).items()})
        rows.append(row)
    return {"n_analyzed": len(rows), "n_failed": len(failed), "molecules": rows}


def handle_enrichment(payload: dict) -> dict:
    """GO enrichment for a gene list (requires the ontology files on disk)."""
    from pathlib import Path

    from nexus.bio.enrichment import enrichment
    from nexus.bio.gaf import parse_gaf
    from nexus.bio.ontology import GeneOntology

    genes = payload.get("genes")
    if not isinstance(genes, list) or not genes:
        raise NexusError("'genes' must be a non-empty list of gene symbols")
    aspect = payload.get("aspect", "P")
    data_dir = Path(payload.get("data_dir", "data/go"))
    obo, gaf = data_dir / "go-basic.obo", data_dir / "goa_human.gaf.gz"
    if not obo.exists() or not gaf.exists():
        raise NexusError(f"Gene Ontology files not found under {data_dir}")

    go = _ontology_cache(str(obo), str(gaf), aspect)
    results = enrichment(go, genes, aspect=aspect,
                         method=payload.get("method", "classic"))
    top = int(payload.get("top", 25))
    return {
        "n_genes": len(genes), "aspect": aspect,
        "n_terms_tested": len(results),
        "terms": [
            {"go_id": r.go_id, "name": r.name, "namespace": r.namespace,
             "n_study_hits": r.n_study_hits, "n_term_genes": r.n_term_genes,
             "fold_enrichment": r.fold_enrichment, "p_value": r.p_value,
             "fdr": r.fdr, "genes": list(r.genes[:40])}
            for r in results[:top]
        ],
    }


_ONTOLOGY: dict[tuple, Any] = {}


def _ontology_cache(obo: str, gaf: str, aspect: str):
    """Load the ontology once; it takes several seconds and is read-only."""
    key = (obo, gaf, aspect)
    if key not in _ONTOLOGY:
        go = GeneOntologyLoader(obo, gaf, aspect)
        _ONTOLOGY[key] = go
    return _ONTOLOGY[key]


def GeneOntologyLoader(obo: str, gaf: str, aspect: str):
    from nexus.bio.gaf import parse_gaf
    from nexus.bio.ontology import GeneOntology

    go = GeneOntology.load(obo)
    go.annotate(parse_gaf(gaf), aspect=aspect)
    return go


ROUTES: dict[str, Callable[[dict], dict]] = {
    "/health": handle_health,
    "/analyze": handle_analyze,
    "/scaffolds": handle_scaffolds,
    "/similarity": handle_similarity,
    "/predict": handle_predict,
    "/screen": handle_screen,
    "/topology": handle_topology,
    "/enrichment": handle_enrichment,
}


class NexusHandler(BaseHTTPRequestHandler):
    server_version = "NEXUS/0.1"
    allowed_origin = "*"

    def _send(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", self.allowed_origin)
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:          # noqa: N802 (http.server naming)
        self._send(204, {})

    def do_GET(self) -> None:              # noqa: N802
        path = self.path.split("?")[0].rstrip("/") or "/health"
        if path in ("/health", "/"):
            self._send(200, handle_health({}))
        else:
            self._send(405, {"error": f"use POST for {path}"})

    def do_POST(self) -> None:             # noqa: N802
        path = self.path.split("?")[0].rstrip("/")
        handler = ROUTES.get(path)
        if handler is None:
            self._send(404, {"error": f"unknown endpoint {path!r}",
                             "available": sorted(ROUTES)})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": "request body too large"})
                return
            payload = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(payload, dict):
                raise NexusError("request body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, {"error": f"invalid request: {exc}"})
            return

        started = time.time()
        try:
            result = handler(payload)
        except NexusError as exc:
            self._send(400, {"error": str(exc)})
            return
        except Exception as exc:                      # noqa: BLE001
            logger.exception("handler %s failed", path)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return
        result["elapsed_seconds"] = round(time.time() - started, 3)
        self._send(200, result)

    def log_message(self, fmt: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), fmt % args)


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer((host, port), NexusHandler)
    logger.info("NEXUS service listening on http://%s:%d", host, port)
    logger.info("endpoints: %s", ", ".join(sorted(ROUTES)))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NEXUS analysis service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.host, args.port)
