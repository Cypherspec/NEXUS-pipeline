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

import os
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


_ACTIONS = {
    "analyze": analyze,
    "neighbors": neighbors,
    "screen": screen,
    "predict": predict,
}


def invoke(action: str, payload: dict, ctx: dict):
    """Single dispatcher, so every action shares the same payload guards."""
    handler = _ACTIONS.get(action)
    if handler is None:
        raise ValueError(f"Unknown action: {action}")
    return handler(payload or {}, ctx or {})
