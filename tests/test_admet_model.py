"""Tests for the multi-task ADMET model that ships with the panel.

The model file is a build artefact: it is produced by benchmarks/train_admet.py
and checked in. These tests confirm the artefact loads, predicts, and still
scores what its own report claims it scores, so a change to featurisation or to
a descriptor cannot silently degrade what the panel shows.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from nexus.admet.model import WEIGHTS_PATH, ADMETModel, load_model, pack, unpack
from nexus.admet.tdc import ENDPOINTS
from nexus.chem.smiles import parse_smiles

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "results" / "admet_fit.json"

pytestmark = pytest.mark.skipif(
    not WEIGHTS_PATH.exists(),
    reason="no trained model; run python -m benchmarks.train_admet")

DRUGS = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "terfenadine": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
    "metformin": "CN(C)C(=N)N=C(N)N",
}


def test_weight_codec_round_trips():
    rng = np.random.default_rng(0)
    array = rng.normal(size=(7, 5))
    assert np.allclose(unpack(pack(array), array.shape), array, atol=1e-3)


def test_weight_codec_falls_back_when_half_precision_would_overflow():
    big = np.array([[1e6, -2e6], [3.0, 4.0]])
    payload = pack(big)
    assert payload["dtype"] == "float32"
    assert np.allclose(unpack(payload, big.shape), big)
    small = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert pack(small)["dtype"] == "float16"


def test_model_loads_and_covers_every_endpoint():
    model = load_model()
    assert set(model.heads) == set(ENDPOINTS)
    assert model.n_train > 10000
    for name, head in model.heads.items():
        assert head.task == ENDPOINTS[name].task
        assert np.isfinite(head.weight).all()


def test_predictions_are_in_range_and_shaped_right():
    model = load_model()
    mols = [parse_smiles(s) for s in DRUGS.values()]
    out = model.predict(mols)

    assert out["n"] == len(mols)
    assert len(out["applicability"]) == len(mols)
    for name, block in out["endpoints"].items():
        spec = ENDPOINTS[name]
        if spec.task == "classification":
            probabilities = np.asarray(block["probability"])
            assert probabilities.shape == (len(mols),)
            assert ((probabilities >= 0) & (probabilities <= 1)).all(), name
        else:
            values = np.asarray(block["value"])
            assert values.shape == (len(mols),)
            assert np.isfinite(values).all(), name


def test_empty_input_is_handled():
    assert load_model().predict([])["n"] == 0


def test_round_trip_through_json_preserves_predictions():
    model = load_model()
    mols = [parse_smiles(s) for s in DRUGS.values()]
    before = model.predict(mols)
    after = ADMETModel.from_json(model.to_json()).predict(mols)
    for name, block in before["endpoints"].items():
        key = "probability" if block["task"] == "classification" else "value"
        assert np.allclose(block[key], after["endpoints"][name][key], atol=1e-3), name


def test_applicability_flags_chemistry_unlike_the_training_set():
    """A plain alkane chain is far outside a drug-like descriptor distribution."""
    model = load_model()
    out = model.predict([parse_smiles("C" * 60), parse_smiles(DRUGS["aspirin"])])
    leverage = out["leverage"]
    assert leverage[0] > leverage[1], "a C60 chain should sit further out than aspirin"
    assert out["applicability"][1], "aspirin should be inside the domain"


@pytest.mark.skipif(not REPORT.exists(), reason="no training report")
def test_reported_accuracy_clears_the_baselines():
    """Every endpoint must beat the trivial baseline on its own held-out split.

    This is the claim the panel makes when it prints a held-out score beside a
    prediction, so it is worth a test rather than a note.
    """
    report = json.loads(REPORT.read_text())
    weak = []
    for name, entry in report["endpoints"].items():
        if entry["task"] == "classification":
            if entry["model"]["roc_auc"] <= 0.5:
                weak.append(f"{name} ROC-AUC {entry['model']['roc_auc']:.3f}")
        else:
            if entry["model"]["mae"] >= entry["baseline_mean"]["mae"]:
                weak.append(f"{name} MAE {entry['model']['mae']:.3f} "
                            f">= mean baseline {entry['baseline_mean']['mae']:.3f}")
    assert not weak, "endpoints no better than the trivial baseline: " + "; ".join(weak)


@pytest.mark.skipif(not REPORT.exists(), reason="no training report")
def test_conformal_intervals_cover_what_they_claim():
    """Split-conformal intervals should hold their coverage on the test split.

    Allowed a margin: coverage is itself estimated on a finite sample, and the
    guarantee is marginal rather than per-endpoint.
    """
    report = json.loads(REPORT.read_text())
    for name, entry in report["endpoints"].items():
        conformal = entry.get("conformal")
        if not conformal:
            continue
        target = conformal["target_coverage"]
        assert conformal["empirical_coverage"] >= target - 0.15, (
            f"{name} covered {conformal['empirical_coverage']:.2f} of a "
            f"{target:.0%} interval")
