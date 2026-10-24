"""Tests for the ADMET screening rules.

The rule-based endpoints are checked on molecules whose classification is not in
doubt, and the one quantitative endpoint is scored against measured values on
the dataset it was fitted on, so a regression in either shows up here.
"""

from __future__ import annotations

import numpy as np
import pytest

from nexus.chem.admet import (HIGH, LOW, MODERATE, admet_profile, admet_summary,
                              aromatic_proportion, cns_exposure, egan_absorption,
                              esol_log_solubility, gleeson_4_400, herg_liability,
                              num_basic_amines, oral_bioavailability_flags,
                              pfizer_3_75, solubility_class)
from nexus.chem.smiles import parse_smiles

BENZENE = "c1ccccc1"
ETHANOL = "CCO"
ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
DIPHENHYDRAMINE = "CN(C)CCOC(c1ccccc1)c1ccccc1"       # basic amine, lipophilic
CYCLOSPORINE_LIKE = "CC(C)CC(NC(=O)C(C)NC(=O)C(CC(C)C)NC(=O)C(C)N)C(=O)NC(C)C(=O)O"


def mol(smiles: str):
    return parse_smiles(smiles)


def test_aromatic_proportion_bounds():
    assert aromatic_proportion(mol(BENZENE)) == pytest.approx(1.0)
    assert aromatic_proportion(mol(ETHANOL)) == pytest.approx(0.0)
    ap = aromatic_proportion(mol(ASPIRIN))
    assert 0.0 < ap < 1.0


def test_basic_amine_counting_excludes_amides_and_aromatics():
    # A tertiary aliphatic amine is counted; caffeine's nitrogens are amide or
    # aromatic and must not be.
    assert num_basic_amines(mol(DIPHENHYDRAMINE)) == 1
    assert num_basic_amines(mol(CAFFEINE)) == 0
    assert num_basic_amines(mol(BENZENE)) == 0


def test_solubility_ordering_matches_chemical_intuition():
    # Ethanol is miscible; benzene is far less soluble. The model must at least
    # order these correctly, whatever its absolute error.
    assert esol_log_solubility(mol(ETHANOL)) > esol_log_solubility(mol(BENZENE))


def test_solubility_class_bands():
    assert solubility_class(0.0) == "soluble"
    assert solubility_class(-3.0) == "moderately soluble"
    assert solubility_class(-5.0) == "poorly soluble"
    assert solubility_class(-7.0) == "insoluble"


def test_egan_absorption_flags_large_polar_molecules():
    assert egan_absorption(mol(CAFFEINE))
    assert not egan_absorption(mol(CYCLOSPORINE_LIKE))


def test_cns_window_reports_the_failing_properties():
    ok, failures = cns_exposure(mol(CYCLOSPORINE_LIKE))
    assert not ok
    assert failures                       # at least one named reason
    assert all(isinstance(f, str) for f in failures)


def test_herg_bands_need_both_basicity_and_lipophilicity():
    # Diphenhydramine: basic amine, logP around 3.3 -> flagged, not top band.
    assert herg_liability(mol(DIPHENHYDRAMINE)) in (MODERATE, HIGH)
    # Caffeine: no basic centre, low logP -> clean.
    assert herg_liability(mol(CAFFEINE)) == LOW


def test_herg_flags_known_blockers_and_clears_known_safe_drugs():
    """A sanity check on drugs whose hERG behaviour is well documented.

    Terfenadine was withdrawn over QT prolongation, and amitriptyline and
    diphenhydramine are long-established hERG binders; caffeine and aspirin are
    not. Five compounds are far too few to call the rule validated -- they only
    show it is not inverted or stuck on one answer.
    """
    blockers = {
        "terfenadine": "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
        "amitriptyline": "CN(C)CCC=C1c2ccccc2CCc2ccccc21",
        "diphenhydramine": DIPHENHYDRAMINE,
    }
    for name, smiles in blockers.items():
        assert herg_liability(mol(smiles)) == HIGH, name

    for name, smiles in {"caffeine": CAFFEINE, "aspirin": ASPIRIN}.items():
        assert herg_liability(mol(smiles)) == LOW, name


def test_pfizer_and_gleeson_rules():
    assert pfizer_3_75(mol(CAFFEINE)) in (LOW, MODERATE)
    assert gleeson_4_400(mol(CAFFEINE))
    assert not gleeson_4_400(mol(CYCLOSPORINE_LIKE))


def test_bioavailability_flags_are_named():
    assert oral_bioavailability_flags(mol(CAFFEINE)) == []
    assert oral_bioavailability_flags(mol(CYCLOSPORINE_LIKE))


def test_profile_is_serialisable_and_counts_flags():
    profile = admet_profile(mol(CAFFEINE))
    data = profile.as_dict()
    assert set(data) == {
        "logSolubility", "solubilityClass", "absorptionEgan", "cnsCandidate",
        "cnsFailures", "hergRisk", "toxRisk375", "gleesonPass",
        "bioavailabilityFailures", "riskFlags",
    }
    assert 0 <= data["riskFlags"] <= 5
    assert data["hergRisk"] in (LOW, MODERATE, HIGH)


def test_summary_over_a_set():
    mols = [mol(s) for s in (BENZENE, ETHANOL, ASPIRIN, CAFFEINE, DIPHENHYDRAMINE)]
    summary = admet_summary(mols)
    assert summary["n"] == 5
    assert 0.0 <= summary["absorbedFraction"] <= 1.0
    assert 0.0 <= summary["cleanFraction"] <= 1.0
    assert admet_summary([]) == {"n": 0}


def test_solubility_accuracy_on_esol():
    """The quantitative endpoint, scored against measured solubilities.

    The thresholds are the fit's held-out numbers with room to spare, so this
    catches a regression in the coefficients or in any descriptor they consume
    without being brittle to small changes.
    """
    from nexus.data import load_dataset

    mols, Y, _, _ = load_dataset("esol")
    y = Y[:, 0]
    keep = np.isfinite(y)
    mols = [m for m, k in zip(mols, keep) if k]
    y = y[keep]

    pred = np.array([esol_log_solubility(m) for m in mols])
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r = float(np.corrcoef(y, pred)[0, 1])

    assert rmse < 1.5, f"ESOL RMSE regressed to {rmse:.3f}"
    assert r > 0.75, f"ESOL correlation regressed to {r:.3f}"
    # Centred, which the published coefficients are not against this logP.
    assert abs(float(np.mean(pred - y))) < 0.25
