"""Tests for structure-activity landscape analysis."""

from __future__ import annotations

import numpy as np
import pytest

from nexus.chem.smiles import parse_smiles
from nexus.space.landscape import (activity_cliffs, diversity, duplicate_values,
                                   landscape_roughness, region_enrichment,
                                   similarity_matrix)

# A homologous alcohol series. Binary Morgan fingerprints cannot separate these
# -- every CH2 environment is identical -- which is why the module counts.
ALCOHOLS = ["CCO", "CCCO", "CCCCO", "CCCCCCO", "CCCCCCCCCCCCCCCCO"]
AROMATICS = ["c1ccccc1", "Cc1ccccc1", "c1ccc2ccccc2c1", "c1ccc2cc3ccccc3cc2c1"]


def mols(smiles):
    return [parse_smiles(s) for s in smiles]


def test_homologues_are_not_identical_under_count_fingerprints():
    """The bug this module exists to avoid: a saturating similarity of 1.0."""
    S = similarity_matrix(mols(ALCOHOLS))
    ethanol, hexadecanol = 0, len(ALCOHOLS) - 1
    assert S[ethanol, hexadecanol] < 0.95
    # And the ordering is sane: ethanol is closer to propanol than to C16.
    assert S[ethanol, 1] > S[ethanol, hexadecanol]


def test_similarity_matrix_is_a_similarity_matrix():
    S = similarity_matrix(mols(ALCOHOLS + AROMATICS))
    assert np.allclose(np.diag(S), 1.0)
    assert np.allclose(S, S.T)
    assert S.min() >= 0.0 and S.max() <= 1.0


def test_cliffs_rank_small_change_large_effect_first():
    # Benzene -> naphthalene -> anthracene: each ring costs solubility.
    smiles = AROMATICS
    values = np.array([-1.6, -2.7, -3.6, -6.4])
    cliffs = activity_cliffs(mols(smiles), values, similarity_min=0.3, top=5)
    assert cliffs
    # SALI must be delta over structural distance, and sorted descending.
    salis = [c.sali for c in cliffs]
    assert salis == sorted(salis, reverse=True)
    for c in cliffs:
        assert c.similarity < 1.0
        assert c.sali == pytest.approx(c.delta / (1.0 - c.similarity))


def test_cliffs_exclude_identical_structures():
    pair = mols(["CCO", "CCO"])
    values = np.array([0.0, 5.0])
    assert activity_cliffs(pair, values, similarity_min=0.3) == []
    duplicates = duplicate_values(pair, values)
    assert len(duplicates) == 1
    assert duplicates[0][2] == pytest.approx(5.0)


def test_roughness_is_higher_for_a_smooth_property():
    """A property that tracks structure should score above one that does not."""
    smiles = ALCOHOLS + AROMATICS
    group = mols(smiles)
    S = similarity_matrix(group)
    # Chain length: varies smoothly with structure.
    smooth = np.array([float(m.heavy_atom_count()) for m in group])
    rng = np.random.default_rng(0)
    noise = rng.normal(size=len(group)) * smooth.std()
    assert (landscape_roughness(group, smooth, sim=S)["correlation"]
            > landscape_roughness(group, noise, sim=S)["correlation"])


def test_region_enrichment_finds_the_feature_that_defines_a_region():
    # Six halogenated compounds and six clean ones; the region is the halogens.
    halogenated = ["CCCl", "CCCCl", "CCCCCl", "c1ccccc1Cl", "CC(Cl)Cl", "CCBr"]
    clean = ["CCO", "CCCO", "CCCCO", "c1ccccc1", "CCN", "CCC"]
    group = mols(halogenated + clean)
    region = np.array([True] * 6 + [False] * 6)
    results = region_enrichment(group, region, min_count=2)
    assert results
    top = results[0]
    assert top.feature == "halogen"
    assert top.fold > 1.0
    assert top.q_value <= 0.05


def test_region_enrichment_handles_degenerate_masks():
    group = mols(ALCOHOLS)
    assert region_enrichment(group, np.zeros(len(group), dtype=bool)) == []
    assert region_enrichment(group, np.ones(len(group), dtype=bool)) == []


def test_diversity_reports_redundancy():
    # The same molecule five times is maximally redundant.
    same = mols(["CCO"] * 5)
    assert diversity(same)["redundantFraction"] == pytest.approx(1.0)
    varied = diversity(mols(AROMATICS + ["CCO", "CCN", "CC(=O)O"]))
    assert varied["redundantFraction"] < 1.0
    assert 0.0 <= varied["meanNearestNeighbour"] <= 1.0


def test_landscape_on_a_real_dataset():
    """ESOL: the least soluble quartile should be enriched for halogens."""
    from nexus.data import load_dataset

    group, Y, _, _ = load_dataset("esol")
    group, y = group[:300], Y[:300, 0]
    S = similarity_matrix(group)

    rough = landscape_roughness(group, y, sim=S)
    assert rough["nPairs"] == 300 * 299 // 2
    assert np.isfinite(rough["correlation"])

    results = region_enrichment(group, y < np.quantile(y, 0.25))
    assert results, "no feature distinguished the least soluble quartile"
    assert any(r.feature == "halogen" for r in results[:5])
