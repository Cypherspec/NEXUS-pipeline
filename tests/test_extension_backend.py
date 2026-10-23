"""The Mantis extension's backend actions.

The backend is loaded the way Mantis loads it -- as a module file, calling
``invoke(action, payload, ctx)`` -- rather than imported as a package, so these
tests exercise the same entry point the host uses.
"""

import random
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "extension" / "backend" / "main.py"

CTX = {"project_id": "test", "user_id": "test",
       "extension_id": "nexus.drug-discovery", "extension_version": "test"}


@pytest.fixture(scope="module")
def backend():
    """Loads the backend module exactly as the host does."""
    spec = importlib.util.spec_from_file_location("nexus_extension_backend", BACKEND)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nexus_extension_backend"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def molecules():
    """A small set with a ring, a chain and a measurable property."""
    return [
        {"id": "a", "smiles": "CCO"},
        {"id": "b", "smiles": "c1ccccc1"},
        {"id": "c", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"id": "d", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        {"id": "e", "smiles": "CCCCCCCC"},
        {"id": "f", "smiles": "c1ccc2ccccc2c1"},
    ]


def test_every_declared_action_is_implemented():
    """The manifest promises a set of actions; the module must provide them."""
    import json

    manifest = json.loads((ROOT / "extension" / "mantis.extension.json").read_text())
    spec = importlib.util.spec_from_file_location("nexus_backend_manifest_check", BACKEND)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    declared = set(manifest["backend"]["actions"])
    assert declared == set(module._ACTIONS), (
        "the manifest's action list and the dispatcher have drifted apart")


def test_unknown_action_is_rejected(backend):
    with pytest.raises(ValueError, match="Unknown action"):
        backend.invoke("no-such-action", {}, CTX)


def test_analyze_returns_a_row_per_molecule(backend, molecules):
    result = backend.invoke("analyze", {"molecules": molecules}, CTX)
    assert result["nAnalyzed"] == len(molecules)
    assert result["nDropped"] == 0
    assert {row["id"] for row in result["molecules"]} == {m["id"] for m in molecules}
    # Every descriptor the package computes crosses the boundary as a number.
    first = result["molecules"][0]
    for key in ("mw", "logp", "tpsa", "drug_likeness", "lipinski_violations"):
        assert isinstance(first[key], (int, float))
    assert result["nScaffolds"] >= 1


def test_analyze_counts_what_it_could_not_read(backend):
    """An unreadable structure is reported, not silently dropped."""
    result = backend.invoke("analyze", {"molecules": [
        {"id": "ok", "smiles": "CCO"},
        {"id": "bad", "smiles": "this is not a molecule"},
        {"id": "empty", "smiles": ""},
    ]}, CTX)
    assert result["nAnalyzed"] == 1
    assert result["nDropped"] == 2


def test_analyze_rejects_a_payload_without_molecules(backend):
    with pytest.raises(ValueError, match="molecules"):
        backend.invoke("analyze", {}, CTX)


def test_neighbours_are_ranked_and_exclude_the_focus(backend, molecules):
    result = backend.invoke("neighbors", {"molecules": molecules,
                                          "focusId": "b", "k": 3}, CTX)
    assert result["focusId"] == "b"
    assert len(result["neighbors"]) == 3
    assert all(n["id"] != "b" for n in result["neighbors"])
    scores = [n["similarity"] for n in result["neighbors"]]
    assert scores == sorted(scores, reverse=True)
    # Benzene's nearest neighbour among these is naphthalene, not an alkane.
    assert result["neighbors"][0]["id"] == "f"


def test_neighbours_reject_an_unknown_focus(backend, molecules):
    with pytest.raises(ValueError, match="focusId"):
        backend.invoke("neighbors", {"molecules": molecules, "focusId": "zz"}, CTX)


def test_screen_cold_starts_without_measurements(backend, molecules):
    result = backend.invoke("screen", {"molecules": molecules, "batchSize": 3}, CTX)
    assert len(result["batch"]) == 3
    assert result["nMeasured"] == 0
    # With too few measurements the backend must say it is ranking by
    # diversity rather than implying a fitted posterior.
    assert "too few" in result["rationale"]


def test_screen_refuses_when_everything_is_measured(backend, molecules):
    measured = {m["id"]: 1.0 for m in molecules}
    with pytest.raises(ValueError, match="already carries a measurement"):
        backend.invoke("screen", {"molecules": molecules, "measured": measured}, CTX)


def test_predict_needs_enough_labels_to_fit_and_calibrate(backend, molecules):
    with pytest.raises(ValueError, match="at least 20"):
        backend.invoke("predict", {"molecules": molecules,
                                   "measured": {"a": 1.0, "b": 2.0}}, CTX)


def test_predict_reports_coverage_and_interval_mode(backend):
    from nexus.chem.smiles import write_smiles
    from nexus.data import load_dataset

    dataset, labels, _, _ = load_dataset("esol")
    rows = [{"id": f"p{i}", "smiles": write_smiles(m)} for i, m in enumerate(dataset[:90])]
    measured = {f"p{i}": float(labels[i][0]) for i in range(60)}

    result = backend.invoke("predict", {"molecules": rows, "measured": measured,
                                        "alpha": 0.1}, CTX)
    assert result["nLabelled"] == 60
    assert result["nominalCoverage"] == pytest.approx(0.9)
    assert result["intervalMode"] == "constant-width"
    assert 0.0 <= result["calibrationCoverage"] <= 1.0
    widths = [r["high"] - r["low"] for r in result["predictions"]]
    # The default score gives one width to every molecule.
    assert max(widths) - min(widths) < 1e-9

    adaptive = backend.invoke("predict", {"molecules": rows, "measured": measured,
                                          "alpha": 0.1, "adaptive": True}, CTX)
    assert adaptive["intervalMode"] == "adaptive"
    adaptive_widths = [r["high"] - r["low"] for r in adaptive["predictions"]]
    assert max(adaptive_widths) - min(adaptive_widths) > 1e-6


def test_conformer_is_centred_and_scaled(backend):
    result = backend.invoke("conformer", {"smiles": "CC(=O)Oc1ccccc1C(=O)O"}, CTX)
    assert result["formula"] == "C9H8O4"
    coords = [(a["x"], a["y"], a["z"]) for a in result["atoms"]]
    assert max(abs(v) for xyz in coords for v in xyz) == pytest.approx(1.0, abs=1e-6)
    assert result["extentAngstrom"] > 0
    assert len(result["bonds"]) >= len(result["atoms"]) - 1


def test_conformer_rejects_an_unparseable_structure(backend):
    with pytest.raises(ValueError, match="parse"):
        backend.invoke("conformer", {"smiles": "not a molecule at all"}, CTX)


def test_space3d_projects_and_reports_variance(backend, molecules):
    result = backend.invoke("space3d", {"molecules": molecules,
                                        "colourBy": "drug_likeness"}, CTX)
    assert len(result["points"]) == len(molecules)
    assert len(result["explainedVariance"]) == 3
    assert all(0.0 <= v <= 1.0 for v in result["explainedVariance"])
    # The projection is normalised into a unit box so the panel can draw it
    # without knowing the chemistry.
    assert max(abs(p[axis]) for p in result["points"] for axis in "xyz") <= 1.0 + 1e-9


def test_space3d_needs_enough_molecules(backend):
    with pytest.raises(ValueError, match="at least four"):
        backend.invoke("space3d", {"molecules": [{"id": "a", "smiles": "CCO"}]}, CTX)


@pytest.mark.parametrize("smiles,name,expected_loops", [
    ("c1ccccc1", "benzene", 1),
    ("CCCCCCCC", "octane", 0),
    ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine", 2),
])
def test_shape_recovers_the_ring_count(backend, smiles, name, expected_loops):
    """Persistence runs on 3D coordinates and never sees the bond graph."""
    result = backend.invoke("shape", {"smiles": smiles}, CTX)
    loops = [f for f in result["features"] if f["dim"] == 1 and f["life"] > 0.35]
    assert len(loops) == expected_loops, f"{name} gave {len(loops)} loops"


def test_shape_curvature_is_exactly_zero_on_a_cycle(backend):
    """A theorem, not a fit: C_n with n >= 6 has curvature exactly zero."""
    result = backend.invoke("shape", {"smiles": "c1ccccc1"}, CTX)
    assert all(bond["ollivier"] == 0.0 for bond in result["bonds"])
    assert result["curvature"]["magnitude"] == 0.0


def test_shape_keeps_loops_when_components_outnumber_them(backend):
    """A large molecule has more components than loops, and the loops matter.

    Capping returned features by lifetime alone drops every loop on a big
    structure, because each of the n-1 components outlives any of them.
    """
    big = ("COc1cc(cc(OC)c1O)C2C3C(COC3=O)C(OC4OC5COC(C)OC5C(O)C4O)"
           "c6cc7c(cc26)OCO7")
    result = backend.invoke("shape", {"smiles": big}, CTX)
    loops = [f for f in result["features"] if f["dim"] == 1]
    assert len(loops) > 0, "every loop was crowded out by H0 features"
    assert result["ringBonds"] > 10


def test_shape_picks_the_most_ring_rich_when_none_is_named(backend):
    result = backend.invoke("shape", {"molecules": [
        {"id": "chain", "smiles": "CCCCCCCC"},
        {"id": "ring", "smiles": "c1ccc2ccccc2c1"},
        {"id": "tiny", "smiles": "CO"},
    ]}, CTX)
    assert result["formula"] == "C10H8", "should have chosen naphthalene"


def test_shape_rejects_a_payload_with_neither_smiles_nor_molecules(backend):
    with pytest.raises(ValueError):
        backend.invoke("shape", {}, CTX)


def test_enrich_recovers_a_planted_process(backend):
    """A study set drawn from one GO term must surface that term."""
    index = backend._go_index()
    term = max(
        (t for t in index["terms"] if 40 <= len(t["gene_set"]) <= 120),
        key=lambda t: len(t["gene_set"]),
    )
    genes = [index["genes"][i] for i in sorted(term["gene_set"])]
    out = backend.invoke("enrich", {"genes": genes}, CTX)

    assert out["nStudyGenes"] == len(genes)
    hit = next((r for r in out["terms"] if r["id"] == term["id"]), None)
    assert hit is not None, f"{term['name']} not recovered"
    assert hit["foldEnrichment"] > 1.0
    assert hit["studyHits"] == len(genes)


def test_enrich_is_calibrated_on_random_gene_sets(backend):
    """Random sets must not come back enriched.

    The earlier implementation filtered tests by hit count before the
    Benjamini-Hochberg step, which selects tests by their own outcome and
    shrinks the correction's denominator. That returned ~18 "significant"
    terms for sets with no biology in them at all.
    """
    index = backend._go_index()
    rng = random.Random(0xC0FFEE)

    counts = []
    for _ in range(8):
        genes = rng.sample(index["genes"], 120)
        counts.append(backend.invoke("enrich", {"genes": genes}, CTX)["nSignificant"])

    mean = sum(counts) / len(counts)
    assert mean < 2.0, f"false positives per random set: {counts}"


def test_enrich_normalises_only_onto_real_genes(backend):
    """A transformation is accepted only when it resolves."""
    lookup = backend._go_index()["gene_lookup"]

    assert backend._resolve_symbol("ABL1", lookup)[1] == "ABL1"
    assert backend._resolve_symbol("ABL1p", lookup)[1] == "ABL1"
    assert backend._resolve_symbol("KIT(D816V)", lookup)[1] == "KIT"
    assert backend._resolve_symbol("CDK4-cyclinD1", lookup)[1] == "CDK4"
    # No stem resolves, so nothing is substituted.
    assert backend._resolve_symbol("NOTAGENE-partner", lookup) == (None, None)
    assert backend._resolve_symbol("", lookup) == (None, None)


def test_enrich_reports_unmatched_rather_than_dropping_them(backend):
    out = backend.invoke("enrich", {"genes": ["EGFR", "QQQNOTAGENE", "ABL1p"]}, CTX)
    assert out["nUnmatched"] == 1
    assert out["unmatched"] == ["QQQNOTAGENE"]
    assert out["nStudyGenes"] == 2
    assert out["nRewritten"] == 1
