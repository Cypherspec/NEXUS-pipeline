"""Conformers, persistent homology and discrete Ricci curvature."""

import numpy as np
import pytest

from nexus.chem.conformer import (
    bond_length_report,
    distance_bounds,
    embed_conformers,
    embed_molecule,
    ideal_bond_length,
    min_nonbonded_distance,
    planar_groups,
    triangle_smooth,
)
from nexus.chem.smiles import parse_smiles
from nexus.topo.homology import (
    PersistenceDiagram,
    bottleneck_distance,
    diagram_statistics,
    h0_persistence,
    pairwise_distances,
    persistence_image,
    persistence_landscape,
    rips_persistence,
    topological_features,
)
from nexus.topo.ricci import (
    forman_ricci_curvature,
    ollivier_ricci_curvature,
    ricci_flow,
    curvature_features,
    wasserstein_1,
)


def circle_points(n=24, radius=1.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[radius * np.cos(t), radius * np.sin(t), np.zeros(n)]


# ------------------------------------------------------------------ conformers

def test_bounds_respect_triangle_inequality():
    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    lower, upper = distance_bounds(mol)
    assert np.all(lower <= upper + 1e-9)
    n = len(upper)
    for _ in range(50):
        i, j, k = np.random.default_rng(0).integers(0, n, 3)
        assert upper[i, j] <= upper[i, k] + upper[k, j] + 1e-6


def test_triangle_smooth_tightens_bounds():
    lower = np.zeros((3, 3))
    upper = np.array([[0.0, 1.0, 10.0], [1.0, 0.0, 1.0], [10.0, 1.0, 0.0]])
    _, tightened = triangle_smooth(lower, upper)
    assert tightened[0, 2] == pytest.approx(2.0)      # 1 + 1


@pytest.mark.parametrize("smi", ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
                                 "C1CCCCC1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"])
def test_conformer_bond_lengths_are_accurate(smi):
    mol = parse_smiles(smi)
    coords, energy = embed_molecule(mol, seed=0, steps=400)
    assert coords.shape == (mol.n_atoms, 3)
    assert np.isfinite(coords).all()
    report = bond_length_report(mol, coords)
    assert report["max_abs_error"] < 0.05, report


def test_conformer_has_no_atom_clashes():
    mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    coords, _ = embed_molecule(mol, seed=0, steps=400)
    assert min_nonbonded_distance(mol, coords) > 2.0


def test_aromatic_rings_are_planar():
    for smi in ("c1ccccc1", "c1ccc2ccccc2c1", "CC(=O)Oc1ccccc1C(=O)O"):
        mol = parse_smiles(smi)
        coords, _ = embed_molecule(mol, seed=0, steps=500)
        for ring in mol.rings:
            if not all(mol.atoms[i].aromatic for i in ring):
                continue
            pts = coords[list(ring)]
            pts = pts - pts.mean(axis=0)
            out_of_plane = np.linalg.svd(pts, compute_uv=False)[2]
            assert out_of_plane < 0.05, f"{smi} ring is not planar: {out_of_plane}"


def test_cyclohexane_is_not_planar():
    """A saturated ring must be free to pucker into a chair."""
    mol = parse_smiles("C1CCCCC1")
    coords, _ = embed_molecule(mol, seed=0, steps=500)
    pts = coords - coords.mean(axis=0)
    assert np.linalg.svd(pts, compute_uv=False)[2] > 0.1


def test_planar_groups_identifies_aromatics_and_sp2():
    # Acetone's carbonyl carbon has three heavy neighbours, so the centre plus
    # its neighbours is a genuine four-atom planar group.
    groups = planar_groups(parse_smiles("CC(=O)C"))
    assert any(len(g) == 4 for g in groups)
    groups = planar_groups(parse_smiles("c1ccc2ccccc2c1"))
    # Fused aromatic rings share one plane, so they merge into a single group.
    assert any(len(g) == 10 for g in groups)


def test_conformer_ensemble_is_diverse():
    mol = parse_smiles("CCCCCCCC")
    confs = embed_conformers(mol, n_conformers=4, seed=0, steps=200)
    assert len(confs) == 4
    spread = max(np.abs(confs[0] - c).max() for c in confs[1:])
    assert spread > 0.1


def test_single_atom_and_empty_molecules():
    coords, energy = embed_molecule(parse_smiles("[Na+]"), seed=0)
    assert coords.shape == (1, 3) and energy == 0.0


# ---------------------------------------------------------- persistent homology

def test_h0_counts_components():
    rng = np.random.default_rng(0)
    cloud = np.vstack([rng.normal(0, 0.05, (12, 3)),
                       rng.normal(0, 0.05, (12, 3)) + np.array([5.0, 0, 0])])
    diagram = rips_persistence(cloud, max_dim=0)
    assert diagram.betti_at(0, 1.0) == 2
    assert diagram.betti_at(0, 10.0) == 1          # merged at large scale


def test_h0_single_point_and_empty():
    assert len(h0_persistence(np.zeros((1, 1)))) == 1
    assert rips_persistence(np.zeros((0, 3)))[0].shape == (0, 2)


def test_circle_has_exactly_one_loop():
    diagram = rips_persistence(circle_points(20), max_dim=1)
    lifetimes = np.sort(diagram.lifetimes(1))[::-1]
    assert len(lifetimes) >= 1
    assert lifetimes[0] > 1.0
    # One dominant loop: any others are an order of magnitude shorter.
    if len(lifetimes) > 1:
        assert lifetimes[0] > 5 * lifetimes[1]


def test_filled_disc_has_no_persistent_loop():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (40, 2))
    pts = pts[(pts ** 2).sum(1) <= 1]
    cloud = np.c_[pts, np.zeros(len(pts))]
    diagram = rips_persistence(cloud, max_dim=1)
    lifetimes = diagram.lifetimes(1)
    assert len(lifetimes) == 0 or lifetimes.max() < 0.8


def test_larger_circle_has_longer_lived_loop():
    small = rips_persistence(circle_points(20, 1.0), max_dim=1).lifetimes(1).max()
    large = rips_persistence(circle_points(20, 3.0), max_dim=1).lifetimes(1).max()
    assert large > small


def test_persistence_features_are_finite_and_fixed_length():
    a = topological_features(circle_points(16), max_dim=1, resolution=8)
    b = topological_features(circle_points(20), max_dim=1, resolution=8)
    assert a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()


def test_persistence_image_responds_to_the_loop():
    loop = persistence_image(rips_persistence(circle_points(20), 1), dim=1)
    rng = np.random.default_rng(0)
    blob = persistence_image(rips_persistence(rng.normal(0, 0.3, (20, 3)), 1), dim=1)
    assert loop.sum() > blob.sum()


def test_persistence_landscape_shape_and_order():
    diagram = rips_persistence(circle_points(20), max_dim=1)
    land = persistence_landscape(diagram, dim=1, n_layers=3, resolution=16)
    assert land.shape == (48,)
    layers = land.reshape(3, 16)
    assert np.all(layers[0] >= layers[1] - 1e-9)   # layers are ordered


def test_diagram_statistics_keys_and_empty_case():
    diagram = rips_persistence(circle_points(16), max_dim=1)
    stats = diagram_statistics(diagram, 1)
    assert stats["h1_count"] >= 1 and stats["h1_max"] > 0
    empty = diagram_statistics(PersistenceDiagram({1: np.zeros((0, 2))}), 1)
    assert empty["h1_count"] == 0.0


def test_bottleneck_distance_properties():
    a = rips_persistence(circle_points(20, 1.0), max_dim=1)
    b = rips_persistence(circle_points(20, 1.0), max_dim=1)
    c = rips_persistence(circle_points(20, 3.0), max_dim=1)
    assert bottleneck_distance(a, b, 1) == pytest.approx(0.0, abs=1e-9)
    assert bottleneck_distance(a, c, 1) > 0.1
    assert bottleneck_distance(a, c, 1) == pytest.approx(bottleneck_distance(c, a, 1))


def test_pairwise_distances_matches_numpy():
    pts = np.random.default_rng(0).normal(size=(5, 3))
    D = pairwise_distances(pts)
    assert D[1, 3] == pytest.approx(np.linalg.norm(pts[1] - pts[3]))
    np.testing.assert_allclose(D, D.T)


# --------------------------------------------------------------- Ricci curvature

@pytest.mark.parametrize("supply,demand,cost,expected", [
    ([1.0], [1.0], [[0.0]], 0.0),
    ([1.0], [1.0], [[2.0]], 2.0),
    ([0.5, 0.5], [0.5, 0.5], [[0.0, 1.0], [1.0, 0.0]], 0.0),
    ([1.0, 0.0], [0.0, 1.0], [[0.0, 1.0], [1.0, 0.0]], 1.0),
    ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0],
     [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], 2.0),
    ([1.0, 0.0, 0.0], [0.0, 0.5, 0.5],
     [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]], 1.5),
])
def test_wasserstein_exact_values(supply, demand, cost, expected):
    assert wasserstein_1(supply, demand, np.array(cost)) == pytest.approx(expected, abs=1e-9)


def test_cycle_graph_has_zero_ollivier_curvature():
    """Ollivier-Ricci curvature vanishes on C_n for n >= 6."""
    curv = ollivier_ricci_curvature(parse_smiles("c1ccccc1"))
    assert len(curv) == 6
    assert all(abs(v) < 1e-9 for v in curv.values())


def test_curvature_is_bounded():
    for smi in ("CCO", "c1ccccc1", "CC(C)(C)c1ccccc1", "c1ccc2ccccc2c1"):
        for v in ollivier_ricci_curvature(parse_smiles(smi)).values():
            assert -2.0 <= v <= 1.0


def test_branching_lowers_curvature():
    chain = np.mean(list(ollivier_ricci_curvature(parse_smiles("CCCCCC")).values()))
    fused = np.mean(list(ollivier_ricci_curvature(parse_smiles("c1ccc2ccccc2c1")).values()))
    assert chain > fused


def test_forman_curvature_closed_form():
    curv = forman_ricci_curvature(parse_smiles("CCO"))
    # Terminal carbon (degree 1) to middle carbon (degree 2): 4 - 1 - 2 = 1
    assert sorted(curv.values()) == [1.0, 1.0]


def test_ricci_flow_reduces_curvature_spread():
    for smi in ("CC(C)(C)c1ccccc1", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"):
        _, trace = ricci_flow(parse_smiles(smi), iterations=8)
        assert trace[-1] < trace[0]
        assert all(t >= -1e-12 for t in trace)


def test_ricci_flow_keeps_weights_positive():
    weights, _ = ricci_flow(parse_smiles("CC(C)(C)c1ccccc1"), iterations=8)
    assert all(w > 0 for w in weights.values())


def test_curvature_features_fixed_length():
    a = curvature_features(parse_smiles("CCO"))
    b = curvature_features(parse_smiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C"))
    assert a.shape == b.shape == (15,)
    assert np.isfinite(a).all() and np.isfinite(b).all()
