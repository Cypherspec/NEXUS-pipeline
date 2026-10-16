"""Fingerprints, descriptors and scaffolds."""

import numpy as np
import pytest

from nexus.chem.descriptors import (
    compute_descriptors,
    count_structural_alerts,
    fraction_csp3,
    num_aromatic_rings,
    num_hbond_acceptors,
    num_lipinski_hba,
    num_lipinski_hbd,
    num_rotatable_bonds,
    drug_likeness,
    tpsa,
)
from nexus.chem.fingerprint import (
    atom_pair_fingerprint,
    bulk_tanimoto,
    cosine,
    dice,
    fingerprint_matrix,
    maccs_like_keys,
    mix32,
    morgan_fingerprint,
    morgan_identifiers,
    path_fingerprint,
    tanimoto,
    topological_torsion_fingerprint,
)
from nexus.chem.scaffold import (
    murcko_scaffold,
    random_split,
    scaffold_overlap,
    scaffold_split,
)
from nexus.chem.smiles import parse_smiles

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
CAFFEINE = "CN1C=NC2=C1C(=O)N(C)C(=O)N2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"


# ---------------------------------------------------------------- fingerprints

def test_hash_is_stable_and_bounded():
    assert mix32(1, 2, 3) == mix32(1, 2, 3)
    assert mix32(1, 2, 3) != mix32(3, 2, 1)
    assert 0 <= mix32(999, 12345) <= 0xFFFFFFFF


def test_morgan_is_deterministic_across_parses():
    a = morgan_fingerprint(parse_smiles(ASPIRIN))
    b = morgan_fingerprint(parse_smiles(ASPIRIN))
    assert np.array_equal(a, b)


def test_morgan_radius_monotonicity():
    mol = parse_smiles(IBUPROFEN)
    sizes = [len(morgan_identifiers(mol, radius=r)) for r in (0, 1, 2, 3)]
    assert sizes == sorted(sizes)


def test_tanimoto_properties():
    a, b, c = (morgan_fingerprint(parse_smiles(s))
               for s in (ASPIRIN, "CC(=O)Oc1ccccc1C(=O)OC", "CCCCCCCC"))
    assert tanimoto(a, a) == pytest.approx(1.0)
    assert 0.0 <= tanimoto(a, c) < tanimoto(a, b) < 1.0
    assert tanimoto(a, b) == pytest.approx(tanimoto(b, a))


def test_similar_molecules_score_higher_than_dissimilar():
    aspirin = morgan_fingerprint(parse_smiles(ASPIRIN))
    salicylic = morgan_fingerprint(parse_smiles("OC(=O)c1ccccc1O"))
    octane = morgan_fingerprint(parse_smiles("CCCCCCCC"))
    assert tanimoto(aspirin, salicylic) > tanimoto(aspirin, octane)


def test_other_similarity_metrics():
    a = morgan_fingerprint(parse_smiles(ASPIRIN))
    b = morgan_fingerprint(parse_smiles(IBUPROFEN))
    for fn in (dice, cosine):
        assert fn(a, a) == pytest.approx(1.0)
        assert 0.0 <= fn(a, b) <= 1.0


def test_bulk_tanimoto_matches_pairwise():
    mols = [parse_smiles(s) for s in (ASPIRIN, CAFFEINE, IBUPROFEN, "CCO")]
    lib = fingerprint_matrix(mols)
    q = lib[0]
    bulk = bulk_tanimoto(q, lib)
    expected = [tanimoto(q, row) for row in lib]
    np.testing.assert_allclose(bulk, expected, atol=1e-6)


@pytest.mark.parametrize("fn", [morgan_fingerprint, atom_pair_fingerprint,
                                topological_torsion_fingerprint, path_fingerprint])
def test_fingerprints_are_binary_and_nonempty(fn):
    v = fn(parse_smiles(IBUPROFEN))
    assert v.shape == (2048,)
    assert set(np.unique(v)) <= {0, 1}
    assert v.sum() > 0


def test_count_fingerprint_counts_repeats():
    binary = morgan_fingerprint(parse_smiles("CCCCCCCC"), counts=False)
    counted = morgan_fingerprint(parse_smiles("CCCCCCCC"), counts=True)
    assert counted.sum() > binary.sum()


def test_maccs_keys_shape():
    v = maccs_like_keys(parse_smiles(CAFFEINE))
    assert v.shape == (166,) and v.sum() > 0


# ----------------------------------------------------------------- descriptors

# Published TPSA values (Ertl 2000 convention, nitrogen and oxygen only).
@pytest.mark.parametrize("smi,expected", [
    (ASPIRIN, 63.60),
    (CAFFEINE, 58.44),
    (IBUPROFEN, 37.30),
    ("CC(=O)Nc1ccc(O)cc1", 49.33),     # paracetamol
    ("CCO", 20.23),                     # ethanol
    ("c1ccccc1", 0.0),                  # benzene
    ("CC(=O)O", 37.30),                 # acetic acid
])
def test_tpsa_matches_published_values(smi, expected):
    assert tpsa(parse_smiles(smi)) == pytest.approx(expected, abs=0.05)


def test_hydrogen_bond_counts():
    mol = parse_smiles(ASPIRIN)
    assert num_lipinski_hbd(mol) == 1        # one COOH hydrogen
    assert num_lipinski_hba(mol) == 4        # four oxygens
    mol = parse_smiles("NCCO")
    assert num_lipinski_hbd(mol) == 3        # NH2 + OH
    assert num_hbond_acceptors(mol) == 2


def test_pyrrole_nitrogen_is_not_an_acceptor():
    assert num_hbond_acceptors(parse_smiles("c1cc[nH]c1")) == 0
    assert num_hbond_acceptors(parse_smiles("c1ccncc1")) == 1   # pyridine is


def test_amide_nitrogen_is_not_an_acceptor():
    """The amide N is excluded; its carbonyl O still counts."""
    from nexus.chem.descriptors import is_hbond_acceptor

    mol = parse_smiles("CC(=O)NC")
    nitrogen = next(i for i, a in enumerate(mol.atoms) if a.symbol == "N")
    oxygen = next(i for i, a in enumerate(mol.atoms) if a.symbol == "O")
    assert not is_hbond_acceptor(mol, nitrogen)
    assert is_hbond_acceptor(mol, oxygen)
    assert num_hbond_acceptors(mol) == 1


def test_rotatable_bonds():
    assert num_rotatable_bonds(parse_smiles("CCCC")) == 1
    assert num_rotatable_bonds(parse_smiles("c1ccccc1")) == 0
    assert num_rotatable_bonds(parse_smiles("CC(=O)NC")) == 0      # amide excluded
    assert num_rotatable_bonds(parse_smiles("c1ccccc1-c1ccccc1")) == 1


def test_fraction_csp3():
    assert fraction_csp3(parse_smiles("CCCC")) == pytest.approx(1.0)
    assert fraction_csp3(parse_smiles("c1ccccc1")) == pytest.approx(0.0)
    assert fraction_csp3(parse_smiles("Cc1ccccc1")) == pytest.approx(1 / 7)


def test_aromatic_ring_count():
    assert num_aromatic_rings(parse_smiles("c1ccc2ccccc2c1")) == 2
    assert num_aromatic_rings(parse_smiles("C1CCCCC1")) == 0


def test_drug_likeness_is_bounded_and_prefers_drug_like():
    drug = drug_likeness(parse_smiles(IBUPROFEN))
    weird = drug_likeness(parse_smiles("C" * 60))
    assert 0.0 <= drug <= 1.0 and 0.0 <= weird <= 1.0
    assert drug > weird


def test_structural_alerts_detects_nitro_and_epoxide():
    assert count_structural_alerts(parse_smiles("O=[N+]([O-])c1ccccc1")) >= 1
    assert count_structural_alerts(parse_smiles("C1CO1")) >= 1
    assert count_structural_alerts(parse_smiles("CCO")) == 0


def test_descriptor_dictionary_is_finite():
    d = compute_descriptors(parse_smiles(CAFFEINE))
    assert len(d) > 15
    assert all(np.isfinite(v) for v in d.values())


# ------------------------------------------------------------------- scaffolds

def test_murcko_scaffold_shared_by_analogues():
    a = murcko_scaffold(parse_smiles(ASPIRIN))
    b = murcko_scaffold(parse_smiles(IBUPROFEN))
    assert a == b == "c1ccccc1"


def test_acyclic_molecule_has_empty_scaffold():
    assert murcko_scaffold(parse_smiles("CCCCO")) == ""


def test_scaffold_keeps_linker_between_rings():
    scaffold = parse_smiles(murcko_scaffold(parse_smiles("c1ccc(cc1)CCc1ccccc1")))
    assert scaffold.n_rings == 2
    assert scaffold.formula() == "C14H14"      # both rings plus the ethyl linker


def test_generic_scaffold_is_all_carbon():
    s = murcko_scaffold(parse_smiles(CAFFEINE), generic=True)
    assert "n" not in s and "N" not in s and "O" not in s


def test_scaffold_split_has_no_leakage():
    smiles = [ASPIRIN, IBUPROFEN, CAFFEINE, "CCO", "CCCC", "c1ccc2ccccc2c1",
              "OC(=O)c1ccccc1O", "c1ccncc1", "C1CCCCC1", "CC(C)Cc1ccccc1"]
    mols = [parse_smiles(s) for s in smiles * 5]
    train, valid, test = scaffold_split(mols)
    assert len(train) + len(valid) + len(test) == len(mols)
    assert set(train).isdisjoint(test)
    assert scaffold_overlap(train, test, mols) == 0.0


def test_random_split_partitions_indices():
    train, valid, test = random_split(100)
    assert len(train) + len(valid) + len(test) == 100
    assert set(train).isdisjoint(test) and set(train).isdisjoint(valid)
