"""Chemistry core: parsing, perception, formulas, round-trips."""

import numpy as np
import pytest

from nexus.chem.mol import AROMATIC, DOUBLE, MoleculeError, SINGLE
from nexus.chem.smiles import parse_smiles, read_smiles, write_smiles

# (SMILES, formula, molecular weight, number of SSSR rings, aromatic atom count)
REFERENCE = [
    ("CCO",                                   "C2H6O",     46.07, 0, 0),   # ethanol
    ("CC(=O)Oc1ccccc1C(=O)O",                 "C9H8O4",   180.16, 1, 6),   # aspirin
    ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C",          "C8H10N4O2",194.19, 2, 5),   # caffeine
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O",            "C13H18O2", 206.28, 1, 6),   # ibuprofen
    ("CC(=O)Nc1ccc(O)cc1",                    "C8H9NO2",  151.16, 1, 6),   # paracetamol
    ("OCC1OC(O)C(O)C(O)C1O",                  "C6H12O6",  180.16, 1, 0),   # glucose
    ("c1ccc2ccccc2c1",                        "C10H8",    128.17, 2, 10),  # naphthalene
    ("C1=CC=CC=C1",                           "C6H6",      78.11, 1, 6),   # Kekule benzene
    ("c1ccncc1",                              "C5H5N",     79.10, 1, 6),   # pyridine
    ("c1cc[nH]c1",                            "C4H5N",     67.09, 1, 5),   # pyrrole
    ("c1ccoc1",                                "C4H4O",     68.07, 1, 5),  # furan
    ("c1ccsc1",                                "C4H4S",     84.14, 1, 5),  # thiophene
    ("C1CCCCC1",                              "C6H12",     84.16, 1, 0),   # cyclohexane
    ("O=S(=O)(N)c1ccccc1",                    "C6H7NO2S", 157.19, 1, 6),   # benzenesulfonamide
    ("[Na+].[Cl-]",                           "ClNa",      58.44, 0, 0),   # salt, 2 components
    ("C1CC2CCC1CC2",                          "C8H14",    110.20, 2, 0),   # bicyclo[2.2.2]octane
    ("N[C@@H](C)C(=O)O",                      "C3H7NO2",   89.09, 0, 0),   # L-alanine
]


@pytest.mark.parametrize("smi,formula,mw,n_rings,n_arom", REFERENCE)
def test_reference_molecules(smi, formula, mw, n_rings, n_arom):
    mol = parse_smiles(smi)
    assert mol.formula() == formula
    assert mol.molecular_weight() == pytest.approx(mw, abs=0.02)
    assert mol.n_rings == n_rings
    assert sum(a.aromatic for a in mol.atoms) == n_arom
    assert mol.is_sane(), mol.valence_errors()


@pytest.mark.parametrize("smi,formula,mw,n_rings,n_arom", REFERENCE)
def test_round_trip_preserves_composition(smi, formula, mw, n_rings, n_arom):
    mol = parse_smiles(smi)
    again = parse_smiles(write_smiles(mol))
    assert again.formula() == formula
    assert again.n_rings == n_rings
    assert again.molecular_weight() == pytest.approx(mw, abs=0.02)


def test_kekule_and_aromatic_benzene_agree():
    a, b = parse_smiles("C1=CC=CC=C1"), parse_smiles("c1ccccc1")
    assert a.formula() == b.formula() == "C6H6"
    assert all(at.aromatic for at in a.atoms)
    assert all(bond.order == AROMATIC for bond in a.bonds)


def test_caffeine_carbonyl_ring_is_not_aromatic():
    """The xanthine six-membered ring is an amide ring, not aromatic."""
    mol = parse_smiles("CN1C=NC2=C1C(=O)N(C)C(=O)N2C")
    five = [r for r in mol.rings if len(r) == 5][0]
    six = [r for r in mol.rings if len(r) == 6][0]
    assert all(mol.atoms[i].aromatic for i in five)
    assert not all(mol.atoms[i].aromatic for i in six)


def test_implicit_hydrogen_counts():
    mol = parse_smiles("CC(=O)O")           # acetic acid
    assert [a.total_h for a in mol.atoms] == [3, 0, 0, 1]
    mol = parse_smiles("c1ccccc1")
    assert all(a.total_h == 1 for a in mol.atoms)
    mol = parse_smiles("c1ccsc1")           # thiophene sulfur carries no H
    assert mol.atoms[3].total_h == 0


def test_charges_isotopes_and_maps():
    mol = parse_smiles("[13CH4]")
    assert mol.atoms[0].isotope == 13 and mol.atoms[0].total_h == 4
    mol = parse_smiles("[NH4+]")
    assert mol.charge() == 1 and mol.formula() == "H4N"
    mol = parse_smiles("[O-]C(=O)C")
    assert mol.charge() == -1
    mol = parse_smiles("[CH3:1][OH:2]")
    assert [a.map_number for a in mol.atoms] == [1, 2]
    mol = parse_smiles("[Fe+2]")
    assert mol.atoms[0].charge == 2


def test_ring_bond_numbering_forms():
    a = parse_smiles("C%10CCCCC%10")
    b = parse_smiles("C1CCCCC1")
    c = parse_smiles("C%(100)CCCCC%(100)")
    assert a.n_rings == b.n_rings == c.n_rings == 1
    assert a.formula() == b.formula() == c.formula() == "C6H12"


def test_ring_perception_counts():
    assert parse_smiles("C1CC1").n_rings == 1
    assert parse_smiles("c1ccc2ccccc2c1").n_rings == 2            # naphthalene
    assert parse_smiles("c1ccc2c(c1)ccc1ccccc12").n_rings == 3    # phenanthrene
    # Cubane: 12 bonds, 8 atoms, 1 component -> cycle rank 5
    assert parse_smiles("C12C3C4C1C5C4C3C25").n_rings == 5


def test_disconnected_components():
    mol = parse_smiles("CCO.CCO.[Na+]")
    assert len(mol.components()) == 3


def test_distance_matrix_is_consistent():
    mol = parse_smiles("CCCC")
    D = mol.distance_matrix()
    assert D[0, 3] == 3 and D[0, 0] == 0
    assert np.array_equal(D, D.T)


def test_valence_sanity_check():
    assert not parse_smiles("C(C)(C)(C)(C)C").is_sane()    # pentavalent carbon
    assert parse_smiles("CS(=O)(=O)O").is_sane()           # sulfonic acid
    assert parse_smiles("O=[N+]([O-])c1ccccc1").is_sane()  # nitrobenzene


def test_wildcard_atom():
    mol = parse_smiles("*CCO")
    assert mol.n_atoms == 4 and mol.formula() == "C2H5O"


@pytest.mark.parametrize("bad", [
    "", "C1CC", "CC)", "C(CC", "[C", "c1ccccc2", "[Qq]", "C%", "[CH4:]", "C1CC11",
])
def test_invalid_smiles_rejected(bad):
    with pytest.raises(MoleculeError):
        parse_smiles(bad)
    assert read_smiles(bad) is None


def test_aromatic_bond_orders_set():
    mol = parse_smiles("c1ccccc1C")
    ring_bonds = [b for b in mol.bonds if b.in_ring]
    assert len(ring_bonds) == 6
    assert all(b.order == AROMATIC for b in ring_bonds)
    exo = [b for b in mol.bonds if not b.in_ring][0]
    assert exo.order == SINGLE
