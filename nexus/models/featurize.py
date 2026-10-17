"""Convert molecules into the tensors a message-passing network consumes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus.chem.descriptors import is_hbond_acceptor, is_hbond_donor
from nexus.chem.mol import AROMATIC, BOND_ORDER, DOUBLE, Molecule, SINGLE, TRIPLE
from nexus.chem.periodic import atomic_number, electronegativity, vdw_radius

# Elements covered explicitly; anything else falls into a shared "other" slot.
ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
DEGREES = [0, 1, 2, 3, 4, 5]
CHARGES = [-2, -1, 0, 1, 2]
H_COUNTS = [0, 1, 2, 3, 4]
RING_SIZES = [3, 4, 5, 6, 7, 8]
BOND_TYPES = [SINGLE, DOUBLE, TRIPLE, AROMATIC]


def _one_hot(value, choices: list) -> list[float]:
    """One-hot with a trailing catch-all slot for unseen values."""
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def atom_features(mol: Molecule, i: int) -> np.ndarray:
    atom = mol.atoms[i]
    f: list[float] = []
    f += _one_hot(atom.symbol, ATOM_TYPES)
    f += _one_hot(mol.degree(i), DEGREES)
    f += _one_hot(atom.charge, CHARGES)
    f += _one_hot(atom.total_h, H_COUNTS)
    f += [1.0 if atom.aromatic else 0.0]
    f += [1.0 if atom.ring_sizes else 0.0]
    f += [1.0 if size in atom.ring_sizes else 0.0 for size in RING_SIZES]
    f += [1.0 if is_hbond_donor(mol, i) else 0.0]
    f += [1.0 if is_hbond_acceptor(mol, i) else 0.0]
    f += [1.0 if atom.chirality else 0.0]
    # Continuous properties, scaled to a comparable range.
    f += [atomic_number(atom.symbol) / 53.0,
          electronegativity(atom.symbol) / 4.0,
          vdw_radius(atom.symbol) / 2.2,
          mol.explicit_valence(i) / 6.0]
    return np.array(f, dtype=np.float64)


def bond_features(mol: Molecule, bond) -> np.ndarray:
    f: list[float] = []
    f += _one_hot(bond.order, BOND_TYPES)
    f += [1.0 if bond.in_ring else 0.0]
    f += [1.0 if bond.stereo else 0.0]
    f += [BOND_ORDER[bond.order] / 3.0]
    conjugated = bond.order in (DOUBLE, AROMATIC) or (
        bond.order == SINGLE and _adjacent_unsaturated(mol, bond))
    f += [1.0 if conjugated else 0.0]
    return np.array(f, dtype=np.float64)


def _adjacent_unsaturated(mol: Molecule, bond) -> bool:
    for atom_idx in (bond.a, bond.b):
        for j in mol.neighbors(atom_idx):
            other = mol.bond_between(atom_idx, j)
            if other is not bond and other.order in (DOUBLE, TRIPLE, AROMATIC):
                return True
    return False


ATOM_FEATURE_DIM = len(ATOM_TYPES) + 1 + len(DEGREES) + 1 + len(CHARGES) + 1 \
    + len(H_COUNTS) + 1 + 2 + len(RING_SIZES) + 3 + 4
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 4


@dataclass
class MolGraph:
    """A batch of molecules merged into one disconnected graph.

    atom_features : (n_atoms, atom_dim)
    bond_features : (n_directed_bonds, bond_dim)  -- each bond appears twice
    bond_src/bond_dst : (n_directed_bonds,) atom indices
    reverse_bond : index of the opposite directed bond
    atom_to_mol : (n_atoms,) which molecule each atom belongs to
    """
    atom_features: np.ndarray
    bond_features: np.ndarray
    bond_src: np.ndarray
    bond_dst: np.ndarray
    reverse_bond: np.ndarray
    atom_to_mol: np.ndarray
    n_molecules: int

    @property
    def n_atoms(self) -> int:
        return len(self.atom_features)

    @property
    def n_bonds(self) -> int:
        return len(self.bond_features)


def featurize_batch(mols: list[Molecule]) -> MolGraph:
    atom_feats: list[np.ndarray] = []
    bond_feats: list[np.ndarray] = []
    src: list[int] = []
    dst: list[int] = []
    reverse: list[int] = []
    atom_to_mol: list[int] = []
    offset = 0

    for mol_index, mol in enumerate(mols):
        if mol.n_atoms == 0:
            # Keep one zero atom so every molecule has a readout row.
            atom_feats.append(np.zeros(ATOM_FEATURE_DIM))
            atom_to_mol.append(mol_index)
            offset += 1
            continue
        for i in range(mol.n_atoms):
            atom_feats.append(atom_features(mol, i))
            atom_to_mol.append(mol_index)
        for bond in mol.bonds:
            f = bond_features(mol, bond)
            a, b = offset + bond.a, offset + bond.b
            k = len(bond_feats)
            bond_feats.append(f)
            src.append(a)
            dst.append(b)
            bond_feats.append(f)
            src.append(b)
            dst.append(a)
            reverse.append(k + 1)
            reverse.append(k)
        offset += mol.n_atoms

    return MolGraph(
        atom_features=np.array(atom_feats, dtype=np.float64) if atom_feats
        else np.zeros((0, ATOM_FEATURE_DIM)),
        bond_features=np.array(bond_feats, dtype=np.float64) if bond_feats
        else np.zeros((0, BOND_FEATURE_DIM)),
        bond_src=np.array(src, dtype=np.int64),
        bond_dst=np.array(dst, dtype=np.int64),
        reverse_bond=np.array(reverse, dtype=np.int64),
        atom_to_mol=np.array(atom_to_mol, dtype=np.int64),
        n_molecules=len(mols),
    )
