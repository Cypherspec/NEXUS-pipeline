"""Bemis-Murcko scaffolds and scaffold-based dataset splitting.

Scaffold splits are the standard way to measure whether a model generalises to
new chemical series rather than memorising close analogues.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from nexus.chem.mol import Molecule
from nexus.chem.smiles import write_smiles


def murcko_scaffold(mol: Molecule, generic: bool = False) -> str:
    """Bemis-Murcko scaffold as SMILES: ring systems plus their linkers.

    Iteratively strips terminal atoms that are not in a ring, which leaves the
    ring systems and the chains connecting them. `generic=True` additionally
    replaces every atom with carbon and every bond with a single bond, giving
    the scaffold topology ("graph framework").
    """
    keep = {i for i in range(mol.n_atoms) if mol.atoms[i].ring_sizes}
    if not keep:
        return ""
    # Grow back the linker paths between ring systems.
    keep |= _linker_atoms(mol, keep)
    # Keep exocyclic double-bond partners (e.g. ring C=O), as Bemis-Murcko does.
    for bond in mol.bonds:
        if bond.order == 2:
            if bond.a in keep and bond.b not in keep and not mol.atoms[bond.b].ring_sizes:
                keep.add(bond.b)
            elif bond.b in keep and bond.a not in keep and not mol.atoms[bond.a].ring_sizes:
                keep.add(bond.a)

    sub = subgraph(mol, sorted(keep))
    if generic:
        for atom in sub.atoms:
            atom.symbol = "C"
            atom.aromatic = False
            atom.charge = 0
            atom.isotope = 0
            atom.explicit_h = 0
            atom.chirality = ""
            atom.in_bracket = False
        for bond in sub.bonds:
            bond.order = 1
        sub.perceive()
    return write_smiles(sub)


def _linker_atoms(mol: Molecule, ring_atoms: set[int]) -> set[int]:
    """Atoms on shortest paths between distinct ring systems."""
    systems = [s for s in mol.ring_systems()]
    if len(systems) < 2:
        return set()
    D = mol.distance_matrix()
    linkers: set[int] = set()
    for a in range(len(systems)):
        for b in range(a + 1, len(systems)):
            best = None
            for i in systems[a]:
                for j in systems[b]:
                    d = D[i, j]
                    if d > 0 and (best is None or d < best[0]):
                        best = (d, i, j)
            if best is None:
                continue
            _, src, dst = best
            path = _shortest_path(mol, src, dst)
            linkers |= set(path)
    return linkers - ring_atoms


def _shortest_path(mol: Molecule, src: int, dst: int) -> list[int]:
    prev = {src: -1}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in mol.neighbors(u):
                if v in prev:
                    continue
                prev[v] = u
                if v == dst:
                    path, cur = [], v
                    while cur != -1:
                        path.append(cur)
                        cur = prev[cur]
                    return path
                nxt.append(v)
        frontier = nxt
    return []


def subgraph(mol: Molecule, atom_indices: list[int]) -> Molecule:
    """Induced subgraph on the given atoms, as a new perceived Molecule."""
    import copy as _copy

    index_map = {old: new for new, old in enumerate(atom_indices)}
    sub = Molecule(
        atoms=[_copy.replace(mol.atoms[i]) for i in atom_indices],
        bonds=[],
    )
    for bond in mol.bonds:
        if bond.a in index_map and bond.b in index_map:
            sub.bonds.append(_copy.replace(bond, a=index_map[bond.a], b=index_map[bond.b]))
    # Atoms lose substituents, so hydrogen counts must be recomputed.
    for atom in sub.atoms:
        if not atom.in_bracket:
            atom.explicit_h = 0
            atom.implicit_h = 0
    return sub.perceive()


def scaffold_groups(mols, generic: bool = False) -> dict[str, list[int]]:
    """Map scaffold SMILES to the indices of molecules sharing it."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, mol in enumerate(mols):
        try:
            key = murcko_scaffold(mol, generic=generic)
        except Exception:
            key = ""
        groups[key].append(i)
    return dict(groups)


def scaffold_split(
    mols, frac_train: float = 0.8, frac_valid: float = 0.1,
    generic: bool = False, balanced: bool = True, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices so that no scaffold appears in more than one subset.

    `balanced=True` assigns the largest scaffold groups to the training set
    first (the deterministic MoleculeNet convention); otherwise groups are
    shuffled before assignment.
    """
    groups = scaffold_groups(mols, generic=generic)
    sets = list(groups.values())
    if balanced:
        sets.sort(key=lambda g: (-len(g), g[0]))
    else:
        rng = np.random.default_rng(seed)
        rng.shuffle(sets)

    n = len(mols)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    train, valid, test = [], [], []
    for group in sets:
        if len(train) + len(group) <= n_train:
            train += group
        elif len(valid) + len(group) <= n_valid:
            valid += group
        else:
            test += group
    return np.array(sorted(train)), np.array(sorted(valid)), np.array(sorted(test))


def random_split(n: int, frac_train: float = 0.8, frac_valid: float = 0.1,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    return (np.sort(idx[:n_train]), np.sort(idx[n_train:n_train + n_valid]),
            np.sort(idx[n_train + n_valid:]))


def scaffold_overlap(train_idx, test_idx, mols, generic: bool = False) -> float:
    """Fraction of test molecules whose scaffold also occurs in train."""
    train_scaffolds = {murcko_scaffold(mols[i], generic) for i in train_idx}
    if not len(test_idx):
        return 0.0
    hits = sum(1 for i in test_idx if murcko_scaffold(mols[i], generic) in train_scaffolds)
    return hits / len(test_idx)
