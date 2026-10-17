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

    balanced=True (default) puts scaffold groups larger than half the
    validation set into training, then shuffles the remaining small groups
    before filling train/valid/test. This is the "scaffold balanced" scheme of
    Yang et al. (2019). The deterministic largest-first alternative
    (balanced=False) is reproducible but can leave the held-out sets
    single-class on datasets such as BACE and BBBP, where activity is strongly
    scaffold-correlated, which makes ROC-AUC undefined.
    """
    groups = scaffold_groups(mols, generic=generic)
    sets = list(groups.values())
    n = len(mols)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)

    if balanced:
        cutoff = n_valid / 2
        # Largest first within the big sets, so the one group too large for any
        # held-out subset lands in training rather than inflating test.
        big = sorted((g for g in sets if len(g) > cutoff), key=len, reverse=True)
        small = [g for g in sets if len(g) <= cutoff]
        rng = np.random.default_rng(seed)
        rng.shuffle(small)
        sets = big + small
    else:
        sets.sort(key=lambda g: (-len(g), g[0]))

    train: list[list[int]] = []
    valid: list[list[int]] = []
    test: list[list[int]] = []
    size = {id(train): 0, id(valid): 0, id(test): 0}
    for group in sets:
        if size[id(train)] + len(group) <= n_train:
            target = train
        elif size[id(valid)] + len(group) <= n_valid:
            target = valid
        else:
            target = test
        target.append(group)
        size[id(target)] += len(group)

    # Greedy assignment can starve the validation set: a group larger than the
    # validation target goes to test, and every later group still fits in
    # train, so nothing is ever left over for valid. FreeSolv does exactly this
    # -- its acyclic group is half the dataset. Rebalance by moving whole
    # groups, which keeps every scaffold inside a single subset.
    if size[id(valid)] < n_valid:
        movable = sorted((g for g in train if len(g) <= n_valid - size[id(valid)]),
                         key=len, reverse=True)
        for group in movable:
            if size[id(valid)] >= n_valid:
                break
            if len(group) > n_valid - size[id(valid)]:
                continue
            train.remove(group)
            valid.append(group)
            size[id(train)] -= len(group)
            size[id(valid)] += len(group)

    flat = lambda parts: np.array(sorted(i for group in parts for i in group), dtype=int)
    return flat(train), flat(valid), flat(test)


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
