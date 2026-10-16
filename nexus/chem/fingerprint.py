"""Molecular fingerprints: Morgan/ECFP, atom-pair, topological torsion, RDKit-style
path fingerprints, and similarity metrics. Deterministic across runs and machines.
"""

from __future__ import annotations

import numpy as np

from nexus.chem.mol import BOND_ORDER, Molecule
from nexus.chem.periodic import atomic_number

MASK32 = 0xFFFFFFFF


def mix32(*values: int) -> int:
    """Stable 32-bit hash (FNV-1a over the little-endian words of each value).

    Python's built-in hash() is salted per process, so fingerprints must not use
    it: identical molecules would hash differently across runs.
    """
    h = 0x811C9DC5
    for v in values:
        v &= MASK32
        for shift in (0, 8, 16, 24):
            h ^= (v >> shift) & 0xFF
            h = (h * 0x01000193) & MASK32
    return h


def atom_invariant(mol: Molecule, i: int, use_chirality: bool = False) -> int:
    """Daylight-style initial atom identifier for Morgan fingerprints."""
    atom = mol.atoms[i]
    heavy_degree = mol.degree(i)
    total_valence = int(round(mol.explicit_valence(i)))
    parts = (
        atomic_number(atom.symbol),
        heavy_degree,
        total_valence - atom.total_h,
        atom.charge & MASK32,
        atom.total_h,
        1 if atom.ring_sizes else 0,
        1 if atom.aromatic else 0,
    )
    if use_chirality:
        parts = parts + ({"": 0, "@": 1, "@@": 2}.get(atom.chirality, 0),)
    return mix32(*parts)


def morgan_identifiers(
    mol: Molecule, radius: int = 2, use_chirality: bool = False,
    use_features: bool = False,
) -> dict[int, list[tuple[int, int]]]:
    """Morgan (ECFP/FCFP) substructure identifiers.

    Returns {identifier: [(center atom, radius), ...]}, which supports both
    bit-vector folding and interpretation of which substructure set a bit.
    """
    if use_features:
        current = [mix32(*feature_invariant(mol, i)) for i in range(mol.n_atoms)]
    else:
        current = [atom_invariant(mol, i, use_chirality) for i in range(mol.n_atoms)]

    found: dict[int, list[tuple[int, int]]] = {}
    for i, ident in enumerate(current):
        found.setdefault(ident, []).append((i, 0))

    # Atom environments seen so far, to suppress duplicate identifiers arising
    # from symmetric neighbourhoods (ECFP duplicate removal).
    seen_envs: set[frozenset[int]] = set()
    env: list[set[int]] = [{i} for i in range(mol.n_atoms)]

    for layer in range(1, radius + 1):
        nxt = list(current)
        new_env = [set(e) for e in env]
        for i in range(mol.n_atoms):
            neighbour_terms = []
            for j in mol.neighbors(i):
                bond = mol.bond_between(i, j)
                neighbour_terms.append((int(BOND_ORDER[bond.order] * 2), current[j]))
                new_env[i] |= env[j]
            neighbour_terms.sort()
            flat: list[int] = [current[i], layer]
            for order, ident in neighbour_terms:
                flat.extend((order, ident))
            nxt[i] = mix32(*flat)
        current, env = nxt, new_env
        for i, ident in enumerate(current):
            key = frozenset(env[i])
            if key in seen_envs:
                continue
            seen_envs.add(key)
            found.setdefault(ident, []).append((i, layer))
    return found


def feature_invariant(mol: Molecule, i: int) -> tuple[int, ...]:
    """Pharmacophoric (FCFP) atom typing: donor, acceptor, aromatic, halogen,
    basic, acidic."""
    from nexus.chem.descriptors import is_hbond_acceptor, is_hbond_donor

    atom = mol.atoms[i]
    return (
        1 if is_hbond_donor(mol, i) else 0,
        1 if is_hbond_acceptor(mol, i) else 0,
        1 if atom.aromatic else 0,
        1 if atom.symbol in ("F", "Cl", "Br", "I") else 0,
        1 if (atom.symbol == "N" and not atom.aromatic) else 0,
        1 if (atom.symbol == "O" and atom.total_h > 0) else 0,
        1 if atom.ring_sizes else 0,
    )


def morgan_fingerprint(
    mol: Molecule, radius: int = 2, n_bits: int = 2048,
    use_chirality: bool = False, use_features: bool = False,
    counts: bool = False,
) -> np.ndarray:
    """Folded Morgan fingerprint (ECFP4 by default: radius 2, 2048 bits)."""
    identifiers = morgan_identifiers(mol, radius, use_chirality, use_features)
    vec = np.zeros(n_bits, dtype=np.int32 if counts else np.uint8)
    for ident, occurrences in identifiers.items():
        bit = ident % n_bits
        if counts:
            vec[bit] += len(occurrences)
        else:
            vec[bit] = 1
    return vec


def atom_pair_fingerprint(mol: Molecule, n_bits: int = 2048,
                          max_distance: int = 30, counts: bool = False) -> np.ndarray:
    """Carhart atom pairs: (type_i, topological distance, type_j)."""
    D = mol.distance_matrix()
    vec = np.zeros(n_bits, dtype=np.int32 if counts else np.uint8)
    types = [_pair_atom_type(mol, i) for i in range(mol.n_atoms)]
    for i in range(mol.n_atoms):
        for j in range(i + 1, mol.n_atoms):
            d = int(D[i, j])
            if d < 1 or d > max_distance:
                continue
            a, b = sorted((types[i], types[j]))
            bit = mix32(a, d, b) % n_bits
            if counts:
                vec[bit] += 1
            else:
                vec[bit] = 1
    return vec


def _pair_atom_type(mol: Molecule, i: int) -> int:
    atom = mol.atoms[i]
    pi = sum(1 for j in mol.neighbors(i)
             if BOND_ORDER[mol.bond_between(i, j).order] > 1)
    return mix32(atomic_number(atom.symbol), mol.degree(i), pi, 1 if atom.aromatic else 0)


def topological_torsion_fingerprint(mol: Molecule, n_bits: int = 2048,
                                    counts: bool = False) -> np.ndarray:
    """Nilakantan topological torsions: linear paths of four bonded atoms."""
    vec = np.zeros(n_bits, dtype=np.int32 if counts else np.uint8)
    types = [_pair_atom_type(mol, i) for i in range(mol.n_atoms)]
    for b in mol.bonds:
        for i in mol.neighbors(b.a):
            if i == b.b:
                continue
            for l in mol.neighbors(b.b):
                if l in (b.a, i):
                    continue
                path = (types[i], types[b.a], types[b.b], types[l])
                key = min(path, path[::-1])
                bit = mix32(*key) % n_bits
                if counts:
                    vec[bit] += 1
                else:
                    vec[bit] = 1
    return vec


def path_fingerprint(mol: Molecule, n_bits: int = 2048, min_path: int = 1,
                     max_path: int = 7) -> np.ndarray:
    """RDKit-style hashed linear path fingerprint."""
    vec = np.zeros(n_bits, dtype=np.uint8)
    types = [_pair_atom_type(mol, i) for i in range(mol.n_atoms)]

    def walk(path: list[int], used: set[tuple[int, int]]) -> None:
        if len(path) - 1 >= min_path:
            fwd = []
            for k in range(len(path)):
                fwd.append(types[path[k]])
                if k + 1 < len(path):
                    bond = mol.bond_between(path[k], path[k + 1])
                    fwd.append(int(BOND_ORDER[bond.order] * 2))
            key = min(tuple(fwd), tuple(fwd[::-1]))
            vec[mix32(*key) % n_bits] = 1
        if len(path) - 1 >= max_path:
            return
        last = path[-1]
        for j in mol.neighbors(last):
            edge = (min(last, j), max(last, j))
            if edge in used or j in path:
                continue
            walk(path + [j], used | {edge})

    for start in range(mol.n_atoms):
        walk([start], set())
    return vec


def maccs_like_keys(mol: Molecule) -> np.ndarray:
    """166 interpretable structural keys (element counts, ring systems, common
    functional groups). Independently defined here, not the MDL key set."""
    from nexus.chem.descriptors import count_functional_groups

    counts = count_functional_groups(mol)
    keys = []
    for sym in ("B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"):
        n = sum(1 for a in mol.atoms if a.symbol == sym)
        keys += [n >= 1, n >= 2, n >= 4, n >= 8]
    for size in range(3, 9):
        n = sum(1 for r in mol.rings if len(r) == size)
        keys += [n >= 1, n >= 2]
    n_arom = sum(1 for r in mol.rings if all(mol.atoms[i].aromatic for i in r))
    keys += [n_arom >= 1, n_arom >= 2, n_arom >= 3, n_arom >= 4]
    for name in sorted(counts):
        n = counts[name]
        keys += [n >= 1, n >= 2]
    keys += [mol.charge() > 0, mol.charge() < 0, len(mol.components()) > 1]
    vec = np.zeros(166, dtype=np.uint8)
    vec[:min(166, len(keys))] = np.array(keys[:166], dtype=np.uint8)
    return vec


# ------------------------------------------------------------------ similarity

def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """Tanimoto/Jaccard similarity; works for binary and count vectors."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    if a.max() <= 1 and b.max() <= 1:
        inter = float(np.sum(np.minimum(a, b)))
        union = float(np.sum(np.maximum(a, b)))
    else:
        inter = float(np.sum(np.minimum(a, b)))
        union = float(a.sum() + b.sum() - inter)
    return inter / union if union > 0 else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = a.sum() + b.sum()
    return float(2 * np.sum(np.minimum(a, b)) / denom) if denom > 0 else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def bulk_tanimoto(query: np.ndarray, library: np.ndarray) -> np.ndarray:
    """Tanimoto of one binary fingerprint against a (n, n_bits) library."""
    q = query.astype(np.float32)
    L = library.astype(np.float32)
    inter = L @ q
    union = L.sum(axis=1) + q.sum() - inter
    out = np.zeros(len(L), dtype=np.float32)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def fingerprint_matrix(mols, kind: str = "morgan", **kwargs) -> np.ndarray:
    """Stack fingerprints for a list of molecules into an (n, n_bits) array."""
    fn = {
        "morgan": morgan_fingerprint,
        "fcfp": lambda m, **kw: morgan_fingerprint(m, use_features=True, **kw),
        "atom_pair": atom_pair_fingerprint,
        "torsion": topological_torsion_fingerprint,
        "path": path_fingerprint,
        "maccs": lambda m, **kw: maccs_like_keys(m),
    }[kind]
    return np.stack([fn(m, **kwargs) for m in mols])
