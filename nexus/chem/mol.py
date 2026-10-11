"""Molecular graph: atoms, bonds, ring perception, aromaticity, valence.

Pure Python/NumPy. A Molecule is an immutable-by-convention graph built by the
SMILES parser (nexus.chem.smiles); downstream code reads its arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nexus.chem.periodic import (
    ELEMENTS,
    WILDCARD,
    atomic_mass,
    atomic_number,
    default_valence,
    max_valence,
)

SINGLE, DOUBLE, TRIPLE, QUADRUPLE, AROMATIC = 1, 2, 3, 4, 5
BOND_ORDER = {SINGLE: 1.0, DOUBLE: 2.0, TRIPLE: 3.0, QUADRUPLE: 4.0, AROMATIC: 1.5}
BOND_SYMBOL = {SINGLE: "-", DOUBLE: "=", TRIPLE: "#", QUADRUPLE: "$", AROMATIC: ":"}


class MoleculeError(ValueError):
    """Raised for chemically invalid or unparseable input."""


@dataclass
class Atom:
    symbol: str
    aromatic: bool = False
    charge: int = 0
    isotope: int = 0
    explicit_h: int = 0        # hydrogens written in brackets, e.g. [nH]
    chirality: str = ""        # "", "@" or "@@"
    map_number: int = 0
    in_bracket: bool = False
    # Filled in by Molecule.perceive()
    implicit_h: int = 0
    pi_electrons: int = 0
    ring_sizes: tuple[int, ...] = ()

    @property
    def total_h(self) -> int:
        return self.explicit_h + self.implicit_h


@dataclass
class Bond:
    a: int
    b: int
    order: int = SINGLE
    stereo: str = ""           # "", "/" or "\\"
    in_ring: bool = False

    def other(self, i: int) -> int:
        return self.b if i == self.a else self.a


@dataclass
class Molecule:
    atoms: list[Atom] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    name: str = ""
    smiles: str = ""
    coords: np.ndarray | None = None       # (n_atoms, 3) if embedded
    rings: list[tuple[int, ...]] = field(default_factory=list)   # SSSR

    # ---------------------------------------------------------------- topology

    def __len__(self) -> int:
        return len(self.atoms)

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    def neighbors(self, i: int) -> list[int]:
        return self._adj[i]

    def bond_between(self, i: int, j: int) -> Bond | None:
        return self._bond_index.get((min(i, j), max(i, j)))

    def build_index(self) -> "Molecule":
        self._adj = [[] for _ in self.atoms]
        self._bond_index = {}
        for bond in self.bonds:
            self._adj[bond.a].append(bond.b)
            self._adj[bond.b].append(bond.a)
            self._bond_index[(min(bond.a, bond.b), max(bond.a, bond.b))] = bond
        return self

    def degree(self, i: int) -> int:
        """Heavy-atom degree (sigma bonds to non-hydrogen atoms)."""
        return len(self._adj[i])

    def sigma_degree(self, i: int) -> int:
        """Total sigma connections including hydrogens."""
        return len(self._adj[i]) + self.atoms[i].total_h

    def adjacency_matrix(self, weighted: bool = False) -> np.ndarray:
        n = self.n_atoms
        A = np.zeros((n, n))
        for bond in self.bonds:
            w = BOND_ORDER[bond.order] if weighted else 1.0
            A[bond.a, bond.b] = A[bond.b, bond.a] = w
        return A

    def distance_matrix(self) -> np.ndarray:
        """All-pairs topological (bond-count) distances; unreachable = -1."""
        n = self.n_atoms
        D = np.full((n, n), -1, dtype=np.int32)
        for src in range(n):
            D[src, src] = 0
            frontier = [src]
            d = 0
            while frontier:
                d += 1
                nxt = []
                for u in frontier:
                    for v in self._adj[u]:
                        if D[src, v] < 0:
                            D[src, v] = d
                            nxt.append(v)
                frontier = nxt
        return D

    def components(self) -> list[list[int]]:
        seen = [False] * self.n_atoms
        out = []
        for s in range(self.n_atoms):
            if seen[s]:
                continue
            comp, stack = [], [s]
            seen[s] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in self._adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            out.append(sorted(comp))
        return out

    # ------------------------------------------------------------------- rings

    def perceive_rings(self) -> list[tuple[int, ...]]:
        """Smallest set of smallest rings (Horton candidates + GF(2) independence)."""
        n_comp = len(self.components())
        cycle_rank = self.n_bonds - self.n_atoms + n_comp
        if cycle_rank <= 0:
            self.rings = []
            return self.rings

        bond_id = {(min(b.a, b.b), max(b.a, b.b)): k for k, b in enumerate(self.bonds)}

        def ring_vector(cycle: list[int]) -> int:
            v = 0
            for idx in range(len(cycle)):
                u, w = cycle[idx], cycle[(idx + 1) % len(cycle)]
                v |= 1 << bond_id[(min(u, w), max(u, w))]
            return v

        # Candidate rings: for each bond, the shortest cycle through it.
        candidates = []
        for bond in self.bonds:
            cycle = self._shortest_cycle_through(bond.a, bond.b)
            if cycle:
                candidates.append(cycle)
        # Plus shortest cycle through each atom, which catches rings no single
        # bond selects uniquely.
        for atom_i in range(self.n_atoms):
            for u in self._adj[atom_i]:
                cycle = self._shortest_cycle_through(atom_i, u)
                if cycle:
                    candidates.append(cycle)

        candidates.sort(key=len)
        chosen: list[tuple[int, ...]] = []
        basis: list[int] = []
        seen_sets = set()
        for cycle in candidates:
            key = tuple(sorted(cycle))
            if key in seen_sets:
                continue
            vec = ring_vector(cycle)
            reduced = vec
            for b in basis:
                reduced = min(reduced, reduced ^ b)
            if reduced == 0:
                continue
            basis.append(reduced)
            basis.sort(reverse=True)
            chosen.append(tuple(cycle))
            seen_sets.add(key)
            if len(chosen) == cycle_rank:
                break

        self.rings = chosen
        ring_atoms: dict[int, list[int]] = {}
        for ring in chosen:
            for a in ring:
                ring_atoms.setdefault(a, []).append(len(ring))
        for i, atom in enumerate(self.atoms):
            atom.ring_sizes = tuple(sorted(ring_atoms.get(i, [])))
        in_ring_bonds = set()
        for ring in chosen:
            for idx in range(len(ring)):
                u, w = ring[idx], ring[(idx + 1) % len(ring)]
                in_ring_bonds.add((min(u, w), max(u, w)))
        for bond in self.bonds:
            bond.in_ring = (min(bond.a, bond.b), max(bond.a, bond.b)) in in_ring_bonds
        return self.rings

    def _shortest_cycle_through(self, a: int, b: int) -> list[int] | None:
        """Shortest cycle containing edge (a, b), found by BFS from b avoiding it."""
        prev = {b: -1}
        frontier = [b]
        while frontier:
            nxt = []
            for u in frontier:
                for v in self._adj[u]:
                    if (u == b and v == a) or (u == a and v == b):
                        continue
                    if v in prev:
                        continue
                    prev[v] = u
                    if v == a:
                        path = [a]
                        cur = u
                        while cur != -1:
                            path.append(cur)
                            cur = prev[cur]
                        return path
                    nxt.append(v)
            frontier = nxt
        return None

    def is_in_ring(self, i: int) -> bool:
        return bool(self.atoms[i].ring_sizes)

    @property
    def n_rings(self) -> int:
        return len(self.rings)

    def ring_systems(self) -> list[set[int]]:
        """Fused ring systems: rings sharing at least one atom, merged."""
        systems: list[set[int]] = []
        for ring in self.rings:
            s = set(ring)
            merged = [s]
            rest = []
            for existing in systems:
                (merged if existing & s else rest).append(existing)
            combined = set().union(*merged)
            systems = rest + [combined]
        return systems

    # ------------------------------------------------------------- aromaticity

    def _pi_contribution(self, i: int, kekule_double: dict[int, int]) -> int | None:
        """Pi electrons atom i donates to a ring system, or None if not sp2-capable."""
        atom = self.atoms[i]
        sym = atom.symbol
        sigma = self.sigma_degree(i)
        has_ring_double = kekule_double.get(i, 0) > 0
        exo_double_to_en = False
        for bond in self.bonds:
            if bond.order == DOUBLE and i in (bond.a, bond.b) and not bond.in_ring:
                other = self.atoms[bond.other(i)].symbol
                if other in ("O", "S", "Se", "N"):
                    exo_double_to_en = True
        if sym == "C":
            if exo_double_to_en:
                return 0                      # e.g. cyclopentadienone carbonyl
            if atom.charge == -1 and sigma <= 3:
                return 2
            if atom.charge == 1:
                return 0
            return 1 if (has_ring_double or atom.aromatic) else None
        if sym == "N":
            if atom.charge == 1:
                return 1 if has_ring_double or sigma == 3 else None
            return 2 if sigma == 3 else 1
        if sym in ("O", "S", "Se", "Te"):
            if atom.charge == 1 and sym != "O":
                return 1
            return 2 if sigma == 2 else None
        if sym in ("P", "As"):
            return 2 if sigma == 3 else 1
        if sym == "B":
            return 0
        if sym == "Si":
            return 1 if has_ring_double else None
        return None

    def perceive_aromaticity(self) -> None:
        """Hueckel 4n+2 perception over SSSR rings and fused ring systems.

        Rings written lowercase in SMILES are trusted; Kekule rings are tested.
        """
        kekule_double: dict[int, int] = {}
        for bond in self.bonds:
            if bond.order == DOUBLE and bond.in_ring:
                kekule_double[bond.a] = kekule_double.get(bond.a, 0) + 1
                kekule_double[bond.b] = kekule_double.get(bond.b, 0) + 1

        exocyclic_double = set()
        for bond in self.bonds:
            if bond.order == DOUBLE and not bond.in_ring:
                exocyclic_double.add(bond.a)
                exocyclic_double.add(bond.b)

        def ring_is_aromatic(ring: tuple[int, ...] | list[int]) -> bool:
            # Conservative Kekule perception: a ring carrying an exocyclic double
            # bond (e.g. the carbonyl ring of caffeine) is not aromatized. Rings
            # written lowercase in the input are trusted and never reach here.
            if any(i in exocyclic_double for i in ring):
                return False
            total = 0
            for i in ring:
                contrib = self._pi_contribution(i, kekule_double)
                if contrib is None:
                    return False
                total += contrib
            return total % 4 == 2 and total >= 2

        aromatic_atoms: set[int] = set()
        aromatic_rings: list[tuple[int, ...]] = []
        for ring in self.rings:
            if all(self.atoms[i].aromatic for i in ring) or ring_is_aromatic(ring):
                aromatic_rings.append(ring)
                aromatic_atoms.update(ring)

        # Fused systems aromatic as a whole but not ring-by-ring (e.g. azulene).
        for system in self.ring_systems():
            if system & aromatic_atoms:
                continue
            member = [r for r in self.rings if set(r) <= system]
            if len(member) > 1 and ring_is_aromatic(sorted(system)):
                aromatic_atoms.update(system)
                aromatic_rings.extend(member)

        for i in aromatic_atoms:
            self.atoms[i].aromatic = True
            contrib = self._pi_contribution(i, kekule_double)
            self.atoms[i].pi_electrons = 0 if contrib is None else contrib
        for bond in self.bonds:
            if bond.a in aromatic_atoms and bond.b in aromatic_atoms and bond.in_ring:
                same_ring = any(bond.a in r and bond.b in r for r in aromatic_rings)
                if same_ring:
                    bond.order = AROMATIC

    # ----------------------------------------------------------------- valence

    def explicit_valence(self, i: int) -> float:
        """Summed bond orders at atom i; aromatic bonds count as sigma + pi share."""
        atom = self.atoms[i]
        if atom.aromatic:
            sigma = len(self._adj[i])
            # A pi-donor (2 electrons) uses a lone pair, so adds nothing to valence;
            # a 1-electron contributor shares a pi bond, adding one.
            return sigma + (1 if atom.pi_electrons == 1 else 0) + atom.explicit_h
        total = 0.0
        for j in self._adj[i]:
            bond = self.bond_between(i, j)
            total += BOND_ORDER[bond.order]
        return total + atom.explicit_h

    def valence_errors(self) -> list[str]:
        """Atoms whose explicit valence exceeds the element maximum.

        Mirrors the sanitization step of mainstream toolkits, so datasets can be
        filtered to the same molecules used in published benchmarks.
        """
        problems = []
        for i, atom in enumerate(self.atoms):
            cap = max_valence(atom.symbol)
            if cap is None:
                continue
            val = self.explicit_valence(i) + atom.implicit_h
            allowed = cap + (atom.charge if atom.symbol in ("N", "P", "S", "O") else abs(atom.charge))
            if val > allowed + 1e-6:
                problems.append(
                    f"atom {i} ({atom.symbol}{atom.charge:+d}) valence {val:g} > {allowed}")
        return problems

    def is_sane(self) -> bool:
        return not self.valence_errors()

    def assign_implicit_hydrogens(self) -> None:
        for i, atom in enumerate(self.atoms):
            if atom.in_bracket or atom.symbol == WILDCARD:
                atom.implicit_h = 0      # brackets state hydrogens exactly
                continue
            val = self.explicit_valence(i)
            target = default_valence(atom.symbol, val - atom.charge * (1 if atom.symbol in ("N", "P") else -1))
            if atom.symbol in ("N", "P") and atom.charge == 1:
                target = 4
            elif atom.charge != 0:
                target = default_valence(atom.symbol, val) + atom.charge
            atom.implicit_h = max(0, int(round(target - val)))

    def perceive(self) -> "Molecule":
        """Run the full perception pipeline. Call once after parsing."""
        self.build_index()
        self.perceive_rings()
        self.perceive_aromaticity()
        self.assign_implicit_hydrogens()
        return self

    # ------------------------------------------------------------- composition

    def formula(self) -> str:
        counts: dict[str, int] = {}
        for atom in self.atoms:
            if atom.symbol == WILDCARD:
                continue
            counts[atom.symbol] = counts.get(atom.symbol, 0) + 1
            h = atom.total_h
            if h:
                counts["H"] = counts.get("H", 0) + h
        order = ["C", "H"] + sorted(k for k in counts if k not in ("C", "H"))
        parts = []
        for sym in order:
            n = counts.get(sym, 0)
            if n:
                parts.append(sym + (str(n) if n > 1 else ""))
        return "".join(parts)

    def molecular_weight(self) -> float:
        total = 0.0
        for atom in self.atoms:
            total += atomic_mass(atom.symbol) + atom.total_h * atomic_mass("H")
        return total

    def heavy_atom_count(self) -> int:
        return sum(1 for a in self.atoms if a.symbol != "H")

    def charge(self) -> int:
        return sum(a.charge for a in self.atoms)

    def atomic_numbers(self) -> np.ndarray:
        return np.array([atomic_number(a.symbol) for a in self.atoms], dtype=np.int32)

    def copy(self) -> "Molecule":
        import copy as _copy
        m = Molecule(
            atoms=[_copy.replace(a) for a in self.atoms],
            bonds=[_copy.replace(b) for b in self.bonds],
            name=self.name, smiles=self.smiles,
            coords=None if self.coords is None else self.coords.copy(),
            rings=list(self.rings),
        )
        return m.build_index()

    def __repr__(self) -> str:
        return f"<Molecule {self.formula()} atoms={self.n_atoms} bonds={self.n_bonds} rings={self.n_rings}>"
