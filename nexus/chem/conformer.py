"""3D conformer generation by distance geometry plus force-field relaxation.

Pipeline: build a distance bounds matrix from bond lengths, bond angles and
van der Waals contacts; smooth it with the triangle inequality; sample a
metric matrix; embed by classical multidimensional scaling; then relax under a
simple force field. No external chemistry toolkit is required.
"""

from __future__ import annotations

import numpy as np

from nexus.chem.mol import AROMATIC, BOND_ORDER, DOUBLE, Molecule, SINGLE, TRIPLE
from nexus.chem.periodic import covalent_radius, vdw_radius

# Bond-order corrections to the sum of covalent radii (Angstrom).
BOND_SHORTENING = {SINGLE: 0.00, AROMATIC: 0.10, DOUBLE: 0.14, TRIPLE: 0.22}
# Ideal bond angles by the number of sigma neighbours (degrees).
IDEAL_ANGLE = {2: 180.0, 3: 120.0, 4: 109.5, 5: 100.0, 6: 90.0}


def ideal_bond_length(mol: Molecule, i: int, j: int) -> float:
    bond = mol.bond_between(i, j)
    order = bond.order if bond else SINGLE
    base = covalent_radius(mol.atoms[i].symbol) + covalent_radius(mol.atoms[j].symbol)
    return max(base - BOND_SHORTENING.get(order, 0.0), 0.7)


def ideal_angle(mol: Molecule, centre: int) -> float:
    atom = mol.atoms[centre]
    n_sigma = mol.degree(centre) + atom.total_h
    if atom.aromatic:
        return 120.0
    has_triple = any(mol.bond_between(centre, j).order == TRIPLE
                     for j in mol.neighbors(centre))
    if has_triple:
        return 180.0
    has_double = any(mol.bond_between(centre, j).order == DOUBLE
                     for j in mol.neighbors(centre))
    if has_double:
        return 120.0
    # Small rings force tighter angles than the ideal hybridisation value.
    if atom.ring_sizes:
        smallest = min(atom.ring_sizes)
        if smallest <= 4:
            return 180.0 * (smallest - 2) / smallest
    return IDEAL_ANGLE.get(n_sigma, 109.5)


def distance_bounds(mol: Molecule) -> tuple[np.ndarray, np.ndarray]:
    """Lower and upper bounds on every interatomic distance."""
    n = mol.n_atoms
    D = mol.distance_matrix()
    lower = np.zeros((n, n))
    upper = np.full((n, n), 100.0)

    for i in range(n):
        upper[i, i] = lower[i, i] = 0.0
        for j in range(i + 1, n):
            if D[i, j] == 1:
                d = ideal_bond_length(mol, i, j)
                lower[i, j] = upper[i, j] = d
            elif D[i, j] == 2:
                centre = _shared_neighbour(mol, i, j)
                a = ideal_bond_length(mol, i, centre)
                b = ideal_bond_length(mol, centre, j)
                theta = np.radians(ideal_angle(mol, centre))
                d = np.sqrt(a * a + b * b - 2 * a * b * np.cos(theta))
                lower[i, j], upper[i, j] = d * 0.95, d * 1.05
            elif D[i, j] == 3:
                # 1-4 pairs range between cis and trans torsions.
                lower[i, j] = _vdw_contact(mol, i, j) * 0.8
                upper[i, j] = 5.0
            else:
                lower[i, j] = _vdw_contact(mol, i, j)
                if D[i, j] > 0:
                    upper[i, j] = min(100.0, 1.6 * D[i, j] + 2.0)
            lower[j, i], upper[j, i] = lower[i, j], upper[i, j]

    _apply_planar_ring_bounds(mol, lower, upper)
    return triangle_smooth(lower, upper)


def _apply_planar_ring_bounds(mol: Molecule, lower: np.ndarray,
                              upper: np.ndarray, tolerance: float = 0.004) -> None:
    """Pin aromatic rings to a planar regular polygon.

    Bond-length and bond-angle restraints alone leave a six-membered ring free
    to pucker into a chair. For a planar regular n-gon of side a, vertices k
    steps apart sit at a*sin(k*pi/n)/sin(pi/n); fixing those distances forces
    planarity without an explicit out-of-plane term.
    """
    for ring in mol.rings:
        if not all(mol.atoms[i].aromatic for i in ring):
            continue
        n = len(ring)
        if n < 5:
            continue
        sides = [ideal_bond_length(mol, ring[k], ring[(k + 1) % n]) for k in range(n)]
        a = float(np.mean(sides))
        for x in range(n):
            for y in range(x + 1, n):
                steps = min(y - x, n - (y - x))
                if steps < 2:
                    continue
                d = a * np.sin(steps * np.pi / n) / np.sin(np.pi / n)
                i, j = ring[x], ring[y]
                lower[i, j] = lower[j, i] = d * (1 - tolerance)
                upper[i, j] = upper[j, i] = d * (1 + tolerance)


def _shared_neighbour(mol: Molecule, i: int, j: int) -> int:
    shared = set(mol.neighbors(i)) & set(mol.neighbors(j))
    return next(iter(shared))


def _vdw_contact(mol: Molecule, i: int, j: int) -> float:
    return 0.75 * (vdw_radius(mol.atoms[i].symbol) + vdw_radius(mol.atoms[j].symbol))


def triangle_smooth(lower: np.ndarray, upper: np.ndarray,
                    iterations: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Enforce the triangle inequality on the bounds (Floyd-Warshall style)."""
    n = len(upper)
    upper = upper.copy()
    lower = lower.copy()
    for _ in range(iterations):
        for k in range(n):
            # Upper bounds obey u_ij <= u_ik + u_kj.
            candidate = upper[:, k][:, None] + upper[k, :][None, :]
            np.minimum(upper, candidate, out=upper)
        for k in range(n):
            # Lower bounds obey l_ij >= l_ik - u_kj.
            candidate = lower[:, k][:, None] - upper[k, :][None, :]
            np.maximum(lower, candidate, out=lower)
            candidate = lower[k, :][None, :] - upper[:, k][:, None]
            np.maximum(lower, candidate, out=lower)
        np.fill_diagonal(lower, 0.0)
        np.fill_diagonal(upper, 0.0)
        lower = np.minimum(lower, upper)
    return lower, upper


def embed_distance_matrix(D: np.ndarray, n_dim: int = 3) -> np.ndarray:
    """Classical multidimensional scaling of a (possibly inexact) distance matrix."""
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    B = (B + B.T) / 2
    values, vectors = np.linalg.eigh(B)
    order = np.argsort(values)[::-1][:n_dim]
    values = np.maximum(values[order], 0.0)
    return vectors[:, order] * np.sqrt(values)


def planar_groups(mol: Molecule) -> list[np.ndarray]:
    """Atom groups that should be coplanar: aromatic rings and their fused
    systems, plus sp2 centres with their immediate neighbours."""
    groups: list[np.ndarray] = []
    aromatic_rings = [set(r) for r in mol.rings
                      if all(mol.atoms[i].aromatic for i in r)]
    # Merge fused aromatic rings so a whole ring system shares one plane.
    merged: list[set[int]] = []
    for ring in aromatic_rings:
        overlapping = [m for m in merged if m & ring]
        rest = [m for m in merged if not (m & ring)]
        merged = rest + [set().union(ring, *overlapping)]
    groups.extend(np.array(sorted(g)) for g in merged)

    for i, atom in enumerate(mol.atoms):
        if atom.aromatic:
            continue
        neighbours = mol.neighbors(i)
        if len(neighbours) != 3:
            continue
        if any(mol.bond_between(i, j).order == DOUBLE for j in neighbours):
            groups.append(np.array([i] + list(neighbours)))
    return groups


def _planarity_energy_and_gradient(
    coords: np.ndarray, groups: list[np.ndarray], weight: float,
) -> tuple[float, np.ndarray]:
    """Penalise out-of-plane displacement within each coplanar group.

    The best-fit plane normal is recomputed each step and treated as fixed when
    differentiating, which gives a stable descent direction.
    """
    energy = 0.0
    grad = np.zeros_like(coords)
    for idx in groups:
        if len(idx) < 4:
            continue
        points = coords[idx]
        centre = points.mean(axis=0)
        centred = points - centre
        # Smallest right-singular vector is the best-fit plane normal.
        normal = np.linalg.svd(centred, full_matrices=False)[2][-1]
        offsets = centred @ normal
        energy += weight * float((offsets ** 2).sum())
        grad[idx] += 2.0 * weight * offsets[:, None] * normal[None, :]
    return energy, grad


def _force_field_energy_and_gradient(
    coords: np.ndarray, lower: np.ndarray, upper: np.ndarray,
    groups: list[np.ndarray] | None = None, planarity_weight: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Flat-bottom distance restraints plus an out-of-plane penalty."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1) + 1e-12)
    np.fill_diagonal(dist, 1.0)

    too_long = dist > upper
    too_short = dist < lower
    violation = np.where(too_long, dist - upper, 0.0) + np.where(too_short, dist - lower, 0.0)
    np.fill_diagonal(violation, 0.0)
    energy = float((violation ** 2).sum() / 2)

    # dE/dx_i = sum_j 2 * violation_ij * (x_i - x_j)/d_ij
    scale = 2.0 * violation / dist
    grad = (scale[:, :, None] * diff).sum(axis=1)

    if groups:
        plane_energy, plane_grad = _planarity_energy_and_gradient(
            coords, groups, planarity_weight)
        energy += plane_energy
        grad += plane_grad
    return energy, grad


def relax(coords: np.ndarray, lower: np.ndarray, upper: np.ndarray,
          steps: int = 300, lr: float = 0.05,
          groups: list[np.ndarray] | None = None) -> tuple[np.ndarray, float]:
    """Gradient descent on the restraint energy with simple step adaptation."""
    coords = coords.copy()
    energy, grad = _force_field_energy_and_gradient(coords, lower, upper, groups)
    for _ in range(steps):
        candidate = coords - lr * grad
        new_energy, new_grad = _force_field_energy_and_gradient(
            candidate, lower, upper, groups)
        if new_energy < energy:
            coords, energy, grad = candidate, new_energy, new_grad
            lr *= 1.1
        else:
            lr *= 0.5
            if lr < 1e-6:
                break
    return coords, energy


def embed_molecule(mol: Molecule, seed: int = 0, n_tries: int = 4,
                   steps: int = 300) -> tuple[np.ndarray, float]:
    """Generate one 3D conformer; returns (coordinates, restraint energy).

    Several random metrizations are tried and the lowest-energy result kept.
    """
    n = mol.n_atoms
    if n == 0:
        return np.zeros((0, 3)), 0.0
    if n == 1:
        return np.zeros((1, 3)), 0.0
    lower, upper = distance_bounds(mol)
    groups = planar_groups(mol)
    rng = np.random.default_rng(seed)
    best, best_energy = None, np.inf
    for _ in range(n_tries):
        sample = lower + rng.random((n, n)) * (upper - lower)
        sample = (sample + sample.T) / 2
        np.fill_diagonal(sample, 0.0)
        coords = embed_distance_matrix(sample, 3)
        coords, energy = relax(coords, lower, upper, steps=steps, groups=groups)
        if energy < best_energy:
            best, best_energy = coords, energy
    return best - best.mean(axis=0), float(best_energy)


def embed_conformers(mol: Molecule, n_conformers: int = 5, seed: int = 0,
                     steps: int = 300) -> list[np.ndarray]:
    """A small conformer ensemble, sorted by restraint energy."""
    out = []
    for k in range(n_conformers):
        coords, energy = embed_molecule(mol, seed=seed + 1000 * k, n_tries=2, steps=steps)
        out.append((energy, coords))
    out.sort(key=lambda t: t[0])
    return [c for _, c in out]


def bond_length_report(mol: Molecule, coords: np.ndarray) -> dict[str, float]:
    """Deviation of realised bond lengths from their ideal values."""
    errors = []
    for bond in mol.bonds:
        d = float(np.linalg.norm(coords[bond.a] - coords[bond.b]))
        errors.append(abs(d - ideal_bond_length(mol, bond.a, bond.b)))
    if not errors:
        return {"mean_abs_error": 0.0, "max_abs_error": 0.0, "n_bonds": 0}
    return {"mean_abs_error": float(np.mean(errors)),
            "max_abs_error": float(np.max(errors)), "n_bonds": len(errors)}


def min_nonbonded_distance(mol: Molecule, coords: np.ndarray) -> float:
    """Closest approach between atoms more than two bonds apart."""
    D = mol.distance_matrix()
    best = np.inf
    for i in range(mol.n_atoms):
        for j in range(i + 1, mol.n_atoms):
            if D[i, j] > 2 or D[i, j] < 0:
                best = min(best, float(np.linalg.norm(coords[i] - coords[j])))
    return best if np.isfinite(best) else 0.0
