"""Discrete Ricci curvature and Ricci flow on molecular graphs.

Ollivier-Ricci curvature compares the cost of transporting the neighbourhood
measure of one atom onto another's with the distance between them: positive
curvature marks densely connected regions such as fused rings, negative
curvature marks bridges and bottlenecks. Forman-Ricci curvature is a
combinatorial alternative with a closed form.

The Wasserstein-1 distance is solved exactly with a transportation simplex, so
curvature values are not approximations.
"""

from __future__ import annotations

import numpy as np

from nexus.chem.mol import Molecule


def wasserstein_1(supply: np.ndarray, demand: np.ndarray,
                  cost: np.ndarray, tol: float = 1e-10) -> float:
    """Exact optimal transport cost between two discrete distributions.

    Transportation simplex (MODI / u-v method): start from a northwest-corner
    basic feasible solution and pivot on the most negative reduced cost until
    optimal. Supplies and demands must have equal total mass.
    """
    supply = np.asarray(supply, dtype=float).copy()
    demand = np.asarray(demand, dtype=float).copy()
    cost = np.asarray(cost, dtype=float)
    m, n = len(supply), len(demand)
    if m == 0 or n == 0:
        return 0.0
    total = supply.sum()
    if total <= tol:
        return 0.0
    demand = demand * (total / demand.sum())

    # Northwest-corner rule for an initial basic feasible solution.
    x = np.zeros((m, n))
    basis: set[tuple[int, int]] = set()
    i = j = 0
    s, d = supply.copy(), demand.copy()
    while i < m and j < n:
        amount = min(s[i], d[j])
        x[i, j] = amount
        basis.add((i, j))
        s[i] -= amount
        d[j] -= amount
        if s[i] <= tol and i < m - 1:
            i += 1
        elif d[j] <= tol and j < n - 1:
            j += 1
        else:
            break
    # A basis needs exactly m + n - 1 cells; pad with degenerate ones.
    k = 0
    while len(basis) < m + n - 1 and k < m * n:
        cell = (k // n, k % n)
        if cell not in basis:
            basis.add(cell)
        k += 1

    for _ in range(200):
        u, v = _potentials(basis, cost, m, n)
        reduced = cost - (u[:, None] + v[None, :])
        reduced[tuple(zip(*basis))] = 0.0
        entering = np.unravel_index(np.argmin(reduced), reduced.shape)
        if reduced[entering] >= -tol:
            break
        cycle = _find_cycle(basis, entering, m, n)
        if cycle is None:
            break
        minus = cycle[1::2]
        theta_idx = min(minus, key=lambda c: x[c])
        theta = x[theta_idx]
        for pos, cell in enumerate(cycle):
            x[cell] += theta if pos % 2 == 0 else -theta
        basis.add(entering)
        basis.discard(theta_idx)
    return float((x * cost).sum())


def _potentials(basis: set[tuple[int, int]], cost: np.ndarray,
                m: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.full(m, np.nan)
    v = np.full(n, np.nan)
    u[0] = 0.0
    cells = list(basis)
    for _ in range(m + n):
        for (i, j) in cells:
            if not np.isnan(u[i]) and np.isnan(v[j]):
                v[j] = cost[i, j] - u[i]
            elif np.isnan(u[i]) and not np.isnan(v[j]):
                u[i] = cost[i, j] - v[j]
        if not (np.isnan(u).any() or np.isnan(v).any()):
            break
    return np.nan_to_num(u), np.nan_to_num(v)


def _find_cycle(basis: set[tuple[int, int]], start: tuple[int, int],
                m: int, n: int) -> list[tuple[int, int]] | None:
    """Closed alternating row/column path from `start` through basis cells."""
    cells = list(basis | {start})
    by_row: dict[int, list[tuple[int, int]]] = {}
    by_col: dict[int, list[tuple[int, int]]] = {}
    for c in cells:
        by_row.setdefault(c[0], []).append(c)
        by_col.setdefault(c[1], []).append(c)

    path = [start]

    def step(current: tuple[int, int], move_along_row: bool) -> bool:
        options = (by_row.get(current[0], []) if move_along_row
                   else by_col.get(current[1], []))
        for nxt in options:
            if nxt == current:
                continue
            if nxt == start and len(path) >= 4 and len(path) % 2 == 0:
                return True
            if nxt in path:
                continue
            path.append(nxt)
            if step(nxt, not move_along_row):
                return True
            path.pop()
        return False

    return path if step(start, True) else None


def _neighbour_measure(mol: Molecule, i: int, alpha: float) -> tuple[list[int], np.ndarray]:
    """Lazy random-walk measure: mass alpha stays at i, the rest spreads."""
    neighbours = mol.neighbors(i)
    if not neighbours:
        return [i], np.array([1.0])
    support = [i] + list(neighbours)
    mass = np.full(len(support), (1.0 - alpha) / len(neighbours))
    mass[0] = alpha
    return support, mass


def ollivier_ricci_curvature(mol: Molecule, alpha: float = 0.5
                             ) -> dict[tuple[int, int], float]:
    """Ollivier-Ricci curvature of every bond.

    kappa(x, y) = 1 - W1(m_x, m_y) / d(x, y), with d the shortest-path distance
    and m the lazy random-walk measure. Values lie in [-2, 1].
    """
    D = mol.distance_matrix().astype(float)
    curvature: dict[tuple[int, int], float] = {}
    for bond in mol.bonds:
        i, j = bond.a, bond.b
        support_i, mass_i = _neighbour_measure(mol, i, alpha)
        support_j, mass_j = _neighbour_measure(mol, j, alpha)
        cost = D[np.ix_(support_i, support_j)]
        if (cost < 0).any():
            continue
        w1 = wasserstein_1(mass_i, mass_j, cost)
        d = D[i, j]
        curvature[(min(i, j), max(i, j))] = float(1.0 - w1 / d) if d > 0 else 0.0
    return curvature


def forman_ricci_curvature(mol: Molecule) -> dict[tuple[int, int], float]:
    """Forman-Ricci curvature of every bond (combinatorial, closed form).

    For an unweighted graph: F(e) = 4 - deg(u) - deg(v) for the edge e = (u, v).
    """
    out: dict[tuple[int, int], float] = {}
    for bond in mol.bonds:
        i, j = bond.a, bond.b
        out[(min(i, j), max(i, j))] = float(4 - mol.degree(i) - mol.degree(j))
    return out


def ricci_flow(mol: Molecule, iterations: int = 10, step: float = 0.5,
               alpha: float = 0.5) -> tuple[dict[tuple[int, int], float], list[float]]:
    """Discrete Ricci flow on bond weights: dw/dt = -kappa * w.

    Edge weights evolve so that curvature becomes uniform; the spread of
    curvature values across iterations is returned as a convergence trace and
    doubles as a descriptor of how far a molecule's graph is from a
    constant-curvature one.
    """
    weights = {(min(b.a, b.b), max(b.a, b.b)): 1.0 for b in mol.bonds}
    trace: list[float] = []
    for _ in range(iterations):
        curvature = _weighted_ollivier(mol, weights, alpha)
        values = np.array(list(curvature.values()))
        trace.append(float(values.std()) if len(values) else 0.0)
        mean_curv = float(values.mean()) if len(values) else 0.0
        for edge, w in weights.items():
            k = curvature.get(edge, 0.0)
            weights[edge] = max(w - step * (k - mean_curv) * w, 1e-3)
        total = sum(weights.values())
        if total > 0:
            scale = len(weights) / total
            weights = {e: w * scale for e, w in weights.items()}
    return weights, trace


def _weighted_ollivier(mol: Molecule, weights: dict[tuple[int, int], float],
                       alpha: float) -> dict[tuple[int, int], float]:
    n = mol.n_atoms
    inf = float("inf")
    dist = np.full((n, n), inf)
    np.fill_diagonal(dist, 0.0)
    for (a, b), w in weights.items():
        dist[a, b] = dist[b, a] = w
    for k in range(n):                     # Floyd-Warshall on weighted edges
        np.minimum(dist, dist[:, k][:, None] + dist[k, :][None, :], out=dist)

    out: dict[tuple[int, int], float] = {}
    for (i, j), w in weights.items():
        support_i, mass_i = _neighbour_measure(mol, i, alpha)
        support_j, mass_j = _neighbour_measure(mol, j, alpha)
        cost = dist[np.ix_(support_i, support_j)]
        if not np.isfinite(cost).all():
            continue
        w1 = wasserstein_1(mass_i, mass_j, cost)
        out[(i, j)] = float(1.0 - w1 / w) if w > 0 else 0.0
    return out


def curvature_features(mol: Molecule, alpha: float = 0.5) -> np.ndarray:
    """Fixed-length curvature summary: Ollivier and Forman statistics plus the
    Ricci-flow convergence trace."""
    ollivier = np.array(list(ollivier_ricci_curvature(mol, alpha).values()))
    forman = np.array(list(forman_ricci_curvature(mol).values()))
    _, trace = ricci_flow(mol, iterations=5, alpha=alpha)
    def stats(v):
        if len(v) == 0:
            return [0.0] * 5
        return [float(v.mean()), float(v.std()), float(v.min()), float(v.max()),
                float(np.median(v))]
    trace = (trace + [0.0] * 5)[:5]
    return np.array(stats(ollivier) + stats(forman) + trace)
