"""Persistent homology of point clouds via the Vietoris-Rips filtration.

H0 is computed with union-find (exact and near-linear). H1 and H2 come from the
standard column reduction of the Z/2 boundary matrix. Diagrams are turned into
fixed-length features by persistence images, landscapes and summary statistics,
so they can be fed to any downstream model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PersistenceDiagram:
    """Birth/death pairs per homology dimension."""
    points: dict[int, np.ndarray]          # dimension -> (n, 2) array

    def __getitem__(self, dim: int) -> np.ndarray:
        return self.points.get(dim, np.zeros((0, 2)))

    @property
    def dimensions(self) -> list[int]:
        return sorted(self.points)

    def lifetimes(self, dim: int) -> np.ndarray:
        pts = self[dim]
        if len(pts) == 0:
            return np.zeros(0)
        return pts[:, 1] - pts[:, 0]

    def total_persistence(self, dim: int) -> float:
        return float(np.sum(self.lifetimes(dim)))

    def betti_at(self, dim: int, epsilon: float) -> int:
        """Betti number of the complex at filtration value epsilon."""
        pts = self[dim]
        if len(pts) == 0:
            return 0
        return int(np.sum((pts[:, 0] <= epsilon) & (pts[:, 1] > epsilon)))


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def h0_persistence(distances: np.ndarray) -> np.ndarray:
    """Connected-component births and deaths, by union-find over sorted edges.

    Every point is born at 0; a component dies when it merges into an older
    one. One component (the whole cloud) never dies and is recorded as
    infinite.
    """
    n = len(distances)
    if n == 0:
        return np.zeros((0, 2))
    iu = np.triu_indices(n, k=1)
    order = np.argsort(distances[iu])
    rows, cols = iu[0][order], iu[1][order]
    weights = distances[iu][order]

    uf = _UnionFind(n)
    deaths = []
    for a, b, w in zip(rows, cols, weights):
        if uf.union(int(a), int(b)):
            deaths.append(w)
        if len(deaths) == n - 1:
            break
    out = [(0.0, float(d)) for d in deaths]
    out.append((0.0, np.inf))
    return np.array(out, dtype=float)


def _simplices_up_to(distances: np.ndarray, max_edge: float, max_dim: int):
    """Vietoris-Rips simplices with their filtration values, in build order."""
    n = len(distances)
    simplices: list[tuple[float, tuple[int, ...]]] = [(0.0, (i,)) for i in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = distances[i, j]
            if d <= max_edge:
                edges.append((float(d), (i, j)))
    simplices.extend(edges)
    if max_dim >= 2:
        edge_set = {e[1] for e in edges}
        for a in range(n):
            for b in range(a + 1, n):
                if (a, b) not in edge_set:
                    continue
                for c in range(b + 1, n):
                    if (a, c) in edge_set and (b, c) in edge_set:
                        value = max(distances[a, b], distances[a, c], distances[b, c])
                        simplices.append((float(value), (a, b, c)))
    if max_dim >= 3:
        tri_set = {s[1] for s in simplices if len(s[1]) == 3}
        for t in list(tri_set):
            a, b, c = t
            for d in range(c + 1, n):
                if all(x in tri_set for x in ((a, b, d), (a, c, d), (b, c, d))):
                    value = max(distances[i, j] for i in (a, b, c, d)
                                for j in (a, b, c, d) if i < j)
                    simplices.append((float(value), (a, b, c, d)))
    simplices.sort(key=lambda s: (s[0], len(s[1])))
    return simplices


def rips_persistence(points: np.ndarray, max_dim: int = 1,
                     max_edge: float | None = None) -> PersistenceDiagram:
    """Persistent homology of a Vietoris-Rips filtration.

    max_dim=1 gives components and loops; 2 adds voids. Cost grows quickly with
    max_dim, so keep it low for anything beyond a few dozen points.
    """
    points = np.asarray(points, dtype=float)
    n = len(points)
    if n == 0:
        return PersistenceDiagram({d: np.zeros((0, 2)) for d in range(max_dim + 1)})
    distances = pairwise_distances(points)
    if max_edge is None:
        max_edge = float(distances.max()) if n > 1 else 0.0

    out: dict[int, np.ndarray] = {0: h0_persistence(distances)}
    if max_dim == 0:
        return PersistenceDiagram(out)

    simplices = _simplices_up_to(distances, max_edge, max_dim + 1)
    index = {s: k for k, (_, s) in enumerate(simplices)}
    values = np.array([v for v, _ in simplices])

    # Z/2 column reduction: column j holds the boundary of simplex j as a set
    # of row indices; reduce until every column has a distinct lowest entry.
    # reduced_by_low[r] is the reduced column whose lowest entry is r; the
    # simplex that created it is paired with the simplex at row r.
    reduced_by_low: dict[int, set[int]] = {}
    pairs: list[tuple[int, int, int]] = []       # (dim, birth index, death index)

    for j, (_, simplex) in enumerate(simplices):
        dim = len(simplex) - 1
        if dim == 0:
            continue
        column = {index[simplex[:k] + simplex[k + 1:]] for k in range(len(simplex))}
        while column:
            low = max(column)
            if low not in reduced_by_low:
                break
            column ^= reduced_by_low[low]
        if column:
            low = max(column)
            reduced_by_low[low] = column
            pairs.append((dim - 1, low, j))

    for dim in range(1, max_dim + 1):
        rows = [(values[b], values[d]) for pd, b, d in pairs
                if pd == dim and values[d] > values[b]]
        out[dim] = np.array(rows, dtype=float) if rows else np.zeros((0, 2))
    return PersistenceDiagram(out)


# ------------------------------------------------------------------- features

def persistence_image(diagram: PersistenceDiagram, dim: int = 1,
                      resolution: int = 8, sigma: float = 0.2,
                      birth_range: tuple[float, float] = (0.0, 6.0),
                      persistence_range: tuple[float, float] = (0.0, 6.0),
                      weight_by_persistence: bool = True) -> np.ndarray:
    """Gaussian-smoothed image of the diagram in (birth, persistence) coords.

    Returns a flat `resolution * resolution` vector, a stable finite-dimensional
    summary suitable as a model input (Adams et al. 2017).
    """
    pts = diagram[dim]
    image = np.zeros((resolution, resolution))
    if len(pts) == 0:
        return image.ravel()
    finite = pts[np.isfinite(pts[:, 1])]
    if len(finite) == 0:
        return image.ravel()
    birth = finite[:, 0]
    persistence = finite[:, 1] - finite[:, 0]

    bx = np.linspace(*birth_range, resolution)
    by = np.linspace(*persistence_range, resolution)
    grid_x, grid_y = np.meshgrid(bx, by, indexing="ij")
    for b, p in zip(birth, persistence):
        weight = p if weight_by_persistence else 1.0
        image += weight * np.exp(-(((grid_x - b) ** 2 + (grid_y - p) ** 2)
                                   / (2 * sigma ** 2)))
    return image.ravel()


def persistence_landscape(diagram: PersistenceDiagram, dim: int = 1,
                          n_layers: int = 3, resolution: int = 16,
                          value_range: tuple[float, float] = (0.0, 6.0)) -> np.ndarray:
    """Persistence landscape (Bubenik 2015), flattened to a feature vector."""
    pts = diagram[dim]
    grid = np.linspace(*value_range, resolution)
    out = np.zeros((n_layers, resolution))
    if len(pts) == 0:
        return out.ravel()
    finite = pts[np.isfinite(pts[:, 1])]
    if len(finite) == 0:
        return out.ravel()
    # Each point contributes a tent function; layer k is the k-th largest value.
    tents = np.zeros((len(finite), resolution))
    for i, (b, d) in enumerate(finite):
        tents[i] = np.maximum(0.0, np.minimum(grid - b, d - grid))
    ordered = -np.sort(-tents, axis=0)
    out[:min(n_layers, len(finite))] = ordered[:n_layers]
    return out.ravel()


def diagram_statistics(diagram: PersistenceDiagram, dim: int) -> dict[str, float]:
    lifetimes = diagram.lifetimes(dim)
    lifetimes = lifetimes[np.isfinite(lifetimes)]
    if len(lifetimes) == 0:
        return {f"h{dim}_count": 0.0, f"h{dim}_total": 0.0, f"h{dim}_max": 0.0,
                f"h{dim}_mean": 0.0, f"h{dim}_entropy": 0.0}
    total = float(lifetimes.sum())
    p = lifetimes / total if total > 0 else np.ones_like(lifetimes) / len(lifetimes)
    entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return {
        f"h{dim}_count": float(len(lifetimes)),
        f"h{dim}_total": total,
        f"h{dim}_max": float(lifetimes.max()),
        f"h{dim}_mean": float(lifetimes.mean()),
        f"h{dim}_entropy": entropy,
    }


def topological_features(points: np.ndarray, max_dim: int = 1,
                         resolution: int = 8) -> np.ndarray:
    """Concatenated persistence images and summary statistics for a point cloud."""
    diagram = rips_persistence(points, max_dim=max_dim)
    parts = []
    for dim in range(max_dim + 1):
        parts.append(persistence_image(diagram, dim=dim, resolution=resolution))
        stats = diagram_statistics(diagram, dim)
        parts.append(np.array([stats[k] for k in sorted(stats)]))
    return np.concatenate(parts)


def bottleneck_distance(a: PersistenceDiagram, b: PersistenceDiagram,
                        dim: int = 1) -> float:
    """Bottleneck distance between two diagrams (exact, via a threshold search).

    Points may be matched to the diagonal, so diagrams of different sizes are
    comparable.
    """
    pa = a[dim][np.isfinite(a[dim][:, 1])] if len(a[dim]) else np.zeros((0, 2))
    pb = b[dim][np.isfinite(b[dim][:, 1])] if len(b[dim]) else np.zeros((0, 2))
    if len(pa) == 0 and len(pb) == 0:
        return 0.0

    def diag_cost(p):
        return (p[:, 1] - p[:, 0]) / 2.0

    if len(pa) == 0:
        return float(diag_cost(pb).max())
    if len(pb) == 0:
        return float(diag_cost(pa).max())

    cost = np.maximum(np.abs(pa[:, None, 0] - pb[None, :, 0]),
                      np.abs(pa[:, None, 1] - pb[None, :, 1]))
    candidates = np.unique(np.concatenate(
        [cost.ravel(), diag_cost(pa), diag_cost(pb), [0.0]]))

    def feasible(eps: float) -> bool:
        return _has_perfect_matching(cost <= eps, diag_cost(pa) <= eps,
                                     diag_cost(pb) <= eps)

    low, high = 0, len(candidates) - 1
    while low < high:
        mid = (low + high) // 2
        if feasible(candidates[mid]):
            high = mid
        else:
            low = mid + 1
    return float(candidates[low])


def _has_perfect_matching(allowed: np.ndarray, a_to_diag: np.ndarray,
                          b_to_diag: np.ndarray) -> bool:
    """Every unmatched point must be able to go to the diagonal (Hopcroft-Karp
    style augmenting-path search on the bipartite graph of allowed pairs)."""
    n_a, n_b = allowed.shape
    match_b = [-1] * n_b

    def try_match(i: int, seen: list[bool]) -> bool:
        for j in range(n_b):
            if allowed[i, j] and not seen[j]:
                seen[j] = True
                if match_b[j] == -1 or try_match(match_b[j], seen):
                    match_b[j] = i
                    return True
        return False

    matched_a = set()
    for i in range(n_a):
        if try_match(i, [False] * n_b):
            matched_a.add(i)
    for i in range(n_a):
        if i not in matched_a and not a_to_diag[i]:
            return False
    matched_b = {j for j in range(n_b) if match_b[j] != -1}
    for j in range(n_b):
        if j not in matched_b and not b_to_diag[j]:
            return False
    return True
