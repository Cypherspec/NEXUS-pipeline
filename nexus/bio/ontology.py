"""The Gene Ontology as a navigable DAG with information content and
semantic similarity.

Implements the true-path rule (a gene annotated to a term is implicitly
annotated to every ancestor of that term), Resnik/Lin/Jiang-Conrath
information-content similarity, and Wang's graph-based similarity.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nexus.bio.gaf import Annotation, parse_gaf
from nexus.bio.obo import OboTerm, parse_obo

ROOTS = {
    "biological_process": "GO:0008150",
    "molecular_function": "GO:0003674",
    "cellular_component": "GO:0005575",
}
# Wang et al. (2007) semantic contribution factors per relation type.
WANG_WEIGHTS = {"is_a": 0.8, "part_of": 0.6, "regulates": 0.5}


@dataclass
class GeneOntology:
    terms: dict[str, OboTerm]
    header: dict[str, str]

    def __post_init__(self) -> None:
        self.alias: dict[str, str] = {}
        for tid, term in self.terms.items():
            for alt in term.alt_ids:
                self.alias[alt] = tid
            if term.is_obsolete and term.replaced_by:
                self.alias[tid] = term.replaced_by
        self.parents: dict[str, dict[str, str]] = {}
        self.children: dict[str, set[str]] = defaultdict(set)
        for tid, term in self.terms.items():
            rel: dict[str, str] = {}
            for p in term.is_a:
                rel[p] = "is_a"
            for p in term.part_of:
                rel.setdefault(p, "part_of")
            for p in term.regulates:
                rel.setdefault(p, "regulates")
            rel = {p: k for p, k in rel.items() if p in self.terms}
            self.parents[tid] = rel
            for p in rel:
                self.children[p].add(tid)
        self._ancestors: dict[str, frozenset[str]] = {}
        self._depth: dict[str, int] = {}
        self.ic: dict[str, float] = {}
        self._term_genes: dict[str, set[str]] = {}
        self._n_annotated_genes = 0

    # ----------------------------------------------------------------- loading

    @classmethod
    def load(cls, obo_path: str | Path) -> "GeneOntology":
        terms, header = parse_obo(obo_path)
        return cls(terms=terms, header=header)

    def resolve(self, go_id: str) -> str | None:
        """Map an alternative or obsolete id to its current primary id."""
        seen = set()
        while go_id in self.alias and go_id not in seen:
            seen.add(go_id)
            go_id = self.alias[go_id]
        return go_id if go_id in self.terms else None

    def __contains__(self, go_id: str) -> bool:
        return go_id in self.terms

    def __len__(self) -> int:
        return len(self.terms)

    def name(self, go_id: str) -> str:
        term = self.terms.get(go_id)
        return term.name if term else ""

    def namespace(self, go_id: str) -> str:
        term = self.terms.get(go_id)
        return term.namespace if term else ""

    # -------------------------------------------------------------- traversal

    def topological_order(self) -> list[str]:
        """Terms ordered leaves first, every term before its parents."""
        indegree = {t: 0 for t in self.terms}
        for t in self.terms:
            for p in self.parents[t]:
                indegree[p] += 1
        queue = deque(t for t, d in indegree.items() if d == 0)
        order = []
        while queue:
            t = queue.popleft()
            order.append(t)
            for p in self.parents[t]:
                indegree[p] -= 1
                if indegree[p] == 0:
                    queue.append(p)
        if len(order) != len(self.terms):      # cycle: fall back to insertion order
            order += [t for t in self.terms if t not in set(order)]
        return order

    def ancestors(self, go_id: str, include_self: bool = True) -> frozenset[str]:
        """All terms reachable by following is_a/part_of/regulates upward."""
        if go_id not in self.terms:
            return frozenset()
        if not self._ancestors:
            self._build_ancestors()
        anc = self._ancestors.get(go_id, frozenset())
        return anc if include_self else anc - {go_id}

    def _build_ancestors(self) -> None:
        # Roots first, so every parent's ancestor set is complete before its
        # children are processed.
        for t in reversed(self.topological_order()):
            acc = {t}
            for p in self.parents[t]:
                acc |= self._ancestors.get(p, frozenset({p}))
            self._ancestors[t] = frozenset(acc)

    def descendants(self, go_id: str, include_self: bool = True) -> set[str]:
        out: set[str] = set()
        stack = [go_id]
        while stack:
            t = stack.pop()
            for c in self.children.get(t, ()):
                if c not in out:
                    out.add(c)
                    stack.append(c)
        if include_self:
            out.add(go_id)
        return out

    def depth(self, go_id: str) -> int:
        """Length of the shortest path to a namespace root."""
        if go_id in self._depth:
            return self._depth[go_id]
        roots = set(ROOTS.values())
        dist = {go_id: 0}
        queue = deque([go_id])
        while queue:
            t = queue.popleft()
            if t in roots:
                self._depth[go_id] = dist[t]
                return dist[t]
            for p in self.parents.get(t, {}):
                if p not in dist:
                    dist[p] = dist[t] + 1
                    queue.append(p)
        self._depth[go_id] = 0
        return 0

    # --------------------------------------------------- annotation statistics

    def annotate(self, annotations: list[Annotation], aspect: str | None = None,
                 propagate: bool = True) -> None:
        """Attach a gene annotation corpus and compute information content.

        With `propagate=True` the true-path rule is applied: each gene is
        counted for every ancestor of the terms it is annotated to.
        """
        if not self._ancestors:
            self._build_ancestors()
        term_genes: dict[str, set[str]] = defaultdict(set)
        genes: set[str] = set()
        for a in annotations:
            if aspect and a.aspect != aspect:
                continue
            tid = self.resolve(a.go_id)
            if tid is None:
                continue
            genes.add(a.gene)
            targets = self._ancestors[tid] if propagate else {tid}
            for t in targets:
                term_genes[t].add(a.gene)
        self._term_genes = dict(term_genes)
        self._n_annotated_genes = len(genes)

        self.ic = {}
        total = float(len(genes)) or 1.0
        for t, g in self._term_genes.items():
            p = len(g) / total
            self.ic[t] = -math.log(p) if p > 0 else 0.0

    @property
    def n_annotated_genes(self) -> int:
        return self._n_annotated_genes

    def genes_for(self, go_id: str) -> set[str]:
        return self._term_genes.get(go_id, set())

    def term_size(self, go_id: str) -> int:
        return len(self._term_genes.get(go_id, ()))

    def information_content(self, go_id: str) -> float:
        return self.ic.get(go_id, 0.0)

    # ------------------------------------------------------ semantic similarity

    def common_ancestors(self, a: str, b: str) -> frozenset[str]:
        return self.ancestors(a) & self.ancestors(b)

    def mica(self, a: str, b: str) -> tuple[str | None, float]:
        """Most informative common ancestor and its information content."""
        shared = self.common_ancestors(a, b)
        if not shared:
            return None, 0.0
        best, best_ic = None, -1.0
        for t in shared:
            v = self.ic.get(t, 0.0)
            if v > best_ic:
                best, best_ic = t, v
        return best, max(best_ic, 0.0)

    def similarity(self, a: str, b: str, method: str = "lin") -> float:
        """Semantic similarity between two GO terms.

        resnik: information content of the most informative common ancestor
                (unbounded, in nats).
        lin:    2*IC(MICA) / (IC(a) + IC(b)), in [0, 1].
        jiang:  1 / (1 + IC(a) + IC(b) - 2*IC(MICA)), in (0, 1].
        wang:   graph-based semantic value (Wang 2007), in [0, 1].
        """
        a, b = self.resolve(a), self.resolve(b)
        if a is None or b is None:
            return 0.0
        if method == "wang":
            return self._wang_similarity(a, b)
        if a == b and method in ("lin", "jiang"):
            return 1.0
        _, ic_mica = self.mica(a, b)
        ic_a, ic_b = self.ic.get(a, 0.0), self.ic.get(b, 0.0)
        if method == "resnik":
            return ic_mica
        if method == "lin":
            denom = ic_a + ic_b
            return (2 * ic_mica / denom) if denom > 0 else 0.0
        if method == "jiang":
            return 1.0 / (1.0 + max(ic_a + ic_b - 2 * ic_mica, 0.0))
        raise ValueError(f"unknown method {method!r}")

    def _semantic_values(self, go_id: str) -> dict[str, float]:
        """Wang semantic contribution of each ancestor to `go_id`."""
        values = {go_id: 1.0}
        queue = deque([go_id])
        while queue:
            t = queue.popleft()
            for p, rel in self.parents.get(t, {}).items():
                w = WANG_WEIGHTS.get(rel, 0.5)
                candidate = values[t] * w
                if candidate > values.get(p, 0.0):
                    values[p] = candidate
                    queue.append(p)
        return values

    def _wang_similarity(self, a: str, b: str) -> float:
        sa, sb = self._semantic_values(a), self._semantic_values(b)
        shared = set(sa) & set(sb)
        if not shared:
            return 0.0
        num = sum(sa[t] + sb[t] for t in shared)
        den = sum(sa.values()) + sum(sb.values())
        return num / den if den > 0 else 0.0

    def gene_similarity(self, terms_a, terms_b, method: str = "lin",
                        combine: str = "bma") -> float:
        """Similarity between two gene products from their GO term sets.

        combine: "bma" (best-match average, the usual choice), "max", or "avg".
        """
        ta = [t for t in (self.resolve(x) for x in terms_a) if t]
        tb = [t for t in (self.resolve(x) for x in terms_b) if t]
        if not ta or not tb:
            return 0.0
        M = np.array([[self.similarity(x, y, method) for y in tb] for x in ta])
        if combine == "max":
            return float(M.max())
        if combine == "avg":
            return float(M.mean())
        return float((M.max(axis=1).mean() + M.max(axis=0).mean()) / 2)


def load_human_ontology(obo_path: str | Path, gaf_path: str | Path,
                        aspect: str | None = None,
                        exclude_electronic: bool = False) -> GeneOntology:
    """Convenience loader: ontology plus human annotations, ready for use."""
    go = GeneOntology.load(obo_path)
    annotations = parse_gaf(gaf_path, exclude_electronic=exclude_electronic)
    go.annotate(annotations, aspect=aspect)
    return go
