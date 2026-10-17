"""Directed message-passing neural network for molecular property prediction.

Follows the bond-centred formulation of Yang et al. (2019) ("Analyzing Learned
Molecular Representations for Property Prediction"): messages live on directed
bonds and exclude the reverse bond, which prevents a message from immediately
flowing back where it came from.
"""

from __future__ import annotations

import numpy as np

from nexus.models.autograd import Tensor, concat, segment_max, segment_mean
from nexus.models.featurize import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    MolGraph,
    featurize_batch,
)
from nexus.models.nn import Dropout, Linear, Module, MLP


class DMPNN(Module):
    """Directed MPNN encoder plus a feed-forward readout head.

    Parameters
    ----------
    hidden : message dimension
    depth : number of message-passing steps
    n_tasks : outputs per molecule
    readout : "sum", "mean" or "max" pooling over atoms
    extra_features : width of optional per-molecule descriptors concatenated
        before the head (0 to disable)
    """

    def __init__(self, hidden: int = 300, depth: int = 3, n_tasks: int = 1,
                 dropout: float = 0.0, readout: str = "mean", ffn_layers: int = 2,
                 extra_features: int = 0, seed: int = 0):
        self.hidden = hidden
        self.depth = depth
        self.readout = readout
        self.extra_features = extra_features
        self.W_input = Linear(ATOM_FEATURE_DIM + BOND_FEATURE_DIM, hidden,
                              bias=False, seed=seed)
        self.W_message = Linear(hidden, hidden, bias=False, seed=seed + 1)
        self.W_atom = Linear(ATOM_FEATURE_DIM + hidden, hidden, seed=seed + 2)
        self.dropout = Dropout(dropout, seed=seed + 3)
        sizes = [hidden + extra_features] + [hidden] * (ffn_layers - 1) + [n_tasks]
        self.head = MLP(sizes, dropout=dropout, seed=seed + 4)
        self.training = True

    def encode(self, graph: MolGraph) -> Tensor:
        """Per-molecule embeddings of shape (n_molecules, hidden)."""
        atoms = Tensor(graph.atom_features)
        if graph.n_bonds == 0:
            atom_hidden = self.W_atom(
                concat([atoms, Tensor(np.zeros((graph.n_atoms, self.hidden)))], axis=1)
            ).relu()
            return self._pool(atom_hidden, graph)

        src_atoms = atoms[graph.bond_src]
        bonds = Tensor(graph.bond_features)
        h0 = self.W_input(concat([src_atoms, bonds], axis=1)).relu()
        h = h0
        for _ in range(self.depth - 1):
            # Sum incoming messages at each atom, then subtract the reverse bond
            # so a message never returns along the edge it arrived on.
            atom_sum = h.index_add(graph.bond_dst, graph.n_atoms)
            message = atom_sum[graph.bond_src] - h[graph.reverse_bond]
            h = (h0 + self.W_message(message)).relu()
            h = self.dropout(h)

        atom_message = h.index_add(graph.bond_dst, graph.n_atoms)
        atom_hidden = self.W_atom(concat([atoms, atom_message], axis=1)).relu()
        atom_hidden = self.dropout(atom_hidden)
        return self._pool(atom_hidden, graph)

    def _pool(self, atom_hidden: Tensor, graph: MolGraph) -> Tensor:
        if self.readout == "sum":
            return atom_hidden.index_add(graph.atom_to_mol, graph.n_molecules)
        if self.readout == "max":
            return segment_max(atom_hidden, graph.atom_to_mol, graph.n_molecules)
        return segment_mean(atom_hidden, graph.atom_to_mol, graph.n_molecules)

    def forward(self, graph: MolGraph, extra: np.ndarray | None = None) -> Tensor:
        embedding = self.encode(graph)
        if self.extra_features and extra is not None:
            embedding = concat([embedding, Tensor(np.asarray(extra, dtype=np.float64))],
                               axis=1)
        return self.head(embedding)


def batches(n: int, batch_size: int, shuffle: bool = True, seed: int = 0):
    idx = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, n, batch_size):
        yield idx[start:start + batch_size]
