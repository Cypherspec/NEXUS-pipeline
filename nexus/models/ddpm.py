"""Discrete denoising diffusion over molecular graphs.

Atom types and bond types are categorical, so the forward process uses discrete
transition matrices that interpolate towards the dataset's marginal
distribution (the D3PM / DiGress formulation) rather than Gaussian noise. A
message-passing network predicts the clean graph from the corrupted one, and
sampling walks the posterior back to t = 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nexus.models.autograd import Tensor, concat
from nexus.models.nn import (Linear, Module, MLP, cross_entropy,
                             weighted_cross_entropy)

# Atom vocabulary for generation; index 0 is the "no atom" padding symbol.
ATOM_VOCAB = ["<pad>", "C", "N", "O", "F", "S", "Cl", "Br"]
# Bond vocabulary; index 0 means "no bond".
BOND_VOCAB = ["<none>", "single", "double", "triple", "aromatic"]
BOND_ORDER_BY_INDEX = {0: 0, 1: 1, 2: 2, 3: 3, 4: 5}   # 5 = AROMATIC in nexus.chem.mol


def cosine_alpha_schedule(n_steps: int, s: float = 0.008) -> np.ndarray:
    """Cumulative signal retention alpha_bar_t, cosine schedule (Nichol & Dhariwal)."""
    t = np.linspace(0, n_steps, n_steps + 1) / n_steps
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    return np.clip(alpha_bar, 1e-5, 1.0)


@dataclass
class DiscreteDiffusion:
    """Forward corruption towards a marginal distribution.

    q(x_t | x_0) = alpha_bar_t * onehot(x_0) + (1 - alpha_bar_t) * marginal
    """
    marginal: np.ndarray
    n_steps: int = 200
    alpha_bar: np.ndarray = field(default=None)

    def __post_init__(self) -> None:
        self.marginal = np.asarray(self.marginal, dtype=float)
        self.marginal = self.marginal / self.marginal.sum()
        if self.alpha_bar is None:
            self.alpha_bar = cosine_alpha_schedule(self.n_steps)

    def q_probs(self, x0_onehot: np.ndarray, t: int) -> np.ndarray:
        a = self.alpha_bar[t]
        return a * x0_onehot + (1.0 - a) * self.marginal

    def sample_q(self, x0: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
        """Draw x_t given x_0 (integer category arrays)."""
        k = len(self.marginal)
        onehot = np.eye(k)[x0]
        probs = self.q_probs(onehot, t)
        flat = probs.reshape(-1, k)
        cumulative = np.cumsum(flat, axis=1)
        u = rng.random((len(flat), 1))
        drawn = (u > cumulative).sum(axis=1)
        return np.clip(drawn, 0, k - 1).reshape(x0.shape)

    def posterior(self, xt: np.ndarray, x0_probs: np.ndarray, t: int) -> np.ndarray:
        """q(x_{t-1} | x_t, x_0) with x_0 supplied as predicted probabilities."""
        k = len(self.marginal)
        if t <= 1:
            return x0_probs
        a_prev = self.alpha_bar[t - 1]
        a_t = self.alpha_bar[t]
        xt_onehot = np.eye(k)[xt]
        # Likelihood of the observed x_t under each candidate x_{t-1}.
        step = a_t / max(a_prev, 1e-8)
        likelihood = step * xt_onehot + (1.0 - step) * self.marginal
        prior = a_prev * x0_probs + (1.0 - a_prev) * self.marginal
        joint = likelihood * prior
        total = joint.sum(axis=-1, keepdims=True)
        return joint / np.maximum(total, 1e-12)


class GraphDenoiser(Module):
    """Predicts clean atom and bond types from a corrupted graph.

    Message passing over the dense bond tensor: each atom aggregates messages
    weighted by its bond embeddings, and bond logits are read out from the pair
    of endpoint embeddings, so atom and bond predictions stay consistent.
    """

    def __init__(self, hidden: int = 128, depth: int = 4, seed: int = 0):
        n_atom, n_bond = len(ATOM_VOCAB), len(BOND_VOCAB)
        self.hidden = hidden
        self.depth = depth
        self.atom_embed = Linear(n_atom + 1, hidden, seed=seed)       # +1 for the timestep
        self.bond_embed = Linear(n_bond + 1, hidden, seed=seed + 1)
        self.message = [Linear(2 * hidden, hidden, seed=seed + 10 + i)
                        for i in range(depth)]
        self.update = [Linear(2 * hidden, hidden, seed=seed + 30 + i)
                       for i in range(depth)]
        self.atom_out = MLP([hidden, hidden, n_atom], seed=seed + 2)
        self.bond_out = MLP([2 * hidden + hidden, hidden, n_bond], seed=seed + 3)
        self.training = True

    def forward(self, atom_types: np.ndarray, bond_types: np.ndarray,
                t_fraction: float) -> tuple[Tensor, Tensor]:
        n = len(atom_types)
        n_atom, n_bond = len(ATOM_VOCAB), len(BOND_VOCAB)
        atom_feat = np.concatenate(
            [np.eye(n_atom)[atom_types], np.full((n, 1), t_fraction)], axis=1)
        bond_feat = np.concatenate(
            [np.eye(n_bond)[bond_types], np.full((n, n, 1), t_fraction)], axis=2)

        h = self.atom_embed(Tensor(atom_feat)).relu()
        e = self.bond_embed(Tensor(bond_feat.reshape(n * n, n_bond + 1))).relu()

        for layer in range(self.depth):
            # Pairwise messages: [h_i broadcast, bond embedding] -> message
            h_rows = Tensor(np.repeat(np.arange(n), n))
            src = h[np.repeat(np.arange(n), n)]
            pair = concat([src, e], axis=1)
            messages = self.message[layer](pair).relu()
            aggregated = messages.reshape(n, n, self.hidden).sum(axis=1) * (1.0 / n)
            h = self.update[layer](concat([h, aggregated], axis=1)).relu()

        atom_logits = self.atom_out(h)
        hi = h[np.repeat(np.arange(n), n)]
        hj = h[np.tile(np.arange(n), n)]
        bond_logits = self.bond_out(concat([hi, hj, e], axis=1)).reshape(n, n, n_bond)
        return atom_logits, bond_logits


@dataclass
class MoleculeDDPM:
    """Training and sampling for the graph diffusion model."""
    n_steps: int = 100
    hidden: int = 128
    depth: int = 3
    max_atoms: int = 24
    seed: int = 0
    bond_weight: float = 1.0
    class_balance: float = 0.5

    def __post_init__(self) -> None:
        self.model = GraphDenoiser(self.hidden, self.depth, self.seed)
        self.atom_marginal = np.ones(len(ATOM_VOCAB)) / len(ATOM_VOCAB)
        self.bond_marginal = np.ones(len(BOND_VOCAB)) / len(BOND_VOCAB)
        self.size_distribution = None
        self.bond_class_weights = np.ones(len(BOND_VOCAB))

    def fit_marginals(self, graphs: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Estimate atom/bond type frequencies and the molecule-size distribution."""
        atom_counts = np.zeros(len(ATOM_VOCAB))
        bond_counts = np.zeros(len(BOND_VOCAB))
        sizes = []
        for atoms, bonds in graphs:
            for a in atoms:
                atom_counts[a] += 1
            n = len(atoms)
            for i in range(n):
                for j in range(i + 1, n):
                    bond_counts[bonds[i, j]] += 1
            sizes.append(n)
        self.atom_marginal = (atom_counts + 1.0) / (atom_counts.sum() + len(atom_counts))
        self.bond_marginal = (bond_counts + 1.0) / (bond_counts.sum() + len(bond_counts))
        self.atom_diffusion = DiscreteDiffusion(self.atom_marginal, self.n_steps)
        self.bond_diffusion = DiscreteDiffusion(self.bond_marginal, self.n_steps)
        # Bond class weights f_c^-class_balance, clipped and renormalised so the
        # mean weight under the bond marginal is 1 (the loss scale is unchanged,
        # only its balance across classes). class_balance=0 leaves the loss
        # unweighted; 1 equalises the classes outright.
        w = self.bond_marginal ** (-float(self.class_balance))
        w = np.clip(w, 0.1, 20.0)
        self.bond_class_weights = w / float(np.dot(self.bond_marginal, w))
        counts = np.bincount(sizes, minlength=self.max_atoms + 1)[: self.max_atoms + 1]
        self.size_distribution = counts / counts.sum()

    def loss(self, atoms: np.ndarray, bonds: np.ndarray, t: int,
             rng: np.random.Generator) -> Tensor:
        noisy_atoms = self.atom_diffusion.sample_q(atoms, t, rng)
        noisy_bonds = self.bond_diffusion.sample_q(bonds, t, rng)
        noisy_bonds = np.triu(noisy_bonds, 1)
        noisy_bonds = noisy_bonds + noisy_bonds.T
        atom_logits, bond_logits = self.model(noisy_atoms, noisy_bonds,
                                              t / self.n_steps)
        n = len(atoms)
        iu = np.triu_indices(n, k=1)
        atom_loss = cross_entropy(atom_logits, atoms)
        flat_logits = bond_logits.reshape(n * n, len(BOND_VOCAB))
        pair_index = iu[0] * n + iu[1]
        targets = bonds[iu]
        bond_loss = weighted_cross_entropy(flat_logits[pair_index], targets,
                                           self.bond_class_weights[targets])
        return atom_loss + self.bond_weight * bond_loss

    def sample(self, n_atoms: int | None = None, seed: int = 0,
               constrain_valence: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Generate one molecular graph by reverse diffusion.

        With `constrain_valence`, bond choices that would exceed an atom's
        valence are masked out at every reverse step.
        """
        rng = np.random.default_rng(seed)
        if n_atoms is None:
            n_atoms = int(rng.choice(len(self.size_distribution),
                                     p=self.size_distribution))
        n_atoms = max(2, min(n_atoms, self.max_atoms))

        atoms = rng.choice(len(ATOM_VOCAB), size=n_atoms, p=self.atom_marginal)
        # Reverse diffusion must start from the noise prior, i.e. the marginal,
        # in both modes. Starting the constrained path from the empty graph
        # biases every later step towards it, because the posterior's
        # likelihood term anchors x_{t-1} to x_t.
        bonds = rng.choice(len(BOND_VOCAB), size=(n_atoms, n_atoms),
                           p=self.bond_marginal)
        bonds = np.triu(bonds, 1)
        bonds = bonds + bonds.T

        for t in range(self.n_steps, 0, -1):
            atom_logits, bond_logits = self.model(atoms, bonds, t / self.n_steps)
            atom_probs = _softmax(atom_logits.data)
            bond_probs = _softmax(bond_logits.data)
            atom_post = self.atom_diffusion.posterior(atoms, atom_probs, t)
            bond_post = self.bond_diffusion.posterior(bonds, bond_probs, t)
            atoms = _categorical(atom_post, rng)
            if constrain_valence:
                # Resolve pairs one at a time in a random order, re-checking
                # headroom after each, so the running structure stays valid.
                new_bonds = np.zeros_like(bonds)
                pairs = [(i, j) for i in range(n_atoms) for j in range(i + 1, n_atoms)]
                rng.shuffle(pairs)
                for i, j in pairs:
                    allowed = valence_mask(atoms, new_bonds)[i, j]
                    probs = bond_post[i, j] * allowed
                    total = probs.sum()
                    probs = probs / total if total > 1e-12 else allowed / allowed.sum()
                    choice = int(_categorical(probs[None, :], rng)[0])
                    new_bonds[i, j] = new_bonds[j, i] = choice
                bonds = new_bonds
            else:
                bonds = _categorical(bond_post, rng)
                bonds = np.triu(bonds, 1)
                bonds = bonds + bonds.T
        return atoms, bonds


# Maximum bond-order sum per generated element, used to constrain sampling.
MAX_VALENCE_BY_ATOM = {"C": 4, "N": 3, "O": 2, "F": 1, "S": 6, "Cl": 1, "Br": 1}
BOND_ORDER_VALUE = {0: 0.0, 1: 1.0, 2: 2.0, 3: 3.0, 4: 1.5}


def valence_mask(atoms: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    """Which bond types each pair may still take without exceeding valence.

    Returns a boolean (n, n, n_bond_types) mask. Sampling inside this mask is
    what makes generated graphs chemically well-formed; without it almost every
    sample is over-valent, because the bond marginal puts far too much mass on
    "bonded" for a sparse molecular graph.
    """
    n = len(atoms)
    used = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i != j:
                used[i] += BOND_ORDER_VALUE[int(bonds[i, j])]
    capacity = np.array([MAX_VALENCE_BY_ATOM.get(ATOM_VOCAB[a], 0) for a in atoms],
                        dtype=float)
    mask = np.ones((n, n, len(BOND_VOCAB)), dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                mask[i, j, 1:] = False          # no self bonds
                continue
            current = BOND_ORDER_VALUE[int(bonds[i, j])]
            headroom_i = capacity[i] - (used[i] - current)
            headroom_j = capacity[j] - (used[j] - current)
            headroom = min(headroom_i, headroom_j)
            for b in range(1, len(BOND_VOCAB)):
                if BOND_ORDER_VALUE[b] > headroom + 1e-9:
                    mask[i, j, b] = False
    # Padding atoms can carry no bonds at all.
    for i in range(n):
        if atoms[i] == 0:
            mask[i, :, 1:] = False
            mask[:, i, 1:] = False
    return mask


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def _categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shape = probs.shape[:-1]
    flat = probs.reshape(-1, probs.shape[-1])
    cumulative = np.cumsum(flat, axis=1)
    u = rng.random((len(flat), 1))
    return (u > cumulative).sum(axis=1).reshape(shape)


# ----------------------------------------------------- molecule <-> graph

def molecule_to_graph(mol) -> tuple[np.ndarray, np.ndarray] | None:
    """Encode a Molecule as (atom type indices, bond type matrix), or None if
    it uses atoms outside the generation vocabulary."""
    from nexus.chem.mol import AROMATIC, DOUBLE, SINGLE, TRIPLE

    n = mol.n_atoms
    atoms = np.zeros(n, dtype=int)
    for i, atom in enumerate(mol.atoms):
        if atom.symbol not in ATOM_VOCAB:
            return None
        atoms[i] = ATOM_VOCAB.index(atom.symbol)
    bonds = np.zeros((n, n), dtype=int)
    order_map = {SINGLE: 1, DOUBLE: 2, TRIPLE: 3, AROMATIC: 4}
    for bond in mol.bonds:
        code = order_map.get(bond.order, 1)
        bonds[bond.a, bond.b] = bonds[bond.b, bond.a] = code
    return atoms, bonds


def graph_to_molecule(atoms: np.ndarray, bonds: np.ndarray):
    """Decode a generated graph into a Molecule, dropping padding atoms."""
    from nexus.chem.mol import Atom, Bond, Molecule

    keep = [i for i, a in enumerate(atoms) if a != 0]
    if not keep:
        return None
    remap = {old: new for new, old in enumerate(keep)}
    # An atom carrying an aromatic-coded bond must be flagged aromatic, exactly
    # as the SMILES parser does for lowercase atoms; otherwise perception sees
    # aromatic bonds between non-aromatic atoms and the ring decodes as
    # aliphatic.
    aromatic_atoms = {i for i in keep
                      if any(bonds[i, j] == 4 for j in keep if j != i)}
    mol = Molecule(atoms=[Atom(symbol=ATOM_VOCAB[atoms[i]],
                               aromatic=i in aromatic_atoms) for i in keep])
    for i in keep:
        for j in keep:
            if i >= j or bonds[i, j] == 0:
                continue
            mol.bonds.append(Bond(remap[i], remap[j],
                                  BOND_ORDER_BY_INDEX[int(bonds[i, j])]))
    try:
        return mol.perceive()
    except Exception:
        return None
