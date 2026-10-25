"""Featurisation shared by every ADMET endpoint.

One vector per molecule, assembled from three views that fail in different
ways, which is why all three are kept rather than picking a favourite:

    Morgan counts (2048)   circular substructure environments; the workhorse
                           for structure-activity, blind to global shape
    atom pairs (1024)      element pairs at a topological distance; carries
                           the long-range layout Morgan misses
    descriptors (24)       size, polarity, lipophilicity, flexibility -- the
                           physical properties most ADMET rules are stated in

Counts are log-compressed rather than binarised: a molecule with eight amide
bonds differs from one with one, but not eightfold, and log1p says that without
letting a repeated motif dominate the vector.

The descriptor block is standardised against statistics computed once on the
training set and shipped with the model, so a molecule scored in the panel is
transformed exactly as the training molecules were.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nexus.chem.descriptors import DESCRIPTOR_NAMES, compute_descriptors
from nexus.chem.fingerprint import atom_pair_fingerprint, morgan_fingerprint
from nexus.chem.mol import Molecule

MORGAN_BITS = 2048
MORGAN_RADIUS = 2
PAIR_BITS = 1024
N_DESCRIPTORS = len(DESCRIPTOR_NAMES)
N_FEATURES = MORGAN_BITS + PAIR_BITS + N_DESCRIPTORS


@dataclass
class FeatureScaler:
    """Mean and scale for the descriptor block, fitted on training molecules."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, descriptors: np.ndarray) -> "FeatureScaler":
        mean = descriptors.mean(axis=0)
        scale = descriptors.std(axis=0)
        # A descriptor that never varies carries no information; dividing by
        # its zero spread would turn it into infinities instead.
        scale[scale < 1e-9] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, descriptors: np.ndarray) -> np.ndarray:
        # Standardised, then clipped: a descriptor twenty deviations out is a
        # parse artefact or an extreme outlier, and either way it should not
        # dominate the first layer's activations.
        return np.clip((descriptors - self.mean) / self.scale, -10.0, 10.0)


def descriptor_block(mols: list[Molecule]) -> np.ndarray:
    """The raw, unscaled descriptor matrix."""
    rows = np.zeros((len(mols), N_DESCRIPTORS), dtype=np.float64)
    for i, mol in enumerate(mols):
        values = compute_descriptors(mol)
        rows[i] = [values[name] for name in DESCRIPTOR_NAMES]
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)


def fingerprint_block(mols: list[Molecule]) -> np.ndarray:
    """Morgan counts and atom-pair counts, log-compressed."""
    rows = np.zeros((len(mols), MORGAN_BITS + PAIR_BITS), dtype=np.float64)
    for i, mol in enumerate(mols):
        morgan = morgan_fingerprint(mol, n_bits=MORGAN_BITS, radius=MORGAN_RADIUS,
                                    counts=True)
        pairs = atom_pair_fingerprint(mol, n_bits=PAIR_BITS, counts=True)
        rows[i, :MORGAN_BITS] = morgan
        rows[i, MORGAN_BITS:] = pairs
    return np.log1p(rows)


def featurize(mols: list[Molecule], scaler: FeatureScaler) -> np.ndarray:
    """The full matrix a model consumes."""
    return np.hstack([fingerprint_block(mols),
                      scaler.transform(descriptor_block(mols))])
