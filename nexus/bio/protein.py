"""Protein sequence featurization for drug-target affinity models."""

from __future__ import annotations

import numpy as np

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AMINO_ACIDS)}

# Side-chain physicochemical properties: hydropathy (Kyte-Doolittle), volume
# (A^3), charge at pH 7, polarity, aromaticity.
AA_PROPERTIES = {
    "A": (1.8, 88.6, 0, 0, 0), "C": (2.5, 108.5, 0, 0, 0),
    "D": (-3.5, 111.1, -1, 1, 0), "E": (-3.5, 138.4, -1, 1, 0),
    "F": (2.8, 189.9, 0, 0, 1), "G": (-0.4, 60.1, 0, 0, 0),
    "H": (-3.2, 153.2, 0.1, 1, 1), "I": (4.5, 166.7, 0, 0, 0),
    "K": (-3.9, 168.6, 1, 1, 0), "L": (3.8, 166.7, 0, 0, 0),
    "M": (1.9, 162.9, 0, 0, 0), "N": (-3.5, 114.1, 0, 1, 0),
    "P": (-1.6, 112.7, 0, 0, 0), "Q": (-3.5, 143.8, 0, 1, 0),
    "R": (-4.5, 173.4, 1, 1, 0), "S": (-0.8, 89.0, 0, 1, 0),
    "T": (-0.7, 116.1, 0, 1, 0), "V": (4.2, 140.0, 0, 0, 0),
    "W": (-0.9, 227.8, 0, 0, 1), "Y": (-1.3, 193.6, 0, 1, 1),
}


def amino_acid_composition(sequence: str) -> np.ndarray:
    """Fraction of each of the 20 standard amino acids."""
    counts = np.zeros(len(AMINO_ACIDS))
    for residue in sequence.upper():
        if residue in AA_INDEX:
            counts[AA_INDEX[residue]] += 1
    total = counts.sum()
    return counts / total if total > 0 else counts


def kmer_counts(sequence: str, k: int = 3, n_bits: int = 2048,
                normalize: bool = True) -> np.ndarray:
    """Hashed k-mer spectrum of a sequence."""
    from nexus.chem.fingerprint import mix32

    vec = np.zeros(n_bits)
    sequence = sequence.upper()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if any(r not in AA_INDEX for r in kmer):
            continue
        code = 0
        for r in kmer:
            code = code * 20 + AA_INDEX[r]
        vec[mix32(code, k) % n_bits] += 1
    if normalize and vec.sum() > 0:
        vec /= vec.sum()
    return vec


def physicochemical_profile(sequence: str) -> np.ndarray:
    """Mean, standard deviation and terminal values of residue properties."""
    rows = [AA_PROPERTIES[r] for r in sequence.upper() if r in AA_PROPERTIES]
    if not rows:
        return np.zeros(5 * 4)
    arr = np.array(rows, dtype=float)
    return np.concatenate([arr.mean(0), arr.std(0), arr[:20].mean(0), arr[-20:].mean(0)])


def protein_features(sequence: str, k: int = 3, n_bits: int = 1024) -> np.ndarray:
    """Composition, hashed k-mers, physicochemical profile and length."""
    return np.concatenate([
        amino_acid_composition(sequence),
        kmer_counts(sequence, k=k, n_bits=n_bits),
        physicochemical_profile(sequence),
        [min(len(sequence), 4000) / 4000.0],
    ])


def protein_feature_matrix(sequences, k: int = 3, n_bits: int = 1024) -> np.ndarray:
    return np.stack([protein_features(s, k, n_bits) for s in sequences])
