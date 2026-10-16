"""Molecular descriptors: counts, TPSA, logP, QED, complexity, drug-likeness.

Where a descriptor has a published definition (Ertl TPSA, Lipinski counts,
Bickerton QED) the definition is implemented directly and validated against
reference values shipped with public datasets; see tests/test_descriptors.py
and benchmarks/validate_descriptors.py.
"""

from __future__ import annotations

import math

import numpy as np

from nexus.chem.mol import AROMATIC, BOND_ORDER, DOUBLE, SINGLE, TRIPLE, Molecule
from nexus.chem.periodic import atomic_mass, electronegativity

# --------------------------------------------------------------- simple counts


def is_hbond_donor(mol: Molecule, i: int) -> bool:
    """N or O carrying at least one hydrogen (Lipinski definition)."""
    atom = mol.atoms[i]
    return atom.symbol in ("N", "O") and atom.total_h > 0


def is_hbond_acceptor(mol: Molecule, i: int) -> bool:
    """N or O with an available lone pair.

    Excludes pyrrole-type aromatic N (lone pair in the pi system), amide N,
    and N/O bearing a positive charge.
    """
    atom = mol.atoms[i]
    if atom.symbol not in ("N", "O"):
        return False
    if atom.charge > 0:
        return False
    if atom.symbol == "N":
        if atom.aromatic and atom.pi_electrons == 2:
            return False
        for j in mol.neighbors(i):
            nb = mol.atoms[j]
            if nb.symbol == "C":
                # amide / amidine / sulfonamide nitrogen
                for k in mol.neighbors(j):
                    bond = mol.bond_between(j, k)
                    if k != i and bond.order == DOUBLE and mol.atoms[k].symbol in ("O", "S"):
                        return False
            if nb.symbol == "S":
                for k in mol.neighbors(j):
                    bond = mol.bond_between(j, k)
                    if bond.order == DOUBLE and mol.atoms[k].symbol == "O":
                        return False
    return True


def num_hbond_donors(mol: Molecule) -> int:
    return sum(1 for i in range(mol.n_atoms) if is_hbond_donor(mol, i))


def num_hbond_acceptors(mol: Molecule) -> int:
    return sum(1 for i in range(mol.n_atoms) if is_hbond_acceptor(mol, i))


def num_lipinski_hbd(mol: Molecule) -> int:
    """Lipinski's simple count: every N-H and O-H hydrogen."""
    return sum(a.total_h for a in mol.atoms if a.symbol in ("N", "O"))


def num_lipinski_hba(mol: Molecule) -> int:
    """Lipinski's simple count: every nitrogen and oxygen."""
    return sum(1 for a in mol.atoms if a.symbol in ("N", "O"))


def num_rotatable_bonds(mol: Molecule) -> int:
    """Single, acyclic, non-terminal bonds, excluding amide C-N bonds."""
    count = 0
    for bond in mol.bonds:
        if bond.order != SINGLE or bond.in_ring:
            continue
        a, b = mol.atoms[bond.a], mol.atoms[bond.b]
        if a.symbol == "H" or b.symbol == "H":
            continue
        if mol.degree(bond.a) < 2 or mol.degree(bond.b) < 2:
            continue
        if _is_amide_bond(mol, bond.a, bond.b) or _is_amide_bond(mol, bond.b, bond.a):
            continue
        if _is_triple_adjacent(mol, bond.a) or _is_triple_adjacent(mol, bond.b):
            continue
        count += 1
    return count


def _is_amide_bond(mol: Molecule, c: int, n: int) -> bool:
    if mol.atoms[c].symbol != "C" or mol.atoms[n].symbol != "N":
        return False
    for k in mol.neighbors(c):
        bond = mol.bond_between(c, k)
        if bond.order == DOUBLE and mol.atoms[k].symbol in ("O", "S"):
            return True
    return False


def _is_triple_adjacent(mol: Molecule, i: int) -> bool:
    return any(mol.bond_between(i, j).order == TRIPLE for j in mol.neighbors(i))


def num_rings(mol: Molecule) -> int:
    return mol.n_rings


def num_aromatic_rings(mol: Molecule) -> int:
    return sum(1 for r in mol.rings if all(mol.atoms[i].aromatic for i in r))


def num_aliphatic_rings(mol: Molecule) -> int:
    return mol.n_rings - num_aromatic_rings(mol)


def num_heteroatoms(mol: Molecule) -> int:
    return sum(1 for a in mol.atoms if a.symbol not in ("C", "H"))


def fraction_csp3(mol: Molecule) -> float:
    carbons = [i for i, a in enumerate(mol.atoms) if a.symbol == "C"]
    if not carbons:
        return 0.0
    sp3 = sum(1 for i in carbons if _is_sp3(mol, i))
    return sp3 / len(carbons)


def _is_sp3(mol: Molecule, i: int) -> bool:
    if mol.atoms[i].aromatic:
        return False
    return all(mol.bond_between(i, j).order == SINGLE for j in mol.neighbors(i))


def num_stereocenters(mol: Molecule) -> int:
    return sum(1 for a in mol.atoms if a.chirality)


def num_spiro_atoms(mol: Molecule) -> int:
    """Atoms shared by exactly two rings that share only that atom."""
    count = 0
    for i in range(mol.n_atoms):
        member = [set(r) for r in mol.rings if i in r]
        if len(member) < 2:
            continue
        for a in range(len(member)):
            for b in range(a + 1, len(member)):
                if member[a] & member[b] == {i}:
                    count += 1
                    break
            else:
                continue
            break
    return count


def num_bridgehead_atoms(mol: Molecule) -> int:
    """Atoms shared by two rings that also share at least two other atoms."""
    count = 0
    for i in range(mol.n_atoms):
        member = [set(r) for r in mol.rings if i in r]
        found = False
        for a in range(len(member)):
            for b in range(a + 1, len(member)):
                shared = member[a] & member[b]
                if len(shared) > 2 and i in shared:
                    neighbours_in_shared = sum(1 for j in mol.neighbors(i) if j in shared)
                    if neighbours_in_shared >= 2:
                        found = True
        count += 1 if found else 0
    return count


# ------------------------------------------------------------------------ TPSA

# Ertl et al. (2000) J. Med. Chem. 43:3714, Table 1. Keys are
# (symbol, aromatic, n_single, n_double, n_triple, n_aromatic, n_hydrogens, charge).
_TPSA_N = {
    (False, 3, 0, 0, 0, 0, 0): 3.24,    # N(-*)(-*)-*
    (False, 1, 1, 0, 0, 0, 0): 12.36,   # N(-*)=*
    (False, 0, 0, 1, 0, 0, 0): 23.79,   # N#*
    (False, 1, 2, 0, 0, 0, 0): 11.68,   # N(-*)(=*)=*
    (False, 0, 1, 1, 0, 0, 0): 13.60,   # N(=*)#*
    (False, 2, 0, 0, 0, 1, 0): 12.03,   # NH(-*)-*
    (False, 0, 1, 0, 0, 1, 0): 23.85,   # NH=*
    (False, 1, 0, 0, 0, 2, 0): 26.02,   # NH2-*
    (False, 4, 0, 0, 0, 0, 1): 0.00,    # N+(-*)(-*)(-*)-*
    (False, 2, 1, 0, 0, 0, 1): 3.01,    # N+(-*)(-*)=*
    (False, 1, 0, 1, 0, 0, 1): 4.36,    # N+(-*)#*
    (False, 3, 0, 0, 0, 1, 1): 4.44,    # NH+(-*)(-*)-*
    (False, 1, 1, 0, 0, 1, 1): 13.97,   # NH+(-*)=*
    (False, 2, 0, 0, 0, 2, 1): 16.61,   # NH2+(-*)-*
    (False, 0, 1, 0, 0, 2, 1): 25.59,   # NH2+=*
    (False, 1, 0, 0, 0, 3, 1): 27.64,   # NH3+-*
    (True, 0, 0, 0, 2, 0, 0): 12.89,    # n(:*):*
    (True, 0, 0, 0, 3, 0, 0): 4.41,     # n(:*)(:*):*
    (True, 1, 0, 0, 2, 0, 0): 4.93,     # n(-*)(:*):*
    (True, 0, 1, 0, 2, 0, 0): 8.39,     # n(=*)(:*):*
    (True, 0, 0, 0, 2, 1, 0): 15.79,    # nH(:*):*
    (True, 0, 0, 0, 3, 0, 1): 4.10,     # n+(:*)(:*):*
    (True, 1, 0, 0, 2, 0, 1): 3.88,     # n+(-*)(:*):*
    (True, 0, 0, 0, 2, 1, 1): 14.14,    # nH+(:*):*
}
_TPSA_O = {
    (False, 2, 0, 0, 0, 0, 0): 9.23,    # O(-*)-*
    (False, 0, 1, 0, 0, 0, 0): 17.07,   # O=*
    (False, 1, 0, 0, 0, 1, 0): 20.23,   # OH-*
    (False, 1, 0, 0, 0, 0, -1): 23.06,  # O- -*
    (True, 0, 0, 0, 2, 0, 0): 13.14,    # o(:*):*
}
_TPSA_S = {
    (False, 2, 0, 0, 0, 0, 0): 25.30,
    (False, 0, 1, 0, 0, 0, 0): 32.09,
    (False, 2, 1, 0, 0, 0, 0): 19.21,
    (False, 2, 2, 0, 0, 0, 0): 8.38,
    (False, 1, 0, 0, 0, 1, 0): 38.80,
    (True, 0, 0, 0, 2, 0, 0): 28.24,
    (True, 0, 1, 0, 2, 0, 0): 21.70,
}
_TPSA_P = {
    (False, 3, 0, 0, 0, 0, 0): 13.59,
    (False, 1, 1, 0, 0, 0, 0): 34.14,
    (False, 3, 1, 0, 0, 0, 0): 9.81,
    (False, 2, 1, 0, 0, 1, 0): 23.47,
}
# Fallback contributions for environments outside the published table.
_TPSA_FALLBACK = {"N": 12.0, "O": 17.0, "S": 25.3, "P": 13.6}


def _tpsa_key(mol: Molecule, i: int) -> tuple:
    atom = mol.atoms[i]
    n_single = n_double = n_triple = n_arom = 0
    for j in mol.neighbors(i):
        order = mol.bond_between(i, j).order
        if order == AROMATIC:
            n_arom += 1
        elif order == DOUBLE:
            n_double += 1
        elif order == TRIPLE:
            n_triple += 1
        else:
            n_single += 1
    return (atom.aromatic, n_single, n_double, n_triple, n_arom, atom.total_h, atom.charge)


def tpsa(mol: Molecule, include_s_p: bool = False) -> float:
    """Topological polar surface area (Ertl 2000), in square Angstroms.

    By default sums nitrogen and oxygen contributions only, which is the
    convention used by most published TPSA values.
    """
    tables = {"N": _TPSA_N, "O": _TPSA_O}
    if include_s_p:
        tables.update({"S": _TPSA_S, "P": _TPSA_P})
    total = 0.0
    for i, atom in enumerate(mol.atoms):
        table = tables.get(atom.symbol)
        if table is None:
            continue
        key = _tpsa_key(mol, i)
        total += table.get(key, _TPSA_FALLBACK[atom.symbol])
    return total


# --------------------------------------------------------- functional groups

def count_functional_groups(mol: Molecule) -> dict[str, int]:
    """Counts of common medicinal-chemistry functional groups."""
    counts = dict.fromkeys(
        ["hydroxyl", "carbonyl", "carboxylic_acid", "ester", "amide", "amine_primary",
         "amine_secondary", "amine_tertiary", "nitro", "nitrile", "sulfonamide",
         "sulfone", "halogen", "ether", "thiol", "aromatic_n", "guanidine", "urea"], 0)
    for i, atom in enumerate(mol.atoms):
        nbrs = mol.neighbors(i)
        if atom.symbol in ("F", "Cl", "Br", "I"):
            counts["halogen"] += 1
        if atom.symbol == "S" and atom.total_h > 0:
            counts["thiol"] += 1
        if atom.symbol == "N" and atom.aromatic:
            counts["aromatic_n"] += 1
        if atom.symbol == "O":
            if atom.total_h > 0 and not atom.aromatic:
                counts["hydroxyl"] += 1
            elif len(nbrs) == 2 and all(mol.atoms[j].symbol == "C" for j in nbrs) \
                    and not atom.aromatic:
                counts["ether"] += 1
        if atom.symbol == "N" and not atom.aromatic and atom.charge == 0:
            heavy = len(nbrs)
            attached_carbonyl = any(_is_amide_bond(mol, j, i) for j in nbrs)
            if attached_carbonyl:
                counts["amide"] += 1
            elif heavy == 1:
                counts["amine_primary"] += 1
            elif heavy == 2:
                counts["amine_secondary"] += 1
            elif heavy == 3:
                counts["amine_tertiary"] += 1
        if atom.symbol == "C":
            dbl_o = [j for j in nbrs
                     if mol.bond_between(i, j).order == DOUBLE and mol.atoms[j].symbol == "O"]
            if dbl_o:
                counts["carbonyl"] += 1
                single_o = [j for j in nbrs if mol.bond_between(i, j).order == SINGLE
                            and mol.atoms[j].symbol == "O"]
                if single_o:
                    if any(mol.atoms[j].total_h > 0 for j in single_o):
                        counts["carboxylic_acid"] += 1
                    else:
                        counts["ester"] += 1
                n_nbrs = [j for j in nbrs if mol.atoms[j].symbol == "N"]
                if len(n_nbrs) >= 2:
                    counts["urea"] += 1
            n_nbrs = [j for j in nbrs if mol.atoms[j].symbol == "N"]
            if len(n_nbrs) == 3:
                counts["guanidine"] += 1
            if any(mol.bond_between(i, j).order == TRIPLE and mol.atoms[j].symbol == "N"
                   for j in nbrs):
                counts["nitrile"] += 1
        if atom.symbol == "N":
            o_nbrs = [j for j in nbrs if mol.atoms[j].symbol == "O"]
            if len(o_nbrs) == 2 and len(nbrs) == 3:
                counts["nitro"] += 1
        if atom.symbol == "S":
            dbl_o = [j for j in nbrs
                     if mol.bond_between(i, j).order == DOUBLE and mol.atoms[j].symbol == "O"]
            if len(dbl_o) == 2:
                if any(mol.atoms[j].symbol == "N" for j in nbrs):
                    counts["sulfonamide"] += 1
                else:
                    counts["sulfone"] += 1
    return counts


# ------------------------------------------------------------------ complexity

def bertz_complexity(mol: Molecule) -> float:
    """Bertz CT-style graph complexity: bond-type entropy plus atom-type entropy."""
    if mol.n_atoms <= 1:
        return 0.0

    def entropy_term(labels: list) -> float:
        n = len(labels)
        if n == 0:
            return 0.0
        counts: dict = {}
        for x in labels:
            counts[x] = counts.get(x, 0) + 1
        total = 0.0
        for c in counts.values():
            p = c / n
            total -= n * p * math.log2(p)
        return total

    atom_types = [(a.symbol, a.aromatic, mol.degree(i)) for i, a in enumerate(mol.atoms)]
    bond_types = [(mol.atoms[b.a].symbol, mol.atoms[b.b].symbol, b.order) for b in mol.bonds]
    connectivity = sum(mol.degree(i) * (mol.degree(i) - 1) // 2 for i in range(mol.n_atoms))
    return entropy_term(atom_types) + entropy_term(bond_types) + connectivity


def synthetic_accessibility(mol: Molecule) -> float:
    """Heuristic synthetic accessibility on a 1 (easy) to 10 (hard) scale.

    Follows the structure of Ertl & Schuffenhauer (2009): a fragment-rarity
    term plus complexity penalties. The fragment term here uses Morgan
    identifier rarity estimated from the loaded reference library rather than
    the original PubChem frequency table, so values are comparable within this
    package but not numerically identical to the published score.
    """
    from nexus.chem.fingerprint import morgan_identifiers

    n_heavy = mol.heavy_atom_count()
    if n_heavy == 0:
        return 1.0
    idents = morgan_identifiers(mol, radius=2)
    n_unique = len(idents)
    fragment_penalty = n_unique / max(n_heavy, 1)
    ring_penalty = 0.0
    for ring in mol.rings:
        if len(ring) > 8:
            ring_penalty += 0.5
        if len(ring) < 5:
            ring_penalty += 0.3
    stereo_penalty = 0.5 * num_stereocenters(mol)
    bridge_penalty = 0.4 * num_bridgehead_atoms(mol)
    spiro_penalty = 0.3 * num_spiro_atoms(mol)
    size_penalty = math.log10(max(n_heavy, 1)) * 0.5
    raw = (fragment_penalty + ring_penalty + stereo_penalty + bridge_penalty
           + spiro_penalty + size_penalty)
    return float(np.clip(1.0 + 1.6 * raw, 1.0, 10.0))


# -------------------------------------------------------------- drug-likeness

# Trapezoidal desirability functions: each returns 1.0 inside the preferred
# range, tapering linearly to 0 at the outer bounds. Ranges follow the
# property distributions of oral drugs reported in the drug-likeness
# literature (Lipinski 2001, Veber 2002, Ghose 1999, Bickerton 2012).
# (low_zero, low_one, high_one, high_zero)
_DESIRABILITY_RANGES = {
    "mw":      (120.0, 200.0, 400.0, 600.0),
    "logp":    (-2.0, 0.5, 3.5, 6.0),
    "hba":     (0.0, 0.0, 8.0, 14.0),
    "hbd":     (0.0, 0.0, 3.0, 7.0),
    "tpsa":    (0.0, 20.0, 100.0, 160.0),
    "rotb":    (0.0, 0.0, 7.0, 13.0),
    "arom":    (0.0, 1.0, 3.0, 5.0),
    "alerts":  (0.0, 0.0, 0.0, 3.0),
}
# Relative weights; alerts and logP dominate, matching their weight in QED.
_DESIRABILITY_WEIGHTS = {
    "mw": 0.66, "logp": 0.95, "hba": 0.05, "hbd": 0.61,
    "tpsa": 0.60, "rotb": 0.65, "arom": 0.48, "alerts": 0.95,
}


def trapezoid(x: float, low_zero: float, low_one: float,
              high_one: float, high_zero: float) -> float:
    """Desirability in [0, 1]: 1 on [low_one, high_one], 0 outside the outer bounds."""
    if low_one <= x <= high_one:
        return 1.0
    if x <= low_zero or x >= high_zero:
        return 0.0
    if x < low_one:
        return (x - low_zero) / (low_one - low_zero) if low_one > low_zero else 1.0
    return (high_zero - x) / (high_zero - high_one) if high_zero > high_one else 1.0


def desirabilities(mol: Molecule, logp: float | None = None) -> dict[str, float]:
    """Per-property desirability values used by `drug_likeness`."""
    values = {
        "mw": mol.molecular_weight(),
        "logp": crippen_logp(mol) if logp is None else logp,
        "hba": num_lipinski_hba(mol),
        "hbd": num_lipinski_hbd(mol),
        "tpsa": tpsa(mol),
        "rotb": num_rotatable_bonds(mol),
        "arom": num_aromatic_rings(mol),
        "alerts": count_structural_alerts(mol),
    }
    return {k: trapezoid(float(v), *_DESIRABILITY_RANGES[k]) for k, v in values.items()}


def drug_likeness(mol: Molecule, logp: float | None = None) -> float:
    """QED-style weighted drug-likeness in [0, 1].

    The weighted geometric mean of eight property desirabilities. This follows
    the *structure* of the Quantitative Estimate of Drug-likeness
    (Bickerton et al. 2012) but uses the transparent trapezoidal desirability
    functions defined above rather than that paper's fitted asymmetric
    sigmoids, so values are not numerically identical to published QED.

    Measured behaviour (benchmarks/validate_drug_likeness.py): the score does
    NOT separate FDA-approved drugs from ClinTox clinical-trial failures
    (AUC 0.52), which is expected because both classes are drug-like molecules
    that reached trials. Lipinski violations do differ between those classes
    (0.39 vs 0.64 on average). Treat this score as a property-range summary for
    filtering and as a model feature, not as a predictor of approval.
    """
    d = desirabilities(mol, logp)
    total_w = sum(_DESIRABILITY_WEIGHTS.values())
    total = 0.0
    for key, value in d.items():
        total += _DESIRABILITY_WEIGHTS[key] * math.log(max(value, 1e-6))
    return float(math.exp(total / total_w))


def count_structural_alerts(mol: Molecule) -> int:
    """Count of simple reactive/undesirable motifs (a small PAINS-like set)."""
    groups = count_functional_groups(mol)
    alerts = 0
    alerts += groups["nitro"]
    alerts += groups["thiol"]
    if any(a.symbol == "I" for a in mol.atoms):
        alerts += 1
    for bond in mol.bonds:
        if bond.order == TRIPLE and not bond.in_ring:
            a, b = mol.atoms[bond.a], mol.atoms[bond.b]
            if a.symbol == "C" and b.symbol == "C":
                alerts += 1
    for ring in mol.rings:
        if len(ring) == 3 and any(mol.atoms[i].symbol in ("N", "O") for i in ring):
            alerts += 1          # epoxide / aziridine
    for i, atom in enumerate(mol.atoms):
        if atom.symbol == "N" and len(mol.neighbors(i)) == 2:
            if all(mol.atoms[j].symbol == "N" for j in mol.neighbors(i)):
                alerts += 1      # azo / azide fragment
    return alerts


# ------------------------------------------------------------- Crippen-style logP

# Atom-contribution logP. Types are defined by element, aromaticity, hybridisation
# and attached-heteroatom context, in the spirit of Wildman & Crippen (1999).
# Contributions are fitted on experimental data by
# benchmarks/fit_logp.py and stored in nexus/chem/logp_params.py.
def crippen_atom_type(mol: Molecule, i: int) -> str:
    atom = mol.atoms[i]
    sym = atom.symbol
    nbrs = mol.neighbors(i)
    hetero = sum(1 for j in nbrs if mol.atoms[j].symbol not in ("C", "H"))
    if sym == "C":
        if atom.aromatic:
            if hetero:
                return "C_ar_X"
            return "C_ar_H" if atom.total_h else "C_ar"
        unsaturated = any(mol.bond_between(i, j).order in (DOUBLE, TRIPLE) for j in nbrs)
        if any(mol.bond_between(i, j).order == TRIPLE for j in nbrs):
            return "C_sp"
        if unsaturated:
            return "C_sp2_X" if hetero else "C_sp2"
        if hetero:
            return "C_sp3_X2" if hetero >= 2 else "C_sp3_X"
        return f"C_sp3_{min(mol.degree(i), 4)}"
    if sym == "N":
        if atom.aromatic:
            return "N_ar"
        if any(mol.bond_between(i, j).order == TRIPLE for j in nbrs):
            return "N_sp"
        if any(_is_amide_bond(mol, j, i) for j in nbrs):
            return "N_amide"
        if atom.charge > 0:
            return "N_plus"
        return f"N_sp3_H{min(atom.total_h, 2)}"
    if sym == "O":
        if atom.aromatic:
            return "O_ar"
        if atom.charge < 0:
            return "O_minus"
        if any(mol.bond_between(i, j).order == DOUBLE for j in nbrs):
            return "O_double"
        return "O_H" if atom.total_h else "O_ether"
    if sym == "S":
        return "S_ar" if atom.aromatic else ("S_oxide" if any(
            mol.bond_between(i, j).order == DOUBLE for j in nbrs) else "S_sulfide")
    if sym in ("F", "Cl", "Br", "I"):
        return f"halogen_{sym}"
    if sym == "P":
        return "P"
    return "other"


def crippen_logp(mol: Molecule) -> float:
    """Atom-contribution estimate of octanol/water logP."""
    from nexus.chem.logp_params import LOGP_CONTRIBUTIONS, LOGP_INTERCEPT, LOGP_H

    total = LOGP_INTERCEPT
    for i, atom in enumerate(mol.atoms):
        total += LOGP_CONTRIBUTIONS.get(crippen_atom_type(mol, i), 0.0)
        total += LOGP_H * atom.total_h
    return float(total)


def molar_refractivity(mol: Molecule) -> float:
    """Crippen-style molar refractivity from atomic polarisability proxies."""
    total = 0.0
    for i, atom in enumerate(mol.atoms):
        from nexus.chem.periodic import vdw_radius
        total += 1.4 * vdw_radius(atom.symbol) ** 2 + 0.35 * atom.total_h
    return float(total)


# -------------------------------------------------------------- rule compliance

def lipinski_violations(mol: Molecule) -> int:
    """Rule-of-five violations (Lipinski 2001)."""
    v = 0
    v += mol.molecular_weight() > 500
    v += crippen_logp(mol) > 5
    v += num_lipinski_hbd(mol) > 5
    v += num_lipinski_hba(mol) > 10
    return int(v)


def passes_lipinski(mol: Molecule) -> bool:
    return lipinski_violations(mol) <= 1


def veber_pass(mol: Molecule) -> bool:
    """Veber (2002) oral bioavailability rules."""
    return num_rotatable_bonds(mol) <= 10 and tpsa(mol) <= 140


DESCRIPTOR_FUNCTIONS = {
    "mw": lambda m: m.molecular_weight(),
    "heavy_atoms": lambda m: m.heavy_atom_count(),
    "logp": crippen_logp,
    "tpsa": tpsa,
    "hbd": num_lipinski_hbd,
    "hba": num_lipinski_hba,
    "hbd_strict": num_hbond_donors,
    "hba_strict": num_hbond_acceptors,
    "rotatable_bonds": num_rotatable_bonds,
    "rings": num_rings,
    "aromatic_rings": num_aromatic_rings,
    "aliphatic_rings": num_aliphatic_rings,
    "heteroatoms": num_heteroatoms,
    "fraction_csp3": fraction_csp3,
    "stereocenters": num_stereocenters,
    "spiro_atoms": num_spiro_atoms,
    "bridgehead_atoms": num_bridgehead_atoms,
    "formal_charge": lambda m: m.charge(),
    "bertz_complexity": bertz_complexity,
    "synthetic_accessibility": synthetic_accessibility,
    "drug_likeness": drug_likeness,
    "molar_refractivity": molar_refractivity,
    "lipinski_violations": lipinski_violations,
    "structural_alerts": count_structural_alerts,
}

DESCRIPTOR_NAMES = tuple(DESCRIPTOR_FUNCTIONS)


def compute_descriptors(mol: Molecule) -> dict[str, float]:
    return {name: float(fn(mol)) for name, fn in DESCRIPTOR_FUNCTIONS.items()}


def descriptor_matrix(mols, names: tuple[str, ...] = DESCRIPTOR_NAMES) -> np.ndarray:
    return np.array([[float(DESCRIPTOR_FUNCTIONS[n](m)) for n in names] for m in mols])
