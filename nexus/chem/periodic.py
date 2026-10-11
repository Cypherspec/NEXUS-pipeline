"""Element data needed for molecular graph construction and featurization.

Values: IUPAC standard atomic weights, Cordero covalent radii (Angstrom),
Bondi van der Waals radii (Angstrom), Pauling electronegativities. Default
valences follow the Daylight/OpenSMILES model used for implicit hydrogens.
"""

from __future__ import annotations

# symbol: (atomic number, atomic mass, covalent radius, vdW radius, electronegativity)
ELEMENTS: dict[str, tuple[int, float, float, float, float]] = {
    "H":  (1,   1.008,  0.31, 1.20, 2.20),
    "He": (2,   4.003,  0.28, 1.40, 0.00),
    "Li": (3,   6.941,  1.28, 1.82, 0.98),
    "Be": (4,   9.012,  0.96, 1.53, 1.57),
    "B":  (5,  10.811,  0.84, 1.92, 2.04),
    "C":  (6,  12.011,  0.76, 1.70, 2.55),
    "N":  (7,  14.007,  0.71, 1.55, 3.04),
    "O":  (8,  15.999,  0.66, 1.52, 3.44),
    "F":  (9,  18.998,  0.57, 1.47, 3.98),
    "Ne": (10, 20.180,  0.58, 1.54, 0.00),
    "Na": (11, 22.990,  1.66, 2.27, 0.93),
    "Mg": (12, 24.305,  1.41, 1.73, 1.31),
    "Al": (13, 26.982,  1.21, 1.84, 1.61),
    "Si": (14, 28.086,  1.11, 2.10, 1.90),
    "P":  (15, 30.974,  1.07, 1.80, 2.19),
    "S":  (16, 32.066,  1.05, 1.80, 2.58),
    "Cl": (17, 35.453,  1.02, 1.75, 3.16),
    "Ar": (18, 39.948,  1.06, 1.88, 0.00),
    "K":  (19, 39.098,  2.03, 2.75, 0.82),
    "Ca": (20, 40.078,  1.76, 2.31, 1.00),
    "Sc": (21, 44.956,  1.70, 2.11, 1.36),
    "Ti": (22, 47.867,  1.60, 2.00, 1.54),
    "V":  (23, 50.942,  1.53, 2.00, 1.63),
    "Cr": (24, 51.996,  1.39, 2.00, 1.66),
    "Mn": (25, 54.938,  1.39, 2.00, 1.55),
    "Fe": (26, 55.845,  1.32, 2.00, 1.83),
    "Co": (27, 58.933,  1.26, 2.00, 1.88),
    "Ni": (28, 58.693,  1.24, 1.63, 1.91),
    "Cu": (29, 63.546,  1.32, 1.40, 1.90),
    "Zn": (30, 65.380,  1.22, 1.39, 1.65),
    "Ga": (31, 69.723,  1.22, 1.87, 1.81),
    "Ge": (32, 72.640,  1.20, 2.11, 2.01),
    "As": (33, 74.922,  1.19, 1.85, 2.18),
    "Se": (34, 78.960,  1.20, 1.90, 2.55),
    "Br": (35, 79.904,  1.20, 1.85, 2.96),
    "Kr": (36, 83.798,  1.16, 2.02, 3.00),
    "Rb": (37, 85.468,  2.20, 3.03, 0.82),
    "Sr": (38, 87.620,  1.95, 2.49, 0.95),
    "Y":  (39, 88.906,  1.90, 2.00, 1.22),
    "Zr": (40, 91.224,  1.75, 2.00, 1.33),
    "Mo": (42, 95.960,  1.54, 2.00, 2.16),
    "Ru": (44, 101.07,  1.46, 2.00, 2.20),
    "Rh": (45, 102.91,  1.42, 2.00, 2.28),
    "Pd": (46, 106.42,  1.39, 1.63, 2.20),
    "Ag": (47, 107.87,  1.45, 1.72, 1.93),
    "Cd": (48, 112.41,  1.44, 1.58, 1.69),
    "In": (49, 114.82,  1.42, 1.93, 1.78),
    "Sn": (50, 118.71,  1.39, 2.17, 1.96),
    "Sb": (51, 121.76,  1.39, 2.06, 2.05),
    "Te": (52, 127.60,  1.38, 2.06, 2.10),
    "I":  (53, 126.90,  1.39, 1.98, 2.66),
    "Xe": (54, 131.29,  1.40, 2.16, 2.60),
    "Cs": (55, 132.91,  2.44, 3.43, 0.79),
    "Ba": (56, 137.33,  2.15, 2.68, 0.89),
    "Pt": (78, 195.08,  1.36, 1.75, 2.28),
    "Au": (79, 196.97,  1.36, 1.66, 2.54),
    "Hg": (80, 200.59,  1.32, 1.55, 2.00),
    "Tl": (81, 204.38,  1.45, 1.96, 1.62),
    "Pb": (82, 207.20,  1.46, 2.02, 2.33),
    "Bi": (83, 208.98,  1.48, 2.07, 2.02),
    # Remaining transition metals, lanthanides and actinides: present in public
    # screening libraries (metal complexes in HIV), needed so parsing does not fail.
    "Sc": (21, 44.956, 1.70, 2.11, 1.36),
    "Ti": (22, 47.867, 1.60, 2.00, 1.54),
    "V":  (23, 50.942, 1.53, 2.00, 1.63),
    "Ge": (32, 72.640, 1.20, 2.11, 2.01),
    "Nb": (41, 92.906, 1.64, 2.00, 1.60),
    "Tc": (43, 98.000, 1.47, 2.00, 1.90),
    "La": (57, 138.910, 2.07, 2.00, 1.10),
    "Ce": (58, 140.120, 2.04, 2.00, 1.12),
    "Pr": (59, 140.910, 2.03, 2.00, 1.13),
    "Nd": (60, 144.240, 2.01, 2.00, 1.14),
    "Pm": (61, 145.000, 1.99, 2.00, 1.13),
    "Sm": (62, 150.360, 1.98, 2.00, 1.17),
    "Eu": (63, 151.960, 1.98, 2.00, 1.20),
    "Gd": (64, 157.250, 1.96, 2.00, 1.20),
    "Tb": (65, 158.930, 1.94, 2.00, 1.10),
    "Dy": (66, 162.500, 1.92, 2.00, 1.22),
    "Ho": (67, 164.930, 1.92, 2.00, 1.23),
    "Er": (68, 167.260, 1.89, 2.00, 1.24),
    "Tm": (69, 168.930, 1.90, 2.00, 1.25),
    "Yb": (70, 173.050, 1.87, 2.00, 1.10),
    "Lu": (71, 174.970, 1.87, 2.00, 1.27),
    "Hf": (72, 178.490, 1.75, 2.00, 1.30),
    "Ta": (73, 180.950, 1.70, 2.00, 1.50),
    "W":  (74, 183.840, 1.62, 2.00, 2.36),
    "Re": (75, 186.210, 1.51, 2.00, 1.90),
    "Os": (76, 190.230, 1.44, 2.00, 2.20),
    "Ir": (77, 192.220, 1.41, 2.00, 2.20),
    "Tl": (81, 204.380, 1.45, 1.96, 1.62),
    "Po": (84, 209.000, 1.40, 1.97, 2.00),
    "At": (85, 210.000, 1.50, 2.02, 2.20),
    "Rn": (86, 222.000, 1.50, 2.20, 2.20),
    "Fr": (87, 223.000, 2.60, 3.48, 0.70),
    "Ra": (88, 226.000, 2.21, 2.83, 0.90),
    "Ac": (89, 227.000, 2.15, 2.00, 1.10),
    "Th": (90, 232.040, 2.06, 2.00, 1.30),
    "Pa": (91, 231.040, 2.00, 2.00, 1.50),
    "U":  (92, 238.030, 1.96, 1.86, 1.38),
    "Np": (93, 237.000, 1.90, 2.00, 1.36),
    "Pu": (94, 244.000, 1.87, 2.00, 1.28),
    "Am": (95, 243.000, 1.80, 2.00, 1.30),
    "Cm": (96, 247.000, 1.69, 2.00, 1.30),
    "Rf": (104, 267.000, 1.57, 2.00, 0.00),
    "Db": (105, 268.000, 1.49, 2.00, 0.00),
    "Sg": (106, 269.000, 1.43, 2.00, 0.00),
    "Bh": (107, 270.000, 1.41, 2.00, 0.00),
    "Hs": (108, 269.000, 1.34, 2.00, 0.00),
    "Mt": (109, 278.000, 1.29, 2.00, 0.00),
}

SYMBOL_BY_NUMBER = {v[0]: k for k, v in ELEMENTS.items()}

# Valences used for implicit-hydrogen filling. The first value that is >= the
# summed explicit bond order is used (OpenSMILES default valence model).
DEFAULT_VALENCES: dict[str, tuple[int, ...]] = {
    "B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5), "S": (2, 4, 6),
    "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,), "At": (1,), "Ts": (1,),
    "H": (1,), "Se": (2, 4, 6), "Si": (4,), "As": (3, 5), "Te": (2, 4, 6),
}

# Elements writable in SMILES without brackets.
ORGANIC_SUBSET = {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}

# Elements that may be written lowercase (aromatic) in SMILES.
AROMATIC_ELEMENTS = {"b", "c", "n", "o", "p", "s", "se", "as", "te", "si"}


# Wildcard atom "*" (unspecified / polymer attachment point) carries no mass
# and no valence requirement.
WILDCARD = "*"

# Maximum valences used for sanity checking, mirroring standard sanitization.
MAX_VALENCE: dict[str, int] = {
    "H": 1, "B": 4, "C": 4, "N": 4, "O": 3, "F": 1, "Si": 6, "P": 6, "S": 7,
    "Cl": 7, "Se": 6, "Br": 7, "I": 7, "As": 5, "Te": 6, "Al": 4, "Mg": 2,
}


def atomic_number(symbol: str) -> int:
    return 0 if symbol == WILDCARD else ELEMENTS[symbol][0]


def atomic_mass(symbol: str) -> float:
    return 0.0 if symbol == WILDCARD else ELEMENTS[symbol][1]


def covalent_radius(symbol: str) -> float:
    return ELEMENTS.get(symbol, ELEMENTS["C"])[2]


def vdw_radius(symbol: str) -> float:
    return ELEMENTS.get(symbol, ELEMENTS["C"])[3]


def electronegativity(symbol: str) -> float:
    return ELEMENTS.get(symbol, ELEMENTS["C"])[4]


def max_valence(symbol: str) -> int | None:
    """Upper bound on valence, or None when unconstrained (metals, wildcards)."""
    return MAX_VALENCE.get(symbol)


def default_valence(symbol: str, bond_order_sum: float) -> int:
    """Lowest standard valence that can accommodate the explicit bonds."""
    valences = DEFAULT_VALENCES.get(symbol)
    if not valences:
        return 0
    for v in valences:
        if bond_order_sum <= v:
            return v
    return valences[-1]
