"""SMILES parser and writer (OpenSMILES subset), with no external dependencies.

Supports: organic-subset and bracket atoms, isotopes, charges, explicit
hydrogens, tetrahedral chirality markers, all bond symbols, branches, ring-bond
closures (including %nn and %(nnn)), disconnected components, and atom maps.
"""

from __future__ import annotations

from nexus.chem.mol import (
    AROMATIC,
    BOND_SYMBOL,
    DOUBLE,
    QUADRUPLE,
    SINGLE,
    TRIPLE,
    Atom,
    Bond,
    Molecule,
    MoleculeError,
)
from nexus.chem.periodic import AROMATIC_ELEMENTS, ELEMENTS, ORGANIC_SUBSET

BOND_CHARS = {"-": SINGLE, "=": DOUBLE, "#": TRIPLE, "$": QUADRUPLE, ":": AROMATIC}
# Two-letter organic-subset symbols must be tried before one-letter ones.
TWO_LETTER_ORGANIC = ("Cl", "Br")


class SmilesSyntaxError(MoleculeError):
    pass


class _Parser:
    def __init__(self, text: str):
        self.text = text.strip()
        self.pos = 0
        self.mol = Molecule(smiles=self.text)
        self.prev: int | None = None
        self.branch_stack: list[int] = []
        self.ring_open: dict[int, tuple[int, int | None, str]] = {}
        self.pending_bond: int | None = None
        self.pending_stereo: str = ""
        self.bond_keys: set[tuple[int, int]] = set()

    # ------------------------------------------------------------------ helpers

    def error(self, msg: str) -> None:
        raise SmilesSyntaxError(f"{msg} at position {self.pos} in {self.text!r}")

    def peek(self, n: int = 1) -> str:
        return self.text[self.pos:self.pos + n]

    def add_atom(self, atom: Atom) -> int:
        self.mol.atoms.append(atom)
        idx = len(self.mol.atoms) - 1
        if self.prev is not None:
            order = self.pending_bond
            if order is None:
                both_aromatic = atom.aromatic and self.mol.atoms[self.prev].aromatic
                order = AROMATIC if both_aromatic else SINGLE
            self.mol.bonds.append(Bond(self.prev, idx, order, self.pending_stereo))
            self.bond_keys.add((min(self.prev, idx), max(self.prev, idx)))
        self.pending_bond = None
        self.pending_stereo = ""
        self.prev = idx
        return idx

    # -------------------------------------------------------------------- scan

    def parse(self) -> Molecule:
        if not self.text:
            raise SmilesSyntaxError("empty SMILES")
        while self.pos < len(self.text):
            ch = self.text[self.pos]
            if ch == "[":
                self.parse_bracket_atom()
            elif ch == "(":
                if self.prev is None:
                    self.error("branch opened before any atom")
                self.branch_stack.append(self.prev)
                self.pos += 1
            elif ch == ")":
                if not self.branch_stack:
                    self.error("unmatched ')'")
                self.prev = self.branch_stack.pop()
                self.pos += 1
            elif ch in BOND_CHARS:
                self.pending_bond = BOND_CHARS[ch]
                self.pos += 1
            elif ch in "/\\":
                self.pending_bond = SINGLE
                self.pending_stereo = ch
                self.pos += 1
            elif ch == ".":
                self.prev = None
                self.pending_bond = None
                self.pos += 1
            elif ch.isdigit() or ch == "%":
                self.parse_ring_closure()
            elif ch.isspace():
                break
            else:
                self.parse_organic_atom()
        if self.branch_stack:
            self.error("unclosed '('")
        if self.ring_open:
            nums = sorted(self.ring_open)
            raise SmilesSyntaxError(f"unclosed ring bond(s) {nums} in {self.text!r}")
        if not self.mol.atoms:
            raise SmilesSyntaxError(f"no atoms parsed from {self.text!r}")
        return self.mol.perceive()

    def parse_organic_atom(self) -> None:
        two = self.peek(2)
        if two in TWO_LETTER_ORGANIC:
            self.pos += 2
            self.add_atom(Atom(symbol=two))
            return
        ch = self.text[self.pos]
        if ch == "*":
            self.pos += 1
            self.add_atom(Atom(symbol="*", in_bracket=True))
            return
        upper = ch.upper()
        if ch.islower():
            if ch not in ("b", "c", "n", "o", "p", "s"):
                self.error(f"unexpected aromatic symbol {ch!r}")
            self.pos += 1
            self.add_atom(Atom(symbol=upper, aromatic=True))
            return
        if upper not in ORGANIC_SUBSET:
            self.error(f"{ch!r} must be written in brackets")
        self.pos += 1
        self.add_atom(Atom(symbol=upper))

    def parse_bracket_atom(self) -> None:
        close = self.text.find("]", self.pos)
        if close < 0:
            self.error("unclosed '['")
        body, start = self.text[self.pos + 1:close], self.pos
        self.pos = close + 1
        i = 0

        isotope = 0
        while i < len(body) and body[i].isdigit():
            isotope = isotope * 10 + int(body[i])
            i += 1

        if i >= len(body):
            self.pos = start
            self.error("bracket atom without element symbol")
        symbol = ""
        if body[i] == "*":
            symbol = "*"
            i += 1
        else:
            two = body[i:i + 2]
            if len(two) == 2 and two.lower() in AROMATIC_ELEMENTS and two[0].islower():
                symbol, i = two, i + 2
            elif len(two) == 2 and two in ELEMENTS:
                symbol, i = two, i + 2
            elif body[i] in ELEMENTS or body[i].lower() in AROMATIC_ELEMENTS:
                symbol, i = body[i], i + 1
            else:
                self.pos = start
                self.error(f"unknown element {body[i:i+2]!r}")
        aromatic = symbol[0].islower()
        canonical = symbol[0].upper() + symbol[1:] if symbol != "*" else "*"
        if canonical != "*" and canonical not in ELEMENTS:
            self.pos = start
            self.error(f"unknown element {symbol!r}")

        chirality = ""
        if body[i:i + 2] == "@@":
            chirality, i = "@@", i + 2
        elif body[i:i + 1] == "@":
            chirality, i = "@", i + 1
            while i < len(body) and body[i].isalpha() and body[i] not in "H":
                i += 1   # @TH1, @AL2, @SP3, @TB5, @OH9 ... treated as tetrahedral
            while i < len(body) and body[i].isdigit():
                i += 1

        explicit_h = 0
        if body[i:i + 1] == "H":
            i += 1
            explicit_h = 1
            num = ""
            while i < len(body) and body[i].isdigit():
                num += body[i]
                i += 1
            if num:
                explicit_h = int(num)

        charge = 0
        while i < len(body) and body[i] in "+-":
            sign = 1 if body[i] == "+" else -1
            i += 1
            num = ""
            while i < len(body) and body[i].isdigit():
                num += body[i]
                i += 1
            if num:
                charge += sign * int(num)
            else:
                charge += sign
                while i < len(body) and body[i] == ("+" if sign > 0 else "-"):
                    charge += sign
                    i += 1

        map_number = 0
        if body[i:i + 1] == ":":
            i += 1
            num = ""
            while i < len(body) and body[i].isdigit():
                num += body[i]
                i += 1
            if not num:
                self.pos = start
                self.error("atom map ':' without number")
            map_number = int(num)

        if i != len(body):
            self.pos = start
            self.error(f"trailing {body[i:]!r} in bracket atom")

        self.add_atom(Atom(symbol=canonical, aromatic=aromatic, charge=charge,
                           isotope=isotope, explicit_h=explicit_h, chirality=chirality,
                           map_number=map_number, in_bracket=True))

    def parse_ring_closure(self) -> None:
        if self.prev is None:
            self.error("ring closure before any atom")
        if self.text[self.pos] == "%":
            self.pos += 1
            if self.peek() == "(":
                close = self.text.find(")", self.pos)
                if close < 0:
                    self.error("unclosed '%(' ring number")
                num = int(self.text[self.pos + 1:close])
                self.pos = close + 1
            else:
                if not self.peek(2).isdigit():
                    self.error("'%' must be followed by two digits")
                num = int(self.text[self.pos:self.pos + 2])
                self.pos += 2
        else:
            num = int(self.text[self.pos])
            self.pos += 1

        order, stereo = self.pending_bond, self.pending_stereo
        self.pending_bond, self.pending_stereo = None, ""
        if num in self.ring_open:
            start_atom, start_order, start_stereo = self.ring_open.pop(num)
            if start_atom == self.prev:
                self.error(f"ring bond {num} closes on its own atom")
            key = (min(start_atom, self.prev), max(start_atom, self.prev))
            if key in self.bond_keys:
                self.error(f"ring bond {num} duplicates an existing bond")
            final = order or start_order
            if final is None:
                both = self.mol.atoms[start_atom].aromatic and self.mol.atoms[self.prev].aromatic
                final = AROMATIC if both else SINGLE
            self.mol.bonds.append(Bond(start_atom, self.prev, final, stereo or start_stereo))
            self.bond_keys.add(key)
        else:
            self.ring_open[num] = (self.prev, order, stereo)


def parse_smiles(text: str, name: str = "") -> Molecule:
    """Parse a SMILES string into a perceived Molecule. Raises MoleculeError."""
    mol = _Parser(text).parse()
    mol.name = name
    return mol


def read_smiles(text: str, name: str = "") -> Molecule | None:
    """Parse, returning None instead of raising on invalid input."""
    try:
        return parse_smiles(text, name)
    except MoleculeError:
        return None


# ------------------------------------------------------------------- writing

def _atom_token(mol: Molecule, i: int) -> str:
    atom = mol.atoms[i]
    sym = atom.symbol.lower() if atom.aromatic else atom.symbol
    simple = (
        atom.symbol in ORGANIC_SUBSET
        and atom.charge == 0
        and atom.isotope == 0
        and not atom.chirality
        and atom.map_number == 0
    )
    if simple:
        # Only omit brackets when the implicit-H model reproduces the H count.
        expected = atom.implicit_h if not atom.in_bracket else None
        if expected is None or atom.total_h == expected:
            if not atom.in_bracket or atom.explicit_h == 0:
                return sym
    inner = sym
    if atom.isotope:
        inner = f"{atom.isotope}{inner}"
    inner += atom.chirality
    h = atom.total_h
    if h == 1:
        inner += "H"
    elif h > 1:
        inner += f"H{h}"
    if atom.charge > 0:
        inner += "+" if atom.charge == 1 else f"+{atom.charge}"
    elif atom.charge < 0:
        inner += "-" if atom.charge == -1 else f"-{-atom.charge}"
    if atom.map_number:
        inner += f":{atom.map_number}"
    return f"[{inner}]"


def write_smiles(mol: Molecule, root: int | None = None) -> str:
    """Write a (non-canonical) SMILES string by DFS over each component."""
    # Pass 1: DFS spanning forest. Edges not in the forest are ring closures.
    parent: dict[int, int] = {}
    order: list[int] = []
    tree_edges: set[tuple[int, int]] = set()
    seen: set[int] = set()
    starts = [min(c) for c in mol.components()] if root is None else [root]

    def walk(start: int) -> None:
        stack = [(start, -1)]
        while stack:
            i, from_atom = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            order.append(i)
            if from_atom >= 0:
                parent[i] = from_atom
                tree_edges.add((min(i, from_atom), max(i, from_atom)))
            for j in reversed(mol.neighbors(i)):
                if j not in seen:
                    stack.append((j, i))

    for s in starts:
        walk(s)
    for i in range(mol.n_atoms):       # any atom missed (isolated component)
        if i not in seen:
            starts.append(i)
            walk(i)

    ring_bonds = [b for b in mol.bonds
                  if (min(b.a, b.b), max(b.a, b.b)) not in tree_edges]
    # Pass 2: assign a ring-closure label to each non-tree bond, reusing labels
    # once a pair has been closed.
    position = {atom: k for k, atom in enumerate(order)}
    ring_bonds.sort(key=lambda b: min(position[b.a], position[b.b]))
    labels: dict[int, int] = {}
    closures_at: dict[int, list[tuple[int, Bond, int]]] = {}
    active: dict[int, int] = {}
    for bond in ring_bonds:
        first, second = sorted((bond.a, bond.b), key=lambda a: position[a])
        label = 1
        while label in active:
            label += 1
        active[label] = second
        labels[id(bond)] = label
        closures_at.setdefault(first, []).append((label, bond, second))
        closures_at.setdefault(second, []).append((label, bond, first))

    def bond_token(bond: Bond, from_atom: int, to_atom: int) -> str:
        if bond.order == AROMATIC:
            return ""
        if bond.order == SINGLE:
            if mol.atoms[from_atom].aromatic and mol.atoms[to_atom].aromatic:
                return "-"
            return ""
        return BOND_SYMBOL[bond.order]

    def label_token(label: int) -> str:
        return str(label) if label < 10 else f"%{label}"

    out: list[str] = []
    emitted: set[int] = set()

    def emit(i: int) -> None:
        emitted.add(i)
        token = _atom_token(mol, i)
        for label, bond, other in closures_at.get(i, []):
            token += bond_token(bond, i, other) + label_token(label)
            if other in emitted:
                active.pop(label, None)
        out.append(token)
        children = [j for j in mol.neighbors(i)
                    if parent.get(j) == i and j not in emitted]
        for k, j in enumerate(children):
            last = k == len(children) - 1
            if not last:
                out.append("(")
            out.append(bond_token(mol.bond_between(i, j), i, j))
            emit(j)
            if not last:
                out.append(")")

    for s_i, s in enumerate(starts):
        if s in emitted:
            continue
        if out:
            out.append(".")
        emit(s)
    return "".join(out)
