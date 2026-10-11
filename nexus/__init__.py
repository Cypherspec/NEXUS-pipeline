"""NEXUS: Non-Equilibrium eXploration with Unified Synthesis.

A dependency-free (NumPy-only) drug discovery pipeline covering hit
identification, lead optimization, binding affinity prediction and de novo
generation.
"""

from nexus.chem.mol import Molecule, MoleculeError
from nexus.chem.smiles import parse_smiles, read_smiles, write_smiles

__all__ = ["Molecule", "MoleculeError", "parse_smiles", "read_smiles", "write_smiles"]
__version__ = "0.1.0"
