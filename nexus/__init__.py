"""NEXUS: cross-validated ensemble selection for tabular classification."""

from nexus.classifier import NexusClassifier
from nexus.ensemble import greedy_ensemble_selection, mix
from nexus.calibration import fit_temperature, apply_temperature

__all__ = [
    "NexusClassifier",
    "greedy_ensemble_selection",
    "mix",
    "fit_temperature",
    "apply_temperature",
]
__version__ = "1.0.0"
