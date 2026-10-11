"""NEXUS: cross-validated ensemble selection for tabular classification."""

from tabular_ensemble.classifier import NexusClassifier
from tabular_ensemble.ensemble import greedy_ensemble_selection, mix
from tabular_ensemble.calibration import fit_temperature, apply_temperature

__all__ = [
    "NexusClassifier",
    "greedy_ensemble_selection",
    "mix",
    "fit_temperature",
    "apply_temperature",
]
__version__ = "1.0.0"
