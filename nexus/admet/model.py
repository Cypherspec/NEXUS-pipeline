"""A multi-task ADMET model over the TDC endpoints.

One trunk, eighteen heads. The endpoints are modelled together rather than
separately because they are not independent problems: the same lipophilicity
and size that drive CYP inhibition drive hERG blockade and solubility, and the
large endpoints (CYP2D6 at 13k, hERG at 13k) carry enough structure-activity
signal to shape a representation that the small ones (DILI at 475,
bioavailability at 640) could never learn alone. Each molecule contributes to
whichever endpoints measured it; a masked loss ignores the rest, so no value is
imputed and no endpoint is diluted by another's chemistry.

What ships with the panel is this trunk plus the heads, trained once offline.
Inference is three matrix multiplies, which is why the extension sandbox can
run it with NumPy alone.

Two things are predicted alongside every value, because a number without them
is not usable for triage:

    calibration        classification heads emit logits, which are not
                       probabilities; a Platt scaling fitted on held-out data
                       maps them to probabilities that mean what they say,
                       and regression heads carry split-conformal intervals
                       with a finite-sample coverage guarantee

    applicability      a leverage score against the training set's descriptor
                       distribution. A model asked about chemistry unlike
                       anything it was fitted on will still return a number;
                       leverage is what says that number is extrapolation.
                       The threshold is the standard Williams-plot h* = 3p/n.
"""

from __future__ import annotations

import base64
import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nexus.admet.features import N_DESCRIPTORS, FeatureScaler, featurize
from nexus.admet.tdc import ENDPOINTS

TRUNK_SIZES = (256, 128)


# ----------------------------------------------------------------- weight codec

def pack(array: np.ndarray) -> dict:
    """Float array to a UTF-8 payload.

    Mantis decodes every file in an extension package as text and refuses
    binaries, so weights cannot ship as .npy. The trunk dominates the file, and
    at half precision it costs about 0.0005 in predicted probability against
    float64 -- far below the model's own error -- so it is stored that way to
    stay inside the package's per-file limit. Anything that does not fit half
    precision's range, such as an inverse covariance with large entries, falls
    back to single precision rather than overflowing to infinity.
    """
    a = np.ascontiguousarray(array, dtype=np.float64)
    finite = a[np.isfinite(a)]
    dtype = "float16" if (finite.size == 0 or np.abs(finite).max() < 6e4) else "float32"
    raw = a.astype(dtype).tobytes()
    return {"dtype": dtype, "b64": base64.b64encode(zlib.compress(raw, 9)).decode("ascii")}


def unpack(payload, shape: tuple[int, ...]) -> np.ndarray:
    # Version 1 of the format stored a bare base64 string at half precision.
    if isinstance(payload, str):
        payload = {"dtype": "float16", "b64": payload}
    raw = zlib.decompress(base64.b64decode(payload["b64"]))
    return np.frombuffer(raw, dtype=payload["dtype"]).astype(np.float64).reshape(shape)


# ----------------------------------------------------------------- the model

@dataclass
class Head:
    """One endpoint's output layer and the calibration fitted on top of it."""

    name: str
    task: str
    weight: np.ndarray                     # (hidden,)
    bias: float
    #: Per-endpoint linear model on the raw feature vector, and how much of the
    #: final score comes from the network rather than from it. Chosen on the
    #: validation split: an endpoint where the trunk adds nothing keeps the
    #: linear model, which on several of these is the stronger of the two.
    linear: np.ndarray | None = None
    blend: float = 1.0
    #: Platt scaling (classification): probability = sigmoid(a * logit + b).
    platt_a: float = 1.0
    platt_b: float = 0.0
    #: Standardisation of the target (regression), to report in real units.
    y_mean: float = 0.0
    y_std: float = 1.0
    #: Half-width of the split-conformal interval at the model's alpha.
    conformal_half_width: float = float("nan")
    #: Held-out metrics, carried so the panel can state them beside a value.
    metrics: dict = field(default_factory=dict)


@dataclass
class ADMETModel:
    """Trunk, heads, feature scaler and applicability domain."""

    trunk_weights: list[np.ndarray]
    trunk_biases: list[np.ndarray]
    heads: dict[str, Head]
    scaler: FeatureScaler
    #: Inverse covariance of the training descriptor block, and the Williams
    #: threshold, for the leverage score.
    leverage_inv_cov: np.ndarray
    leverage_threshold: float
    alpha: float = 0.1
    n_train: int = 0

    # -- forward

    def trunk(self, X: np.ndarray) -> np.ndarray:
        h = X
        for W, b in zip(self.trunk_weights, self.trunk_biases):
            h = np.maximum(h @ W + b, 0.0)
        return h

    def leverage(self, descriptors_scaled: np.ndarray) -> np.ndarray:
        """Mahalanobis leverage of each molecule against the training set."""
        d = descriptors_scaled
        return np.einsum("ij,jk,ik->i", d, self.leverage_inv_cov, d)

    def predict(self, mols, endpoints: list[str] | None = None) -> dict:
        """Predict every endpoint for a list of molecules.

        @returns A dict keyed by endpoint, each carrying the value in the
                 endpoint's own units, its uncertainty, and the applicability
                 flag; plus `applicability` per molecule.
        """
        if not mols:
            return {"n": 0, "endpoints": {}, "applicability": []}

        X = featurize(mols, self.scaler)
        H = self.trunk(X)
        descriptors_scaled = X[:, -N_DESCRIPTORS:]
        h = self.leverage(descriptors_scaled)
        in_domain = h <= self.leverage_threshold

        out: dict[str, dict] = {}
        for name in (endpoints or list(self.heads)):
            head = self.heads.get(name)
            if head is None:
                continue
            raw = H @ head.weight + head.bias
            if head.linear is not None and head.blend < 1.0:
                raw = head.blend * raw + (1.0 - head.blend) * (X @ head.linear)
            spec = ENDPOINTS[name]
            if head.task == "classification":
                logits = head.platt_a * raw + head.platt_b
                probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
                out[name] = {
                    "task": "classification",
                    "category": spec.category,
                    "units": spec.units,
                    "probability": probability.tolist(),
                    "metrics": head.metrics,
                    "higherIsBetter": spec.higher_is_better,
                }
            else:
                value = raw * head.y_std + head.y_mean
                out[name] = {
                    "task": "regression",
                    "category": spec.category,
                    "units": spec.units,
                    "value": value.tolist(),
                    "intervalHalfWidth": head.conformal_half_width,
                    "metrics": head.metrics,
                    "higherIsBetter": spec.higher_is_better,
                }

        return {
            "n": len(mols),
            "endpoints": out,
            "applicability": in_domain.tolist(),
            "leverage": h.tolist(),
            "leverageThreshold": self.leverage_threshold,
        }

    # -- persistence

    def to_json(self) -> str:
        return json.dumps({
            "version": 2,
            "alpha": self.alpha,
            "n_features": int(self.trunk_weights[0].shape[0]),
            "n_train": self.n_train,
            "trunk": [
                {"shape": list(W.shape), "weight": pack(W), "bias": pack(b)}
                for W, b in zip(self.trunk_weights, self.trunk_biases)
            ],
            "scaler": {"mean": pack(self.scaler.mean), "scale": pack(self.scaler.scale)},
            "leverage": {
                "inv_cov": pack(self.leverage_inv_cov),
                "threshold": self.leverage_threshold,
            },
            "heads": {
                name: {
                    "task": h.task, "weight": pack(h.weight), "bias": h.bias,
                    "linear": pack(h.linear) if h.linear is not None else None,
                    "blend": h.blend,
                    "platt_a": h.platt_a, "platt_b": h.platt_b,
                    "y_mean": h.y_mean, "y_std": h.y_std,
                    "conformal_half_width": h.conformal_half_width,
                    "metrics": h.metrics,
                }
                for name, h in self.heads.items()
            },
        })

    @classmethod
    def from_json(cls, text: str) -> "ADMETModel":
        blob = json.loads(text)
        weights, biases = [], []
        for layer in blob["trunk"]:
            shape = tuple(layer["shape"])
            weights.append(unpack(layer["weight"], shape))
            biases.append(unpack(layer["bias"], (shape[1],)))
        hidden = weights[-1].shape[1]
        heads = {
            name: Head(
                name=name, task=h["task"],
                weight=unpack(h["weight"], (hidden,)), bias=h["bias"],
                linear=(unpack(h["linear"], (blob["n_features"],))
                        if h.get("linear") is not None else None),
                blend=h.get("blend", 1.0),
                platt_a=h["platt_a"], platt_b=h["platt_b"],
                y_mean=h["y_mean"], y_std=h["y_std"],
                conformal_half_width=h["conformal_half_width"],
                metrics=h.get("metrics", {}),
            )
            for name, h in blob["heads"].items()
        }
        n_desc = N_DESCRIPTORS
        return cls(
            trunk_weights=weights, trunk_biases=biases, heads=heads,
            scaler=FeatureScaler(mean=unpack(blob["scaler"]["mean"], (n_desc,)),
                                 scale=unpack(blob["scaler"]["scale"], (n_desc,))),
            leverage_inv_cov=unpack(blob["leverage"]["inv_cov"], (n_desc, n_desc)),
            leverage_threshold=blob["leverage"]["threshold"],
            alpha=blob["alpha"], n_train=blob.get("n_train", 0),
        )


WEIGHTS_PATH = Path(__file__).resolve().parent / "weights.json"
_CACHED: ADMETModel | None = None


def load_model(path: Path | None = None) -> ADMETModel:
    """Load the shipped model, once per process."""
    global _CACHED
    if path is None and _CACHED is not None:
        return _CACHED
    target = path or WEIGHTS_PATH
    if not target.exists():
        raise FileNotFoundError(
            f"no trained ADMET model at {target}; run benchmarks/train_admet.py")
    model = ADMETModel.from_json(target.read_text(encoding="utf-8"))
    if path is None:
        _CACHED = model
    return model
