"""Recompute every compared method and ablation from saved fold predictions.

All ensemble variants share the exact same bagged base-model predictions, so
differences between them isolate the combination strategy.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from tabular_ensemble.calibration import apply_temperature, fit_temperature
from tabular_ensemble.ensemble import EPS, greedy_ensemble_selection, log_loss_from_proba, mix
from tabular_ensemble.zoo import ZOO_BY_NAME

DISPLAY = {
    "L2:lightgbm": "L2 LightGBM", "L2:logreg": "L2 logistic regression",
    "lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost",
    "random_forest": "Random forest", "extra_trees": "Extra trees",
    "logreg": "Logistic regression", "knn": "k-NN", "mlp": "MLP",
}
FAMILIES = ["gbdt", "bagged_trees", "linear", "neighbors", "neural"]
FAMILY_DISPLAY = {
    "L2:lightgbm": "L2 LightGBM", "L2:logreg": "L2 logistic regression","gbdt": "GBDTs", "bagged_trees": "RF/ET", "linear": "linear",
                  "neighbors": "k-NN", "neural": "MLP"}

MAIN_METHODS = [
    "NEXUS", "NEXUS v1", "AutoGluon", "Stacking (LR)", "Uniform average",
    "Best model by CV", "LightGBM (tuned)", *(v for k, v in DISPLAY.items() if not k.startswith("L2:")),
]
ENSEMBLE_SIZES = [1, 2, 5, 10, 25, 50, 100]


def _nexus(oof, test, y, n_iter=50, calibrate=True):
    w = greedy_ensemble_selection(oof, y, n_iter=n_iter)
    t = fit_temperature(mix(oof, w), y) if calibrate else 1.0
    return apply_temperature(mix(test, w), t), w, t


def _stacking(oof, test, y):
    feats = lambda P: np.log(np.clip(P, EPS, 1)).transpose(1, 0, 2).reshape(P.shape[1], -1)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(feats(oof), y)
    out = np.zeros(test.shape[1:])
    out[:, lr.classes_] = lr.predict_proba(feats(test))
    return out


def compute_methods(fold: dict) -> tuple[dict[str, np.ndarray], dict]:
    names = list(fold["names"])
    oof, test, y = fold["oof"].astype(np.float64), fold["bagged_test"].astype(np.float64), fold["y_train"].astype(int)
    k = test.shape[2]
    out: dict[str, np.ndarray] = {}
    extra: dict = {}

    out["NEXUS v1"], w, t = _nexus(oof, test, y)
    extra["weights"] = dict(zip(names, w))
    extra["temperature"] = t
    extra["max_abs_diff_vs_saved"] = float(np.abs(out["NEXUS v1"] - fold["nexus_test"]).max())

    out["Stacking (LR)"] = _stacking(oof, test, y)
    out["Uniform average"] = test.mean(axis=0)
    cv_losses = [log_loss_from_proba(y, p) for p in oof]
    out["Best model by CV"] = test[int(np.argmin(cv_losses))]
    extra["best_by_cv"] = names[int(np.argmin(cv_losses))]
    out["LightGBM (tuned)"] = fold["tuned_lgbm_test"].astype(np.float64)
    for i, n in enumerate(names):
        out[DISPLAY[n]] = fold["single_test"][i].astype(np.float64)
        out[f"{DISPLAY[n]} (bagged)"] = test[i]

    # Ablations
    out["NEXUS v1 w/o calibration"], _, _ = _nexus(oof, test, y, calibrate=False)
    out["Uniform average + calibration"] = apply_temperature(
        test.mean(axis=0), fit_temperature(oof.mean(axis=0), y))
    for fam in FAMILIES:
        keep = [i for i, n in enumerate(names) if ZOO_BY_NAME[n].family != fam]
        out[f"NEXUS v1 w/o {FAMILY_DISPLAY[fam]}"], _, _ = _nexus(oof[keep], test[keep], y)
    gbdt = [i for i, n in enumerate(names) if ZOO_BY_NAME[n].family == "gbdt"]
    out["NEXUS v1 (GBDTs only)"], _, _ = _nexus(oof[gbdt], test[gbdt], y)
    for m in ENSEMBLE_SIZES:
        out[f"NEXUS v1 (size {m})"], _, _ = _nexus(oof, test, y, n_iter=m)

    for key, p in out.items():
        assert p.shape == (len(fold["y_test"]), k), key
    return out, extra
