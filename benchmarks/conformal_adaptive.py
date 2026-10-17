"""Do locally-adaptive conformal intervals beat constant-width ones?

Split conformal with an absolute-residual score gives every molecule the same
interval width. That is valid -- marginal coverage is guaranteed either way --
but useless for triage: it cannot say which predictions to trust. Normalising
the score by a per-point difficulty estimate (here the GP posterior standard
deviation) lets width vary while keeping the same guarantee.

Whether that is an improvement is an empirical question, so this measures it:

  marginal coverage   must stay at the nominal level for both
  mean width          narrower is better at equal coverage
  conditional coverage coverage inside each difficulty quintile; the point of
                      adaptive intervals is that this stays near nominal
                      instead of over-covering easy points and under-covering
                      hard ones
  CWC                 coverage-width criterion, which penalises any shortfall

    python -m benchmarks.conformal_adaptive --datasets esol freesolv lipo
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")

import numpy as np

from nexus.al.gp import GaussianProcess
from nexus.chem.fingerprint import fingerprint_matrix
from nexus.chem.scaffold import scaffold_split
from nexus.data import load_dataset
from nexus.uq.conformal import ConformalRegressor, coverage_width_criterion

ROOT = Path(__file__).resolve().parent.parent
N_QUINTILES = 5


def conditional_coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray,
                         difficulty: np.ndarray, n_bins: int = N_QUINTILES) -> list[float]:
    """Coverage inside each difficulty bin, hardest bin last.

    @param y True values.
    @param low Interval lower bounds.
    @param high Interval upper bounds.
    @param difficulty Per-point difficulty estimate used to bin.
    @param n_bins Number of equal-count bins.
    @returns Coverage in each bin.
    """
    order = np.argsort(difficulty)
    covered = (y >= low) & (y <= high)
    return [float(np.mean(covered[chunk]))
            for chunk in np.array_split(order, n_bins) if len(chunk)]


def run_dataset(name: str, alpha: float, seeds: int) -> dict:
    """Compares constant-width and normalised intervals on one dataset."""
    dataset, labels, _, _ = load_dataset(name)
    y_all = np.asarray(labels, dtype=float).ravel()
    keep = np.isfinite(y_all)
    mols = [m for m, k in zip(dataset, keep) if k]
    y_all = y_all[keep]
    X_all = fingerprint_matrix(mols, kind="morgan", n_bits=1024).astype(np.float64)

    out: dict[str, list] = {}
    for seed in range(seeds):
        train_idx, calib_idx, test_idx = scaffold_split(
            mols, frac_train=0.6, frac_valid=0.2, seed=seed, balanced=True)
        train_idx = np.asarray(train_idx); calib_idx = np.asarray(calib_idx)
        test_idx = np.asarray(test_idx)
        if min(len(train_idx), len(calib_idx), len(test_idx)) < 20:
            continue

        model = GaussianProcess(kernel="tanimoto", noise=1e-2).fit(
            X_all[train_idx], y_all[train_idx])
        mean_cal, std_cal = model.predict(X_all[calib_idx])
        mean_test, std_test = model.predict(X_all[test_idx])
        y_cal, y_test = y_all[calib_idx], y_all[test_idx]

        # A second difficulty estimate, fitted rather than borrowed: hold an
        # inner fold out of training, and regress log|residual| on it. The GP's
        # own posterior sd measures distance from the training set, which is
        # not the same thing as being wrong.
        inner = np.random.default_rng(seed).permutation(len(train_idx))
        cut = max(10, int(0.75 * len(inner)))
        fit_idx, held_idx = train_idx[inner[:cut]], train_idx[inner[cut:]]
        resid_cal = resid_test = None
        if len(held_idx) >= 10:
            inner_model = GaussianProcess(kernel="tanimoto", noise=1e-2).fit(
                X_all[fit_idx], y_all[fit_idx])
            held_pred, _ = inner_model.predict(X_all[held_idx])
            log_resid = np.log(np.abs(y_all[held_idx] - held_pred) + 1e-3)
            difficulty = GaussianProcess(kernel="tanimoto", noise=1e-1).fit(
                X_all[held_idx], log_resid)
            resid_cal = np.exp(difficulty.predict(X_all[calib_idx])[0])
            resid_test = np.exp(difficulty.predict(X_all[test_idx])[0])

        variants = [("standard", None, None), ("gp_sigma", std_cal, std_test)]
        if resid_cal is not None:
            variants.append(("residual_fitted", resid_cal, resid_test))

        for label, sigma_cal, sigma_test in variants:
            normalize = sigma_cal is not None
            reg = ConformalRegressor(alpha=alpha, normalize=normalize)
            reg.fit(mean_cal, y_cal, sigma_cal)
            low, high = reg.predict_interval(mean_test, sigma_test)
            widths = high - low
            covered = float(np.mean((y_test >= low) & (y_test <= high)))
            binner = sigma_test if sigma_test is not None else std_test
            out.setdefault(label, []).append({
                "coverage": covered,
                "mean_width": float(np.mean(widths)),
                "width_spread": float(np.std(widths)),
                "conditional": conditional_coverage(y_test, low, high, binner),
                "cwc": float(coverage_width_criterion(covered, float(np.mean(widths)),
                                                      1.0 - alpha)),
                "n_test": int(len(test_idx)),
            })
    return out


def summarise(runs: list[dict]) -> dict:
    """Means over seeds, plus the worst conditional bin."""
    if not runs:
        return {}
    conditional = np.array([r["conditional"] for r in runs], dtype=float)
    return {
        "coverage": float(np.mean([r["coverage"] for r in runs])),
        "mean_width": float(np.mean([r["mean_width"] for r in runs])),
        "width_spread": float(np.mean([r["width_spread"] for r in runs])),
        "conditional_by_quintile": [float(v) for v in conditional.mean(axis=0)],
        "worst_quintile_coverage": float(conditional.mean(axis=0).min()),
        # How far conditional coverage drifts across difficulty. Constant-width
        # intervals over-cover easy molecules and under-cover hard ones, so a
        # large spread is the failure adaptive widths are meant to fix; the
        # worst single quintile alone does not show it.
        "conditional_spread": float(conditional.mean(axis=0).max()
                                    - conditional.mean(axis=0).min()),
        "cwc": float(np.mean([r["cwc"] for r in runs])),
        "n_seeds": len(runs),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["esol", "freesolv", "lipo"])
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    report = {"alpha": args.alpha, "nominal_coverage": 1.0 - args.alpha,
              "seeds": args.seeds, "datasets": {}}
    for name in args.datasets:
        print(f"{name}...", flush=True)
        runs = run_dataset(name, args.alpha, args.seeds)
        report["datasets"][name] = {k: summarise(v) for k, v in runs.items()}
        for label, s in report["datasets"][name].items():
            if not s:
                continue
            print(f"  {label:<11} coverage {s['coverage']:.3f}  width "
                  f"{s['mean_width']:.3f} (sd {s['width_spread']:.3f})  "
                  f"worst quintile {s['worst_quintile_coverage']:.3f}  "
                  f"spread {s['conditional_spread']:.3f}  "
                  f"CWC {s['cwc']:.3f}", flush=True)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "conformal_adaptive.json").write_text(json.dumps(report, indent=2))
    print("wrote results/conformal_adaptive.json")


if __name__ == "__main__":
    main()
