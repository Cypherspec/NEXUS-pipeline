"""Train the multi-task ADMET model and report held-out accuracy.

The eighteen TDC endpoints are pooled into one table of unique compounds, with
a mask marking which endpoints measured each one. A single scaffold split is
taken over that union, so a Bemis-Murcko scaffold that appears in the training
set appears in no endpoint's test set -- the leak that makes random splits
flatter than they should be, since the same chemical series is often assayed
across several endpoints.

Three numbers are reported per endpoint and written to results/admet_fit.json:
the model's held-out score, the score of a baseline that predicts the training
majority or mean, and the score of a logistic or ridge regression on the same
features. The third is there because a multi-task network that cannot beat a
linear model on the same inputs has not earned the complexity.

    python -m benchmarks.train_admet [--epochs 40] [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nexus.admet.features import N_DESCRIPTORS, FeatureScaler, descriptor_block, featurize
from nexus.admet.model import ADMETModel, Head, TRUNK_SIZES
from nexus.admet.tdc import ENDPOINTS, load_endpoint
from nexus.chem.scaffold import scaffold_split
from nexus.models.metrics import average_precision, roc_auc
from nexus.models.nn import (AdamW, CosineSchedule, MLP,
                             binary_cross_entropy_with_logits, mse_loss)
from nexus.models.autograd import Tensor

ROOT = Path(__file__).resolve().parent.parent
ALPHA = 0.1                      # conformal intervals at 90% coverage
RIDGE = 1.0


# ------------------------------------------------------------------ data

def build_table(quick: bool = False):
    """Pool every endpoint into one table of unique compounds.

    @returns mols, Y (n, 18) with NaN where unmeasured, mask, endpoint names.
    """
    names = list(ENDPOINTS)
    by_smiles: dict[str, dict[str, float]] = {}
    for name in names:
        smiles, y, _ = load_endpoint(name, parse=False)
        if quick:
            smiles, y = smiles[:1200], y[:1200]
        for text, value in zip(smiles, y):
            by_smiles.setdefault(text, {})[name] = float(value)

    from nexus.chem.smiles import read_smiles

    mols, rows = [], []
    for text, values in by_smiles.items():
        mol = read_smiles(text)
        if mol is None:
            continue
        mols.append(mol)
        rows.append([values.get(name, np.nan) for name in names])

    Y = np.asarray(rows, dtype=float)
    return mols, Y, ~np.isnan(Y), names


# ------------------------------------------------------------------ metrics

def endpoint_scores(task: str, y: np.ndarray, pred: np.ndarray) -> dict:
    if task == "classification":
        return {"roc_auc": float(roc_auc(y, pred)),
                "pr_auc": float(average_precision(y, pred)), "n": int(len(y))}
    resid = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "mae": float(np.mean(np.abs(resid))),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "r2": float(1 - float(np.sum(resid ** 2)) / ss_tot) if ss_tot > 0 else float("nan"),
        "n": int(len(y)),
    }


# ------------------------------------------------------------------ training

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, nargs="+", default=list(TRUNK_SIZES))
    ap.add_argument("--quick", action="store_true", help="small subset, for a smoke run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    started = time.time()
    print("loading endpoints ...", flush=True)
    mols, Y, mask, names = build_table(quick=args.quick)
    print(f"  {len(mols)} unique compounds, {int(mask.sum())} measurements, "
          f"{len(names)} endpoints  ({time.time() - started:.0f}s)", flush=True)

    train, valid, test = scaffold_split(mols, frac_train=0.8, frac_valid=0.1)
    print(f"  scaffold split: {len(train)} train, {len(valid)} valid, {len(test)} test",
          flush=True)

    # The scaler is fitted on training molecules only.
    t0 = time.time()
    descriptors = descriptor_block(mols)
    scaler = FeatureScaler.fit(descriptors[train])
    X = featurize(mols, scaler)
    print(f"  featurised to {X.shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)

    # Regression targets are standardised on the training split so that one
    # head's units cannot dominate the shared trunk's gradients.
    Z = Y.copy()
    stats = {}
    for j, name in enumerate(names):
        if ENDPOINTS[name].task != "regression":
            continue
        seen = mask[train, j]
        values = Y[train, j][seen]
        mean, std = float(values.mean()), float(values.std() or 1.0)
        stats[name] = (mean, std)
        Z[:, j] = (Y[:, j] - mean) / std
    Z = np.nan_to_num(Z, nan=0.0)
    M = mask.astype(float)

    is_class = np.array([ENDPOINTS[n].task == "classification" for n in names])

    # Each endpoint contributes equally to the loss regardless of how many
    # molecules measured it. Without this the three CYP sets (37k rows between
    # them) steer the shared trunk and DILI's 475 rows never move it.
    # Positive examples in a skewed endpoint are upweighted to its own balance
    # for the same reason: a 19%-positive endpoint otherwise trains towards
    # predicting "no".
    counts = mask[train].sum(axis=0).astype(float)
    weights = np.zeros_like(M)
    for j in range(len(names)):
        if counts[j] < 1:
            continue
        column = mask[:, j]
        weights[column, j] = 1.0 / counts[j]
        if is_class[j]:
            positives = Y[train, j][mask[train, j]] > 0.5
            rate = float(positives.mean()) if positives.size else 0.5
            if 0.0 < rate < 1.0:
                pos_scale = (1 - rate) / rate
                hit = column & (np.nan_to_num(Y[:, j]) > 0.5)
                weights[hit, j] *= pos_scale
    # Scale so the two task families enter the sum on comparable footing.
    for block in (is_class, ~is_class):
        if block.any():
            total = weights[:, block][train].sum()
            if total > 0:
                weights[:, block] *= block.sum() / total

    model = MLP([X.shape[1], *args.hidden, len(names)], activation="relu",
                dropout=args.dropout, layer_norm=True, seed=0)
    optimiser = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(1, len(train) // args.batch_size)
    schedule = CosineSchedule(optimiser, total_steps=steps, warmup=min(200, steps // 10))
    rng = np.random.default_rng(0)

    def forward(rows: np.ndarray) -> np.ndarray:
        was_training = model.training
        model.training = False
        out = []
        for start in range(0, len(rows), 2048):
            out.append(model(Tensor(X[rows[start:start + 2048]])).data)
        model.training = was_training
        return np.vstack(out)

    def validation_score(P: np.ndarray) -> float:
        """Mean over endpoints of ROC-AUC, or of R^2 floored at zero.

        One number so early stopping has something to rank epochs by, and both
        task families enter it on a 0-1 scale so neither dominates the choice.
        """
        scores = []
        for j, name in enumerate(names):
            seen = mask[valid, j]
            if seen.sum() < 10:
                continue
            y, p = Y[valid, j][seen], P[seen, j]
            if is_class[j]:
                if 0 < y.mean() < 1:
                    scores.append(roc_auc(y, p))
            else:
                mean, std = stats[name]
                resid = (p * std + mean) - y
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                scores.append(max(0.0, 1 - float(np.sum(resid ** 2)) / ss_tot)
                              if ss_tot > 0 else 0.0)
        return float(np.mean(scores)) if scores else float("nan")

    print(f"training up to {args.epochs} epochs ...", flush=True)
    best = {"score": -np.inf, "epoch": -1,
            "state": [p.data.copy() for p in model.parameters()]}
    history = []
    for epoch in range(args.epochs):
        model.training = True
        order = rng.permutation(train)
        total, batches = 0.0, 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start:start + args.batch_size]
            logits = model(Tensor(X[idx]))
            loss = (binary_cross_entropy_with_logits(
                        logits[:, is_class], Z[idx][:, is_class],
                        weights[idx][:, is_class])
                    + mse_loss(logits[:, ~is_class], Z[idx][:, ~is_class],
                               weights[idx][:, ~is_class]))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.data)
            batches += 1

        score = validation_score(forward(valid))
        history.append({"epoch": epoch, "loss": total / max(batches, 1),
                        "valid_score": score})
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": [p.data.copy() for p in model.parameters()]}
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {total / max(batches, 1):.4f}"
                  f"  valid {score:.4f}  best {best['score']:.4f}@{best['epoch']}"
                  f"  ({time.time() - started:.0f}s)", flush=True)
        # Stop once the validation score has not improved for a while: the
        # training loss keeps falling long after the model stops generalising.
        if epoch - best["epoch"] >= args.patience:
            print(f"  early stop at epoch {epoch}; best was {best['epoch']}", flush=True)
            break

    # Restore the best epoch rather than keeping the last one.
    for parameter, saved in zip(model.parameters(), best["state"]):
        parameter.data[...] = saved
    model.training = False
    P_valid, P_test = forward(valid), forward(test)
    print(f"  restored epoch {best['epoch']} (valid {best['score']:.4f})", flush=True)

    # -- a linear model per endpoint, on the same features and the same split.
    # It is both the honesty check and, where it wins, part of what ships.
    print("fitting linear comparators ...", flush=True)
    linear_weights: dict[str, np.ndarray] = {}
    L_valid = np.full_like(P_valid, np.nan)
    L_test = np.full_like(P_test, np.nan)
    for j, name in enumerate(names):
        rows = train[mask[train, j]]
        if len(rows) < 50:
            continue
        spec = ENDPOINTS[name]
        y = Y[rows, j]
        target = y if spec.task == "regression" else np.where(y > 0.5, 1.0, -1.0)
        if spec.task == "regression":
            mean, std = stats[name]
            target = (target - mean) / std
        A = X[rows].T @ X[rows] + RIDGE * np.eye(X.shape[1])
        beta = np.linalg.solve(A, X[rows].T @ target)
        linear_weights[name] = beta
        L_valid[:, j] = X[valid] @ beta
        L_test[:, j] = X[test] @ beta

    # -- blend. The weight is chosen per endpoint on the validation split, so
    # an endpoint where the network adds nothing simply gets the linear model.
    blend: dict[str, float] = {}
    grid = np.linspace(0.0, 1.0, 11)
    for j, name in enumerate(names):
        if name not in linear_weights:
            blend[name] = 1.0
            continue
        seen = mask[valid, j]
        if seen.sum() < 20:
            blend[name] = 0.5
            continue
        y = Y[valid, j][seen]
        net, lin = P_valid[seen, j], L_valid[seen, j]
        best_w, best_s = 1.0, -np.inf
        for w in grid:
            combined = w * net + (1 - w) * lin
            if is_class[j]:
                if not 0 < y.mean() < 1:
                    continue
                s = roc_auc(y, combined)
            else:
                mean, std = stats[name]
                resid = (combined * std + mean) - y
                s = -float(np.mean(resid ** 2))
            if s > best_s:
                best_s, best_w = s, float(w)
        blend[name] = best_w

    def combine(P, L, j, name):
        if name not in linear_weights:
            return P[:, j]
        w = blend[name]
        return w * P[:, j] + (1 - w) * L[:, j]

    # -- calibrate, score, and assemble the shipped model
    hidden_dim = args.hidden[-1]
    trunk_weights = [l.weight.data for l in model.linears[:-1]]
    trunk_biases = [l.bias.data for l in model.linears[:-1]]
    final = model.linears[-1]

    heads, report = {}, {}
    for j, name in enumerate(names):
        spec = ENDPOINTS[name]
        seen_v, seen_t = mask[valid, j], mask[test, j]
        yv, yt = Y[valid, j][seen_v], Y[test, j][seen_t]
        pv = combine(P_valid, L_valid, j, name)[seen_v]
        pt = combine(P_test, L_test, j, name)[seen_t]

        head = Head(name=name, task=spec.task,
                    weight=final.weight.data[:, j].copy(),
                    bias=float(final.bias.data[j]),
                    linear=linear_weights.get(name),
                    blend=blend.get(name, 1.0))

        entry: dict = {"category": spec.category, "task": spec.task,
                       "units": spec.units, "n_test": int(seen_t.sum())}

        if spec.task == "classification":
            # Platt scaling on the validation split: two parameters fitted by
            # Newton steps on the logistic likelihood.
            a, b = 1.0, 0.0
            if len(yv) > 20 and 0 < yv.mean() < 1:
                a, b = _fit_platt(pv, yv)
            head.platt_a, head.platt_b = float(a), float(b)
            probs = 1.0 / (1.0 + np.exp(-np.clip(a * pt + b, -30, 30)))
            entry["model"] = endpoint_scores("classification", yt, probs)
            entry["baseline_majority"] = {
                "roc_auc": 0.5, "n": int(len(yt))}
        else:
            mean, std = stats[name]
            head.y_mean, head.y_std = mean, std
            values = pt * std + mean
            entry["model"] = endpoint_scores("regression", yt, values)
            entry["baseline_mean"] = endpoint_scores(
                "regression", yt, np.full(len(yt), Y[train, j][mask[train, j]].mean()))
            # Split-conformal half width from the validation residuals.
            resid = np.abs((pv * std + mean) - yv)
            if len(resid) >= 20:
                k = int(np.ceil((len(resid) + 1) * (1 - ALPHA))) - 1
                head.conformal_half_width = float(np.sort(resid)[min(k, len(resid) - 1)])
                covered = np.abs(values - yt) <= head.conformal_half_width
                entry["conformal"] = {
                    "half_width": head.conformal_half_width,
                    "empirical_coverage": float(covered.mean()),
                    "target_coverage": 1 - ALPHA,
                }

        # The linear model alone and the network alone, both on this same
        # split, so the blend can never flatter itself.
        if name in linear_weights:
            lin = L_test[seen_t, j]
            if spec.task == "regression":
                lin = lin * stats[name][1] + stats[name][0]
            entry["baseline_linear"] = endpoint_scores(spec.task, yt, lin)
            net_only = P_test[seen_t, j]
            if spec.task == "regression":
                net_only = net_only * stats[name][1] + stats[name][0]
            entry["network_only"] = endpoint_scores(spec.task, yt, net_only)
            entry["blend_weight"] = blend[name]

        head.metrics = entry["model"]
        heads[name] = head
        report[name] = entry

        key = "roc_auc" if spec.task == "classification" else "mae"
        print(f"  {name:26s} {spec.task[:5]:5s} n={entry['n_test']:5d} "
              f"{key} {entry['model'][key]:.3f}", flush=True)

    # Applicability domain on the training descriptor block.
    D = scaler.transform(descriptors[train])
    cov = np.cov(D, rowvar=False) + 1e-6 * np.eye(N_DESCRIPTORS)
    inv_cov = np.linalg.inv(cov)
    # Williams' h* = 3p/n, the usual QSAR cut for an outlier in model space.
    threshold = float(3.0 * N_DESCRIPTORS / max(len(train), 1))
    train_leverage = np.einsum("ij,jk,ik->i", D, inv_cov, D)
    # Expressed as a quantile instead when the asymptotic cut is implausible
    # for this sample size, so the flag stays meaningful either way.
    threshold = max(threshold, float(np.quantile(train_leverage, 0.99)))

    admet = ADMETModel(
        trunk_weights=trunk_weights, trunk_biases=trunk_biases, heads=heads,
        scaler=scaler, leverage_inv_cov=inv_cov, leverage_threshold=threshold,
        alpha=ALPHA, n_train=len(train))

    out = Path(args.out) if args.out else ROOT / "nexus" / "admet" / "weights.json"
    out.write_text(admet.to_json(), encoding="utf-8")

    summary = {
        "n_compounds": len(mols),
        "n_measurements": int(mask.sum()),
        "n_endpoints": len(names),
        "split": "scaffold over the pooled union (80/10/10)",
        "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
        "epochs": args.epochs,
        "hidden": args.hidden,
        "features": int(X.shape[1]),
        "alpha": ALPHA,
        "leverage_threshold": threshold,
        "weights_bytes": out.stat().st_size,
        "seconds": round(time.time() - started, 1),
        "endpoints": report,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "admet_fit.json").write_text(json.dumps(summary, indent=2))

    classification = [v for v in report.values() if v["task"] == "classification"]
    regression = [v for v in report.values() if v["task"] == "regression"]
    print(f"\nmean held-out ROC-AUC {np.mean([v['model']['roc_auc'] for v in classification]):.3f}"
          f" over {len(classification)} classification endpoints")
    print(f"mean held-out R^2     {np.mean([v['model']['r2'] for v in regression]):.3f}"
          f" over {len(regression)} regression endpoints")
    print(f"weights {out.stat().st_size / 1e6:.2f} MB -> {out}")


def _fit_platt(scores: np.ndarray, y: np.ndarray, iterations: int = 60) -> tuple[float, float]:
    """Two-parameter Platt scaling by Newton's method on the logistic loss."""
    a, b = 1.0, 0.0
    for _ in range(iterations):
        z = np.clip(a * scores + b, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-9, None)
        g = np.array([np.sum((p - y) * scores), np.sum(p - y)])
        H = np.array([[np.sum(w * scores * scores), np.sum(w * scores)],
                      [np.sum(w * scores), np.sum(w)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        a -= float(step[0])
        b -= float(step[1])
        if np.max(np.abs(step)) < 1e-9:
            break
    return a, b


if __name__ == "__main__":
    main()
