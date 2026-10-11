"""Score candidate NEXUS designs on the development datasets from cached members.

    python -m benchmark.dev_eval
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from benchmark.dev import DEV_DATASETS, N_OUTER, member_path
from nexus.calibration import apply_temperature, fit_temperature
from nexus.ensemble import greedy_ensemble_selection, mix
from nexus.metrics import all_metrics


def greedy_temp(oof, test, y, n_iter=50):
    w = greedy_ensemble_selection(oof, y, n_iter=n_iter)
    t = fit_temperature(mix(oof, w), y)
    return apply_temperature(mix(test, w), t)


def bagged_temp(oof, test, y):
    from nexus.ensemble import bagged_greedy_selection
    w = bagged_greedy_selection(oof, y)
    t = fit_temperature(mix(oof, w), y)
    return apply_temperature(mix(test, w), t)


def convex_temp(oof, test, y):
    from nexus.ensemble import optimal_convex_weights
    w = optimal_convex_weights(oof, y)
    t = fit_temperature(mix(oof, w), y)
    return apply_temperature(mix(test, w), t)


def load(did, fold, names):
    """Member names may be "a+b+c": repetitions whose OOF and test preds are averaged.
    "a|b+c" uses OOF from a only but averages test preds over a, b, c."""
    oofs, tests, d0 = [], [], None
    for n in names:
        oof_names, _, extra = n.partition("|")
        sel = oof_names.split("+")
        allt = sel + (extra.split("+") if extra else [])
        zs = {m: np.load(member_path(did, fold, m)) for m in dict.fromkeys(allt)}
        d0 = d0 or zs[sel[0]]
        oofs.append(np.mean([zs[m]["oof"] for m in sel], axis=0))
        tests.append(np.mean([zs[m]["test"] for m in allt], axis=0))
    return (np.stack(oofs).astype(np.float64), np.stack(tests).astype(np.float64),
            d0["y_train"].astype(int), d0["y_test"].astype(int))


def evaluate(variants: dict[str, tuple[list[str], callable]]) -> pd.DataFrame:
    rows = []
    for did in DEV_DATASETS:
        for f in range(N_OUTER):
            for name, (members, combine) in variants.items():
                try:
                    oof, test, y_tr, y_te = load(did, f, members)
                except FileNotFoundError:
                    continue
                rows.append({"did": did, "fold": f, "variant": name, **all_metrics(y_te, combine(oof, test, y_tr))})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    # Only datasets where every variant has all folds.
    full = df.groupby(["did", "variant"]).fold.nunique().unstack()
    keep = full.index[(full == N_OUTER).all(axis=1)]
    t = df[df.did.isin(keep)].groupby(["did", "variant"])[["log_loss", "accuracy", "roc_auc"]].mean()
    ll = t.log_loss.unstack()
    acc = t.accuracy.unstack()
    out = pd.DataFrame({
        "n_datasets": len(keep),
        "mean_rank_ll": ll.rank(axis=1).mean(),
        "rel_ll_vs_ref_%": 100 * ((ll[ref].to_numpy()[:, None] - ll) / ll[ref].to_numpy()[:, None]).mean(),
        "wins_ll_vs_ref": (ll.lt(ll[ref], axis=0)).sum(),
        "mean_acc": acc.mean(),
        "rel_err_vs_ref_%": 100 * (((1 - acc[ref]).to_numpy()[:, None] - (1 - acc)) / (1 - acc[ref]).clip(1e-9).to_numpy()[:, None]).mean(),
    })
    return out.sort_values("mean_rank_ll")


if __name__ == "__main__":
    from benchmark.dev import V1
    variants = {"v1": (V1, greedy_temp)}
    print(summarize(evaluate(variants), "v1").to_string())
