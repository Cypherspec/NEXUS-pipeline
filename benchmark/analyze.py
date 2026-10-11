"""Turn raw fold predictions into metrics, statistics and tables.

    python -m benchmark.analyze

Writes results/fold_metrics.csv, results/task_metrics.csv, results/fold_info.csv,
results/stats/*.csv, results/summary.json and paper/tables/*.tex|md.
Only tasks with all 10 folds complete enter the statistics.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from benchmark.methods import DISPLAY, ENSEMBLE_SIZES, MAIN_METHODS
from nexus.metrics import LOWER_IS_BETTER, all_metrics

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
RES = ROOT / "results"
TABLES = ROOT / "paper" / "tables"
METRICS = ["log_loss", "roc_auc", "accuracy", "balanced_accuracy", "brier", "ece"]
METRIC_NAMES = {"log_loss": "Log loss", "roc_auc": "ROC-AUC", "accuracy": "Accuracy",
                "balanced_accuracy": "Balanced accuracy", "brier": "Brier score", "ece": "ECE"}
N_BINS = 15
RELIABILITY_METHODS = ["NEXUS", "NEXUS v1", "NEXUS v1 w/o calibration", "AutoGluon", "Uniform average", "LightGBM (tuned)"]
RAW_V2 = ROOT / "results" / "raw_v2"
RAW_AG = ROOT / "results" / "raw_autogluon"
N_FOLDS = int(os.environ.get("NEXUS_MIN_FOLDS", "10"))
ABLATION_REF = "NEXUS v1"
ABLATIONS = ["NEXUS v1 w/o calibration", "Uniform average + calibration",
             "NEXUS v1 w/o GBDTs", "NEXUS v1 w/o RF/ET", "NEXUS v1 w/o linear", "NEXUS v1 w/o k-NN",
             "NEXUS v1 w/o MLP", "NEXUS v1 (GBDTs only)", "Best model by CV", "Uniform average",
             "Stacking (LR)"]


def _process(path: Path):
    from benchmark.methods import compute_methods

    fold = dict(np.load(path, allow_pickle=False))
    task, f = int(path.parent.name), int(path.stem[4:])
    methods, extra = compute_methods(fold)
    y = fold["y_test"].astype(int)
    v2_path = RAW_V2 / path.parent.name / path.name
    ag_path = RAW_AG / path.parent.name / path.name
    v2 = dict(np.load(v2_path)) if v2_path.exists() else None
    if v2 is not None:
        assert np.array_equal(v2["y_test"], fold["y_test"])
        methods["NEXUS"] = v2["nexus_test"].astype(np.float64)
    if ag_path.exists():
        ag = np.load(ag_path)
        assert np.array_equal(ag["y_test"], fold["y_test"])
        methods["AutoGluon"] = ag["proba"].astype(np.float64)
    prior = np.bincount(fold["y_train"].astype(int), minlength=fold["oof"].shape[2]) / len(fold["y_train"])
    prior_ll = float(-np.mean(np.log(np.clip(prior[y], 1e-6, 1))))
    rows = [{"task_id": task, "fold": f, "method": m, "prior_ll": prior_ll, **all_metrics(y, p)}
            for m, p in methods.items()]
    rel = []
    for m in RELIABILITY_METHODS:
        if m not in methods:
            continue
        p = methods[m]
        conf, correct = p.max(axis=1), p.argmax(axis=1) == y
        bins = np.minimum((conf * N_BINS).astype(int), N_BINS - 1)
        for b in range(N_BINS):
            sel = bins == b
            rel.append({"task_id": task, "fold": f, "method": m, "bin": b, "count": int(sel.sum()),
                        "conf_sum": float(conf[sel].sum()), "correct_sum": float(correct[sel].sum())})
    info = {
        "task_id": task, "fold": f, "temperature": extra["temperature"],
        "best_by_cv": extra["best_by_cv"], "recompute_max_abs_diff": extra["max_abs_diff_vs_saved"],
        "nexus_fit_seconds": float(fold["nexus_fit_seconds"]),
        "nexus_predict_seconds": float(fold["nexus_predict_seconds"]),
        "tuned_lgbm_seconds": float(fold["tuned_lgbm_seconds"]),
        **{f"w_{n}": w for n, w in extra["weights"].items()},
        **{f"single_seconds_{n}": s for n, s in zip(fold["names"], fold["single_fit_seconds"])},
        **{f"member_seconds_{n}": s for n, s in zip(fold["names"], fold["member_fit_seconds"])},
    }
    if v2 is not None:
        info["v2_fit_seconds"] = float(v2["nexus_fit_seconds"])
        info["v2_temperature"] = float(v2["nexus_temperature"])
        info.update({f"w2_{n}": w for n, w in zip(v2["names"], v2["nexus_weights"])})
    if ag_path.exists():
        info["ag_fit_seconds"] = float(ag["fit_seconds"])
        info["ag_budget"] = float(ag["budget"])
        info["ag_best_model"] = str(ag["best_model"])
    return rows, info, rel


def collect(workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(RAW.glob("*/fold*.npz"))
    paths = [p for p in paths if not p.name.endswith(".tmp.npz")]
    with ProcessPoolExecutor(workers) as pool:
        out = list(pool.map(_process, paths, chunksize=4))
    fm = pd.DataFrame([r for rows, _, _ in out for r in rows])
    info = pd.DataFrame([i for _, i, _ in out])
    rel = pd.DataFrame([r for _, _, rel in out for r in rel])
    return fm, info, rel


# ----------------------------------------------------------------- statistics

def rank_matrix(task: pd.DataFrame, methods: list[str], metric: str) -> pd.DataFrame:
    wide = task[task.method.isin(methods)].pivot(index="task_id", columns="method", values=metric)[methods]
    return wide.rank(axis=1, ascending=LOWER_IS_BETTER[metric], method="average")


def friedman_nemenyi(ranks: pd.DataFrame, alpha=0.05) -> dict:
    n, k = ranks.shape
    R = ranks.mean()
    chi2 = 12 * n / (k * (k + 1)) * (np.sum(R ** 2) - k * (k + 1) ** 2 / 4)
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2)
    p = 1 - stats.f.cdf(ff, k - 1, (k - 1) * (n - 1))
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * n))
    return {"n_tasks": n, "k_methods": k, "friedman_chi2": chi2, "iman_davenport_F": ff,
            "p_value": p, "nemenyi_cd": cd, "mean_ranks": R.sort_values().to_dict()}


def holm(pvals: np.ndarray) -> np.ndarray:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def bootstrap_ci(x: np.ndarray, stat=np.mean, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def pairwise_vs(task: pd.DataFrame, ref: str, others: list[str], metric: str) -> pd.DataFrame:
    wide = task.pivot(index="task_id", columns="method", values=metric)
    lower = LOWER_IS_BETTER[metric]
    rows = []
    for o in others:
        a, b = wide[ref], wide[o]
        diff = (b - a) if lower else (a - b)  # >0 means ref is better
        tol = 1e-9
        wins, losses = int((diff > tol).sum()), int((diff < -tol).sum())
        ties = len(diff) - wins - losses
        p = stats.wilcoxon(a, b, zero_method="zsplit").pvalue if (diff.abs() > tol).any() else 1.0
        if metric in ("log_loss", "brier", "ece"):
            rel = (b - a) / b  # relative reduction of the loss, ref vs other
        else:
            rel = (a - b) / (1 - b).clip(lower=1e-12)  # relative reduction of (1 - score)
        lo, hi = bootstrap_ci(rel.to_numpy())
        h = task[task.method == ref].set_index("task_id").prior_ll.reindex(wide.index)
        dr2 = 100 * diff / h if metric == "log_loss" else pd.Series(np.nan, index=wide.index)
        dlo, dhi = bootstrap_ci(dr2.to_numpy()) if metric == "log_loss" else (np.nan, np.nan)
        rows.append({"reference": ref, "versus": o, "metric": metric, "wins": wins, "ties": ties,
                     "losses": losses, "p_wilcoxon": p, "mean_rel_improvement": rel.mean(),
                     "median_rel_improvement": rel.median(), "ci95_low": lo, "ci95_high": hi,
                     "mean_ref": a.mean(), "mean_other": b.mean(),
                     "mean_dR2_pp": dr2.mean(), "dR2_ci95_low": dlo, "dR2_ci95_high": dhi})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_wilcoxon"].to_numpy())
    return df


# ----------------------------------------------------------------- tables

def _fmt_p(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def write_tables(task, stats_main, pair, abl, ranks_by_metric, tasks_meta, info):
    TABLES.mkdir(parents=True, exist_ok=True)

    # Table 1: main results: mean rank per metric + mean score
    rows = []
    for m in MAIN_METHODS:
        r = {"Method": m}
        for met in ["log_loss", "roc_auc", "accuracy", "ece"]:
            r[f"{METRIC_NAMES[met]} rank"] = ranks_by_metric[met][m]
            r[f"{METRIC_NAMES[met]} mean"] = task[task.method == m][met].mean()
        rows.append(r)
    t1 = pd.DataFrame(rows).sort_values("Log loss rank")
    t1.to_csv(RES / "stats" / "table_main.csv", index=False)
    with open(TABLES / "main.md", "w") as fh:
        fh.write("| Method | Log loss (rank) | ROC-AUC (rank) | Accuracy (rank) | ECE (rank) |\n|---|---|---|---|---|\n")
        for _, r in t1.iterrows():
            fh.write(f"| {r.Method} | {r['Log loss mean']:.4f} ({r['Log loss rank']:.2f}) | "
                     f"{r['ROC-AUC mean']:.4f} ({r['ROC-AUC rank']:.2f}) | "
                     f"{r['Accuracy mean']:.4f} ({r['Accuracy rank']:.2f}) | "
                     f"{r['ECE mean']:.4f} ({r['ECE rank']:.2f}) |\n")
    with open(TABLES / "main.tex", "w") as fh:
        fh.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n"
                 "& \\multicolumn{2}{c}{Log loss $\\downarrow$} & \\multicolumn{2}{c}{ROC-AUC $\\uparrow$} "
                 "& \\multicolumn{2}{c}{Accuracy $\\uparrow$} & \\multicolumn{2}{c}{ECE $\\downarrow$}\\\\\n"
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\n"
                 "Method & mean & rank & mean & rank & mean & rank & mean & rank\\\\\n\\midrule\n")
        best = {c: (t1[c].min() if ("rank" in c or c.startswith(("Log", "ECE"))) else t1[c].max())
                for c in t1.columns if c != "Method"}
        for _, r in t1.iterrows():
            cells = []
            for met in ["Log loss", "ROC-AUC", "Accuracy", "ECE"]:
                for suf, fmt in [("mean", ".4f"), ("rank", ".2f")]:
                    c = f"{met} {suf}"
                    s = format(r[c], fmt)
                    cells.append(f"\\textbf{{{s}}}" if np.isclose(r[c], best[c]) else s)
            name = r.Method.replace("&", "\\&")
            fh.write(f"{name} & " + " & ".join(cells) + "\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # Table 2: pairwise NEXUS vs baselines (log loss, AUC, accuracy)
    sub = pair[pair.versus.isin(MAIN_METHODS)]
    with open(TABLES / "pairwise.md", "w") as fh:
        fh.write("| NEXUS vs. | Metric | W / T / L | Mean rel. improvement [95% CI] | Holm p |\n|---|---|---|---|---|\n")
        for _, r in sub.iterrows():
            fh.write(f"| {r.versus} | {METRIC_NAMES[r.metric]} | {r.wins} / {r.ties} / {r.losses} | "
                     f"{100*r.mean_rel_improvement:+.1f}% [{100*r.ci95_low:+.1f}, {100*r.ci95_high:+.1f}] | {_fmt_p(r.p_holm)} |\n")
    with open(TABLES / "pairwise.tex", "w") as fh:
        fh.write("\\begin{tabular}{llcrr}\n\\toprule\nNEXUS vs. & Metric & W/T/L & Rel. impr. [95\\% CI] & $p_\\text{Holm}$\\\\\n\\midrule\n")
        for i, (_, r) in enumerate(sub.iterrows()):
            fh.write(f"{r.versus} & {METRIC_NAMES[r.metric]} & {r.wins}/{r.ties}/{r.losses} & "
                     f"{100*r.mean_rel_improvement:+.1f}\\% [{100*r.ci95_low:+.1f}, {100*r.ci95_high:+.1f}] & {_fmt_p(r.p_holm)}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # Table 3: ablations (log loss)
    with open(TABLES / "ablation.md", "w") as fh:
        fh.write("| Variant | Mean log loss | NEXUS v1 rel. improvement [95% CI] | NEXUS v1 W / T / L | Holm p |\n|---|---|---|---|---|\n")
        base = task[task.method == ABLATION_REF].set_index("task_id").log_loss
        fh.write(f"| NEXUS v1 (full) | {base.mean():.4f} | — | — | — |\n")
        for _, r in abl.iterrows():
            fh.write(f"| {r.versus} | {r.mean_other:.4f} | {100*r.mean_rel_improvement:+.1f}% "
                     f"[{100*r.ci95_low:+.1f}, {100*r.ci95_high:+.1f}] | {r.wins} / {r.ties} / {r.losses} | {_fmt_p(r.p_holm)} |\n")
    with open(TABLES / "ablation.tex", "w") as fh:
        fh.write("\\begin{tabular}{lrrcr}\n\\toprule\nVariant & Log loss & v1 rel. impr. [95\\% CI] & W/T/L & $p_\\text{Holm}$\\\\\n\\midrule\n")
        fh.write(f"NEXUS v1 (full) & {base.mean():.4f} & --- & --- & ---\\\\\n")
        for _, r in abl.iterrows():
            fh.write(f"{r.versus} & {r.mean_other:.4f} & {100*r.mean_rel_improvement:+.1f}\\% "
                     f"[{100*r.ci95_low:+.1f}, {100*r.ci95_high:+.1f}] & {r.wins}/{r.ties}/{r.losses} & {_fmt_p(r.p_holm)}\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")

    # Table 4: per-task results (appendix)
    meta = tasks_meta.set_index("task_id")
    wide_ll = task.pivot(index="task_id", columns="method", values="log_loss")
    wide_acc = task.pivot(index="task_id", columns="method", values="accuracy")
    cols = [c for c in ["NEXUS", "NEXUS v1", "AutoGluon", "Stacking (LR)", "LightGBM (tuned)", "CatBoost"] if c in wide_ll.columns]
    with open(TABLES / "per_task.md", "w") as fh:
        fh.write("| Task | Dataset | n | d | k | " + " | ".join(cols) + " |\n|" + "---|" * (5 + len(cols)) + "\n")
        for tid in wide_ll.sort_values("NEXUS").index:
            mm = meta.loc[tid]
            best = wide_ll.loc[tid, cols].min()
            cells = [(f"**{wide_ll.loc[tid, c]:.4f}**" if np.isclose(wide_ll.loc[tid, c], best) else f"{wide_ll.loc[tid, c]:.4f}") for c in cols]
            fh.write(f"| {tid} | {mm['name']} | {mm.n_instances} | {mm.n_features} | {mm.n_classes} | " + " | ".join(cells) + " |\n")
    with open(TABLES / "per_task.tex", "w") as fh:
        fh.write("\\begin{longtable}{llrrr" + "r" * len(cols) + "}\n\\toprule\nTask & Dataset & $n$ & $d$ & $k$ & "
                 + " & ".join(c.replace("Best model by CV", "Best-CV").replace("Stacking (LR)", "Stack.").replace("LightGBM (tuned)", "LGBM-T") for c in cols)
                 + "\\\\\n\\midrule\n\\endhead\n")
        for tid in wide_ll.sort_values("NEXUS").index:
            mm = meta.loc[tid]
            best = wide_ll.loc[tid, cols].min()
            cells = [(f"\\textbf{{{wide_ll.loc[tid, c]:.4f}}}" if np.isclose(wide_ll.loc[tid, c], best) else f"{wide_ll.loc[tid, c]:.4f}") for c in cols]
            name = str(mm['name']).replace("_", "\\_")
            fh.write(f"{tid} & {name[:28]} & {mm.n_instances} & {mm.n_features} & {mm.n_classes} & " + " & ".join(cells) + "\\\\\n")
        fh.write("\\bottomrule\n\\end{longtable}\n")


def _macro_name(*parts) -> str:
    out = ""
    for p in parts:
        p = str(p)
        for a, b in {"(": "", ")": "", "-": "", "/": "", "+": "Plus", " ": "", "1": "One",
                     "2": "Two", "5": "Five", "0": "Zero", "_": ""}.items():
            p = p.replace(a, b)
        out += p[:1].upper() + p[1:]
    return out


def write_numbers(summary, pair, abl, abl_acc, task, info):
    """LaTeX macros so every number quoted in the paper comes from the data."""
    m = {"NTasks": summary["n_tasks_complete"], "NFolds": summary["n_tasks_complete"] * N_FOLDS}
    for met, st in summary["friedman"].items():
        key = _macro_name(met)
        m[f"FriedmanP{key}"] = f"{st['p_value']:.1e}"
        m[f"CD{key}"] = f"{st['nemenyi_cd']:.2f}"
        ranks = pd.Series(st["mean_ranks"]).sort_values()
        m[f"NexusRank{key}"] = f"{ranks['NEXUS']:.2f}"
        m[f"NexusPlace{key}"] = int(list(ranks.index).index("NEXUS") + 1)
        nxt = ranks.drop("NEXUS")
        m[f"BestOtherName{key}"] = nxt.index[0]
        m[f"BestOtherRank{key}"] = f"{nxt.iloc[0]:.2f}"
    for _, r in pair.iterrows():
        k = _macro_name(r.versus, r.metric)
        m[f"Win{k}"], m[f"Tie{k}"], m[f"Loss{k}"] = r.wins, r.ties, r.losses
        m[f"Rel{k}"] = f"{100 * r.mean_rel_improvement:.1f}"
        m[f"RelMed{k}"] = f"{100 * r.median_rel_improvement:.1f}"
        m[f"CILo{k}"] = f"{100 * r.ci95_low:.1f}"
        m[f"CIHi{k}"] = f"{100 * r.ci95_high:.1f}"
        m[f"P{k}"] = _fmt_p(r.p_holm)
        if r.metric == "log_loss":
            m[f"DRTwo{k}"] = f"{r.mean_dR2_pp:.2f}"
            m[f"DRTwoLo{k}"] = f"{r.dR2_ci95_low:.2f}"
            m[f"DRTwoHi{k}"] = f"{r.dR2_ci95_high:.2f}"
    for df, met in [(abl, "logloss"), (abl_acc, "accuracy")]:
        for _, r in df.iterrows():
            k = _macro_name("Abl", r.versus, met)
            m[f"Rel{k}"] = f"{100 * r.mean_rel_improvement:.1f}"
            m[f"CILo{k}"] = f"{100 * r.ci95_low:.1f}"
            m[f"CIHi{k}"] = f"{100 * r.ci95_high:.1f}"
            m[f"P{k}"] = _fmt_p(r.p_holm)
            m[f"Win{k}"], m[f"Loss{k}"] = r.wins, r.losses
    means = task.groupby("method")[METRICS].mean()
    for meth in ["NEXUS", "NEXUS v1", "AutoGluon", "LightGBM (tuned)", "Best model by CV", "Stacking (LR)", "CatBoost",
                 "NEXUS v1 w/o calibration"]:
        if meth not in means.index:
            continue
        for met in METRICS:
            m[f"Mean{_macro_name(meth, met)}"] = f"{means.loc[meth, met]:.4f}"
    t = summary["timing_median_seconds"]
    m["MedianFitNexus"] = f"{t.get('v2_fit_seconds', np.nan):.1f}"
    m["MedianFitVOne"] = f"{t['nexus_fit_seconds']:.1f}"
    m["MedianFitAutoGluon"] = f"{t.get('ag_fit_seconds', np.nan):.1f}"
    m["MedianFitTuned"] = f"{t['tuned_lgbm_seconds']:.1f}"
    m["MedianFitSingles"] = f"{t['all_singles_seconds']:.1f}"
    m["MeanMembers"] = f"{summary['mean_nonzero_members']:.1f}"
    m["TempMedian"] = f"{summary['temperature_median']:.2f}"
    m["TempQOne"], m["TempQThree"] = (f"{x:.2f}" for x in summary["temperature_iqr"])
    m["OracleFrac"] = f"{100 * summary['nexus_beats_test_oracle_single_frac']:.0f}"
    m["RecomputeDiff"] = f"{summary['recompute_max_abs_diff']:.0e}"
    with open(ROOT / "paper" / "numbers.tex", "w") as fh:
        fh.write("% Auto-generated by benchmark/analyze.py. Do not edit.\n")
        for k, v in m.items():
            v = str(v).replace("%", "\\%").replace("&", "\\&")
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")


def main():
    (RES / "stats").mkdir(parents=True, exist_ok=True)
    fm, info, rel = collect(workers=int(os.environ.get("WORKERS", "4")))
    fm.to_csv(RES / "fold_metrics.csv", index=False)
    info.to_csv(RES / "fold_info.csv", index=False)

    present = [m for m in MAIN_METHODS if m in set(fm.method)]
    missing = [m for m in MAIN_METHODS if m not in present]
    if missing:
        print("WARNING: methods with no results yet, excluded:", missing)
    MAIN_METHODS[:] = present
    cnt = fm[fm.method.isin(present)].groupby(["task_id", "method"]).fold.nunique().unstack()
    complete = cnt.index[(cnt.reindex(columns=present) >= N_FOLDS).all(axis=1)]
    fm = fm[fm.task_id.isin(complete)]
    info = info[info.task_id.isin(complete)]
    rel = rel[rel.task_id.isin(complete)]
    rel.groupby(["method", "bin"])[["count", "conf_sum", "correct_sum"]].sum().to_csv(RES / "stats" / "reliability.csv")
    task = fm.groupby(["task_id", "method"], as_index=False)[METRICS + ["prior_ll"]].mean()
    task.to_csv(RES / "task_metrics.csv", index=False)
    tasks_meta = pd.read_csv(ROOT / "benchmark" / "tasks.csv")

    summary = {"n_tasks_complete": int(len(complete)),
               "recompute_max_abs_diff": float(info.recompute_max_abs_diff.max())}
    ranks_by_metric, fried = {}, {}
    for met in METRICS:
        r = rank_matrix(task, MAIN_METHODS, met)
        r.to_csv(RES / "stats" / f"ranks_{met}.csv")
        fried[met] = friedman_nemenyi(r)
        ranks_by_metric[met] = r.mean()
    summary["friedman"] = fried

    others = [m for m in MAIN_METHODS if m != "NEXUS"]
    pair = pd.concat([pairwise_vs(task, "NEXUS", others, met) for met in ["log_loss", "roc_auc", "accuracy", "ece"]])
    pair.to_csv(RES / "stats" / "pairwise_nexus.csv", index=False)
    abl = pairwise_vs(task, ABLATION_REF, ABLATIONS, "log_loss")
    abl.to_csv(RES / "stats" / "ablation_log_loss.csv", index=False)
    abl_acc = pairwise_vs(task, ABLATION_REF, ABLATIONS, "accuracy")
    abl_acc.to_csv(RES / "stats" / "ablation_accuracy.csv", index=False)

    # Ensemble-size sweep
    sizes = task[task.method.str.startswith("NEXUS v1 (size")].copy()
    sizes["size"] = sizes.method.str.extract(r"(\d+)").astype(int)
    sizes.groupby("size")[METRICS].mean().to_csv(RES / "stats" / "ensemble_size.csv")

    # Oracle: best bagged single model chosen on the test fold (upper bound for selection)
    bag = task[task.method.str.endswith("(bagged)")]
    oracle = bag.groupby("task_id").log_loss.min()
    nexus_ll = task[task.method == ABLATION_REF].set_index("task_id").log_loss
    summary["nexus_beats_test_oracle_single_frac"] = float((nexus_ll < oracle).mean())

    # Timing
    tcols = [c for c in ["nexus_fit_seconds", "v2_fit_seconds", "ag_fit_seconds", "tuned_lgbm_seconds"] if c in info]
    t = info.groupby("task_id")[tcols].mean()
    single_cols = [c for c in info.columns if c.startswith("single_seconds_")]
    t["all_singles_seconds"] = info.groupby("task_id")[single_cols].mean().sum(axis=1)
    t.to_csv(RES / "stats" / "timing.csv")
    summary["timing_median_seconds"] = t.median().to_dict()

    # Weights
    wcols = [c for c in info.columns if c.startswith("w2_")] or [c for c in info.columns if c.startswith("w_")]
    summary["mean_weights"] = info[wcols].mean().rename(lambda c: c.split("_", 1)[1]).to_dict()
    summary["mean_nonzero_members"] = float((info[wcols] > 0).sum(axis=1).mean())
    if "ag_best_model" in info:
        summary["autogluon_best_model_counts"] = info.ag_best_model.value_counts().to_dict()
    summary["best_by_cv_counts"] = info.best_by_cv.value_counts().to_dict()
    summary["temperature_median"] = float(info.temperature.median())
    summary["temperature_iqr"] = [float(info.temperature.quantile(.25)), float(info.temperature.quantile(.75))]

    headline = pair[pair.versus.isin(["NEXUS v1", "AutoGluon", "LightGBM (tuned)", "Best model by CV", "Stacking (LR)", "CatBoost"])]
    summary["headline"] = headline.to_dict(orient="records")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    write_tables(task, fried, pair, abl, ranks_by_metric, tasks_meta, info)
    write_numbers(summary, pair, abl, abl_acc, task, info)
    print(json.dumps({k: summary[k] for k in ["n_tasks_complete", "recompute_max_abs_diff"]}, indent=1))
    print(pd.Series(ranks_by_metric["log_loss"]).sort_values().round(2).to_string())
    print(pair[pair.metric == "log_loss"][["versus", "wins", "ties", "losses", "mean_rel_improvement", "p_holm"]].to_string())


if __name__ == "__main__":
    main()
