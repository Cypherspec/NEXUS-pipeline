"""Paper figures from results/ (run benchmark.analyze first).

    python -m benchmark.figures
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from benchmark.analyze import METRIC_NAMES, friedman_nemenyi, rank_matrix
from benchmark.methods import DISPLAY, ENSEMBLE_SIZES, MAIN_METHODS
from nexus.zoo import DEFAULT_ZOO

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
FIG = ROOT / "paper" / "figures"

# Reference palette (light mode). NEXUS is emphasised; baselines recede.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BASE = "#a9a8a1"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 220,
    "legend.frameon": False, "text.color": INK,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def cd_diagram(ax, ranks: pd.Series, cd: float, title: str):
    """Demšar-style critical-difference diagram; bars join methods within CD."""
    ranks = ranks.sort_values()
    k = len(ranks)
    lo, hi = 1, k
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.82
    ax.plot([lo, hi], [y0, y0], color=INK2, lw=0.8)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y0, y0 + 0.03], color=INK2, lw=0.8)
        ax.text(r, y0 + 0.05, str(r), ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.plot([lo, lo + cd], [0.96, 0.96], color=INK, lw=1.2)
    ax.text(lo + cd + 0.1, 0.96, f"critical difference (Nemenyi, α = 0.05) = {cd:.2f}",
            ha="left", va="center", fontsize=7.2, color=INK2)
    names = list(ranks.index)
    half = (k + 1) // 2
    step = 0.62 / max(half, 1)
    for i, (n, r) in enumerate(ranks.items()):
        left = i < half
        yy = y0 - 0.08 - step * (i if left else k - 1 - i)
        xx = lo - 0.25 if left else hi + 0.25
        c = BLUE if n == "NEXUS" else INK2
        w = "bold" if n == "NEXUS" else "normal"
        ax.plot([r, r, xx], [y0, yy, yy], color=c, lw=0.8 if n != "NEXUS" else 1.4)
        ax.text(xx + (-0.05 if left else 0.05), yy, f"{n} ({r:.2f})", ha="right" if left else "left",
                va="center", fontsize=7.5, color=c, fontweight=w)
    # cliques: maximal groups whose rank span < cd
    vals = ranks.to_numpy()
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for c, (a, b) in enumerate(cliques):
        yy = y0 - 0.035 - 0.022 * c
        ax.plot([vals[a] - 0.03, vals[b] + 0.03], [yy, yy], color=INK, lw=2.2, solid_capstyle="round")
    ax.set_title(title, loc="left", pad=10)


def fig_cd(task):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4))
    for ax, met in zip(axes, ["log_loss", "roc_auc", "accuracy"]):
        r = rank_matrix(task, MAIN_METHODS, met)
        st = friedman_nemenyi(r)
        cd_diagram(ax, r.mean(), st["nemenyi_cd"],
                   f"{METRIC_NAMES[met]}: mean rank over {st['n_tasks']} tasks "
                   f"(Friedman p = {st['p_value']:.1e})")
    fig.tight_layout(h_pad=1.5)
    save(fig, "fig1_critical_difference")


def fig_scatter(task):
    wide = task.pivot(index="task_id", columns="method", values="log_loss")
    refs = ["LightGBM (tuned)", "Best model by CV", "Stacking (LR)"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, ref in zip(axes, refs):
        x, y = wide[ref], wide["NEXUS"]
        better = y < x
        lo = min(x.min(), y.min()) * 0.8
        hi = max(x.max(), y.max()) * 1.2
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.8, zorder=1)
        ax.scatter(x[~better], y[~better], s=16, color=BASE, edgecolor="white", lw=0.6, zorder=2)
        ax.scatter(x[better], y[better], s=16, color=BLUE, edgecolor="white", lw=0.6, zorder=3)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
        ax.set_xlabel(f"{ref} log loss")
        ax.text(0.04, 0.96, f"NEXUS better on {better.sum()}/{len(x)}", transform=ax.transAxes,
                va="top", fontsize=7.5, color=BLUE)
    axes[0].set_ylabel("NEXUS log loss")
    fig.tight_layout()
    save(fig, "fig2_per_task_scatter")


def _forest(ax, df, label_col, color=BLUE, xlabel=""):
    df = df.reset_index(drop=True)
    ys = np.arange(len(df))[::-1]
    ax.axvline(0, color=AXIS, lw=0.8)
    for y, (_, r) in zip(ys, df.iterrows()):
        sig = r.p_holm < 0.05
        c = color if sig else MUTED
        ax.plot([100 * r.ci95_low, 100 * r.ci95_high], [y, y], color=c, lw=1.6, solid_capstyle="round")
        ax.scatter([100 * r.mean_rel_improvement], [y], s=22, color=c, zorder=3, edgecolor="white", lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels(df[label_col])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(xlabel)


def fig_forest(pair):
    mets = ["log_loss", "roc_auc", "accuracy"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.6), sharey=True)
    order = pair[pair.metric == "log_loss"].sort_values("mean_rel_improvement").versus.tolist()
    for ax, met in zip(axes, mets):
        sub = pair[pair.metric == met].set_index("versus").loc[order].reset_index()
        _forest(ax, sub, "versus", xlabel=f"% reduction in {'log loss' if met == 'log_loss' else '1 − ' + METRIC_NAMES[met]}")
        ax.set_title(METRIC_NAMES[met], loc="left")
    fig.text(0.99, 0.005, "Blue: Holm-corrected Wilcoxon p < 0.05; grey: not significant. Bars: 95% bootstrap CI over tasks.",
             ha="right", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig3_relative_improvement")


def fig_ablation(abl):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    df = abl.sort_values("mean_rel_improvement").copy()
    # Present as the change in log loss when the component is removed/replaced.
    _forest(ax, df, "versus", xlabel="% log-loss reduction of full NEXUS over the variant")
    ax.set_title("Ablations (log loss)", loc="left")
    save(fig, "fig4_ablation")


def fig_size():
    s = pd.read_csv(RES / "stats" / "ensemble_size.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.4))
    for ax, met in zip(axes, ["log_loss", "accuracy"]):
        ax.plot(s["size"], s[met], color=BLUE, lw=2, marker="o", ms=4.5, mec="white", mew=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ENSEMBLE_SIZES); ax.set_xticklabels(ENSEMBLE_SIZES)
        ax.minorticks_off()
        ax.set_xlabel("greedy selection iterations")
        ax.set_title(f"Mean {METRIC_NAMES[met].lower()}", loc="left")
    axes[0].annotate("size 1 = best model by CV\n+ temperature", (1, s[s["size"] == 1].log_loss.iloc[0]),
                     xytext=(3, s.log_loss.max()), fontsize=7, color=INK2,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    fig.tight_layout()
    save(fig, "fig5_ensemble_size")


def fig_weights(info):
    names = [m.name for m in DEFAULT_ZOO]
    w = info[[f"w_{n}" for n in names]]
    mean_w = w.mean().to_numpy()
    used = (w > 0).mean().to_numpy()
    order = np.argsort(mean_w)
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    lbl = [DISPLAY[names[i]] for i in order]
    axes[0].barh(lbl, mean_w[order], color=BLUE, height=0.62)
    axes[0].set_title("Mean ensemble weight", loc="left")
    axes[1].barh(lbl, 100 * used[order], color=BLUE, height=0.62)
    axes[1].set_title("% of fits where selected", loc="left")
    for ax in axes:
        ax.grid(axis="y", visible=False)
    for i, v in enumerate(mean_w[order]):
        axes[0].text(v + 0.004, i, f"{v:.2f}", va="center", fontsize=7, color=INK2)
    for i, v in enumerate(100 * used[order]):
        axes[1].text(v + 1.5, i, f"{v:.0f}%", va="center", fontsize=7, color=INK2)
    axes[1].set_xlim(0, 112)
    fig.tight_layout()
    save(fig, "fig6_ensemble_weights")


def fig_pareto(task, info):
    names = [m.name for m in DEFAULT_ZOO]
    ranks = rank_matrix(task, MAIN_METHODS, "log_loss").mean()
    cost = {DISPLAY[n]: info[f"single_seconds_{n}"].median() for n in names}
    members = info[[f"member_seconds_{n}" for n in names]].sum(axis=1).median()
    for m in ["NEXUS", "Stacking (LR)", "Uniform average", "Best model by CV"]:
        cost[m] = info.nexus_fit_seconds.median() if m == "NEXUS" else members
    cost["LightGBM (tuned)"] = info.tuned_lgbm_seconds.median()
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    placed = []  # (log10 cost, rank) of labels already placed, to avoid collisions
    for m in sorted(MAIN_METHODS, key=lambda m: ranks[m]):
        c = BLUE if m == "NEXUS" else BASE
        ax.scatter(cost[m], ranks[m], s=34 if m == "NEXUS" else 22, color=c, edgecolor="white", lw=0.7, zorder=3)
        lx, ly = np.log10(cost[m]), ranks[m]
        clash = any(abs(lx - px) < 0.45 and abs(ly - py) < 0.45 for px, py in placed)
        offset, ha = ((-4, -9), "right") if clash else ((4, 3), "left")
        placed.append((lx, ly))
        ax.annotate(m, (cost[m], ranks[m]), xytext=offset, textcoords="offset points", fontsize=6.8, ha=ha,
                    color=BLUE if m == "NEXUS" else INK2, fontweight="bold" if m == "NEXUS" else "normal")
    ax.set_xscale("log")
    ax.set_xlabel("median training time per fold (s, single core)")
    ax.set_ylabel("mean rank, log loss (lower is better)")
    ax.invert_yaxis()
    ax.set_title("Accuracy–cost trade-off", loc="left")
    save(fig, "fig7_cost_vs_rank")


def fig_profiles(task):
    wide = task.pivot(index="task_id", columns="method", values="log_loss")
    methods = [("NEXUS", BLUE), ("LightGBM (tuned)", ORANGE), ("Best model by CV", AQUA), ("Stacking (LR)", YELLOW)]
    ratio = wide[[m for m, _ in methods]].div(wide[[m for m, _ in methods]].min(axis=1), axis=0)
    taus = np.linspace(1, 1.5, 400)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for m, c in methods:
        prof = [(ratio[m] <= t).mean() for t in taus]
        ax.plot(taus, prof, color=c, lw=2)
        ax.text(taus[-1] + 0.005, prof[-1], m, color=INK2, fontsize=7, va="center")
    ax.set_xlim(1, 1.62)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("τ  (log loss within factor τ of the best of these four)")
    ax.set_ylabel("fraction of tasks")
    ax.set_title("Performance profile (log loss)", loc="left")
    save(fig, "fig8_performance_profile")


def fig_reliability():
    rel = pd.read_csv(RES / "stats" / "reliability.csv")
    panels = ["NEXUS w/o calibration", "NEXUS", "Uniform average", "LightGBM (tuned)"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.25), sharey=True)
    for ax, m in zip(axes, panels):
        d = rel[rel.method == m]
        d = d[d["count"] >= max(30, 0.002 * d["count"].sum())]  # hide near-empty bins
        conf, acc, n = d.conf_sum / d["count"], d.correct_sum / d["count"], d["count"]
        allm = rel[rel.method == m]
        ece = float(np.sum(np.abs(allm.correct_sum - allm.conf_sum)) / allm["count"].sum())
        ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8)
        c = BLUE if m == "NEXUS" else INK2
        ax.plot(conf, acc, color=c, lw=1.6, marker="o", ms=3.2, mec="white", mew=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
        ax.set_title(m, loc="left", fontsize=8, color=c)
        ax.text(0.97, 0.04, f"pooled ECE {ece:.3f}", transform=ax.transAxes, ha="right", fontsize=7, color=INK2)
        ax.set_xlabel("confidence")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    save(fig, "fig9_reliability")


def main():
    task = pd.read_csv(RES / "task_metrics.csv")
    info = pd.read_csv(RES / "fold_info.csv")
    info = info[info.task_id.isin(task.task_id.unique())]
    pair = pd.read_csv(RES / "stats" / "pairwise_nexus.csv")
    abl = pd.read_csv(RES / "stats" / "ablation_log_loss.csv")
    fig_cd(task)
    fig_scatter(task)
    fig_forest(pair)
    fig_ablation(abl)
    fig_size()
    fig_weights(info)
    fig_pareto(task, info)
    fig_profiles(task)
    fig_reliability()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
