"""Render the development-set design study (results/dev_summary.csv) as tables."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def bagging_gain():
    """Single-model OOF vs bagged-test log loss per algorithm (needs the dev cache)."""
    import numpy as np

    from tabular_ensemble_benchmark.dev import DEV_DATASETS, N_OUTER, V1, member_path
    from tabular_ensemble.ensemble import log_loss_from_proba
    from tabular_ensemble_benchmark.methods import DISPLAY

    rows = []
    for n in V1:
        for d in DEV_DATASETS:
            for f in range(N_OUTER):
                z = np.load(member_path(d, f, n))
                oof = log_loss_from_proba(z["y_train"].astype(int), z["oof"].astype(float))
                test = log_loss_from_proba(z["y_test"].astype(int), z["test"].astype(float))
                rows.append({"model": DISPLAY[n], "did": d, "oof": oof, "test": test})
    df = pd.DataFrame(rows).groupby(["model", "did"]).mean().reset_index()
    df["gain_%"] = 100 * (df.oof - df.test) / df.oof
    g = df.groupby("model").agg(oof=("oof", "mean"), bagged_test=("test", "mean"),
                                gain_mean=("gain_%", "mean"), gain_median=("gain_%", "median"),
                                gain_positive=("gain_%", lambda x: int((x > 0).sum())))
    return g.sort_values("gain_mean")


def main():
    cache = ROOT / "results" / "dev" / "cache"
    if cache.exists():
        bagging_gain().to_csv(ROOT / "results" / "dev_bagging_gain.csv")
    g = pd.read_csv(ROOT / "results" / "dev_bagging_gain.csv", index_col=0)
    with open(ROOT / "paper_tabular" / "tables" / "bagging_gain.tex", "w") as fh:
        fh.write("\\begin{tabular}{lrrrr}\n\\toprule\nAlgorithm & OOF LL & Bagged test LL & Gain (mean) & Datasets with gain\\\\\n\\midrule\n")
        for name, r in g.iterrows():
            fh.write(f"{name} & {r.oof:.3f} & {r.bagged_test:.3f} & {r.gain_mean:.1f}\\% & {int(r.gain_positive)}/16\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(ROOT / "paper_tabular" / "tables" / "bagging_gain.md", "w") as fh:
        fh.write("| Algorithm | OOF log loss | Bagged test log loss | Mean gain | Datasets with gain |\n|---|---|---|---|---|\n")
        for name, r in g.iterrows():
            fh.write(f"| {name} | {r.oof:.3f} | {r.bagged_test:.3f} | {r.gain_mean:.1f}% | {int(r.gain_positive)}/16 |\n")
    df = pd.read_csv(ROOT / "results" / "dev_summary.csv", index_col=0)
    df = df.sort_values("dR2_pp", ascending=False)
    out = ROOT / "paper_tabular" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "dev.md", "w") as fh:
        fh.write("| Candidate (vs. v1) | Mean rank | ΔR² (pp) | Mean rel. log loss | Median rel. log loss | Wins / 16 | Rel. error change |\n|---|---|---|---|---|---|---|\n")
        for name, r in df.iterrows():
            fh.write(f"| {name} | {r.mean_rank_ll:.2f} | {r.dR2_pp:+.2f} | {r['rel_ll_vs_ref_%']:+.1f}% | "
                     f"{r['median_rel_ll_%']:+.2f}% | {int(r.wins_ll_vs_ref)} | {r['rel_err_vs_ref_%']:+.1f}% |\n")
    with open(out / "dev.tex", "w") as fh:
        fh.write("\\begin{tabular}{lrrrrr}\n\\toprule\nCandidate & Rank & $\\Delta R^2$ (pp) & Rel.\\ LL & Wins/16 & Rel.\\ err.\\\\\n\\midrule\n")
        for name, r in df.iterrows():
            n = name.replace("{", "\\{").replace("}", "\\}").replace("_", "\\_")
            bold = "(v2)" in name
            cells = [n, f"{r.mean_rank_ll:.2f}", f"{r.dR2_pp:+.2f}", f"{r['rel_ll_vs_ref_%']:+.1f}\\%",
                     str(int(r.wins_ll_vs_ref)), f"{r['rel_err_vs_ref_%']:+.1f}\\%"]
            if bold:
                cells = [f"\\textbf{{{c}}}" for c in cells]
            fh.write(" & ".join(cells) + "\\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")


if __name__ == "__main__":
    main()
