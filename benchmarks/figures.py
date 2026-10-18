"""Figures for the NEXUS results. Every panel is drawn from a file in results/.

    python -m benchmarks.figures

Writes paper/figures/*.png and *.pdf. Colours follow a validated categorical
palette; sequential scales use a single hue, and no chart uses two y-scales.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIG = ROOT / "paper" / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# Single-hue sequential ramp, light to dark.
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 220,
    "legend.frameon": False, "text.color": INK, "savefig.facecolor": SURFACE,
})


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def load(name: str):
    path = RESULTS / name
    return json.loads(path.read_text()) if path.exists() else None


# ------------------------------------------------------------------ figure 1

def fig_molnet() -> None:
    """Method comparison per dataset, with seed-to-seed spread."""
    summary = load("molnet_summary.json")
    if not summary:
        return
    datasets = [d for d in summary if summary[d]["methods"]]
    if not datasets:
        return
    order = ["D-MPNN", "D-MPNN + descriptors", "ECFP + RF", "ECFP+descriptors + RF",
             "ECFP + linear", "descriptors + RF"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(2.5 * len(datasets), 3.1))
    axes = np.atleast_1d(axes)

    for ax, name in zip(axes, datasets):
        block = summary[name]
        higher = block["metric"] == "roc_auc"
        present = [m for m in order if m in block["methods"]]
        means = [block["methods"][m]["mean"] for m in present]
        stds = [block["methods"][m]["std"] for m in present]
        y = np.arange(len(present))[::-1]
        colours = [BLUE if m.startswith("D-MPNN") else "#a9a8a1" for m in present]
        ax.barh(y, means, xerr=stds, height=0.6, color=colours,
                error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
        ref = block.get("reference", {}).get("chemprop_dmpnn")
        if ref:
            ax.axvline(ref, color=ORANGE, lw=1.2, ls=(0, (4, 2)), zorder=3)
            ax.text(ref, len(present) - 0.3, " published", color=ORANGE, fontsize=6.5,
                    va="top")
        ax.set_yticks(y)
        ax.set_yticklabels(present if name == datasets[0] else [""] * len(present),
                           fontsize=7)
        ax.set_title(f"{name}\n{block['metric']} "
                     f"({'higher' if higher else 'lower'} is better)", loc="left",
                     fontsize=8)
        ax.grid(axis="y", visible=False)
        lo = min(means) - max(stds) - 0.05
        ax.set_xlim(max(0, lo), max(means) + max(stds) + 0.05)
    fig.suptitle("MoleculeNet, scaffold-balanced splits, 3 seeds", x=0.01, ha="left",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "fig1_molnet_benchmark")


# ------------------------------------------------------------------ figure 2

def fig_active_learning() -> None:
    """Hits found over screening rounds, against the random control."""
    data = load("active_learning_validation.json")
    if not data:
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9),
                                  gridspec_kw={"width_ratios": [1.5, 1]})
    strategies = list(data["strategies"])
    batch, initial = data["batch_size"], data["initial_size"]
    n_actives = data["n_actives"]

    ends: list[tuple[float, str, str]] = []
    for i, name in enumerate(strategies):
        curve = data["strategies"][name]["cumulative_hits_mean"]
        x = [initial + k * batch for k in range(len(curve))]
        colour = MUTED if name == "random" else SERIES[i % len(SERIES)]
        ax.plot(x, curve, color=colour, lw=2 if name != "random" else 1.4,
                ls="--" if name == "random" else "-",
                marker="o", ms=3, mec="white", mew=0.6)
        ends.append((curve[-1], name, colour))
    # Spread the end labels vertically so near-identical curves stay readable.
    ends.sort(reverse=True)
    span = max(e[0] for e in ends) - min(e[0] for e in ends)
    step = max(span * 0.11, n_actives * 0.035)
    placed: list[float] = []
    for value, name, colour in ends:
        y_label = value
        while placed and abs(y_label - placed[-1]) < step:
            y_label -= step - abs(y_label - placed[-1])
        placed.append(y_label)
        ax.plot([x[-1], x[-1] + 55], [value, y_label], color=colour, lw=0.7)
        ax.text(x[-1] + 65, y_label, name, color=colour, fontsize=7, va="center")
    ax.set_xlabel("compounds screened")
    ax.set_ylabel("actives found")
    ax.set_title(f"Retrospective screen: {data['library_size']} compounds, "
                 f"{n_actives} actives", loc="left")
    ax.set_xlim(0, initial + batch * (len(curve) - 1) + 230)

    enrich = [data["strategies"][s]["enrichment_mean"] for s in strategies]
    err = [data["strategies"][s]["enrichment_std"] for s in strategies]
    y = np.arange(len(strategies))[::-1]
    colours = [MUTED if s == "random" else BLUE for s in strategies]
    ax2.barh(y, enrich, xerr=err, height=0.6, color=colours,
             error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 2})
    ax2.axvline(1.0, color=AXIS, lw=0.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels(strategies, fontsize=7.5)
    ax2.set_xlabel("enrichment over random")
    ax2.set_title("Enrichment", loc="left")
    ax2.grid(axis="y", visible=False)
    for yy, v, e in zip(y, enrich, err):
        ax2.text(v + e + 0.08, yy, f"{v:.2f}x", va="center", fontsize=7, color=INK2)
    ax2.set_xlim(0, (max(enrich) + max(err)) * 1.3)
    fig.tight_layout()
    save(fig, "fig2_active_learning")


# ------------------------------------------------------------------ figure 3

def fig_conformal() -> None:
    """Empirical coverage against the nominal level, measured live."""
    from nexus.uq.conformal import ConformalClassifier, ConformalRegressor

    alphas = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
    n, trials = 1500, 40
    reg, clf = [], []
    for alpha in alphas:
        r_cov, c_cov = [], []
        for t in range(trials):
            rng = np.random.default_rng(t)
            x = rng.normal(size=n)
            y = 2 * x + rng.normal(0, 1, n)
            pred = 2 * x
            half = n // 2
            model = ConformalRegressor(alpha=float(alpha)).fit(pred[:half], y[:half])
            r_cov.append(model.coverage(pred[half:], y[half:]))

            probs = rng.dirichlet(np.ones(3), size=n)
            labels = np.array([rng.choice(3, p=p) for p in probs])
            cc = ConformalClassifier(alpha=float(alpha)).fit(probs[:half], labels[:half])
            c_cov.append(cc.coverage(probs[half:], labels[half:]))
        reg.append((np.mean(r_cov), np.std(r_cov)))
        clf.append((np.mean(c_cov), np.std(c_cov)))

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    nominal = 1 - alphas
    ax.plot([0.55, 1.0], [0.55, 1.0], color=AXIS, lw=1.0, zorder=1)
    ax.text(0.98, 0.955, "exact", color=MUTED, fontsize=7, ha="right")
    for (series, colour, label) in ((reg, BLUE, "regression intervals"),
                                    (clf, ORANGE, "classification sets")):
        mean = np.array([s[0] for s in series])
        std = np.array([s[1] for s in series])
        ax.errorbar(nominal, mean, yerr=std, color=colour, lw=1.8, marker="o",
                    ms=4.5, mec="white", mew=0.7, capsize=2, elinewidth=0.8,
                    label=label, zorder=3)
    ax.set_xlabel("nominal coverage (1 - alpha)")
    ax.set_ylabel("empirical coverage")
    ax.set_title(f"Conformal prediction: guarantee holds\n"
                 f"{trials} trials per level, {n // 2} test points each", loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_xlim(0.55, 1.01)
    ax.set_ylim(0.55, 1.01)
    ax.set_aspect("equal")
    save(fig, "fig3_conformal_coverage")


# ------------------------------------------------------------------ figure 4

def fig_free_energy() -> None:
    """Estimator accuracy against an analytically known free energy."""
    from nexus.physics.jarzynski import (
        SwitchingProtocol, bar_free_energy, crooks_free_energy,
        equilibrium_sample, jarzynski_cumulant_estimate, jarzynski_free_energy,
    )

    beta, k0, k1 = 1.0, 1.0, 4.0
    exact = 0.5 * np.log(k1 / k0) / beta

    def energy(x, lam):
        return 0.5 * ((1 - lam) * k0 + lam * k1) * np.sum(x ** 2, axis=1)

    protocol = SwitchingProtocol(energy=energy, beta=beta, n_steps=200, dt=0.005)
    x0 = equilibrium_sample(energy, 0.0, 3000, 1, beta, seed=0)
    forward, _ = protocol.run(x0, forward=True, seed=1)
    x1 = equilibrium_sample(energy, 1.0, 3000, 1, beta, seed=2)
    reverse, _ = protocol.run(x1, forward=False, seed=3)

    estimates = {
        "BAR": bar_free_energy(forward, reverse, beta),
        "Crooks": crooks_free_energy(forward, reverse, beta),
        "Jarzynski": jarzynski_free_energy(forward, beta),
        "cumulant": jarzynski_cumulant_estimate(forward, beta),
        "mean work": float(forward.mean()),
    }

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    ax.hist(forward, bins=45, color=BLUE, alpha=0.75, density=True, label="forward W")
    ax.hist(-reverse, bins=45, color=ORANGE, alpha=0.6, density=True,
            label="-reverse W")
    ax.axvline(exact, color=INK, lw=1.4, zorder=5)
    ax.text(exact, ax.get_ylim()[1] * 0.96, f" exact dF = {exact:.3f}", fontsize=7.5,
            color=INK, va="top")
    ax.set_xlim(min(forward.min(), (-reverse).min()) - 0.1,
                float(np.percentile(np.concatenate([forward, -reverse]), 99.5)))
    ax.set_xlabel("work (kT)")
    ax.set_ylabel("density")
    ax.set_title("Work distributions cross at dF", loc="left")
    ax.legend(fontsize=7.5, loc="upper right")

    names = list(estimates)
    errors = [abs(estimates[n] - exact) for n in names]
    y = np.arange(len(names))[::-1]
    colours = [BLUE if e < 0.02 else ("#a9a8a1" if e < 0.3 else ORANGE) for e in errors]
    ax2.barh(y, errors, height=0.6, color=colours)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=7.5)
    ax2.set_xscale("log")
    ax2.set_xlabel("absolute error vs exact (kT)")
    ax2.set_title("Estimator accuracy", loc="left")
    ax2.grid(axis="y", visible=False)
    for yy, (n, e) in zip(y, zip(names, errors)):
        ax2.text(e * 1.15, yy, f"{estimates[n]:.4f}", va="center", fontsize=7,
                 color=INK2)
    ax2.set_xlim(min(errors) * 0.4, max(errors) * 6)
    fig.tight_layout()
    save(fig, "fig4_free_energy")


# ------------------------------------------------------------------ figure 5

def fig_descriptors() -> None:
    """Descriptor agreement with reference values shipped in public datasets."""
    import csv

    from nexus.chem.descriptors import num_rotatable_bonds, tpsa
    from nexus.chem.smiles import read_smiles

    path = ROOT / "data" / "molnet" / "bace.csv"
    if not path.exists():
        return
    rows = list(csv.DictReader(path.read_text().splitlines()))
    mine_tpsa, ref_tpsa, mine_mw, ref_mw = [], [], [], []
    for row in rows:
        mol = read_smiles((row.get("mol") or "").strip())
        if mol is None:
            continue
        mine_tpsa.append(tpsa(mol, include_s_p=True))
        ref_tpsa.append(float(row["PSA"]))
        mine_mw.append(mol.molecular_weight())
        ref_mw.append(float(row["MW"]))

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for ax, mine, ref, label, unit in (
        (axes[0], np.array(mine_tpsa), np.array(ref_tpsa), "TPSA", "A^2"),
        (axes[1], np.array(mine_mw), np.array(ref_mw), "Molecular weight", "Da"),
    ):
        lo = min(mine.min(), ref.min())
        hi = max(mine.max(), ref.max())
        ax.plot([lo, hi], [lo, hi], color=AXIS, lw=0.9, zorder=1)
        ax.scatter(ref, mine, s=7, color=BLUE, alpha=0.5, edgecolor="none", zorder=2)
        r = np.corrcoef(ref, mine)[0, 1]
        exact = np.mean(np.abs(mine - ref) < 0.05)
        ax.set_xlabel(f"reference {label} ({unit})")
        ax.set_ylabel(f"NEXUS {label} ({unit})")
        ax.set_title(f"{label}\nr = {r:.4f}, {100 * exact:.1f}% exact", loc="left")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    fig.suptitle(f"Descriptor validation against BACE reference columns "
                 f"({len(mine_tpsa)} molecules)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig5_descriptor_validation")


# ------------------------------------------------------------------ figure 6

def fig_topology() -> None:
    """Persistence diagrams for molecules of increasing ring complexity."""
    from nexus.chem.conformer import embed_molecule
    from nexus.chem.smiles import parse_smiles
    from nexus.topo.homology import rips_persistence

    examples = [("CCCCCCCC", "octane"), ("c1ccccc1", "benzene"),
                ("c1ccc2ccccc2c1", "naphthalene"),
                ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "caffeine")]
    fig, axes = plt.subplots(1, len(examples), figsize=(2.0 * len(examples), 2.5),
                             sharex=True, sharey=True)
    diagrams = []
    for smi, label in examples:
        mol = parse_smiles(smi)
        coords, _ = embed_molecule(mol, seed=0, steps=400)
        diagram = rips_persistence(coords, max_dim=1)
        diagrams.append((label, mol, diagram))
    limit = 0.0
    for _, _, diagram in diagrams:
        for dim in (0, 1):
            pts = diagram[dim]
            finite = pts[np.isfinite(pts[:, 1])] if len(pts) else pts
            if len(finite):
                limit = max(limit, float(finite[:, 1].max()))
    limit = limit * 1.25 or 1.0

    for ax, (label, mol, diagram) in zip(axes, diagrams):
        ax.plot([0, limit], [0, limit], color=AXIS, lw=0.8, zorder=1)
        h0 = diagram[0]
        finite0 = h0[np.isfinite(h0[:, 1])]
        if len(finite0):
            # Coincident points are common here, so draw them hollow and
            # semi-transparent: overlaps show as darker marks instead of one dot.
            ax.scatter(finite0[:, 0], finite0[:, 1], s=26, facecolor="none",
                       edgecolor=BLUE, lw=1.1, alpha=0.75,
                       label="H0 (components)", zorder=3)
        h1 = diagram[1]
        if len(h1):
            ax.scatter(h1[:, 0], h1[:, 1], s=44, facecolor="none", edgecolor=ORANGE,
                       marker="^", lw=1.3, alpha=0.85, label="H1 (loops)", zorder=4)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal")
        ax.set_title(f"{label}\n{mol.n_rings} ring(s), {len(h1)} loop(s)", loc="left",
                     fontsize=8)
        ax.set_xlabel("birth (A)")
    axes[0].set_ylabel("death (A)")
    axes[-1].legend(fontsize=6.5, loc="lower right")
    fig.suptitle("Persistent homology of 3D conformers", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig6_persistence_diagrams")


# ------------------------------------------------------------------ figure 7

def fig_ricci() -> None:
    """Ricci flow drives curvature towards uniformity."""
    from nexus.chem.smiles import parse_smiles
    from nexus.topo.ricci import ollivier_ricci_curvature, ricci_flow

    examples = [("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "caffeine"),
                ("CC(C)(C)c1ccccc1", "tert-butylbenzene"),
                ("c1ccc(cc1)CCc1ccccc1", "bibenzyl"),
                ("c1ccccc1", "benzene")]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    for i, (smi, label) in enumerate(examples):
        mol = parse_smiles(smi)
        _, trace = ricci_flow(mol, iterations=10)
        ax.plot(range(len(trace)), trace, color=SERIES[i], lw=2, marker="o", ms=3.5,
                mec="white", mew=0.6)
        ax.text(len(trace) - 0.9, trace[-1], f" {label}", color=SERIES[i], fontsize=7,
                va="center")
    ax.set_xlabel("Ricci flow iteration")
    ax.set_ylabel("curvature spread (std)")
    ax.set_title("Flow drives curvature to uniform", loc="left")
    ax.set_xlim(0, len(trace) + 3.2)

    labels, values = [], []
    for smi, label in examples:
        curv = list(ollivier_ricci_curvature(parse_smiles(smi)).values())
        labels.append(label)
        values.append(curv)
    positions = np.arange(len(labels))
    for i, vals in enumerate(values):
        jitter = np.random.default_rng(i).normal(0, 0.05, len(vals))
        ax2.scatter(positions[i] + jitter, vals, s=18, color=SERIES[i], alpha=0.75,
                    edgecolor="white", lw=0.5)
    ax2.axhline(0, color=AXIS, lw=0.8)
    ax2.set_xticks(positions)
    ax2.set_xticklabels([l[:12] for l in labels], fontsize=7, rotation=20, ha="right")
    ax2.set_ylabel("Ollivier-Ricci curvature per bond")
    ax2.set_title("Benzene (a cycle) has curvature 0 exactly", loc="left")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig7_ricci_curvature")


# ------------------------------------------------------------------ figure 8

def fig_gene_ontology() -> None:
    """GO validation: controls recovered, false positives, similarity separation."""
    data = load("go_validation.json")
    if not data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))

    controls = data["positive_controls"]
    labels = [c["label"] for c in controls]
    ranks = [(c["rank_of_target"] + 1) if c["rank_of_target"] is not None else np.nan
             for c in controls]
    y = np.arange(len(labels))[::-1]
    colours = [BLUE if np.isfinite(r) and r == 1 else ORANGE for r in ranks]
    plotted = [r if np.isfinite(r) else 50 for r in ranks]
    axes[0].barh(y, plotted, height=0.6, color=colours)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].set_xlabel("rank of the seeded term")
    axes[0].set_title("Positive controls", loc="left")
    axes[0].grid(axis="y", visible=False)
    for yy, r in zip(y, ranks):
        axes[0].text(1.2, yy, "rank 1" if r == 1 else "not in top 50", va="center",
                     fontsize=6.5, color=INK2)
    axes[0].set_xlim(0, 60)

    fp = data["false_positive_control"]
    axes[1].bar([0, 1], [fp["mean_significant_terms"], fp["max_significant_terms"]],
                width=0.55, color=[BLUE, "#a9a8a1"])
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["mean", "worst trial"], fontsize=7.5)
    axes[1].set_ylabel("significant terms (FDR 0.05)")
    axes[1].set_title(f"Random gene sets\n{fp['n_trials']} trials of "
                      f"{fp['genes_per_trial']} genes", loc="left")
    axes[1].grid(axis="x", visible=False)
    for x, v in zip([0, 1], [fp["mean_significant_terms"], fp["max_significant_terms"]]):
        axes[1].text(x, v + 0.05, f"{v:.2f}", ha="center", fontsize=7.5, color=INK2)

    sim = data["semantic_similarity"]
    axes[2].bar([0, 1], [sim["mean_same_pathway"], sim["mean_random"]], width=0.55,
                color=[BLUE, "#a9a8a1"])
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["same pathway", "random pair"], fontsize=7.5)
    axes[2].set_ylabel("Lin semantic similarity")
    axes[2].set_title(f"Gene similarity\nAUC {sim['auc_same_vs_random']:.3f}", loc="left")
    axes[2].grid(axis="x", visible=False)
    for x, v in zip([0, 1], [sim["mean_same_pathway"], sim["mean_random"]]):
        axes[2].text(x, v + 0.006, f"{v:.3f}", ha="center", fontsize=7.5, color=INK2)
    fig.suptitle("Gene Ontology layer validation", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig8_gene_ontology")


# ------------------------------------------------------------------ figure 9

def fig_scaffold_split() -> None:
    """Scaffold splitting removes the leakage a random split leaves behind."""
    from nexus.chem.scaffold import random_split, scaffold_overlap, scaffold_split
    from nexus.data import load_dataset

    names = ["bace", "bbbp", "esol"]
    scaffold_leak, random_leak = [], []
    for name in names:
        mols, _, _, _ = load_dataset(name)
        tr, _, te = scaffold_split(mols, seed=0)
        rtr, _, rte = random_split(len(mols), seed=0)
        scaffold_leak.append(scaffold_overlap(tr, te, mols))
        random_leak.append(scaffold_overlap(rtr, rte, mols))

    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(len(names))
    width = 0.36
    ax.bar(x - width / 2, [100 * v for v in random_leak], width, color="#a9a8a1",
           label="random split")
    ax.bar(x + width / 2, [100 * v for v in scaffold_leak], width, color=BLUE,
           label="scaffold split")
    for xi, v in zip(x - width / 2, random_leak):
        ax.text(xi, 100 * v + 1.5, f"{100 * v:.0f}%", ha="center", fontsize=7.5,
                color=INK2)
    for xi, v in zip(x + width / 2, scaffold_leak):
        ax.text(xi, 1.5, f"{100 * v:.0f}%", ha="center", fontsize=7.5, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("test molecules whose scaffold\nalso appears in training (%)")
    ax.set_title("Scaffold leakage", loc="left")
    ax.legend(fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 100)
    save(fig, "fig9_scaffold_leakage")


def fig_adaptive_conformal() -> None:
    """Adaptive interval widths: what they buy, and what they cost.

    Marginal coverage is guaranteed for every score, so the question is only
    whether width adapts usefully. Left: mean width at matched coverage.
    Right: coverage inside difficulty quintiles, where a flat line at the
    nominal level is the ideal and the hardest quintile is the one that
    matters for triage.
    """
    data = load("conformal_adaptive.json")
    nominal = data["nominal_coverage"]
    datasets = list(data["datasets"].keys())
    variants = ["standard", "gp_sigma", "residual_fitted"]
    pretty = {"standard": "constant width", "gp_sigma": "scaled by GP sd",
              "residual_fitted": "fitted residual model"}
    colours = {"standard": "#a9a8a1", "gp_sigma": BLUE, "residual_fitted": ORANGE}

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))

    ax = axes[0]
    x = np.arange(len(datasets))
    width = 0.26
    for k, variant in enumerate(variants):
        widths, labels = [], []
        for name in datasets:
            entry = data["datasets"][name].get(variant, {})
            widths.append(entry.get("mean_width", np.nan))
            labels.append(entry.get("coverage", np.nan))
        offset = (k - 1) * width
        ax.bar(x + offset, widths, width, color=colours[variant],
               label=pretty[variant])
        for xi, w, cov in zip(x + offset, widths, labels):
            if np.isfinite(w):
                ax.text(xi, w, f"{cov:.2f}", ha="center", va="bottom",
                        fontsize=6.5, color=INK2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("mean interval width")
    ax.set_title("Width at matched coverage", loc="left")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(ax.get_ylim()[1] * 1.28, 1))
    # Numbers above each bar are the measured marginal coverage.
    ax.text(0.0, -0.26, "labels: measured marginal coverage",
            transform=ax.transAxes, fontsize=6.5, color=MUTED)

    ax = axes[1]
    for variant in variants:
        series = [data["datasets"][name].get(variant, {}).get(
            "conditional_by_quintile") for name in datasets]
        series = [s for s in series if s]
        if not series:
            continue
        mean = np.mean(np.array(series, dtype=float), axis=0)
        ax.plot(np.arange(1, len(mean) + 1), mean, marker="o", markersize=4,
                linewidth=2, color=colours[variant], label=pretty[variant])
    ax.axhline(nominal, color=INK2, linewidth=1, linestyle="--")
    ax.text(0.99, nominal, f"nominal {nominal:.0%}", fontsize=7, color=INK2,
            va="bottom", ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("difficulty quintile (hardest last)")
    ax.set_ylabel("coverage")
    ax.set_title("Coverage by difficulty", loc="left")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    save(fig, "fig10_adaptive_conformal")


def fig_shape_signatures() -> None:
    """Topology and curvature separate molecules the descriptors do not.

    Three panels, each a check against something known rather than a display:
    a benzene ring must give exactly one long-lived loop and curvature exactly
    zero, an acyclic chain neither; and across a real dataset the loop count
    should track the ring count without being a restatement of it.
    """
    import numpy as _np

    from nexus.chem.conformer import embed_molecule
    from nexus.chem.descriptors import num_rings
    from nexus.chem.smiles import parse_smiles
    from nexus.data import load_dataset
    from nexus.topo.homology import rips_persistence
    from nexus.topo.ricci import ollivier_ricci_curvature

    probes = [("benzene", "c1ccccc1"), ("octane", "CCCCCCCC"),
              ("caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
              ("morphine", "CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OC3C(C=C4)O")]

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.7))

    # Panel 1: persistence diagrams of the probes.
    ax = axes[0]
    for i, (name, smiles) in enumerate(probes[:3]):
        mol = parse_smiles(smiles)
        xyz, _ = embed_molecule(mol, seed=0)
        diagram = rips_persistence(_np.asarray(xyz), max_dim=1)
        pairs = _np.asarray(diagram.points.get(1, []), dtype=float).reshape(-1, 2)
        finite = pairs[_np.isfinite(pairs[:, 1])]
        if len(finite):
            ax.scatter(finite[:, 0], finite[:, 1], s=34, color=SERIES[i],
                       edgecolor=SURFACE, linewidth=0.6, label=f"{name} ({len(finite)})",
                       zorder=3)
        else:
            ax.scatter([], [], s=34, color=SERIES[i], label=f"{name} (0)")
    lo, hi = ax.get_xlim()[0], max(ax.get_ylim()[1], 3.0)
    ax.plot([0, hi], [0, hi], color=MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("birth (Å)")
    ax.set_ylabel("death (Å)")
    ax.set_title("H$_1$ features", loc="left")
    ax.legend(fontsize=6.5, loc="lower right")

    # Panel 2: curvature distribution per probe.
    ax = axes[1]
    for i, (name, smiles) in enumerate(probes):
        mol = parse_smiles(smiles)
        values = list(ollivier_ricci_curvature(mol).values())
        ax.scatter([i] * len(values), values, s=22, color=SERIES[i % len(SERIES)],
                   alpha=0.75, edgecolor=SURFACE, linewidth=0.5)
    ax.axhline(0.0, color=MUTED, linewidth=1, linestyle="--")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels([name for name, _ in probes], rotation=20, ha="right")
    ax.set_ylabel("Ollivier–Ricci curvature")
    ax.set_title("Zero on a cycle", loc="left")
    ax.grid(axis="x", visible=False)

    # Panel 3: loops against ring count over a real dataset.
    ax = axes[2]
    dataset, _, _, _ = load_dataset("esol")
    loops, rings = [], []
    for mol in dataset[:120]:
        if mol.n_atoms < 3 or mol.n_atoms > 40:
            continue
        try:
            xyz, _ = embed_molecule(mol, seed=0)
            diagram = rips_persistence(_np.asarray(xyz), max_dim=1)
            pairs = _np.asarray(diagram.points.get(1, []), dtype=float).reshape(-1, 2)
            finite = pairs[_np.isfinite(pairs[:, 1])]
            # Count only features that survive a meaningful range of scales;
            # short-lived pairs are embedding noise, not rings.
            loops.append(int((finite[:, 1] - finite[:, 0] > 0.35).sum()))
            rings.append(num_rings(mol))
        except Exception:                                    # noqa: BLE001
            continue
    jitter = np.random.default_rng(0).normal(0, 0.06, size=len(rings))
    ax.scatter(np.array(rings) + jitter, loops, s=16, color=BLUE, alpha=0.55,
               edgecolor="none")
    if rings:
        r = float(np.corrcoef(rings, loops)[0, 1])
        ax.text(0.04, 0.92, f"r = {r:.2f}  (n = {len(rings)})", transform=ax.transAxes,
                fontsize=7, color=INK2)
    ax.set_xlabel("rings (perceived)")
    ax.set_ylabel("long-lived H$_1$ features")
    ax.set_title("Loops track rings", loc="left")

    fig.tight_layout()
    save(fig, "fig11_shape_signatures")


def fig_conformal_width_distribution() -> None:
    """What adaptive widths actually do to the distribution of interval widths.

    The summary table reports a mean; this shows the spread behind it. A
    constant-width score is a spike, which is the point: it cannot tell a
    confident prediction from a guess.
    """
    data = load("conformal_adaptive.json")
    if not data:
        raise FileNotFoundError("results/conformal_adaptive.json")

    names = list(data["datasets"].keys())
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.5), sharey=False)
    if len(names) == 1:
        axes = [axes]
    pretty = {"standard": ("constant width", "#a9a8a1"),
              "gp_sigma": ("scaled by GP sd", BLUE),
              "residual_fitted": ("fitted residual model", ORANGE)}

    for ax, name in zip(axes, names):
        variants = data["datasets"][name]
        for key, (label, colour) in pretty.items():
            entry = variants.get(key)
            if not entry:
                continue
            mean = entry["mean_width"]
            spread = entry["width_spread"]
            # Each score is drawn as its mean with a +/- 1 sd bar; a constant
            # width has no spread and collapses to a point, which is the
            # distinction worth seeing.
            ax.errorbar([label], [mean], yerr=[spread], fmt="o", markersize=6,
                        color=colour, ecolor=colour, elinewidth=2, capsize=4)
        ax.set_title(name, loc="left")
        ax.set_ylabel("interval width" if name == names[0] else "")
        ax.tick_params(axis="x", labelrotation=24)
        ax.grid(axis="x", visible=False)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_fontsize(7)

    fig.suptitle("Interval width, mean ± 1 sd", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig12_interval_width_spread")


def main() -> None:
    print("rendering figures from results/ ...")
    for fn in (fig_molnet, fig_active_learning, fig_conformal, fig_free_energy,
               fig_descriptors, fig_topology, fig_ricci, fig_gene_ontology,
               fig_scaffold_split, fig_adaptive_conformal,
               fig_shape_signatures, fig_conformal_width_distribution):
        try:
            fn()
        except Exception as exc:                      # noqa: BLE001
            print(f"  {fn.__name__} failed: {type(exc).__name__}: {exc}")
    print(f"figures in {FIG}")


if __name__ == "__main__":
    main()
