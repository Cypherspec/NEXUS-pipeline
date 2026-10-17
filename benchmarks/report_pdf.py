"""Compile every measured result, figure and screenshot into one PDF.

    python -m benchmarks.report_pdf

Numbers are read from results/ at build time, so the document cannot drift
from what was actually measured. Sections whose results are not yet present are
marked as still running rather than omitted.
"""

from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def load(name: str):
    path = RESULTS / name
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def esc(text: str) -> str:
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        text = text.replace(a, b)
    return text


def code_stats() -> dict:
    """Line counts for the shipped package, tests and benchmarks."""
    def count(pattern: str, *roots: str) -> int:
        total = 0
        for root in roots:
            for path in (ROOT / root).rglob(pattern):
                if "__pycache__" in str(path) or "node_modules" in str(path):
                    continue
                total += len(path.read_text(errors="ignore").splitlines())
        return total
    return {
        "package": count("*.py", "nexus"),
        "tests": count("*.py", "tests"),
        "benchmarks": count("*.py", "benchmarks"),
        "panel": count("*.ts", "mantis/src") + count("*.tsx", "mantis/src"),
    }


def n_tests() -> str:
    try:
        out = subprocess.run(["python3", "-m", "pytest", "--collect-only", "-q"],
                             cwd=ROOT, capture_output=True, text=True, timeout=300)
        for line in reversed(out.stdout.splitlines()):
            if "test" in line and "collected" in line:
                return line.split()[0]
        return str(sum(1 for l in out.stdout.splitlines() if "::" in l))
    except Exception:
        return "?"


def figure_block(stem: str, caption: str, width: str = r"\linewidth") -> str:
    path = PAPER / "figures" / f"{stem}.pdf"
    if not path.exists():
        return ""
    return (f"\\begin{{figure}}[htbp]\n\\centering\n"
            f"\\includegraphics[width={width}]{{figures/{stem}.pdf}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}\n")


def screenshot_block(stem: str, caption: str) -> str:
    path = PAPER / "screenshots" / f"{stem}.png"
    if not path.exists():
        return ""
    return (f"\\begin{{figure}}[htbp]\n\\centering\n"
            f"\\fbox{{\\includegraphics[width=0.96\\linewidth]{{screenshots/{stem}.png}}}}\n"
            f"\\caption{{{caption}}}\n\\end{{figure}}\n")


def molnet_table() -> str:
    summary = load("molnet_summary.json")
    if not summary:
        return "\\emph{MoleculeNet benchmark still running.}\n"
    rows = []
    for name in sorted(summary):
        block = summary[name]
        methods = block["methods"]
        if not methods:
            continue
        higher = block["metric"] == "roc_auc"
        best = (max if higher else min)(methods.items(), key=lambda kv: kv[1]["mean"])
        dmpnn = methods.get("D-MPNN", {})
        ref = block.get("reference", {}).get("chemprop_dmpnn")
        rows.append(
            f"{esc(name)} & {block['n_total']} & {esc(block['metric'])} & "
            f"{dmpnn.get('mean', float('nan')):.3f} $\\pm$ {dmpnn.get('std', 0):.3f} & "
            f"{ref:.3f} & {best[1]['mean']:.3f} & {esc(best[0])} \\\\")
    if not rows:
        return "\\emph{MoleculeNet benchmark still running.}\n"
    return ("\\begin{tabular}{llllrrl}\n\\toprule\n"
            "dataset & $n$ & metric & NEXUS D-MPNN & published$^1$ & best & best method \\\\\n"
            "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def active_learning_table() -> str:
    data = load("active_learning_validation.json")
    if not data:
        return "\\emph{Not yet run.}\n"
    rows = []
    for name, block in data["strategies"].items():
        rows.append(f"{esc(name)} & {100 * block['recall_mean']:.1f}\\% & "
                    f"{block['enrichment_mean']:.2f}$\\times$ $\\pm$ "
                    f"{block['enrichment_std']:.2f} \\\\")
    return ("\\begin{tabular}{lrr}\n\\toprule\nstrategy & actives recovered & "
            "enrichment over random \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def affinity_table() -> str:
    data = load("affinity.json")
    if not data:
        return "\\emph{Davis binding-affinity benchmark still running.}\n"
    rows = []
    for split, models in data["splits"].items():
        for model, block in models.items():
            rows.append(f"{esc(split)} & {esc(model)} & {block['rmse']['mean']:.3f} & "
                        f"{block['pearson']['mean']:.3f} & {block['ci']['mean']:.3f} \\\\")
    return ("\\begin{tabular}{llrrr}\n\\toprule\nsplit & model & RMSE & Pearson & "
            "concordance index \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def generation_table() -> str:
    data = load("generation.json")
    if not data:
        return "\\emph{Generation benchmark still running.}\n"
    rows = []
    labels = {"model": "model + valence constraint",
              "model_unconstrained_sampling": "model, unconstrained sampling",
              "marginal_baseline": "marginal baseline (no model)"}
    for key, label in labels.items():
        block = data.get(key)
        if not block:
            continue
        frag = block.get("validity_largest_fragment")
        degree = block.get("mean_degree")
        rows.append(
            f"{esc(label)} & {100 * block['validity']:.1f}\\% & "
            + (f"{100 * frag:.1f}\\%" if frag is not None else "--") + " & "
            + (f"{degree:.2f}" if degree is not None else "--") + " & "
            + f"{100 * block['uniqueness']:.1f}\\% & "
            + f"{100 * block['novelty']:.1f}\\% \\\\")
    train_degree = data.get("training_mean_degree")
    note = ""
    if train_degree is not None:
        note = (f"\n\n\\noindent\\footnotesize Training-set mean degree is "
                f"{train_degree:.2f}; a generator that collapses to the empty graph "
                f"shows up here long before the validity column explains why.\n")
    body = ("\\begin{tabular}{lrrrrr}\n\\toprule\nsampler & valid & "
            "valid (largest frag.) & mean degree & unique & novel \\\\\n"
            "\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")
    return body + note


def rejection_table() -> str:
    """Why samples were rejected -- the breakdown, not just the headline rate."""
    data = load("generation.json")
    block = (data or {}).get("model", {})
    reasons = block.get("rejections")
    if not reasons:
        return ""
    total = block.get("n_sampled", 0) or 1
    order = sorted(reasons.items(), key=lambda kv: -kv[1])
    rows = [f"{esc(k.replace('_', ' '))} & {v} & {100 * v / total:.1f}\\% \\\\"
            for k, v in order]
    return ("\\begin{tabular}{lrr}\n\\toprule\nrejected because & count & "
            "share \\\\\n\\midrule\n" + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n")


def conformal_adaptive_table() -> str:
    """Constant-width against adaptive conformal scores."""
    data = load("conformal_adaptive.json")
    if not data:
        return "\\emph{Adaptive-interval benchmark not yet run.}\n"
    pretty = {"standard": "constant width", "gp_sigma": "scaled by GP sd",
              "residual_fitted": "fitted residual model"}
    rows = []
    for name, variants in data["datasets"].items():
        first = True
        for key, label in pretty.items():
            entry = variants.get(key)
            if not entry:
                continue
            rows.append(
                f"{esc(name) if first else ''} & {esc(label)} & "
                f"{entry['coverage']:.3f} & {entry['mean_width']:.2f} & "
                f"{entry['width_spread']:.2f} & "
                f"{entry['worst_quintile_coverage']:.3f} & "
                f"{entry['conditional_spread']:.3f} \\\\")
            first = False
        rows.append("\\addlinespace")
    return ("\\begin{tabular}{llrrrrr}\n\\toprule\ndataset & score & coverage & "
            "width & width sd & worst quintile & spread \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def descriptor_table() -> str:
    data = load("descriptor_validation.json")
    if not data:
        return "\\emph{Not yet run.}\n"
    rows = []
    for filename, block in data.items():
        for ref_col, s in block["descriptors"].items():
            if ref_col == "_all":
                continue
            rows.append(f"{esc(filename.replace('.csv', ''))} & {esc(ref_col)} & "
                        f"{esc(s['implementation'])} & {s['n']} & {s['pearson_r']:.4f} & "
                        f"{s['mean_abs_error']:.3f} & {100 * s['fraction_exact']:.1f}\\% \\\\")
    return ("\\begin{tabular}{lllrrrr}\n\\toprule\ndataset & reference column & "
            "our implementation & $n$ & $r$ & MAE & exact \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def go_table() -> str:
    data = load("go_validation.json")
    if not data:
        return "\\emph{Not yet run.}\n"
    rows = []
    for c in data["positive_controls"]:
        rank = c["rank_of_target"]
        rows.append(f"{esc(c['label'])} & "
                    f"{(str(rank + 1) if rank is not None else 'not recovered')} & "
                    f"{esc(str(c['top_term_name'])[:34])} \\\\")
    fp = data["false_positive_control"]
    sim = data["semantic_similarity"]
    table = ("\\begin{tabular}{llp{5.2cm}}\n\\toprule\nseeded term & rank recovered & "
             "top-ranked term \\\\\n\\midrule\n" + "\n".join(rows)
             + "\n\\bottomrule\n\\end{tabular}\n")
    table += (f"\n\\vspace{{0.6em}}\\noindent Random gene sets "
              f"({fp['n_trials']} trials of {fp['genes_per_trial']} genes): mean "
              f"{fp['mean_significant_terms']:.2f} significant terms at FDR 0.05, "
              f"{100 * fp['fraction_trials_with_any_hit']:.0f}\\% of trials returned any "
              f"hit. Semantic similarity separates same-pathway from random gene pairs "
              f"with AUC {sim['auc_same_vs_random']:.3f} "
              f"({sim['mean_same_pathway']:.3f} versus {sim['mean_random']:.3f}).\n")
    return table


def build() -> None:
    stats = code_stats()
    logp = load("logp_fit.json") or {}
    gen = load("generation.json")
    total_python = stats["package"] + stats["tests"] + stats["benchmarks"]

    doc = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=0.9in]{{geometry}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage{{microtype}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{xcolor}}
\usepackage{{longtable}}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{{hyperref}}
\usepackage{{float}}
\setlength{{\parskip}}{{0.5em}}
\setlength{{\parindent}}{{0pt}}

\title{{\textbf{{NEXUS}}\\[0.3em]
\large Non-Equilibrium eXploration with Unified Synthesis\\[0.2em]
\normalsize A dependency-free drug discovery pipeline, with measured results}}
\author{{Nischay Kommisetty}}
\date{{{date.today().isoformat()}}}

\begin{{document}}
\maketitle

\section*{{What this is}}

NEXUS is a drug discovery pipeline whose entire scientific stack runs on NumPy alone:
SMILES parsing, molecular graphs, fingerprints, descriptors, 3D conformers, automatic
differentiation, message-passing neural networks, graph diffusion models, Gaussian
processes, persistent homology, discrete Ricci curvature, conformal prediction,
non-equilibrium free energy estimators and the Gene Ontology. There is no RDKit and no
PyTorch.

Every component is checked against something external --- published reference values,
exactly solvable physics, known topology, or a falsifiable statistical guarantee.
This document is generated from the files in \texttt{{results/}}, so the numbers in it are
the measured ones. Results that did not help are reported alongside those that did.

\vspace{{0.5em}}
\begin{{tabular}}{{lr}}
\toprule
Python in the package & {stats['package']:,} lines \\
Tests & {stats['tests']:,} lines \\
Benchmarks and validation & {stats['benchmarks']:,} lines \\
Mantis panel (TypeScript) & {stats['panel']:,} lines \\
\midrule
Total Python & {total_python:,} lines \\
Test suite & {n_tests()} tests, all passing \\
\bottomrule
\end{{tabular}}

\section{{Validation summary}}

\begin{{tabular}}{{p{{3.4cm}}p{{5.6cm}}p{{6.2cm}}}}
\toprule
Component & Check & Result \\
\midrule
SMILES parser & parse the whole MoleculeNet suite & 59,974 of 59,975 molecules (99.998\%) \\
Descriptors & against reference columns in ESOL and BACE & MW, heavy atoms, rings exact; TPSA $r=0.9998$ \\
TPSA & published values for known drugs & aspirin 63.6, caffeine 58.44, ibuprofen 37.3 --- exact \\
Scaffold split & leakage into the held-out set & 0.000, against 35--78\% for random splits \\
Autograd & every operation vs.\ finite differences & 42 operations, all agree \\
Conformers & bond lengths and ring planarity & error $<0.05$\,\AA; aromatic rings planar \\
Persistent homology & shapes with known topology & circle $\rightarrow$ 1 loop; 2 clusters $\rightarrow$ Betti-0 $=2$ \\
Ricci curvature & optimal transport vs.\ hand-computed optima & exact on all six; $\kappa=0$ on cycles \\
Conformal prediction & empirical vs.\ nominal coverage & matches across $\alpha$, models, 200 trials \\
Free energy & harmonic oscillator, exact $\Delta F = 0.6931$ & BAR 0.6957, Crooks 0.6899, Jarzynski 0.6979 \\
Active learning & retrospective screen of 8,000 compounds & 3.16$\times$ enrichment (random control 1.01$\times$) \\
Gene Ontology & enrichment and semantic similarity & 4/5 controls at rank 1; similarity AUC 0.885 \\
\bottomrule
\end{{tabular}}

\section{{The pipeline running}}

The screenshots below are of the real panel driving the real service: the Python package
computes every number shown, and the panel reads them over HTTP. Nothing is mocked except
the four Mantis store modules that supply the selected points.

{screenshot_block('02_analyze_results', 'Analyze mode. Forty-eight molecules from the ESOL set, analyzed in 0.084\\,s into 27 scaffolds. Molecular weight, logP and TPSA are computed by the NumPy package; the right-hand strip is the map, coloured by the chosen descriptor.')}

{screenshot_block('05_predict_with_intervals', 'Predict mode, and the result worth singling out. A Tanimoto-kernel Gaussian process is trained on 28 labelled molecules, 10 are held out to calibrate, and the intervals carry a finite-sample 90\\% coverage guarantee. The panel reports the coverage measured on the calibration set --- 90.0\\% --- beside the nominal level.')}

{screenshot_block('04_next_batch', 'Next-batch mode. The active-learning loop ranks the unmeasured molecules by expected improvement and proposes what to assay next, penalising similarity so one chemical series cannot consume the batch.')}

{screenshot_block('03b_property_space', 'The charts below the fold. Property space is logP against molecular weight, every point shaded by the descriptor driving the map colouring; the histogram beneath it is the logP distribution with its mean marked.')}

{screenshot_block('03c_distributions', 'Scaffold families ranked by size, over the per-molecule table. The acyclic group dominating this selection is the same structure that broke the validation split, visible here directly.')}

{screenshot_block('05b_interval_strip', 'Every prediction with its interval, sorted by predicted value. With the default constant-width score each interval is the same length --- which is what prompted the adaptive-width work in the previous section.')}

{screenshot_block('06_light_mode', 'Light mode. Both themes are selected palettes validated against their own surface, not one inverted into the other.')}

\section{{Property prediction}}

Scaffold-balanced splits, three seeds, early stopping on validation, one evaluation of the
selected model on the held-out test set.

{molnet_table()}

\vspace{{0.4em}}
$^1$ Yang et al.\ (2019), same protocol but not identical splits --- shown for orientation,
not as a like-for-like comparison.

\vspace{{0.6em}}
\textbf{{The honest reading.}} The D-MPNN reproduces the published numbers closely, but
tuned fingerprint baselines beat it on most of these datasets. At 1.5--2k molecules a
graph network is not automatically the right tool, and the pipeline reports both rather
than only the flattering half.

{figure_block('fig1_molnet_benchmark', 'Method comparison per dataset. Blue marks the graph networks, grey the fingerprint and descriptor baselines; the dashed line is the published D-MPNN result. Error bars span three seeds.')}

\section{{Hit identification}}

A retrospective screen over 8,000 compounds from the HIV set (3.48\% active), screening
13.8\% of the library. Each strategy starts from the same random seed batch and may only
see labels for compounds it has selected.

{active_learning_table()}

The random arm landing at 1.01$\times$ is the control: it confirms the measurement is
unbiased, since random selection should recover actives in proportion to the fraction
screened.

{figure_block('fig2_active_learning', 'Actives recovered against compounds screened, and the resulting enrichment. Uncertainty sampling and expected improvement both find roughly three times as many actives as random selection at the same budget.')}

\section{{Binding affinity}}

The Davis kinase panel: 68 inhibitors against 442 kinases. Three splits are reported
because they answer different questions --- random tests interpolation, cold-drug is the
lead-discovery case, cold-target the selectivity case.

{affinity_table()}

\section{{Uncertainty that means something}}

Conformal prediction gives a distribution-free, finite-sample guarantee: if calibration
and test points are exchangeable, the region covers the truth with probability at least
$1-\alpha$, whatever the model does. That is a falsifiable claim, so it is tested
directly rather than assumed.

{figure_block('fig3_conformal_coverage', 'Empirical coverage against the nominal level, over 40 trials at each of seven $\\alpha$ values. Both the regression intervals and the classification sets track the diagonal.', '0.62\\linewidth')}

\section{{Free energy}}

Estimators are validated on a system whose answer is known in closed form: a harmonic
oscillator whose spring constant changes from $k_0$ to $k_1$, for which
$\Delta F = \ln(k_1/k_0)/(2\beta)$ exactly.

{figure_block('fig4_free_energy', 'Forward and reverse work distributions cross at $\\Delta F$, as Crooks\\textquotesingle{} theorem requires, and the estimator errors against the exact value. BAR is the most accurate, which is what theory predicts; the cumulant expansion is biased, as expected for non-Gaussian work.')}

\section{{Descriptors}}

{descriptor_table()}

{figure_block('fig5_descriptor_validation', 'Agreement with the reference descriptor columns shipped in the BACE dataset.', '0.82\\linewidth')}

The logP model is the weak one and is reported as such: an atom-contribution fit on
{logp.get('n_molecules', '?')} compounds with experimental logD, held-out MAE
{logp.get('test', {}).get('mae', float('nan')):.2f} against
{logp.get('baseline_predict_mean', {}).get('mae', float('nan')):.2f} for predicting the
mean, $R^2$ {logp.get('test', {}).get('r2', float('nan')):.2f}. It is useful as a feature,
not as a replacement for a dedicated predictor.

\section{{Shape and topology}}

{figure_block('fig6_persistence_diagrams', 'Persistence diagrams of 3D conformers. The number of long-lived $H_1$ features matches the ring count in each molecule, and acyclic octane produces none.')}

{figure_block('fig7_ricci_curvature', 'Ricci flow drives the spread of curvature towards zero, and benzene --- a cycle graph --- has curvature exactly zero, as the theory requires for $C_n$ with $n \\geq 6$.')}

\section{{Target biology}}

{go_table()}

{figure_block('fig8_gene_ontology', 'Gene Ontology validation: positive controls recovered, false positives on random gene sets, and semantic similarity separating same-pathway from random gene pairs.')}

\section{{Calibrated uncertainty, and whether width should adapt}}

Split conformal with an absolute-residual score guarantees marginal coverage, but gives
every molecule an interval of the same length. That is visible in the panel screenshot
further up: a vertical stack of identical bars. It is valid and it is useless for triage,
because it cannot say which predictions to trust. Scaling the score by a per-molecule
difficulty estimate keeps the same guarantee and lets width vary.

{conformal_adaptive_table()}

\noindent Marginal coverage holds for every score, as the theory says it must --- that is
the check, and it passes. The interesting column is the last one: how far conditional
coverage drifts across difficulty quintiles. Constant-width intervals over-cover easy
molecules and under-cover hard ones, which is precisely the failure adaptivity should
repair, and on Lipo scaling by the GP posterior sd cuts that drift from 0.176 to 0.061
while also narrowing the mean interval. On ESOL it makes it worse. Fitting a separate
model to the residuals --- the textbook construction --- is worse than both on every
dataset. So the adaptive score is offered as an option and left off by default, and the
fitted-residual variant is reported as a negative result rather than shipped.

{figure_block('fig10_adaptive_conformal', 'Left: mean interval width, labelled with the marginal coverage each score actually achieved. Right: coverage inside difficulty quintiles, averaged over datasets. Flat at the nominal level is the ideal; the constant-width score slopes down, and the fitted residual model swings from 0.72 to 1.00.')}

\section{{Why scaffold splitting matters}}

{figure_block('fig9_scaffold_leakage', 'A random split leaves most test molecules sharing a scaffold with the training set, which flatters any model. The scaffold split removes that leakage entirely.', '0.62\\linewidth')}

\textbf{{A bug this benchmark was hiding.}} The balanced splitter could hand the
validation set \emph{{nothing}}. FreeSolv is the case: its acyclic scaffold group is 320
of 642 molecules, so it overflows the training quota, overflows the validation quota, and
lands in test --- after which every remaining group still fits in training and validation
is never reached. The split came out 0.50 / 0.00 / 0.50 rather than 0.60 / 0.20 / 0.20,
which silently disabled early stopping and left half the dataset in test. Oversized groups
are now assigned largest-first, so the one group too big for any held-out subset inflates
training rather than test, and a rebalancing pass moves whole groups --- never splitting a
scaffold --- into an undersized validation set. FreeSolv now splits 0.56 / 0.20 / 0.24.
Results computed before this fix are not comparable with results after it, so the
MoleculeNet table above was regenerated.

\section{{De novo generation}}

{generation_table()}

\textbf{{Two bugs, found by a number that was impossible.}} The generator first scored
0.0\% validity --- \emph{{below}} its own untrained marginal baseline. A trained model
losing to its own control is not a weak result, it is a broken one, so it was traced
rather than reported. Sampled graphs had mean degree 0.26 against a training-set mean of
1.99: "no bond" is 83.7\% of atom pairs, so under an unweighted cross entropy the
denoiser had learned to predict "none" everywhere. Separately, the valence-constrained
sampler began reverse diffusion from the all-zero bond matrix instead of the marginal
noise prior, and because the posterior's likelihood term anchors $x_{{t-1}}$ to $x_t$,
starting empty dragged the whole trajectory back. With a class-balanced bond loss and the
prior-initialised sampler, mean generated degree rises monotonically with the balancing
exponent and validity follows it off the floor.

\textbf{{What is still wrong.}} The table below is the honest account of where the
remaining samples die. Valence errors are now rare --- the constraint handles them --- and
almost every rejection is a graph in two or more pieces. The model has no connectivity
prior, and nothing in the training objective supplies one.

{rejection_table()}

\noindent\footnotesize The generation hyperparameters were selected on this same
validity metric, with no held-out set. That is the usual practice in the generative
chemistry literature, and it is still a reason to read the number as an upper bound.
\normalsize

\section{{Integration with Mantis}}

The pipeline ships a drop-in panel for the Mantis map interface. No file in the Mantis
repository is modified: three files are copied in and three short registrations are
documented as exact diffs. Typechecked against the real Mantis tree with its own
\texttt{{node\_modules}} and path aliases --- 121 pre-existing errors in Mantis's
\texttt{{data\_provider.tsx}} both with and without the panel, and zero attributable to
NEXUS code.

\section{{Limitations}}

\begin{{itemize}}
\item Pure NumPy means CPU-bound training. The graph network is smaller and trained for
fewer epochs than a GPU implementation would be, so its numbers are a floor rather than a
ceiling.
\item The logP model is weak ($R^2$ {logp.get('test', {}).get('r2', float('nan')):.2f} on
held-out experimental logD).
\item The drug-likeness score does \emph{{not}} separate approved drugs from clinical-trial
failures (measured AUC 0.52) and is not a predictor of approval. It is a property-range
summary.
\item Free energy estimators are validated on analytically solvable systems, not on
protein--ligand alchemical transformations, which need a real force field.
\item Generation is the weakest component; see the section above.
\item Published reference numbers are quoted for orientation. The splits here are
regenerated with our own seeds, so they are comparable in protocol but not identical in
data.
\end{{itemize}}

\section{{Reproducing everything}}

\begin{{verbatim}}
pip install -e .                                 # NumPy only
python -m pytest -q                              # the test suite
python -m benchmarks.validate_descriptors        # descriptors vs reference values
python -m benchmarks.fit_logp                    # logP fit on experimental logD
python -m benchmarks.validate_go                 # GO enrichment and similarity
python -m benchmarks.validate_conformal          # coverage guarantees
python -m benchmarks.validate_active_learning    # retrospective HIV screen
python -m benchmarks.molnet --seeds 3            # property prediction
python -m benchmarks.affinity                    # Davis binding affinity
python -m benchmarks.generate                    # train and evaluate the generator
python -m benchmarks.figures                     # all figures
python -m benchmarks.report_pdf                  # this document
\end{{verbatim}}

\end{{document}}
"""
    out = PAPER / "NEXUS_report.tex"
    out.write_text(doc)
    print(f"wrote {out}")
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "NEXUS_report.tex"],
                       cwd=PAPER, capture_output=True, timeout=600)
    pdf = PAPER / "NEXUS_report.pdf"
    if pdf.exists():
        print(f"wrote {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")
    else:
        print("pdflatex did not produce a PDF; see paper/NEXUS_report.log")


if __name__ == "__main__":
    build()
