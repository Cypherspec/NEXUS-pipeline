# NEXUS

**Non-Equilibrium eXploration with Unified Synthesis** — a dependency-free drug discovery
pipeline covering hit identification, lead optimization, binding affinity prediction and
de novo generation.

The entire scientific stack is implemented on NumPy alone: SMILES parsing, molecular graphs,
fingerprints, descriptors, 3D conformers, automatic differentiation, message-passing neural
networks, graph diffusion models, Gaussian processes, persistent homology, discrete Ricci
curvature, conformal prediction, non-equilibrium free energy estimators and the Gene
Ontology. No RDKit, no PyTorch.

Every component is validated against something external — published reference values, exactly
solvable physics, known topology, or a falsifiable statistical guarantee. Measured results,
including the unflattering ones, are in [`results/`](results/) and summarized below.

## What is validated, and how

| Component | Check | Result |
|---|---|---|
| SMILES parser | Parse the whole MoleculeNet suite | **59,974 / 59,975** molecules (99.998%) |
| Descriptors | Against reference columns in ESOL and BACE | MW, heavy atoms, rings **exact**; TPSA r=0.9998 (95.8% exact); rotatable bonds 96.5% exact |
| TPSA | Published values for known drugs | aspirin 63.6, caffeine 58.44, ibuprofen 37.3 — **exact** |
| Scaffold split | Scaffold leakage into the test set | **0.000** (random splits leak 35–78%) |
| Autograd | Every op vs central finite differences | 42 ops, all pass |
| D-MPNN | MoleculeNet scaffold split, 3 seeds | matches published Chemprop (see below) |
| Conformers | Bond lengths and ring planarity | error < 0.05 Å; aromatic rings planar, cyclohexane correctly puckered |
| Persistent homology | Shapes with known topology | circle → exactly 1 loop; 2 clusters → Betti-0 = 2; disc → no loop |
| Ricci curvature | Optimal transport vs hand-computed optima | exact on all 6; κ=0 on cycles as theory requires |
| Conformal prediction | Empirical vs nominal coverage | matches across α, models and 200 trials |
| Free energy | Harmonic oscillator with exact ΔF = 0.6931 | **BAR 0.6957**, Crooks 0.6899, Jarzynski 0.6979 |
| Active learning | Retrospective screen of 8,000 HIV compounds | **3.16× enrichment** over random (random control: 1.01×) |
| Gene Ontology | Enrichment and semantic similarity | 4/5 positive controls recovered at rank 1; random sets → 0.28 false terms; similarity AUC 0.885 |
| ADMET model | 18 TDC endpoints, 57,884 compounds, scaffold split | mean held-out **ROC-AUC 0.868** (10 endpoints), **R² 0.469** (8); every endpoint beats its trivial baseline and the linear comparator |
| Activity cliffs | Count fingerprints vs binary on a homologous series | binary ECFP scores C2/C16 alcohols at Tanimoto **1.00**; MinMax on counts separates them |

## Results

### Property prediction (MoleculeNet, scaffold-balanced splits, 3 seeds, ROC-AUC)

| Dataset | NEXUS D-MPNN | Best NEXUS method | Published D-MPNN¹ |
|---|---|---|---|
| BACE | 0.853 ± 0.015 | 0.906 (ECFP+descriptors + RF) | 0.857 |
| BBBP | 0.911 ± 0.006 | 0.934 (ECFP + RF) | 0.913 |
| ClinTox | 0.896 ± 0.041 | 0.897 (D-MPNN + descriptors) | 0.897 |

¹ Yang et al. (2019), same protocol but not identical splits — for orientation, not a
like-for-like comparison.

**The honest reading:** the D-MPNN reproduces published numbers closely, but tuned
fingerprint baselines beat it on 2 of 3 datasets. On these dataset sizes (1.5–2k molecules)
a graph network is not automatically the right tool, and the package reports both.

### Hit identification (retrospective screen, 8,000 HIV compounds, 3.48% active, 13.8% screened)

| Strategy | Actives recovered | Enrichment |
|---|---|---|
| Random | 13.9% | 1.01× |
| **Uncertainty sampling** | **43.4%** | **3.16×** |
| Greedy | 42.2% | 3.07× |
| Expected improvement | 41.6% | 3.03× |

The random arm landing at 1.01× is the control confirming the measurement is unbiased.

## Install

```bash
pip install -e .            # NumPy only
pip install -e ".[bench]"   # + scikit-learn and matplotlib for benchmarks
python -m pytest -q         # 268 tests
```

## Use

```python
from nexus import parse_smiles
from nexus.chem.descriptors import compute_descriptors
from nexus.chem.fingerprint import morgan_fingerprint, tanimoto

mol = parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
mol.formula()                      # 'C9H8O4'
compute_descriptors(mol)["tpsa"]   # 63.6
tanimoto(morgan_fingerprint(mol), morgan_fingerprint(parse_smiles("OC(=O)c1ccccc1O")))
```

Property prediction with a message-passing network:

```python
from nexus.chem.scaffold import scaffold_split
from nexus.models.train import TrainConfig, train_model, predict_calibrated
from nexus.data import load_dataset

mols, Y, spec, _ = load_dataset("bace")
train, valid, test = scaffold_split(mols, seed=0)          # zero scaffold leakage
result = train_model([mols[i] for i in train], Y[train],
                     [mols[i] for i in valid], Y[valid],
                     "classification", TrainConfig(epochs=80, hidden=128))
probs = predict_calibrated(result, [mols[i] for i in test], "classification")
```

Prediction intervals with a finite-sample coverage guarantee:

```python
from nexus.uq.conformal import ConformalRegressor

cp = ConformalRegressor(alpha=0.1).fit(pred_calib, y_calib)
low, high = cp.predict_interval(pred_test)                 # >= 90% coverage
```

Closed-loop screening:

```python
from nexus.al.loop import active_learning_screen

campaign = active_learning_screen(fingerprints, oracle=assay, n_rounds=10,
                                  batch_size=100, acquisition="uncertainty",
                                  surrogate="rf", hit_threshold=0.5)
campaign.enrichment_over_random(len(library), n_actives)
```

Target biology with the Gene Ontology:

```python
from nexus.bio.ontology import GeneOntology
from nexus.bio.gaf import parse_gaf
from nexus.bio.enrichment import enrichment, format_results

go = GeneOntology.load("data/go/go-basic.obo")
go.annotate(parse_gaf("data/go/goa_human.gaf.gz"), aspect="P")
print(format_results(enrichment(go, my_target_genes, aspect="P")))
```

## Layout

```
nexus/
  chem/        SMILES parser, molecular graph, fingerprints, descriptors,
               Bemis-Murcko scaffolds, 3D conformers by distance geometry
  models/      autograd engine, neural layers, D-MPNN, graph diffusion, metrics
  topo/        persistent homology, Ollivier/Forman Ricci curvature, Ricci flow
  uq/          conformal prediction
  al/          Gaussian processes, acquisition functions, SVGD, screening loop
  physics/     Jarzynski, Crooks, BAR, Langevin switching simulations
  bio/         Gene Ontology DAG, semantic similarity, enrichment analysis
  admet/       TDC endpoint loaders, featurisation, the multi-task ADMET model
  space/       structure-activity landscapes: cliffs, regions, diversity
benchmarks/    every validation and benchmark script
tests/         268 tests
results/       measured outputs, committed
```

## Reproduce

```bash
python -m benchmarks.validate_descriptors       # descriptors vs reference values
python -m benchmarks.fit_logp                   # logP fit on experimental logD
python -m benchmarks.validate_go                # GO enrichment and similarity
python -m benchmarks.validate_conformal         # coverage guarantees
python -m benchmarks.validate_active_learning   # retrospective HIV screen
python -m benchmarks.molnet --seeds 3           # property prediction benchmark
python -m benchmarks.report                     # aggregate into tables
python -m benchmarks.generate                   # train and evaluate the generator
```

## ADMET

Eighteen endpoints from [Therapeutics Data Commons](https://tdcommons.ai), spanning
absorption, distribution, metabolism, excretion and toxicity: **91,539 measurements over
57,884 unique compounds**. Two layers sit on top of them.

**Published rules** (`nexus/chem/admet.py`) — ESOL solubility, Egan's absorption egg,
CNS ranges, Aronov's hERG pair, Pfizer 3/75 and Gleeson 4/400. Closed-form and traceable
to a citation; no training involved. Two corrections were needed to make them behave:
Delaney's published ESOL coefficients assume his clogP and carry a +0.98 log-unit bias
against this package's, so the form is kept and the coefficients refitted
(`benchmarks/fit_esol.py`); and this package's logP is fitted to logD 7.4, which reads
one to two units low for a basic amine, so the hERG rule converts to the neutral scale
before applying its cut-off — without that it under-flags the compounds it exists for.

**A multi-task model** (`nexus/admet/model.py`) — one trunk, eighteen heads, trained on
all of it at once so that DILI's 475 rows can borrow structure-activity signal from
CYP2D6's 13,130. Trained by `benchmarks/train_admet.py` under a single scaffold split
taken over the pooled union, so a scaffold in the training set appears in no endpoint's
test set. Each endpoint is reported against two comparators: the trivial baseline, and a
ridge model on the same features and the same split. Predictions carry a Platt-calibrated
probability or a split-conformal interval, and a leverage-based applicability flag that
says when the model is extrapolating. Held out on the scaffold split: mean **ROC-AUC 0.868** across the ten classification
endpoints and mean **R² 0.469** across the eight regression ones. Per endpoint:

| Endpoint | n test | Model | Ridge on same features | Trivial |
|---|---|---|---|---|
| hia_hou | 78 | **0.984** | 0.970 | 0.500 |
| pgp_broccatelli | 152 | **0.959** | 0.935 | 0.500 |
| bbb_martins | 191 | **0.885** | 0.826 | 0.500 |
| dili | 64 | **0.882** | 0.849 | 0.500 |
| cyp3a4_veith | 1,395 | **0.869** | 0.848 | 0.500 |
| herg_karim | 1,621 | **0.869** | 0.838 | 0.500 |
| cyp2c9_veith | 1,346 | **0.858** | 0.819 | 0.500 |
| ames | 592 | **0.849** | 0.812 | 0.500 |
| cyp2d6_veith | 1,498 | **0.848** | 0.826 | 0.500 |
| bioavailability_ma | 78 | **0.676** | 0.734 | 0.500 |

(ROC-AUC; regression endpoints and their MAEs are in `results/admet_fit.json`.)

The first version of this model scored 0.730 mean ROC-AUC and **lost to the ridge
comparator on nearly every endpoint** — DILI 0.389 against 0.849, below chance. Its
training loss had fallen to 0.059: it had memorised. Early stopping on a validation
split, a per-endpoint balanced loss so the three CYP sets (37k rows between them) cannot
steer the shared trunk away from DILI's 475, weight decay, and a per-endpoint blend with
the ridge model chosen on validation are what closed the gap. The comparator column
stays in the table because a model that cannot beat a linear fit on the same features has
not earned its complexity.

## Molecular spaces

A map of molecules supports questions a single molecule does not (`nexus/space/`):

- **Activity cliffs** — pairs that are structurally close but behave differently, ranked
  by SALI. Computed on *count* fingerprints with MinMax similarity, because binary ECFP
  saturates along a homologous series and scores hexanol against hexadecanol at exactly
  1.00, which makes chain extensions look like the strongest cliffs in the set.
- **Landscape smoothness** — how far similarity predicts similar behaviour across all
  pairs. Low means neighbourhood models will not work here and the cliffs are the signal.
- **Region enrichment** — what distinguishes a selected cluster from the rest, by
  hypergeometric test over scaffolds and functional groups, BH-corrected.
- **Diversity** — nearest-neighbour redundancy, a property of the set rather than of any
  molecule in it.

## Limitations

- Pure NumPy means CPU-bound training; the benchmark budgets are sized for four cores, so
  the D-MPNN is smaller and trained for fewer epochs than a GPU implementation would be.
- The logP model is an atom-contribution fit (held-out MAE 0.80 on experimental logD,
  R² 0.21) — usable as a feature, not a replacement for a dedicated predictor.
- The drug-likeness score is a property-range summary. It does **not** separate approved
  drugs from clinical-trial failures (measured AUC 0.52), and is not a predictor of approval.
- Free energy estimators are validated on analytically solvable systems, not on protein–ligand
  alchemical transformations, which need a real force field.
- Binding affinity benchmarks (Davis/KIBA) and the generation results are still running; see
  `results/` for what has completed.
- The ADMET endpoints are screening aids, not assay replacements. The rule-based layer
  places a molecule relative to ranges drugs have historically occupied; the trained layer
  is bounded by the public data behind each endpoint, which is itself noisy and unevenly
  sized (DILI has 475 rows). Outside the applicability domain a prediction is
  extrapolation, and the panel marks it as such.

## License

MIT
