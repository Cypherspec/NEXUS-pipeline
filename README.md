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

## License

MIT
