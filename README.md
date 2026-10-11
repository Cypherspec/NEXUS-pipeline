# NEXUS

**Cross-fitted ensemble selection with stacking and calibration for tabular classification**, plus a
pre-registered, fully reproducible benchmark on OpenML-CC18.

NEXUS is a scikit-learn compatible classifier. It trains a fixed library of eight diverse
algorithms with bagged K-fold cross-fitting, adds a small second stacking layer, picks
sparse convex ensemble weights by greedy forward selection on out-of-fold predictions, and
calibrates the result with a single temperature. There is no hyperparameter search and no
time budget: given the same data and seed it always produces the same model.

<!-- RESULTS:START -->
<!-- RESULTS:END -->

## Install

```bash
pip install -e .            # library only
pip install -e ".[dev]"     # + tests and figure dependencies
```

Python ≥ 3.10. Exact versions used for the paper are in `requirements-lock.txt`.

## Use

```python
from nexus import NexusClassifier

clf = NexusClassifier(n_jobs=4)          # NEXUS v2 (default)
clf.fit(X_train, y_train)                # pandas DataFrame or numpy; categoricals + NaNs OK
proba = clf.predict_proba(X_test)
labels = clf.predict(X_test)

clf.weights_       # ensemble weight per member, e.g. {"catboost": 0.36, "L2:lightgbm": 0.22, ...}
clf.temperature_   # fitted calibration temperature
```

| Parameter | Default | Meaning |
|---|---|---|
| `models` | `"default"` | layer-1 library (8 algorithms, see paper Table 1) or a list of names |
| `stack_models` | `("lightgbm", "logreg")` | layer-2 models on features + layer-1 OOF probabilities; `None` = NEXUS v1 |
| `n_folds` | `5` | cross-fitting folds (shrinks automatically for rare classes) |
| `n_repeats` | `1` | repetitions of cross-fitting |
| `ensemble_size` | `50` | greedy selection iterations |
| `calibrate` | `True` | temperature scaling of the final mixture |
| `n_jobs` | `1` | parallel model fits |

## Reproduce the paper

```bash
./reproduce.sh            # tests → benchmark → analysis → figures → paper/main.pdf
```

or step by step:

```bash
python -m benchmark.run --workers 4                    # NEXUS v1 + all baselines, 54 tasks × 10 folds
python -m benchmark.run --variant v2 --workers 4       # NEXUS v2
/path/to/autogluon-env/bin/python -m benchmark.run_autogluon --workers 4   # time-matched AutoGluon
python -m benchmark.analyze                            # metrics, statistics, tables, numbers.tex
python -m benchmark.figures                            # paper/figures/*.pdf|png
```

Development-set study (datasets disjoint from CC18, used for all design decisions):

```bash
python -m benchmark.dev members --members lightgbm xgboost ...   # cache member predictions
python -m benchmark.dev_report                                   # paper/tables/dev.*
```

## Repository layout

```
nexus/               the library (classifier, model zoo, ensemble selection, calibration, metrics)
benchmark/           suite loader, runners, analysis, figures, development-set harness
tests/               unit tests (pytest)
results/             per-fold/per-task metrics and statistics (raw predictions are regenerable, not committed)
paper/               LaTeX source, auto-generated tables, numbers.tex and figures
```

## Protocol in one paragraph

All design decisions were made on 16 OpenML datasets that are not in (or derived from) CC18,
and the final configuration was committed before its first run on CC18. Evaluation uses the
official OpenML 10-fold splits of the 54 CC18 tasks with ≤12,000 rows and ≤300 features
(an inclusion rule fixed in advance for a 4-CPU budget). Every method sees identical
splits; ensembling baselines share NEXUS's exact base-model predictions; AutoGluon gets the
same per-fold wall-clock budget NEXUS used. Statistics: Friedman + Nemenyi critical
differences, Holm-corrected Wilcoxon signed-rank tests, bootstrap confidence intervals.

## Citation

See `CITATION.cff`.

## License

MIT
