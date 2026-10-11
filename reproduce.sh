#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper from scratch.
#   ./reproduce.sh            full run (~10 h on 4 CPU cores incl. AutoGluon)
#   WORKERS=8 ./reproduce.sh  use more cores
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"

pip install -r requirements-lock.txt
python -m pytest -q
python -m tabular_ensemble_benchmark.run --workers "$WORKERS" | tee results/run.log
python -m tabular_ensemble_benchmark.run --variant v2 --workers "$WORKERS" | tee results/run_v2.log
# AutoGluon needs its own environment (autogluon.tabular 1.6); set AG_PYTHON to its interpreter.
if [ -n "${AG_PYTHON:-}" ]; then
  "$AG_PYTHON" -m tabular_ensemble_benchmark.run_autogluon --workers "$WORKERS" | tee results/run_autogluon.log
else
  echo "AG_PYTHON not set: skipping AutoGluon (analysis will omit it)"
fi
WORKERS="$WORKERS" python -m tabular_ensemble_benchmark.analyze
python -m tabular_ensemble_benchmark.figures
python -m tabular_ensemble_benchmark.dev_report
if command -v pdflatex >/dev/null; then
  (cd paper_tabular && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
  echo "paper_tabular/main.pdf built"
fi
