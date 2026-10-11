#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper from scratch.
#   ./reproduce.sh            full run (~10 h on 4 CPU cores incl. AutoGluon)
#   WORKERS=8 ./reproduce.sh  use more cores
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"

pip install -r requirements-lock.txt
python -m pytest -q
python -m benchmark.run --workers "$WORKERS" | tee results/run.log
python -m benchmark.run --variant v2 --workers "$WORKERS" | tee results/run_v2.log
# AutoGluon needs its own environment (autogluon.tabular 1.6); set AG_PYTHON to its interpreter.
if [ -n "${AG_PYTHON:-}" ]; then
  "$AG_PYTHON" -m benchmark.run_autogluon --workers "$WORKERS" | tee results/run_autogluon.log
else
  echo "AG_PYTHON not set: skipping AutoGluon (analysis will omit it)"
fi
WORKERS="$WORKERS" python -m benchmark.analyze
python -m benchmark.figures
python -m benchmark.dev_report
if command -v pdflatex >/dev/null; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
  echo "paper/main.pdf built"
fi
