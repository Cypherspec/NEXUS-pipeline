#!/usr/bin/env bash
# Reproduce every number, table and figure in the paper from scratch.
#   ./reproduce.sh            full run (~3 h on 4 CPU cores)
#   WORKERS=8 ./reproduce.sh  use more cores
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-4}"

pip install -r requirements-lock.txt
python -m pytest -q
python -m benchmark.run --workers "$WORKERS" | tee results/run.log
WORKERS="$WORKERS" python -m benchmark.analyze
python -m benchmark.figures
if command -v pdflatex >/dev/null; then
  (cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null && bibtex main >/dev/null \
     && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null)
  echo "paper/main.pdf built"
fi
