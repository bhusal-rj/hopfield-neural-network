#!/usr/bin/env bash
set -euo pipefail
OUT=hopfield_report.pdf
TEX=hopfield_report.tex
FIGDIR=figures

# ensure figures exist or warn
if [ ! -d "${FIGDIR}" ]; then
  echo "Warning: ${FIGDIR} not found. Create it and add figures (patterns.png, image_denoising.png, capacity.png) before compiling."
fi

# Run pdflatex twice for crossrefs
pdflatex -interaction=nonstopmode "${TEX}"
pdflatex -interaction=nonstopmode "${TEX}"

echo "Built ${OUT}"