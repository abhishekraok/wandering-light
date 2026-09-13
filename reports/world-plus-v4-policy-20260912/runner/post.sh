#!/usr/bin/env bash
# world-plus-v4-policy (#134), after run.sh: the GPU reads that follow the cells.
# Cross-entropy first, on the world-v3 revision the cells were scored on (the
# contaminated eval behaviours are removed only after this), then the depth-192
# structural cells. Plain .venv python: `uv run` swallows SIGTERM.
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260912
OUT=reports/world-plus-v4-policy-20260912
PY=.venv/bin/python
mkdir -p "$OUT"
echo "=== ce reads"
$PY experiments/step3_ce.py \
  --checkpoint $R/v4only/policies/1m-s0 --checkpoint $R/v4only/policies/1m-s1 --checkpoint $R/v4only/policies/1m-s2 \
  --checkpoint $R/mixed/policies/1m-s0 --checkpoint $R/mixed/policies/1m-s1 --checkpoint $R/mixed/policies/1m-s2 \
  --dropped codearc-128 --dropped mbpp-404 --dropped mbpp-970 \
  --device cuda --out $OUT/ce.json || exit 1
echo "=== structural"
bash $R/structural.sh || exit 1
echo "=== post done"
