#!/usr/bin/env bash
# keep-below-ablation (#136), after study.sh: verify every solved program in the
# #134 epoch study and the two ablation arms with the same 10k-draw check, then
# the ablation tables. The #134 study dirs live in the step-3 worktree's
# checkpoints, so they are passed by absolute path.
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260913-ablation
S3=/home/abhishekrao/repos/wandering-light/.claude/worktrees/stack-machine-training-data-07b097/checkpoints
REP=reports/keep-below-ablation-20260913
PY=.venv/bin/python
mkdir -p $REP
echo "=== verify (10k draws per mode, dense, floor 500)"
$PY experiments/step3_verify_solves.py \
  --run-dir $S3/step3-20260912/v4only --run-dir $S3/step3-20260912/mixed \
  --run-dir $S3/step3-20260913-epochs/v4only --run-dir $S3/step3-20260913-epochs/mixed \
  --run-dir $R/ablated --run-dir $R/random \
  --draws 10000 --dense --min-tested 500 \
  --out $REP/verified_solves_ablation.json > $R/verify_10k.log 2>&1 || { echo "FAILED verify"; exit 1; }
echo "=== ablation tables"
$PY experiments/step3_ablation_report.py \
  --arm-dir v4only=$S3/step3-20260913-epochs/v4only --arm-dir mixed=$S3/step3-20260913-epochs/mixed \
  --arm-dir ablated=$R/ablated --arm-dir random=$R/random \
  --best v4only=$S3/step3-20260912/v4only --best mixed=$S3/step3-20260912/mixed \
  --verified $REP/verified_solves_ablation.json --out $REP > $R/report.log 2>&1 || { echo "FAILED report"; exit 1; }
echo "=== post done"
