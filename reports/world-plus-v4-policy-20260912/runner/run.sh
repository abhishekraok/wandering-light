#!/usr/bin/env bash
# world-plus-v4-policy (#134): two matched arms, 1m, seeds 0/1/2, 6 epochs.
# Called with .venv/bin/python directly: `uv run` swallows SIGTERM, so a
# timeout or a kill would not reach the trainer.
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260912
V4=wandering_light/evals/data/stack_pbe_v4/manifest.json
W3=wandering_light/evals/data/stack_pbe_world_v3/manifest.json
PY=.venv/bin/python
COMMON="--arms 1m --seeds 0 1 2 --epochs 6 --train-device cuda --devices cuda --no-wandb"

echo "=== arm A train (v4 only)"
$PY experiments/compute_optimal_mark28.py train --run-dir $R/v4only $COMMON --corpus $V4 || exit 1
echo "=== arm B train (v4 + world_v3, overlap removed, validation pinned to v4)"
$PY experiments/compute_optimal_mark28.py train --run-dir $R/mixed $COMMON \
  --corpus $V4 $W3 --validation-corpus stack_pbe_v4 --drop-behaviours $R/mixed/drop.txt || exit 1

for arm in v4only mixed; do
  if [ "$arm" = mixed ]; then CORPORA="--corpus $V4 $W3"; else CORPORA="--corpus $V4"; fi
  echo "=== $arm: v4 cohort (300 tasks, beam 64 x depth 48)"
  $PY experiments/compute_optimal_mark28.py time --run-dir $R/$arm $COMMON $CORPORA || exit 1
  echo "=== $arm: world eval (55 tasks, beam 64 x depth 48)"
  $PY experiments/compute_optimal_mark28.py time --run-dir $R/$arm $COMMON \
    --corpus $V4 $W3 --eval-corpus stack_pbe_world_v3 || exit 1
  echo "=== $arm: replay hits"
  $PY experiments/compute_optimal_mark28.py replay --run-dir $R/$arm --corpus $V4 $W3 || exit 1
done
echo "=== done"
