#!/usr/bin/env bash
# The world-eval floor at depth 48 is a witness-length fact, not a policy fact:
# 48 of 55 witnesses exceed the cap, median 138 ops. This cell re-searches world
# eval at depth 192 (34 of 55 in range) for both arms at seed 0, into its own
# filename so the primary cells are untouched. (#134)
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260912
V4=wandering_light/evals/data/stack_pbe_v4/manifest.json
W3=wandering_light/evals/data/stack_pbe_world_v3/manifest.json
for arm in v4only mixed; do
  echo "=== $arm: world eval at depth 192"
  .venv/bin/python experiments/compute_optimal_mark28.py time --run-dir $R/$arm \
    --arms 1m --seeds 0 --devices cuda --no-wandb --train-device cuda \
    --corpus $V4 $W3 --eval-corpus stack_pbe_world_v3 --max-depth 192 || exit 1
done
echo "=== structural done"
