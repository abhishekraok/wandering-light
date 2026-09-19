#!/usr/bin/env bash
# keep-below-ablation (#136): the session restart killed study.sh during replay
# (8 of 24 cells replayed, 0 mismatches). Replay every cell again, then post.sh.
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260913-ablation
V4=wandering_light/evals/data/stack_pbe_v4/manifest.json
W3=wandering_light/evals/data/stack_pbe_world_v3/manifest.json
PY=.venv/bin/python
D=experiments/compute_optimal_mark28.py
echo "=== replay (resumed)"
$PY $D replay --run-dir $R/ablated --corpus $V4 $W3 --no-wandb || { echo "FAILED replay ablated"; exit 1; }
$PY $D replay --run-dir $R/random --corpus $V4 $W3 --no-wandb || { echo "FAILED replay random"; exit 1; }
echo "=== ablation study done"
bash $R/post.sh
