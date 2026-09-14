#!/usr/bin/env bash
# keep-below-ablation (#136): does the #134 gain survive removing the 156 world
# programs that contain keep_below's witness? Arm `ablated` is #134's arm B with
# those 156 behaviours added to the drop list (seeds 0-2); arm `random` drops 156
# world train behaviours drawn at random instead (seed 0), a corpus-size control.
# Same recipe as reports/world-plus-v4-policy-20260912/runner/study.sh: 1m, 6
# epochs, snapshots at every epoch, validation pinned to the v4 holdout, then the
# 300-task v4 cohort searched at every epoch in three concurrent lanes.
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260913-ablation
V4=wandering_light/evals/data/stack_pbe_v4/manifest.json
W3=wandering_light/evals/data/stack_pbe_world_v3/manifest.json
PY=.venv/bin/python
D=experiments/compute_optimal_mark28.py
COMMON="--arms 1m --no-wandb --train-device cuda --devices cuda"
MIXED="--corpus $V4 $W3 --validation-corpus stack_pbe_v4"

echo "=== train ablated (s0-2) and random (s0), snapshots at epochs 1-6"
$PY $D train --run-dir $R/ablated $COMMON --seeds 0 1 2 --epochs 6 --save-epochs 1 2 3 4 5 6 \
  $MIXED --drop-behaviours $R/ablated/drop.txt > $R/train_ablated.log 2>&1 &
TA=$!
$PY $D train --run-dir $R/random $COMMON --seeds 0 --epochs 6 --save-epochs 1 2 3 4 5 6 \
  $MIXED --drop-behaviours $R/random/drop.txt > $R/train_random.log 2>&1 &
TB=$!
wait $TA; SA=$?
wait $TB; SB=$?
[ "$SA" = 0 ] && [ "$SB" = 0 ] || { echo "FAILED training (ablated exit $SA, random exit $SB)"; exit 1; }

echo "=== search 24 cells in 3 lanes"
run_lane() {
  local status=0 spec arm epoch seed
  for spec in "$@"; do
    IFS=: read -r arm epoch seed <<< "$spec"
    $PY $D time --run-dir $R/$arm $COMMON --seeds "$seed" --policy-epoch "$epoch" --corpus $V4 $W3 || status=1
  done
  return $status
}
ORDER=(ablated:3:0 ablated:3:1 ablated:3:2 ablated:5:0 ablated:5:1 ablated:5:2
       ablated:4:0 ablated:4:1 ablated:4:2 random:3:0 random:5:0 random:4:0
       ablated:2:0 ablated:2:1 ablated:2:2 ablated:6:0 ablated:6:1 ablated:6:2
       ablated:1:0 ablated:1:1 ablated:1:2 random:2:0 random:6:0 random:1:0)
L0=(); L1=(); L2=()
for i in "${!ORDER[@]}"; do
  case $((i % 3)) in 0) L0+=("${ORDER[$i]}");; 1) L1+=("${ORDER[$i]}");; 2) L2+=("${ORDER[$i]}");; esac
done
run_lane "${L0[@]}" > $R/lane0.log 2>&1 & P0=$!
run_lane "${L1[@]}" > $R/lane1.log 2>&1 & P1=$!
run_lane "${L2[@]}" > $R/lane2.log 2>&1 & P2=$!
wait $P0; S0=$?; wait $P1; S1=$?; wait $P2; S2=$?
[ "$S0" = 0 ] && [ "$S1" = 0 ] && [ "$S2" = 0 ] || { echo "FAILED a search lane (exits $S0 $S1 $S2)"; exit 1; }

echo "=== replay"
$PY $D replay --run-dir $R/ablated --corpus $V4 $W3 --no-wandb || { echo "FAILED replay ablated"; exit 1; }
$PY $D replay --run-dir $R/random --corpus $V4 $W3 --no-wandb || { echo "FAILED replay random"; exit 1; }
echo "=== ablation study done"
