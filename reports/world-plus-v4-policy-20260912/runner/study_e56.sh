#!/usr/bin/env bash
# world-plus-v4-policy (#134): extend the epoch study to epochs 5 and 6.
# v4-only solves were still rising at epoch 4 on two seeds, so the matched
# comparison needs the later snapshots. Weights come from study.sh's training
# (snapshots e1-e6 already saved). Same three concurrent lanes, then replay, then
# the 10,000-draw solve check with dense draws over every cell, then the table.
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260913-epochs
V4=wandering_light/evals/data/stack_pbe_v4/manifest.json
W3=wandering_light/evals/data/stack_pbe_world_v3/manifest.json
PY=.venv/bin/python
D=experiments/compute_optimal_mark28.py
COMMON="--arms 1m --no-wandb --train-device cuda --devices cuda"
REP=reports/world-plus-v4-policy-20260912

echo "=== search 12 cells (epochs 5, 6) in 3 lanes"
run_lane() {
  local status=0 spec arm epoch seed corpora
  for spec in "$@"; do
    IFS=: read -r arm epoch seed <<< "$spec"
    if [ "$arm" = mixed ]; then corpora="--corpus $V4 $W3"; else corpora="--corpus $V4"; fi
    $PY $D time --run-dir $R/$arm $COMMON --seeds "$seed" --policy-epoch "$epoch" $corpora || status=1
  done
  return $status
}
ORDER=(v4only:5:0 v4only:5:1 v4only:5:2 mixed:5:0 mixed:5:1 mixed:5:2
       v4only:6:0 v4only:6:1 v4only:6:2 mixed:6:0 mixed:6:1 mixed:6:2)
L0=(); L1=(); L2=()
for i in "${!ORDER[@]}"; do
  case $((i % 3)) in 0) L0+=("${ORDER[$i]}");; 1) L1+=("${ORDER[$i]}");; 2) L2+=("${ORDER[$i]}");; esac
done
run_lane "${L0[@]}" > $R/lane0_e56.log 2>&1 & P0=$!
run_lane "${L1[@]}" > $R/lane1_e56.log 2>&1 & P1=$!
run_lane "${L2[@]}" > $R/lane2_e56.log 2>&1 & P2=$!
wait $P0; S0=$?; wait $P1; S1=$?; wait $P2; S2=$?
[ "$S0" = 0 ] && [ "$S1" = 0 ] && [ "$S2" = 0 ] || { echo "FAILED a search lane (exits $S0 $S1 $S2)"; exit 1; }

echo "=== replay"
$PY $D replay --run-dir $R/v4only --corpus $V4 --no-wandb || { echo "FAILED replay v4only"; exit 1; }
$PY $D replay --run-dir $R/mixed --corpus $V4 $W3 --no-wandb || { echo "FAILED replay mixed"; exit 1; }

echo "=== solve check, all cells"
$PY experiments/step3_verify_solves.py \
  --run-dir checkpoints/step3-20260912/v4only --run-dir checkpoints/step3-20260912/mixed \
  --run-dir $R/v4only --run-dir $R/mixed \
  --draws 10000 --dense --min-tested 500 \
  --out $REP/verified_solves_epochs.json > $R/verify_10k_e56.log 2>&1 || { echo "FAILED verify"; exit 1; }

echo "=== epochs table"
$PY experiments/step3_epochs_report.py --baseline $R/v4only --mixed $R/mixed \
  --baseline-best checkpoints/step3-20260912/v4only --mixed-best checkpoints/step3-20260912/mixed \
  --verified $REP/verified_solves_epochs.json --out $REP || { echo "FAILED epochs table"; exit 1; }
echo "=== e56 done"
