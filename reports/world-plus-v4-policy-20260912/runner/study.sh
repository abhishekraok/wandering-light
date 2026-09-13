#!/usr/bin/env bash
# world-plus-v4-policy (#134): is the step-3 solve gain the data or the selected epoch?
# Retrain both arms deterministically saving epochs 1-6, check the snapshots at the
# original best epochs equal the original checkpoints bit for bit, then search the
# 300-task v4 cohort at epochs 1-4 for both arms. The six (arm, seed, best epoch)
# cells already exist in checkpoints/step3-20260912 and are reused. Cells run in
# three concurrent lanes: solve counts are exact (the beam has no time budget), but
# wall-clock from this study is not comparable with sequential cells.
set -u
cd "$(git rev-parse --show-toplevel)"
R=checkpoints/step3-20260913-epochs
V4=wandering_light/evals/data/stack_pbe_v4/manifest.json
W3=wandering_light/evals/data/stack_pbe_world_v3/manifest.json
PY=.venv/bin/python
D=experiments/compute_optimal_mark28.py
COMMON="--arms 1m --no-wandb --train-device cuda --devices cuda"

echo "=== train both arms, snapshots at epochs 1-6"
$PY $D train --run-dir $R/v4only $COMMON --seeds 0 1 2 --epochs 6 --save-epochs 1 2 3 4 5 6 \
  --corpus $V4 > $R/train_v4only.log 2>&1 &
TA=$!
$PY $D train --run-dir $R/mixed $COMMON --seeds 0 1 2 --epochs 6 --save-epochs 1 2 3 4 5 6 \
  --corpus $V4 $W3 --validation-corpus stack_pbe_v4 --drop-behaviours $R/mixed/drop.txt \
  > $R/train_mixed.log 2>&1 &
TB=$!
wait $TA; SA=$?
wait $TB; SB=$?
[ "$SA" = 0 ] && [ "$SB" = 0 ] || { echo "FAILED training (v4only exit $SA, mixed exit $SB)"; exit 1; }

echo "=== determinism gate"
$PY - <<'PY' || { echo "FAILED determinism: the original best-epoch cells cannot be reused"; exit 1; }
import json, sys, torch
from pathlib import Path
from safetensors.torch import load_file
bad = 0
for arm in ("v4only", "mixed"):
    for s in (0, 1, 2):
        old = Path(f"checkpoints/step3-20260912/{arm}/policies/1m-s{s}")
        epoch = json.loads((old / "trainer_state.json").read_text())["best_epoch"]
        new = Path(f"checkpoints/step3-20260913-epochs/{arm}/policies/1m-s{s}-e{epoch}")
        a, b = load_file(old / "model.safetensors"), load_file(new / "model.safetensors")
        same = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
        print(f"RESULT determinism {arm} s{s} e{epoch}: {'identical' if same else 'DIFFERENT'}", flush=True)
        bad += not same
sys.exit(1 if bad else 0)
PY

echo "=== search 18 cells in 3 lanes"
run_lane() {
  local status=0 spec arm epoch seed corpora
  for spec in "$@"; do
    IFS=: read -r arm epoch seed <<< "$spec"
    if [ "$arm" = mixed ]; then corpora="--corpus $V4 $W3"; else corpora="--corpus $V4"; fi
    $PY $D time --run-dir $R/$arm $COMMON --seeds "$seed" --policy-epoch "$epoch" $corpora || status=1
  done
  return $status
}
ORDER=(v4only:3:0 v4only:3:1 v4only:3:2 v4only:4:0 v4only:4:1 v4only:4:2
       mixed:1:0 mixed:1:1 mixed:1:2 mixed:2:1 mixed:2:2
       v4only:2:0 v4only:2:2 v4only:1:1 mixed:3:0 mixed:3:1 mixed:4:0 mixed:4:2)
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
$PY $D replay --run-dir $R/v4only --corpus $V4 --no-wandb || { echo "FAILED replay v4only"; exit 1; }
$PY $D replay --run-dir $R/mixed --corpus $V4 $W3 --no-wandb || { echo "FAILED replay mixed"; exit 1; }
echo "=== epochs done"
