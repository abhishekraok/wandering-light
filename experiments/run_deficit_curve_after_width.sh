#!/usr/bin/env bash
# Does the per-step deficit fall with training, and does it approach ln W / L?
#
# The budget argument says a 41+ behaviour needs a sustained 0.053 nats/step at
# beam 64. The measured deficit is 1-3 nats. The open question that closes the
# "just train better" door -- or opens it -- is whether more training, or the
# lower learning rate #134 noted was never tried, moves that number at all.
#
# Two learning rates, one seed, 18 epochs, snapshots every epoch. The deficit is
# then read off each snapshot by search_proxy with no search at all, so the
# expensive half of this is training, not evaluation.
#
# Queued behind the epoch grid; every run skips when its epoch-18 snapshot
# exists.
set -u
cd "$(dirname "$0")/.." || exit 1

OUT=checkpoints/deficit-curve-20260919
LOG=$OUT/train.log
mkdir -p "$OUT"

while ! grep -q "self-shorten done" checkpoints/self-shorten-20260920/run.log 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) width sweep done; starting the deficit curve" | tee -a "$LOG"

for lr in 1.25e-4 6.25e-5; do
  tag="lr$lr"
  if [ -d "$OUT/$tag/policies/witness-s0-e18" ]; then
    echo "$(date +%H:%M:%S) skip $tag (snapshots exist)" | tee -a "$LOG"
    continue
  fi
  echo "$(date +%H:%M:%S) START train $tag" | tee -a "$LOG"
  EPOCHS=18 LEARNING_RATE="$lr" uv run python - "$OUT/$tag" "$lr" <<'PY' 2>&1 | tee -a "$LOG"
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[0] if False else Path.cwd()
sys.path.insert(0, str(REPO))
import experiments.hindsight_subtargets as hs

run_dir, lr = Path(sys.argv[1]), float(sys.argv[2])
hs.EPOCHS = 18
hs.LEARNING_RATE = lr
tasks = hs.co.corpus()
hs.train(run_dir, tasks, "cuda", ["witness"], [0], lambda line: print(line, flush=True))
PY
done
echo "$(date +%H:%M:%S) deficit curve training done" | tee -a "$LOG"

exec bash experiments/run_analysis.sh
