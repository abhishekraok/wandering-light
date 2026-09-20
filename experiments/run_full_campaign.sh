#!/usr/bin/env bash
# Train what is missing, then run the timing cells one at a time.
#
# Both halves are resumable: training skips an (arm, seed) whose epoch-6
# snapshot exists, and a cell skips when its json exists. A container restart
# therefore costs at most the run that was in flight.
set -u
cd "$(dirname "$0")/.." || exit 1
uv run python experiments/hindsight_subtargets.py train \
  --run-dir checkpoints/hindsight-20260919 \
  --arms witness,hindsight,hindsight-all --seeds 0,1,2 \
  >> checkpoints/hindsight-20260919-train.out 2>&1
exec bash experiments/run_cache_campaign.sh
