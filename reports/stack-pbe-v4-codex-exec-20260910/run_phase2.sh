#!/usr/bin/env bash
# Phase 2 of codex-exec-frontier-effort-curve (#123): gpt-6-astra at effort low
# on the full 1,800-task eval split of stack_pbe_v4, one sample per task,
# through `codex exec` trimmed to the API protocol. Six shards run concurrently
# (tasks[k::6]); `task_index` in each JSON is the join, analyse.py merges them.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=reports/stack-pbe-v4-codex-exec-20260910
N=6
shard() {
  echo "RUN STARTED $(date)" | tee -a "$R/astra_low_v4_full_shard$1.log"
  .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider codex \
    --model gpt-6-astra --effort low --tasks 0 --split eval --shard "$1/$N" \
    --out "$R/astra_low_v4_full_shard$1.json" 2>&1 | tee -a "$R/astra_low_v4_full_shard$1.log"
  echo "RUN FINISHED $(date)" | tee -a "$R/astra_low_v4_full_shard$1.log"
}
for k in $(seq 0 $((N-1))); do shard "$k" & done
wait
echo "PHASE 2 DONE $(date)"
