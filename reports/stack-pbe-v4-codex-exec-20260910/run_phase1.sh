#!/usr/bin/env bash
# Phase 1 of codex-exec-frontier-effort-curve (#123): five arms on the seed-17
# 40-task eval sample of stack_pbe_v4, one sample per task, through `codex exec`
# trimmed to the API protocol (CodexExecTokenGenerator). Arms run concurrently,
# each sequential over its 40 tasks. Same prompt, parser and grader as the
# luna API row in ../stack-pbe-v4-baseline-20260901.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=reports/stack-pbe-v4-codex-exec-20260910

# Each arm is a pipeline through `tee`, so the Python exit status has to be read
# out of PIPESTATUS and *returned*. An arm runs in a background subshell, so it
# cannot report failure by setting a variable -- the parent would never see it.
# The parent waits on each PID instead. Without this a Python process that died
# on an auth or config error would leave a stale JSON in place and the script
# would still exit 0 (Codex reviews on PR #126).
arm() {  # model effort tag
  local status
  echo "RUN STARTED $(date)" | tee -a "$R/$3.log"
  .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider codex \
    --model "$1" --effort "$2" --tasks 40 --split eval --seed 17 \
    --out "$R/$3.json" 2>&1 | tee -a "$R/$3.log"
  status=${PIPESTATUS[0]}
  echo "RUN FINISHED $(date) (exit $status)" | tee -a "$R/$3.log"
  [ "$status" -eq 0 ] || echo "ARM FAILED: $1 $2 $3" | tee -a "$R/$3.log"
  return "$status"
}

pids=()
arm gpt-5.6-luna  high   luna_high_v4_sample40 &    pids+=($!)
arm gpt-5.6-terra high   terra_high_v4_sample40 &   pids+=($!)
arm gpt-6-astra   low    astra_low_v4_sample40 &    pids+=($!)
arm gpt-6-astra   medium astra_medium_v4_sample40 & pids+=($!)
arm gpt-6-astra   high   astra_high_v4_sample40 &   pids+=($!)

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "PHASE 1 INCOMPLETE: an arm failed"; exit 1; }
echo "PHASE 1 DONE $(date)"
