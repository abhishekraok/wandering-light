#!/usr/bin/env bash
# Phase 1 of opus-effort-curve (#137): Claude Opus 5 at effort low / high /
# xhigh on the seed-17 40-task eval sample of stack_pbe_v4, one sample per
# task, through `claude -p` trimmed to the API protocol
# (ClaudeExecTokenGenerator). Arms run concurrently, each sequential over its
# 40 tasks. Same prompt, parser and grader as every other LLM row; same sample
# as ../stack-pbe-v4-codex-exec-20260910.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=reports/stack-pbe-v4-claude-exec-20260913

# An arm is a pipeline through `tee` in a background subshell, so its Python
# exit status is read from PIPESTATUS and returned; the parent waits per PID.
arm() {  # effort tag
  local status
  echo "RUN STARTED $(date)" | tee -a "$R/$2.log"
  .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider claude \
    --model opus --effort "$1" --tasks 40 --split eval --seed 17 \
    --out "$R/$2.json" 2>&1 | tee -a "$R/$2.log"
  status=${PIPESTATUS[0]}
  echo "RUN FINISHED $(date) (exit $status)" | tee -a "$R/$2.log"
  [ "$status" -eq 0 ] || echo "ARM FAILED: $1 $2" | tee -a "$R/$2.log"
  return "$status"
}

pids=()
arm low   opus_low_v4_sample40 &   pids+=($!)
arm high  opus_high_v4_sample40 &  pids+=($!)
arm xhigh opus_xhigh_v4_sample40 & pids+=($!)

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "PHASE 1 INCOMPLETE: an arm failed"; exit 1; }
echo "PHASE 1 DONE $(date)"
