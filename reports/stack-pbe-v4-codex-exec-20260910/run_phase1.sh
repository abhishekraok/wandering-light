#!/usr/bin/env bash
# Phase 1 of codex-exec-frontier-effort-curve (#123): five arms on the seed-17
# 40-task eval sample of stack_pbe_v4, one sample per task, through `codex exec`
# trimmed to the API protocol (CodexExecTokenGenerator). Arms run concurrently,
# each sequential over its 40 tasks. Same prompt, parser and grader as the
# luna API row in ../stack-pbe-v4-baseline-20260901.
set -uo pipefail
# Each arm is a pipeline through `tee`; without this a Python process that dies
# on an auth or config error would leave an empty JSON and the script would
# still exit 0 (Codex review on PR #126).
failed=0
cd "$(dirname "$0")/../.."
R=reports/stack-pbe-v4-codex-exec-20260910
arm() {  # model effort tag
  echo "RUN STARTED $(date)" | tee -a "$R/$3.log"
  .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider codex \
    --model "$1" --effort "$2" --tasks 40 --split eval --seed 17 \
    --out "$R/$3.json" 2>&1 | tee -a "$R/$3.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "ARM FAILED: $*"; failed=1; }
  echo "RUN FINISHED $(date)" | tee -a "$R/$3.log"
}
arm gpt-5.6-luna  high   luna_high_v4_sample40 &
arm gpt-5.6-terra high   terra_high_v4_sample40 &
arm gpt-6-astra   low    astra_low_v4_sample40 &
arm gpt-6-astra   medium astra_medium_v4_sample40 &
arm gpt-6-astra   high   astra_high_v4_sample40 &
wait
[ "$failed" -eq 0 ] || { echo "PHASE 1 INCOMPLETE: an arm failed"; exit 1; }
echo "PHASE 1 DONE $(date)"
