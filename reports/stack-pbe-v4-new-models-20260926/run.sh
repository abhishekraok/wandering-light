#!/usr/bin/env bash
# new-models-sample40 (#173): gpt-6-sol and gpt-6-luna through `codex exec`,
# claude-opus-5-5 at four efforts through `claude -p`, on the seed-17 40-task
# eval sample of stack_pbe_v4. Same protocol, prompt, parser and grader as
# ../stack-pbe-v4-codex-exec-20260910 and ../stack-pbe-v4-claude-exec-20260913.
# Arms run concurrently, each sequential over its 40 tasks. Output goes to $R
# (default: this directory).
set -uo pipefail
cd "$(dirname "$0")/../.."
R=${R:-reports/stack-pbe-v4-new-models-20260926}
mkdir -p "$R"

arm() {  # provider model effort tag
  local status
  echo "RUN STARTED $(date)" | tee -a "$R/$4.log"
  .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider "$1" \
    --model "$2" --effort "$3" --tasks 40 --split eval --seed 17 \
    --out "$R/$4.json" 2>&1 | tee -a "$R/$4.log"
  status=${PIPESTATUS[0]}
  echo "RUN FINISHED $(date) (exit $status)" | tee -a "$R/$4.log"
  [ "$status" -eq 0 ] || echo "ARM FAILED: $2 $3 $4" | tee -a "$R/$4.log"
  return "$status"
}

pids=()
arm codex  gpt-6-sol       high   sol6_high_v4_sample40 &     pids+=($!)
arm codex  gpt-6-luna      high   luna6_high_v4_sample40 &    pids+=($!)
arm claude claude-opus-5-5 low    opus55_low_v4_sample40 &    pids+=($!)
arm claude claude-opus-5-5 medium opus55_medium_v4_sample40 & pids+=($!)
arm claude claude-opus-5-5 high   opus55_high_v4_sample40 &   pids+=($!)
arm claude claude-opus-5-5 xhigh  opus55_xhigh_v4_sample40 &  pids+=($!)

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo "INCOMPLETE: an arm failed"; exit 1; }
echo "ALL DONE $(date)"
