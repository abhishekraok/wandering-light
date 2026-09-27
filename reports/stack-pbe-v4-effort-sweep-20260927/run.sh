#!/usr/bin/env bash
# effort-sweep-5x: gpt-5.6 sol/terra/luna through `codex exec` and claude-fable-5 /
# claude-fable-5-1 through `claude -p`, on the seed-17 40-task eval sample of
# stack_pbe_v4. Same protocol as ../stack-pbe-v4-new-models-20260926. terra and
# luna high are the #123 runs, so they are not repeated. Eight arms at a time.
set -uo pipefail
cd "$(dirname "$0")/../.."
R=${R:-reports/stack-pbe-v4-effort-sweep-20260927}
mkdir -p "$R"

arm() {  # provider model effort tag
  echo "RUN STARTED $(date)" >> "$R/$4.log"
  .venv/bin/python experiments/stack_pbe_baseline.py --backend llm --provider "$1" \
    --model "$2" --effort "$3" --tasks 40 --split eval --seed 17 \
    --out "$R/$4.json" >> "$R/$4.log" 2>&1
  local status=$?
  echo "RUN FINISHED $(date) (exit $status)" >> "$R/$4.log"
  [ "$status" -eq 0 ] || echo "ARM FAILED: $2 $3 $4"
}
export -f arm; export R

{
  for e in low medium high xhigh max; do echo "codex gpt-5.6-sol $e sol56_${e}_v4_sample40"; done
  for m in terra luna; do for e in low medium xhigh max; do
    echo "codex gpt-5.6-$m $e ${m}56_${e}_v4_sample40"; done; done
  for m in 5 5-1; do for e in low medium high xhigh max; do
    echo "claude claude-fable-$m $e fable${m/-/}_${e}_v4_sample40"; done; done
} | xargs -P 8 -L 1 bash -c 'arm "$@"' _
echo "ALL DONE $(date)"
