#!/usr/bin/env bash
# haiku55-sample40 (#181): same untooled protocol and seed-17 sample as the
# README plot. PYTHON may point at an existing environment in another checkout.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-.venv/bin/python}
R=reports/stack-pbe-v4-haiku55-20261011
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$R"

arm() {
  local effort=$1 status
  local tag="haiku55_${effort}_v4_sample40"
  printf 'RUN STARTED %s\n' "$(date -Is)" >> "$R/$tag.log"
  "$PYTHON" -u experiments/stack_pbe_baseline.py --backend llm --provider claude \
    --model claude-haiku-5-5 --effort "$effort" --tasks 40 --split eval --seed 17 \
    --out "$R/$tag.json" >> "$R/$tag.log" 2>&1
  status=$?
  printf 'RUN FINISHED %s (exit %s)\n' "$(date -Is)" "$status" >> "$R/$tag.log"
  return "$status"
}

pids=()
for effort in low medium high xhigh max; do
  arm "$effort" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then
  echo 'INCOMPLETE: an arm failed'
  exit 1
fi
printf 'ALL DONE %s\n' "$(date -Is)"
