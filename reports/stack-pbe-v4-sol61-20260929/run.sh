#!/usr/bin/env bash
# sol61-sample40 (#177): gpt-6.1-sol through untooled `codex exec`, same
# seed-17 sample and protocol as the README plot. EFFORTS defaults to
# low..xhigh (max only if xhigh is below the top band). OUT defaults to this
# directory; PYTHON may point at an environment in another checkout.
set -uo pipefail
cd "$(dirname "$0")/../.."
PYTHON=${PYTHON:-.venv/bin/python}
OUT=${OUT:-reports/stack-pbe-v4-sol61-20260929}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"

arm() {
  local effort=$1 status
  local tag="sol61_${effort}_v4_sample40"
  printf 'RUN STARTED %s codex %s\n' "$(date -Is)" "$(codex --version)" >> "$OUT/$tag.log"
  "$PYTHON" -u experiments/stack_pbe_baseline.py --backend llm --provider codex \
    --model gpt-6.1-sol --effort "$effort" --tasks 40 --split eval --seed 17 \
    --out "$OUT/$tag.json" >> "$OUT/$tag.log" 2>&1
  status=$?
  printf 'RUN FINISHED %s (exit %s)\n' "$(date -Is)" "$status" >> "$OUT/$tag.log"
  return "$status"
}

pids=()
for effort in ${EFFORTS:-low medium high xhigh}; do
  arm "$effort" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[ "$failed" -eq 0 ] || { echo 'INCOMPLETE: an arm failed'; exit 1; }
printf 'ALL DONE %s\n' "$(date -Is)"
