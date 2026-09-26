#!/usr/bin/env bash
# Launch one codex exec author per letter in a shared authoring worktree and
# wait for all of them (distinct-programs, #174). Adapted from
# experiments/world_behaviours/run_authors.sh (#127).
#
#   experiments/distinct_programs/run_authors.sh <worktree> <report-dir> <BRIEF.md> <kind> "<letters>"
#   kind is "refs" or "programs": the output file each author writes.
#
# The worktree holds <report-dir>/<BRIEF.md>, the behaviours file and
# assign_<letter>.txt. Authors write only under <report-dir>. Each author's
# log, timing, token count and exit status land beside its output.
set -u
W="$1"; R="$2"; BRIEF="$3"; KIND="$4"; LETTERS="$5"
MODEL="${CODEX_MODEL:-gpt-6-sol}"; EFFORT="${CODEX_EFFORT:-xhigh}"
pids=""
for k in $LETTERS; do
  (
    cd "$W" || exit 1
    date +%s > "$R/codex_$k.start"
    printf '{"model": "%s", "effort": "%s", "brief": "%s"}\n' "$MODEL" "$EFFORT" "$BRIEF" > "$R/codex_$k.settings"
    codex exec --skip-git-repo-check -s workspace-write -m "$MODEL" -c "model_reasoning_effort=$EFFORT" \
      -o "$R/codex_${k}_last.md" \
      "Read $R/$BRIEF and follow it exactly. You are author $k. Your ids are listed in $R/assign_$k.txt. Write your output to $R/${KIND}_$k.jsonl." \
      < /dev/null > "$R/codex_$k.log" 2>&1
    status=$?
    date +%s > "$R/codex_$k.end"
    echo "$status" > "$R/codex_$k.status"
    exit "$status"
  ) &
  pids="$pids $!"
done
failed=0
for pid in $pids; do wait "$pid" || failed=1; done
for k in $LETTERS; do
  el=$(( $(cat "$W/$R/codex_$k.end") - $(cat "$W/$R/codex_$k.start") ))
  tok=$(grep -A1 "^tokens used" "$W/$R/codex_$k.log" | tail -1)
  echo "{\"author\": \"$k\", \"seconds\": $el, \"tokens\": \"$tok\", \"status\": $(cat "$W/$R/codex_$k.status"), \"model\": \"$MODEL\", \"effort\": \"$EFFORT\"}"
done
[ "$failed" = 0 ] || { echo "one or more authors exited non-zero" >&2; exit 1; }
