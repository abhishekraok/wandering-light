#!/usr/bin/env bash
# Launch one `claude -p` author per letter in a shared authoring worktree and
# wait for all of them (distinct-programs, #174): the cross-family witness set.
# Same brief and checker as the Codex authors (run_authors.sh).
#
#   experiments/distinct_programs/run_claude_authors.sh <worktree> <report-dir> <BRIEF.md> <kind> "<letters>"
set -u
W="$1"; R="$2"; BRIEF="$3"; KIND="$4"; LETTERS="$5"
MODEL="${CLAUDE_MODEL:-claude-opus-5-5}"; EFFORT="${CLAUDE_EFFORT:-high}"
pids=""
for k in $LETTERS; do
  (
    cd "$W" || exit 1
    date +%s > "$R/claude_$k.start"
    printf '{"model": "%s", "effort": "%s", "brief": "%s"}\n' "$MODEL" "$EFFORT" "$BRIEF" > "$R/claude_$k.settings"
    env -u ANTHROPIC_API_KEY claude -p --model "$MODEL" --effort "$EFFORT" \
      --allowedTools "Bash,Read,Write,Edit" --output-format json --no-session-persistence \
      "Read $R/$BRIEF and follow it exactly. You are author $k. Your ids are listed in $R/assign_$k.txt. Write your output to $R/${KIND}_$k.jsonl. Do not open any other author's output files." \
      < /dev/null > "$R/claude_$k.json" 2> "$R/claude_$k.err"
    status=$?
    date +%s > "$R/claude_$k.end"
    echo "$status" > "$R/claude_$k.status"
    exit "$status"
  ) &
  pids="$pids $!"
done
failed=0
for pid in $pids; do wait "$pid" || failed=1; done
for k in $LETTERS; do
  el=$(( $(cat "$W/$R/claude_$k.end") - $(cat "$W/$R/claude_$k.start") ))
  echo "{\"author\": \"$k\", \"seconds\": $el, \"status\": $(cat "$W/$R/claude_$k.status"), \"model\": \"$MODEL\", \"effort\": \"$EFFORT\"}"
done
[ "$failed" = 0 ] || { echo "one or more authors exited non-zero" >&2; exit 1; }
