#!/usr/bin/env bash
# Launch one codex exec author per worktree and wait for all of them (#127).
#
#   experiments/world_behaviours/run_authors.sh <report-dir> <letters>
#   e.g. experiments/world_behaviours/run_authors.sh reports/world-authoring-20260911-w2 "a b c d"
#
# Expects worktrees at .claude/worktrees/codex-<letter> (outside /tmp) checked out
# at a commit that contains <report-dir>/BRIEF.md, behaviours.jsonl and
# assign_<letter>.txt. Writes codex_<letter>.{log,start,end} and
# codex_<letter>_last.md into the worktree's copy of <report-dir>; collect
# them with the report's authors/ convention afterwards.
set -u
R="$1"; LETTERS="$2"
ROOT="$(git rev-parse --show-toplevel)"
MAIN_ROOT="${ROOT%%/.claude/worktrees/*}"
MODEL="${CODEX_MODEL:-gpt-6-astra}"; EFFORT="${CODEX_EFFORT:-high}"
for k in $LETTERS; do
  W="$MAIN_ROOT/.claude/worktrees/codex-$k"
  (
    cd "$W" || exit 1
    date +%s > "$R/codex_$k.start"
    codex exec --skip-git-repo-check -s workspace-write -m "$MODEL" -c "model_reasoning_effort=$EFFORT" \
      -o "$R/codex_${k}_last.md" \
      "Read $R/BRIEF.md and follow it exactly. You are author $k. Your ids are listed in $R/assign_$k.txt. Write your output to $R/programs_$k.jsonl; your macro file is experiments/world_behaviours/macros_$k.txt." \
      > "$R/codex_$k.log" 2>&1
    date +%s > "$R/codex_$k.end"
  ) &
done
wait
for k in $LETTERS; do
  W="$MAIN_ROOT/.claude/worktrees/codex-$k"
  el=$(( $(cat "$W/$R/codex_$k.end") - $(cat "$W/$R/codex_$k.start") ))
  tok=$(grep -A1 "^tokens used" "$W/$R/codex_$k.log" | tail -1)
  echo "{\"author\": \"$k\", \"seconds\": $el, \"tokens\": \"$tok\", \"model\": \"$MODEL\", \"effort\": \"$EFFORT\"}"
done
