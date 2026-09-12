#!/usr/bin/env bash
# Collect a wave's author outputs and verify them raw, before the merge pass (#127).
#
#   experiments/world_behaviours/collect_wave.sh <report-dir> <letters>
#
# Copies programs_<k>.jsonl, macros_<k>.txt and codex_<k>_last.md from each
# author worktree into <report-dir>/authors/, writes authors/runs.jsonl
# (seconds and tokens per author), keeps the codex logs outside the repo under
# .claude/codex-logs/<report-name>/, then re-verifies every program with the
# author macro files loaded and writes authors/README.md + report.json (the
# raw stage). The merge pass (macros.txt, aliases.txt, migrate.py, report.py
# on <report-dir>) follows by hand.
set -eu
R="$1"; LETTERS="$2"
ROOT="$(git rev-parse --show-toplevel)"
MAIN_ROOT="${ROOT%%/.claude/worktrees/*}"
NAME="$(basename "$R")"
mkdir -p "$R/authors" "$MAIN_ROOT/.claude/codex-logs/$NAME"
: > "$R/authors/runs.jsonl"
for k in $LETTERS; do
  W="$MAIN_ROOT/.claude/worktrees/codex-$k"
  cp "$W/$R/programs_$k.jsonl" "$W/$R/codex_${k}_last.md" "$R/authors/"
  cp "$W/experiments/world_behaviours/macros_$k.txt" "$R/authors/" 2>/dev/null || true
  cp "$W/$R/codex_$k.log" "$MAIN_ROOT/.claude/codex-logs/$NAME/"
  el=$(( $(cat "$W/$R/codex_$k.end") - $(cat "$W/$R/codex_$k.start") ))
  tok=$(grep -A1 "^tokens used" "$W/$R/codex_$k.log" | tail -1)
  status=$(cat "$W/$R/codex_$k.status" 2>/dev/null || echo null)
  settings=$(cat "$W/$R/codex_$k.settings")  # written by run_authors.sh: the settings actually used
  echo "{\"author\": \"$k\", \"seconds\": $el, \"tokens\": \"$tok\", \"status\": $status, \"model\": $(python3 -c "import json,sys;print(json.dumps(json.loads(sys.argv[1])['model']))" "$settings"), \"effort\": $(python3 -c "import json,sys;print(json.dumps(json.loads(sys.argv[1])['effort']))" "$settings")}" >> "$R/authors/runs.jsonl"
  git -C "$W" status --short | grep -v '^??' && { echo "author $k changed tracked files"; exit 1; } || true
done
cat "$R/authors/runs.jsonl"
# Raw verification loads the canonical library plus this wave's author files from
# a scratch directory, so nothing in experiments/world_behaviours/ is touched.
STAGE="$(mktemp -d)"
cp experiments/world_behaviours/macros.txt "$STAGE/"
for k in $LETTERS; do cp "$R/authors/macros_$k.txt" "$STAGE/" 2>/dev/null || true; done
cp "$R/behaviours.jsonl" "$R/authors/behaviours.jsonl"
uv run python experiments/world_behaviours/report.py "$R/authors" --no-findings --macros "$STAGE/macros.txt" \
  --title "$NAME raw (author dialects)" --intro "raw author outputs before the merge pass" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({k:d[k] for k in ('solved','claim_mismatches','by_source_difficulty','by_author','length_median','macro_length_median')}, indent=1))"
rm -rf "$STAGE" "$R/authors/behaviours.jsonl"
